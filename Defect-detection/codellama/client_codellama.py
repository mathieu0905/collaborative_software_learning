from collections import OrderedDict

from cc_func import load_data, load_model, train, evaluate, parse_args
from torch.utils.data import DataLoader

import flwr as fl
import torch
import logging
import pickle
import numpy as np
import os

from peft import get_peft_model_state_dict, set_peft_model_state_dict

def set_parameters(model, parameters) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


logger = logging.getLogger(__name__)


args = parse_args()

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
               args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
               torch.distributed.get_world_size() if args.local_rank != -1 else 1)
# 使用FileHandler输出到文件
fh = logging.FileHandler(args.log_file)
logger.addHandler(fh)

logger.info("Loading model")

try:
    model, tokenizer, args = load_model(args)
    logger.info("Loading model succesfully")
except Exception as e:
    logger.error(e)

args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, trainset, testset, cid, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.trainset = trainset
        self.testset = testset
        self.cid = cid
        self.logger = logger
        
    def get_parameters(self, config):
        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]
        # 筛选出LoRA微调部分的参数
        # lora_parameters = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()  if 'q_proj' in k or 'v_proj' in k or 'k_proj' in k or 'o_proj' in k}
        # return lora_parameters

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(args, self.trainset, self.model, self.tokenizer, self.logger)
        return self.get_parameters({}), len(self.trainset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        eval_loss, perplexity = evaluate(args, self.model, self.tokenizer, self.testset, self.logger)
        return eval_loss, len(self.testset), {"perplexity": perplexity}
    

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    logger.info(f"Client {cid} is starting training...")
    
    trainset, testset = load_data(cid=cid, tokenizer=tokenizer, args=args)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.info(f"Using {device}")
    # Create a client-specific Flower client
    return FlowerClient(model.to("cuda"), tokenizer, trainset, testset, cid, logger)


class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        logger.info(f"Server: Aggregating parameters in round {rnd}")
        # Aggregate parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        try:
            # Convert `Parameters` to `List[np.ndarray]`
            # aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # # Save aggregated_ndarrays
            # print(f"Saving round {rnd} aggregated_ndarrays...")
            # np.savez(f"round-{rnd}-weights.npz", *aggregated_ndarrays)

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            logger.info(state_dict.keys())

            model.load_state_dict(state_dict, strict=False)

            # Save the model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.output_dir}/model_round_{rnd}.pth")
        except Exception as e:
            logger.error(e)

        return aggregated_parameters, aggregated_metrics


NUM_CLIENTS = 2
client_resources = {"num_cpus": 16, "num_gpus": 1}
# client_fn(1)
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    client_resources=client_resources,
    strategy=CustomFedAvg()
)
# fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
with open(f"{args.output_dir}/results.pkl", "wb") as f:
    pickle.dump(history, f)