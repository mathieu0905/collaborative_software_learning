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

os.environ["RAY_TMPDIR"] = "/data//FL/my_ray_tmp"

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


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

args.n_gpu = 1
args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, trainset, testset, cid):
        self.model = model
        self.tokenizer = tokenizer
        self.trainset = trainset
        self.testset = testset
        self.cid = cid
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(args, self.trainset, self.model, self.tokenizer)
        return self.get_parameters({}), len(self.trainset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        eval_loss, perplexity = evaluate(args, self.model, self.tokenizer, self.testset)
        return eval_loss, len(self.testset), {"perplexity": perplexity}
    

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    logger.info(f"Client {cid} is starting training...")
    
    trainset, testset = load_data(cid=cid, tokenizer=tokenizer, args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device}")
    # Create a client-specific Flower client
    return FlowerClient(model.to(device), tokenizer, trainset, testset, cid)

class CustomFedAvg(fl.server.strategy.FedAvg):
    min_fit_clients=3,  # Minimum number of clients to be sampled for the next round
    min_available_clients=10,  # Minimum number of clients that need to be connected to the server before a training round can start
    def aggregate_fit(self, rnd, results, failures):
        logger.info(f"Server: Aggregating parameters in round {rnd}")
        # Aggregate parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        # Convert `Parameters` to `List[np.ndarray]`
        aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # Convert `List[np.ndarray]` to PyTorch`state_dict`
        params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Save the model
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{args.output_dir}/model_round_{rnd}.bin")

        return aggregated_parameters, aggregated_metrics



NUM_CLIENTS = 10
client_resources = {"num_cpus": 8, "num_gpus": 1.0}
# client_fn(1)
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    client_resources=client_resources,
    strategy=CustomFedAvg()
)

with open(f"{args.output_dir}/results.pkl", "wb") as f:
    pickle.dump(history, f)
# fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())