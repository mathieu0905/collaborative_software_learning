from collections import OrderedDict

from centralized_uneven_nums import load_data, load_model, train, evaluate, DEVICE, parse_args
from torch.utils.data import DataLoader

import flwr as fl
import torch
import ray
import logging
import pickle
import os

logging.basicConfig(filename='training_uneven_nums.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

args = parse_args()

logging.info("Loading model")

try:
    model, args = load_model(args)
    logging.info("Loading model succesfully")
except Exception as e:
    logging.error(e)

args.n_gpu = 1
args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
args.device = DEVICE


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainset, testset, cid):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.cid = cid

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(args, self.model, self.trainset, self.testset, self.cid)
        return self.get_parameters({}), len(self.trainset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, result = evaluate(args, self.model, self.testset)
        return loss, len(self.testset), {"accuracy": result['precision']}
    

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    logging.info(f"Client {cid} is starting training...")
    
    trainset, testset = load_data(cid=cid, args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device}")
    # Create a client-specific Flower client
    return FlowerClient(model.to(device), trainset, testset, cid)

NUM_CLIENTS = 10
client_resources = {"num_cpus": 16, "num_gpus": 1.0}
# client_fn(1)

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        logging.info(f"Server: Aggregating parameters in round {rnd}")
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None and aggregated_parameters[0] is not None:
            logging.info("Server: Parameters aggregated successfully")

            # 提取模型参数
            model_parameters = aggregated_parameters[0]

            # 定义保存模型参数的路径
            output_dir = f'{args.output_dir}/round_{rnd}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 设置模型文件的名称和路径
            model_file_path = os.path.join(output_dir, 'model.bin')
            torch.save(model_parameters, model_file_path)
            logging.info(f"Aggregated parameters saved in round {rnd} at {model_file_path}")
        else:
            logging.info("Server: Parameter aggregation failed or received None parameters")
        return aggregated_parameters


fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    client_resources=client_resources,
    strategy=CustomFedAvg()
)
# fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())