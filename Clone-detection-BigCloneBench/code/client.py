from collections import OrderedDict

from centralized import load_data, load_model, train, evaluate, DEVICE, parse_args
from torch.utils.data import DataLoader

import flwr as fl
import torch
import ray
import logging
import pickle
import numpy as np

logging.basicConfig(filename='training.log', level=logging.INFO,
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

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures
    ):
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
    
    def aggregate_fit(self, rnd, results, failures):
        logging.info(f"Server: Aggregating parameters in round {rnd}")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(model.state_dict(), f"model_round_{rnd}.pth")

        return aggregated_parameters, aggregated_metrics
    

NUM_CLIENTS = 2
client_resources = {"num_cpus": 16, "num_gpus": 1.0}
# client_fn(1)
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=2),
    client_resources=client_resources,
    strategy=CustomFedAvg()
)
# fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())