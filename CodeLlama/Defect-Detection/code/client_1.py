from collections import OrderedDict

from centralized import load_data, load_model, train, evaluate, DEVICE, parse_args
from torch.utils.data import DataLoader

import flwr as fl
import torch
import logging
import pickle
import numpy as np
import os

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

args = parse_args()
if args.type == "even":
    logging.basicConfig(filename='training.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
elif args.type == "uneven_nums":
    logging.basicConfig(filename='training_uneven_nums.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
elif args.type == "project":
    logging.basicConfig(filename='training_project.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

else:
    logging.basicConfig(filename='training_uneven_tags.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

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
        train(args, self.model, self.trainset, self.cid)
        return self.get_parameters({}), len(self.trainset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(args, self.model, self.testset)
        return loss, len(self.testset), {"accuracy": acc}
    

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    logging.info(f"Client {cid} is starting training...")
    
    trainset, testset = load_data(cid=cid, args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device}")
    # Create a client-specific Flower client
    return FlowerClient(model.to(device), trainset, testset, cid)


fl.client.start_numpy_client(server_address="localhost:8080", client=client_fn(1))