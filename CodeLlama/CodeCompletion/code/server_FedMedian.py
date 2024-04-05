import flwr as fl
import torch
import logging
from collections import OrderedDict
import os
import numpy as np
from flwr.common import Parameters

def parameters_to_ndarrays(params: Parameters):
    """将Parameters对象转换为NumPy数组列表"""
    return [np.array(p, dtype=np.float32) for p in params.tensors]

# 定义聚合策略（例如，使用内置的FedAvg策略）
class CustomFedMedian(fl.server.strategy.FedMedian):
    def aggregate_fit(self, rnd: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        self.final_parameters = aggregated_parameters
        self.save_final_model(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

    def save_final_model(self, parameters: Parameters):
        # 这里编写保存模型参数的代码
        # 例如，转换参数并保存为NumPy文件
        try:
            np.savez("models/FedMedian/model.npz", *parameters.tensors)
            logging.info("Model parameters saved successfully.")
        except Exception as e:
            logging.error(f"Error saving model parameters: {e}")

if __name__ == "__main__":
    # 启动Flower服务器，设定轮数和聚合策略
    fl.server.start_server(
        server_address="localhost:8081", 
        config=fl.server.ServerConfig(num_rounds=5), 
        strategy=CustomFedMedian()  # 使用自定义的聚合策略
    )

