先执行开启服务端
python server_FedMedian.py(或者另外两种策略：server_FedProx.py, server_FerTrimmedAvg.py)

再开启两个终端，分别执行
export CUDA_VISIBLE_DEVICES=0
bash client_0.sh

export CUDA_VISIBLE_DEVICES=1
bash client_1.sh