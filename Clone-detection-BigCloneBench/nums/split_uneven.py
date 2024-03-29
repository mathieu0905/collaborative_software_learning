import json
import pandas as pd
import numpy as np
import os

# 读取 data.jsonl 文件
def read_jsonl_to_df(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 读取 train.txt 文件
def read_train_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    return pd.DataFrame(data, columns=['func1_id', 'func2_id', 'label'])

try:
    # 加载数据
    data_df = read_jsonl_to_df('../dataset/data.jsonl')
    train_df = read_train_txt('../dataset/train.txt')

    # 转换ID和标签为整型
    train_df['func1_id'] = train_df['func1_id'].astype(int)
    train_df['func2_id'] = train_df['func2_id'].astype(int)
    train_df['label'] = train_df['label'].astype(int)

    n_splits = 10

    os.makedirs('split_uneven', exist_ok=True)
    for i in range(n_splits):
        # 从数据中随机选择一部分样本
        split_df = train_df.sample(frac=np.random.uniform(0.03, 0.3), replace=False)

        # 移除已选取的样本
        train_df = train_df.drop(split_df.index)

        # 保存为TXT文件
        split_df.to_csv(f'split_uneven/split_{i}.txt', sep='\t', index=False, header=False)

        # 打印目标为0和1的个数
        print(f"Random Split {i}:")
        print("Label counts:", split_df['label'].value_counts().to_dict())

except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except Exception as e:
    print(f"发生错误: {e}")
