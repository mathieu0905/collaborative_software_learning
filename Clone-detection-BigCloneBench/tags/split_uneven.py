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
    split_size = len(train_df) // n_splits

    os.makedirs('split_uneven', exist_ok=True)
    for i in range(n_splits):
        selected_labels = np.random.choice(train_df['label'].unique(), size=np.random.randint(1, len(train_df['label'].unique())), replace=False)
        split_df = train_df[train_df['label'].isin(selected_labels)].sample(frac=0.1, replace=False)

        if len(split_df) < split_size:
            additional_samples = train_df[~train_df.index.isin(split_df.index)].sample(n=split_size - len(split_df), replace=False)
            split_df = pd.concat([split_df, additional_samples])

        train_df = train_df.drop(split_df.index)

        # 保存为文本文件
        split_df.to_csv(f'split_uneven/split_{i}.txt', index=False, header=None, sep='\t', mode='w')

        # print(f"Uneven Split {i} saved.")
        # 统计并输出标签为0和1的数量
        label_counts = split_df['label'].value_counts().to_dict()
        print(f"Uneven Split {i} label counts: {label_counts}")

except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except Exception as e:
    print(f"发生错误: {e}")
