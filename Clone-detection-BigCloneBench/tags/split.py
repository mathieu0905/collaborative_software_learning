import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

def read_jsonl_to_df(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def read_train_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    return pd.DataFrame(data, columns=['func1_id', 'func2_id', 'label'])


try:
    data_df = read_jsonl_to_df('../dataset/data.jsonl')
    train_df = read_train_txt('../dataset/train.txt')

    train_df['func1_id'] = train_df['func1_id'].astype(int)
    train_df['func2_id'] = train_df['func2_id'].astype(int)
    train_df['label'] = train_df['label'].astype(int)

    skf = StratifiedKFold(n_splits=10)
    os.makedirs('splits_even', exist_ok=True)

    label_counts = []
    for i, (_, test_index) in enumerate(skf.split(train_df, train_df['label'])):
        split_df = train_df.iloc[test_index]
        split_df.to_csv(f'splits_even/split_{i}.txt', index=False, header=None, sep='\t', mode='w')

        label_count = split_df['label'].value_counts().to_dict()
        label_counts.append(label_count)
        print(f"Split {i} label counts: {label_count}")

    saved_files = [f'splits_even/split_{i}.txt' for i in range(10)]
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error: {e}")

