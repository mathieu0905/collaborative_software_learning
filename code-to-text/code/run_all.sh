#!/bin/bash

# Define constant parameters
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
pretrained_model=../../model/codebert-base # Roberta: roberta-base

# Array of languages
languages=("Ruby" "Javascript" "Go" "Python" "Java" "PHP")

# Loop through each language
for lang in "${languages[@]}"; do
    output_dir=model/$lang
    train_dir=$data_dir/$lang/split_jsonl_files
    dev_file=$data_dir/$lang/valid.jsonl
    epochs=1
    model=model/$lang
    output_log=$output_dir/train.log

    # Run the training command
    python client.py \
        --do_train \
        --do_eval \
        --model_type roberta \
        --model_name_or_path $pretrained_model \
        --train_dir $train_dir \
        --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --eval_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epochs 2>&1 | tee $output_log
done
