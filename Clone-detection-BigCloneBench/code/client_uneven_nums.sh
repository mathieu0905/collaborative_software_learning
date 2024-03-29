#!/bin/bash
python client_uneven_nums.py \
    --output_dir=./saved_models_uneven_nums \
    --model_type=roberta \
    --config_name=../../../model/codebert-base/config.json \
    --model_name_or_path=../../../model/codebert-base \
    --tokenizer_name=../../../model/codebert-base \
    --train_data_dir=../../nums/split_uneven/ \
    --eval_data_file=../../dataset/valid.txt \
    --test_data_file=../../dataset/valid.txt\
    --epoch 1 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training True \
    --seed 123456 2>&1| tee train_uneven_nums.log &
