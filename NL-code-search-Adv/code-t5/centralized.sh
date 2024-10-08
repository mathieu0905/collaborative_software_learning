python centralized.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=codebert-base \
    --model_name_or_path=codebert-base \
    --tokenizer_name=codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --train_data_dir=../tags/split \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 1 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training True \
    --type even  2>&1 | tee train.log