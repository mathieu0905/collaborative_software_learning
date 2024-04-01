lang=ruby #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
epochs=10
pretrained_model=codebert-base #Roberta: roberta-base

# Array of languages
languages=("ruby" "javascript" "go" "python" "java" "php")

# Loop through each language
for lang in "${languages[@]}"; do
    output_dir=model_client1/$lang
    train_file=$data_dir/$lang/split_jsonl_files/subset_0.jsonl
    dev_file=$data_dir/$lang/valid.jsonl
    python run.py \
        --do_train \
        --do_eval \
        --model_type roberta \
        --model_name_or_path $pretrained_model \
        --train_filename $train_file \
        --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --eval_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epochs 2>&1 | tee $output_dir/train.log
done