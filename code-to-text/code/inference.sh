lang=ruby
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../dataset

# Array of languages
languages=("go" "python" "java" "php")

# Loop through each language
for lang in "${languages[@]}"; do
    output_dir=model_client1/$lang
    output_log=$output_dir/inference.log
    dev_file=$data_dir/$lang/valid.jsonl
    test_file=$data_dir/$lang/test.jsonl
    test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
    output_log=$output_dir/inference.log

    python run.py \
        --do_test \
        --model_type roberta \
        --model_name_or_path ../../model/codebert-base \
        --load_model_path $test_model \
        --dev_filename $dev_file \
        --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size 2>&1 | tee $output_log
done