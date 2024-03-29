export CUDA_VISIBLE_DEVICES=0
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
LITFILE=../dataset/javaCorpus/literals.json

OUTPUTDIR=../save_codellama_1_client/javaCorpus
PRETRAINDIR=../save_codellama_1_client/javaCorpus/checkpoint-last     # directory of your saved model
LOGFILE=completion_javaCorpus_eval_codellama_1_client.log

types=("save_codellama_1_client" "save_codellama")
for type in "${types[@]}"; do
    OUTPUTDIR=../$type/javaCorpus
    PRETRAINDIR=../$type/javaCorpus/checkpoint-last     # directory of your saved model
    LOGFILE=completion_javaCorpus_$type.log
    python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --fp16 \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=llama \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=8 \
        --logging_steps=100 \
        --seed=42  2>&1 | tee $LOGFILE
done