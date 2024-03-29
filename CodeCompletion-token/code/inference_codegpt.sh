export CUDA_VISIBLE_DEVICES=0
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion_1_client
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save_codegpt_client_1/javaCorpus
PRETRAINDIR=../save_codegpt_client_1/javaCorpus/checkpoint-last       # directory of your saved model
LOGFILE=completion_javaCorpus_eval_codegpt_1_client.log

python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=100 \
        --seed=42  2>&1 | tee $LOGFILE 