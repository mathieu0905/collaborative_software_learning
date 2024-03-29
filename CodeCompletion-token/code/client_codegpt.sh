LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/group
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save_codegpt_fl/javaCorpus
PRETRAINDIR=microsoft/CodeGPT-small-java       # microsoft/CodeGPT-small-py for py150
LOGFILE=completion_javaCorpus_codegpt_fl.log
PER_NODE_GPU=1       # modify YOUR_GPU_NUM

python client_codegpt.py \
    --data_dir=$DATADIR \
    --lit_file=$LITFILE \
    --langs=$LANG \
    --cache_dir=/data/model \
    --output_dir=$OUTPUTDIR \
    --pretrain_dir=$PRETRAINDIR \
    --log_file=$LOGFILE \
    --model_type=gpt2  \
    --block_size=1024 \
    --do_train \
    --gpu_per_node $PER_NODE_GPU \
    --learning_rate=8e-5 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=1 \
    --logging_steps=100 \
    --save_steps=1000 \
    --seed=42 \
    --overwrite_output_dir \
    --not_pretrain 2>&1 | tee $LOGFILE

