BASE_PATH=$1
OUTPUT=$2

# 训练数据路径
if [ "$OUTPUT" == "" ]; then
    OUTPUT=$BASE_PATH/output
fi

mkdir -p $OUTPUT

deepspeed --num_nodes 3 --hostfile /data/hostfile $BASE_PATH/bloom-chat/train.py \
   --model_name_or_path /data/model/bloom-2b5-zh \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 1024 \
   --learning_rate 6e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 6  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --dataloader_num_workers 1 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
