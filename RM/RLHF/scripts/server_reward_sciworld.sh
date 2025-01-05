set -e
set -x

export FLASK_PORT=15678

export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR="rm"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=1
export OMP_NUM_THREADS=1

# MODEL CONFIG
VISION_TOWER=VILA1.5-3b/vision_tower
LM_MODEL_NAME=VILA1.5-3b
MM_PROJECTOR=VILA1.5-3b/mm_projector

# SAVE CONFIG
MODEL_NAME=RM-sciworld
LORA_CKPT=checkpoint-120

# TRAINING CONFIG
NUM_EPOCHS=3
LEARNING_RATE=5e-4
BATCH_SIZE=1
GRAD_ACCUMULATION=1

python server_lora_rm.py \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME/llm \
    --vision_tower $MODEL_DIR/$VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 4096 \
    --query_len 4000 \
    --response_len 2000 \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 1 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --lora_dir "$MODEL_DIR/$MODEL_NAME/$LORA_CKPT" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 10 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --image_aspect_ratio 'resize'
