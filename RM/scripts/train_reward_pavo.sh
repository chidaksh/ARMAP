#!/bin/bash

set -e
set -x


export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR="data"
export MODEL_DIR="."
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=2
export OMP_NUM_THREADS=8

# MODEL CONFIG
VISION_TOWER=VILA1.5-3b/vision_tower
LM_MODEL_NAME=VILA1.5-3b
MM_PROJECTOR=VILA1.5-3b/mm_projector

# DATA CONFIG
PREFERENCE_DATA=pavo/pavo_preference.json
CAPTION_DATA=pavo/pavo_factual.json
IMAGE_FOLDER=pavo_image

# SAVE CONFIG
MODEL_NAME=RM-pavo

# TRAINING CONFIG
NUM_EPOCHS=10
LEARNING_RATE=1e-5
BATCH_SIZE=8
GRAD_ACCUMULATION=2

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME/llm \
    --tokenizer $MODEL_DIR/$LM_MODEL_NAME/llm \
    --image_folder $DATA_DIR/$IMAGE_FOLDER \
    --vision_tower $MODEL_DIR/$VISION_TOWER \
    --mm_projector_path  $MODEL_DIR/$MM_PROJECTOR \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature  cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter False \
    --model_max_length 4096 \
    --query_len 4000 \
    --response_len 2000 \
    --dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --eval_dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 500 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 10 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --reward_prompt_file "./prompts/multi_frame_prompt.txt" \
    --image_to_caption_file $DATA_DIR/$CAPTION_DATA \
    --image_aspect_ratio 'resize'
