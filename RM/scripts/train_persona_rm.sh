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
MOD_PREFERENCE_DATA=pavo/reformatted_preferences.json
CAPTION_DATA=pavo/pavo_factual.json
IMAGE_FOLDER=pavo_image
PERSONA_TRAIN_DATA=pavo/company_personas.json
# Add a sample file for post-training inference. Make sure this file exists.
INFERENCE_KNOWLEDGE_FILE=pavo/persona/coco_cola_2.txt

# SAVE CONFIG
MODEL_NAME=RM-pavo
MOD_MODEL_NAME=output

# TRAINING CONFIG
NUM_EPOCHS=5
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRAD_ACCUMULATION=16

# --- Train Persona Encoder-Decoder ---
# echo "--- Training Persona Encoder-Decoder ---"
# PERSONA_MODULE_OUTPUT_DIR="$MODEL_DIR/$MODEL_NAME/persona"
# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc-per-node=$GPUS_PER_NODE \
#     train_persona_module.py \
#     --dataset_path $DATA_DIR/$PERSONA_TRAIN_DATA \
#     --output_dir $PERSONA_MODULE_OUTPUT_DIR \
#     --model_name "microsoft/deberta-v3-base" \
#     --epochs $NUM_EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --learning_rate $LEARNING_RATE \
#     --use_lora \
#     --inference_knowledge_file $DATA_DIR/$INFERENCE_KNOWLEDGE_FILE

# echo "--- Finished Training Persona Encoder-Decoder ---"

# Train the reward model
torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE finetune_persona_rm.py \
    --persona_model_checkpoint "./RM-pavo/persona/final_checkpoint" \
    --company_data_path "./trajectories/generated_pos_data.json" \
    --knowledge_base_path "./data/pavo/company_personas.json" \
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
    --dataset_path $DATA_DIR/$MOD_PREFERENCE_DATA \
    --eval_dataset_path $DATA_DIR/$MOD_PREFERENCE_DATA \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 45 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_DIR/$MOD_MODEL_NAME" \
    --persona_dir "$MODEL_DIR/$MODEL_NAME/personas" \
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
    --image_aspect_ratio 'resize' \
    # --data_handling_config "{'data_type': 'pavo', 'image_folder': '$DATA_DIR/$IMAGE_FOLDER', 'caption_file': '$DATA_DIR/$CAPTION_DATA'}"
