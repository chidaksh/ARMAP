# Reward Model
## Install Dependencies
Firstly, install dependencies of [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF/tree/830a083fd83e607da3c35f2e6aef833523c1e722/RLHF)


Then:
```
pip install -e git+ssh://git@github.com/Efficient-Large-Model/VILA.git@d7d54bc4ca1e582f59516ba2f94a0217ad2430a0#egg=vila

pip install -r requirements.txt
```

## Prepare Models and Datas
Download [`VILA1.5-3b`](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b/tree/main) and put it into `rm`.

If you don't want to train the model, we provide pre-trained [LoRA checkpoints](https://huggingface.co/Heaplax/ARMAP-RM-LoRA), which you can download and place in the `rm` folder.

Otherwise, download training data of [`Webshop`](https://huggingface.co/datasets/Heaplax/ARMAP-RM-WebShop), [`SciWorld`](https://huggingface.co/datasets/Heaplax/ARMAP-RM-SciWorld), [`Game24`](https://huggingface.co/datasets/Heaplax/ARMAP-RM-Game24) and [`ALFWorld`](https://huggingface.co/datasets/Heaplax/ARMAP-RM-ALFWorld), and put them into `data`

## Train Reward Model

### game24
```
bash scripts/train_reward_game24.sh
```
### webshop
```
bash scripts/train_reward_webshop.sh
```
### sciworld
```
bash scripts/train_reward_sciworld.sh
```
### alfworld
```
bash scripts/train_reward_alfworld.sh
```

## Start Reward Model Server

### sciworld
Replace FLASK_PORT with the port you want the server to run on, and LORA_CKPT with the LoRA checkpoint you want to use.
<details>
 <summary><b>scripts/server_reward_sciworld.sh</b></summary>

```
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
```

</details>
