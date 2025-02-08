LM_API=""
RM_API=""
echo ${LM_API}
echo ${RM_API}
python run_mcts.py \
    --task game24MCTS \
    --task_start_index 900 \
    --task_end_index 1000 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --mcts_run \
    --prompt_sample cot \
    --output_dir output/llm380b4bit_rm07_debug_v3 \
    --rollouts 100 \
    --llm_api ${LM_API} \
    --reward_func ${RM_API} \
    --temperature 0.7 \
    --max_token 999 \
    ${@}

#--method_generate sample \
#--task_start_index 900 \
