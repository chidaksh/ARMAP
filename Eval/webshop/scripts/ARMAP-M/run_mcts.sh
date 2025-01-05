llm_port=7780
rm_port=15678
python src/mcts_agents/env_webshop_wrapper.py \
        --llm_api http://172.17.0.1:${llm_port}/v1 \
        --rm_api http://172.17.0.1:${rm_port}/api/generate \
        --seq_num 10 \
        --horizon 10 \
        --rollouts 10 \
        --with_vllm_api \
        --temperature 1.5 \
        --output_dir ./outputs/mcts_sampling_temp_15 \