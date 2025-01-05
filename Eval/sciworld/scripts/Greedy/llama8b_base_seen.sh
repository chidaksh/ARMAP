RESULT_DIR="outputs/llama8b_base_seen"
python main.py --agent_config local_base \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --exp_config sciworld \
    --split dev \
    --verbose \
    --output_path ${RESULT_DIR} \
    --task sample-test \
