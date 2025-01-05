RESULT_DIR="outputs/llama8b_re_seen"
python main_re.py --agent_config local_base \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --exp_config sciworld \
    --split dev \
    --verbose \
    --output_path ${RESULT_DIR} \
    --task sample-test \
    --rm http://0.0.0.0:15678/api/generate \
    --threshold 7.25 \
    --re_iters 10 \
