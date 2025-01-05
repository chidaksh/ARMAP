for idx in {0..9}
    do
    echo "Loop Iteration: $idx"

    RESULT_DIR="outputs/llama8b_sample_seen/llama8b_sample_seen_${idx}"
    python main.py --agent_config local_0.5 \
        --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --exp_config sciworld \
        --split dev \
        --output_path ${RESULT_DIR} \
        --task sample-test \
        --verbose &
    done


