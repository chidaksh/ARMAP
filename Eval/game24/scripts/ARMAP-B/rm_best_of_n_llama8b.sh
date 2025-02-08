LM_API=""
MD_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
echo ${LM_API}
python run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 1000 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 100 \
    --temperature 0.7 \
    --llm_api ${LM_API} \
    --backend ${MD_ID} \
