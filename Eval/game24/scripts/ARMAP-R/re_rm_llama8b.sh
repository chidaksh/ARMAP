#LM_API="http://9.33.168.174:7778/v1"
LM_API="http://localhost:17777/v1"
MD_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
echo ${LM_API}
ulimit -n 10000
python run_re.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 1000 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 1 \
    --temperature 0.0 \
    --llm_api ${LM_API} \
    --backend ${MD_ID} \
    --reward http://localhost:15601/api/generate \
    --re_iterations 10 \
