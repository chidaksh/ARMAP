reward_url=15678
model_port=7780
reward_func=http://172.30.150.31:${reward_url}/api/generate
model_url="http://172.30.150.31:${model_port}/v1"
start_id=0
echo $job_id
echo $reward_func
echo $model_url
echo $start_id

python main_mcts.py --agent_config openllm_llama31_70b_mcts \
    --model_name llama-3.1-70b \
    --exp_config sciworld \
    --split dev \
    --verbose \
    --exp_name _mcts_vllm_llama31_70b \
    --seq_num 10 \
    --horizon 10 \
    --rollouts 10 \
    --start_id ${start_id} \
    --minimal_sample_num 10 \
    --model_url_add ${model_url} \
    --reward_func ${reward_func}