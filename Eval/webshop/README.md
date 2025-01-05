# Webshop Evaluation
## Install Dependencies
Install dependencies of [AgentBench](https://github.com/THUDM/AgentBench/tree/ac57ad0fd30dc2ec6c7e8991fd21eb643d864783).

```
conda activate agent-bench
```
## Inference

Start the task server.
```
python -m src.start_task -a
```

### Greedy

```
bash scripts/Greedy/new_llama8b_base.sh
```

### Sampling

```
bash scripts/Sampling/new_llama8b_sample.sh
```

### ARMAP-R

TBD

### ARMAP-B

First, run the process to obtain the Sampling results.
```
bash scripts/Sampling/new_llama8b_sample.sh
```

Then, use the reward model to obtain the best-of-n results.
```
python scripts/ARMAP-B/calc_sample.py outputs/new_llama8b_sample webshop-std 12345
```

### ARMAP-M

#### port transfer
```
bash script/ARMAP-M/trans_local_llm.sh
bash script/ARMAP-M/trans_rm.sh
```

#### docker setup
```
docker ps
docker exec -it {docker_id}  /bin/bash
```

#### mcts
```
bash script/ARMAP-M/run_mcts.sh
```
#### performance analysis
```
python src/mcts_agents/performance_analysis.py
```