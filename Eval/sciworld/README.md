# Sciworld Evaluation
## Install Dependencies
Install dependencies of [Exploration-Based Trajectory Optimization for LLM Agents](https://github.com/Yifan-Song793/ETO/tree/a2fc5da38f8d00cfaf3f9b6370d586eebaf72904).


## Inference


### Greedy

```
bash scripts/Greedy/llama8b_base_seen.sh
```

### Sampling

```
bash scripts/Sampling/llama8b_sample_seen.sh
```

### ARMAP-R

```
bash scripts/ARMAP-R/llama8b_re_seen.sh
```

### ARMAP-B

First, run the process to obtain the Sampling results.
```
bash scripts/Sampling/llama8b_sample_seen.sh
```

Then, use the reward model to obtain the best-of-n results.
```
python scripts/ARMAP-B/calc_sample.py outputs/llama8b_sample_seen 12345
```

### ARMAP-M

TBD
