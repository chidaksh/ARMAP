import os
import sys
sys.path.append(".")
from mcts_agents.sci_world_sample import score_sampling


def test_sampling():
    data_dir = "PATH_TO_OUTPUTS_DIR"
    url = "http://172.30.150.31:15678/api/generate"
    out_fn = "sampling_mcts.txt"
    fh = open(out_fn, "w+")
    print(url+"\n")
    fh.flush()
    output_file = os.path.basename(
        data_dir) + "mcts_result.json"
    if os.path.isfile(output_file):
        continue
    score_sampling(data_dir, output_file, url, fh)
    fh.flush()


if __name__ == "__main__":
    test_sampling()
