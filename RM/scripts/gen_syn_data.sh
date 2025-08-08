#!/bin/bash
set -e
set -x

# python syndata/personas/gen_personas.py

python syndata/trajectories/main.py

python syndata/trajectories/reformat.py --input_file syndata/trajectories/generated_trajectories.json --output_file data/pavo/reformatted_preferences.json
