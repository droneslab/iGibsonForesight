#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

algos=("DDPG" "PPO" "A2C")
envs=("Rs_int" "Beechwood_1_int" "Wainscott_1_int")

for a in "${algos[@]}"; do

    for e in "${envs[@]}"; do


        python main.py --algo "$a" --env "$e"
        # echo "python main.py --algo $a --env $e"

        sleep 10s

    done

done

