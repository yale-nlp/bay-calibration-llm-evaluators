#!/bin/bash

datasets=("LLMEval2")

for ratio in $(seq 0.1 0.1 1.0)
do
    for dataset in ${datasets[@]}
    do
        mkdir -p results/plots/${dataset}-bwrs
        mkdir -p results/logs/${dataset}-bwrs
        python main.py --estimator BetaBernoulli \
                        --dataset ${dataset} \
                        --calibrator None \
                        --compare_models All \
                        --q_prior_data_ratio ${ratio} \
                        --plot_dir results/plots/${dataset}-bwrs/in_dist_${ratio} |& tee results/logs/${dataset}-bwrs/in_dist_${ratio}.log
    done
done
