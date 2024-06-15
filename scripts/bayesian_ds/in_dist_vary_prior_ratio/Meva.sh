#!/bin/bash

datasets=("Meva")

for ratio in $(seq 0.0 0.1 1.0)
do
    for dataset in ${datasets[@]}
    do
        mkdir -p results/plots/${dataset}-bayds
        mkdir -p results/logs/${dataset}-bayds
        python main.py --estimator BetaBernoulli \
                        --dataset ${dataset} \
                        --calibrator BayesianDawidSkene \
                        --compare_models gpt___All \
                        --q_prior_data_ratio ${ratio} \
                        --plot_dir results/plots/${dataset}-bayds/in_dist_${ratio} |& tee results/logs/${dataset}-bayds/in_dist_${ratio}.log
    done
done
