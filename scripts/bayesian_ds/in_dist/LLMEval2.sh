#!/bin/bash

datasets=("LLMEval2")

for ratio in $(seq 0.0 0.1 1.0)
do
    for dataset in ${datasets[@]}
    do
        mkdir -p results/plots/${dataset}-bayds
        mkdir -p results/logs/${dataset}-bayds
        python main.py --estimator BetaBernoulli \
                        --dataset ${dataset} \
                        --calibrator BayesianDawidSkene \
                        --compare_models All \
                        --q_prior_data_ratio ${ratio} \
                        --q_prior_data_usage gold_labels \
                        --plot_dir results/plots/${dataset}-bayds/gold_labels_${ratio} |& tee results/logs/${dataset}-bayds/gold_labels_${ratio}.log
    done
done
