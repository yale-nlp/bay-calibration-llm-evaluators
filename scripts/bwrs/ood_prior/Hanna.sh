#!/bin/bash

# "LLMBar" "MTBench" "LLMEval2" is not able to do ood_prior with no generator information
datasets=("Hanna")

for dataset in ${datasets[@]}
do
    mkdir -p results/plots/${dataset}-bwrs
    mkdir -p results/logs/${dataset}-bwrs
    python main.py --estimator BetaBernoulli \
                    --dataset ${dataset} \
                    --calibrator None \
                    --compare_models All \
                    --q_prior_ood \
                    --plot_dir results/plots/${dataset}-bwrs/ood_prior |& tee results/logs/${dataset}-bwrs/ood_prior.log
done
