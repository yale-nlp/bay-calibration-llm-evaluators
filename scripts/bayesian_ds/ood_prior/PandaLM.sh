#!/bin/bash

datasets=("PandaLM")

for dataset in ${datasets[@]}
do
    mkdir -p results/plots/${dataset}-bayds
    mkdir -p results/logs/${dataset}-bayds
    python main.py --estimator BetaBernoulli \
                    --dataset ${dataset} \
                    --calibrator BayesianDawidSkene \
                    --compare_models llama-7b___All \
                    --q_prior_ood \
                    --plot_dir results/plots/${dataset}-bayds/ood_prior |& tee results/logs/${dataset}-bayds/ood_prior.log
done
