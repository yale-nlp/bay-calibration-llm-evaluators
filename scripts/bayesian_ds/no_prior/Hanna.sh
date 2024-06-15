#!/bin/bash

datasets=("Hanna")

for dataset in ${datasets[@]}
do
    mkdir -p results/plots/${dataset}-bayds
    mkdir -p results/logs/${dataset}-bayds
    python main.py --estimator None \
                    --dataset ${dataset} \
                    --calibrator BayesianDawidSkene \
                    --compare_models GPT-2___All \
                    --plot_dir results/plots/${dataset}-bayds/no_prior |& tee results/logs/${dataset}-bayds/no_prior.log
done
