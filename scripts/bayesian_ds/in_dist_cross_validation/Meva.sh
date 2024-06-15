#!/bin/bash

datasets=("Meva")

for dataset in ${datasets[@]}
do
    mkdir -p results/plots/${dataset}-bayds
    mkdir -p results/logs/${dataset}-bayds
    python main.py --estimator BetaBernoulli \
                    --dataset ${dataset} \
                    --calibrator BayesianDawidSkene \
                    --compare_models gpt___All \
                    --q_prior_cv_folds 5 \
                    --plot_dir results/plots/${dataset}-bayds/in_dist_cv |& tee results/logs/${dataset}-bayds/in_dist_cv.log
done