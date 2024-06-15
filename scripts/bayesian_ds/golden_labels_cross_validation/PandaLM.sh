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
                    --q_prior_cv_folds 5 \
                    --q_prior_data_usage gold_labels \
                    --plot_dir results/plots/${dataset}-bayds/gold_labels_cv |& tee results/logs/${dataset}-bayds/gold_labels_cv.log
done
