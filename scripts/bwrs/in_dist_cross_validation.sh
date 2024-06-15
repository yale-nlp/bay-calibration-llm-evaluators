#!/bin/bash

datasets=("Hanna" "Meva" "SummEval" "LLMBar" "MTBench" "LLMEval2" "FairEval")

for dataset in ${datasets[@]}
do
    mkdir -p results/plots/${dataset}-bwrs
    mkdir -p results/logs/${dataset}-bwrs
    python main.py --estimator BetaBernoulli \
                    --dataset ${dataset} \
                    --calibrator None \
                    --compare_models All \
                    --q_prior_cv_folds 5 \
                    --plot_dir results/plots/${dataset}-bwrs/in_dist_cv |& tee results/logs/${dataset}-bwrs/in_dist_cv.log
done
