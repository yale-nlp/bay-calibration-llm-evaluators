#!/bin/bash

# Bayesian Dawid Skene
# no prior
python main.py --estimator None --dataset Hanna --calibrator BayesianDawidSkene --compare_models All --plot_dir results/plots/hanna/no_prior
python main.py --estimator None --dataset Meva --calibrator BayesianDawidSkene --compare_models All --plot_dir results/plots/meva/no_prior
python main.py --estimator None --dataset SummEval --calibrator BayesianDawidSkene --compare_models All --plot_dir results/plots/summeval/no_prior
python main.py --estimator None --dataset LLMBar --calibrator BayesianDawidSkene --compare_models All --plot_dir results/plots/llmbar/no_prior
# in-distribution prior -- vary prior ratio
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --plot_dir results/plots/hanna/in_dist_0.2
python main.py --estimator BetaBernoulli --dataset Meva --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --plot_dir results/plots/meva/in_dist_0.2
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --plot_dir results/plots/summeval/in_dist_0.2
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --plot_dir results/plots/llmbar/in_dist_0.2
# in-distribution prior -- cross-validation
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset Meva --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
# out-of-distribution prior
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator BayesianDawidSkene --compare_models All --q_prior_ood --plot_dir results/plots/hanna/out_dist
python main.py --estimator BetaBernoulli --dataset Meva --calibrator BayesianDawidSkene --compare_models All --q_prior_ood --plot_dir results/plots/meva/out_dist
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator BayesianDawidSkene --compare_models All --q_prior_ood --plot_dir results/plots/summeval/out_dist
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator BayesianDawidSkene --compare_models All --q_prior_ood --plot_dir results/plots/llmbar/out_dist
# golden labels -- vary prior ratio
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --q_prior_data_usage gold_labels --plot_dir /dev/null 
python main.py --estimator BetaBernoulli --dataset Meva --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --q_prior_data_usage gold_labels --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --q_prior_data_usage gold_labels --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator BayesianDawidSkene --compare_models All --q_prior_data_ratio 0.2 --q_prior_data_usage gold_labels --plot_dir /dev/null
# golden labels -- cross-validation
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --q_prior_data_usage gold_labels --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset Meva --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --q_prior_data_usage gold_labels --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --q_prior_data_usage gold_labels --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator BayesianDawidSkene --compare_models All --q_prior_cv_folds 5 --q_prior_data_usage gold_labels --plot_dir /dev/null

# bwrs
# in-distribution prior -- vary prior ratio
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator None --compare_models All --q_prior_data_ratio 0.2 --plot_dir results/plots/hanna-ours/in_dist_0.2
python main.py --estimator BetaBernoulli --dataset Meva --calibrator None --compare_models All  --q_prior_data_ratio 0.2 --plot_dir results/plots/meva-ours/in_dist_0.2
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator None --compare_models All  --q_prior_data_ratio 0.2 --plot_dir results/plots/summeval-ours/in_dist_0.2
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator None --compare_models All  --q_prior_data_ratio 0.2 --plot_dir results/plots/llmbar-ours/in_dist_0.2
# in-distribution prior -- cross-validation
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator None --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset Meva --calibrator None --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator None --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator None --compare_models All --q_prior_cv_folds 5 --plot_dir /dev/null
# out-of-distribution prior
python main.py --estimator BetaBernoulli --dataset Hanna --calibrator None --compare_models All --q_prior_ood --plot_dir results/plots/hanna-ours/out_dist
python main.py --estimator BetaBernoulli --dataset Meva --calibrator None --compare_models All --q_prior_ood --plot_dir results/plots/meva-ours/out_dist
python main.py --estimator BetaBernoulli --dataset SummEval --calibrator None --compare_models All --q_prior_ood --plot_dir results/plots/summeval-ours/out_dist
python main.py --estimator BetaBernoulli --dataset LLMBar --calibrator None --compare_models All --q_prior_ood --plot_dir results/plots/llmbar-ours/out_dist
