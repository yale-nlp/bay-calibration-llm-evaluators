# Bayesian Calibration of Win Rate Estimation with LLM Evaluators

This is the code repo for EMNLP 2024 paper "Bayesian Calibration of Win Rate Estimation with LLM Evaluators".

## Datasets

Our datasets will be released publicly on huggingface after the ARR cycle ends. These repos will be accessed by modules under `dataset_loaders`, there's no need to download them manually. During the ARR cycle, the datasets are available in uploaded files in the format of pandas Dataframe.

## Run experiments

First, install the required packages:

```pip install -r requirements.txt```

The scripts under `scripts/` can be used to run the experiments. For example, to run the Bayesian DS experiment with no prior, run the following command:

```bash scripts/bayesian_ds_no_prior.sh```

Please note that the Bayesian experiments may take a long time to run. To run all experiments that replicates our results, please run:

```bash scripts/run_all.sh```

The script will run the experiment and save the results to `./results/`. To process the results, please refer to the "process logs" section in `experiments.ipynb`.