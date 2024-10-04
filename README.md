# Bayesian Calibration of Win Rate Estimation with LLM Evaluators

This is the code repo for EMNLP 2024 paper "Bayesian Calibration of Win Rate Estimation with LLM Evaluators".

## Introduction

Recent advances in large language models (LLMs) show the potential of using LLMs as evaluators for assessing the quality of text generations from LLMs. However, applying LLM evaluators naively to compare different systems can lead to unreliable results due to inaccuracy and intrinsic bias of LLM evaluators. In order to mitigate this problem, we propose two calibration methods, Bayesian Win-Rate Sampling (BWRS) and Bayesian Dawid-Skene, both of which leverage Bayesian inference to more accurately infer the true win rate of generative language models. We empirically validate our methods on six datasets covering story generation, summarization, and instruction following tasks. We show that both our methods are effective in improving the accuracy of win rate estimation using LLMs as evaluators, offering a promising direction for reliable automatic text quality evaluation.

## Datasets

Our datasets can be accessed in [our huggingface organization](https://huggingface.co/bay-calibration-llm-evaluators). These datasets will be accessed by modules under `dataset_loaders`, there's no need to download them manually.

## Run experiments

First, install the required packages:

```pip install -r requirements.txt```

All the experiments can be replicated using the scripts under `scripts/`. To run an experiment, please use:

```bash scripts/<method name>/<prior setting>/<dataset name>.sh```

where `method name` can be one of `bayesian_ds` or `bwrs`, `prior setting` can be one of `in_dist`, `no_prior`, or `ood_prior` for Bayesian Dawid-Skene, and one of `in_dist` or `ood_prior` for BWRS. For example, to run the Bayesian DS experiment with no prior, please use the following command:

```bash scripts/bayesian_ds_no_prior.sh```

The script will run the experiment and save the results to `./results/`. To process the results, please refer to the "process logs" section in `experiments.ipynb`. Specifically, `plot_all_in_dist_error_curves` can be used to produce curves similar to Figure 2, and `parse_repeat_results` can be used to generate result summaries similar to Table 2 and 3 in our paper.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use our code, data, or methodology in your work, please cite our paper:

```

```