from math import nan
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import datetime
from dataclasses import dataclass
from scipy.stats import truncnorm, beta, gaussian_kde
from scipy.stats import beta

font = {'size': 16}
matplotlib.rc('font', **font)


def estimate_bernoulli_parameter(s, n, prior_alpha=1, prior_beta=1):
    return beta(prior_alpha + s, prior_beta + n - s)

def sample_q(q_method, q_dist_list, sample_size=1000):
    if q_method == 'OneCoinBetaBernoulli':
        return np.array([beta(*q_dist).rvs(sample_size) for q_dist in q_dist_list])
    elif q_method == 'BetaBernoulli':
        # reverse q_1 because q_dist_list is in the form of confusion matrix
        return np.array([[beta(*q_dist).rvs(sample_size) for q_dist in q_dist_list[:, 0, :]],
                         [beta(*reversed(q_dist)).rvs(sample_size) for q_dist in q_dist_list[:, 1, :]]])
    else:
        raise NotImplementedError(f"Unknown q method {q_method}")

@dataclass
class ResultSamples:
    date: datetime.datetime
    true_p: float
    k: float
    p_mean: float
    p_mode: float
    freq_p: float
    truth_mat_p: float
    p_samples: np.ndarray

def parse_results(dirs, methods):
    results_dir = {'generator_1': [], 'generator_2': [], '|Mean - p|': [], '|Mode - p|': [], '|k - p|': []}
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith('.pkl'):
                generator_1, generator_2 = f.split('.')[0].split('___')
                results_dir['generator_1'].append(generator_1)
                results_dir['generator_2'].append(generator_2)
                result = pickle.load(open(f'{d}/{f}', 'rb'))
                results_dir['|Mean - p|'].append(abs(result.p_mean - result.true_p))
                results_dir['|Mode - p|'].append(abs(result.p_mode - result.true_p))
                results_dir['|k - p|'].append(abs(result.k - result.true_p))
    results_df = pd.DataFrame(results_dir)


def plot_p(p, plot_title, confidence=0.95, sample_size=10000, file_name=None, true_p=None, k_as_p=None, freq_p=None, truth_mat_p=None, save_dir="plots"):
    p = np.sort(p)

    # Plot the distribution of p
    plt.figure(figsize=(10, 6), dpi=100)
    plt.hist(p, density=True, alpha=0.2)
    plt.title(plot_title)
    plt.xlabel('$p$ Values')
    plt.ylabel('Density')

    kde_x_base = np.arange(0, 1, 1e-5)
    if p.shape[0] < 0.5 * sample_size:
        print('p sampling failed, falling back to k')
        if k_as_p is not None:
            p = np.array([k_as_p])
            kde_val = np.array([k_as_p])
            # plot point mass
            plt.scatter(k_as_p, 0, label='k', color='b')
        else:
            print('k is not provided, skipping')
            return nan, nan, nan
    else:
        kde_func = gaussian_kde(p)
        kde_val = np.array(kde_func(kde_x_base))
        plt.plot(kde_x_base, kde_val, label='Gaussian KDE', color='g')
        confidence_lower_x = p[int(len(p) * (1 - confidence))]
        confidence_upper_x = p[int(len(p) * confidence)]
        plt.fill_between(kde_x_base, kde_val, where=(kde_x_base > confidence_lower_x) & (kde_x_base < confidence_upper_x), alpha=0.8)
        plt.text(confidence_lower_x, 0, round(confidence_lower_x, 2), ha='center')
        plt.text(confidence_upper_x, 0, round(confidence_upper_x, 2), ha='center')
        plt.axvline(p.mean(), color='g', linestyle='--', label='estimated p mean')
        plt.axvline(kde_x_base[np.argmax(kde_val)], color='c', linestyle='--', label='estimated p mode')

    if true_p is not None:
        print(f'Difference between true p and estimated p mean: {abs(true_p - p.mean())}')
        print(f'Difference between true p and estimated p mode: {abs(true_p - kde_x_base[np.argmax(kde_val)])}')
        plt.axvline(true_p, color='r', linestyle='--', label='ground truth p')
    if k_as_p is not None:
        print(f'Difference between true p and k: {abs(k_as_p - true_p)}')
        plt.axvline(k_as_p, color='b', linestyle='--', label='k')
    plt.legend()
    if file_name is None:
        plt.show()
    else:
        if save_dir != '/dev/null':
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/{file_name}')
            result = ResultSamples(date=datetime.datetime.now(), 
                                   true_p=true_p, 
                                   k=k_as_p, 
                                   p_mean=p.mean(), 
                                   p_mode=kde_x_base[np.argmax(kde_val)], 
                                   truth_mat_p=truth_mat_p, 
                                   freq_p=freq_p,
                                   p_samples=p)
            pickle.dump(result, open(f'{save_dir}/{file_name}.pkl', 'wb'))
        else:
            # We're not actually writing to /dev/null, just bypassing the save
            pass
    plt.close()
    return abs(true_p - p.mean()), abs(kde_x_base[np.argmax(kde_val)] - true_p), abs(k_as_p - true_p)


def get_real_q(voting_matrix, format='one_coin'):
    if format == 'one_coin':
        return voting_matrix.groupby('worker').apply(lambda x: (x['human_label'] == x['llm_label']).mean())
    elif format == 'conf_mat':
        return voting_matrix.groupby(['worker', 'human_label']).apply(lambda x: (x['human_label'] == x['llm_label']).mean())

def get_real_p(truth_matrix):
    return 1 - truth_matrix['human_label'].mean()

def get_k(voting_matrix):
    return 1 - voting_matrix['llm_label'].mean()
