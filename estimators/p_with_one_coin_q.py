import numpy as np
from utils import estimate_bernoulli_parameter, get_k, get_real_p, plot_p


def estimate_p_with_one_coin_q(voting_matrix, truth_matrix, q_value_list=None, q_sample_list=None, plot_dir='plots', file_name='p_estimate.png', **kwargs):
    alpha_prior, beta_prior = 1, 1
    worker_votes = voting_matrix.groupby('worker').apply(lambda x: x['llm_label'].sum()).to_list()
    worker_counts = voting_matrix.groupby('worker').apply(lambda x: len(x)).to_list()
    if q_sample_list is not None:
        sample_size = q_sample_list.shape[1]
    else:
        sample_size = 10000
    all_k_samples, all_q_samples = [], []
    for i, (k_s, k_n) in enumerate(zip(worker_votes, worker_counts)):
        k_s = k_n - k_s
        all_k_samples.append(estimate_bernoulli_parameter(k_s, k_n, alpha_prior, beta_prior).rvs(sample_size))
        if q_value_list is not None:
            all_q_samples.append(np.array([q_value_list[i]] * sample_size))
            title = r'Distribution of $\hat{{p}}$ ($s_k={}$ $n_k={}$)'.format(k_s, k_n)
        elif q_sample_list is not None:
            all_q_samples.append(np.array(q_sample_list[i, :]))
            title = r'Distribution of $\hat{{p}}$ ($s_k={}$ $n_k={}$)'.format(k_s, k_n)
        else:
            raise ValueError
    # Use all samples
    k_samples = np.concatenate(all_k_samples)
    q_samples = np.concatenate(all_q_samples)
    # Only use the best evaluator
    # best_evaluator = np.argmax(q_sample_list.mean(axis=1))
    # print(f'Best evaluator: {best_evaluator} with q={q_sample_list[best_evaluator].mean()}')
    # k_samples = all_k_samples[best_evaluator]
    # q_samples = all_q_samples[best_evaluator]

    valid_indices = (2 * q_samples - 1) != 0  # Avoid division by zero
    p = (k_samples[valid_indices] - 1 + q_samples[valid_indices]) / (2 * q_samples[valid_indices] - 1)
    # Removing p values that are not within the (0, 1) interval
    p = p[(p > 0) & (p < 1)]
    print(f'Mean {np.mean(p)}')
    return plot_p(p, title, save_dir=plot_dir, file_name=file_name, true_p=get_real_p(voting_matrix), k_as_p=get_k(voting_matrix), truth_mat_p=get_real_p(truth_matrix))
