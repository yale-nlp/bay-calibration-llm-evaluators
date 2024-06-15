import numpy as np
from utils import estimate_bernoulli_parameter, get_k, get_real_p, plot_p


def estimate_p_with_conf_mat_q(voting_matrix, truth_matrix, q_value_list=None,
                                         q_sample_list=None,
                                         plot_dir='plots', file_name='p_estimate.png', **kwargs):
    alpha_prior, beta_prior = 1, 1
    # number of 1 votes
    worker_votes = voting_matrix.groupby('worker').apply(lambda x: x['llm_label'].sum()).to_list()
    # total number of votes
    worker_counts = voting_matrix.groupby('worker').apply(lambda x: len(x)).to_list()
    if q_sample_list is not None:
        q_0_sample_list = q_sample_list[0]
        q_1_sample_list = q_sample_list[1]
        sample_size = q_0_sample_list.shape[1]
    elif q_value_list is not None:
        q_0_value_list = q_value_list[0]
        q_1_value_list = q_value_list[1]
        sample_size = 10000
    else:
        raise ValueError('Either q_value_list or q_sample_list must be provided')
    
    all_k_samples, all_q_0_samples, all_q_1_samples = [], [], []
    for i, (k_s, k_n) in enumerate(zip(worker_votes, worker_counts)):
        # number of 0 votes
        k_s = k_n - k_s
        
        if q_value_list is not None:
            # scalar k, fake samples
            all_k_samples.append(np.array([k_s / k_n] * sample_size))
            all_q_0_samples.append(np.array([q_0_value_list[i]] * sample_size))
            all_q_1_samples.append(np.array([q_1_value_list[i]] * sample_size))
            title = r'Distribution of $\hat{{p}}$ ($s_k={}$ $n_k={}$)'.format(k_s, k_n)
        elif q_sample_list is not None:
            # sample k, q
            all_k_samples.append(estimate_bernoulli_parameter(k_s, k_n, alpha_prior, beta_prior).rvs(sample_size))
            all_q_0_samples.append(np.array(q_0_sample_list[i, :]))
            all_q_1_samples.append(np.array(q_1_sample_list[i, :]))
            title = r'Distribution of $\hat{{p}}$ ($s_k={}$ $n_k={}$)'.format(k_s, k_n)
        else:
            raise ValueError
    # Use all samples
    k_samples = np.concatenate(all_k_samples)
    q_0_samples = np.concatenate(all_q_0_samples)
    q_1_samples = np.concatenate(all_q_1_samples)
    # Only use the best evaluator
    # TODO implement this (if needed)
    # best_evaluator = np.argmax(q_sample_list.mean(axis=1))
    # print(f'Best evaluator: {best_evaluator} with q={q_sample_list[best_evaluator].mean()}')
    # k_samples = all_k_samples[best_evaluator]
    # q_samples = all_q_samples[best_evaluator]

    valid_indices = (q_0_samples + q_1_samples - 1) != 0  # Avoid division by zero
    p = (k_samples[valid_indices] - 1 + q_1_samples[valid_indices]) / (q_0_samples[valid_indices] + q_1_samples[valid_indices] - 1)
    # Removing p values that are not within the (0, 1) interval
    p = p[(p > 0) & (p < 1)]
    print(f'Mean {np.mean(p)}')
    # q1 = len(voting_matrix[(voting_matrix['llm_label'] == 1) & (voting_matrix['human_label'] == 1)]) / len(voting_matrix[voting_matrix['human_label'] == 1])
    # q0 = len(voting_matrix[(voting_matrix['llm_label'] == 0) & (voting_matrix['human_label'] == 0)]) / len(voting_matrix[voting_matrix['human_label'] == 0])
    # k = get_k(voting_matrix)
    # return (k + q1 - 1) / (q0 + q1 - 1)


    return plot_p(p, title, save_dir=plot_dir, file_name=file_name, true_p=get_real_p(voting_matrix), k_as_p=get_k(voting_matrix), truth_mat_p=get_real_p(truth_matrix), sample_size=sample_size)
