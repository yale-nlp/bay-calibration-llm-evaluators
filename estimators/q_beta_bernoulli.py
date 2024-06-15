import pandas as pd
import numpy as np
from utils import get_real_p, get_real_q


def estimate_q_beta_bernoulli(truth_matrix):
    s_0_list = truth_matrix.groupby('worker').apply(lambda x: ((x['llm_label'] == 0) & (x['human_label'] == 0)).sum())
    s_1_list = truth_matrix.groupby('worker').apply(lambda x: ((x['llm_label'] == 1) & (x['human_label'] == 1)).sum())
    n_0_list = truth_matrix.groupby('worker').apply(lambda x: (x['human_label'] == 0).sum())
    n_1_list = truth_matrix.groupby('worker').apply(lambda x: (x['human_label'] == 1).sum())
    prior_alpha, prior_beta = 1, 1
    beta_parameters = np.array([[[prior_alpha + s_0, prior_beta + n_0 - s_0], [prior_beta + n_1 - s_1, prior_alpha + s_1]] for s_0, n_0, s_1, n_1 in zip(s_0_list.values, n_0_list.values, s_1_list.values, n_1_list.values)])
    print(f"Before calibration:\n{get_real_q(truth_matrix, 'conf_mat')}")
    return beta_parameters