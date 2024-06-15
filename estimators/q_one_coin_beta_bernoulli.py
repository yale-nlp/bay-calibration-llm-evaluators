import pandas as pd
import numpy as np
from utils import get_real_q


def estimate_q_one_coin_beta_bernoulli(truth_matrix):
    s_list = truth_matrix.groupby('worker').apply(lambda x: (x['llm_label'] == x['human_label']).sum())
    n_list = truth_matrix.groupby('worker').apply(lambda x: len(x))
    prior_alpha, prior_beta = 1, 1
    beta_parameters = np.array([[prior_alpha + s, prior_beta + n - s] for s, n in zip(s_list.values, n_list.values)])
    print(f"Before calibration: {get_real_q(truth_matrix, 'one_coin')}")
    return beta_parameters