import pandas as pd
from utils import get_real_q


def estimate_q_scalar(truth_matrix):
    return get_real_q(truth_matrix, 'conf_mat').to_numpy().reshape([2, -1]).tolist()