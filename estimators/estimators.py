from utils import estimate_bernoulli_parameter, plot_p, get_real_p, get_k
from estimators import estimate_q_beta_bernoulli, estimate_q_scalar, estimate_q_one_coin_beta_bernoulli, estimate_p_with_one_coin_q, estimate_p_with_conf_mat_q

def estimate_q(method, truth_matrix):
    if method == 'BetaBernoulli':
        return estimate_q_beta_bernoulli(truth_matrix)
    elif method == 'OneCoinBetaBernoulli':
        return estimate_q_one_coin_beta_bernoulli(truth_matrix)
    elif method == 'Scalar':
        return estimate_q_scalar(truth_matrix)
    elif method == 'None':
        return None
    else:
        raise ValueError(f"Unknown method {method}")

def estimate_p(q_method, q_calibration_method, voting_matrix, truth_matrix, **kwargs):
    if q_method == 'BetaBernoulli' and q_calibration_method == 'None':
        # BetaBernoulli without calibration
        return estimate_p_with_conf_mat_q(voting_matrix, truth_matrix, **kwargs)
    elif q_method == 'OneCoinBetaBernoulli' and q_calibration_method == 'None':
        # OneCoinBetaBernoulli without calibration
        return estimate_p_with_one_coin_q(voting_matrix, truth_matrix, **kwargs)
    elif q_method == 'Scalar' and q_calibration_method == 'None':
        # Scalar without calibration, currently only support conf_mat scalar
        return estimate_p_with_conf_mat_q(voting_matrix, truth_matrix, **kwargs)
    elif q_calibration_method == 'DawidSkene':
        # Frequentist DawidSkene
        return estimate_p_with_conf_mat_q(voting_matrix, truth_matrix, **kwargs)
    elif q_calibration_method == 'OneCoinDawidSkene':
        # Frequentist OneCoinDawidSkene
        return estimate_p_with_one_coin_q(voting_matrix, truth_matrix, **kwargs)
    else:
        raise NotImplementedError
