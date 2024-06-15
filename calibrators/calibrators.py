from calibrators.bayesian_dawid_skene import calibrate_q_bayesian_dawid_skene
from calibrators.dawid_skene import calibrate_q_dawid_skene
from calibrators.one_coin_dawid_skene import calibrate_q_one_coin_dawid_skene


def calibrate_q(method, voting_matrix, q_priors=None, **kwargs):
    if method == 'DawidSkene':
        return calibrate_q_dawid_skene(voting_matrix, q_priors, **kwargs)
    elif method == 'OneCoinDawidSkene':
        return calibrate_q_one_coin_dawid_skene(voting_matrix, q_priors, **kwargs)
    elif method == 'BayesianDawidSkene':
        return calibrate_q_bayesian_dawid_skene(voting_matrix, q_priors, **kwargs)
    elif method == 'BayesianOneCoinDawidSkene':
        return calibrate_q_bayesian_one_coin_dawid_skene(voting_matrix, q_priors, **kwargs)
    elif method == 'GLAD':
        return calibrate_q_glad(voting_matrix, q_priors, **kwargs)
    else:
        raise NotImplementedError


def calibrate_q_bayesian_one_coin_dawid_skene(voting_matrix, q_priors=None, **kwargs):
    # model = BayesianOneCoinDawidSkene()
    # result_q_samples = model.fit(voting_matrix, **kwargs)
    # return result_q_samples
    raise NotImplementedError

def calibrate_q_glad(voting_matrix, q_priors=None, **kwargs):
    # model = BayesianGLAD(n_iter=10000, tol=1e-5)
    # if q_priors is not None:
    #     result = model.fit(voting_matrix, **kwargs).alphas_.to_list()
    # else:
    #     result = model.fit(voting_matrix, **kwargs).alphas_.to_list()
    # return result
    raise NotImplementedError