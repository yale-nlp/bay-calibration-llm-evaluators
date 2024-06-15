from tqdm import tqdm
import dataset_loaders
from argparse import ArgumentParser
from estimators import estimate_q, estimate_p
from calibrators import calibrate_q
import openai
import itertools
import numpy as np

from utils import get_real_q, plot_p, get_real_p, sample_q, get_k

def parse_args():
    parser = ArgumentParser()
    # dataset arguments
    parser.add_argument("--dataset", type=str, default='RandomSamples')

    parser.add_argument("--q_prior_data_ratio", type=float, default=None) # (Ben) the ratio of data used for q prior estimation. This argument will default to none if not explictitly specified, meaning that no q prior will be used.
    parser.add_argument("--q_prior_data_usage", type=str, default='q_prior', choices=['q_prior', 'gold_labels']) # controls whether we want to use q_prior or gold_labels for utilizing labelled data.
    parser.add_argument("--load_cache", action='store_true')
    parser.add_argument("--q_prior_in_distribution", action='store_true')
    parser.add_argument('--q_prior_ood', dest='q_prior_in_distribution', action='store_false')
    parser.add_argument('--q_prior_ood_source', type=str, choices=['exclude_generators', 'all_others'], default='all_others')

    parser.set_defaults(q_prior_in_distribution=True)

    parser.add_argument("--dataset_p", type=float, default=0.8) # The customized p value for reorganizing the original dataset.

    # calibrator arguments
    parser.add_argument("--calibrator", type=str, default='BayesianDawidSkene')
    parser.add_argument("--calibrator_sample_size", type=int, default=10000)
    parser.add_argument("--calibrator_sample_cores", type=int, default=4)
    parser.add_argument("--q_prior", type=str, default=None)
    
    # estimator arguments
    parser.add_argument("--estimator", type=str, default='Mean')
    parser.add_argument("--plot_dir", type=str, default='results/plots')
    parser.add_argument("--p_sample_size", type=int, default=10000)
    parser.add_argument("--p_confidence", type=float, default=0.95)

    # overall control arguments
    parser.add_argument("--compare_models", type=str, default='All')
    parser.add_argument("--q_prior_cv_folds", type=int, default=-1)

    args = parser.parse_args()
    if args.q_prior is not None:
        args.q_prior = [float(q) for q in args.q_prior.split(',')]
    
    # This is actually possible, but we don't support it for now
    assert 'OneCoin' not in args.calibrator or 'OneCoin' in args.estimator, "Please use a one coin estimator if you use a one coin calibrator"
    assert not (args.q_prior_cv_folds != -1 and args.q_prior_data_ratio is not None), "When doing q_prior cross-validation, please don't specify q_prior_data_ratio since it's not used"
    return args


def do_estimate(args, voting_matrix, truth_matrix, **kwargs):
    # use q value instead of samples in the following cases: user given q, freq dawid-skene, estimator is scalar q
    use_scalar_q = args.q_prior is not None or ('Bayesian' not in args.calibrator and args.calibrator != 'None') or args.estimator == 'Scalar'
    # estimate q or use q_prior
    if args.estimator != 'None':
        if use_scalar_q:
            q_value_list = estimate_q(args.estimator, truth_matrix)
        else:
            q_dist_list = estimate_q(args.estimator, truth_matrix)
    elif args.q_prior is not None:
        q_value_list = args.q_prior
    else:
        q_value_list = None
        q_dist_list = None
    print(f'phat by human label error: {abs(get_real_p(truth_matrix) - get_real_p(voting_matrix))}')
    # calibrate q
    if args.calibrator != 'None':
        if use_scalar_q:
            q_value_list = calibrate_q(args.calibrator, voting_matrix, q_priors=q_value_list, n_samples=args.calibrator_sample_size, n_cores=args.calibrator_sample_cores)
            p_sample_list = None 
            print(f'After calibration:\n{q_value_list.T}')
        elif args.q_prior_data_usage == 'gold_labels':
            q_sample_list, p_sample_list = calibrate_q(args.calibrator, voting_matrix, q_priors=None, n_samples=args.calibrator_sample_size, n_cores=args.calibrator_sample_cores, gold_labels=truth_matrix)
            print(f'After calibration:\n{q_sample_list.mean(axis=(0, 1)).tolist()}')
        else:
            # assert args.q_prior_data_usage == 'q_prior'
            q_sample_list, p_sample_list = calibrate_q(args.calibrator, voting_matrix, q_priors=q_dist_list, n_samples=args.calibrator_sample_size, n_cores=args.calibrator_sample_cores)
            print(f'After calibration:\n{q_sample_list.mean(axis=(0, 1)).tolist()}')
    elif args.q_prior is None:
        # no calibration, use q as estimated
        if not use_scalar_q:
            # sample
            q_sample_list = sample_q(args.estimator, q_dist_list, sample_size=args.p_sample_size)
            p_sample_list = None
        else:
            q_sample_list = None
            p_sample_list = None
    # estimate p
    if use_scalar_q:
        return estimate_p(args.estimator, args.calibrator, voting_matrix, truth_matrix, q_value_list=q_value_list, file_name=f"{kwargs['compare_models']}.png", plot_dir=args.plot_dir, true_p=get_real_p(voting_matrix), k_as_p=get_k(voting_matrix), truth_mat_p=get_real_p(truth_matrix))
    elif p_sample_list is not None:
        # p generated by calibrator
        return plot_p(p_sample_list, 'Distribution of $\hat{{p}}$', file_name=f"{kwargs['compare_models']}.png", true_p=get_real_p(voting_matrix), k_as_p=get_k(voting_matrix), truth_mat_p=get_real_p(truth_matrix), save_dir=args.plot_dir)
    else:
        return estimate_p(args.estimator, args.calibrator, voting_matrix, truth_matrix, q_sample_list=q_sample_list, file_name=f"{kwargs['compare_models']}.png", plot_dir=args.plot_dir, true_p=get_real_p(voting_matrix), k_as_p=get_k(voting_matrix), truth_mat_p=get_real_p(truth_matrix))


def q_prior_cross_validation(args, voting_matrix, truth_matrix, compare_models):
    # split the data into k folds
    
    # split folds by task
    fold_truth_idxs = [np.where(truth_matrix['task'].isin(truth_matrix['task'].unique()[i::args.q_prior_cv_folds]))[0] for i in range(args.q_prior_cv_folds)]
    
    # The following code is deprecated:
    # if args.q_prior_data_usage == 'gold_labels':
    #     # split folds by task
    #     fold_truth_idxs = [np.where(truth_matrix['task'].isin(truth_matrix['task'].unique()[i::args.q_prior_cv_folds]))[0] for i in range(args.q_prior_cv_folds)]
    # else:
    #     idx = np.arange(len(voting_matrix))
    #     np.random.shuffle(idx)
    #     fold_truth_idxs = [idx[i::args.q_prior_cv_folds] for i in range(args.q_prior_cv_folds)]
    
    # do estimate for each fold
    p_hat_errors, p_mode_errors, k_errors = [], [], []
    for i, fold_truth_idx in enumerate(fold_truth_idxs):
        # in cross-validation, we only evaluate on voting matrix not truth matrix, so truth matrix is excluded from voting matrix
        # note that in non-cross-validation, voting matrix contains all tasks, so that q prior is perfect when q_prior_data_ratio is 1
        fold_voting_idx = voting_matrix.index.difference(fold_truth_idx)
        p_hat_error, p_mode_error, k_error = do_estimate(args, voting_matrix.iloc[fold_voting_idx], truth_matrix.iloc[fold_truth_idx], compare_models=compare_models)
        print(f'CV #{i + 1}/{args.q_prior_cv_folds} p_hat_error: {p_hat_error}, p_mode_error: {p_mode_error}, k_error: {k_error}')
        p_hat_errors.append(p_hat_error)
        p_mode_errors.append(p_mode_error)
        k_errors.append(k_error)
    print('CV average:')
    print(f'Difference between true p and estimated p mean: {np.mean(p_hat_errors)}')
    print(f'Difference between true p and estimated p mode: {np.mean(p_mode_errors)}')
    print(f'Difference between true p and k: {np.mean(k_errors)}')


def compare_models(args, voting_matrix, truth_matrix, model1, model2):
    print('\n**************************')
    print(f'Comparing {model1} and {model2}...')
    print('**************************\n')
    if args.q_prior_cv_folds != -1:
        q_prior_cross_validation(args, voting_matrix, truth_matrix, compare_models=f'{model1}___{model2}')
    else:
        do_estimate(args, voting_matrix, truth_matrix, compare_models=f'{model1}___{model2}')
    print('True q:\n', get_real_q(voting_matrix, format='one_coin' if 'OneCoin' in args.estimator else 'conf_mat'))
    print('True p:\n', get_real_p(voting_matrix))

def main():
    args = parse_args()
    dataset_module = getattr(dataset_loaders, args.dataset + 'Dataset')
    # load matrices
    if args.compare_models == 'All':
        models = dataset_module.get_generator_list()
        for model1, model2 in tqdm(itertools.combinations(models, 2), total=len(models) * (len(models) - 1) // 2):
            voting_matrix, truth_matrix = dataset_module.get_matrices(compare_models=f'{model1}___{model2}', use_ood_q=not args.q_prior_in_distribution, q_prior_data_ratio=args.q_prior_data_ratio, q_prior_data_usage=args.q_prior_data_usage, load_cache=args.load_cache, dataset_p=args.dataset_p, q_prior_ood_source=args.q_prior_ood_source)
            compare_models(args, voting_matrix, truth_matrix, model1, model2)
    elif 'All' in args.compare_models:
        # compare one model and every other
        to_compare = args.compare_models.split('___')
        to_compare.remove('All')
        baseline_model = to_compare[0] # to_compare[0] will be the baseline model, i.e. the 1 in the 1 vs. n case.

        # Get the list of all generators in this dataset.
        models = dataset_module.get_generator_list()
        for model in models:
            if model == baseline_model:
                continue # Ignore the baseline model.
            voting_matrix, truth_matrix = dataset_module.get_matrices(
                compare_models=f'{baseline_model}___{model}', 
                use_ood_q=not args.q_prior_in_distribution, 
                q_prior_data_ratio=args.q_prior_data_ratio, 
                q_prior_data_usage=args.q_prior_data_usage, 
                load_cache=args.load_cache, 
                dataset_p=args.dataset_p,
                q_prior_ood_source=args.q_prior_ood_source)
            compare_models(args, voting_matrix, truth_matrix, baseline_model, model)
    else:
        voting_matrix, truth_matrix = dataset_module.get_matrices(compare_models=args.compare_models, use_ood_q=not args.q_prior_in_distribution, q_prior_data_ratio=args.q_prior_data_ratio, q_prior_data_usage=args.q_prior_data_usage, load_cache=args.load_cache, dataset_p=args.dataset_p, q_prior_ood_source=args.q_prior_ood_source)
        compare_models(args, voting_matrix, truth_matrix, *args.compare_models.split('___'))


if __name__ == '__main__':
    main()
