import pymc as pm
import pandas as pd
import numpy as np
import pytensor.tensor as pt
import arviz as az
from utils import get_k


def calibrate_q_bayesian_dawid_skene(voting_matrix, skills_priors=None, **kwargs):
    model = BayesianDawidSkene()
    result_q_samples, return_p_samples = model.fit(voting_matrix, skills_priors=skills_priors, return_p=True, **kwargs)
    return result_q_samples, return_p_samples.reshape(-1, 2)[:, 0]


class BayesianDawidSkene:
    def __init__(self):
        pass

    def fit(self, data: pd.DataFrame, n_samples=10000, n_chains=None, n_cores=16, plot_trace=True,
            skills_priors=None, proba_prior: pd.Series = None, 
            gold_labels: pd.DataFrame = None, return_p=False):
        if n_chains is None:
            n_chains = n_cores
        # probas = MajorityVote().fit_predict_proba(data)
        w = len(data['worker'].unique())
        t = len(set(data['task'].unique()).union(set(gold_labels['task'].unique())) if gold_labels is not None else data['task'].unique())
        c = len(data['llm_label'].unique())

        with pm.Model() as model:
            # label_prior = proba_prior if proba_prior is not None else data['llm_label'].mean()
            class_prevalence = pm.Dirichlet('class_prevalence', a=np.ones(c), shape=c)
            if skills_priors is not None:
                # OOD
                skills = pm.Dirichlet('skills', a= c * skills_priors / skills_priors.sum(axis=2, keepdims=True), shape=(w, c, c))
            else:
                # No prior
                skills = pm.Dirichlet('skills', a=np.ones((c, c)) + np.diag(np.ones(c)), shape=(w, c, c))
            
            if gold_labels is None:
                labels = pm.Categorical('label', p=class_prevalence, shape=t)
            else:
                labels = []
                all_tasks = list(set(data['task'].unique()).union(set(gold_labels['task'].unique())))
                for i in range(t):
                    gold_label_row = gold_labels[gold_labels['task'] == all_tasks[i]]
                    if len(gold_label_row) > 0:
                        # assert len(gold_label_row) == 1, 'Multiple gold labels for the same task'
                        gold_label = gold_label_row.iloc[0]['human_label'].item()
                        labels.append(pm.Categorical(f'label_{i}', p=class_prevalence, shape=1, observed=np.array([gold_label])))
                    else:
                        labels.append(pm.Categorical(f'label_{i}', p=class_prevalence, shape=1))
                labels = pm.math.concatenate(labels)

            pm.Deterministic(f'worker_accuracy', var=pt.as_tensor_variable([sum(skills[i, j, j] for j in range(c)) / pt.sum(skills[i]) for i in range(w)]))
            data_workers = data['worker'].factorize()[0].tolist()
            data_observations = data['llm_label'].to_list()
            data_tasks = data['task'].factorize()[0].tolist()
            pm.Categorical(f'predictions', p=skills[data_workers, labels[data_tasks]], observed=data_observations)

        with model:
            trace = pm.sample(n_samples, tune=n_samples, chains=n_chains, cores=n_cores, progressbar=False)
        if plot_trace:
            # pm.plot_trace(trace, var_names=['worker_accuracy', 'class_prevalence'])
            # plt.show()
            print(az.summary(trace['posterior']['worker_accuracy']))
            print(az.summary(trace['posterior']['class_prevalence']))
        if return_p:
            return trace['posterior']['worker_accuracy'].to_numpy(), trace['posterior']['class_prevalence'].to_numpy()
        else:
            return trace['posterior']['worker_accuracy'].to_numpy()
