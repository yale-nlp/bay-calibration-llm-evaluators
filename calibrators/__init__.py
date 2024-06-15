from .bayesian_dawid_skene import BayesianDawidSkene
from .dawid_skene import DawidSkene
from .one_coin_dawid_skene import OneCoinDawidSkene

from .calibrators import calibrate_q

# Not yet implemented methods
# class BayesianOneCoinDawidSkene:
#     def __init__(self):
#         pass

#     def fit(self, data: pd.DataFrame, n_samples=1000, n_chains=10, plot_trace=True, 
#             skills_prior_alphas=None, skills_prior_betas=None, 
#             proba_prior: pd.Series = None, golden_labels: pd.DataFrame = None, return_p=False):
#         probas = MajorityVote().fit_predict_proba(data)
#         w = len(data['worker'].unique())
#         t = len(data['task'].unique())
#         c = len(data['label'].unique())

#         with pm.Model() as model:
#             # Prior for class prevalence
#             class_prevalence = pm.Dirichlet('class_prevalence', a=np.ones(c), shape=c)

#             # Skills prior
#             if skills_prior_alphas is not None and skills_prior_betas is not None:
#                 skills = pm.Beta('skills', alpha=skills_prior_alphas, beta=skills_prior_betas, shape=w)
#             else:
#                 # Uniform prior for skills if no prior is provided
#                 skills = pm.Beta('skills', alpha=1., beta=1., shape=w)

#             # Categorical distribution for the labels
#             labels = pm.Categorical('label', p=class_prevalence, shape=t)

#             # Deterministic variable for worker accuracy (not necessary for inference, but useful for inspection)
#             pm.Deterministic('worker_accuracy', skills)

#             # Mapping workers and tasks from data to indices
#             worker_indices = data['worker'].factorize()[0]
#             task_indices = data['task'].factorize()[0]
#             observed_labels = data['label'].values

#             correct_label_prob = skills[worker_indices]
#             incorrect_label_prob = (1 - skills[worker_indices]) / (c - 1)
#             correct_label_prob_matrix = correct_label_prob[:, None] * pt.eq(labels[task_indices][:, None], np.arange(c))
#             incorrect_label_prob_matrix = incorrect_label_prob[:, None] * pt.neq(labels[task_indices][:, None], np.arange(c))
#             final_prob = pm.math.sum(correct_label_prob_matrix + incorrect_label_prob_matrix, axis=1)
#             pm.Categorical('observations', p=final_prob, observed=observed_labels)
#         with model:
#             trace = pm.sample(n_samples, chains=n_chains, cores=n_cores, progressbar=False)
#         if plot_trace:
#             pm.plot_trace(trace, var_names=['worker_accuracy'])
#             plt.show()
#             print(az.summary(trace['posterior']['worker_accuracy']))
#         if return_p:
#             return trace['posterior']['worker_accuracy'].to_numpy(), trace['posterior']['class_prevalence'].to_numpy()
#         else:
#             return trace['posterior']['worker_accuracy'].to_numpy()


# class BayesianGLAD:
#     def __init__(self):
#         pass

#     def fit(self, data: pd.DataFrame, n_samples=1000, plot_trace=True, skills_prior_alpha=1, skills_prior_beta=1, proba_prior: pd.Series = None):
#         w = len(data['worker'].unique())
#         t = len(data['task'].unique())
#         c = len(data['label'].unique())
#         with pm.Model() as model:
#             # label_prior = proba_prior if proba_prior is not None else probas.mean()
#             # if skills_prior is not None:
#             #     skills_prior = [pm.Beta(f'skills_{i}', alpha=skills_prior[i][0], beta=skills_prior[i][1], shape=1) for i in range(w)]

#             # For now, we don't support prior parameters, we will use a uniform prior for skills and label
#             class_prevalence = pm.Dirichlet('class_prevalence', a=np.ones(c), shape=c)
#             skills = pm.Dirichlet('skills', a=np.ones((c, c)) + np.diag(np.ones(c)), shape=(w, c, c))
#             labels = pm.Categorical('label', p=class_prevalence, shape=t)
#             pm.Deterministic(f'worker_accuracy', var=pt.as_tensor_variable([sum(skills[i, j, j] for j in range(c)) / pt.sum(skills[i]) for i in range(w)]))
#             data_workers = data['worker'].apply(lambda x: int(x.split('_')[1])).to_list()
#             data_observations = data['label'].to_list()
#             data_tasks = data['task'].apply(lambda x: int(x.split('_')[1])).to_list()
#             pm.Categorical(f'predictions', p=skills[data_workers, labels[data_tasks]], observed=data_observations)

#         with model:
#             trace = pm.sample(n_samples, progressbar=False)
#         if plot_trace:
#             pm.plot_trace(trace, var_names=['worker_accuracy'])
#             plt.show()
#             print(az.summary(trace['posterior']['worker_accuracy']))
#         return trace['posterior']['worker_accuracy'].to_numpy()
