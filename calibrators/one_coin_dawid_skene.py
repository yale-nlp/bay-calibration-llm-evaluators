import pandas as pd
import numpy as np
from crowdkit.aggregation.classification.dawid_skene import OneCoinDawidSkene as OneCoinDawidSkeneBase, MajorityVote
from crowdkit.aggregation.utils import get_most_probable_labels


def calibrate_q_one_coin_dawid_skene(voting_matrix, q_priors=None, **kwargs) -> list:
    model = OneCoinDawidSkene(n_iter=10000, tol=1e-5)
    if q_priors is not None:
        result = model.fit(voting_matrix, skills_prior=pd.Series({f'w_{i}': q_p for i, q_p in enumerate(q_priors)}), **kwargs).skills_.to_numpy()
    else:
        result = model.fit(voting_matrix, **kwargs).skills_.to_numpy()
    return result


class OneCoinDawidSkene(OneCoinDawidSkeneBase):
    def fit(self, data: pd.DataFrame, skills_prior: pd.Series = None, proba_prior: pd.Series = None, **kwargs):
        """Fits the model to the training data with the EM algorithm.
        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.
        Returns:
            DawidSkene: self.
        """

        data = data[['task', 'worker', 'llm_label']]

        # Preprocess for crowdkit requirements
        data = data.rename(columns={'llm_label': 'label'})
        data['worker'] = np.vectorize(lambda x: f'w_{x}')(pd.factorize(data['worker'])[0])
        data['task'] = np.vectorize(lambda x: f't_{x}')(pd.factorize(data['task'])[0])
        if skills_prior.iloc[0].shape[0] == 2:
            skills_prior = skills_prior.apply(lambda x: x[0] / (x[0] + x[1]))

        # Early exit
        if not data.size:
            self.probas_ = pd.DataFrame()
            self.priors_ = pd.Series(dtype=float)
            self.errors_ = pd.DataFrame()
            self.labels_ = pd.Series(dtype=float)
            return self

        # Initialization
        probas = MajorityVote().fit_predict_proba(data)
        priors = proba_prior if proba_prior is not None else probas.mean()
        skills = skills_prior if skills_prior is not None else self._m_step(data, probas)
        errors = self._process_skills_to_errors(data, probas, skills)
        loss = -np.inf
        self.loss_history_ = []

        # Updating proba and errors n_iter times
        for _ in range(self.n_iter):
            probas = self._e_step(data, priors, errors)
            priors = probas.mean()
            skills = self._m_step(data, probas)
            errors = self._process_skills_to_errors(data, probas, skills)
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(data)
            self.loss_history_.append(new_loss)

            if new_loss - loss < self.tol:
                break
            loss = new_loss

        # Saving results
        self.probas_ = probas
        self.priors_ = priors
        self.skills_ = skills
        self.errors_ = errors
        self.labels_ = get_most_probable_labels(probas)

        return self