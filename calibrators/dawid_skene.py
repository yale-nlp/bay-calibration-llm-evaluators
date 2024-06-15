import pandas as pd
import numpy as np
from crowdkit.aggregation.classification.dawid_skene import DawidSkene as DawidSkeneBase, MajorityVote
from crowdkit.aggregation.utils import get_most_probable_labels


def calibrate_q_dawid_skene(voting_matrix, q_priors=None, **kwargs) -> list:
    model = DawidSkene(n_iter=10000, tol=1e-5)
    if q_priors is not None:
        q_priors = pd.Series({f'w_{i}': q_p for i, q_p in enumerate(q_priors)})
    result = model.fit(voting_matrix, q_priors, **kwargs).errors_.reset_index()
    w = len(result['worker'].unique())
    result_list = np.zeros((2, w))
    for i, row in result.iterrows():
        worker = int(row['worker'].split('_')[1])
        if row['label'] == 0:
            result_list[0, worker] = row[0]
        elif row['label'] == 1:
            result_list[1, worker] = row[1]
    return result_list


class DawidSkene(DawidSkeneBase):
    def fit(self, data: pd.DataFrame, q_priors, **kwargs):
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

        if not data.size:
            self.probas_ = pd.DataFrame()
            self.priors_ = pd.Series(dtype=float)
            self.errors_ = pd.DataFrame()
            self.labels_ = pd.Series(dtype=float)
            return self

        # Initialization
        probas = MajorityVote().fit_predict_proba(data)
        priors = probas.mean()
        if q_priors is not None:
            errors = []
            for worker, conf_mat in q_priors.to_dict().items():
                errors.append({'worker': worker, 'label': 0, 0: conf_mat[0, 0] / conf_mat[0].sum(), 1: conf_mat[0, 1] / conf_mat[0].sum()})
                errors.append({'worker': worker, 'label': 1, 0: conf_mat[1, 0] / conf_mat[1].sum(), 1: conf_mat[1, 1] / conf_mat[1].sum()})
            errors = pd.DataFrame.from_dict(errors)
            errors = errors.set_index(['worker', 'label'])
        else:
            errors = self._m_step(data, probas)
        loss = -np.inf
        self.loss_history_ = []

        # Updating proba and errors n_iter times
        for _ in range(self.n_iter):
            probas = self._e_step(data, priors, errors)
            priors = probas.mean()
            errors = self._m_step(data, probas)
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(data)
            self.loss_history_.append(new_loss)

            if new_loss - loss < self.tol:
                break
            loss = new_loss

        probas.columns = pd.Index(probas.columns, name='label', dtype=probas.columns.dtype)
        # Saving results
        self.probas_ = probas
        self.priors_ = priors
        self.errors_ = errors
        self.labels_ = get_most_probable_labels(probas)

        return self
    