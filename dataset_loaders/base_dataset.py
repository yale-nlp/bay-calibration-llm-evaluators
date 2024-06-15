import datasets
import numpy as np
from abc import ABC, abstractmethod
from utils import get_k, get_real_q

class BaseDataset(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @classmethod
    def get_matrices(cls, dataset_name, use_ood_q=False, q_prior_data_for_gold_labels=False, **kwargs):
        assert not (use_ood_q and q_prior_data_for_gold_labels), "Gold labels not supported for ood q."

        df = datasets.load_dataset(dataset_name)['train'].to_pandas()
        model_1, model_2 = kwargs['compare_models'].split('___')

        all_models = cls.get_generator_list(df)
        assert model_1 in all_models and model_2 in all_models, "Models not found in dataset"
        df = df.rename(columns={'task_id': 'task', 'worker_id': 'worker'})
        # drop human
        df = df[(df['generator_1'] != 'Human') & (df['generator_2'] != 'Human')]
        if kwargs['q_prior_data_ratio'] is None and not use_ood_q: # This if block essentially handles the case where no prior information is used, and the entire filtered dataset is used as both the voting matrix and the truth matrix for further processing or evaluation.
            # no prior
            df = df[((df['generator_1'] == model_1) & (df['generator_2'] == model_2)) | ((df['generator_1'] == model_2) & (df['generator_2'] == model_1))]
            return df.reset_index(drop=True), df.reset_index(drop=True)
        elif not use_ood_q:
            # in distribution q
            df = df[((df['generator_1'] == model_1) & (df['generator_2'] == model_2)) | ((df['generator_1'] == model_2) & (df['generator_2'] == model_1))]
            df = cls.drop_too_few_workers(df)
            if q_prior_data_for_gold_labels:
                all_tasks = df['task'].unique()
                selected_tasks = np.random.choice(all_tasks, int(len(all_tasks) * kwargs['q_prior_data_ratio']), replace=False)
                truth_matrix = df.groupby('task').first().reset_index()
                truth_matrix = truth_matrix[truth_matrix['task'].isin(selected_tasks)]
            else:
                truth_matrix = df.groupby('worker').apply(lambda x: x.sample(frac=kwargs['q_prior_data_ratio'])).reset_index(drop=True)
            voting_matrix = df
            # voting_matrix, truth_matrix = cls.drop_bad_prediction_workers(voting_matrix, truth_matrix)
            return voting_matrix, truth_matrix
        else:
            # out of distribution q
            all_generators = cls.get_generator_list()
            # all_generators.remove(model_1)
            # all_generators.remove(model_2)
            # find most similar generator pair in terms of k
            voting_matrix = df[((df['generator_1'] == model_1) & (df['generator_2'] == model_2)) | ((df['generator_1'] == model_2) & (df['generator_2'] == model_1))]
            true_k = get_k(voting_matrix)
            best_k = -10
            for generator_1 in all_generators:
                for generator_2 in all_generators:
                    if generator_1 == generator_2 or (model_1 == generator_1 and model_2 == generator_2) or (model_1 == generator_2 and model_2 == generator_1):
                        continue
                    temp_df = df[((df['generator_1'] == generator_1) & (df['generator_2'] == generator_2)) | ((df['generator_1'] == generator_2) & (df['generator_2'] == generator_1))]
                    temp_k = get_k(temp_df)
                    if abs(true_k - temp_k) < abs(true_k - best_k):
                        best_k = temp_k
                        best_df = temp_df
            print('best ood error:', abs(true_k - best_k))
            truth_matrix = best_df
            # if kwargs['q_prior_ood_source'] == 'exclude_generators':
            #     # also exclude any current generator vs other generators
            #     truth_matrix = df[(df['generator_1'] != model_1) & (df['generator_2'] != model_2) & (df['generator_1'] != model_2) & (df['generator_2'] != model_1)]
            # elif kwargs['q_prior_ood_source'] == 'all_others':
            #     # only exclude current generator vs current generator
            #     truth_matrix = df[((df['generator_1'] != model_1) & (df['generator_2'] != model_1)) | ((df['generator_2'] != model_2) & (df['generator_1'] != model_2))]
            # voting_matrix = voting_matrix.reset_index(drop=True)
            # truth_matrix = truth_matrix.reset_index(drop=True)
            # # voting_matrix, truth_matrix = cls.drop_bad_prediction_workers(voting_matrix, truth_matrix)
            return voting_matrix, truth_matrix
    
    @classmethod
    def get_generator_list(cls, df=None) -> list:
        if df is None:
            df = datasets.load_dataset(cls.dataset_name)['train'].to_pandas()
        generator_list = df[['generator_1', 'generator_2']].stack().unique().tolist()
        if 'Human' in generator_list:
            generator_list.remove('Human')
        return generator_list
    
    @classmethod
    def drop_too_few_workers(cls, df, threshold=10):
        # drop worker if too few data points
        worker_counts = df['worker'].value_counts()
        workers_to_drop = worker_counts[worker_counts < threshold]
        return df[~df['worker'].isin(workers_to_drop)]

    @classmethod
    def drop_bad_prediction_workers(cls, voting_matrix, truth_matrix):
        annotators = voting_matrix['worker'].unique()
        bad_annotators = []
        for annotator in annotators:
            worker_votes = voting_matrix[voting_matrix['worker'] == annotator]
            worker_truth = truth_matrix[truth_matrix['worker'] == annotator]
            k = get_k(worker_votes)
            q = get_real_q(worker_truth, 'conf_mat').to_list()
            # if only one human label, then q[0] and q[1] is same
            if len(q) == 1:
                q.append(q[0])
            # two cases of bad annotators, see paper for details
            if q[0] + q[1] >= 1 and not 1 - q[1] <= k <= q[0]:
                bad_annotators.append(annotator)
            elif q[0] + q[1] < 1 and not q[0] <= k <= 1 - q[1]:
                bad_annotators.append(annotator)
        return voting_matrix, truth_matrix
        # if len(bad_annotators) == len(annotators):
        #     print('All annotators are bad :( can\'t proceed with the experiment.')
        #     return voting_matrix, truth_matrix
        # elif len(bad_annotators) > 0:
        #     print(f'Dropping bad annotators: {bad_annotators}')
        # else:
        #     print('All annotators are good :)')
        # return voting_matrix[~voting_matrix['worker'].isin(bad_annotators)].reset_index(drop=True), truth_matrix[~truth_matrix['worker'].isin(bad_annotators)].reset_index(drop=True)
