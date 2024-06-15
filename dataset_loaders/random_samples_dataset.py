import random
import pandas as pd
import pickle
from pathlib import Path
from .base_dataset import BaseDataset
from .utils import cache_matrices


class RandomSamplesDataset(BaseDataset):
    def __init__(self):
        pass

    @staticmethod
    def sample_k(p, q_list, k_n):
        truths, k_samples = [], []

        # sample p
        for _ in range(k_n):
            truths.append(0 if random.random() < p else 1)

        # sample q
        for q in q_list:
            k_samples.append([])
            for t in truths:
                k_samples[-1].append(t if random.random() < q else 1 - t)

        return truths, k_samples

    @staticmethod
    # @cache_matrices(load_path='data_cache/random_samples_matrices.pkl')
    def get_matrices(use_ood_q=False, **kwargs):
        q_truth_list = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
        p_truth = 0.8
        n = 100
        voting_matrix, truth_matrix = [], []
        truth_list, k_matrix = RandomSamplesDataset.sample_k(p_truth, q_truth_list, n)
        for i, k_list in enumerate(k_matrix):
            for j, k in enumerate(k_list):
                voting_matrix.append({
                    "worker": f'w_{i}',
                    "task": f't_{j}',
                    "label": k,
                })
        for j, t in enumerate(truth_list):
            truth_matrix.append({
                "task": f't_{j}',
                "label": t,
            })

        return pd.DataFrame(voting_matrix), pd.DataFrame(truth_matrix)

