import random
import pandas as pd
import pickle
import datasets
from pathlib import Path

from dataset_loaders.evaluators.alpaca_farm_evaluators import AlpacaFarmEvaluators
from .base_dataset import BaseDataset
from .utils import cache_matrices, flip_labels


class LLMEval2Dataset(BaseDataset):
    dataset_name = 'llm-aes/llmeval2-annotated-latest'

    def __init__(self):
        pass

    @classmethod
    @cache_matrices(load_path='data_cache/llmeval2_matrices.pkl')
    def get_matrices(cls, use_ood_q=False, **kwargs):
        kwargs['q_prior_data_for_gold_labels'] = kwargs['q_prior_data_usage'] == 'gold_labels'
        voting_matrix, truth_matrix = super().get_matrices(dataset_name=cls.dataset_name, use_ood_q=use_ood_q, **kwargs)
        p_value = kwargs.get('dataset_p', 0.8)
        flip_labels(voting_matrix, p_value)
        flip_labels(truth_matrix, p_value)
        return voting_matrix.reset_index(drop=True), truth_matrix.reset_index(drop=True)
    