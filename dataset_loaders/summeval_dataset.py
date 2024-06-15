import random
import pandas as pd
import pickle
import datasets
from pathlib import Path

from dataset_loaders.evaluators.alpaca_farm_evaluators import AlpacaFarmEvaluators
from .base_dataset import BaseDataset
from .utils import cache_matrices


class SummEvalDataset(BaseDataset):
    dataset_name = 'llm-aes/summeval-gpt2-vs-all-annotated-latest'

    def __init__(self):
        pass

    @classmethod
    @cache_matrices(load_path='data_cache/summeval_matrices.pkl')
    def get_matrices(cls, use_ood_q=False, **kwargs):
        kwargs['q_prior_data_for_gold_labels'] = kwargs['q_prior_data_usage'] == 'gold_labels'
        return super().get_matrices(dataset_name=cls.dataset_name, use_ood_q=use_ood_q, **kwargs)
    