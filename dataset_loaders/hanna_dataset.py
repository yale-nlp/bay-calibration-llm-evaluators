import random
import pandas as pd
import pickle
import datasets
from pathlib import Path

from dataset_loaders.evaluators.alpaca_farm_evaluators import AlpacaFarmEvaluators
from .base_dataset import BaseDataset
from .utils import cache_matrices


class HannaDataset(BaseDataset):
    dataset_name = 'xxx'

    def __init__(self):
        pass
    
    @staticmethod
    def generate_matrices(evaluators=None):
        raise NotImplementedError

    @classmethod
    @cache_matrices(load_path='data_cache/hanna_matrices.pkl')
    def get_matrices(cls, use_ood_q=False, **kwargs):
        kwargs['q_prior_data_for_gold_labels'] = kwargs['q_prior_data_usage'] == 'gold_labels'
        return super().get_matrices(dataset_name=cls.dataset_name, use_ood_q=use_ood_q, **kwargs)

if __name__ == '__main__':
    HannaDataset.generate_matrices('AlpacaFarm')