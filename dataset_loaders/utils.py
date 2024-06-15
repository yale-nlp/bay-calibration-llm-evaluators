import pickle
import random
import datasets
import itertools
import pandas as pd
from copy import deepcopy
from pathlib import Path
from functools import wraps


def cache_matrices(load_path):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_path = Path(load_path)
            load_cache = kwargs.pop('load_cache', True)
            # Check if we should load from cache
            if cache_path.exists() and load_cache:
                print("Loading matrices from cache...")
                return pickle.load(open(cache_path, 'rb'))

            # Call the decorated function to get matrices
            voting_matrix, truth_matrix = func(*args, **kwargs)

            # Cache the matrices if the cache file does not exist
            if not cache_path.exists() and load_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                print("Saving matrices to cache...")
                pickle.dump((voting_matrix, truth_matrix), open(cache_path, 'wb'))

            return voting_matrix, truth_matrix

        return wrapper
    return decorator


def preprocess_hanna():
    df = datasets.load_dataset('xxx')
    df = pd.DataFrame(df['train'])
    # add label column
    df['label'] = df['Relevance'] + df['Coherence'] + df['Empathy'] + df['Surprise'] + df['Engagement'] + df['Complexity']
    total_scores = df.groupby('Story_ID')['label'].sum()
    df = df.drop(columns='label').join(total_scores, on='Story_ID')
    df = df.groupby('Story_ID').first()

    pairwise_comparisons = []

    def get_pairwise_comparison(group):
        pairwise_dict = {}
        for model1, model2 in itertools.combinations(group['Model'], 2):
            model1_row = group[group['Model'] == model1]
            model2_row = group[group['Model'] == model2]
            pairwise_dict['output_1'] = model1_row['Story'].item()
            pairwise_dict['output_2'] = model2_row['Story'].item()
            if model1_row['label'].values[0] > model2_row['label'].values[0]:
                pairwise_dict['human_label'] = 0
            elif model1_row['label'].values[0] < model2_row['label'].values[0]:
                pairwise_dict['human_label'] = 1
            else:
                pairwise_dict['human_label'] = 2
            pairwise_dict['instruction'] = "Write a story with the following prompt."
            pairwise_dict['input'] = model1_row['Prompt'].iloc[0]
            pairwise_dict['generator_1'] = model1
            pairwise_dict['generator_2'] = model2
            pairwise_comparisons.append(deepcopy(pairwise_dict))

    df.groupby('Prompt').apply(get_pairwise_comparison)
    return pd.DataFrame(pairwise_comparisons)


def merge_annotators(dataset_names):
    dfs = []
    for name in dataset_names:
        df = datasets.load_dataset(name)['train'].to_pandas()
        dfs.append(df)
    return datasets.Dataset.from_pandas(pd.concat(dfs))


def flip_labels(df, p_value=0.8):
    all_tasks = df['task'].unique()
    num_sample_tasks = int((1 - p_value) * len(all_tasks))
    sampled_tasks = random.sample(list(all_tasks), num_sample_tasks)
    sampled_indices = df[df['task'].isin(sampled_tasks)].index
    df.loc[sampled_indices, 'human_label'] = 1 - df.loc[sampled_indices, 'human_label']
    df.loc[sampled_indices, 'llm_label'] = 1 - df.loc[sampled_indices, 'llm_label']
    return df


if __name__ == '__main__':
    summeval_full = merge_annotators(['xxx'])
    summeval_full.push_to_hub('xxx')