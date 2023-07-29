import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, FixedLocator, FixedFormatter
import matplotlib as mpl
import sys
from pathlib import Path 
import os
import yaml



def get_data_folder_path():

    project_folder = Path(__file__).resolve().parents[2]

    return os.path.join(project_folder, 'output', 'clean_folder')

def get_csv(experiment_name):
    data_folder_path = get_data_folder_path()

    path = os.path.join(data_folder_path, experiment_name + '.csv')

    return pd.read_csv(path)

sizes_df = get_csv('test_data_size_df')
print(sizes_df.columns)

seq_df = sizes_df.loc[(~sizes_df['Query Plan Or.'].str.contains("Index Scan using test_index")) | (~sizes_df['Query Plan mod.'].str.contains("Index Scan using test_index")), :]
ix_df = sizes_df.loc[(sizes_df['Query Plan mod.'].str.contains("Index Scan using test_index")) | (sizes_df['Query Plan Or.'].str.contains("Index Scan using test_index")), :]

seq_agg_df = seq_df.groupby(['Number of Features in Index', 'Batch Size']).aggregate({'Prediction Latency [ms] qpo': ['mean', 'std'], 'Prediction Latency [ms] mod': ['mean', 'std']})
ix_agg_df = ix_df.groupby(['Number of Features in Index', 'Batch Size']).aggregate({'Prediction Latency [ms] mod': ['mean', 'std'], 'Prediction Latency [ms] qpo': ['mean', 'std']})

script_folder = Path(__file__).resolve().parents[1]
seq_data_path = os.path.join(script_folder, 'output', 'varying_sizes_seq_scan.tex')
idx_data_path = os.path.join(script_folder, 'output', 'varying_sizes_index_scan.tex')


seq_agg_df.to_latex(buf=seq_data_path, multirow=True, multicolumn=True)

ix_agg_df.to_latex(buf=idx_data_path, multirow=True, multicolumn=True)

print(seq_agg_df)
print(ix_agg_df)

std_10feat_10k_instances = np.array([3047.9630, 2742.2620]).std()
mean_10feat_10k_instances = np.array([3047.9630, 2742.2620]).mean()

print(mean_10feat_10k_instances, std_10feat_10k_instances)

