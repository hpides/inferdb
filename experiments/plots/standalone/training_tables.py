import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import sys
from pathlib import Path 
import os
import yaml


project_folder = Path(__file__).resolve().parents[2]
# data_path = os.path.join(project_folder, 'output', 'clean_folder')
data_path = os.path.join(project_folder, 'output')

complete_datasets = [
            'creditcard',
            'mnist',
            'rice',
            'nyc_rides_complex',
            'pm25',
            'hits'
]

report_columns = [
                'name',
                'model',
                'training_preprocessing_runtime',
                'training_runtime',
                'model_end_to_end_runtime',
                'encoding_runtime',
                'solution_runtime',
                'index_runtime',
                'index_end_to_end_runtime'
            ]


df = pd.DataFrame()
for c in complete_datasets:

    path = os.path.join(data_path, c + '_standalone.csv')

    df_m = pd.read_csv(path)
    df = pd.concat((df, df_m[report_columns]))

df['end-to-end-training'] = df.apply(lambda x: x['training_preprocessing_runtime'] + x['training_runtime'], axis=1)
df['end-to-end-index'] = df.apply(lambda x: x['encoding_runtime'] + x['solution_runtime'] + x['index_runtime'], axis=1)

write_path = os.path.join(data_path, 'standalone_experiments.csv')
df.to_csv(write_path, index=False)

df['index_end_to_end_runtime'] = df['index_end_to_end_runtime'] * 1000 ### convert to ms
df['model_end_to_end_runtime'] = df['model_end_to_end_runtime'] * 1000 ### convert to ms

agg_training_df = df.groupby(['name', 'model']).agg({
                                    'training_preprocessing_runtime':['mean', 'std'],
                                    'training_runtime':['mean', 'std'],
                                    'end-to-end-training': ['mean', 'std'],
                                    'model_end_to_end_runtime':['mean', 'std'],
                                    'encoding_runtime':['mean', 'std'],
                                    'solution_runtime':['mean', 'std'],
                                    'index_runtime':['mean', 'std'],
                                    'end-to-end-index':['mean', 'std'],
                                    'index_end_to_end_runtime':['mean', 'std']
                                    })

# print(agg_training_df.loc[:, [('model_end_to_end_runtime', 'mean'), ('model_end_to_end_runtime', 'std')]])

output_folder = Path(__file__).resolve().parents[1]
write_path = os.path.join(output_folder, 'latex', 'training_table', 'tr_table.tex') 
agg_training_df.to_latex(write_path, float_format="%.2f", escape=True)


