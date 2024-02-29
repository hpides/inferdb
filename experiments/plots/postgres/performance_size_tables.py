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

# Impute Featurize Latency (ms),Encode Scale Latency (ms),Score Latency (ms),Batch Size (Records),End-to-End Latency (ms)

project_folder = Path(__file__).resolve().parents[2]
# data_path = os.path.join(project_folder, 'output', 'clean_folder')
data_path = os.path.join(project_folder, 'output')

complete_datasets = [
            'creditcard',
            'mnist',
            'rice',
            'nyc_rides',
            'pm25',
            'hits'
]

report_columns = [
                'Experiment',
                'Algorithm',
                'Solution',
                'Impute Featurize Latency (ms)',
                'Encode Scale Latency (ms)',
                'Score Latency (ms)',
                'Batch Size (Records)',
                'Size (B)',
                'End-to-End Latency (ms)'
            ]



df = pd.DataFrame()
for c in complete_datasets:

    path = os.path.join(data_path, c + '_pgml.csv')

    df_m = pd.read_csv(path)
    df = pd.concat((df, df_m[report_columns]))

df['Preprocessing Latency (ms)'] = df.apply(lambda x: x['Impute Featurize Latency (ms)'] + x['Encode Scale Latency (ms)'], axis=1)
df['Size (MB)'] = df['Size (B)'] / 1e6

write_path = os.path.join(data_path, 'pg_performance_size_experiments.csv')
df.to_csv(write_path, index=False)

agg_inference_df = df.groupby(['Experiment', 'Algorithm', 'Solution', 'Batch Size (Records)'], as_index=False).agg({
                                    'Preprocessing Latency (ms)':['mean', 'std'],
                                    'Score Latency (ms)':['mean', 'std'],
                                    'End-to-End Latency (ms)': ['mean', 'std']
                                    , 'Size (MB)':['mean', 'std']
                                    })

agg_inference_df["Batch Size (Records)"] = agg_inference_df["Batch Size (Records)"].astype(int)

agg_inference_df.set_index(['Experiment', 'Algorithm', 'Solution', 'Batch Size (Records)'], inplace=True)

output_folder = Path(__file__).resolve().parents[1]
write_path_clf = os.path.join(output_folder, 'latex', 'performance_size', 'performance_size.tex') 

agg_inference_df.to_latex(write_path_clf, float_format="%.2f", escape=True)



# output_folder = Path(__file__).resolve().parents[1]
# write_path = os.path.join(output_folder, 'output', 'standalone', 'training_table.tex') 
# agg_training_df.to_latex(write_path, float_format="%.2f")


# pm25_path = os.path.join(data_path, 'scalability_pm25_pgml.csv')

# pol_df = pd.read_csv(pm25_path)

# pol_df['Preprocessing Latency (ms)'] = pol_df.apply(lambda x: x['Impute Featurize Latency (ms)'] + x['Encode Scale Latency (ms)'], axis=1)
# pol_df['Size (MB)'] = pol_df['Size (B)'] / 1e6


# agg_inference_df = pol_df.groupby(['Experiment', 'Algorithm', 'Solution', 'Batch Size (Records)', 'Size (MB)']).agg({
#                                     'Preprocessing Latency (ms)':['mean', 'std'],
#                                     'Score Latency (ms)':['mean', 'std'],
#                                     'End-to-End Latency (ms)': ['mean', 'std']
#                                     })

# print(agg_inference_df)