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

models = [
            'decisiontreeclassifier',
            'kneighborsclassifier',
            'lgbmclassifier',
            'logisticregression',
            'mlpclassifier',
            'xgbclassifier'
]

complete_datasets = [
            'creditcard',
            'mnist',
            'rice',
            'nyc_rides_complex',
            'pm25',
            'hits'
]

regression = ['nyc_rides_complex', 'pm25' ]
classification = ['creditcard',
            'mnist',
            'rice',
            'hits' ]

report_columns_regression = [
                'name',
                'model',
                'model_error',
                'index_error'
            ]

report_columns_classification = [
                'name',
                'model',
                'model_f1', 'model_recall', 'model_precision', 'index_f1', 'index_recall', 'index_precision'
            ]


regression_df = pd.DataFrame()
for c in regression:

    path = os.path.join(data_path, c + '_standalone.csv')

    df_m = pd.read_csv(path)
    regression_df = pd.concat((regression_df, df_m.loc[:, report_columns_regression]))

classification_df = pd.DataFrame()
for c in classification:

    path = os.path.join(data_path, c + '_standalone.csv')

    df_m = pd.read_csv(path)
    classification_df = pd.concat((classification_df, df_m.loc[:, report_columns_classification]))

agg_df_clf = classification_df.groupby(['name', 'model']).agg(['mean']).sort_values(by=['name', ('model_f1', 'mean')], ascending=False)
agg_df_reg = regression_df.groupby(['name', 'model']).agg(['mean']).sort_values(by=['name', ('model_error', 'mean')], ascending=True)

# print(agg_df_reg)
# print(agg_df_clf)

output_folder = Path(__file__).resolve().parents[1]
write_path_clf = os.path.join(output_folder, 'latex', 'effectiveness_tables', 'clf_effectiveness.tex') 
write_path_reg = os.path.join(output_folder, 'latex', 'effectiveness_tables', 'reg_effectiveness.tex') 

agg_df_clf.to_latex(write_path_clf, float_format="%.2f", escape=True)
agg_df_reg.to_latex(write_path_reg, float_format="%.2f", escape=True)


