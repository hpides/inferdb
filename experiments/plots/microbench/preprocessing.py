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

def get_csv(experiment_name, model_name):
    data_folder_path = get_data_folder_path()

    path = os.path.join(data_folder_path, experiment_name + '_' + model_name + '_pg.csv')

    return pd.read_csv(path)

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_config_file():

    project_folder = Path(__file__).resolve().parents[1]

    path = os.path.join(project_folder, 'plot_config.yml')

    cfg_file = read_yaml(path)

    return cfg_file

def plot_inference_breakdown(experiment_name, model_name, fig_name, ax, font_size):

    df = get_csv(experiment_name, model_name)

    df = df.loc[df['Batch Size (Records)'] == 729000,:]
    
    if model_name in ['mlpregressor', 'mlpclassifier']:
        df = df.loc[df['Solution'].isin(['model', 'index']),:]
        x_array_labels = ['Model', 'Index']
        df['Solution'] = pd.Categorical(df['Solution'], ["model", "index"])
        labels = [['Impute/Featurize', 'Encode-Scale', 'Predict'], ['Impute/Featurize', 'Translate', 'Lookup']]
    else:
        x_array_labels = ['Model', 'PGML', 'Index']
        df['Solution'] = pd.Categorical(df['Solution'], ["model", "pgml", "index"])
        labels = [['Impute/Featurize', 'Encode-Scale', 'Predict'], ['Impute/Featurize', 'Encode-Scale', 'Predict'], ['Impute/Featurize', 'Translate', 'Lookup']]

    param_dict = get_config_file()
    x_array = np.arange(len(x_array_labels))
    pipelines = df['Pipeline'].unique()
    x = np.arange(pipelines.size)

    
    df['Pipeline'] = pd.Categorical(df['Pipeline'], ['simple', 'complex'])
    df = df.sort_values(["Solution", "Pipeline"], ascending=[True, True])

    df = df.groupby(["Solution", "Pipeline"], as_index=False).agg({'End-to-End Latency (ms)': ['mean', 'std'],
                                                                                'Impute Featurize Latency (ms)': ['mean', 'std'], 
                                                                                'Encode Scale Latency (ms)': ['mean', 'std'], 
                                                                                'Score Latency (ms)': ['mean', 'std']
                                                                                })

    width = 0.3  # the width of the bars
    
    normalized_values = []
    runtimes = []
    for ids, s in enumerate(df[('Solution', '')].unique()):
        runtimes.append(df.loc[df[('Solution', '')]==s, ('End-to-End Latency (ms)', 'mean')].reset_index(drop=True).to_numpy())
        normalized_values.append(df.loc[(df[('Solution', '')]==s) ,[('Impute Featurize Latency (ms)', 'mean'), ('Encode Scale Latency (ms)', 'mean'), ('Score Latency (ms)', 'mean')]].to_numpy())
    
    for r in range(len(x_array_labels)):

        for b in x:

            normalized_values[r][b] = [(x/runtimes[r][b]) * 100 for x in normalized_values[r][b]]
    
    for idb, b in enumerate(pipelines):
        multiplier = 0
        for ids, s in enumerate(x_array_labels): 
            cum = 0
            for idv, v in enumerate(normalized_values[ids][idb]):
                offset = width * multiplier
                # if idv == len(normalized_values[ids][idb]) - 1 and ids == len(x_array_labels) - 1 and idb == batches.size - 1:
                if idb == pipelines.size - 1:
                    ax.bar(x[idb] + offset, v, bottom=cum, tick_label = b,width=width, label= s + '-' + labels[ids][idv], hatch=param_dict['hatches'][labels[ids][idv]] , color=param_dict['colors'][s])
                else:
                    ax.bar(x[idb] + offset, v, bottom=cum, tick_label = b,width=width, hatch=param_dict['hatches'][labels[ids][idv]] , color=param_dict['colors'][s])

                cum += v
            multiplier += 1
           

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    ax.set_xlabel('Pipeline', fontsize = font_size, labelpad=1)
    ax.set_ylabel('\% of Inference Latency', fontsize = font_size)

    ax.set_ylim(top=100)

    ax.set_xticks(x + width, pipelines)

    ax.title.set_text(fig_name)
    ax.title.set_size(font_size)

    # labels = [str(f'{value:,}') for value in batches]

    x_labels = ['Simple', 'Complex']

    ax.xaxis.set_major_formatter(FixedFormatter(x_labels))

def plot_experiment_breakdown(experiment_name, model_names, figure_names, font_size, wspace):

    column_width = 3.3374
    fig_width = column_width * 0.475

    plt.rcParams.update({'text.usetex' : True
                        , 'pgf.rcfonts': False
                        , 'text.latex.preamble':r"""\usepackage{iftex}
                                                \ifxetex
                                                    \usepackage[libertine]{newtxmath}
                                                    \usepackage[tt=false]{libertine}
                                                    \setmonofont[StylisticSet=3]{inconsolata}
                                                \else
                                                    \RequirePackage[tt=false, type1=true]{libertine}
                                                \fi"""   
                        })

    fig, ax = plt.subplots(1, 2, sharey='row')

    for idm, m in enumerate(model_names):

        plot_inference_breakdown(experiment_name, m, figure_names[idm], ax[idm], font_size)

    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', experiment_name + '_prep_complex_inference_breakdown_pg.pdf')
        
    ax[1].set_ylabel('')

    fig.set_size_inches(fig_width*2, fig_width)

    fig.subplots_adjust(wspace=wspace)

    h, l = ax[0].get_legend_handles_labels()

    order = [2,1,0,5,4,3,8,7,6]

    h = [h[idx] for idx in order]
    l = [l[idx] for idx in order]

    lgd = plt.legend(handles=h, labels=l, loc='upper center', bbox_to_anchor=(-0.2,-0.25), ncols=2, fontsize = 'x-small')

    # plt.show()
    fig.savefig(data_path, format='pdf', bbox_extra_artists=(lgd,) , bbox_inches="tight")

def create_latex_summary_table(experiment_list, model_list):

    summary_df = pd.DataFrame()
    param_dict = get_config_file()

    for ide, e in enumerate(experiment_list):
        for m in model_list[ide]:
            df = get_csv(e, m)
            df = df.loc[df['Batch Size (Records)']==729000,:]
            summary_df = pd.concat([summary_df, df])
    
    df = summary_df.groupby(["Pipeline", "Solution", "Algorithm"], as_index=True).agg({'End-to-End Latency (ms)': ['mean', 'std']})

    # print(df)
    
    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', 'pg_preprocessing_complexity.tex')

    df.to_latex(buf=data_path, multirow=True, multicolumn=True)

plot_experiment_breakdown('nyc_rides', ['linearregression', 'mlpregressor'], ['Linear Regression', 'MLP Regressor'], 9, 0.1)
create_latex_summary_table(['nyc_rides'],[['linearregression', 'mlpregressor']])

