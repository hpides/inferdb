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
from math import log

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_config_file():

    project_folder = Path(__file__).resolve().parents[1]

    path = os.path.join(project_folder, 'plot_config.yml')

    cfg_file = read_yaml(path)

    return cfg_file

def get_data_folder_path():

    project_folder = Path(__file__).resolve().parents[2]

    return os.path.join(project_folder, 'output', 'clean_folder')

def get_csv(experiment_name, model_name):
    data_folder_path = get_data_folder_path()

    path = os.path.join(data_folder_path, experiment_name + '_' + model_name + '_pg.csv')

    return pd.read_csv(path)


def plot_inference_runtimes(experiment_name, model_name, ax, font_size, title_name, top):

    df = get_csv(experiment_name, model_name)

    if 'Pipeline' in df.columns:

        df = df.loc[df['Pipeline']=='complex',:]
    
    if model_name in ['mlpregressor', 'mlpclassifier']:
        df = df.loc[df['Solution'].isin(['model', 'index']),:]
        x_array_labels = ['SQL Model', 'InferDB']
        df['Solution'] = pd.Categorical(df['Solution'], ["model", "index"])
    else:
        x_array_labels = ['SQL Model', 'PGML', 'InferDB']
        df['Solution'] = pd.Categorical(df['Solution'], ["model", "pgml", "index"])

    param_dict = get_config_file()
    x_array = np.arange(len(x_array_labels))

    df = df.loc[df['Batch Size (Records)'].isin(param_dict['batches'][experiment_name]), :]
    batches = df['Batch Size (Records)'].unique()
    x = np.arange(batches.size)

    df = df.sort_values(["Solution", 'Batch Size (Records)'], ascending=[True, True])

    # df.reset_index(inplace=True, drop=True)
    
    df = df.groupby(["Solution", 'Batch Size (Records)'], as_index=False).agg({'End-to-End Latency (ms)': ['mean', 'std']})

    y_array = []
    error = []
    for s in df['Solution'].unique():

        y_array.append(df.loc[df[('Solution', '')]==s,('End-to-End Latency (ms)', 'mean')].to_numpy())
        error.append(df.loc[df[('Solution', '')]==s,('End-to-End Latency (ms)', 'std')].to_numpy())

    width = 0.25  # the width of the bars
    multiplier = 0

    # bars = ax.bar(x_array, y_array, tick_label=x_array_labels, color=[param_dict['colors'][i] for i in x_array_labels])

    for ids, s in enumerate(x_array_labels):
        offset = width * multiplier
        mpl.rcParams['hatch.linewidth'] = 1.5
        ax.bar(x + offset, y_array[ids], width, label=s, color='none', edgecolor=param_dict['colors'][s], hatch=param_dict['competitor_hatches'][s], lw=1.5)
        ax.errorbar(x + offset, y_array[ids], yerr=error[ids], fmt='none', ecolor='k', elinewidth=0.5)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(title_name, fontsize = font_size-1, labelpad=-9)
    # ax.title.set_text(fig_name)
    # ax.title.set_size(font_size)
    
    # ax.set_xlabel('Batch Size (Records)', fontsize = font_size, labelpad=1)

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)
    
    max_y = max(max([max(i) for i in y_array]), top)
    # ax.set_ylim(top=max_y+ max_y*(.3))

    ax.set_xticks(x + width, batches, rotation=45)
    
    x_labels = param_dict['batches_labels'][experiment_name]

    ax.xaxis.set_major_formatter(FixedFormatter(x_labels))
    ax.set_yscale('log')
    ax.set_ylim(bottom=1, top=max_y + max_y*10)
    ax.grid(axis='y', linestyle='--', linewidth=0.3)
    ax.set_ylabel('')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.1)

    ax.tick_params(length=0.05, width=0.03, color='0.8', pad=0.01)

    return max_y


def create_latex_summary_table(experiment_list, model_list):

    summary_df = pd.DataFrame()
    param_dict = get_config_file()

    for ide, e in enumerate(experiment_list):
        for m in model_list[ide]:
            df = get_csv(e, m)
            df = df.loc[df['Batch Size (Records)'].isin(param_dict['batches'][e]),:]
            if 'Pipeline' in df.columns:
                df = df.loc[df['Pipeline']=='complex',:]
            summary_df = pd.concat([summary_df, df])
    
    summary_df['End-to-End Latency (ms)'] = summary_df['End-to-End Latency (ms)']
    summary_df['End-to-End Latency (ms)'] = summary_df['End-to-End Latency (ms)']
    
    df = summary_df.groupby(['Experiment', 'Solution', 'Algorithm', 'Batch Size (Records)'], as_index=True).agg({'End-to-End Latency (ms)': ['mean', 'std']})

    # print(df)
    
    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', 'pg_batch_performance.tex')

    df.to_latex(buf=data_path, multirow=True, multicolumn=False)

def plot_experiment_inference(experiment_names, model_names, figure_names, font_size, wspace, subplot_titles):

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
                        , 'ytick.labelsize' : 'xx-small'
                        })

    fig, ax = plt.subplots(1, 4, sharey='row')
    # fig, ax = plt.subplots(1, 2)

    ax_count = 0
    top = 0
    for ide, experiment_name in enumerate(experiment_names):

        for idm, m in enumerate(model_names[ide]):

            max_y = plot_inference_runtimes(experiment_name, m, ax[ax_count], font_size, subplot_titles[ide][idm], top)
            if max_y > top:
                top = max_y
            ax_count += 1

    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', 'inference_runtimes_batches_pg.pdf')
        
    ax[0].set_ylabel('Inference Latency [ms]', fontsize = font_size-1, labelpad=0.5)

    fig.set_size_inches(fig_width*4, fig_width)

    fig.subplots_adjust(wspace=wspace)

    h, l = ax[0].get_legend_handles_labels()

    lgd = ax[0].legend(handles=h, labels=l, loc='upper center', bbox_to_anchor=(2.1, 1.35), ncols=3, fontsize = font_size-1, frameon=False)
    # lgd.set_in_layout(False)
    # fig.tight_layout()
    
    fig.text(0.5, -0.15, 'Batch Size (Records)', ha='center', fontsize=font_size-1)

    fig.text(0.3, 0.9, figure_names[0], ha='center', fontsize=font_size-1)
    fig.text(0.7, 0.9, figure_names[1], ha='center', fontsize=font_size-1)

    plt.minorticks_off()
    plt.tight_layout()
    
    fig.savefig(data_path, format='pdf', transparent=True, bbox_extra_artists=(lgd,) , bbox_inches="tight", pad_inches = 0.1)

    # plt.show()

def plot_inference_breakdown(experiment_name, model_name, ax, font_size, title_name):

    df = get_csv(experiment_name, model_name)

    if 'Pipeline' in df.columns:

        df = df.loc[df['Pipeline']=='complex',:]
    
    if model_name in ['mlpregressor', 'mlpclassifier']:
        df = df.loc[df['Solution'].isin(['model', 'index']),:]
        x_array_labels = ['SQL Model', 'InferDB']
        labels = [['Impute/Featurize', 'Encode-Scale', 'Predict'], ['Impute/Featurize', 'Translate', 'Predict']]
        df['Solution'] = pd.Categorical(df['Solution'], ["model", "index"])
    else:
        x_array_labels = ['SQL Model', 'PGML', 'InferDB']
        labels = [['Impute/Featurize', 'Encode-Scale', 'Predict'], ['Impute/Featurize', 'Encode-Scale', 'Predict'], ['Impute/Featurize', 'Translate', 'Predict']]
        df['Solution'] = pd.Categorical(df['Solution'], ["model", "pgml", "index"])

    param_dict = get_config_file()
    x_array = np.arange(len(x_array_labels))

    df = df.loc[df['Batch Size (Records)'].isin(param_dict['breakdown_batches'][experiment_name]),:]
    batches = df['Batch Size (Records)'].unique()
    x = np.arange(batches.size)

    
    df = df.sort_values(["Solution", 'Batch Size (Records)'], ascending=[True, True])

    if experiment_name == 'creditcard':
        df = df.groupby(["Solution", 'Batch Size (Records)'], as_index=False).agg({'End-to-End Latency (ms)': ['mean', 'std'], 
                                                                                'Imputation Latency (ms)': ['mean', 'std'], 
                                                                                'Encode Scale Latency (ms)': ['mean', 'std'], 
                                                                                'Score Latency (ms)': ['mean', 'std']
                                                                                })
    else:
        df = df.groupby(["Solution", 'Batch Size (Records)'], as_index=False).agg({'End-to-End Latency (ms)': ['mean', 'std'],
                                                                                'Impute Featurize Latency (ms)': ['mean', 'std'], 
                                                                                'Encode Scale Latency (ms)': ['mean', 'std'], 
                                                                                'Score Latency (ms)': ['mean', 'std']
                                                                                })

    width = 0.275  # the width of the bars

    normalized_values = []
    runtimes = []
    for ids, s in enumerate(df[('Solution', '')].unique()):
        runtimes.append(df.loc[df[('Solution', '')]==s, ('End-to-End Latency (ms)', 'mean')].reset_index(drop=True).to_numpy())
        if experiment_name == 'creditcard':
            normalized_values.append(df.loc[(df[('Solution', '')]==s) ,[('Imputation Latency (ms)', 'mean'), ('Encode Scale Latency (ms)', 'mean'), ('Score Latency (ms)', 'mean')]].to_numpy())
        else:
            normalized_values.append(df.loc[(df[('Solution', '')]==s) ,[('Impute Featurize Latency (ms)', 'mean'), ('Encode Scale Latency (ms)', 'mean'), ('Score Latency (ms)', 'mean')]].to_numpy())

    for r in range(len(x_array_labels)):

        for b in range(batches.size):

            normalized_values[r][b] = [(x/runtimes[r][b]) * 100 for x in normalized_values[r][b]]
    
    for idb, b in enumerate(batches):
        multiplier = 0
        for ids, s in enumerate(x_array_labels): 
            cum = 0
            for idv, v in enumerate(normalized_values[ids][idb]):
                offset = width * multiplier
                # if idv == len(normalized_values[ids][idb]) - 1 and ids == len(x_array_labels) - 1 and idb == batches.size - 1:
                mpl.rcParams['hatch.linewidth'] = 2
                if idb == batches.size - 1:
                    ax.bar(x[idb] + offset, v, bottom=cum, tick_label = b,width=width, label= s + '-' + labels[ids][idv], color='none', edgecolor=param_dict[s][labels[ids][idv]], hatch=param_dict['competitor_hatches'][s], lw=3)
                else:
                    ax.bar(x[idb] + offset, v, bottom=cum, tick_label = b,width=width, color='none', edgecolor=param_dict[s][labels[ids][idv]], hatch=param_dict['competitor_hatches'][s], lw=3)

                cum += v
            multiplier += 1
           

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    # ax.set_xlabel('Batch Size (Records)', fontsize = font_size, labelpad=1)
    # ax.set_ylabel('\% of Inference Latency', fontsize = font_size)

    ax.xaxis.set_label_position('top')
    ax.set_xlabel(title_name, fontsize = font_size-1, labelpad=0.5)

    ax.set_ylim(top=100)

    ax.set_xticks(x + width, batches)

    # ax.title.set_text(fig_name)
    # ax.title.set_size(font_size)

    # labels = [str(f'{value:,}') for value in batches]

    ax.grid(axis='y', linestyle='--', linewidth=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.1)

    ax.tick_params(length=0.05, width=0.03, color='0.8', pad=0.01)

    x_labels = param_dict['breakdown_batches_labels'][experiment_name]

    ax.xaxis.set_major_formatter(FixedFormatter(x_labels))

def plot_experiment_breakdown(experiment_names, model_names, figure_names, font_size, wspace, plot_names):

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

    fig, ax = plt.subplots(1, 4, sharey='row')
    ax_count = 0
    for ide, experiment_name in enumerate(experiment_names):

        for idm, m in enumerate(model_names[ide]):

            plot_inference_breakdown(experiment_name, m, ax[ax_count], font_size, plot_names[ide][idm])
            ax_count += 1

    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', 'inference_breakdown_pg.pdf')
        
    ax[1].set_ylabel('')

    fig.set_size_inches(fig_width*5, fig_width)

    fig.subplots_adjust(wspace=wspace)

    h, l = ax[0].get_legend_handles_labels()

    order = [2,1,0,5,4,3,8,7,6]

    h = [h[idx] for idx in order]
    l = [l[idx] for idx in order]

    lgd = plt.legend(handles=h, labels=l, loc='upper center', bbox_to_anchor=(-1.3,-0.15), ncols=3, fontsize = font_size-2, frameon=False)

    fig.text(0.5, -0.05, 'Batch Size (Records)', ha='center', fontsize=font_size)
    ax[0].set_ylabel('\% of Inference Latency', fontsize = font_size, labelpad=1)

    fig.text(0.3, 0.975, figure_names[0], ha='center', fontsize=font_size)
    fig.text(0.7, 0.975, figure_names[1], ha='center', fontsize=font_size)

    plt.minorticks_off()

    # plt.show()
    fig.savefig(data_path, format='pdf', bbox_extra_artists=(lgd,) , bbox_inches="tight", pad_inches=0.15, transparent=True)


# create_latex_summary_table(['nyc_rides', 'creditcard'], [['linearregression', 'mlpregressor'], ['logisticregression', 'mlpclassifier']])
plot_experiment_inference(['nyc_rides', 'creditcard'], [['linearregression', 'mlpregressor'], ['logisticregression', 'mlpclassifier']], ['a) NYC Rides', 'b) Credit Card Fraud'], 12, 0.1, [['LR', 'NN'], ['LR', 'NN']])
# plot_experiment_inference('nyc_rides', ['linearregression', 'mlpregressor'], ['Linear Regression', 'MLP Regressor'], 3, 0.2)
plot_experiment_breakdown(['nyc_rides', 'creditcard'], [['linearregression', 'mlpregressor'], ['logisticregression', 'mlpclassifier']], ['a) NYC Rides', 'b) Credit Card Fraud'], 12, 0.1, [['LR', 'NN'], ['LR', 'NN']])
# plot_experiment_breakdown('creditcard', ['logisticregression', 'mlpclassifier'], ['Logistic Regression', 'MLP Classifier'], 3, 0.2)
