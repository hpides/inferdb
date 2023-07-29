
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path 
import os
from matplotlib.ticker import FuncFormatter
import yaml

def get_data_folder_path():

    project_folder = Path(__file__).resolve().parents[2]

    return os.path.join(project_folder, 'output', 'clean_folder')

def get_csv(experiment_name, model_name):
    data_folder_path = get_data_folder_path()

    path = os.path.join(data_folder_path, experiment_name + '_' + model_name + '_standalone.csv')

    return pd.read_csv(path)

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_config_file():

    project_folder = Path(__file__).resolve().parents[1]

    path = os.path.join(project_folder, 'plot_config.yml')

    cfg_file = read_yaml(path)

    return cfg_file


def plot_training_runtimes(experiment_name, model_name, fig_name, ax, font_size):

    df = get_csv(experiment_name, model_name)

    x_array_labels = ['Model', 'Index']
    param_dict = get_config_file()
    x_array = np.arange(len(x_array_labels))
    if df['model'][0] == 'mlpregressor':
        y_array = [df.loc[df['iteration']!=4, 'training_runtime'].mean(), df['encoding_runtime'].mean() + df['solution_runtime'].mean() + df['index_runtime'].mean()]
        error_array = [df.loc[df['iteration']!=4, 'training_runtime'].std(), df['encoding_runtime'].std() + df['solution_runtime'].std() + df['index_runtime'].std()]
    else:
        y_array = [df['training_runtime'].mean(), df['encoding_runtime'].mean() + df['solution_runtime'].mean() + df['index_runtime'].mean()]
        error_array = [df['training_runtime'].std(), df['encoding_runtime'].std() + df['solution_runtime'].std() + df['index_runtime'].std()]

    bars = ax.bar(x_array, y_array, tick_label=x_array_labels, color=[param_dict['colors'][i] for i in x_array_labels])
    # ax.errorbar(x_array, y_array, error_array, fmt="o")

    for idb, bar in enumerate(bars):

        height = round(bar.get_height(), 2)
        ax.annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
        textcoords="offset points", ha='center', va='bottom', fontsize=font_size)
        # bar.set_hatch(param_dict['hatches'][x_array_labels[idb]])
    
    ax.set_xlabel(fig_name, fontsize = font_size, labelpad=1)
    ax.set_ylabel('Training Runtime [s]', fontsize = font_size)

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    ax.set_ylim(top=max(y_array)+ max(y_array)*(.3))

    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

def create_latex_summary_table(experiment_list, model_list):

    summary_df = pd.DataFrame()

    for ide, e in enumerate(experiment_list):
        for m in model_list[ide]:
            df = get_csv(e, m)
            if df['model'][0] == 'mlpregressor':
                df = df.loc[df['iteration']!=4, :] 
                summary_df = pd.concat([summary_df, df])
            else:
                summary_df = pd.concat([summary_df, df])
    
    summary_df['index_training_runtime'] = summary_df['encoding_runtime'] + summary_df['solution_runtime'] + summary_df['index_runtime']
    summary_df['model_training_runtime'] = summary_df['training_preprocessing_runtime'] + summary_df['training_runtime']
    
    df = summary_df.groupby(["name", 'model']).agg({'training_preprocessing_runtime': ['mean', 'std']
                                                    , 'training_runtime': ['mean', 'std']
                                                    , 'model_training_runtime': ['mean', 'std']
                                                    , 'encoding_runtime': ['mean', 'std']
                                                    , 'solution_runtime': ['mean', 'std']
                                                    , 'index_runtime': ['mean', 'std']
                                                    , 'index_training_runtime': ['mean', 'std']})
    
    print(df)
    
    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', 'training_performance.tex')

    df.to_latex(buf=data_path, multirow=True)
    

def plot_experiment_training(experiment_name, model_names, figure_names, font_size, wspace):

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

    fig, ax = plt.subplots(1, 2)

    for idm, m in enumerate(model_names):

        plot_training_runtimes(experiment_name, m, figure_names[idm], ax[idm], font_size)

    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', experiment_name + '_training_runtimes.pdf')
        
    ax[1].set_ylabel('')

    fig.set_size_inches(fig_width * 2.3, fig_width)

    fig.subplots_adjust(wspace=wspace)

    # plt.show()
    
    fig.savefig(data_path, format='pdf', bbox_inches="tight")

def plot_training_breakdown(experiment_name, model_name, fig_name, ax, font_size):

    df = get_csv(experiment_name, model_name)

    if df['model'][0] == 'mlpregressor':
        df = df.loc[df['iteration']!=4, :] 

    x_array_labels = ['Model', 'Index']
    param_dict = get_config_file()
    x_array = np.arange(len(x_array_labels))

    index_runtime = df['encoding_runtime'].mean() + df['solution_runtime'].mean() + df['index_runtime'].mean()
    model_runtime = df['training_runtime'].mean()
    normalized_values = [[(df['training_preprocessing_runtime'].mean() / model_runtime) * 100, ((df['training_runtime'].mean() - df['training_preprocessing_runtime'].mean()) / model_runtime) * 100], [(df['encoding_runtime'].mean() / index_runtime) * 100, (df['solution_runtime'].mean() / index_runtime) * 100, (df['index_runtime'].mean() / index_runtime) * 100]]
    labels = [['Preprocess', 'Learn'], ['Learn Encodings', 'Feature Selection', 'Index Population']]

    for ids, s in enumerate(x_array_labels):
        cum = 0
        for idv, v in enumerate(normalized_values[ids]):
            if idv == 0:
                ax.bar(x_array[ids], v, label=labels[ids][idv], tick_label=s, color=param_dict['colors'][s], hatch=param_dict['hatches'][labels[ids][idv]])
                cum += normalized_values[ids][idv] 
            else:
                ax.bar(x_array[ids], v, bottom=cum ,label=labels[ids][idv], color=param_dict['colors'][s], hatch=param_dict['hatches'][labels[ids][idv]])
                cum += normalized_values[ids][idv]

    ax.set_ylabel('\% Of Training Runtime', fontsize = font_size)

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    ax.set_ylim(top=100)

    ax.set_xlabel(fig_name, fontsize = font_size, labelpad=1)

    ax.set_xticks(x_array, x_array_labels)

def plot_index_breakdown(experiment_name, model_names, figure_names, font_size, wspace):

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

        plot_training_breakdown(experiment_name, m, figure_names[idm], ax[idm], font_size)

    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', experiment_name + '_index_breakdown.pdf')
        
    ax[1].set_ylabel('')

    fig.set_size_inches(fig_width * 2, fig_width)

    fig.subplots_adjust(wspace=wspace)

    h, l = ax[0].get_legend_handles_labels()

    order = [1,0,4,3,2]

    h = [h[idx] for idx in order]
    l = [l[idx] for idx in order]

    lgd = plt.legend(handles=h, labels=l, loc='upper center', bbox_to_anchor=(0,-0.3), ncols=3, fontsize = 'x-small')

    # plt.show()
    
    fig.savefig(data_path, format='pdf', bbox_extra_artists=(lgd,) , bbox_inches="tight")

create_latex_summary_table(['nyc_rides', 'creditcard'], [['linearregression_complex', 'mlpregressor_complex'], ['logisticregression', 'mlpclassifier']])

# plot_index_breakdown('nyc_rides', ['linearregression_complex', 'mlpregressor_complex'], ['Linear Regression', 'MLP Regressor'], 9, 0.1)
# plot_index_breakdown('creditcard', ['logisticregression', 'mlpclassifier'], ['Logistic Regression', 'MLP Classifier'], 9, 0.1)
# plot_experiment_training('nyc_rides', ['linearregression_complex', 'mlpregressor_complex'], ['Linear Regression', 'MLP Regressor'], 9, 0.4)
# plot_experiment_training('creditcard', ['logisticregression', 'mlpclassifier'], ['Logistic Regression', 'MLP Classifier'], 9, 0.3)