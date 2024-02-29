import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, FixedLocator, FixedFormatter,  MaxNLocator
import matplotlib as mpl
import matplotlib.ticker as ticker
import sys
from pathlib import Path 
import os
import yaml

def get_data_folder_path():

    project_folder = Path(__file__).resolve().parents[2]

    return os.path.join(project_folder, 'output', 'clean_folder')

def get_csv():
    data_folder_path = get_data_folder_path()

    path_1 = os.path.join(data_folder_path, 'query_integration.csv')
    path_2 = os.path.join(data_folder_path, 'query_integration_inferdb.csv')

    df_1 = pd.read_csv(path_1)
    df_1 = df_1.loc[~df_1['Competitor'].isin(['InferDB']), :]
    df_2 = pd.read_csv(path_2)

    df = pd.concat([df_1,df_2])
    
    return df

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_config_file():

    project_folder = Path(__file__).resolve().parents[1]

    path = os.path.join(project_folder, 'plot_config.yml')

    cfg_file = read_yaml(path)

    return cfg_file

def plot_curves(wspace, fontsize):

    df = get_csv()
    param_dict = get_config_file()

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
    fig.subplots_adjust(wspace=wspace)

    fig.set_size_inches(fig_width*2, fig_width/1.5)

    x_array = np.arange(1, df['Limit'].unique().size + 1)

    print(x_array)
    
    for idm, m in enumerate(df['Algorithm'].unique()):

        agg_df = df.loc[df['Algorithm']==m,:].groupby(['Competitor', 'Limit'], as_index=False).agg({'Runtime': ['mean', 'std']})

        #### INFERDB Numbers
        inferdb_y_array = agg_df.loc[agg_df[('Competitor', '')]=='InferDB',('Runtime', 'mean')].to_numpy()
        print(inferdb_y_array)
        inferdb_error = agg_df.loc[agg_df[('Competitor', '')]=='InferDB',('Runtime', 'std')].to_numpy()

        #### SQL Model Numbers
        sql_y_array = agg_df.loc[agg_df[('Competitor', '')]=='SQL Model',('Runtime', 'mean')].to_numpy()
        sql_error = agg_df.loc[agg_df[('Competitor', '')]=='SQL Model',('Runtime', 'std')].to_numpy()

        if m == 'linearregression':

            #### SQL Model Numbers
            pgml_y_array = agg_df.loc[agg_df[('Competitor', '')]=='PGML',('Runtime', 'mean')].to_numpy()
            pgml_error = agg_df.loc[agg_df[('Competitor', '')]=='PGML',('Runtime', 'std')].to_numpy()

            ax[idm].plot(x_array, pgml_y_array, label='PGML', linewidth=1, color=param_dict['colors']['PGML'])
            ax[idm].plot(x_array, sql_y_array, label='SQL Model', linewidth=1, color=param_dict['colors']['SQL Model'])
            ax[idm].plot(x_array, inferdb_y_array, label='InferDB', linewidth=1, color=param_dict['colors']['InferDB'])

            ax[idm].errorbar(x_array, sql_y_array, yerr=sql_error, fmt='none', ecolor=param_dict['colors']['SQL Model'], elinewidth=1)
            ax[idm].errorbar(x_array, inferdb_y_array, yerr=inferdb_error, fmt='none', ecolor=param_dict['colors']['InferDB'], elinewidth=1)
            ax[idm].errorbar(x_array, pgml_y_array, yerr=pgml_error, fmt='none', ecolor=param_dict['colors']['PGML'], elinewidth=1)

            x_labels = ['1', '25', '50', '75', '99']

            ax[idm].xaxis.set_major_locator(ticker.FixedLocator(x_array))
            ax[idm].xaxis.set_major_formatter(FixedFormatter(x_labels))
            ax[idm].grid(axis='y', linestyle='--', linewidth=0.3)

            # ax[idm].set_xlabel('Selectivity [\%]', fontsize=fontsize)
            ax[idm].set_ylabel('I. Latency [ms]', fontsize=fontsize)

            ax[idm].tick_params(axis='both', which='major', labelsize=fontsize)
            ax[idm].tick_params(axis='both', which='minor', labelsize=fontsize)

            ax[idm].set_yscale('log')
            ax[idm].minorticks_off()

            # ax[idm].legend(loc='best', ncols=1, fontsize = fontsize, frameon=False, columnspacing=0.5, labelspacing=0.2)

            ax[idm].title.set_text('LR')
            ax[idm].title.set_size(fontsize)
            ax[idm].legend(loc='best', ncols=1, fontsize = fontsize, frameon=False, columnspacing=0.5, labelspacing=0.2)
            ax[idm].set_ylim(bottom=1e2)
            ax[idm].yaxis.set_ticks([1e2, 1e3, 1e4])
        elif m =='mlpregressor':
        
            ax[idm].plot(x_array, sql_y_array, label='SQL Model', linewidth=1, color=param_dict['colors']['SQL Model'])
            ax[idm].plot(x_array, inferdb_y_array, label='InferDB', linewidth=1, color=param_dict['colors']['InferDB'])

            ax[idm].errorbar(x_array, sql_y_array, yerr=sql_error, fmt='none', ecolor=param_dict['colors']['SQL Model'], elinewidth=1)
            ax[idm].errorbar(x_array, inferdb_y_array, yerr=inferdb_error, fmt='none', ecolor=param_dict['colors']['InferDB'], elinewidth=1)
            ax[idm].set_yscale('log')

            ax[idm].tick_params(axis='both', which='major', labelsize=fontsize)
            ax[idm].tick_params(axis='both', which='minor', labelsize=fontsize)

            x_labels = ['1', '25', '50', '75', '99']

            ax[idm].xaxis.set_major_locator(ticker.FixedLocator(x_array))
            ax[idm].xaxis.set_major_formatter(FixedFormatter(x_labels))
            ax[idm].grid(axis='y', linestyle='--', linewidth=0.3)

            ax[idm].tick_params(axis='both', which='major', labelsize=fontsize)
            ax[idm].tick_params(axis='both', which='minor', labelsize=fontsize)
            

            # ax[idm].set_xlabel('Selectivity [\%]', fontsize=fontsize)
            # ax[idm].set_ylabel('Inference Latency [ms]', fontsize=fontsize)

            ax[idm].title.set_text('NN')
            ax[idm].title.set_size(fontsize)

            # ax[idm].legend(loc='best', ncols=1, fontsize = fontsize, frameon=False, columnspacing=0.5, labelspacing=0.2)
        
            ax[idm].minorticks_off()
    
            # ax[idm].legend(loc='best', ncols=1, fontsize = fontsize, frameon=False, columnspacing=0.5, labelspacing=0.2)

            ax[idm].set_ylim(bottom=1e2)

            ax[idm].yaxis.set_ticks([1e2, 1e3, 1e6])
    
    

    fig.text(0.5, -0.15, 'Selectivity [\%]', ha='center', fontsize=fontsize)
    h, l = ax[0].get_legend_handles_labels()
    lgd = ax[0].legend(handles=h, labels=l, loc='upper center', bbox_to_anchor=(0.9, 1.5), columnspacing=0.1, labelspacing=0.1, ncols=3, fontsize = fontsize, frameon=False)

    plt.show()

    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', 'query_integration.pdf')

    fig.savefig(data_path, format='pdf',bbox_extra_artists=(lgd,) , bbox_inches="tight", transparent=True, pad_inches = 0.15)

plot_curves(0.3, 8)