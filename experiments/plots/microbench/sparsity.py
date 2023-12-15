import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, FixedLocator, FixedFormatter, MaxNLocator
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

def plot_sparsity_curves(ax, font_size, title):

    df = get_csv('sparsity_df')

    # column_width = 3.3374
    # fig_width = column_width * 0.475

    # plt.rcParams.update({'text.usetex' : True
    #                     , 'pgf.rcfonts': False
    #                     , 'text.latex.preamble':r"""\usepackage{iftex}
    #                                             \ifxetex
    #                                                 \usepackage[libertine]{newtxmath}
    #                                                 \usepackage[tt=false]{libertine}
    #                                                 \setmonofont[StylisticSet=3]{inconsolata}
    #                                             \else
    #                                                 \RequirePackage[tt=false, type1=true]{libertine}
    #                                             \fi"""   
    #                     })

    # fig, ax = plt.subplots()

    # fig.set_size_inches(fig_width/1.5, fig_width/1.5)

    x_array = df['Number of Features']
    filling_degree_array = df['Filling Degree'] * 100
    path_miss_rate = df['Path Miss Rate'] * 100
    
    ax.plot(x_array, filling_degree_array, '^-k' , label='Fill-factor [\%]', markersize=4, linewidth=1)
    ax.plot(x_array, path_miss_rate, 'o-r', label='Miss-rate [\%]', markersize=4, linewidth=1)

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    ax.set_xlabel('\# Features in Index', fontsize = font_size, labelpad=1)
    # ax.set_ylabel('\%', fontsize = font_size)

    ax.set_ylim(top=103)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    ax.title.set_text(title)
    ax.title.set_size(font_size)

    ax.grid(axis='y', linestyle='--', linewidth=0.3)

    ax.legend(loc='upper right', ncols=1, fontsize = font_size-2, frameon=False, columnspacing=0.5, labelspacing=0.2)

    # plt.show()
    
    # script_folder = Path(__file__).resolve().parents[1]
    # data_path = os.path.join(script_folder, 'output', 'sparsity_analysis.pdf')

    # fig.savefig(data_path, format='pdf', bbox_extra_artists=(lgd,) , bbox_inches="tight")


def plot_index_benchmark(ax, font_size, title):

    df = get_csv('index_bench_df')

    # column_width = 3.3374
    # fig_width = column_width * 0.475

    # plt.rcParams.update({'text.usetex' : True
    #                     , 'pgf.rcfonts': False
    #                     , 'text.latex.preamble':r"""\usepackage{iftex}
    #                                             \ifxetex
    #                                                 \usepackage[libertine]{newtxmath}
    #                                                 \usepackage[tt=false]{libertine}
    #                                                 \setmonofont[StylisticSet=3]{inconsolata}
    #                                             \else
    #                                                 \RequirePackage[tt=false, type1=true]{libertine}
    #                                             \fi"""   
    #                     })

    # fig, ax = plt.subplots()

    # fig.set_size_inches(fig_width/1.5, fig_width/1.5)

    x_array = df['Number of Features in Index'].unique()
    y_df = df.groupby(['Index Type', 'Number of Features in Index'], as_index=True).agg({'Prediction Latency [ms]': ['mean', 'std']})
    
    btree_y = y_df.loc['btree', ('Prediction Latency [ms]', 'mean')].to_numpy()
    hash_y = y_df.loc['hash', ('Prediction Latency [ms]', 'mean')].to_numpy()
    trie_y = y_df.loc['trie', ('Prediction Latency [ms]', 'mean')].to_numpy()

    btree_error = y_df.loc['btree', ('Prediction Latency [ms]', 'std')].to_numpy()
    hash_error = y_df.loc['hash', ('Prediction Latency [ms]', 'std')].to_numpy()
    trie_error = y_df.loc['trie', ('Prediction Latency [ms]', 'std')].to_numpy()

    ax.plot(x_array, btree_y, '--',label='B-tree', color='#5e3c99', linewidth=1, markersize=6)
    ax.plot(x_array, hash_y, '.-', label='Hash', color='#ca0020', linewidth=1, markersize=6)
    ax.plot(x_array, trie_y, '*-',label='Trie', color='#008837', linewidth=1, markersize=6)

    ax.errorbar(x_array, btree_y, yerr=btree_error, fmt='none', ecolor='#5e3c99', elinewidth=1)
    ax.errorbar(x_array, hash_y, yerr=hash_error, fmt='none', ecolor='#ca0020', elinewidth=1)
    ax.errorbar(x_array, trie_y, yerr=trie_error, fmt='none', ecolor='#008837', elinewidth=1)

    ax.set_yscale('log')

    ax.set_xlabel('\# Features in Index', fontsize = font_size, labelpad=1)
    ax.set_ylabel('Prediction Latency [ms]', fontsize = font_size, labelpad=1)

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.title.set_text(title)
    ax.title.set_size(font_size)

    ax.grid(axis='y', linestyle='--', linewidth=0.3)

    ax.legend(loc='best', ncols=1, fontsize = font_size-2, frameon=False, columnspacing=0.5, labelspacing=0.2)

    # script_folder = Path(__file__).resolve().parents[1]
    # data_path = os.path.join(script_folder, 'output', 'index_benchmark.pdf')

    # fig.savefig(data_path, format='pdf', bbox_extra_artists=(lgd,) , bbox_inches="tight")

    # plt.show()

def create_combined_plot(font_size, wspace, subplot_titles = ['a) Sparsity', 'b) Index performance - $1k$ instances']):

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

    fig.set_size_inches(fig_width*4, fig_width*2)
    fig.subplots_adjust(wspace=wspace)

    plot_index_benchmark(ax[1], font_size, subplot_titles[1])
    plot_sparsity_curves(ax[0], font_size, subplot_titles[0])

    plt.minorticks_off()
    plt.tight_layout()

    script_folder = Path(__file__).resolve().parents[1]
    data_path = os.path.join(script_folder, 'output', 'index_benchmark.pdf')

    fig.savefig(data_path, format='pdf', bbox_inches="tight", transparent=True, pad_inches=0.05)

    # plt.show()

create_combined_plot(16, 0.1)






