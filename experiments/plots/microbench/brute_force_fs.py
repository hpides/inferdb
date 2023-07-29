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

    path = os.path.join(data_folder_path, 'brute_force_search_creditcard_df.csv')

    return pd.read_csv(path)

