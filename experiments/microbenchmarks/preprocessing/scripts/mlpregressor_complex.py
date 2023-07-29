import sys
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder))
from sklearn.neural_network import MLPRegressor
from process_experiments_pg import nyc_experiment

def mlpregressor_complex():

    df = nyc_experiment('deep', MLPRegressor(max_iter=10000, activation='logistic'), [1, 10000, 100000, 729000])

    return df