import sys
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder))
from sklearn.linear_model import LinearRegression
from process_experiments_pg import nyc_experiment

def linearregression_simple():

    df = nyc_experiment('shallow', LinearRegression(), [1, 10000, 100000, 729000])

    return df