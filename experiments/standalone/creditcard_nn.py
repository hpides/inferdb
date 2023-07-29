import sys
import os
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.

parent = os.path.dirname(os.path.dirname(current))

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from experiments.experiment_handler import ExperimentHandler
from sklearn.neural_network import MLPClassifier
import pandas as pd
from pathlib import Path 

def creditcard_nn():

    exp = ExperimentHandler('creditcard', MLPClassifier(max_iter=10000, activation='logistic'), 'classification', False, 1, False)

    df = pd.DataFrame()
    for i in range(5):
        exp.fit()
        d = exp.create_report()
        d['iteration'] = i
        df = pd.concat([df, d])
    
    project_folder = Path(__file__).resolve().parents[1]

    path = os.path.join(project_folder, 'output')
    path = os.path.join(path, exp.experiment_name + '_' + exp.model_name)
    df.to_csv(path + '_standalone.csv', index=False)


if __name__ == "__main__":
    creditcard_nn()
    
