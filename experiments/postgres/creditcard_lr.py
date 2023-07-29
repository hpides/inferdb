from experiment_script import run_experiment
from sklearn.linear_model import LogisticRegression

def creditcard_lr():

    run_experiment('creditcard', LogisticRegression(max_iter=10000), 'classification', False, 1, False, [1, 10, 100, 1000, 10000, 100000, 140000])

