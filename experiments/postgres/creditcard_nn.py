from experiment_script import run_experiment
from sklearn.neural_network import MLPClassifier

def creditcard_nn():

    run_experiment('creditcard', MLPClassifier(max_iter=10000, activation='logistic'), 'classification', False, 1, False, [1, 10, 100, 1000, 10000, 100000, 140000])

