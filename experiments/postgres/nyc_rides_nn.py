from experiment_script import run_experiment
from sklearn.neural_network import MLPRegressor
from get_nyc_rides import get_nyc_rides

def nyc_rides_nn():
    get_nyc_rides()
    run_experiment('nyc_rides_simple', MLPRegressor(max_iter=10000, activation='logistic'), 'regression', False, 1, False, [1, 10, 100, 1000, 10000, 100000, 500000, 700000])

