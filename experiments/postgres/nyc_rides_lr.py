from experiment_script import run_experiment
from sklearn.linear_model import LinearRegression
from get_nyc_rides import get_nyc_rides

def nyc_rides_lr():
    get_nyc_rides()
    run_experiment('nyc_rides_simple', LinearRegression(), 'regression', False, 1, False, [1, 10, 100, 1000, 10000, 100000, 500000, 700000])

