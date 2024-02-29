from credit_card_new_pipeline import creditcard_experiment
from hits import hits_experiment
from mnist import mnist_experiment
from nyc_rides_complex import nycrides_experiment
from pm25 import pm_experiment
from rice import rice_experiment
import sys


def standalone_experiments(iterations=5, paper_models=False):

    creditcard_experiment(iterations, paper_models)
    hits_experiment(iterations, paper_models)
    mnist_experiment(iterations, paper_models)
    nycrides_experiment(iterations, paper_models)
    pm_experiment(iterations, paper_models)
    rice_experiment(iterations, paper_models)


if __name__ == "__main__":

    standalone_experiments(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))