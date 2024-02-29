from creditcard.creditcard_pgml import creditcard_experiment
from hits.hits_pgml import hits_experiment
from mnist.mnist_pgml import mnist_experiment
from nyc_rides.nyc_rides_pgml import nycrides_experiment
from pollution.pollution_pgml import pm_experiment
from rice.rice_pgml import rice_experiment
import sys

def pg_experiments(iterations=5, paper_models=False):

    creditcard_experiment(iterations, paper_models)
    hits_experiment(iterations, paper_models)
    mnist_experiment(iterations, paper_models)
    nycrides_experiment(iterations, paper_models)
    pm_experiment(iterations, paper_models)
    rice_experiment(iterations, paper_models)

if __name__ == "__main__":

    pg_experiments(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))
    


