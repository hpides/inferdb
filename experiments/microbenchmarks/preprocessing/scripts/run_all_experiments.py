from linearregression_complex import linearregression_complex
from linearregression_simple import linearregression_simple
from mlpregressor_complex import mlpregressor_complex
from mlpregressor_simple import mlpregressor_simple
import pandas as pd
import os
from pathlib import Path 
import sys

def run_all_experiments():

    report_df = pd.DataFrame()

    lr_c = linearregression_complex()
    lr_s = linearregression_simple()
    mlp_c = mlpregressor_complex()
    mlp_s = mlpregressor_simple()

    report_df = pd.concat([report_df, lr_c, lr_s, mlp_c, mlp_s], ignore_index=True)

    project_folder = Path(__file__).resolve().parents[3]
    path = os.path.join(project_folder, 'output')
    Path(path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(path, 'nyc_rides_pg.csv')
    report_df.to_csv(path, index=False)


if __name__ == "__main__":

    run_all_experiments()

