import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[2]
src_folder = os.path.join(project_folder, 'src')
sys.path.append(str(src_folder))
from transpiler import SQLmodel, InferDB, PGML
from sklearn.linear_model import LogisticRegression
import pandas as pd

def creditcard_lr(batch_sizes):

    inferdb = InferDB('creditcard', LogisticRegression(max_iter=10000), 'classification', 1, False, False)
    sqlmodel = SQLmodel('creditcard', LogisticRegression(max_iter=10000), 'classification', False)
    pgml = PGML('creditcard', LogisticRegression(max_iter=10000), 'classification', False)

    df = pd.DataFrame()
    
    for i in range(5):
        for b in batch_sizes:
            inferdb_df = inferdb.create_report_pg(b)
            inferdb_df['Iteration'] = i
            sqlmodel_df = sqlmodel.create_report_pg(b)
            sqlmodel_df['Iteration'] = i
            pgml_df = pgml.create_report_pg(b)
            pgml_df['Iteration'] = i

            df = pd.concat([df, inferdb_df, sqlmodel_df, pgml_df])

    experiment_folder = Path(__file__).resolve().parents[1]

    path = os.path.join(experiment_folder, 'output')
    path = os.path.join(path, inferdb.experiment_name + '_' + inferdb.model_name)
    df.to_csv(path + '_pg.csv', index=False)

if __name__ == "__main__":
    creditcard_lr()

        



