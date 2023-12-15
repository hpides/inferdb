import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[2]
src_folder = os.path.join(project_folder, 'src')
sys.path.append(str(src_folder))
from transpiler import SQLmodel, InferDB, PGML
from sklearn.neural_network import MLPClassifier
import pandas as pd

def creditcard_nn(batch_sizes):

    inferdb = InferDB('creditcard', MLPClassifier(max_iter=10000, activation='logistic'), 'classification', 1, False, False)
    sqlmodel = SQLmodel('creditcard', MLPClassifier(max_iter=10000, activation='logistic'), 'classification', False)
    pgml = PGML('creditcard', MLPClassifier(max_iter=10000, activation='logistic'), 'classification', False)

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
    creditcard_nn()



