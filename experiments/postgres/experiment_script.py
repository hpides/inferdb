import sys
import os
import pandas as pd
from pathlib import Path
project_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder))
from experiment_handler import ExperimentHandler


def run_experiment(name, model, task, is_balanced, sample_factor, populate_empty_paths, batch_list):

    exp = ExperimentHandler(name, model, task, is_balanced, sample_factor, populate_empty_paths)

    exp.create_utils()
    exp.fit()
    exp.create_test_artifacts()
    exp.create_train_artifacts()
    
    print('donde creating artefacts')
    df = pd.DataFrame()
    for b in batch_list:
        exp.create_arbes_solution(b)
        exp.create_model_postgres_representation(b)
        exp.create_pgml_artifacts()
        exp.generate_report_function_pg()
        
        for i in range(5):
            df_b = exp.create_report_pg(b)
            df_b['iteration'] = i
            df = pd.concat([df, df_b])

    current = os.path.realpath(__file__)
    path = os.path.join(os.path.dirname(os.path.dirname(current)), 'output')
    Path(path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(path, exp.input_name + '_pg')

    if exp.is_balanced:
        path += '_balanced'
    if exp.populate_paths:
        path += '_pp'

    df.to_csv(path + '.csv', index=False)

    print('done')

