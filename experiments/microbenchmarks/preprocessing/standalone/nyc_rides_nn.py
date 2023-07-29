import sys
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder))
from preprocessing_experiment import preprocessing_experiment
from nyc_rides_featurizer import NYC_Featurizer
from sklearn.neural_network import MLPRegressor
import os
from pickle import load, dump
import pandas as pd


def nyc_rides_nn():
    project_folder = Path(__file__).resolve().parents[4]

    path = os.path.join(project_folder, 'data', 'nyc_rides', 'train.csv')
    # Path(path).mkdir(parents=True, exist_ok=True)

    test = preprocessing_experiment('nyc_rides', NYC_Featurizer('deep'), MLPRegressor(max_iter=10000, activation='logistic'))
    summary_df = pd.DataFrame()
    for i in range(5):

        pipeline = test.train_pipeline(path, 'trip_duration', False)
        test.persist_pipeline(pipeline)
        test.get_kv_tuples(pipeline, 'db', path, 'trip_duration', True)
        test.persist_experiment()

        project_folder = Path(__file__).resolve().parents[4]
        pipeline_path = os.path.join(project_folder, 'objects', 'nyc_rides_mlpregressor_complex_pipeline.joblib')
        object_path = os.path.join(project_folder, 'objects', 'nyc_rides_mlpregressor_complex_experiment.joblib')

        with open(pipeline_path, "rb") as File:
            pipeline = load(File)
        with open(object_path, "rb") as File:
            test = load(File)
        
        exp_folder = Path(__file__).resolve().parents[4]
        data_path = os.path.join(exp_folder, 'data', 'nyc_rides', 'train.csv')

        df = test.create_report(pipeline, data_path, 'trip_duration')
        df['iteration'] = i

        summary_df = pd.concat([summary_df, df])
    
    save_path = os.path.join(exp_folder, 'experiments', 'output', test.name + '_' + 'mlpregressor')

    if test.featurizer.depth == 'deep':
        summary_df.to_csv(save_path + '_complex_standalone.csv', index=False)
    else:
        summary_df.to_csv(save_path + '_simple_standalone.csv', index=False)

if __name__ == "__main__":
    nyc_rides_nn()