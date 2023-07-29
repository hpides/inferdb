import numpy as np
import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(project_folder,'src')
sys.path.append(str(src_folder))
from pickle import load, dump
from database_connect import connect
from psycopg2 import sql
from psycopg2.extensions import register_adapter, AsIs, register_type, new_type, DECIMAL
import re
from database_connect import connect
from preprocessing_experiment import preprocessing_experiment
from nyc_rides_featurizer import NYC_Featurizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import time
from math import floor

# models = ['linearregression']
# models = ['linearregression']
models = ['mlpregressor']
pipelines = ['complex']
# pipelines = ['complex']
batch_sizes = [729000]
# batch_sizes = [729000]

for m in models:
    summary_df = pd.DataFrame()
    for p in pipelines:
        with open(str(project_folder) + '/objects/nyc_rides_' + m + '_' + p + '_pipeline.joblib', "rb") as File:
            pipeline = load(File)
        with open(str(project_folder) + '/objects/nyc_rides_' + m + '_' + p + '_encoder.joblib', "rb") as File:
            encoder = load(File)
        with open(str(project_folder) + '/objects/nyc_rides_' + m + '_' + p + '_features_names.joblib', "rb") as File:
            features_names = load(File)
        with open(str(project_folder) + '/objects/nyc_rides_' + m + '_' + p + '_solution.joblib', "rb") as File:
            solution = load(File)
        with open(str(project_folder) + '/objects/nyc_rides_' + m + '_' + p + '_tuples.joblib', "rb") as File:
            tuples = load(File)
        with open(str(project_folder) + '/objects/nyc_rides_' + m + '_' + p + '_experiment.joblib', "rb") as File:
            test = load(File)

        #### train
        test.create_train_table(pipeline, str(project_folder) + '/data/nyc_rides/train.csv', 'trip_duration')
        test.insert_train_tuples(pipeline, str(project_folder) + '/data/nyc_rides/train.csv', 'trip_duration')

        ##### Test
        test.create_test_table(['pickup_datetime', 'dropoff_datetime'], ['id', 'store_and_fwd_flag'], ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'])
        test.insert_test_tuples(str(project_folder) + '/data/nyc_rides/train.csv', 'trip_duration')
        test.geo_distance()
        test.create_prefix_search_udf()

        for b in batch_sizes:

            if p == 'complex':
                dropoff_encoder = pipeline.named_steps['featurizer'].dropoff_encoder.splits
                pickup_encoder = pipeline.named_steps['featurizer'].pickup_encoder.splits
                route_freq_dict = pipeline.named_steps['featurizer'].route_freq_mappers
                clusters_encoder = pipeline.named_steps['featurizer'].clusters_encoder.splits
                test.create_featurizer(pickup_encoder, dropoff_encoder, clusters_encoder, route_freq_dict, b)
                test.create_encode_scale_set(pipeline)
                test.create_preprocessing_pipeline()
                if m == 'mlpregressor':
                    test.create_nn_table(pipeline)
                    test.create_nn_scorer_function()
                elif m == 'linearregression':
                    test.create_coef_table(pipeline)
                    test.create_scorer_function_lr()
            elif p == 'simple':
                test.create_featurizer(batch_size=b)
                test.create_encode_scale_set(pipeline)
                test.create_preprocessing_pipeline()
                if m == 'mlpregressor':
                    test.create_nn_table(pipeline)
                    test.create_nn_scorer_function()
                elif m == 'linearregression':
                    test.create_coef_table(pipeline)
                    test.create_scorer_function_lr()
            # #### kv
            test.create_kv_featurizer(b)
            test.create_preproccesing_function(encoder, features_names, solution)
            test.create_kv_table()
            test.insert_tuples_in_kv_table(tuples)
            test.create_index()
            test.create_scoring_function_kv()

            ## PGML
            test.train_model_in_pgml()
            test.deploy_pgml_trained_model()
            test.create_scorer_function_pgml()

            #### report
            test.generate_report_function_pg()
            
            # for i in range(5):
            #     df = test.create_report_pg(b)
            #     df['Pipeline'] = p
            #     df['iteration'] = i
            #     summary_df = pd.concat([summary_df, df])

    # path_folder = Path(__file__).resolve().parents[2]
    # path = os.path.join(path_folder, 'output')
    # Path(path).mkdir(parents=True, exist_ok=True)
    # path = os.path.join(path, 'nyc_rides_' + m + '_pg.csv')
    # summary_df.to_csv(path, index=False)