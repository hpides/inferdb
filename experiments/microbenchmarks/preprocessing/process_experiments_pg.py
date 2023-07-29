import sys
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[0]
sys.path.append(str(project_folder))
from preprocessing_experiment import preprocessing_experiment
from nyc_rides_featurizer import NYC_Featurizer
import os
import pandas as pd

def nyc_experiment(featurizer, model, batch_sizes):

    project_folder = Path(__file__).resolve().parents[4]

    path = os.path.join(project_folder, 'data', 'nyc_rides', 'train.csv')

    test = preprocessing_experiment('nyc_rides', NYC_Featurizer(featurizer), model)
    pipeline = test.train_pipeline(path, 'trip_duration', False)
    tuples, encoder, features_names, solution = test.get_kv_tuples(pipeline, 'db', path, 'trip_duration', False)

    dropoff_encoder = pipeline.named_steps['featurizer'].dropoff_encoder.splits
    pickup_encoder = pipeline.named_steps['featurizer'].pickup_encoder.splits
    route_freq_dict = pipeline.named_steps['featurizer'].route_freq_mappers
    clusters_encoder = pipeline.named_steps['featurizer'].clusters_encoder.splits

    test.create_test_table(['pickup_datetime', 'dropoff_datetime'], ['id', 'store_and_fwd_flag'], ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'])
    test.insert_test_tuples('data/nyc_rides/train.csv', 'trip_duration')
    test.geo_distance()

    # #### PGML
    test.create_train_table(pipeline, 'data/nyc_rides/train.csv', 'trip_duration')
    test.insert_train_tuples(pipeline, 'data/nyc_rides/train.csv', 'trip_duration')
    test.train_model_in_pgml()
    test.deploy_pgml_trained_model()
    test.create_scorer_function_pgml()

    summary_df = pd.DataFrame()

    for b in batch_sizes:
        ###### CREATE SQL MODEL
        test.create_featurizer(pickup_encoder, dropoff_encoder, clusters_encoder, route_freq_dict, b)
        test.create_encode_scale_set(pipeline)
        test.create_preprocessing_pipeline()
        test.create_nn_table(pipeline)
        test.create_nn_scorer_function()
        # #### kv
        test.create_kv_featurizer(b)
        test.create_preproccesing_function(encoder, features_names, solution)
        test.create_kv_table()
        test.insert_tuples_in_kv_table(tuples)
        test.create_index()
        test.create_scoring_function_kv()

        test.generate_report_function_pg()

        for i in range(5):
            
            df = test.create_report_pg(b)
            if featurizer == 'deep':
                df['Pipeline'] = 'complex'
            else:
                df['Pipeline'] = 'simple'
            df['iteration'] = i
            summary_df = pd.concat([summary_df, df])
    

    return summary_df
        