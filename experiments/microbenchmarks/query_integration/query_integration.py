import numpy as np
import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(project_folder,'src')
sys.path.append(str(src_folder))
preproc_folder = os.path.join(project_folder,'experiments', 'microbenchmarks', 'preprocessing')
sys.path.append(str(preproc_folder))
from pickle import load, dump
from database_connect import connect
from psycopg2 import sql
from nyc_rides_featurizer import NYC_Featurizer
from psycopg2.extensions import register_adapter, AsIs, register_type, new_type, DECIMAL
import re
from database_connect import connect
import pandas as pd

def create_postgres_connection():
        conn = connect()
        cur = conn.cursor()

        return conn, cur

def create_pg_artifacts(model, pipeline_type, batch_size):

    with open(str(project_folder) + '/objects/nyc_rides_' + model + '_' + pipeline_type + '_pipeline.joblib', "rb") as File:
        pipeline = load(File)
    with open(str(project_folder) + '/objects/nyc_rides_' + model + '_' + pipeline_type + '_encoder.joblib', "rb") as File:
        encoder = load(File)
    with open(str(project_folder) + '/objects/nyc_rides_' + model + '_' + pipeline_type + '_features_names.joblib', "rb") as File:
        features_names = load(File)
    with open(str(project_folder) + '/objects/nyc_rides_' + model + '_' + pipeline_type + '_solution.joblib', "rb") as File:
        solution = load(File)
    with open(str(project_folder) + '/objects/nyc_rides_' + model + '_' + pipeline_type + '_tuples.joblib', "rb") as File:
        tuples = load(File)
    with open(str(project_folder) + '/objects/nyc_rides_' + model + '_' + pipeline_type + '_experiment.joblib', "rb") as File:
        test = load(File)

    #### train
    test.create_train_table(pipeline, str(project_folder) + '/data/nyc_rides/train.csv', 'trip_duration')
    test.insert_train_tuples(pipeline, str(project_folder) + '/data/nyc_rides/train.csv', 'trip_duration')

    ##### Test
    test.create_test_table(['pickup_datetime', 'dropoff_datetime'], ['id', 'store_and_fwd_flag'], ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'])
    test.insert_test_tuples(str(project_folder) + '/data/nyc_rides/train.csv', 'trip_duration')
    test.geo_distance()
    test.create_prefix_search_udf()

        
    dropoff_encoder = pipeline.named_steps['featurizer'].dropoff_encoder.splits
    pickup_encoder = pipeline.named_steps['featurizer'].pickup_encoder.splits
    route_freq_dict = pipeline.named_steps['featurizer'].route_freq_mappers
    clusters_encoder = pipeline.named_steps['featurizer'].clusters_encoder.splits
    test.create_featurizer(pickup_encoder, dropoff_encoder, clusters_encoder, route_freq_dict, batch_size)
    test.create_encode_scale_set(pipeline)
    test.create_preprocessing_pipeline()
    if model == 'mlpregressor':
        test.create_nn_table(pipeline)
        test.create_nn_scorer_function()
    elif model == 'linearregression':
        test.create_coef_table(pipeline)
        test.create_scorer_function_lr()
        
    # #### kv
    test.create_kv_featurizer(batch_size)
    test.create_preproccesing_function(encoder, features_names, solution)
    test.create_kv_table()
    test.insert_tuples_in_kv_table(tuples)
    test.create_index()
    test.create_scoring_function_kv()

    ## PGML
    test.train_model_in_pgml()
    test.deploy_pgml_trained_model()
    test.create_scorer_function_pgml()

def set_pgml_query(p_limit):

    query = sql.SQL(""" 
                    analyze;
                    explain analyze 
                    with featurized_data as
                        (
                            SELECT
                            s.row_id
                            , s.passenger_count
                            , s.day
                            , s.hour
                            , s.is_weekend
                            , s.month
                            , s.pickup_hour_of_day
                            , s.distance
                            , COALESCE(rq.count, 0) as route_freq
                            , s.distance * COALESCE(rq.count, 0) as freq_dist
                            , CASE  WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (13, 8) THEN 0 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (13, 8) THEN 1 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (13, 8) THEN 2 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (13, 8) THEN 3 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (13, 8) THEN 4 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (18, 12) THEN 5 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (18, 12) THEN 6 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (18, 12) THEN 7 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (18, 12) THEN 8 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (18, 12) THEN 9 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (7, 2, 14) THEN 10 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (7, 2, 14) THEN 11 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (7, 2, 14) THEN 12 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (7, 2, 14) THEN 13 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (7, 2, 14) THEN 14 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 15 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 16 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 17 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 18 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 19 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 20 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 21 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 22 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 23 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 24 END  as clusters_cluster
                            FROM 
                            (
                                SELECT
                                row_id
                                , passenger_count
                                , TRIM(to_char(pickup_datetime, 'Day')) as day
                                , extract(hour from pickup_datetime) as hour
                                , CASE  WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 0 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 1 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 2 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 3 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 4 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 5 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 6 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 7 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 8 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 9 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 10 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 11 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 12 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 13 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 14 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 15 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 16 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 17 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 18 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 19 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 20 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 21 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 22 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 23 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 24 END  as pickup_cluster
                                , CASE  WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 0 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 1 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 2 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 3 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 4 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 5 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 6 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 7 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 8 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 9 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 10 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 11 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 12 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 13 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 14 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 15 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 16 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 17 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 18 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 19 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 20 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 21 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 22 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 23 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 24 END  as dropoff_cluster
                                , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                , extract(month from pickup_datetime) as month
                                , CASE WHEN extract(hour from pickup_datetime) BETWEEN 6 and 11 THEN 'morning'
                                WHEN extract(hour from pickup_datetime) BETWEEN 12 and 15 THEN 'afternoon'
                                WHEN extract(hour from pickup_datetime) BETWEEN 16 and 21 THEN 'evening'
                                ELSE 'late_night' END as pickup_hour_of_day
                                , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                                FROM (SELECT * FROM pgml.nyc_rides_test limit 350000) zt
                            ) as s
                            LEFT JOIN pgml.route_freq as rq
                            ON
                            s.pickup_cluster || '_' || s.dropoff_cluster = rq.cluster_combination
                            AND
                            s.day = rq.day
                            AND
                            s.hour = rq.hour
                        ),

                        encode_scaled as
                        (
                            
                            SELECT row_id,ARRAY[((passenger_count-(1.6628402817959693))/1.3122565670353086)::real, ((distance-(3.4471878326333023))/4.314937461985763)::real, 
                                        ((hour-(13.60859126695753))/6.401555533353099)::real, ((route_freq-(356.7411458861792))/417.43107981115435)::real, 
                                        ((freq_dist-(650.1091490302825))/820.8054068657466)::real,  
                                        (CASE WHEN day='Friday' THEN 1 ELSE 0 END)::real,  (CASE WHEN day='Monday' THEN 1 ELSE 0 END)::Real,  
                                        CASE WHEN day='Saturday' THEN 1 ELSE 0 END,  CASE WHEN day='Sunday' THEN 1 ELSE 0 END,  
                                        CASE WHEN day='Thursday' THEN 1 ELSE 0 END,  CASE WHEN day='Tuesday' THEN 1 ELSE 0 END,  
                                        CASE WHEN day='Wednesday' THEN 1 ELSE 0 END,  CASE WHEN is_weekend='0' THEN 1 ELSE 0 END,  
                                        CASE WHEN is_weekend='1' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='0' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='1' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='2' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='3' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='4' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='5' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='6' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='7' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='8' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='9' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='10' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='11' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='12' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='13' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='14' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='15' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='16' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='17' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='18' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='19' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='20' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='21' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='22' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='23' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='24' THEN 1 ELSE 0 END,  
                                        CASE WHEN month='1' THEN 1 ELSE 0 END,  CASE WHEN month='2' THEN 1 ELSE 0 END,  CASE WHEN month='3' THEN 1 ELSE 0 END,  
                                        CASE WHEN month='4' THEN 1 ELSE 0 END,  CASE WHEN month='5' THEN 1 ELSE 0 END,  CASE WHEN month='6' THEN 1 ELSE 0 END,  
                                        CASE WHEN pickup_hour_of_day='afternoon' THEN 1 ELSE 0 END,  CASE WHEN pickup_hour_of_day='evening' THEN 1 ELSE 0 END,  
                                        CASE WHEN pickup_hour_of_day='late_night' THEN 1 ELSE 0 END,  CASE WHEN pickup_hour_of_day='morning' THEN 1 ELSE 0 END] as m 
                            FROM featurized_data
                        ),

                        predictions as(
                        select row_id
                                , pgml.predict_batch('nyc_rides', d.m) AS prediction
                                FROM encode_scaled d
                        )

                        select * from predictions where prediction < {p_limit}

                    
                    """
                    
                ).format(p_limit = sql.SQL(str(p_limit)))
    
    return query

def set_sqlmodel_query(p_limit, model):

    if model == 'linearregression':

        prediction_function = sql.SQL(""" intercept as (select val from pgml.nyc_rides_linearregression_coefficients where col_id = -1),

                                            coef as (select pre.row_id, sum(pre.val * b.val) as val
                                                        from preprocessed_data pre
                                                        left join pgml.nyc_rides_linearregression_coefficients b
                                                        on pre.col_id = b.col_id
                                                        group by 1),
                                            
                                            predictions as
                                            (
                                                select coef.row_id, coef.val + intercept.val as prediction
                                                from coef, intercept

                                            )

                                            select * from predictions where prediction < {p_limit} 

                                                
                                            """
                                    ).format(p_limit=sql.SQL(str(p_limit)))
    elif model == 'mlpregressor':

        prediction_function = sql.SQL(""" input_weights as
                                            (
                                                SELECT m1.row_id, m2.col, sum(m1.val * m2.val) AS val
                                                FROM   preprocessed_data m1
                                                join (select * from pgml.nn_matrix_nyc_rides where id=0) m2
                                                    on m1.col_id = m2.row
                                                GROUP BY m1.row_id, m2.col
                                            )
                                            ,

                                            activation as
                                            (
                                                select iw.row_id, iw.col, 1/(1 + EXP(-(iw.val::numeric + nn.val::numeric))) as val
                                                from input_weights iw
                                                join (select * from pgml.nn_matrix_nyc_rides where id=2) nn
                                                    on iw.col = nn.row
                                            )
                                            ,

                                            output_weights as
                                            (
                                                select m1.row_id, sum(m1.val * nn2.val) as val
                                                from activation m1
                                                join (select * from pgml.nn_matrix_nyc_rides where id=1) nn2
                                                    on m1.col = nn2.row
                                                group by 1
                                            ),
                                      
                                            predictions as (
                                                select m1.row_id, m1.val + nn.val as prediction
                                                from output_weights m1, pgml.nn_matrix_nyc_rides nn
                                                where nn.id = 3
                                            )

                                            select * from predictions where prediction < {p_limit} 
                                                                                                                                                                
                                              """
                                      ).format(p_limit=sql.SQL(str(p_limit)))


    query = sql.SQL(""" 
                    analyze;
                    explain analyze 
                    with featurized_data as
                        (
                            SELECT
                            s.row_id
                            , s.passenger_count
                            , s.day
                            , s.hour
                            , s.is_weekend
                            , s.month
                            , s.pickup_hour_of_day
                            , s.distance
                            , COALESCE(rq.count, 0) as route_freq
                            , s.distance * COALESCE(rq.count, 0) as freq_dist
                            , CASE  WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (13, 8) THEN 0 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (13, 8) THEN 1 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (13, 8) THEN 2 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (13, 8) THEN 3 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (13, 8) THEN 4 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (18, 12) THEN 5 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (18, 12) THEN 6 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (18, 12) THEN 7 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (18, 12) THEN 8 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (18, 12) THEN 9 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (7, 2, 14) THEN 10 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (7, 2, 14) THEN 11 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (7, 2, 14) THEN 12 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (7, 2, 14) THEN 13 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (7, 2, 14) THEN 14 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 15 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 16 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 17 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 18 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 19 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 20 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 21 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 22 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 23 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 24 END  as clusters_cluster
                            FROM 
                            (
                                SELECT
                                row_id
                                , passenger_count
                                , TRIM(to_char(pickup_datetime, 'Day')) as day
                                , extract(hour from pickup_datetime) as hour
                                , CASE  WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 0 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 1 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 2 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 3 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 4 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 5 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 6 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 7 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 8 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 9 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 10 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 11 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 12 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 13 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 14 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 15 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 16 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 17 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 18 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 19 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 20 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 21 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 22 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 23 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 24 END  as pickup_cluster
                                , CASE  WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 0 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 1 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 2 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 3 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 4 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 5 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 6 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 7 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 8 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 9 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 10 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 11 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 12 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 13 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 14 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 15 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 16 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 17 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 18 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 19 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 20 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 21 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 22 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 23 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 24 END  as dropoff_cluster
                                , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                , extract(month from pickup_datetime) as month
                                , CASE WHEN extract(hour from pickup_datetime) BETWEEN 6 and 11 THEN 'morning'
                                WHEN extract(hour from pickup_datetime) BETWEEN 12 and 15 THEN 'afternoon'
                                WHEN extract(hour from pickup_datetime) BETWEEN 16 and 21 THEN 'evening'
                                ELSE 'late_night' END as pickup_hour_of_day
                                , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                                FROM (SELECT * FROM pgml.nyc_rides_test limit 350000) zt
                            ) as s
                            LEFT JOIN pgml.route_freq as rq
                            ON
                            s.pickup_cluster || '_' || s.dropoff_cluster = rq.cluster_combination
                            AND
                            s.day = rq.day
                            AND
                            s.hour = rq.hour
                        ),

                        encode_scaled as
                        (
                            
                            SELECT row_id,ARRAY[((passenger_count-(1.6628402817959693))/1.3122565670353086)::real, ((distance-(3.4471878326333023))/4.314937461985763)::real, 
                                        ((hour-(13.60859126695753))/6.401555533353099)::real, ((route_freq-(356.7411458861792))/417.43107981115435)::real, 
                                        ((freq_dist-(650.1091490302825))/820.8054068657466)::real,  
                                        (CASE WHEN day='Friday' THEN 1 ELSE 0 END)::real,  (CASE WHEN day='Monday' THEN 1 ELSE 0 END)::Real,  
                                        CASE WHEN day='Saturday' THEN 1 ELSE 0 END,  CASE WHEN day='Sunday' THEN 1 ELSE 0 END,  
                                        CASE WHEN day='Thursday' THEN 1 ELSE 0 END,  CASE WHEN day='Tuesday' THEN 1 ELSE 0 END,  
                                        CASE WHEN day='Wednesday' THEN 1 ELSE 0 END,  CASE WHEN is_weekend='0' THEN 1 ELSE 0 END,  
                                        CASE WHEN is_weekend='1' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='0' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='1' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='2' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='3' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='4' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='5' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='6' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='7' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='8' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='9' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='10' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='11' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='12' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='13' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='14' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='15' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='16' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='17' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='18' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='19' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='20' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='21' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='22' THEN 1 ELSE 0 END,  
                                        CASE WHEN clusters_cluster='23' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='24' THEN 1 ELSE 0 END,  
                                        CASE WHEN month='1' THEN 1 ELSE 0 END,  CASE WHEN month='2' THEN 1 ELSE 0 END,  CASE WHEN month='3' THEN 1 ELSE 0 END,  
                                        CASE WHEN month='4' THEN 1 ELSE 0 END,  CASE WHEN month='5' THEN 1 ELSE 0 END,  CASE WHEN month='6' THEN 1 ELSE 0 END,  
                                        CASE WHEN pickup_hour_of_day='afternoon' THEN 1 ELSE 0 END,  CASE WHEN pickup_hour_of_day='evening' THEN 1 ELSE 0 END,  
                                        CASE WHEN pickup_hour_of_day='late_night' THEN 1 ELSE 0 END,  CASE WHEN pickup_hour_of_day='morning' THEN 1 ELSE 0 END] as m 
                            FROM featurized_data
                        ),


                        preprocessed_data
                        as(

                            SELECT m.row_id, u.ord - 1 as col_id, u.val
                            FROM   encode_scaled m,
                                LATERAL unnest(m.m) WITH ORDINALITY AS u(val, ord)
                            where u.val != 0
                        ),  
                        
                        {prediction_function}
                        """
                    ).format(prediction_function = prediction_function)
    
    return query

def set_inferdb_query(p_limit, model):

    query = sql.SQL(""" 
                    analyze;
                    explain analyze 
                    with featurized_data as
                        (

                            SELECT 
                            row_id
                            , passenger_count
                            , TRIM(to_char(pickup_datetime, 'Day')) as day
                            , extract(hour from pickup_datetime) as hour
                            , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                            , extract(month from pickup_datetime) as month
                            , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                            FROM pgml.nyc_rides_test limit 350000

                        ),

                        translated_data as
                        (
                            SELECT row_id, CASE WHEN hour <= 2.5 THEN '0.' WHEN hour <= 9.5 AND hour > 2.5 THEN '1.' WHEN hour <= 11.5 AND hour > 9.5 THEN '2.' ELSE '3.' END || 
                                            CASE WHEN month::TEXT= ANY(ARRAY['1']) THEN '0.' WHEN month::TEXT= ANY(ARRAY['2']) THEN '1.' WHEN month::TEXT= ANY(ARRAY['3']) THEN '2.' WHEN month::TEXT= ANY(ARRAY['4']) THEN '3.' WHEN month::TEXT= ANY(ARRAY['5']) THEN '4.' WHEN month::TEXT= ANY(ARRAY['6']) THEN '5.' ELSE '6.' END || 
                                            CASE WHEN day::TEXT= ANY(ARRAY['Monday']) THEN '0.' WHEN day::TEXT= ANY(ARRAY['Sunday']) THEN '1.' WHEN day::TEXT= ANY(ARRAY['Saturday']) THEN '2.' WHEN day::TEXT= ANY(ARRAY['Wednesday']) THEN '3.' WHEN day::TEXT= ANY(ARRAY['Friday']) THEN '4.' WHEN day::TEXT= ANY(ARRAY['Tuesday']) THEN '5.' WHEN day::TEXT= ANY(ARRAY['Thursday']) THEN '6.' ELSE '7.' END || 
                                            CASE WHEN distance <= 0.7610131204128265 THEN '0' WHEN distance <= 1.0520523190498352 AND distance > 0.7610131204128265 THEN '1' WHEN distance <= 1.2976540923118591 AND distance > 1.0520523190498352 THEN '2' WHEN distance <= 1.5451794266700745 AND distance > 1.2976540923118591 THEN '3' WHEN distance <= 1.7953275442123413 AND distance > 1.5451794266700745 THEN '4' WHEN distance <= 2.1136789321899414 AND distance > 1.7953275442123413 THEN '5' WHEN distance <= 2.4617480039596558 AND distance > 2.1136789321899414 THEN '6' WHEN distance <= 2.791672468185425 AND distance > 2.4617480039596558 THEN '7' WHEN distance <= 3.507714867591858 AND distance > 2.791672468185425 THEN '8' WHEN distance <= 4.474145889282227 AND distance > 3.507714867591858 THEN '9' WHEN distance <= 5.80251407623291 AND distance > 4.474145889282227 THEN '10' ELSE '11' END as key  
                            FROM featurized_data
                        )


                        select k1.row_id, case when k2.value is not null then k2.value else pgml.prefix_search(k1.key, 'nyc_rides_{model}_kv') end as prediction
                        from translated_data k1
                        left join pgml.nyc_rides_{model}_kv k2
                        on k1.key = k2.key
                        where k2.value < {p_limit}  """
                    ).format(model=sql.SQL(model),
                             p_limit=sql.SQL(str(p_limit)))
    return query

def run_benchmark(model, p_limits):

    conn, cur = create_postgres_connection()

    results = []
    create_pg_artifacts(model, 'complex', 350000)
    for l in p_limits:
        for i in range(3):

            inferdb_query = set_inferdb_query(l, model)
            cur.execute(inferdb_query)
            analyze_fetched = cur.fetchall()
            runtime = float(analyze_fetched[-1][0].split(':')[1].strip()[:-3])
            results.append(['InferDB', model, l, i, runtime, analyze_fetched])

            sqlmodel_query = set_sqlmodel_query(l, model)
            cur.execute(sqlmodel_query)
            analyze_fetched = cur.fetchall()
            runtime = float(analyze_fetched[-1][0].split(':')[1].strip()[:-3])
            results.append(['SQL Model', model, l, i, runtime, analyze_fetched])

            if model == 'linearregression':

                pgml_query = set_pgml_query(l)
                cur.execute(pgml_query)
                analyze_fetched = cur.fetchall()
                runtime = float(analyze_fetched[-1][0].split(':')[1].strip()[:-3])
                results.append(['PGML', model, l, i, runtime, analyze_fetched])
            else:
                continue
    
    summary_df = pd.DataFrame(results, columns=['Competitor', 'Algorithm', 'Limit', 'Iteration', 'Runtime', 'Query Plan'])

    return summary_df

p_limits = [303.8971749026446,	545.4022631244602,	767.670044657267,	1076.1988712279758,	2305.2487750444584]

df =pd.DataFrame()

lr_df = run_benchmark('linearregression', p_limits)
nn_df = run_benchmark('mlpregressor', p_limits)

df = pd.concat([df, lr_df, nn_df], ignore_index=True)


path_folder = Path(__file__).resolve().parents[2]
path = os.path.join(path_folder, 'output')
Path(path).mkdir(parents=True, exist_ok=True)
path = os.path.join(path, 'query_integration_inferdb.csv')
df.to_csv(path, index=False)