import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_log_error
import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[3]
sys.path.append(str(project_folder))
parent_folder = os.path.join(project_folder,'experiments', 'microbenchmarks', 'preprocessing')
sys.path.append(str(parent_folder))
src_folder = os.path.join(project_folder,'src')
sys.path.append(str(src_folder))
from nyc_rides_featurizer import NYC_Featurizer
from pickle import load, dump
from database_connect import connect
from psycopg2 import sql
from psycopg2.extensions import register_adapter, AsIs, register_type, new_type, DECIMAL
import re
from database_connect import connect
from encoder import Encoder
from optimizer import Problem, Optimizer
from scipy import sparse
from io import StringIO
from os.path import getsize
from inference_trie import Trie
import time
from itertools import product

def create_postgres_connection():
        conn = connect()
        cur = conn.cursor()

        return conn, cur

def get_pipeline(featurizer, model):
     
    numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', StandardScaler())
                            ]
                    )
    categorical_transformer = Pipeline(
        steps=
                [
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
        )

    column_transformer = ColumnTransformer(
                                            transformers=[
                                                            ('num', numerical_transformer, featurizer.num_features)
                                                            , ('cat', categorical_transformer, featurizer.cat_features)
                                                        ]
                                            , remainder='passthrough'

                                        )
    pipeline = Pipeline(
                                steps=
                                        [   ('featurizer', featurizer)
                                            , ('column_transformer', column_transformer)
                                            , ('clf', model)
                                        ]
                        )
    
    return pipeline

def train_pipeline(path_to_data, target_feature, pipeline):

        df = pd.read_csv(path_to_data)

        training_features = [i for i in list(df) if i != target_feature]
        X = df[training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        start = time.time()
        trained_pipeline = pipeline.fit(X_train, y_train)
        end =  time.time() - start

        transformed_df = pipeline[0].transform(X_train.head(5))
        transformed_features = list(transformed_df)

        # training_time = end 

        # start = time.time()
        # pipeline[:-1].transform(X_train)
        # end = time.time() - start

        # preproc_training_time = end

        
        return trained_pipeline, training_features
        


def get_kv_tuples(trained_pipeline, training_features, path_to_data, target_feature, complex=False, optimize=True):

        df = pd.read_csv(path_to_data)
        encoder = Encoder('regression')
        
        X = df[training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        if complex == False:
            featurizer = NYC_Featurizer('shallow')
        else:
            featurizer = NYC_Featurizer('deep')

        featurizer.fit(X_train, y_train)
        features_names = list(featurizer.transform(X_train.head(5)))
        x_train_featurized = np.array(featurizer.transform(X_train))
        x_test_featurized = np.array(featurizer.transform(X_test))
        y_train_pred = trained_pipeline.predict(X_train)

        cat_mask = [idf for idf, i in enumerate(features_names) if i in featurizer.cat_features]

        encoder.fit(x_train_featurized, y_train_pred, cat_mask)
        start = time.time()
        encoded_training_set = encoder.transform_dataset(x_train_featurized, [i for i in range(x_train_featurized.shape[1])])
        end = time.time() - start
        my_problem = Problem(encoded_training_set, y_train, encoder.num_bins, 'regression', 1)
        encoding_time = end

        encoded_test_set = encoder.transform_dataset(x_test_featurized, [i for i in range(x_test_featurized.shape[1])])
        
        if optimize == True:
            start = time.time()
            my_problem.set_costs()
            print('Costs set')
            my_optimizer = Optimizer(my_problem, 1)
            my_optimizer.greedy_search()
            end = time.time() - start
            solution_time = end
            encoded_training_set = encoded_training_set[:, my_optimizer.greedy_solution]
            encoded_test_set = encoded_test_set[:, my_optimizer.greedy_solution]
            print('Greedy Search Complete, index:' + str(my_optimizer.greedy_solution))
            solution = my_optimizer.greedy_solution
        else:
            my_problem.set_costs()
            results = []
            for i in range(1, encoded_training_set.shape[1]+1):
                solution = np.argsort(my_problem.costs_array)[::-1][:i]
                # solution = list(np.argsort(bin_array)[:i])
                encoded_training_set_ = encoded_training_set[:, solution]
                encoded_test_set_ = encoded_test_set[:, solution]
                print(solution)

        ########### Calculates the freq of hitting a sparse region in the test data ##########

                training_tuples = [tuple(x) for x in encoded_training_set_]
                testing_tuples = [tuple(x) for x in encoded_test_set_]
                paths_present = len(set(training_tuples))
                paths_missing = len(set(testing_tuples) - set(training_tuples))
                bin_set = []
                for feature in solution:
                    bin_set.append([i for i in range(encoder.num_bins[feature]+1)])

                combinations_set = product(*bin_set)
                num_bins_solution = np.array([encoder.num_bins[feature]+1 for feature in solution])
                num_possible_paths = np.prod(num_bins_solution)
                print(num_bins_solution, num_possible_paths)
                # print('finished combinations')
                filling_degree = paths_present / num_possible_paths
                sparse_hit_ratio = paths_missing / (paths_present + paths_missing)

                results.append([i, paths_present, paths_missing, num_possible_paths, filling_degree, sparse_hit_ratio])
            
            summary_df = pd.DataFrame(results, columns=['Number of Features', 'Found Paths', 'Missing Paths', 'All Paths', 'Filling Degree', 'Path Miss Rate'])
         
        return summary_df

def create_prefix_search_udf():

        conn, cur = create_postgres_connection()

        function_statement = sql.SQL(""" CREATE OR REPLACE FUNCTION pgml.prefix_search(searched_key TEXT, search_table TEXT) returns NUMERIC AS 
                                            $$

                                            DECLARE 
                                            r INTEGER;
                                            prediction NUMERIC;
                                            not_found BOOLEAN;
                                            BEGIN
                                                r := 0;
                                                not_found := TRUE;
                                                WHILE not_found LOOP
                                                    r := r + 1;
                                                    execute format('SELECT count(*)=0 from pgml.%I where key ^@ left(%L, length(%L) - (%s*2))', search_table, searched_key, searched_key, r) INTO not_found;
                                                END LOOP;
                                                
                                                execute format('SELECT avg(value) from pgml.%I where key ^@ left(%L, length(%L) - (%s*2))', search_table, searched_key, searched_key, r) INTO prediction;
                                                
                                                RETURN prediction;
                                            END;
                                            $$
                                            LANGUAGE plpgsql STABLE PARALLEL SAFE;   
                                        """
                                     )
        cur.execute(function_statement)
        conn.commit()

def create_test_pg(path_to_data, target_feature, trained_pipeline, index_lengths):
     
    conn, cur = create_postgres_connection()

    df = pd.read_csv(path_to_data)
    encoder = Encoder('regression')
    
    X = df[training_features]
    y = df[target_feature].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

    if complex == False:
        featurizer = NYC_Featurizer('shallow')
    else:
        featurizer = NYC_Featurizer('deep')

    featurizer.fit(X_train, y_train)
    features_names = list(featurizer.transform(X_train.head(5)))
    x_train_featurized = np.array(featurizer.transform(X_train))
    x_test_featurized = np.array(featurizer.transform(X_test))
    y_train_pred = trained_pipeline.predict(X_train)
    y_train_pred = np.where(y_train_pred < 0, 0, y_train_pred)

    cat_mask = [idf for idf, i in enumerate(features_names) if i in featurizer.cat_features]

    encoder.fit(x_train_featurized, y_train_pred, cat_mask)
    start = time.time()
    encoded_training_set = encoder.transform_dataset(x_train_featurized, [i for i in range(x_train_featurized.shape[1])])
    end = time.time() - start
    my_problem = Problem(encoded_training_set, y_train, encoder.num_bins, 'regression', 1)
    encoding_time = end

    encoded_test_set = encoder.transform_dataset(x_test_featurized, [i for i in range(x_test_featurized.shape[1])])
    
    my_problem.set_costs()
    results = []

    for l in index_lengths:
        solution = np.argsort(my_problem.costs_array)[::-1][:l]
        # solution = list(np.argsort(bin_array)[:i])
        encoded_training_set_ = encoded_training_set[:, solution]
        encoded_test_set_ = encoded_test_set[:, solution]

    ########### finds distinct paths in the training and test data ##########

        training_tuples = [tuple(x) for x in encoded_training_set_]
        testing_tuples = [tuple(x) for x in encoded_test_set_]
        bin_set = []
        for feature in solution:
            bin_set.append([i for i in range(encoder.num_bins[feature]+1)])

        encoded_df= pd.DataFrame(encoded_training_set_)
        target_variable_number = encoded_df.shape[1]
        encoded_df[target_variable_number] = y_train_pred
        agg_df = encoded_df.groupby([i for i in range(len(solution))], as_index=False)[target_variable_number].mean()

        buffer = StringIO()
        agg_df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        keys_statement = ''
        for idf, f in enumerate(solution):
            if idf < len(solution) - 1:
                keys_statement += 'v_' + str(idf) + ' INTEGER,'
            else:
                keys_statement += 'v_' + str(idf) + ' INTEGER'


        ###### First: Create prediction table with oberved training keys
        cur.execute(sql.SQL("""DROP TABLE IF EXISTS pgml.kv_table_disc CASCADE; 
                                CREATE TABLE pgml.kv_table_disc ({keys}, value REAL NOT NULL)
                            """).format(keys = sql.SQL(keys_statement)))
        conn.commit()

        cur.copy_expert("COPY pgml.kv_table_disc FROM STDIN WITH CSV DELIMITER ';'", buffer)
        conn.commit()

        #### Create the keys as text
        select_statement = 'SELECT '
        for idf, f in enumerate(solution):
            if idf < len(solution) - 1:
                select_statement += 'v_' + str(idf) + """::text || '.' || """
            else:
                select_statement += 'v_' + str(idf) + '::text as key, value'
        
        cur.execute(sql.SQL(""" DROP TABLE IF EXISTS pgml.kv_table_test CASCADE; CREATE TABLE pgml.kv_table_test as 

                                {select_statement} FROM pgml.kv_table_disc  """
                            ).format(select_statement = sql.SQL(select_statement))
                    )
        conn.commit()

        ###### Now create the test table

        test_frame = pd.DataFrame(testing_tuples)
        test_frame['value'] = y_test

        buffer = StringIO()
        test_frame.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        keys_statement = ''
        for idf, f in enumerate(solution):
            if idf < len(solution) - 1:
                keys_statement += 'v_' + str(idf) + ' INTEGER,'
            else:
                keys_statement += 'v_' + str(idf) + ' INTEGER'
        
        cur.execute(sql.SQL("""DROP TABLE IF EXISTS pgml.test_table_disc CASCADE; 
                                CREATE TABLE pgml.test_table_disc ({keys}, value REAL NOT NULL)
                            """).format(keys = sql.SQL(keys_statement)))
        conn.commit()

        cur.copy_expert("COPY pgml.test_table_disc FROM STDIN WITH CSV DELIMITER ';'", buffer)
        conn.commit()

        #### Create the keys as text
        select_statement = 'SELECT '
        for idf, f in enumerate(solution):
            if idf < len(solution) - 1:
                select_statement += 'v_' + str(idf) + """::text || '.' || """
            else:
                select_statement += 'v_' + str(idf) + '::text as key, value'
        
        cur.execute(sql.SQL(""" DROP TABLE IF EXISTS pgml.test_table CASCADE; CREATE TABLE pgml.test_table as 

                                {select_statement} FROM pgml.test_table_disc  """
                            ).format(select_statement = sql.SQL(select_statement))
                    )
        conn.commit()

        create_prefix_search_udf()

        #####CREATE EXPERIMENT

        for iteration in range(5):
            for index in ['btree', 'trie', 'hash']:
                
                if index == 'btree':
                    cur.execute(sql.SQL("""DROP INDEX IF EXISTS pgml.test_index CASCADE; create index test_index on pgml.kv_table_test (key text_pattern_ops, value); analyze;"""))
                if index == 'trie':
                    cur.execute(sql.SQL("""DROP INDEX IF EXISTS pgml.test_index CASCADE; create index test_index on pgml.kv_table_test using spgist(key); analyze;"""))
                if index == 'hash':
                    cur.execute(sql.SQL("""DROP INDEX IF EXISTS pgml.test_index CASCADE; create index test_index on pgml.kv_table_test using hash(key); analyze;"""))
                

                scoring_function = sql.SQL("""case when k2.value is not null then k2.value else pgml.prefix_search(k1.key, 'kv_table_test') end as prediction
                                                from (select * from pgml.test_table limit 1000) k1
                                                left join pgml.kv_table_test k2
                                                on k1.key = k2.key
                                            """)

                function_statement = sql.SQL("""
                            DROP FUNCTION IF EXISTS pgml.measure_runtime CASCADE;
                            CREATE OR REPLACE FUNCTION pgml.measure_runtime() returns NUMERIC AS $$

                            DECLARE
                            _timing1  timestamptz;
                            _start_ts timestamptz;
                            _end_ts   timestamptz;
                            _overhead numeric;     -- in ms
                            _timing   numeric;     -- in ms
                            runtime numeric;
                            BEGIN
                            _timing1  := clock_timestamp();
                            _start_ts := clock_timestamp();
                            _end_ts   := clock_timestamp();
                            -- take minimum duration as conservative estimate
                            _overhead := 1000 * extract(epoch FROM LEAST(_start_ts - _timing1
                                                                        , _end_ts   - _start_ts));

                            _start_ts := clock_timestamp();
                            PERFORM {sc_function};  -- your query here, replacing the outer SELECT with PERFORM
                            _end_ts   := clock_timestamp();
                            
                            -- RAISE NOTICE 'Timing overhead in ms = %', _overhead;
                            runtime := 1000 * (extract(epoch FROM _end_ts - _start_ts)) - _overhead;

                            RETURN runtime;
                            END;
                            $$ LANGUAGE plpgsql;
                        """).format(sc_function = scoring_function)

                cur.execute(function_statement)

                measure_runtime = sql.SQL(""" select * from  pgml.measure_runtime() """)
                cur.execute(measure_runtime)

                result = cur.fetchall()

                for row in result:
                    results.append([l, iteration, index, result[0][0]])
    
    results_df = pd.DataFrame(results, columns=['Number of Features in Index', 'Iteration', 'Index Type', 'Prediction Latency [ms]'])

    return results_df



pipeline = get_pipeline(NYC_Featurizer('deep'), LinearRegression())
path = os.path.join(project_folder, 'data', 'nyc_rides', 'train.csv')
trained_pipeline, training_features = train_pipeline(path, 'trip_duration', pipeline)

# summary_df = get_kv_tuples(trained_pipeline, training_features, path, 'trip_duration', True, False)
# data_folder = os.path.join(project_folder, 'experiments', 'output', 'sparsity_df.csv')
# summary_df.to_csv(data_folder, index=False)

pg_df = create_test_pg(path, 'trip_duration', trained_pipeline, [i for i in range(1,11)])
data_folder = os.path.join(project_folder, 'experiments', 'output', 'index_bench_df.csv')
pg_df.to_csv(data_folder, index=False)

