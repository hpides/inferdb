import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_log_error
import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[3]
sys.path.append(str(project_folder))
parent_folder = os.path.join(project_folder,'experiments', 'microbenchmarks', 'preprocessing')
src_folder = os.path.join(project_folder,'src')
sys.path.append(str(parent_folder))
sys.path.append(str(src_folder))
from nyc_rides_featurizer import NYC_Featurizer
from pickle import load, dump
from psycopg2.extensions import register_adapter, AsIs, register_type, new_type, DECIMAL
from src.encoder import Encoder
from src.optimizer import Problem, Optimizer
from os.path import getsize
from src.inference_trie import Trie
import time

def get_pipeline_nyc(featurizer, model):
     
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

def get_pipeline_ccf(model):
     
    numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', StandardScaler())
                            ]
                    )

    column_transformer = ColumnTransformer(
                                            transformers=[
                                                            ('num', numerical_transformer, [i for i in range(29)])
                                                        ]
                                            , remainder='passthrough'

                                        )
    pipeline = Pipeline(
                                steps=
                                        [   
                                            ('column_transformer', column_transformer)
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

def get_kv_tuples_ccf(trained_pipeline, training_features, path_to_data, target_feature):

    df = pd.read_csv(path_to_data)
    encoder = Encoder('classification')
    
    X = df[training_features]
    y = df[target_feature].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

    y_train_pred = trained_pipeline.predict(X_train)

    encoder.fit(X_train.to_numpy(), y_train_pred, [])
    start = time.time()
    encoded_training_set = encoder.transform_dataset(X_train.to_numpy(), [i for i in range(X_train.shape[1])])
    end = time.time() - start
    my_problem = Problem(encoded_training_set, y_train, encoder.num_bins, 'classification', 1)
    encoding_time = end

    encoded_test_set = encoder.transform_dataset(X_test.to_numpy(), [i for i in range(X_test.to_numpy().shape[1])])
    
    
    start = time.time()
    my_problem.set_costs()
    print('Costs set')
    my_optimizer = Optimizer(my_problem, 1)
    my_optimizer.greedy_search()
    end = time.time() - start
    greedy_solution_time = end
    encoded_training_set = encoded_training_set[:, my_optimizer.greedy_solution]
    encoded_test_set = encoded_test_set[:, my_optimizer.greedy_solution]
    print('Greedy Search Complete, index:' + str(my_optimizer.greedy_solution))
    solution = my_optimizer.greedy_solution
    start = time.time()
    brute_force_df = my_optimizer.brute_force(len(solution))
    end = time.time() - start
    brute_force_end = end

    brute_force_df = brute_force_df.sort_values(['IV'], ignore_index=True, ascending=False)
    brute_force_solution = brute_force_df['Features'][0]

    greedy_index = Trie('classification')
    bruteforce_index = Trie('classification')
    st = time.time()

    encoded_df = pd.DataFrame(encoded_training_set)
    target_variable_number = encoded_training_set.shape[1] + 1
    encoded_df[target_variable_number] = y_train_pred

    greedy_agg_df = encoded_df.groupby([i for i in range(len(solution))], as_index=False)[target_variable_number].mean()
    greedy_tuples = greedy_agg_df.to_numpy()

    brute_force_agg_df = encoded_df.groupby([i for i in range(len(brute_force_solution))], as_index=False)[target_variable_number].mean()
    brute_force_tuples = brute_force_agg_df.to_numpy()
    
    for idx, i in enumerate(greedy_tuples):
        key = i[:len(solution)]
        value = round(i[-1])
        greedy_index.insert(key, value)
    
    for idx, i in enumerate(brute_force_tuples):
        key = i[:len(solution)]
        value = round(i[-1])
        bruteforce_index.insert(key, value)
    
    results = [('Greedy', solution, my_optimizer.greedy_iv, sys.getsizeof(greedy_index), greedy_solution_time)]
    results.append(('Brute Force', brute_force_df['Features'][0] ,brute_force_df['IV'][0], sys.getsizeof(bruteforce_index), brute_force_end))

    results_df = pd.DataFrame(results, columns=['Solution', 'Features', 'IV', 'Size (B)', 'Runtime'])

    return results_df
        


def get_kv_tuples_nyc(trained_pipeline, training_features, path_to_data, target_feature, complex=False, optimize=True):

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
        
        
        start = time.time()
        my_problem.set_costs()
        print('Costs set')
        my_optimizer = Optimizer(my_problem, 1)
        my_optimizer.greedy_search()
        end = time.time() - start
        greedy_solution_time = end
        encoded_training_set = encoded_training_set[:, my_optimizer.greedy_solution]
        encoded_test_set = encoded_test_set[:, my_optimizer.greedy_solution]
        print('Greedy Search Complete, index:' + str(my_optimizer.greedy_solution))
        solution = my_optimizer.greedy_solution
        start = time.time()
        brute_force_df = my_optimizer.brute_force(len(solution))
        end = time.time() - start
        brute_force_end = end

        brute_force_df = brute_force_df.sort_values(['IV'], ignore_index=True, ascending=False)
        brute_force_solution = brute_force_df['Features'][0]

        greedy_index = Trie('regression')
        bruteforce_index = Trie('regression')
        st = time.time()

        encoded_df = pd.DataFrame(encoded_training_set)
        target_variable_number = encoded_training_set.shape[1] + 1
        encoded_df[target_variable_number] = y_train_pred

        greedy_agg_df = encoded_df.groupby([i for i in range(len(solution))], as_index=False)[target_variable_number].mean()
        greedy_tuples = greedy_agg_df.to_numpy()

        brute_force_agg_df = encoded_df.groupby([i for i in range(len(brute_force_solution))], as_index=False)[target_variable_number].mean()
        brute_force_tuples = brute_force_agg_df.to_numpy()
        
        for idx, i in enumerate(greedy_tuples):
            key = i[:len(solution)]
            value = round(i[-1])
            greedy_index.insert(key, value)
        
        for idx, i in enumerate(brute_force_tuples):
            key = i[:len(solution)]
            value = round(i[-1])
            bruteforce_index.insert(key, value)
        
        results = [('Greedy', my_optimizer.greedy_iv, sys.getsizeof(greedy_index), greedy_solution_time)]
        results.append(('Brute Force', brute_force_df['IV'][0], sys.getsizeof(bruteforce_index), brute_force_end))

        results_df = pd.DataFrame(results, columns=['Solution', 'IV', 'Size (B)', 'Runtime'])

        return results_df

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def build_deterministic_trie(trained_pipeline, training_features, path_to_data, target_feature, greedy_solution, bs_solution):

    df = pd.read_csv(path_to_data)
    encoder = Encoder('classification')
    
    X = df[training_features]
    y = df[target_feature].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

    y_train_pred = trained_pipeline.predict(X_train)

    encoder.fit(X_train.to_numpy(), y_train_pred, [])
    start = time.time()
    encoded_training_set = encoder.transform_dataset(X_train.to_numpy(), [i for i in range(X_train.shape[1])])
    end = time.time() - start
    my_problem = Problem(encoded_training_set, y_train, encoder.num_bins, 'classification', 1)
    encoding_time = end

    encoded_test_set = encoder.transform_dataset(X_test.to_numpy(), [i for i in range(X_test.to_numpy().shape[1])])

    bin_array = np.fromiter(encoder.num_bins.values(), dtype=int)

    encoded_df = pd.DataFrame(encoded_training_set)
    target_variable_number = encoded_training_set.shape[1] + 1
    encoded_df[target_variable_number] = y_train_pred

    greedy_agg_df = encoded_df.groupby(greedy_solution, as_index=False)[target_variable_number].mean()
    greedy_tuples = greedy_agg_df.to_numpy()

    brute_force_agg_df = encoded_df.groupby(bs_solution, as_index=False)[target_variable_number].mean()
    brute_force_tuples = brute_force_agg_df.to_numpy()

    greedy_index = Trie('classification')
    bruteforce_index = Trie('classification')
    
    for idx, i in enumerate(greedy_tuples):
        key = i[:len(greedy_solution)]
        value = round(i[-1])
        greedy_index.insert(key, value)
    
    for idx, i in enumerate(brute_force_tuples):
        key = i[:len(bs_solution)]
        value = round(i[-1])
        bruteforce_index.insert(key, value)
    
    results = [('Greedy', greedy_solution, get_size(greedy_index))]
    results.append(('Brute Force', bs_solution, get_size(bruteforce_index)))

    object_folder = os.path.join(project_folder, 'objects')
    with open(object_folder + '/gs_index.joblib', "wb") as File:
        dump(greedy_index, File)
    gs_size = getsize(object_folder + '/gs_index.joblib')
    print(gs_size) 

    results_df = pd.DataFrame(results, columns=['Solution', 'Features', 'Size (B)'])

    print('Bins found by Greedy search solution: ' + str(bin_array[greedy_solution]))
    print('Bins found by Brute Force solution: ' + str(bin_array[bs_solution]))

    return results_df

# pipeline = get_pipeline_nyc(NYC_Featurizer('deep'), LinearRegression())
# path = os.path.join(project_folder, 'data', 'nyc_rides', 'train.csv')
# trained_pipeline, training_features = train_pipeline(path, 'trip_duration', pipeline)

# results_df = get_kv_tuples_nyc(trained_pipeline, training_features, path, 'trip_duration', True, True)

# data_folder = os.path.join(project_folder, 'experiments', 'output', 'brute_force_search_df.csv')
# results_df.to_csv(data_folder, index=False)

pipeline = get_pipeline_ccf(LogisticRegression(max_iter=10000))
path = os.path.join(project_folder, 'data', 'creditcard', 'creditcard.csv')
trained_pipeline, training_features = train_pipeline(path, 'class', pipeline)

# results_df = get_kv_tuples_ccf(trained_pipeline, training_features, path, 'class')

results_sizes_df = build_deterministic_trie(trained_pipeline, training_features, path, 'class', [3, 9, 15, 16], [3, 15, 16, 2])

data_folder = os.path.join(project_folder, 'experiments', 'output', 'brute_force_search_creditcard_sizes_df.csv')
results_sizes_df.to_csv(data_folder, index=False)


