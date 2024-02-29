import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(x_folder, 'src')
sys.path.append(str(src_folder))
featurizer_folder = os.path.join(src_folder, 'featurizers')
sys.path.append(str(featurizer_folder))
from transpiler import Standalone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_log_error, make_scorer
from pathlib import Path
from nyc_rides_featurizer import NYC_Featurizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from datetime import time
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.impute import SimpleImputer
from inference_trie import Trie
from encoder import Encoder
from optimizer import Problem, Optimizer
from tqdm import tqdm

data_path = os.path.join(x_folder, 'data', 'paper_data', 'nyc_rides', 'nyc_rides_augmented_dirty.csv')

df = pd.read_csv(data_path)
training_features = [i for i in list(df) if i != 'trip_duration']

X = df[training_features]
y = df['trip_duration'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

model = LGBMRegressor(n_estimators=1000, n_jobs=-1, objective='regression', reg_lambda=1, reg_alpha=1)

numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', RobustScaler())
                            ]
                    )
categorical_transformer = Pipeline(
    steps=
            [
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]
    )

categorical_transformer_trees = Pipeline(
    steps=
            [
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]
    )

featurizer = NYC_Featurizer('deep')

column_transformer = ColumnTransformer(
                                        transformers=[
                                                        ('num', numerical_transformer, featurizer.num_features)
                                                        , ('cat', categorical_transformer, [])
                                                    ]
                                        , remainder='passthrough'
                                        , n_jobs=-1

                                    )

pipeline = Pipeline(
                        steps=
                                [   
                                ('featurizer', featurizer),
                                ('column_transformer', column_transformer),
                                ('imputer', SimpleImputer()),
                                ('clf', model)
                                ]
                    )

X_trans = pipeline[:-1].fit_transform(X_train, y_train)

pipeline[-1].fit(X_trans, np.log(y_train))

print('training done')

X_test_trans = pipeline[:-1].transform(X_test)
y_pred = pipeline[-1].predict(X_test_trans)

error = mean_squared_log_error(y_test, np.exp(y_pred), squared=False)

print(error)

results_list = [['standalone', X_trans.shape, model.__class__.__name__, error]]

############################ INFERDB

encoder = Encoder('regression')

y_train_pred = np.exp(pipeline[-1].predict(X_trans))

subselection = [
                'vendor_id',
                'passenger_count', 
                'pickup_longitude', 
                'pickup_latitude', 
                'dropoff_longitude', 
                'dropoff_latitude',
                'pickup_weekday', 
                'pickup_hour', 
                'is_weekend',
                'distance'
                ]

x_featurized_inferdb = pipeline[0].transform_for_inferdb(X_train)
X_test_featurized_inferdb = pipeline[0].transform_for_inferdb(X_test)
# y_train_pred = np.where(y_train_pred > y_train.max(), y_train.max(), y_train_pred)
print(y_train_pred.max())
encoder.fit(x_featurized_inferdb, y_train_pred, [])

encoded_training_set = encoder.transform_dataset(x_featurized_inferdb.loc[:, subselection], [i for i in range(len(subselection))])

my_problem = Problem(encoded_training_set, y_train_pred, encoder.num_bins, 'regression', 1)
my_problem.set_costs()
my_optimizer = Optimizer(my_problem, 1)
my_optimizer.greedy_search()

print(my_optimizer.greedy_solution)

encoded_training_set = encoded_training_set[:, my_optimizer.greedy_solution]

encoded_df= pd.DataFrame(encoded_training_set)
target_variable_number = encoded_df.shape[1]

encoded_df[target_variable_number] = y_train_pred

agg_df = encoded_df.groupby([i for i in range(len(my_optimizer.greedy_solution))], as_index=False)[target_variable_number].mean()
tuples = agg_df.to_numpy()

model_index = Trie('regression')

for i in tuples:
    key = i[:len(my_optimizer.greedy_solution)]
    value = round(i[-1], 2)
    model_index.insert(key, value)

X_test_arr = X_test_featurized_inferdb.loc[:, subselection].to_numpy()
y_pred_trie = np.zeros_like(y_test)
for index, row in tqdm(enumerate(X_test_arr)):
    instance = row[my_optimizer.greedy_solution]
    preprocessed_instance = encoder.transform_single(instance, my_optimizer.greedy_solution)
    y_pred_trie[index] = model_index.query(preprocessed_instance)

error = mean_squared_log_error(y_test, y_pred_trie, squared=False)

print(error)

results_list.append(['InferDB', encoded_training_set.shape, model.__class__.__name__, error])

############################ INFERDB - True values

encoder = Encoder('regression')

y_train_pred = np.exp(pipeline[-1].predict(X_trans))

subselection = [
                'vendor_id',
                'passenger_count', 
                'pickup_longitude', 
                'pickup_latitude', 
                'dropoff_longitude', 
                'dropoff_latitude',
                'pickup_weekday', 
                'pickup_hour', 
                'is_weekend',
                'distance'
                ]

x_featurized_inferdb = pipeline[0].transform_for_inferdb(X_train)
X_test_featurized_inferdb = pipeline[0].transform_for_inferdb(X_test)
# y_train_pred = np.where(y_train_pred > y_train.max(), y_train.max(), y_train_pred)
print(y_train_pred.max())
encoder.fit(x_featurized_inferdb, y_train_pred, [])

encoded_training_set = encoder.transform_dataset(x_featurized_inferdb.loc[:, subselection], [i for i in range(len(subselection))])

my_problem = Problem(encoded_training_set, y_train, encoder.num_bins, 'regression', 1)
my_problem.set_costs()
my_optimizer = Optimizer(my_problem, 1)
my_optimizer.greedy_search()

print(my_optimizer.greedy_solution)

encoded_training_set = encoded_training_set[:, my_optimizer.greedy_solution]

encoded_df= pd.DataFrame(encoded_training_set)
target_variable_number = encoded_df.shape[1]

encoded_df[target_variable_number] = y_train

agg_df = encoded_df.groupby([i for i in range(len(my_optimizer.greedy_solution))], as_index=False)[target_variable_number].mean()
tuples = agg_df.to_numpy()

model_index = Trie('regression')

for i in tuples:
    key = i[:len(my_optimizer.greedy_solution)]
    value = round(i[-1], 2)
    model_index.insert(key, value)

X_test_arr = X_test_featurized_inferdb.loc[:, subselection].to_numpy()
y_pred_trie = np.zeros_like(y_test)
for index, row in tqdm(enumerate(X_test_arr)):
    instance = row[my_optimizer.greedy_solution]
    preprocessed_instance = encoder.transform_single(instance, my_optimizer.greedy_solution)
    y_pred_trie[index] = model_index.query(preprocessed_instance)

error = mean_squared_log_error(y_test, y_pred_trie, squared=False)

print(error)

results_list.append(['InferDB', encoded_training_set.shape, model.__class__.__name__, error])

########################################### KNN

# X_train_featurized_knn = pipeline[0].transform(X_train)
# X_test_featurized_knn = pipeline[0].transform(X_test)

# knn = KNeighborsRegressor()

# X_train_featurized_knn['target'] = y_train
# X_test_featurized_knn['target'] = y_test

# X_train_featurized_knn.dropna(inplace=True)
# X_test_featurized_knn.dropna(inplace=True)

# y_train_knn = X_train_featurized_knn['target'].to_numpy()
# y_test_knn = X_test_featurized_knn['target'].to_numpy()

# X_train_featurized_knn.drop(columns=['target'], inplace=True)
# X_test_featurized_knn.drop(columns=['target'], inplace=True)

# knn.fit(X_train_featurized_knn, y_train_knn)

# y_pred_knn = knn.predict(X_test_featurized_knn)

# error = mean_squared_log_error(y_test_knn, y_pred_knn, squared=False)

# print(error)

# results_list.append(['knn', X_train_featurized_knn.shape, knn.__class__.__name__, error])

# ################################################# kNN constrained

# knn = KNeighborsRegressor()

# selected_features = [subselection[i] for i in my_optimizer.greedy_solution]

# input_size = X_train_featurized_knn.loc[:, selected_features].shape

# knn.fit(X_train_featurized_knn.loc[:, selected_features], y_train_knn)

# y_pred_knn = knn.predict(X_test_featurized_knn.loc[:, selected_features])

# error = mean_squared_log_error(y_test_knn, y_pred_knn, squared=False)

# print(error)

# results_list.append(['knn-constrained', input_size, knn.__class__.__name__, error])

# summary_df = pd.DataFrame(results_list, columns=['Method', 'Input Size', 'Model', 'RMSLE'])

# print(summary_df)

# write_path = os.path.join(x_folder, 'experiments', 'output', 'generalization.csv')
# summary_df.to_csv(write_path, index=False)