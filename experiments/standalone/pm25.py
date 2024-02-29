import sys
import os
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))
src_folder = os.path.join(project_folder, 'src')
sys.path.append(str(src_folder))
from transpiler import Standalone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer

def get_pm_experiment_data():

    data_path = os.path.join(project_folder, 'data', 'paper_data', 'pm', 'pm_2016_2019_augmented.csv')
    df = pd.read_csv(data_path, nrows=30000000)
    df.set_index("date", inplace=True)

    training_features = [
                        'latitude', 'longitude', 'weekday', 'weekofyear', 'day', 
                        'previous_value', 
                        'rolling_2',
                        'trend'
                        ]

    inferdb_subset = [
                        'latitude', 'longitude', 'weekday', 'weekofyear', 'day', 
                        'previous_value', 
                        'rolling_2',
                        'trend'
                    ]
    inferdb_indices = [training_features.index(i) for i in inferdb_subset]

    cat_features = []
    num_features = [i for i in training_features if i not in cat_features]

    cat_mask = [training_features.index(i) for i in cat_features]
    num_mask = [idx for idx, i in enumerate(training_features) if idx not in cat_mask]

    y = df['DS_PM_pred'].to_numpy()

    training_threshold = '2017-01-01'
    test_threshold = '2018-01-01'
    X_train = df.loc[df.index < training_threshold, training_features].bfill()
    X_test = df.loc[(df.index > training_threshold) & (df.index <= test_threshold), training_features].bfill()
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = df.loc[df.index < training_threshold, 'DS_PM_pred'].to_numpy()
    y_test = df.loc[(df.index > training_threshold) & (df.index <= test_threshold), 'DS_PM_pred'].to_numpy()

    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()

    return X_train, X_test, y_train, y_test, inferdb_indices, training_features, num_mask

def get_pipeline(model, num_mask):

    numerical_transformer = Pipeline(
                        steps=
                                [
                                    ('scaler', RobustScaler())
                                ]
                        )

    column_transformer = ColumnTransformer(
                                                transformers=[
                                                                ('num', numerical_transformer, [])
                                                            ]
                                                , remainder='passthrough'
                                                , n_jobs=-1

                                            )

    pipeline = Pipeline(
                            steps=
                                    [   
                                        ('imputer', SimpleImputer()),
                                        ('column_transformer', column_transformer),
                                        
                                    ]
                        )

    tree_based_models = ['xgbregressor', 'decisiontreeregressor', 'lgbmregressor']

    if model.__class__.__name__ in tree_based_models:

        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, [])
                                                                ]
        
    else:
        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, num_mask)
                                                                ]

    pipeline.steps.append(['clf', model])

    return pipeline

def get_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, inferdb_indices, training_features, num_mask = get_pm_experiment_data()

    df = pd.DataFrame()
    exp = Standalone(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline, inferdb_indices)
    for i in range(iterations):
        d = exp.create_report(cat_mask=[])
        d['iteration'] = i
        df = pd.concat([df, d])
    
    return df


def pm_experiment(iterations=5, paper_models=False):

    xgboost = xgb.XGBRegressor(objective="reg:squarederror", max_delta_step=10, gamma=1, random_state=42, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
    lr = LinearRegression(n_jobs=-1)
    dt = DecisionTreeRegressor()
    nn = MLPRegressor(hidden_layer_sizes=(8,), max_iter=10000, activation='relu')
    knn = KNeighborsRegressor(algorithm='kd_tree', n_jobs=-1)
    lgbm = LGBMRegressor(n_jobs=-1, objective='regression', random_state=42, reg_lambda=1, reg_alpha=1)

    num_mask = [0, 1, 2, 3, 4, 5, 6, 7]

    if paper_models:
        models = [
                    xgboost,  
                    knn,  
                    lgbm
                ]
    else:
        models = [
            xgboost, 
            lr, 
            dt, 
            nn, 
            knn, 
            lgbm
            ]

    df = pd.DataFrame()
    for model in models:

        pipeline = get_pipeline(model, num_mask)
        d = get_report(pipeline, iterations)
        df = pd.concat([df, d])
        export_path = os.path.join(project_folder, 'experiments', 'output', 'pm25')
        df.to_csv(export_path + '_standalone.csv', index=False)

if __name__ == "__main__":

    pm_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))