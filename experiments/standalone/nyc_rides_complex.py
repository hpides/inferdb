import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[2]
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

def get_experiment_data():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'nyc_rides', 'nyc_rides_augmented.csv')
    df = pd.read_csv(data_path)

    training_features = [i for i in list(df) if i != 'trip_duration']
    X = df[training_features]
    y = df['trip_duration'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test

def get_pipeline(model):

    tree_based_models = ['xgbregressor', 'decisiontreeregressor', 'lgbmregressor']

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
                                            , remainder='drop'
                                            , n_jobs=-1

                                        )

    featurizer = NYC_Featurizer('deep')

    pipeline = Pipeline(
                            steps=
                                    [   
                                        ('featurizer', featurizer),
                                        ('imputer', SimpleImputer()),
                                        ('column_transformer', column_transformer)
                                    ]
                        )
    
    if model.__class__.__name__ in tree_based_models:
        params = {'remainder':'passthrough'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_transformer, [])]
    else:
        params = {'remainder':'drop'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [
                                                                ('num', numerical_transformer, [i for i in range(len(featurizer.num_features))])
                                                                ]
            
    pipeline.steps.append(['clf', model])

    return pipeline

def get_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()

    exp = Standalone(X_train, X_test, y_train, y_test, 'nyc_rides', 'regression', True, True, pipeline)
    df = pd.DataFrame()
    for i in range(iterations):
        d = exp.create_report(cat_mask=[])
        d['iteration'] = i
        df = pd.concat([df, d])
    
    return df

def nycrides_experiment(iterations=5, paper_models=False):

    xgboost = xgb.XGBRegressor(n_estimators=1000, objective="reg:squaredlogerror", random_state=42, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
    lr = LinearRegression(n_jobs=-1)
    dt = DecisionTreeRegressor()
    nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000, activation='logistic', alpha=0.005)
    knn = KNeighborsRegressor(algorithm='kd_tree', n_jobs=-1)
    lgbm = LGBMRegressor(n_estimators=1000, n_jobs=-1, objective='regression', reg_lambda=1, reg_alpha=1)

    if paper_models:
        models = [
            xgboost,  
            nn,  
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
        
        pipeline = get_pipeline(model)
        d = get_report(pipeline, iterations)
        df = pd.concat([df, d])
        export_path = os.path.join(x_folder, 'experiments', 'output', 'nyc_rides')
        df.to_csv(export_path + '_complex_standalone.csv', index=False)
        
if __name__ == "__main__":

    nycrides_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))