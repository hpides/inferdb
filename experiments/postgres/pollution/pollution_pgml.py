import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(x_folder, 'src')
sys.path.append(str(src_folder))
from transpiler import SQLmodel, PGML, MADLIB, InferDB, Transpiler
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from pickle import load
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from time import time
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.impute import SimpleImputer

def get_pm_experiment_data():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'pm', 'pm_2016_2019_augmented.csv')
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

def create_pg_artifacts(exp:Transpiler):

    exp.create_aux_functions()
    exp.create_test_table()
    exp.insert_test_tuples()
    exp.create_train_table()
    exp.insert_train_tuples()

def get_SQL_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, inferdb_indices, training_features, num_mask = get_pm_experiment_data()

    exp = SQLmodel(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline=pipeline, feature_names=training_features)
    create_pg_artifacts(exp)

    exp.translate_imputer()
    exp.translate_column_transfomer() ###### needs to be tailored for each dataset. e.g., for credit no imputation
    exp.create_preprocessing_pipeline() ######### If no preprocessing, source is test

    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()
    
    return summary_df

def get_PGML_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, inferdb_indices, training_features, num_mask = get_pm_experiment_data()

    exp = PGML(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline=pipeline, feature_names=training_features)

    PGML_models_parameters = {
                                'XGBRegressor':{ 
                                            'name': "'xgboost'",
                                            'parameters': """'{"n_estimators":350, "max_delta_step":10, "gamma":1, "min_child_weight":10, "subsample":0.8, "eta":0.05}'"""
                                            },
                                'LGBMRegressor': {
                                                'name': "'lightgbm'",
                                                'parameters':"""'{"n_estimators":350, "reg_lambda":1, "reg_alpha":1}'"""
                                            },
                                
                                'LinearRegression': {
                                                'name': "'linear'",
                                                'parameters': []
                                        
                                            }
                            }
    
    model_name_pgml = PGML_models_parameters[exp.model.__class__.__name__]['name']
    model_name = exp.model.__class__.__name__
    model_parameters = PGML_models_parameters[exp.model.__class__.__name__]['parameters']

    create_pg_artifacts(exp)
    
    exp.translate_imputer()
    exp.translate_column_transfomer() ###### needs to be tailored for each dataset. e.g., for credit no imputation
    
    summary_df = exp.create_report_pg(model_name_pgml, PGML_models_parameters[model_name]['parameters'], iterations=iterations)

    exp.clean_up()
    
    return summary_df

def get_INFERDB_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, inferdb_indices, training_features, num_mask = get_pm_experiment_data()

    exp = InferDB(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline=pipeline, feature_names=training_features)
    create_pg_artifacts(exp)

    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()

    return summary_df

def pm_experiment(iterations=5, paper_models=False):

    write_path = os.path.join(x_folder, 'experiments', 'output', 'pm25_pgml.csv')

    xgboost = xgb.XGBRegressor(objective="reg:squarederror", max_delta_step=10, gamma=1, random_state=42, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
    lr = LinearRegression(n_jobs=-1)
    dt = DecisionTreeRegressor()
    nn = MLPRegressor(hidden_layer_sizes=(8,), max_iter=10000, activation='relu')
    knn = KNeighborsRegressor(algorithm='kd_tree', n_jobs=-1, leaf_size=10)
    lgbm = LGBMRegressor(n_jobs=-1, objective='regression', reg_lambda=1, reg_alpha=1)

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
    ######## SQLModel

    SQL_models = [
        lr, 
        nn
        ]
    for model in SQL_models:

        if model in models:

            pipeline = get_pipeline(model, num_mask)

            summary_df = get_SQL_report(pipeline, iterations)

            df = pd.concat((df, summary_df))

            df.to_csv(write_path, index=False)

        else:
            continue

    ############ PGML

    PGML_models = [
                    xgboost, 
                    lr, 
                    lgbm
                    ]

    for model in PGML_models:

        if model in models:

            pipeline = get_pipeline(model, num_mask)
            
            summary_df = get_PGML_report(pipeline, iterations)

            df = pd.concat((df, summary_df))

            df.to_csv(write_path, index=False)
        
        else:

            continue

    ####################################### InferDB


    infer_db_models = [
        xgboost,
        lr, 
        nn,  
        lgbm,
        knn,
        dt
        ]

    for model in infer_db_models:

        if model in models:

            pipeline = get_pipeline(model, num_mask)

            summary_df = get_INFERDB_report(pipeline, iterations)

            df = pd.concat((df, summary_df))
            df.to_csv(write_path, index=False)

        else:

            continue

if __name__ == "__main__":

    pm_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))