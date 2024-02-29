import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(x_folder, 'src')
sys.path.append(str(src_folder))
from transpiler import Standalone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, RobustScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from transpiler import SQLmodel, PGML, MADLIB, InferDB, Transpiler
from pickle import load
from sklearn.feature_selection import SelectFromModel

def get_experiment_data():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'hits_augmented.csv')

    df = pd.read_csv(data_path)
    mask_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'cat_mask')
    with open(mask_path, 'rb') as d:
        cat_mask = load(d)

    training_features = [i for i in list(df) if i != 'class']
    X = df[training_features].to_numpy()
    y = df['class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    num_mask = [i for i in range(X_train.shape[1]) if i not in cat_mask]

    return X_train, X_test, y_train, y_test

def get_masks():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'hits_augmented.csv')

    df = pd.read_csv(data_path, nrows=5)
    mask_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'cat_mask')
    with open(mask_path, 'rb') as d:
        cat_mask = load(d)

    training_features = [i for i in list(df) if i != 'class']

    num_mask = [i for i in range(len(training_features)) if i not in cat_mask]

    return num_mask, cat_mask, training_features

def create_pg_artifacts(exp:Transpiler):

    exp.create_aux_functions()
    exp.create_test_table()
    exp.insert_test_tuples()
    exp.create_train_table()
    exp.insert_train_tuples()

def get_pipeline(model):

    num_mask, cat_mask, training_features = get_masks()

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


    column_transformer = ColumnTransformer(
                                                transformers=[
                                                                ('num', numerical_transformer, num_mask),
                                                                ('cat', categorical_transformer, cat_mask)
                                                            ]
                                                , remainder='drop'
                                                , n_jobs=-1

                                            )

    pipeline = Pipeline(
                            steps=
                                    [   
                                    ('column_transformer', column_transformer)
                                    ]
                        )
    
    pipeline.steps.append(['clf', model])

    return pipeline

def get_SQL_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()
    num_mask, cat_mask, training_features = get_masks()

    exp = SQLmodel(X_train, X_test, y_train, y_test, 'hits', 'classification', False, False, pipeline=pipeline, feature_names=training_features)
    create_pg_artifacts(exp)

    exp.translate_column_transfomer(source_table='test') ###### needs to be tailored for each dataset. e.g., for credit no imputation
    exp.create_preprocessing_pipeline() ######### If no preprocessing, source is test
    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()

    return summary_df

def get_PGML_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()
    num_mask, cat_mask, training_features = get_masks()

    exp = PGML(X_train, X_test, y_train, y_test, 'hits', 'classification', False, False, pipeline=pipeline, feature_names=training_features)

    PGML_models_parameters = {
                                'XGBClassifier':{ 
                                            'name': "'xgboost'",
                                            'parameters': """'{"n_estimators":500, "max_delta_step":10, "gamma":1, "min_child_weight":10, "subsample":0.8, "eta":0.05}'"""
                                            },
                                'LGBMClassifier': {
                                                'name': "'lightgbm'",
                                                'parameters':"""'{"n_estimators":500, "class_weight":"balanced", "reg_lambda":1, "reg_alpha":1}'"""
                                            },
                                
                                'LogisticRegression': {
                                                'name': "'linear'",
                                                'parameters': """'{"max_iter": 100000}'"""
                                        
                                            }
                            }
    
    model_name_pgml = PGML_models_parameters[exp.model.__class__.__name__]['name']
    model_name = exp.model.__class__.__name__
    model_parameters = PGML_models_parameters[exp.model.__class__.__name__]['parameters']

    create_pg_artifacts(exp)

    exp.translate_column_transfomer(source_table='test') ### No imputation
    
    summary_df = exp.create_report_pg(model_name_pgml, PGML_models_parameters[model_name]['parameters'], iterations=iterations)

    exp.clean_up()

    return summary_df
    
def get_INFERDB_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()
    num_mask, cat_mask, training_features = get_masks()

    exp = InferDB(X_train, X_test, y_train, y_test, 'hits', 'classification', False, False, pipeline=pipeline, feature_names=training_features)

    create_pg_artifacts(exp)

    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()

    return summary_df

def hits_experiment(iterations=5, paper_models=False):

    write_path = os.path.join(x_folder, 'experiments', 'output', 'hits_pgml.csv')

    xgboost = xgb.XGBClassifier(n_estimators=500, objective="multi:softmax", num_class=10, random_state=42, n_jobs=-1)
    lr = LogisticRegression(n_jobs=-1, max_iter=10000)
    dt = DecisionTreeClassifier()
    nn = MLPClassifier(hidden_layer_sizes=100, max_iter=10000, activation='relu', alpha=0.005)
    knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=500, n_jobs=-1, reg_lambda=1, reg_alpha=1)

    if paper_models:
        models = [
            xgboost,  
            dt, 
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

    SQL_models = [lr, nn]
    for model in SQL_models:

        if model in models:

            pipeline = get_pipeline(model)
            summary_df = get_SQL_report(pipeline, iterations)

            df = pd.concat((df, summary_df))

            df.to_csv(write_path, index=False)
        
        else:
            continue

    ############# PGML

    PGML_models = [
                    lgbm,
                    xgboost, 
                    lr
                    
                    ]

    for model in PGML_models:

        if model in models:
                
            pipeline = get_pipeline(model)
            
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
        
            pipeline = get_pipeline(model)

            summary_df = get_INFERDB_report(pipeline, iterations)

            df = pd.concat((df, summary_df))
            df.to_csv(write_path, index=False)

        else:
            continue

if __name__ == "__main__":

    hits_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))