import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(x_folder, 'src')
sys.path.append(str(src_folder))
from transpiler import SQLmodel, PGML, MADLIB, InferDB
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pickle import load
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.feature_selection import SelectFromModel
import numpy as np

def get_experiment_data():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'creditcard', 'creditcard.csv')
    mask_path = os.path.join(x_folder, 'data', 'paper_data', 'creditcard', 'cat_mask')
    
    with open(mask_path, 'rb') as d:
        cat_mask = load(d)

    df = pd.read_csv(data_path)

    training_features = [i for i in list(df) if i != 'Class']
    X = df[training_features].to_numpy()
    y = df['Class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    return X_train, X_test, y_train, y_test, training_features

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
                                                            ('num', numerical_transformer, [i for i in range(30)])
                                                        ]
                                            , remainder='passthrough'
                                            , n_jobs=-1

                                        )

    pipeline = Pipeline(
                            steps=
                                    [   
                                        ('column_transformer', column_transformer)
                                        
                                    ]
                        )
    
    if model.__class__.__name__ in tree_based_models:

        numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [])]
        
    else:
        numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [i for i in range(30)])]
    
    pipeline.steps.append(['clf', model])
    
    return pipeline

def get_SQL_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, training_features = get_experiment_data()

    exp = SQLmodel(X_train, X_test, y_train, y_test, 'creditcard', 'classification', False, False, pipeline=pipeline, feature_names=training_features)
    exp.create_aux_functions()
    exp.create_test_table()
    exp.insert_test_tuples()
    exp.create_train_table()
    exp.insert_train_tuples()
    exp.translate_column_transfomer(source_table='test') ###### needs to be tailored for each dataset. e.g., for credit no imputation
    exp.create_preprocessing_pipeline() ######### If no preprocessing, source is test
    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()

    return summary_df

def get_PGML_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, training_features = get_experiment_data()

    exp = PGML(X_train, X_test, y_train, y_test, 'creditcard', 'classification', False, False, pipeline=pipeline, feature_names=training_features)

    PGML_models_parameters = {
                                'XGBClassifier':{ 
                                            'name': "'xgboost'",
                                            'parameters': """'{"n_estimators":350, "gamma":1, "subsample":0.5, "eta":0.05}'"""
                                            },
                                'LGBMClassifier': {
                                                'name': "'lightgbm'",
                                                'parameters':"""'{"n_estimators":350, "class_weight":"balanced", "reg_lambda":1, "reg_alpha":1}'"""
                                            },
                                
                                'LogisticRegression': {
                                                'name': "'linear'",
                                                'parameters': """'{"max_iter": 100000}'"""
                                        
                                            }
                            }
    
    model_name_pgml = PGML_models_parameters[exp.model.__class__.__name__]['name']
    model_name = exp.model.__class__.__name__
    model_parameters = PGML_models_parameters[exp.model.__class__.__name__]['parameters']

    exp.create_aux_functions()
    exp.create_test_table()
    exp.insert_test_tuples()
    exp.create_train_table()
    exp.insert_train_tuples()
    exp.translate_column_transfomer(source_table='test') ### No imputation
    summary_df = exp.create_report_pg(model_name_pgml, PGML_models_parameters[model_name]['parameters'], iterations=iterations)
    exp.clean_up()

    return summary_df
    
def get_INFERDB_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, training_features = get_experiment_data()

    exp = InferDB(X_train, X_test, y_train, y_test, 'creditcard', 'classification', False, False, pipeline=pipeline, feature_names=training_features)
    exp.create_aux_functions()
    exp.create_train_table()
    exp.insert_train_tuples()

    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()

    return summary_df

def creditcard_experiment(iterations=5, paper_models=False):

    write_path = os.path.join(x_folder, 'experiments', 'output', 'creditcard_pgml.csv')

    xgboost = xgb.XGBClassifier(n_estimators=350, objective="binary:logistic", random_state=42, n_jobs=-1, gamma=1, subsample=0.5, eta=0.05)
    lr = LogisticRegression(n_jobs=-1, max_iter=10000)
    nn = MLPClassifier(hidden_layer_sizes=(30,), max_iter=10000, activation='logistic', alpha=0.005)
    lgbm = LGBMClassifier(n_estimators=350, n_jobs=-1, objective='binary', class_weight='balanced', reg_lambda=1, reg_alpha=1)
    knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
    dt = DecisionTreeClassifier()

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
                    xgboost, 
                    lr, 
                    lgbm
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

    creditcard_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))
    

    