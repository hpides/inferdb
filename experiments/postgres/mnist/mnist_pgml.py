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

def get_experiment_data():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'mnist_784', 'mnist_784.csv')

    df = pd.read_csv(data_path)

    training_features = [i for i in list(df) if i != 'class']
    X = df[training_features].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(df['class'].to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    return X_train, X_test, y_train, y_test

def get_masks():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'mnist_784', 'mnist_784.csv')
    write_path = os.path.join(x_folder, 'experiments', 'output', 'mnist_pgml.csv')

    df = pd.read_csv(data_path, nrows=5)

    training_features = [i for i in list(df) if i != 'class']

    cat_mask = []
    num_mask = [i for i in range(len(training_features))]

    return num_mask, cat_mask, training_features

def create_pg_artifacts(exp:Transpiler):

    exp.create_aux_functions()
    exp.create_test_table()
    exp.insert_test_tuples()
    exp.create_train_table()
    exp.insert_train_tuples()

def get_pipeline(model):

    tree_based_models = ['xgbclassifier', 'decisiontreeclassifier', 'lgbmclassifier']

    num_mask, cat_mask, training_features = get_masks()

    numerical_transformer = Pipeline(
                        steps=
                                [
                                    ('scaler', RobustScaler())
                                ]
                        )

    column_transformer = ColumnTransformer(
                                            transformers=[
                                                            ('num', numerical_transformer, [i for i in range(len(training_features))])
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
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [i for i in range(len(training_features))])]
    
    pipeline.steps.append(['clf', model])

    return pipeline

def get_SQL_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()
    n_mask, cat_mask, training_features = get_masks()

    exp = SQLmodel(X_train, X_test, y_train, y_test, 'mnist', 'multi-class', False, False, pipeline=pipeline, feature_names=training_features)
    create_pg_artifacts(exp)

    exp.translate_column_transfomer(source_table='test') ###### needs to be tailored for each dataset. e.g., for credit no imputation
    exp.create_preprocessing_pipeline() ######### If no preprocessing, source is test
    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()

    return summary_df

def get_PGML_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()
    n_mask, cat_mask, training_features = get_masks()

    exp = PGML(X_train, X_test, y_train, y_test, 'mnist', 'multi-class', False, False, pipeline=pipeline, feature_names=training_features)
    create_pg_artifacts(exp)

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

    exp.translate_column_transfomer(source_table='test') ### No imputation
            
    summary_df = exp.create_report_pg(model_name_pgml, PGML_models_parameters[model_name]['parameters'], iterations=iterations)

    exp.clean_up()

    return summary_df

def get_INFERDB_report(pipeline, iterations):


    X_train, X_test, y_train, y_test = get_experiment_data()
    n_mask, cat_mask, training_features = get_masks()

    exp = InferDB(X_train, X_test, y_train, y_test, 'mnist', 'multi-class', False, False, pipeline=pipeline, feature_names=training_features)
    create_pg_artifacts(exp)

    summary_df = exp.create_report_pg(iterations=iterations)

    exp.clean_up()

    return summary_df




def mnist_experiment(iterations=5, paper_models=False):

    write_path = os.path.join(x_folder, 'experiments', 'output', 'mnist_pgml.csv')

    xgboost = xgb.XGBClassifier(n_estimators=500, objective="multi:softmax", num_class=10, random_state=42, n_jobs=-1)
    lr = LogisticRegression(n_jobs=-1, max_iter=10000)
    dt = DecisionTreeClassifier()
    nn = MLPClassifier(hidden_layer_sizes=(784,), max_iter=10000, activation='logistic', alpha=0.005)
    knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=500, n_jobs=-1, reg_lambda=1, reg_alpha=1)

    if paper_models:
        models = [
            xgboost, 
            lr, 
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

    mnist_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))