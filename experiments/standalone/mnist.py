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

def get_experiment_data():

    data_path = os.path.join(project_folder, 'data', 'paper_data', 'mnist_784', 'mnist_784.csv')

    df = pd.read_csv(data_path)

    training_features = [i for i in list(df) if i != 'class']
    X = df[training_features].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(df['class'].to_numpy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    return X_train, X_test, y_train, y_test

def get_pipeline(model):

    tree_based_models = ['xgbclassifier', 'decisiontreeclassifier', 'lgbmclassifier']

    num_mask = [i for i in range(784)]

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
                                        ('column_transformer', column_transformer)
                                        
                                    ]
                        )
    

    if model.__class__.__name__ in tree_based_models:

        pipeline.named_steps.column_transformer.transformers = [('num', numerical_transformer, [])]
    else:
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_transformer, num_mask)]
            
    pipeline.steps.append(['clf', model])
    
    return pipeline

def get_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()

    exp = Standalone(X_train, X_test, y_train, y_test, 'mnist', 'multi-class', False, False, pipeline=pipeline)
    df = pd.DataFrame()
    for i in range(iterations):
        d = exp.create_report(cat_mask=[])
        d['iteration'] = i
        df = pd.concat([df, d])
    
    return df

def mnist_experiment(iterations=5, paper_models=False):

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
    for model in models:

        pipeline = get_pipeline(model)
        d = get_report(pipeline, iterations)
        df = pd.concat([df, d])
        export_path = os.path.join(project_folder, 'experiments', 'output', 'mnist')
        df.to_csv(export_path + '_standalone.csv', index=False)

if __name__ == "__main__":

    mnist_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))  




