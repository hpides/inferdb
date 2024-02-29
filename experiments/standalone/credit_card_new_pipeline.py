import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[2]
src_folder = os.path.join(x_folder, 'src')
sys.path.append(str(src_folder))
from transpiler import Standalone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import  f1_score, recall_score, precision_score
from pathlib import Path
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from pickle import load

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

    return X_train, X_test, y_train, y_test

def get_pipeline(model):

    tree_based_models = ['xgbclassifier', 'decisiontreeclassifier', 'lgbmclassifier']

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
        pipeline.named_steps.column_transformer.transformers[0] = (numerical_scaler[0], numerical_scaler[1], [])
    else:
        numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
        pipeline.named_steps.column_transformer.transformers[0] = (numerical_scaler[0], numerical_scaler[1], [i for i in range(30)])
        
    pipeline.steps.append(['clf', model])

    return pipeline

def get_report(pipeline, iterations):

    X_train, X_test, y_train, y_test = get_experiment_data()

    df = pd.DataFrame()

    exp = Standalone(X_train, X_test, y_train, y_test, 'creditcard', 'classification', False, False, pipeline=pipeline)
    for i in range(iterations):
        d = exp.create_report(cat_mask=[])
        d['iteration'] = i
        df = pd.concat([df, d])
    
    return df


def creditcard_experiment(iterations=5, paper_models=False):

    xgboost = xgb.XGBClassifier(n_estimators=350, objective="binary:logistic", random_state=42, n_jobs=-1, gamma=1, subsample=0.5, eta=0.05)
    lr = LogisticRegression(n_jobs=-1, max_iter=10000)
    dt = DecisionTreeClassifier()
    nn = MLPClassifier(hidden_layer_sizes=(30,), max_iter=10000, activation='logistic', alpha=0.005)
    knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=350, n_jobs=-1, objective='binary', class_weight='balanced', reg_lambda=1, reg_alpha=1)

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
        export_path = os.path.join(x_folder, 'experiments', 'output', 'creditcard')
        df.to_csv(export_path + '_standalone.csv', index=False)

if __name__ == "__main__":

    creditcard_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))
    


        