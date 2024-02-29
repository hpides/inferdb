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
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
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
from song_hits_featurizer import Hits_Featurizer
from sklearn.feature_selection import SelectFromModel

def get_experiment_data():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'hits_augmented.csv')
    mask_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'cat_mask')
    with open(mask_path, 'rb') as d:
        cat_mask = load(d)

    df = pd.read_csv(data_path)

    training_features = [i for i in list(df) if i != 'class']
    X = df[training_features]
    y = df['class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    return X_train, X_test, y_train, y_test, cat_mask

def get_feature_names_list():

    data_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'hits_augmented.csv')
    df = pd.read_csv(data_path, nrows=5)
    training_features = [i for i in list(df) if i != 'class']

    mask_path = os.path.join(x_folder, 'data', 'paper_data', 'hits', 'cat_mask')
    with open(mask_path, 'rb') as d:
        cat_mask = load(d)

    return training_features, cat_mask


def get_pipeline(model):

    feature_list, cat_mask = get_feature_names_list()
    cat_mask_names = [i for idx, i in enumerate(feature_list) if idx in cat_mask]
    num_mask_names = [i for idx, i in enumerate(feature_list) if idx not in cat_mask]

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
                                                                ('num', numerical_transformer, num_mask_names),
                                                                ('cat', categorical_transformer, cat_mask_names)
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

def get_report(pipeline, iterations):

    X_train, X_test, y_train, y_test, cat_mask = get_experiment_data()

    df = pd.DataFrame()

    exp = Standalone(X_train, X_test, y_train, y_test, 'hits', 'classification', False, False, pipeline=pipeline)
    for i in range(iterations):
        d = exp.create_report(cat_mask=cat_mask)
        d['iteration'] = i
        df = pd.concat([df, d])
    
    return df


def hits_experiment(iterations=5, paper_models=False):

    xgboost = xgb.XGBClassifier(n_estimators=500, objective="binary:logistic", max_delta_step=10, eval_metric='auc', random_state=42, gamma=1, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
    lr = LogisticRegression(n_jobs=-1, max_iter=100000000, C=0.05, solver='saga' )
    dt = DecisionTreeClassifier()
    nn = MLPClassifier(max_iter=10000, activation='logistic', alpha=0.005)
    knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=500, n_jobs=-1, objective='binary', class_weight='balanced', reg_lambda=1, reg_alpha=1)

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
    for model in models:
            
        pipeline = get_pipeline(model)
        d = get_report(pipeline, iterations)
        df = pd.concat([df, d])
        export_path = os.path.join(x_folder, 'experiments', 'output', 'hits')
        df.to_csv(export_path + '_standalone.csv', index=False)

if __name__ == "__main__":

    hits_experiment(iterations=int(sys.argv[1]), paper_models=bool(sys.argv[2]))