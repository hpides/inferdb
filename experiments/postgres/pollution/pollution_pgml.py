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

data_path = os.path.join(x_folder, 'data', 'pm', 'pm_2016_2019_augmented.csv')
write_path = os.path.join(x_folder, 'experiments', 'output', 'pm25_pgml.csv')

df = pd.read_csv(data_path, nrows=50000000)

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
test_threshold = '2017-03-31'
X_train = df.loc[df.index < training_threshold, training_features].bfill()
X_test = df.loc[(df.index > training_threshold) & (df.index <= test_threshold), training_features].bfill()
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = df.loc[df.index < training_threshold, 'DS_PM_pred'].to_numpy()
y_test = df.loc[(df.index > training_threshold) & (df.index <= test_threshold), 'DS_PM_pred'].to_numpy()

X_train, X_test = X_train.to_numpy(), X_test.to_numpy()

sample_indices = np.random.randint(X_train.shape[0], size=round(X_train.shape[0] * 0.3))

X_train = X_train[sample_indices]
y_train = y_train[sample_indices]

xgboost = xgb.XGBRegressor(n_estimators=350, objective="reg:squarederror", random_state=42, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
lr = LinearRegression(n_jobs=-1)
dt = DecisionTreeRegressor()
nn = MLPRegressor(hidden_layer_sizes=(len(training_features),), max_iter=10000, activation='relu')
knn = KNeighborsRegressor(algorithm='kd_tree', n_jobs=-1)
lgbm = LGBMRegressor(n_estimators=350, n_jobs=-1, objective='regression', reg_lambda=1, reg_alpha=1)


models = [
    xgboost, 
    lr, 
    nn,  
    lgbm,
    knn,
    dt
    ]
tree_based_models = [xgboost.__class__.__name__, lgbm.__class__.__name__, dt.__class__.__name__]

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

df = pd.DataFrame()
######## SQLModel

SQL_models = [
    lr, 
    nn
    ]
for model in SQL_models:

    pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, num_mask)
                                                                ]
        
    pipeline.steps.append(['clf', model])
    exp = SQLmodel(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline=pipeline, feature_names=training_features)
    exp.create_test_table()
    exp.insert_test_tuples()
    exp.create_train_table()
    exp.insert_train_tuples()

    exp.translate_imputer()
    exp.translate_column_transfomer() ###### needs to be tailored for each dataset. e.g., for credit no imputation
    exp.create_preprocessing_pipeline() ######### If no preprocessing, source is test

    summary_df = exp.create_report_pg()

    df = pd.concat((df, summary_df))

    df.to_csv(write_path, index=False)

    pipeline.steps.pop(-1)

############ PGML

PGML_models = [
                xgboost, 
                lr, 
                lgbm
                ]

PGML_models_parameters = {
                            xgboost.__class__.__name__:{ 
                                        'name': "'xgboost'",
                                        'parameters': """'{"n_estimators":500}'"""
                                        },
                            lgbm.__class__.__name__: {
                                            'name': "'lightgbm'",
                                            'parameters':"""'{"n_estimators":500, "reg_lambda":1, "reg_alpha":1}'"""
                                        },
                            
                            lr.__class__.__name__: {
                                            'name': "'linear'",
                                            'parameters': []
                                    
                                        }
                        }

for model in PGML_models:

    model_name_pgml = PGML_models_parameters[model.__class__.__name__]['name']
    model_name = model.__class__.__name__
    model_parameters = PGML_models_parameters[model.__class__.__name__]['parameters']

    if model.__class__.__name__ in tree_based_models:

        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, [])
                                                                ]
        
    else:
        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, num_mask)
                                                                ]
        

    pipeline.steps.append(['clf', model])
    exp = PGML(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline=pipeline, feature_names=training_features)
    
    exp.translate_imputer()
    exp.translate_column_transfomer() ###### needs to be tailored for each dataset. e.g., for credit no imputation

    exp.create_train_table()
    exp.insert_train_tuples()
    
    summary_df = exp.create_report_pg(model_name_pgml, PGML_models_parameters[model_name]['parameters'])

    df = pd.concat((df, summary_df))

    df.to_csv(write_path, index=False)

    pipeline.steps.pop(-1)

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

    if model.__class__.__name__ in tree_based_models:

        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, [])
                                                                ]
        
    else:
        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, num_mask)
                                                                ]

    pipeline.steps.append(['clf', model])

    exp = InferDB(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline=pipeline, feature_names=training_features)
    exp.create_aux_functions()

    summary_df = exp.create_report_pg()

    df = pd.concat((df, summary_df))
    df.to_csv(write_path, index=False)

    pipeline.steps.pop(-1)

    

    