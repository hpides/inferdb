import sys
import os
from pathlib import Path
x_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(x_folder, 'src')
featurizer_folder = os.path.join(src_folder, 'featurizers')
sys.path.append(str(featurizer_folder))
sys.path.append(str(src_folder))
from transpiler import SQLmodel, PGML, MADLIB, InferDB
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
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
from nyc_rides_featurizer import NYC_Featurizer 
from create_featurizer_pg import NYC_Featurizer_pg

data_path = os.path.join(x_folder, 'data', 'nyc_rides', 'nyc_rides_augmented.csv')
write_path = os.path.join(x_folder, 'experiments', 'output', 'nyc_rides_pgml.csv')

df = pd.read_csv(data_path)

df.rename(columns={'primary':'primarie'}, inplace=True)

training_features = [i for i in list(df) if i != 'trip_duration']
X = df[training_features]
y = df['trip_duration'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', RobustScaler())
                            ]
                    )
featurizer = NYC_Featurizer('deep')
column_transformer = ColumnTransformer(
                                        transformers=[
                                                        ('num', numerical_transformer, [i for i in range(len(featurizer.num_features))])
                                                        # ('num', numerical_transformer, [])
                                                    ]
                                        , remainder='drop'
                                        , n_jobs=-1

                                    )



pipeline = Pipeline(
                        steps=
                                [   
                                    
                                    ('featurizer', featurizer),
                                    ('imputer', SimpleImputer()),
                                    ('column_transformer', column_transformer)
                                    
                                    # , ('stacking_regressor', StackingRegressor(estimators=estimators, passthrough=True, final_estimator=model, n_jobs=-1))
                                    # ('clf', model)
                                ]
                    )

pipeline.fit(X_train, y_train)

trained_featurizer = pipeline.named_steps['featurizer']

pg_featurizer = NYC_Featurizer_pg(trained_featurizer)
feat_query = pg_featurizer.create_featurizer_query()
pg_featurizer.create_aux_functions()
pg_featurizer.push_cluster_table()
pg_featurizer.push_mappers_table()

xgboost = xgb.XGBRegressor(n_estimators=1000, objective="reg:squaredlogerror", random_state=42, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
lr = LinearRegression(n_jobs=-1)
dt = DecisionTreeRegressor()
nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=10000, activation='relu', alpha=0.005)
knn = KNeighborsRegressor(algorithm='kd_tree', n_jobs=-1)
lgbm = LGBMRegressor(n_estimators=1000, n_jobs=-1, objective='regression', reg_lambda=1, reg_alpha=1)


models = [
    xgboost, 
    lr, 
    nn,  
    lgbm,
    knn,
    dt
    ]
tree_based_models = [xgboost.__class__.__name__, lgbm.__class__.__name__, dt.__class__.__name__]


df = pd.DataFrame()
# ######## SQLModel

SQL_models = [
    lr, 
    nn
    ]
for model in SQL_models:

    # pipeline.named_steps.column_transformer.transformers = [
    #                                                                 ('num', numerical_transformer, trained_featurizer.num_features)
    #                                                             ]
        
    pipeline.steps.append(['clf', model])
    exp = SQLmodel(X_train, X_test, y_train, y_test, 'NYC_rides', 'regression', True, True, pipeline=pipeline, feature_names=training_features, featurize_query=feat_query)
    
    exp.create_test_table()
    exp.insert_test_tuples()
    # exp.create_train_table()
    # exp.insert_train_tuples()

    exp.create_featurizer()
    exp.translate_imputer(source_table='featurized', source_table_feature_names=featurizer.num_features)
    exp.translate_column_transfomer(source_table='imputed', source_table_feature_names=featurizer.num_features) ###### needs to be tailored for each dataset. e.g., for credit no imputation
    exp.create_preprocessing_pipeline() ######### If no preprocessing, source is test

    summary_df = exp.create_report_pg()

    df = pd.concat((df, summary_df))

    df.to_csv(write_path, index=False)

    pipeline.steps.pop(-1)

# # ############ PGML

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
        

    pipeline.steps.append(['clf', model])
    exp = PGML(X_train, X_test, y_train, y_test, 'nyc_rides', 'regression', True, True, pipeline=pipeline, feature_names=training_features, featurizer_query=feat_query)
    
    exp.translate_imputer(source_table='featurized', source_table_feature_names=featurizer.num_features)
    exp.translate_column_transfomer(source_table='imputed', source_table_feature_names=featurizer.num_features)
    # exp.translate_column_transfomer(source_table='featurized', source_table_feature_names=featurizer.num_features) ###### needs to be tailored for each dataset. e.g., for credit no imputation

    exp.create_train_table()
    exp.insert_train_tuples()
    
    summary_df = exp.create_report_pg(model_name_pgml, PGML_models_parameters[model_name]['parameters'])

    df = pd.concat((df, summary_df))

    df.to_csv(write_path, index=False)

    pipeline.steps.pop(-1)

# ####################################### InferDB


infer_db_models = [
    xgboost,
    lr, 
    nn,  
    lgbm,
    knn,
    dt
    ]

for model in infer_db_models:

    pipeline.steps.append(['clf', model])

    exp = InferDB(X_train, X_test, y_train, y_test, 'nyc_rides', 'regression', False,False, True, pipeline=pipeline, inferdb_columns=featurizer.infer_db_subset,feature_names=training_features)
    exp.create_aux_functions()

    summary_df = exp.create_report_pg()

    df = pd.concat((df, summary_df))
    df.to_csv(write_path, index=False)

    pipeline.steps.pop(-1)

    

    