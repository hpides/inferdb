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
from sklearn.impute import SimpleImputer

data_path = os.path.join(x_folder, 'data', 'rice', 'rice.csv')
write_path = os.path.join(x_folder, 'experiments', 'output', 'rice_pgml.csv')

df = pd.read_csv(data_path)

training_features = [i for i in list(df) if i != 'CLASS']
X = df[training_features].to_numpy()
le = LabelEncoder()
y = le.fit_transform(df['CLASS'].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)


xgboost = xgb.XGBClassifier(objective="multi:softmax", num_class=5, random_state=42, n_jobs=-1)
lr = LogisticRegression(n_jobs=-1, max_iter=10000)
dt = DecisionTreeClassifier()
nn = MLPClassifier(hidden_layer_sizes=(X_train.shape[1],), max_iter=10000, activation='logistic', alpha=0.005)
knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
lgbm = LGBMClassifier(n_estimators=500, n_jobs=-1, reg_lambda=1, reg_alpha=1)


models = [
    xgboost, 
    lr, 
    nn,  
    lgbm,
    knn,
    dt
    ]
tree_based_models = [xgboost.__class__.__name__, lgbm.__class__.__name__]

numerical_imputer = Pipeline(
                    steps= 
                    [   
                        ('imputer', SimpleImputer())
                    ]
)

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
                                    ('column_transformer', column_transformer)
                                    
                                ]
                    )

df = pd.DataFrame()
######## SQLModel

SQL_models = [
    lr, 
    nn
    ]
for model in SQL_models:

    numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
    pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [i for i in range(len(training_features))])]
        
    pipeline.steps.append(['clf', model])
    exp = SQLmodel(X_train, X_test, y_train, y_test, 'rice', 'multi-class', False, False, pipeline=pipeline, feature_names=training_features)
    exp.create_test_table()
    exp.insert_test_tuples()
    exp.create_train_table()
    exp.insert_train_tuples()
    exp.translate_imputer()
    exp.translate_column_transfomer(source_table='imputed') ###### needs to be tailored for each dataset. e.g., for credit no imputation
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
                                            'parameters': """'{"max_iter": 100000}'"""
                                    
                                        }
                        }

# for model in PGML_models:

#     model_name_pgml = PGML_models_parameters[model.__class__.__name__]['name']
#     model_name = model.__class__.__name__
#     model_parameters = PGML_models_parameters[model.__class__.__name__]['parameters']

#     if model.__class__.__name__ in tree_based_models:

#         numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
#         pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [])]
        
#     else:
#         numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
#         pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [i for i in range(len(training_features))])]
        

#     pipeline.steps.append(['clf', model])
#     exp = PGML(X_train, X_test, y_train, y_test, 'rice', 'multi-class', False, False, pipeline=pipeline, feature_names=training_features)
#     # exp.create_test_table()
#     # exp.insert_test_tuples()
    
#     exp.translate_imputer()
#     exp.translate_column_transfomer(source_table='imputed') ###### needs to be tailored for each dataset. e.g., for credit no imputation

#     exp.create_train_table()
#     exp.insert_train_tuples()
    
#     summary_df = exp.create_report_pg(model_name_pgml, PGML_models_parameters[model_name]['parameters'])

#     df = pd.concat((df, summary_df))

#     df.to_csv(write_path, index=False)

#     pipeline.steps.pop(-1)

# ####################################### InferDB


# infer_db_models = [
#     xgboost,
#     lr, 
#     nn,  
#     lgbm,
#     knn,
#     dt
#     ]

# for model in infer_db_models:

#     if model.__class__.__name__ in tree_based_models:

#         numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
#         pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [])]
        
#     else:
#         numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
#         pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [i for i in range(len(training_features))])]

    
#     pipeline.steps.append(['clf', model])

#     exp = InferDB(X_train, X_test, y_train, y_test, 'rice', 'multi-class', False, False, pipeline=pipeline, feature_names=training_features)
#     exp.create_aux_functions()

#     exp.create_train_table()
#     exp.insert_train_tuples()

#     summary_df = exp.create_report_pg()

#     df = pd.concat((df, summary_df))
#     df.to_csv(write_path, index=False)

#     pipeline.steps.pop(-1)

    

    