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

data_path = os.path.join(x_folder, 'data', 'creditcard', 'creditcard.csv')
mask_path = os.path.join(x_folder, 'data', 'creditcard', 'cat_mask')
write_path = os.path.join(x_folder, 'experiments', 'output', 'creditcard_madlib.csv')
with open(mask_path, 'rb') as d:
    cat_mask = load(d)

df = pd.read_csv(data_path)

training_features = [i for i in list(df) if i != 'Class']
X = df[training_features].to_numpy()
y = df['Class'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)


lr = LogisticRegression(n_jobs=-1, max_iter=10000)
nn = MLPClassifier(hidden_layer_sizes=(X_train.shape[1],), max_iter=10000, activation='logistic', alpha=0.005)
knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
dt = DecisionTreeClassifier()


madlib_models = [
    lr, 
    nn,
    dt,
    knn
    ]
tree_based_models = [dt.__class__.__name__]

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


for model in madlib_models:

    if model.__class__.__name__ in tree_based_models:

        numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [])]
        
    else:
        numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [i for i in range(len(training_features))])]
 
    if model.__class__.__name__ == 'KNeighborsClassifier':
        pipeline.steps.append(
                                [
                                'feature_selection', SelectFromModel(xgb.XGBClassifier(objective="binary:logistic", max_delta_step=10, eval_metric='auc', random_state=42, gamma=1, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05))
                                ]
                            )
        
        pipeline.steps.append(['clf', model])
        
        pipeline[:-1].fit(X_train, y_train)

        support = pipeline.named_steps.feature_selection.get_support()

        support = [idx for idx, i in enumerate(support) if i]

        training_features = [training_features[i] for i in support]

        sample = np.random.randint(X_train.shape[0], size=round(X_train.shape[0] * 0.10))

        X_train = X_train[sample, :]
        X_train = X_train[:, support]
        y_train = y_train[sample]
        X_test = X_test[:, support]

        numerical_scaler = pipeline.named_steps.column_transformer.transformers[0]
        pipeline.named_steps.column_transformer.transformers = [('num', numerical_scaler[1], [i for i in range(len(support))])]

        pipeline.steps.pop(-1)
        pipeline.steps.pop(-1)
    
    pipeline.steps.append(['clf', model])

    exp = MADLIB(X_train, X_test, y_train, y_test, 'creditcard', 'classification', False, False, pipeline=pipeline, feature_names=training_features)
    exp.create_test_table()
    exp.insert_test_tuples()

    exp.create_train_table()
    exp.insert_train_tuples()
    exp.translate_column_transfomer(source_table='test')

    if exp.model_name in ('linearregression', 'logisticregression'):
        exp.create_lr_solution()
    elif exp.model_name in ('decisiontreeclassifier', 'decisiontreeregressor'):
        exp.create_dt_solution()
    elif exp.model_name in ('mlpclassifier', 'mlpregressor'):
        exp.create_mlp_solution('n_iterations=1000', 100)
    elif exp.model_name in ('kneighborsclassifier', 'kneighborsregressor'):
        exp.create_knn_solution()

    summary_df = exp.create_report_pg()

    df = pd.concat((df, summary_df))

    df.to_csv(write_path, index=False)

    pipeline.steps.pop(-1)