import sys
import os
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[3]
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

exp_folder = Path(__file__).resolve().parents[3]
data_path = os.path.join(exp_folder, 'data', 'paper_data', 'rice', 'rice.csv')

df = pd.read_csv(data_path)

training_features = [i for i in list(df) if i != 'CLASS']
X = df[training_features]
le = LabelEncoder()
y = le.fit_transform(df['CLASS'].to_numpy())


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

# model = xgb.XGBClassifier(n_estimators=1000, objective="multi:softmax", num_class=10, random_state=42, n_jobs=-1)
model = xgb.XGBClassifier(objective="multi:softmax", num_class=5, random_state=42, n_jobs=-1)


numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', RobustScaler())
                            ]
                    )

column_transformer = ColumnTransformer(
                                        transformers=[
                                                        ('num', numerical_transformer, list(X_train))
                                                    ]
                                        , remainder='passthrough'
                                        , n_jobs=-1

                                    )

pipeline = Pipeline(
                        steps=
                                [   
                                    ('column_transformer', column_transformer)
                                    , ('clf', model)
                                ]
                    )

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)

exp = Standalone(X_train, X_test, y_train, y_test, 'rice', 'multi-class', False, pipeline=pipeline)

d = exp.create_report(cat_mask=[], with_pred=True)

export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
d.to_csv(export_path + '_standalone.csv', index=False)


X_train_sub, X_test_sub = X_train.loc[:, [training_features[i] for i in exp.solution]], X_test.loc[:, [training_features[i] for i in exp.solution]]
# new_cat_mask = [training_features[i] for i in exp.solution if i in cat_mask]
new_num_mask = [training_features[i] for i in exp.solution]

numerical_imputer = pipeline.named_steps.column_transformer.transformers[0]
pipeline.named_steps.column_transformer.transformers[0] = (numerical_imputer[0], numerical_imputer[1], list(X_train_sub))

exp = Standalone(X_train_sub, X_test_sub, y_train, y_test, 'rice', 'multi-class', False, False, pipeline)

d = exp.create_report([], True)

export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
d.to_csv(export_path + '_standalone_rice_simple.csv', index=False)

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, exp.y_pred_trie, display_labels=le.classes_, ax=axs[1])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, ax=axs[0])

plt.show()




