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
from sklearn.impute import SimpleImputer

data_path = os.path.join(project_folder, 'data', 'rice', 'rice.csv')

df = pd.read_csv(data_path)

training_features = [i for i in list(df) if i != 'CLASS']
X = df[training_features].to_numpy()
le = LabelEncoder()
y = le.fit_transform(df['CLASS'].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

cat_mask = []
num_mask = [i for i in range(X_train.shape[1])]

xgboost = xgb.XGBClassifier(objective="multi:softmax", num_class=5, random_state=42, n_jobs=-1)
lr = LogisticRegression(n_jobs=-1, max_iter=10000)
dt = DecisionTreeClassifier()
nn = MLPClassifier(hidden_layer_sizes=(X_train.shape[1],), max_iter=10000, activation='logistic', alpha=0.005)
knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
lgbm = LGBMClassifier(n_estimators=500, n_jobs=-1, reg_lambda=1, reg_alpha=1)

models = [
    xgboost, 
    lr, 
    dt, 
    nn, 
    knn, 
    lgbm
    ]
tree_based_models = [xgboost.__class__.__name__, dt.__class__.__name__, lgbm.__class__.__name__]

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
for model in models:

    if model.__class__.__name__ in tree_based_models:
        params = {'remainder':'passthrough'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, [])
                                                                ]
    else:
        params = {'remainder':'passthrough'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, num_mask)
                                                                ]
        
    pipeline.steps.append(['clf', model])
    
    for i in range(5):
        x_trans = pipeline[:-1].fit_transform(X_train)
        print(np.argwhere(np.isnan(x_trans)))
        exp = Standalone(X_train, X_test, y_train, y_test, 'rice', 'multi-class', False, False, pipeline=pipeline)
        d = exp.create_report(cat_mask=cat_mask)
        d['iteration'] = i
        df = pd.concat([df, d])
        export_path = os.path.join(project_folder, 'experiments', 'output', exp.experiment_name)
        df.to_csv(export_path + '_standalone.csv', index=False)

    pipeline.steps.pop(-1)

# pipeline.fit(X_train, y_train)

# y_pred = pipeline.predict(X_test)

# acc = accuracy_score(y_test, y_pred)

# exp = Standalone(X_train, X_test, y_train, y_test, 'rice', 'multi-class', False, pipeline=pipeline)

# d = exp.create_report(cat_mask=[], with_pred=True)

# export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
# d.to_csv(export_path + '_standalone.csv', index=False)


# X_train_sub, X_test_sub = X_train.loc[:, [training_features[i] for i in exp.solution]], X_test.loc[:, [training_features[i] for i in exp.solution]]
# # new_cat_mask = [training_features[i] for i in exp.solution if i in cat_mask]
# new_num_mask = [training_features[i] for i in exp.solution]

# numerical_imputer = pipeline.named_steps.column_transformer.transformers[0]
# pipeline.named_steps.column_transformer.transformers[0] = (numerical_imputer[0], numerical_imputer[1], list(X_train_sub))

# exp = Standalone(X_train_sub, X_test_sub, y_train, y_test, 'rice', 'multi-class', False, False, pipeline)

# d = exp.create_report([], True)

# export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
# d.to_csv(export_path + '_standalone_rice_simple.csv', index=False)

# import matplotlib.pyplot as plt

# from sklearn.metrics import ConfusionMatrixDisplay

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# ConfusionMatrixDisplay.from_predictions(y_test, exp.y_pred_trie, display_labels=le.classes_, ax=axs[1])
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_, ax=axs[0])

# plt.show()




