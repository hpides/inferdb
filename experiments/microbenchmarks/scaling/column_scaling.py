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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, RobustScaler
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from pickle import load, dump
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.neighbors import NearestNeighbors
from optbinning import OptimalBinning
from song_hits_featurizer import Hits_Featurizer


exp_folder = Path(__file__).resolve().parents[3]
data_path = os.path.join(exp_folder, 'data', 'hits', 'hits.csv')
mask_path = os.path.join(exp_folder, 'data', 'hits', 'cat_mask')
with open(mask_path, 'rb') as d:
    cat_mask = load(d)

df = pd.read_csv(data_path)

training_features = [i for i in list(df) if i != 'class']
X = df[training_features]
y = df['class'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

sample_weights = compute_sample_weight('balanced', y_train)
# model = xgb.XGBClassifier(n_estimators=1000, objective="binary:logistic", random_state=42, gamma=1, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
n_positive = y_train.sum()
n_negative = y_train.shape[0] - n_positive
model = xgb.XGBClassifier(n_estimators=500, objective="binary:logistic", max_delta_step=10, eval_metric='auc', random_state=42, gamma=1, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)


feature_list = list(X_train)
cat_mask_names = [i for idx, i in enumerate(feature_list) if idx in cat_mask]
num_mask_names = [i for idx, i in enumerate(feature_list) if idx not in cat_mask]

featurizer = Hits_Featurizer(cat_mask_names)

for c in cat_mask_names:

    cat_mask_names_encoded = []
    cat_mask_names_encoded.extend([c + '_encoded'])
    feature_list.extend([c + '_encoded'])

numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', RobustScaler())
                            ]
                    )

categorical_transformer = Pipeline(
                    steps=
                            [
                                ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
                            ]
                    )


column_transformer = ColumnTransformer(
                                            transformers=[
                                                            ('num', numerical_transformer, num_mask_names)
                                                            , ('cat', categorical_transformer, cat_mask_names_encoded)
                                                        ]
                                            , remainder='drop'
                                            , n_jobs=-1

                                        )

pipeline = Pipeline(
                        steps=
                                [   
                                ('featurizer', featurizer),
                                    ('column_transformer', column_transformer),
                                    ('clf', model)
                                ]
                    )

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Results for Full Model:')
print('ACC: ' + str(acc))
print('f1: ' + str(f1))
print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('_________________________')

exp = Standalone(X_train, X_test, y_train, y_test, 'hits', 'classification', True, False, pipeline)

d = exp.create_report(cat_mask, True)

export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
d.to_csv(export_path + '_standalone_column_scaling.csv', index=False)

# X_train_sub, X_test_sub = X_train.loc[:, [training_features[i] for i in exp.solution]], X_test.loc[:, [training_features[i] for i in exp.solution]]
# new_cat_mask = [training_features[i] for i in exp.solution if i in cat_mask]
# new_num_mask = [training_features[i] for i in exp.solution if i not in cat_mask]

# for c in new_cat_mask:

#     cat_mask_names_encoded = []
#     cat_mask_names_encoded.extend([c + '_encoded'])

# numerical_imputer = pipeline.named_steps.column_transformer.transformers[0]
# categorical_imputer = pipeline.named_steps.column_transformer.transformers[1]
# pipeline.named_steps.column_transformer.transformers[0] = (numerical_imputer[0], numerical_imputer[1], new_num_mask)
# pipeline.named_steps.column_transformer.transformers[1] = (categorical_imputer[0], categorical_imputer[1], cat_mask_names_encoded)


# pipeline.set_params(featurizer__cat_mask_names=new_cat_mask, clf__n_estimators=10)


# exp = Standalone(X_train_sub, X_test_sub, y_train, y_test, 'hits', 'classification', True, False, pipeline)

# d = exp.create_report([0], True)

# export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
# d.to_csv(export_path + '_standalone_column_scaling_simple.csv', index=False)


# pipeline.fit(X_train_sub, y_train)

# y_pred = pipeline.predict(X_test_sub)

# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

# print('Results for Partial Model:')
# print('ACC: ' + str(acc))
# print('f1: ' + str(f1))
# print('precision: ' + str(precision))
# print('recall: ' + str(recall))
# print('_________________________')

# for s in np.arange(0.2, 1.2, 0.2):

#     column_sample = np.random.choice(X_train.shape[1], round(X_train.shape[1] * s), replace=False)

#     selected_columns = [feature_list[i] for i in column_sample]

#     X_train_sub = X_train.loc[:, selected_columns]
#     X_test_sub = X_test.loc[:, selected_columns]

#     cat_mask_names_sub = [i for i in selected_columns if i in cat_mask_names]
#     num_mask_names_sub = [i for i in selected_columns if i not in cat_mask_names]

#     for c in cat_mask_names_sub:

#         cat_mask_names_encoded = []
#         cat_mask_names_encoded.extend([c + '_encoded'])
#         feature_list.extend([c + '_encoded'])

#     pipeline.fit(X_train_sub, y_train)
#     # pipeline.fit(X_res, y_res)

#     y_pred = pipeline.predict(X_test_sub)

#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     print('Results for ' + str(s*100) + '%:')
#     print('ACC: ' + str(acc))
#     print('f1: ' + str(f1))
#     print('precision: ' + str(precision))
#     print('recall: ' + str(recall))
#     print('_________________________')

#     exp = Standalone('hits', model, 'classification', False, column_subset=selected_columns)

#     d = exp.create_report()

#     export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
#     d.to_csv(export_path + '_standalone_column_scaling' + str(s*100) + '.csv', index=False)

# import matplotlib.pyplot as plt

# from sklearn.metrics import ConfusionMatrixDisplay

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# ConfusionMatrixDisplay.from_predictions(y_test, exp.y_pred_trie, ax=axs[1])
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axs[0])

# plt.show()




