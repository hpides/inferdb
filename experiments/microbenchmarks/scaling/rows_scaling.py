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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, mean_squared_log_error
from pickle import load, dump
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.neighbors import NearestNeighbors
from optbinning import OptimalBinning
from song_hits_featurizer import Hits_Featurizer
from sklearn.linear_model import LinearRegression
from pm_featurizer import PM_Featurizer


exp_folder = Path(__file__).resolve().parents[3]
data_path = os.path.join(exp_folder, 'data', 'pm', 'pm_2016.csv')

df = pd.read_csv(data_path)

df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)

df['weekday'] = df['date'].dt.weekday
df['weekofyear'] = df['date'].dt.isocalendar().week
df['day'] = df['date'].dt.day
df = df.sort_values(by=["ctfips", "date"])
df["previous_value"] = df.groupby("ctfips")["DS_PM_pred"].shift()
df["previous_value_2"] = df.groupby("ctfips")["DS_PM_pred"].shift(2)
df["previous_value_3"] = df.groupby("ctfips")["DS_PM_pred"].shift(3)
df["previous_value_4"] = df.groupby("ctfips")["DS_PM_pred"].shift(4)
df["previous_value_5"] = df.groupby("ctfips")["DS_PM_pred"].shift(5)
df["previous_value_6"] = df.groupby("ctfips")["DS_PM_pred"].shift(6)
df['previous_value_7'] = df.groupby("ctfips")["DS_PM_pred"].shift(7)
df['previous_value_30'] = df.groupby("ctfips")["DS_PM_pred"].shift(31)
df['trend_sign'] = df.apply(lambda x: 1 if x['previous_value']/x['previous_value_30'] > 1.05 else -1 if x['previous_value']/x['previous_value_30'] < .95 else 0, axis=1)
df['trend'] = df.apply(lambda x: x['previous_value']/x['previous_value_30'], axis=1)


df.set_index(["ctfips", "date"], inplace=True)
df['rolling_2'] = df.rolling(3, closed='right')["DS_PM_pred"].mean()
df['rolling_3'] = df.rolling(4, closed='right')["DS_PM_pred"].mean()
df['rolling_7'] = df.rolling(8, closed='right')["DS_PM_pred"].mean()
df['rolling_15'] = df.rolling(16, closed='right')["DS_PM_pred"].mean()
df['rolling_30'] = df.rolling(31, closed='right')["DS_PM_pred"].mean()
df.reset_index(inplace=True)

training_features = ['statefips', 'countyfips', 'ctfips', 'latitude', 'longitude', 
                     'weekday', 'weekofyear', 'day', 
                     'previous_value', 'previous_value_2', 'previous_value_3', 'previous_value_4', 'previous_value_5', 'previous_value_6', 'previous_value_7', 'previous_value_30', 
                     'rolling_2', 'rolling_3', 'rolling_7', 'rolling_15', 'rolling_30',
                     'trend', 'trend_sign']
# X = df[training_features]
# y = np.log(df['DS_PM_pred'].to_numpy())
y = df['DS_PM_pred'].to_numpy()

threshold = '2016-02-28'
X_train = df.loc[df['date'] <= threshold, training_features].bfill()
X_test = df.loc[df['date'] > threshold, training_features].bfill()
X_train = df.loc[df['date'] <= threshold, training_features].fillna(0)
X_test = df.loc[df['date'] > threshold, training_features].fillna(0)
y_train = y[X_train.index.array]
y_test = y[X_test.index.array]

X_train, X_test = X_train.to_numpy(), X_test.to_numpy()

# print(X_train.shape, X_test.shape)

model = xgb.XGBRegressor(n_estimators=100, objective="reg:squarederror", random_state=42, n_jobs=-1)
# model = LinearRegression(n_jobs=-1)

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
                                                            ('num', numerical_transformer, []),
                                                            ('cat', categorical_transformer, [])
                                                        ]
                                            , remainder='passthrough'
                                            , n_jobs=-1

                                        )

pipeline = Pipeline(
                        steps=
                                [   
                                    # ('featurizer', PM_Featurizer()),
                                    ('column_transformer', column_transformer),
                                    ('clf', model)
                                ]
                    )

pipeline.fit(X_train, np.log(y_train))

y_pred = pipeline.predict(X_test)

error = mean_squared_log_error(y_test, np.exp(y_pred), squared=False)

print(error)

inferdb_columns = ['statefips', 'countyfips', 'ctfips', 'latitude', 'longitude', 
                   'weekday', 'day'
                   ]
inderdb_column_indices = [training_features.index(i) for i in inferdb_columns]

exp = Standalone(X_train, X_test, y_train, y_test, 'pm', 'regression', False, False, pipeline, inderdb_column_indices)

d = exp.create_report([0, 1, 2])

export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
d.to_csv(export_path + '_standalone_rows_scaling.csv', index=False)

# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

# print('Results for Full Model:')
# print('ACC: ' + str(acc))
# print('f1: ' + str(f1))
# print('precision: ' + str(precision))
# print('recall: ' + str(recall))
# print('_________________________')

# exp = Standalone(X_train, X_test, y_train, y_test, 'hits', 'classification', True, False, pipeline)

# d = exp.create_report(cat_mask, True)

# export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
# d.to_csv(export_path + '_standalone_column_scaling.csv', index=False)

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




