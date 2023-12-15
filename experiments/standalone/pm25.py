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
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer


data_path = os.path.join(project_folder, 'data', 'pm', 'pm_2016_2019_augmented.csv')

# data_path = os.path.join(project_folder, 'data', 'pm', 'pm_2016_2019.csv')
df = pd.read_csv(data_path, nrows=50000000)

# df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)

# df['weekday'] = df['date'].dt.weekday
# df['weekofyear'] = df['date'].dt.isocalendar().week
# df['day'] = df['date'].dt.day
# df = df.sort_values(by=["ctfips", "date"])
# df["previous_value"] = df.groupby("ctfips")["DS_PM_pred"].shift()
# df["previous_value_2"] = df.groupby("ctfips")["DS_PM_pred"].shift(2)
# df["previous_value_3"] = df.groupby("ctfips")["DS_PM_pred"].shift(3)
# df["previous_value_4"] = df.groupby("ctfips")["DS_PM_pred"].shift(4)
# df["previous_value_5"] = df.groupby("ctfips")["DS_PM_pred"].shift(5)
# df["previous_value_6"] = df.groupby("ctfips")["DS_PM_pred"].shift(6)
# df['previous_value_7'] = df.groupby("ctfips")["DS_PM_pred"].shift(7)
# df['previous_value_30'] = df.groupby("ctfips")["DS_PM_pred"].shift(31)
# df['trend_sign'] = df.apply(lambda x: 1 if x['previous_value']/x['previous_value_30'] > 1.05 else -1 if x['previous_value']/x['previous_value_30'] < .95 else 0, axis=1)
# df['trend'] = df.apply(lambda x: x['previous_value']/x['previous_value_30'], axis=1)


# df.set_index(["ctfips", "date"], inplace=True)
# df['rolling_2'] = df.rolling(3, closed='right')["DS_PM_pred"].mean()
# df['rolling_3'] = df.rolling(4, closed='right')["DS_PM_pred"].mean()
# df['rolling_7'] = df.rolling(8, closed='right')["DS_PM_pred"].mean()
# df['rolling_15'] = df.rolling(16, closed='right')["DS_PM_pred"].mean()
# df['rolling_30'] = df.rolling(31, closed='right')["DS_PM_pred"].mean()
# df.reset_index(inplace=True)
# data_path = os.path.join(project_folder, 'data', 'pm', 'pm_2016_2019_augmented.csv')
# df.to_csv(data_path, index=False)
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

categorical_transformer = Pipeline(
                    steps=
                            [
                                ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
                            ]
                    )

categorical_transformer_trees = Pipeline(
    steps=
            [
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]
    )

column_transformer = ColumnTransformer(
                                            transformers=[
                                                            ('num', numerical_transformer, []),
                                                            ('cat', categorical_transformer, [])
                                                        ]
                                            , remainder='drop'
                                            , n_jobs=-1

                                        )

pipeline = Pipeline(
                        steps=
                                [   
                                    ('column_transformer', column_transformer),
                                    ('imputer', SimpleImputer())
                                ]
                    )

xgboost = xgb.XGBRegressor(n_estimators=350, objective="reg:squarederror", random_state=42, n_jobs=-1, min_child_weight=10, subsample=0.8, eta=0.05)
lr = LinearRegression(n_jobs=-1)
dt = DecisionTreeRegressor()
nn = MLPRegressor(hidden_layer_sizes=(len(training_features),), max_iter=10000, activation='logistic')
knn = KNeighborsRegressor(algorithm='kd_tree', n_jobs=-1)
lgbm = LGBMRegressor(n_estimators=350, n_jobs=-1, objective='regression', reg_lambda=1, reg_alpha=1)

models = [
            xgboost, 
            lr, 
            dt, 
            nn, 
            knn, 
            lgbm
        ]
tree_based_models = [xgboost.__class__.__name__, dt.__class__.__name__, lgbm.__class__.__name__]

df = pd.DataFrame()
for model in models:

    if model.__class__.__name__ in tree_based_models:

        params = {'remainder':'passthrough'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('cat', categorical_transformer_trees, [])
                                                                ]
    else:
        params = {'remainder':'drop'}
        column_transformer.set_params(**params)
        pipeline.named_steps.column_transformer.transformers = [
                                                                    ('num', numerical_transformer, num_mask)
                                                                ]

    pipeline.steps.append(['clf', model])

    exp = Standalone(X_train, X_test, y_train, y_test, 'pm25', 'regression', False, False, pipeline, inferdb_indices)
    for i in range(5):
        d = exp.create_report(cat_mask=[])
        d['iteration'] = i
        df = pd.concat([df, d])
        export_path = os.path.join(project_folder, 'experiments', 'output', exp.experiment_name)
        df.to_csv(export_path + '_standalone.csv', index=False)
    
    pipeline.steps.pop(-1)





