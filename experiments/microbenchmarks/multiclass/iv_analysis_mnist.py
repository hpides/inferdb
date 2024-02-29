import sys
import os
from pathlib import Path 
project_folder = Path(__file__).resolve().parents[3]
sys.path.append(str(project_folder))
src_folder = os.path.join(project_folder, 'src')
sys.path.append(str(src_folder))
from encoder import Encoder
from optimizer import Optimizer, Problem
from transpiler import Standalone
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, RobustScaler
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from copy import deepcopy
from collections import Counter

exp_folder = Path(__file__).resolve().parents[3]
data_path = os.path.join(exp_folder, 'data', 'paper_data', 'mnist_784', 'mnist_784.csv')

df = pd.read_csv(data_path)

training_features = [i for i in list(df) if i != 'class']
X = df[training_features]
y = df['class'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

# model = xgb.XGBClassifier(n_estimators=1000, objective="multi:softmax", num_class=10, random_state=42, n_jobs=-1)
model = xgb.XGBClassifier(objective="multi:softmax", num_class=10, random_state=42, n_jobs=-1)


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

plt.rcParams.update({'text.usetex' : True
                    , 'pgf.rcfonts': False
                    , 'text.latex.preamble':r"""\usepackage{iftex}
                                            \ifxetex
                                                \usepackage[libertine]{newtxmath}
                                                \usepackage[tt=false]{libertine}
                                                \setmonofont[StylisticSet=3]{inconsolata}
                                            \else
                                                \RequirePackage[tt=false, type1=true]{libertine}
                                            \fi"""   
                    })

pipeline.fit(X_train, y_train)

y_pred_train = pipeline.predict(X_train)

encoder = Encoder('multi-class')

encoder.fit(X_train.to_numpy(), y_pred_train, [])

X_train_encoded = encoder.transform_dataset(X_train, [i for i in range(X_train.shape[1])])

problem = Problem(X_train_encoded, y_pred_train, encoder.num_bins, 'multi-class', 1)
problem.set_information_value()

plt.hist(problem.iv_array, bins = 5)
plt.xlabel("Information Value", fontsize=16)
plt.ylabel("Feature Frequency", fontsize=16)
plt.show()

##################### Binary analysis. Idea: select features for each class as a binary problem. Then check the histogram of the features
##################### i.e., how many time a feature is shared across classes
#####################

solution_array = []
for i in range(10):

    model = xgb.XGBClassifier(objective="binary:logistic", n_jobs=-1, random_state=42)

    df_copy = deepcopy(df)
    df_copy['class'] = df_copy.apply(lambda x: 1 if i == x['class'] else 0, axis=1)

    X = df_copy[training_features]
    y = df_copy['class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)

    encoder = Encoder('classification')

    encoder.fit(X_train.to_numpy(), y_pred_train, [])

    X_train_encoded = encoder.transform_dataset(X_train, [i for i in range(X_train.shape[1])])

    problem = Problem(X_train_encoded, y_pred_train, encoder.num_bins, 'classification', 1)

    problem.set_costs()
    
    optimizer = Optimizer(problem, 1)

    optimizer.greedy_search()

    solution_array.extend(optimizer.greedy_solution)

counter = Counter(solution_array)

features = list(counter.keys())
appeareances = list(counter.values())

app_counter = Counter(appeareances)

number_of_shares = list(app_counter.keys())
freq = list(app_counter.values())

# plot

fig, ax = plt.subplots()
column_width = 3.3374
fig_width = column_width * 0.475
fig.set_size_inches(fig_width * 2, fig_width)

ax.bar(number_of_shares, freq, width=1, linewidth=0.7)
ax.set_xticks(np.arange(min(number_of_shares), max(number_of_shares) + 1, 1))
ax.set_ylabel("Number of Features", fontsize = 16)
ax.set_xlabel("Sharing Frequency", fontsize = 16)
plt.show()

# opt = Optimizer(problem, 1)

# acc = accuracy_score(y_test, y_pred)

# exp = Standalone(X_train, X_test, y_train, y_test, 'mnist_784', 'multi-class', False, pipeline=pipeline)

# d = exp.create_report(cat_mask=[], with_pred=True)

# export_path = os.path.join(exp_folder, 'experiments', 'output', exp.experiment_name + '_' + exp.model_name)
# d.to_csv(export_path + '_standalone.csv', index=False)



# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# ConfusionMatrixDisplay.from_predictions(y_test, exp.y_pred_trie, ax=axs[1])
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axs[0])

# plt.show()




