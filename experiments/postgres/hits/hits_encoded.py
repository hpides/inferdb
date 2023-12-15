import sys
import os
from pathlib import Path 
x_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(x_folder, 'src')
featurizer_folder = os.path.join(x_folder, 'src', 'featurizers')
sys.path.append(str(src_folder))
sys.path.append(str(featurizer_folder))
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
from transpiler import SQLmodel, PGML, MADLIB, InferDB
from pickle import load, dump
from sklearn.feature_selection import SelectFromModel
from optbinning import OptimalBinning

data_path = os.path.join(x_folder, 'data', 'hits', 'hits.csv')
write_path = os.path.join(x_folder, 'data', 'hits', 'hits_augmented.csv')
mask_path = os.path.join(x_folder, 'data', 'hits', 'cat_mask')

df = pd.read_csv(data_path)

cat_mask = list(df.columns.get_indexer(df.select_dtypes('object').columns))

training_shape = df.shape[1] - 1

new_features = np.empty((df.shape[0], len(cat_mask)))
new_names = []
to_extend = []
to_drop = []
for idx, i in enumerate(cat_mask):

    feature_name = list(df)[i]

    encoder = OptimalBinning(name=feature_name, dtype='categorical')
    new_features[:, idx] = encoder.fit_transform(df[feature_name].to_numpy(), df['class'].to_numpy(), metric='indices')

    new_names.extend([feature_name + '_encoded'])
    
    to_extend.extend([training_shape + idx])

    to_drop.extend([feature_name])


new_features_df = pd.DataFrame(new_features, columns=new_names)

new_features_df = new_features_df.astype(object)

df = pd.concat([df, new_features_df], axis=1)
df.drop(columns=to_drop, inplace=True)

df.to_csv(write_path, index=False)

duo = df.drop(columns='class')

cat_mask = [list(duo).index(i) for i in new_names]

with open(mask_path, 'wb') as d:
    dump(cat_mask, d)