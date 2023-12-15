import pandas as pd
import openml
import os
import pickle
from sklearn import preprocessing
from sklearn.datasets import load_digits
from pathlib import Path

datasets = [
            ('adult', 1590)
            , ('mushroom', 24)
            , ('credit-g', 31)
            , ('creditcard', 1597)
            , ('higgs', 23512)
            , ('madelon', 1485)
            , ('Speech', 40910)
            , ('kr-vs-kp', 3)
            , ('bank-marketing', 1461)
            , ('nasa', 42821)
            , ('airlines', 1169)
            , ('clicks', 1219)
            , ('digits', 554)
            , ('iris', 61)
            ]


for d in datasets:

    dataset = openml.datasets.get_dataset(d[1])
    

    df, y, cat_mask, names = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)

    df = pd.concat([df, y], axis=1)
    df.columns = [*df.columns[:-1], 'class']
    
    # Parent Directory path
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    directory = dataset.name
    
    # Path
    path = os.path.join(parent_dir, directory)
    Path(path).mkdir(parents=True, exist_ok=True)
        

    cat_mask = [idx for idx, i in enumerate(cat_mask) if i]
    
    with open('data/' + dataset.name + '/cat_mask', 'wb') as d:
        pickle.dump(cat_mask, d)

    df.to_csv('data/' + dataset.name + '/' + dataset.name + '.csv', index=False)