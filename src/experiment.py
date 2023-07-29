import pandas as pd
from pickle import load, dump
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
import numpy as np



class Experiment:

    def __init__(self, name, path, task) -> None:
        self.name = name
        self.path = path
        self.task = task

    def prepare_dataset(self):

        df = pd.read_csv(self.path + '/' + self.name + '.csv')

        with open(self.path + '/cat_mask', 'rb') as d:
            self.cat_mask = load(d)
        
        target = 'class'
        training_features = [i for i in list(df) if i != target]
        self.datetime_features = [i for i in training_features if re.search("datetime", i)]
        for f in self.datetime_features:

            df[f] = pd.to_datetime(df[f])
            df['day'] = df.apply(lambda x: x[f].day_name(), axis=1)
            df['is_weekend'] = df.apply(lambda x: 1 if x['day'] in ('Saturday', 'Sunday') else 0, axis=1)
            df['hour'] = df[f].dt.hour
            df['month'] = df[f].dt.month

            def time_of_day(x):
                if x in range(6,12):
                    return 'Morning'
                elif x in range(12,16):
                    return 'Afternoon'
                elif x in range(16,22):
                    return 'Evening'
                else:
                    return 'Late_night'
            df['pickup_hour_of_day'] = df['hour'].apply(time_of_day)

            #### adjust cat_mask accordingly:
            
            self.cat_mask = [i if i < training_features.index(f) else i - 1 for i in self.cat_mask]
            df.drop(columns=[f], inplace=True)
            training_features.remove(f)
            training_features.extend(['day', 'is_weekend', 'hour', 'month', 'pickup_hour_of_day'])

            df = df.loc[:, training_features + ['class']]
            day_index = list(df).index('day')
            self.cat_mask.append(day_index)
            pickup_hour_index = list(df).index('pickup_hour_of_day')
            self.cat_mask.append(pickup_hour_index)
        
        self.feature_names = list(df)

        X = df[training_features].to_numpy()
        y = df[target].to_numpy()

        if self.task == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=42, test_size=0.5)

