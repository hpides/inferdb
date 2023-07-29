from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.inspection import permutation_importance
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
import numpy as np
from sklearn.impute import SimpleImputer
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier, MLPRegressor
import multiprocessing as mp
import yaml
import os


class Trainer:

    def __init__(self, clf, balanced):
        
        self.clf = clf
        self.balanced = balanced
        self.preprocess_time = 0
        self.training_time = 0
        self.postprocessing_time = 0

        numerical_transformer = Pipeline(
            steps=
                    [
                        ('num_imputer', SimpleImputer())
                        # , ('generator', PolynomialFeatures(interaction_only=True))
                        # , ('feature_selector', VarianceThreshold())
                        , ('scaler', StandardScaler())
                    ]
            )
        categorical_transformer = Pipeline(
            steps=
                    [
                        ('cat_imputer', SimpleImputer(strategy='most_frequent'))
                        , ('encoder', OneHotEncoder(handle_unknown='ignore'))
                    ]
            )


        column_transformer = ColumnTransformer(
                                                transformers=[
                                                                ('num', numerical_transformer, [])
                                                                , ('cat', categorical_transformer, [])
                                                            ]
                                                , remainder='passthrough'
                                                # , n_jobs=-1
                                            )
        self.pipeline = Pipeline(
                                    steps=
                                            [   
                                                ('column_transformer', column_transformer)
                                                , ('clf', self.clf)
                                            ]
                                )

    
    def preprocess(self, x, y, categorical_mask):

        numerical_mask = [i for i in range(x.shape[1]) if i not in categorical_mask]

        numerical_imputer = self.pipeline.named_steps.column_transformer.transformers[0]
        categorical_imputer = self.pipeline.named_steps.column_transformer.transformers[1]
        self.pipeline.named_steps.column_transformer.transformers[0] = (numerical_imputer[0], numerical_imputer[1], numerical_mask)
        self.pipeline.named_steps.column_transformer.transformers[1] = (categorical_imputer[0], categorical_imputer[1], categorical_mask)

        x_train = x.copy()
        
        if categorical_mask and numerical_mask:
            
            start = time.time()
            numerical_imputer = SimpleImputer()
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            x_train[:, numerical_mask] = numerical_imputer.fit_transform(x_train[:, numerical_mask])
            x_train[:, categorical_mask] = categorical_imputer.fit_transform(x_train[:, categorical_mask])

            self.preprocess_time += time.time() - start

            if self.balanced:
                start = time.time()
                sm = SMOTENC(categorical_mask, 
                                random_state=42
                            )
                x_resampled, y_resampled = sm.fit_resample(x_train, y)
                self.preprocess_time += time.time() - start
            else:
                x_resampled = x_train
                y_resampled = y

        elif not categorical_mask:
            
            start = time.time()
            numerical_imputer = SimpleImputer()
            
            self.preprocess_time += time.time() - start

            if self.balanced:
                start = time.time()
                sm = SMOTE(random_state=42
                            )
                x_resampled, y_resampled = sm.fit_resample(x_train, y)
                self.preprocess_time += time.time() - start

            else:
                x_resampled = x_train
                y_resampled = y
          
        elif not numerical_mask:
            start = time.time()
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            x_resampled = x_train
            y_resampled = y
            self.preprocess_time += time.time() - start

        t = time.time()
        self.pipeline[:-1].fit(x_resampled)
        self.preprocess_time += (time.time() - t)
        
        x_train_preprocessed = self.pipeline[:-1].transform(x_resampled)
        
        return x_resampled, y_resampled, x_train_preprocessed

    
    def fit(self, x, y):

        if isinstance(self.pipeline.named_steps['clf'], (MLPClassifier, MLPRegressor)):

            input_size = x.shape[1]
            if self.pipeline.named_steps['clf'].__class__.__name__ == 'MLPClassifier':
                output_size = np.unique(y).size
            else:
                output_size = 1

            neurons = min(100, round((2/3) * (input_size + output_size)))
            params ={
                        'clf__hidden_layer_sizes':(neurons,)
                    }
            self.pipeline.set_params(**params)

        tr1 = time.time()
        self.pipeline[-1].fit(x, y)
        self.training_time += (time.time() - tr1) 
    
    def infer(self, x):
        
        start = time.time()
        self.pipeline[:-1].transform(x)
        preprocess_runtime = (time.time() - start) * 1e6
        start = time.time()
        self.pipeline.predict(x)
        inference_runtime = (time.time() - start) * 1e6
        prediction = self.pipeline.predict(x)
        
        return (preprocess_runtime, inference_runtime, prediction)
    
    def parallel_infer(self, X):

        with mp.Pool(processes=self.processes) as pool:

            results = pool.map(self.infer, X)
        
        y_pred = np.array([c for a, b, c in results])
        preprocess_runtimes = np.array([a for a, b, c in results])
        inference_times = np.array([b for a, b, c in results])

        return preprocess_runtimes, inference_times, y_pred








