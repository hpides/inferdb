import pandas as pd
from pandas.api.types import is_string_dtype
from optbinning import ContinuousOptimalBinning2D
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from copy import copy, deepcopy
from numpy import percentile
from sklearn.linear_model import LinearRegression
from geographiclib.geodesic import Geodesic
from geographiclib.constants import Constants
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

class PM_Featurizer(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        super().__init__()
    
    def learn_clustering(self, X):

        coords = X[['latitude', 'longitude']].values

        sample_ind = np.random.permutation(len(coords))[:500000]
        kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=10000).fit(coords[sample_ind])

        return kmeans

    def fit(self, X, y):

        X_ = deepcopy(X)

        self.kmeans = self.learn_clustering(X_)

        X_['cluster'] = self.kmeans.predict(X_[['latitude', 'longitude']].values)

        return self
        
    def transform(self, X):

        X_ = deepcopy(X)
        if isinstance(X_, pd.Series):
            X_ = X_.to_frame().transpose()

        X_['cluster'] = self.kmeans.predict(X_[['latitude', 'longitude']].values)
        
        self.feature_names = list(X_)

        return X_


