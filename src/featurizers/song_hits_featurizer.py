import pandas as pd
from optbinning import OptimalBinning
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy


class Hits_Featurizer(BaseEstimator, TransformerMixin):

    def __init__(self, cat_mask_names) -> None:
        super().__init__()

        self.cat_mask_names = cat_mask_names
    
    def learn_encoding(self, X, y, feature_name):

        encoder = OptimalBinning(name=feature_name, dtype='categorical')

        encoder.fit(X[feature_name].to_numpy(), y)

        return encoder

    def get_feature_names(self):

        return self.feature_names
    
    def fit(self, X, y):

        X_ = deepcopy(X)

        self.encoder_dict = {}

        for c in self.cat_mask_names:

            self.encoder_dict[c] = self.learn_encoding(X_, y, c)
        
        return self
    
    def transform(self, X):

        X_ = deepcopy(X)
        if isinstance(X_, pd.Series):
            X_ = X_.to_frame().transpose()

        if self.cat_mask_names:
            for c in self.cat_mask_names:
                X_[c + '_encoded'] = self.encoder_dict[c].transform(X_[c].to_numpy(), metric='indices')
        
        self.feature_names = list(X_)
        
        return X_