from optbinning import OptimalBinning, ContinuousOptimalBinning, MulticlassOptimalBinning
import numpy as np
import pandas as pd

class Encoder:

    def __init__(self, type):
        """Encodes input data based on a target feature

        Args:
            type (str): 'regression', 'classification' for binary classification, 'multi-class' for multi label classification
        """        
        self.bin_ranges = {}
        self.cat_map = {}
        self.max_bins = None
        self.max_bin_size = None
        self.task = type
        self.num_bins = {}
        self.encoders = {}
        self.embeddings = {}

    def fit(self, x, y, mask):
        """Learns an encoding

        Args:
            x (ndarray): input data
            y (array): true values or predictions
            mask (list): list containing indices for categorical features in the input data
        """        
        if isinstance(x, pd.DataFrame):
            x_ = np.array(x)
        else:
            x_ = x.copy()
        
        for i in range(x_.shape[1]):
            if i not in mask:
                if self.task == 'classification':
                    optb = OptimalBinning(name=str(i), dtype="numerical", monotonic_trend='auto_heuristic', outlier_detector='range')
                    optb.fit(x_[:, i], y)
                    self.bin_ranges[i] = optb.splits
                    self.num_bins[i] = len(optb.splits)
                    self.encoders[i] = optb
                elif self.task == 'regression':
                    optb = ContinuousOptimalBinning(name=str(i), dtype="numerical", monotonic_trend='auto_heuristic', outlier_detector='range')
                    optb.fit(x_[:, i], y)
                    self.bin_ranges[i] = optb.splits
                    self.num_bins[i] = len(optb.splits)
                    self.encoders[i] = optb
                elif self.task == 'multi-class':
                    optb = MulticlassOptimalBinning(name=str(i), monotonic_trend='auto_heuristic', outlier_detector='range')
                    optb.fit(x_[:, i], y)
                    self.bin_ranges[i] = optb.splits
                    self.num_bins[i] = len(optb.splits)
                    self.encoders[i] = optb
            else:
                if self.task == 'classification':
                    optb = OptimalBinning(name=str(i), dtype="categorical", monotonic_trend='auto_heuristic', cat_cutoff=0.05)
                    try:
                        optb.fit(x_[:, i], y)
                        unique_values = np.unique(x_[:, i])
                        binarized_x = optb.transform(unique_values, metric='indices')
                        self.cat_map[i] = {}
                        for idj, j in enumerate(unique_values):
                            self.cat_map[i][j] = binarized_x[idj]
                        self.embeddings[i] = optb.splits
                        self.num_bins[i] = len(optb.splits)
                        self.encoders[i] = optb
                    except ValueError:
                        unique_values = np.unique(x_[:, i])
                        binarized_x = np.array([i for i in range(unique_values.size)])
                        self.cat_map[i] = {}
                        for idj, j in enumerate(unique_values):
                            self.cat_map[i][j] = binarized_x[idj]
                        self.embeddings[i] = unique_values
                        self.num_bins[i] = unique_values.size
                        self.encoders[i] = self.cat_map[i]
                elif self.task == 'regression':
                    optb = ContinuousOptimalBinning(name=str(i), dtype="categorical", monotonic_trend='auto_heuristic', cat_cutoff=0.05)
                    try:
                        optb.fit(x_[:, i], y)
                        unique_values = np.unique(x_[:, i])
                        binarized_x = optb.transform(unique_values, metric='indices')
                        self.cat_map[i] = {}
                        for idj, j in enumerate(unique_values):
                            self.cat_map[i][j] = binarized_x[idj]
                        self.embeddings[i] = optb.splits
                        self.num_bins[i] = len(optb.splits)
                        self.encoders[i] = optb
                    except ValueError:
                        unique_values = np.unique(x_[:, i])
                        binarized_x = np.array([i for i in range(unique_values.size)])
                        self.cat_map[i] = {}
                        for idj, j in enumerate(unique_values):
                            self.cat_map[i][j] = binarized_x[idj]
                        self.embeddings[i] = unique_values
                        self.num_bins[i] = unique_values.size
                        self.encoders[i] = self.cat_map[i]
        

    def transform_single(self, x, mask):
        """Transforms a single datapoint

        Args:
            x (1-d array): one dimensional array describing one single data point
            mask (list): indices of features to encode

        Returns:
            1-d array: encoded data point
        """        
        x_ = np.empty_like(x)

        for idx, i in enumerate(x):
            if mask[idx] in self.bin_ranges:
                encoder = self.bin_ranges[mask[idx]]
                key = len(encoder)
                try:
                    key = np.nonzero(encoder > i)[0].item(0)
                except IndexError:
                    key = encoder.size
                except (TypeError, KeyError):
                    key = len(encoder)
            else:
                try:
                    key = self.cat_map[mask[idx]][i]
                except (TypeError, KeyError):
                    key = len(self.cat_map[mask[idx]])
            
            x_[idx] = int(key)

        return x_

    def transform_dataset(self, x, mask):
        """Encodes an ndarray

        Args:
            x (ndarray): contains many datapoints to be encoded
            mask (list): list containing the indices of features to encode

        Returns:
            ndarray: encoded ndarray
        """        

        x_ = x.copy()
        if isinstance(x_, pd.DataFrame):
            x_ = np.array(x_)
        
        for i in mask:
            if isinstance(self.encoders[i], ContinuousOptimalBinning) or isinstance(self.encoders[i], OptimalBinning):
                x_[:, i] = self.encoders[i].transform(x_[:, i], metric='indices')
            else:
                for idj, j in enumerate(x_[:, i]):
                    x_[idj, i] = self.encoders[i][j]
        return x_

    

        





