from optbinning import OptimalBinning, ContinuousOptimalBinning
import numpy as np
import math

class Encoder:

    def __init__(self, bin_strategy, type):

        self.binning_strategy = bin_strategy
        self.bin_ranges = {}
        self.cat_map = {}
        self.max_bins = None
        self.max_bin_size = None
        self.task = type
        self.num_bins = {}
        self.encoders = {}
        self.embeddings = {}

    def fit(self, x, y, mask):
        if self.binning_strategy == 'optimal':
            for i in range(x.shape[1]):
                if i not in mask:
                    if self.task in ('classification'):
                        optb = OptimalBinning(name=str(i), dtype="numerical", monotonic_trend='auto_heuristic', outlier_detector='range')
                        optb.fit(x[:, i], y)
                        self.bin_ranges[i] = optb.splits
                        self.num_bins[i] = len(optb.splits)
                        self.encoders[i] = optb
                    else:
                        optb = ContinuousOptimalBinning(name=str(i), dtype="numerical", monotonic_trend='auto_heuristic', outlier_detector='range')
                        optb.fit(x[:, i], y)
                        self.bin_ranges[i] = optb.splits
                        self.num_bins[i] = len(optb.splits)
                        self.encoders[i] = optb
                else:
                    if self.task in ('classification'):
                        optb = OptimalBinning(name=str(i), dtype="categorical", monotonic_trend='auto_heuristic', cat_cutoff=0.05)
                        try:
                            optb.fit(x[:, i], y)
                            unique_values = np.unique(x[:, i])
                            binarized_x = optb.transform(unique_values, metric='indices')
                            self.cat_map[i] = {}
                            for idj, j in enumerate(unique_values):
                                self.cat_map[i][j] = binarized_x[idj]
                            self.embeddings[i] = optb.splits
                            self.num_bins[i] = len(optb.splits)
                            self.encoders[i] = optb
                        except ValueError:
                            unique_values = np.unique(x[:, i])
                            binarized_x = np.array([i for i in range(unique_values.size)])
                            self.cat_map[i] = {}
                            for idj, j in enumerate(unique_values):
                                self.cat_map[i][j] = binarized_x[idj]
                            self.embeddings[i] = unique_values
                            self.num_bins[i] = unique_values.size
                            self.encoders[i] = self.cat_map[i]
                    else:
                        optb = ContinuousOptimalBinning(name=str(i), dtype="categorical", monotonic_trend='auto_heuristic', cat_cutoff=0.05)
                        try:
                            optb.fit(x[:, i], y)
                            unique_values = np.unique(x[:, i])
                            binarized_x = optb.transform(unique_values, metric='indices')
                            self.cat_map[i] = {}
                            for idj, j in enumerate(unique_values):
                                self.cat_map[i][j] = binarized_x[idj]
                            self.embeddings[i] = optb.splits
                            self.num_bins[i] = len(optb.splits)
                            self.encoders[i] = optb
                        except ValueError:
                            unique_values = np.unique(x[:, i])
                            binarized_x = np.array([i for i in range(unique_values.size)])
                            self.cat_map[i] = {}
                            for idj, j in enumerate(unique_values):
                                self.cat_map[i][j] = binarized_x[idj]
                            self.embeddings[i] = unique_values
                            self.num_bins[i] = unique_values.size
                            self.encoders[i] = self.cat_map[i]
        

    def transform_single(self, x, mask):
        x_ = x.copy()

        for idx, i in enumerate(x_):
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

        x_ = x.copy()
        
        for i in mask:
            if isinstance(self.encoders[i], ContinuousOptimalBinning) or isinstance(self.encoders[i], OptimalBinning):
                x_[:, i] = self.encoders[i].transform(x[:, i], metric='indices')
            else:
                for idj, j in enumerate(x_[:, i]):
                    x_[idj, i] = self.encoders[i][j]
        return x_

    

        





