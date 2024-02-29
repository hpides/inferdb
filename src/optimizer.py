import numpy as np
from math import log
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from itertools import permutations

class Problem:
    """This class defines a feature selection problem.
    """    

    def __init__(self, encoded_x, y, encoder_num_bins, task, sample_factor) -> None:
        """Constructor method for the problem class

        Args:
            encoded_x (ndarray): ndarray containing the encoded training data
            y (array): array containing predictions or true values
            encoder_num_bins (dict): dictionary containing the encoders of the features
            task (str): 'classification' for binary classification, 'regression', 'multi-class' for muti-label classification
            sample_factor (float): proportion of the data to use for feature selection
        """        
        self.encoded_x = encoded_x
        self.y = y
        self.iv_array = np.empty
        self.bin_array = np.empty
        self.costs_array = np.empty
        self.encoders = encoder_num_bins
        self.task = task
        self.sample_factor = sample_factor
        self.df= pd.DataFrame(self.encoded_x)
        self.target_variable_number = self.df.shape[1] + 1
        self.df[self.target_variable_number] = self.y
        if self.sample_factor < 1:
            sample_indices = np.random.choice(self.encoded_x.shape[0], round(self.encoded_x.shape[0] * self.sample_factor))
            self.sample = self.encoded_x[sample_indices]
            self.sample_y = self.y[sample_indices]
        else:
            self.sample = self.encoded_x
            self.sample_y = self.y
        self.labels = np.unique(self.y)
        if self.task in ('regression', 'classification'):
            self.global_events = self.y.sum()
            self.global_instances = self.y.shape[0]
            self.global_nonevents = self.global_instances - self.global_events
            self.global_mean = np.mean(self.y)

        elif self.task == 'multi-class':
            global_events = self.df.groupby(self.target_variable_number)[self.target_variable_number].agg(['count'])
            global_events['non-events'] = self.y.shape[0] - global_events['count']
            global_events.rename(columns={'count': 'events'}, inplace=True)
            self.global_events = global_events.to_dict()

        self.adjustement_factor = 0.5
    
    def set_number_of_bins(self):
        """Gets the number of bins in order for each feature
        """        

        self.bin_array = np.fromiter(self.encoders.values(), dtype=int)
    
    def weird_division(self, n, d):
        """avoids division by 0

        Args:
            n (int, float): dividend
            d (int, float): divisor

        Returns:
            int, float: division result
        """        
        return n / d if d else 0
    
    def set_information_value(self):
        """computes information value for all features in the encoded set
        """        

        iv_list = []
        for idx in range(self.encoded_x.shape[1]):
            if self.task == 'classification':
                agg_df = self.df.groupby(idx)[self.target_variable_number].agg(['count', 'sum'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['count']-x['sum']) + self.adjustement_factor) - log(self.global_nonevents)) - (log(x['sum'] + self.adjustement_factor) - log(self.global_events))) * ((x['count']-x['sum'])/self.global_nonevents - x['sum']/self.global_events), axis=1)
                iv_list.append(agg_df['woe'].sum())
            elif self.task == 'multi-class':
                class_agg = self.df.groupby(self.target_variable_number).size()

                local_instances = self.df.groupby(idx, as_index=False).size()
                index_classes = [idx]
                index_classes.extend([self.target_variable_number])
                class_instances = self.df.groupby(index_classes, as_index=False).size()

                # print(class_instances)

                agg_df = local_instances.set_index(idx).join(class_instances.set_index(idx), lsuffix='_other').fillna(0)
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['size_other'] - x['size']) + self.adjustement_factor) - log(sum(class_agg) - class_agg[x[self.target_variable_number]] + self.adjustement_factor)) - (log(x['size'] + self.adjustement_factor) - log(class_agg[x[self.target_variable_number]] + self.adjustement_factor))) * (self.weird_division((x['size_other'] - x['size']), sum(class_agg) - class_agg[x[self.target_variable_number]]) - self.weird_division(x['size'], class_agg[x[self.target_variable_number]])), axis=1)

                iv = agg_df['woe'].sum()

                iv_list.append(iv)
            elif self.task == 'regression':
                agg_df = self.df.groupby(idx)[self.target_variable_number].agg(['count', 'mean'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: abs(x['mean'] - self.global_mean) * (x['count'] / self.global_instances), axis=1)
                iv_list.append(agg_df['woe'].sum())

        self.iv_array = np.array(iv_list)
    
    def set_costs(self):
        """Sets the cost array
        """        

        self.set_number_of_bins()
        # self.set_information_value()
        self.set_information_value()

        self.costs_array = self.iv_array ### --> beneficial for IV
    
class Optimizer:
    """This class contains the methods to find a solution for a feature selection problem using heuristics
    """    
    
    def __init__(self, problem, sample_factor) -> None:
        """Constructor method for the optimizer

        Args:
            problem (Problem): Feature selection problem
            sample_factor (float): Proportion of the encoded data to use
        """        
        self.problem = problem
        self.sample_factor = sample_factor
        self.class_agg = self.problem.df.groupby(self.problem.target_variable_number).size()
    
    def weird_division(self, n, d):
        """avoids division by 0

        Args:
            n (int, float): dividend
            d (int, float): divisor

        Returns:
            int, float: division result
        """  
        return n / d if d else 0
    
    def compute_iv(self, index):
        """Computes information value for a candidate index

        Args:
            index (list): list containing the candidate features indices

        Returns:
            float: information value
        """        

        if self.problem.task == 'classification':
            agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'sum'])
            agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['count']-x['sum']) + self.problem.adjustement_factor) - log(self.problem.global_nonevents)) - (log(x['sum'] + self.problem.adjustement_factor) - log(self.problem.global_events))) * ((x['count']-x['sum'])/self.problem.global_nonevents - x['sum']/self.problem.global_events), axis=1)
            iv = agg_df['woe'].sum()
        elif self.problem.task == 'regression':
            agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'mean'])
            agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: abs(x['mean'] - self.problem.global_mean) * (x['count'] / self.problem.global_instances), axis=1)
            iv = agg_df['woe'].sum()
        elif self.problem.task == 'multi-class':
            
            local_instances = self.problem.df.groupby(index, as_index=False).size()
            index_classes = deepcopy(index)
            index_classes.extend([self.problem.target_variable_number])
            class_instances = self.problem.df.groupby(index_classes, as_index=False).size()

            agg_df = local_instances.set_index(index).join(class_instances.set_index(index), lsuffix='_other').fillna(0)
            agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['size_other'] - x['size']) + self.problem.adjustement_factor) - log(sum(self.class_agg) - self.class_agg[x[self.problem.target_variable_number]] + self.problem.adjustement_factor)) - (log(x['size'] + self.problem.adjustement_factor) - log(self.class_agg[x[self.problem.target_variable_number]] + self.problem.adjustement_factor))) * (self.weird_division((x['size_other'] - x['size']), sum(self.class_agg) - self.class_agg[x[self.problem.target_variable_number]]) - self.weird_division(x['size'], self.class_agg[x[self.problem.target_variable_number]])), axis=1)

            iv = agg_df['woe'].sum()
        
        return iv

    def greedy_search(self): 
        """Performs greedy search for feature selection problem
        """        

        sort_time = 0
        score_paths_time = 0

        set_of_candidates = np.argsort(self.problem.costs_array)[::-1] ## --> sort features by IV desc

        index = []
        information_value = 0
        if self.sample_factor < 1:
            sample_indices = np.random.choice(self.problem.df.shape[0], round(self.problem.df.shape[0] * self.sample_factor))
            self.df = self.problem.df.iloc[sample_indices]

        for idf, feature_index in tqdm(enumerate(set_of_candidates)):
            if self.problem.iv_array[feature_index] == 0:
                continue
            
            index.append(feature_index)
            if self.problem.task == 'classification':
                agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'sum'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['count']-x['sum']) + self.problem.adjustement_factor) - log(self.problem.global_nonevents)) - (log(x['sum'] + self.problem.adjustement_factor) - log(self.problem.global_events))) * ((x['count']-x['sum'])/self.problem.global_nonevents - x['sum']/self.problem.global_events), axis=1)
                iv = agg_df['woe'].sum()
            elif self.problem.task == 'multi-class':
                iv = self.compute_iv(index)
            elif self.problem.task == 'regression':
                agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'mean'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: abs(x['mean'] - self.problem.global_mean) * (x['count'] / self.problem.global_instances), axis=1)
                iv = agg_df['woe'].sum()

            if iv <= information_value * 1.002:
                index.remove(feature_index)
            elif(len(index) > 1 and iv > information_value * 1.002):
                information_value = iv
                index_bins = np.argsort(self.problem.bin_array[index]) ## -> If we sort by number of bins asc, the index storage decreases.
                index = [index[i] for i in index_bins]
            else:
                information_value = iv
            
            # print("Current Information Value: " + str(information_value))
            # print("Current Index Size: " + str(len(index)))
            # print("----------------")

        self.greedy_solution = index
        self.greedy_iv = information_value
        self.sort_time = sort_time
        self.score_paths_time = score_paths_time
        self.total_runtime = self.sort_time + self.score_paths_time

        self.problem.df.set_index(list(self.problem.df.columns[index]), inplace=True)
        self.problem.df.sort_index(inplace=True)
        compund_keys = self.problem.df.index.unique().values

        self.number_of_paths = compund_keys.shape[0]
        self.number_of_features = self.problem.df.shape[1]
        self.all_possible_paths = np.prod(self.problem.bin_array[index])
    
    def brute_force(self, permutations_length):

        set_of_candidates = [i for i in range(self.problem.encoded_x.shape[1])]
        self.problem.df.reset_index(inplace=True)
        results = []
        for length in range(1, permutations_length + 1):
            perm = permutations(set_of_candidates, length)
            for i in tqdm(perm):
                index = list(i)
                iv = self.compute_iv(index)
                results.append((index, iv))
        
        results_df = pd.DataFrame(results, columns=['Features', 'IV'])

        return results_df


    def sensitivity(self, sample_factor=1, index_size=None):
        """computes the filling degree of a candidate index until certain max path length (index size)

        Args:
            sample_factor (int, optional): If 1 uses all input data. If float uses a defined proportion of the data. Defaults to 1.
            index_size (int, optional): max length path of the index. Defaults to None.

        Returns:
            tuple(int, float, int, int, float): (index_size, sample_factor, compund_keys.shape[0], all_possible_paths, filling_degree)
        """        

        df = self.df.copy()
        
        set_of_candidates = np.argsort(self.problem.costs_array)[::-1] ## --> sort features by IV desc

        index = set_of_candidates[:index_size]
        df.set_index(list(df.columns[index]), inplace=True)
        df.sort_index(inplace=True)
        if sample_factor < 1:
            sample_indices = np.random.choice(df.shape[0], round(df.shape[0] * sample_factor)) 
            df = df.iloc[sample_indices]
        
        compund_keys = df.index.unique().values
        all_possible_paths = np.prod(self.problem.bin_array[index])
        filling_degree = compund_keys.shape[0] / all_possible_paths

        return (index_size, sample_factor, compund_keys.shape[0], all_possible_paths, filling_degree)




        





