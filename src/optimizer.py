import numpy as np
from math import log
import pandas as pd
import multiprocessing as mp
import yaml
import os
from tqdm import tqdm
from itertools import permutations



class Problem:

    def __init__(self, encoded_x, y, encoder_num_bins, task, sample_factor) -> None:
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
        self.global_events = self.y.sum()
        self.global_instances = self.y.shape[0]
        self.global_nonevents = self.global_instances - self.global_events
        self.global_mean = np.mean(self.y)
        self.adjustement_factor = 1
    
    def set_number_of_bins(self):

        self.bin_array = np.fromiter(self.encoders.values(), dtype=int)
    
    def set_information_value(self):

        iv_list = []
        for idx in range(self.encoded_x.shape[1]):
            if self.task == 'classification':
                agg_df = self.df.groupby(idx)[self.target_variable_number].agg(['count', 'sum'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['count']-x['sum']) + self.adjustement_factor) - log(self.global_nonevents)) - (log(x['sum'] + self.adjustement_factor) - log(self.global_events))) * ((x['count']-x['sum'])/self.global_nonevents - x['sum']/self.global_events), axis=1)
                iv_list.append(agg_df['woe'].sum())
            elif self.task == 'regression':
                agg_df = self.df.groupby(idx)[self.target_variable_number].agg(['count', 'mean'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: abs(x['mean'] - self.global_mean) * (x['count'] / self.global_instances), axis=1)
                iv_list.append(agg_df['woe'].sum())

        self.iv_array = np.array(iv_list)
    
    def set_costs(self):

        self.set_number_of_bins()
        # self.set_information_value()
        self.set_information_value()

        self.costs_array = self.iv_array ### --> beneficial for IV
    
class Optimizer:

    
    def __init__(self, problem, sample_factor) -> None:
        self.problem = problem
        self.sample_factor = sample_factor
    
    def compute_iv(self, index):

        if self.problem.task == 'classification':
            agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'sum'])
            agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['count']-x['sum']) + self.problem.adjustement_factor) - log(self.problem.global_nonevents)) - (log(x['sum'] + self.problem.adjustement_factor) - log(self.problem.global_events))) * ((x['count']-x['sum'])/self.problem.global_nonevents - x['sum']/self.problem.global_events), axis=1)
            iv = agg_df['woe'].sum()
        elif self.problem.task == 'regression':
            agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'mean'])
            agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: abs(x['mean'] - self.problem.global_mean) * (x['count'] / self.problem.global_instances), axis=1)
            iv = agg_df['woe'].sum()
        
        return iv

    def greedy_search(self): 

        sort_time = 0
        score_paths_time = 0

        set_of_candidates = np.argsort(self.problem.costs_array)[::-1] ## --> sort features by IV desc

        index = []
        information_value = 0
        if self.sample_factor < 1:
            sample_indices = np.random.choice(self.problem.df.shape[0], round(self.problem.df.shape[0] * self.sample_factor))
            self.df = self.problem.df.iloc[sample_indices]

        for idf, feature_index in tqdm(enumerate(set_of_candidates)):
            if self.problem.iv_array[feature_index] == 0 or self.problem.bin_array[feature_index] <= 1:
                continue
            
            index.append(feature_index)
            if self.problem.task == 'classification':
                agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'sum'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: ((log((x['count']-x['sum']) + self.problem.adjustement_factor) - log(self.problem.global_nonevents)) - (log(x['sum'] + self.problem.adjustement_factor) - log(self.problem.global_events))) * ((x['count']-x['sum'])/self.problem.global_nonevents - x['sum']/self.problem.global_events), axis=1)
                iv = agg_df['woe'].sum()
            elif self.problem.task == 'regression':
                agg_df = self.problem.df.groupby(index)[self.problem.target_variable_number].agg(['count', 'mean'])
                agg_df.loc[:, 'woe'] = agg_df.apply(lambda x: abs(x['mean'] - self.problem.global_mean) * (x['count'] / self.problem.global_instances), axis=1)
                iv = agg_df['woe'].sum()

            if iv <= information_value:
                index.remove(feature_index)
            elif(len(index) > 1 and iv > information_value):
                information_value = iv
                index_bins = np.argsort(self.problem.bin_array[index]) ## -> If we sort by number of bins asc, the index storage decreases.
                index = [index[i] for i in index_bins]
            else:
                information_value = iv

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




        





