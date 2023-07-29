import os
import sys
from pathlib import Path
project_folder = Path(__file__).resolve().parents[1]
src_folder = os.path.join(project_folder,'src')
sys.path.append(str(src_folder))
from experiment import Experiment
from optimizer import Problem, Optimizer
from trainer import Trainer
from encoder import Encoder
import pandas as pd
import numpy as np
from database_connect import connect
from psycopg2 import sql
from psycopg2.extensions import register_adapter, AsIs, register_type, new_type, DECIMAL
import re
from itertools import product
import time
from pathlib import Path 
from sklearn.metrics import accuracy_score, mean_squared_log_error, recall_score, precision_score, f1_score
from joblib import dump
from copy import deepcopy
from os.path import getsize
from inference_trie import Trie
from scipy import sparse
from io import StringIO

class ExperimentHandler:

    def __init__(self, experiment_name, model, task, is_balanced, sample_factor, populate_paths) -> None:
        self.experiment_name = self.text_normalizer(experiment_name).lower()
        self.model = model
        self.task = task
        self.is_balanced = is_balanced
        self.sample_factor = sample_factor
        self.populate_paths = populate_paths
        self.model_name = (model.__class__.__name__).lower()
        self.input_name = self.experiment_name + '_' + self.model_name

    def create_postgres_connection(self):
        conn = connect()
        cur = conn.cursor()

        return conn, cur

    def addapt_numpy_float64(numpy_float64):
        return AsIs(numpy_float64)
    
    def addapt_numpy_int64(numpy_int64):
        return AsIs(numpy_int64)
    
    def adapt_psycopg2(self):
        register_adapter(np.float64, self.addapt_numpy_float64)
        register_adapter(np.int64, self.addapt_numpy_int64)

    def text_normalizer(self, text):

        rep = {"-": "_", ".": "_", "?":"", "/":"", "(":"_", ")":"_", "&":"_"} # define desired replacements here

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pattern = re.compile("|".join(rep.keys()))

        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    def fit(self):

        encoder = Encoder('optimal', self.task)
        from pathlib import Path
        project_folder = Path(__file__).resolve().parents[1]
        data_folder = os.path.join(project_folder, 'data', self.experiment_name)
        self.path = data_folder
        experiment = Experiment(self.experiment_name, self.path, self.task)
        trainer = Trainer(self.model, self.is_balanced)
        experiment.prepare_dataset()
        
        st = time.time()
        x_train_resampled, y_train_resampled, x_train_preprocessed = trainer.preprocess(experiment.X_train, experiment.y_train, experiment.cat_mask)
        trainer.fit(x_train_preprocessed, y_train_resampled)
        self.training_time = time.time() - st
        st = time.time()
        trainer.pipeline[:-1].transform(x_train_resampled)
        self.training_preprocessing_runtime = time.time() - st
        print('Training Complete')
        st = time.time()
        y_train_pred = trainer.pipeline.predict(x_train_resampled)
        encoder.fit(x_train_resampled, y_train_pred, experiment.cat_mask)
        encoded_training_set = encoder.transform_dataset(x_train_resampled, [i for i in range(x_train_resampled.shape[1])])
        self.encoding_time = time.time() - st
        st = time.time()
        print('Encoding Complete')
        my_problem = Problem(encoded_training_set, y_train_resampled, encoder.num_bins, self.task, self.sample_factor)
        my_problem.set_costs()
        print('Costs set')
        my_optimizer = Optimizer(my_problem, self.sample_factor)
        my_optimizer.greedy_search()
        self.solution_time = time.time() - st
        print('Greedy Search Complete, index:' + str(my_optimizer.greedy_solution))
        st = time.time() 

        self.pipeline = trainer.pipeline
        self.encoder = encoder
        self.solution = my_optimizer.greedy_solution
        self.experiment_feature_names = experiment.feature_names
        self.experiment_cat_mask = experiment.cat_mask
        self.experiment_training_instances = x_train_resampled.shape[0]
        self.experiment_input_dimensions = experiment.X_train.shape[1]
        self.experiment_test_instances = experiment.X_test.shape[0]
        self.experiment_transformed_input_size = x_train_preprocessed.shape[1]
        # self.encoded_training_set = encoded_training_set
        self.y_train_pred = y_train_pred
        self.max_path_length = len(self.solution)
        self.preprocessing_output_shape = x_train_preprocessed.shape

        print('A solution was found')

        # new_cat_mask = [idf for idf, i in enumerate(my_optimizer.greedy_solution) if i in experiment.cat_mask]
        # x_train_resampled, y_train_resampled, x_train_preprocessed = trainer.preprocess(experiment.X_train[:, my_optimizer.greedy_solution], experiment.y_train, new_cat_mask)
        # trainer.fit(x_train_preprocessed, y_train_resampled)

        # print(trainer.pipeline[:-1].transform(experiment.X_test[:5]))

    def get_kv_tuples(self, purpose):    
        if self.populate_paths:
            bin_set = []
            for feature in self.solution:
                bin_set.append([i for i in range(self.encoder.num_bins[feature]+1)])
            combinations_set = product(*bin_set)

        experiment = Experiment(self.experiment_name, self.path, self.task)
        trainer = Trainer(self.model, self.is_balanced)
        experiment.prepare_dataset()
        
        
        x_train_resampled, y_train_resampled, x_train_preprocessed = trainer.preprocess(experiment.X_train, experiment.y_train, experiment.cat_mask)
        encoded_training_set = self.encoder.transform_dataset(x_train_resampled, [i for i in range(x_train_resampled.shape[1])])
        encoded_training_set = encoded_training_set[:, self.solution]
        training_tuples = [tuple(x) for x in encoded_training_set]

        if self.populate_paths:
            encoded_pipeline = deepcopy(self.pipeline)
            encoded_pipeline[-1].fit(encoded_training_set, self.y_train_pred)
            diff_set = set(combinations_set) - set(training_tuples)

            y_pred_diff = encoded_pipeline[-1].predict(np.array(list(diff_set)))

        if purpose != 'standalone':
            
            training_output = []
            for idt, t in enumerate(encoded_training_set):
                key = ''
                for idf, feature in enumerate(t):
                    if idf < len(t) - 1:
                        key = key + str(int(feature)) + '.'
                    else:
                        key = key + str(int(feature))
                    # key = key + str(feature)
                training_output.append((key, self.y_train_pred[idt]))

            if self.populate_paths:      
                for iddiff, t in enumerate(diff_set):
                    key = ''
                    for idf, feature in enumerate(t):
                        if idf < len(t) - 1:
                            key = key + str(int(feature)) + '.'
                        else:
                            key = key + str(int(feature))
                        # key = key + str(feature)
                    training_output.append((key, y_pred_diff[iddiff]))
            
            k_v_frame = pd.DataFrame(training_output, columns=['key', 'value'])
            k_v_aggregates = k_v_frame.groupby(['key']).mean()
            k_v_aggregates.reset_index(drop=False, inplace=True)
            tuples = k_v_aggregates.to_numpy()
        else:

            if self.populate_paths:
                encoded_training_set = np.append(encoded_training_set, np.array(list(diff_set)), axis=0)
                y_pred_all = np.append(self.y_train_pred, y_pred_diff)

            encoded_df= pd.DataFrame(encoded_training_set)
            target_variable_number = encoded_df.shape[1]
            if self.populate_paths:
                encoded_df[target_variable_number] = y_pred_all
            else:
                encoded_df[target_variable_number] = self.y_train_pred
            agg_df = encoded_df.groupby([i for i in range(len(self.solution))], as_index=False)[target_variable_number].mean()
            tuples = agg_df.to_numpy()
        
        return tuples

    def create_kv_table(self, table_name):
        conn, cur = self.create_postgres_connection()
        ## Creates a table where key, predictions will be stored
        cur.execute(sql.SQL('DROP TABLE IF EXISTS pgml.{table_name} CASCADE; CREATE TABLE pgml.{table_name} (key TEXT NOT NULL, value NUMERIC NOT NULL)').format(table_name=sql.Identifier(table_name)))
        conn.commit()

    def insert_tuples_in_kv_table(self, table_name):
        conn, cur = self.create_postgres_connection()
        ## Inserts keys and predictions in an already created table
        tuples = self.get_kv_tuples('db')
        # sql_insert_query = sql.SQL("INSERT INTO pgml.{} (key, value) VALUES (%s,%s)").format(sql.Identifier(table_name))
        # cur.executemany(sql_insert_query, tuples)

        buffer = StringIO()
        df = pd.DataFrame(tuples)
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY pgml.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name)), buffer)
        conn.commit()

        conn.commit()
        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")

    def create_index(self, index_name, table_name):
        conn, cur = self.create_postgres_connection()
        ## Creates an index for the keys using spgist
        cur.execute(sql.SQL('DROP INDEX IF EXISTS {index_name} CASCADE;CREATE INDEX {index_name} ON pgml.{table_name} using spgist(key)').format(index_name = sql.Identifier(index_name), table_name = sql.Identifier(table_name)))
        conn.commit()
    
    def create_scoring_function_kv(self):
        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS pgml.{input_name}_score_kv;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_score_kv(OUT row_id INTEGER, OUT prediction NUMERIC) RETURNS SETOF record AS 
                                        $func$
                                        select k1.ROW_ID, case when k2.value is not null then k2.value else pgml.prefix_search(k1.key, '{input_name}_kv') end as prediction
                                        from pgml.{name}_translated k1
                                        left join pgml.{input_name}_kv k2
                                        on k1.key = k2.key
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    """).format(input_name = sql.SQL(self.input_name),
                                                name = sql.SQL(self.experiment_name))
        
        ## Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score_kv')

        if self.task == 'classification':

            self.create_acc_measure_function('score_kv', 'acc')
            self.create_acc_measure_function('score_kv', 'f1')
            self.create_acc_measure_function('score_kv', 'precision')
            self.create_acc_measure_function('score_kv', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score_kv', 'rmsle')

    def create_aux_functions(self):

        conn, cur = self.create_postgres_connection()

        parent = Path(__file__).resolve().parents[1]
        
        linear_search_path = os.path.join(parent, 'sql_scripts/linear_search_text.sql')
        binary_search_path = os.path.join(parent, 'sql_scripts/binary_search_numeric.sql')
        # crazy_exp_path = os.path.join(parent, 'sql_scripts/crazy_exp.sql')
        # measure_runtime = os.path.join(parent, 'sql_scripts/measure_runtime.sql')
        prefix_search = os.path.join(parent, 'sql_scripts/prefix_search.sql')

        bst = open(linear_search_path, 'r')
        bsn = open(binary_search_path, 'r')
        # exp = open(crazy_exp_path, 'r')
        # run = open(measure_runtime, 'r')
        ps = open(prefix_search, 'r')

        bstr = bst.read()
        bsnr = bsn.read()
        # expr = exp.read()
        # runr = run.read()
        psr = ps.read()

        sql_bst = sql.SQL(bstr)
        sql_bsn = sql.SQL(bsnr)
        # sql_exp = sql.SQL(expr)
        # sql_run = sql.SQL(runr)
        sql_ps = sql.SQL(psr)

        cur.execute(sql_bst)
        conn.commit()
        cur.execute(sql_bsn)
        conn.commit()
        # cur.execute(sql_exp)
        # conn.commit()
        # cur.execute(sql_run)
        # conn.commit()
        cur.execute(sql_ps)
        conn.commit()
    
    def create_preproccesing_function(self, limit_factor):

        conn, cur = self.create_postgres_connection()

        ### Creates a function to transform raw values into keys

        feature_list = [(i, self.text_normalizer(self.experiment_feature_names[i])) for i in self.solution]

        select_statement = """ SELECT ROW_ID, """
        for idf, feature in enumerate(feature_list):
            feature_index = feature[0]
            feature_name = feature[1]
            if feature_index not in self.experiment_cat_mask:
                ranges_list = list(self.encoder.bin_ranges[feature_index])
                for idr, range in enumerate(ranges_list):
                    if idf < len(feature_list) - 1:
                        if idr == 0:
                            select_statement += 'CASE WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """.'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' AND ' + feature_name + ' > ' + str(ranges_list[idr-1]) + ' THEN ' + """'""" + str(idr) + """.'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' AND ' + feature_name + ' > ' + str(ranges_list[idr-1]) + ' THEN ' + """'""" + str(idr) + """.' ELSE """ + """'""" + str(idr + 1) + """.' END || """
                    else:
                        if idr == 0:
                            select_statement += 'CASE WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' AND ' + feature_name + ' > ' + str(ranges_list[idr-1]) + ' THEN ' + """'""" + str(idr) + """'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' AND ' + feature_name + ' > ' + str(ranges_list[idr-1]) + ' THEN ' + """'""" + str(idr) + """' ELSE """ + """'""" + str(idr + 1) + """' END """
            else:
                ranges_list = self.encoder.embeddings[feature_index]
                embeddings = [[str(j) for j in i] for i in ranges_list]
                for idr, range in enumerate(embeddings):
                    if idf < len(feature_list) - 1:
                        if idr == 0:
                            select_statement += 'CASE WHEN ' + feature_name + '= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + '= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + '= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.' ELSE """ + """'""" + str(idr + 1) + """.' END || """ 
                    else:
                        if idr == 0:
                            select_statement += 'CASE WHEN ' + feature_name + '= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + '= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + '= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """' ELSE """ + """'""" + str(idr + 1) + """' END """ 


        function_statement = sql.SQL(""" DROP FUNCTION IF EXISTS pgml.{input_name}_translate CASCADE ;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_translate(OUT ROW_ID INTEGER, OUT key TEXT) returns SETOF record AS 
                                        $func$
                                        
                                        {select_statement} 
                                        FROM pgml.{experiment_name}_test
                                        ORDER BY ROW_ID ASC LIMIT {limit_factor};

                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    """).format(input_name = sql.SQL(self.input_name)
                                                , experiment_name = sql.SQL(self.experiment_name)
                                                , select_statement = sql.SQL(select_statement)
                                                , limit_factor = sql.SQL(str(limit_factor))
                                                )

        # print(function_statement)
        ## Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('translate')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_translated;
                                        CREATE MATERIALIZED VIEW pgml.{name}_translated AS 
                                        
                                        SELECT * FROM  pgml.{input_name}_translate();
                                        
                                    """).format(name=sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , limit_factor = sql.SQL(str(limit_factor))
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

    def create_test_table(self):

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.text_normalizer(self.experiment_name + '_test').lower()
        initial_statement = 'DROP TABLE IF EXISTS pgml.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE pgml.' + table_name + '(ROW_ID INTEGER,'
        for idf, feature in enumerate(self.experiment_feature_names):
            if idf not in self.experiment_cat_mask:
                feature_definition = self.text_normalizer(feature) + ' REAL'
            else:
                feature_definition = self.text_normalizer(feature) + ' TEXT'

            if idf < len(self.experiment_feature_names) - 1:
                initial_statement = initial_statement + feature_definition + ', '
            else:
                initial_statement = initial_statement + feature_definition + ')'

        ### Create table
        cur.execute(initial_statement)
        conn.commit()
    
    def create_train_table(self):

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.text_normalizer(self.experiment_name + '_train').lower()
        initial_statement = 'DROP TABLE IF EXISTS pgml.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE pgml.' + table_name + '('
        for idf in range(self.preprocessing_output_shape[1]):
            
            initial_statement += 'f_' + str(idf) + ' REAL, '
            
        initial_statement += 'target REAL)'

        ### Create table
        cur.execute(initial_statement)
        conn.commit()

    def insert_test_tuples(self):

        conn, cur = self.create_postgres_connection()
        ### Inserts tuples into test table
        table_name = self.text_normalizer(self.experiment_name + '_test').lower()
        ids = (np.arange(self.experiment_test_instances, dtype=np.intc)).reshape((self.experiment_test_instances, 1))
        experiment = Experiment(self.experiment_name, self.path, self.task)
        experiment.prepare_dataset()
        test_input = np.hstack((experiment.X_test, experiment.y_test.reshape((experiment.y_test.shape[0], 1))))
        test_input = np.hstack((ids, test_input))
        ### Insert test data
        
        insert_statement = 'INSERT INTO pgml.{} VALUES ('
        insert_statement += "%s" + ','
        for idf, feature in enumerate(self.experiment_feature_names):

            if idf < len(self.experiment_feature_names) - 1:
                insert_statement += "%s" + ','
            else:
                insert_statement += "%s" + ')'

        # sql_insert_query = sql.SQL(insert_statement).format(sql.Identifier(table_name))
        # cur.executemany(sql_insert_query, test_input)

        buffer = StringIO()
        df = pd.DataFrame(test_input)
        df[0] = df[0].astype('int')
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY pgml.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name)), buffer)
       
        conn.commit()
        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")
    
    def insert_train_tuples(self):

        conn, cur = self.create_postgres_connection()
        ### Inserts tuples into test table
        table_name = self.text_normalizer(self.experiment_name + '_train').lower()
        # ids = (np.arange(self.experiment_training_instances, dtype=np.intc)).reshape((self.experiment_training_instances, 1))
        experiment = Experiment(self.experiment_name, self.path, self.task)
        experiment.prepare_dataset()
        trainer = Trainer(self.model, self.is_balanced)
        experiment.prepare_dataset()
        
        x_train_resampled, y_train_resampled, x_train_preprocessed = trainer.preprocess(experiment.X_train, experiment.y_train, experiment.cat_mask)
        
        try:
            train_input = np.hstack((x_train_preprocessed, y_train_resampled.reshape((y_train_resampled.shape[0], 1))))
        except ValueError:
            train_input = sparse.hstack((x_train_preprocessed, y_train_resampled.reshape((y_train_resampled.shape[0], 1))))
            
            train_input = np.asarray(train_input.todense())
        
        insert_statement = 'INSERT INTO pgml.{} VALUES ('
        for idf in range(self.preprocessing_output_shape[1]):

            insert_statement += "%s" + ','
        
        #### Extra value for the target
        insert_statement += "%s" + ')'
            
        # sql_insert_query = sql.SQL(insert_statement).format(sql.Identifier(table_name))
        # cur.executemany(sql_insert_query, train_input)
        # conn.commit()

        buffer = StringIO()
        df = pd.DataFrame(train_input)
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY pgml.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name)), buffer)
        conn.commit()

        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")

    def create_imputation_set(self, limit_factor):

        conn, cur = self.create_postgres_connection()

        num_mask = self.pipeline.named_steps['column_transformer'].transformers_[0][2]
        cat_mask = self.pipeline.named_steps['column_transformer'].transformers_[1][2]
        model_name = (self.pipeline[-1].__class__.__name__).lower()
        function_name = self.experiment_name + '_' + model_name + '_impute_set'
        return_type = self.experiment_name + '_test'
        table_name = self.experiment_name + '_test'
        normalized_feature_names = [self.text_normalizer(feature.lower()) for feature in self.experiment_feature_names]

        if cat_mask:
            cat_imputer_array = self.pipeline.named_steps['column_transformer'].transformers_[1][1].named_steps['cat_imputer'].statistics_
        if num_mask:
            num_imputer_array = self.pipeline.named_steps['column_transformer'].transformers_[0][1].named_steps['num_imputer'].statistics_

        selection_statement = 'SELECT ROW_ID, '
        for idf in range(self.experiment_input_dimensions):
            feature_name = normalized_feature_names[idf]
            
            if idf in cat_mask:
                position_in_cat_mask = cat_mask.index(idf)
                imputation_value = """'""" + str(cat_imputer_array[position_in_cat_mask]) + """'"""
            else:
                position_in_num_mask = num_mask.index(idf)
                imputation_value = num_imputer_array[position_in_num_mask]

            imputation_statement = 'CASE WHEN ' + feature_name + """='NaN' THEN """ + str(imputation_value) + ' ELSE ' + feature_name + ' END'
            
        
            if idf < self.experiment_input_dimensions - 1:
                selection_statement += imputation_statement + ', '
            else:
                selection_statement += imputation_statement + ', class'

        function_statement = sql.SQL(
                            """
                                DROP FUNCTION IF EXISTS pgml.{function_name} CASCADE;
                                CREATE OR REPLACE FUNCTION pgml.{function_name}() RETURNS SETOF pgml.{return_type} AS
                                $func$
                                    {selection_statement} FROM pgml.{table_name} ORDER BY ROW_ID ASC LIMIT {limit_factor};
                                $func$
                                LANGUAGE SQL STABLE PARALLEL SAFE;
                            """).format(function_name = sql.Identifier(function_name)
                                        , return_type = sql.Identifier(return_type)
                                        , table_name = sql.Identifier(table_name)
                                        , selection_statement = sql.SQL(selection_statement)
                                        , limit_factor = sql.SQL(str(limit_factor))
                                        )
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('impute_set')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_imputed;
                                        CREATE MATERIALIZED VIEW pgml.{name}_imputed AS 
                                        
                                        SELECT * FROM  pgml.{function_name}();
                                        
                                    """).format(name=sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , limit_factor = sql.SQL(str(limit_factor))
                                                , function_name = sql.Identifier(function_name)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

    def create_encode_scale_set(self):

        conn, cur = self.create_postgres_connection()

        num_mask = self.pipeline.named_steps['column_transformer'].transformers_[0][2]
        cat_mask = self.pipeline.named_steps['column_transformer'].transformers_[1][2]
        function_name = self.experiment_name + '_' + self.model_name + '_encode_scale_set'
        return_type = 'NUMERIC[]'
        table_name = self.experiment_name + '_imputed'

        normalized_feature_names = [self.text_normalizer(feature.lower()) for feature in self.experiment_feature_names]

        numerical_feature_names = [normalized_feature_names[i] for i in num_mask]
        categorical_feature_names = [normalized_feature_names[i] for i in cat_mask]

        scaler = self.pipeline.named_steps['column_transformer'].transformers_[0][1].named_steps['scaler']
        means = scaler.mean_
        vars = np.sqrt(scaler.var_)

        select_statement = 'SELECT ARRAY['
        for idf, feature in enumerate(numerical_feature_names):
            if vars[idf] > 0:
                if cat_mask:
                    select_statement += '(' + feature + '-(' + str(means[idf]) + '))' + '/' + str(vars[idf]) + ', '
                # elif cat_mask and idf == len(numerical_feature_names) - 1:
                #     select_statement += '(' + feature + '-(' + str(means[idf]) + '))' + '/' + str(vars[idf])
                elif not cat_mask and idf < len(numerical_feature_names) - 1:
                    select_statement += '(' + feature + '-(' + str(means[idf]) + '))' + '/' + str(vars[idf]) + ', '
                elif not cat_mask and idf == len(numerical_feature_names) - 1: 
                    select_statement += '(' + feature + '-(' + str(means[idf]) + '))' + '/' + str(vars[idf]) + ']'
            else:
                if cat_mask and idf < len(numerical_feature_names) - 1:
                    select_statement +=  feature + ', '
                elif cat_mask and idf == len(numerical_feature_names) - 1:
                    select_statement +=  feature
                elif not cat_mask and idf < len(numerical_feature_names) - 1:
                    select_statement +=  feature + ', '
                elif not cat_mask and idf == len(numerical_feature_names) - 1:
                    select_statement +=  feature + ']'
        
        for idf, feature in enumerate(categorical_feature_names):
            unique_categories = self.pipeline.named_steps['column_transformer'].transformers_[1][1].named_steps['encoder'].categories_[idf]
            for idc, category in enumerate(unique_categories):
                if idf == len(categorical_feature_names) - 1 and idc == len(unique_categories) - 1:
                    select_statement += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END] '
                else:
                    select_statement += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END, '

        function_statement = sql.SQL(
                            """
                                DROP FUNCTION IF EXISTS pgml.{function_name} CASCADE;
                                CREATE OR REPLACE FUNCTION pgml.{function_name}() RETURNS SETOF {return_type} AS
                                $func$
                                    {select_statement} FROM pgml.{table_name};
                                $func$
                                LANGUAGE SQL STABLE PARALLEL SAFE;
                            """).format(function_name = sql.Identifier(function_name)
                                        , return_type = sql.SQL(return_type)
                                        , table_name = sql.SQL(table_name)
                                        , select_statement = sql.SQL(select_statement))
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('encode_scale_set')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_encoded;
                                        CREATE MATERIALIZED VIEW pgml.{name}_encoded AS 
                                        
                                        SELECT m.* as m FROM  pgml.{function_name}() as m;
                                        
                                    """).format(name=sql.SQL(self.experiment_name)
                                                , function_name = sql.Identifier(function_name)
                                                
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

    def create_preprocessing_pipeline(self):

        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS pgml.{input_name}_preprocess CASCADE;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_preprocess(OUT row_id INTEGER, OUT col_id INTEGER, OUT val NUMERIC) RETURNS SETOF record AS
                                        $func$
                                        with d as (select row_number() over () as id, m.* from pgml.{name}_encoded as m)

                                        SELECT m.id - 1 as row_id, u.ord - 1 as col_id, u.val
                                        FROM   d m,
                                            LATERAL unnest(m.m) WITH ORDINALITY AS u(val, ord)
                                        where u.val != 0
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;

                                    """).format(input_name = sql.SQL(self.input_name)
                                                , name = sql.SQL(self.experiment_name))
        ### Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('preprocess')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_preprocessed;
                                        CREATE MATERIALIZED VIEW pgml.{name}_preprocessed AS 
                                        
                                        SELECT * FROM  pgml.{input_name}_preprocess();
                                        
                                    """).format(name=sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

        
    
    def create_measure_function(self, function):

        conn, cur = self.create_postgres_connection()
    
        function_statement = sql.SQL("""
                    DROP FUNCTION IF EXISTS pgml.measure_{function}_runtime_{input_name} ;
                    CREATE OR REPLACE FUNCTION pgml.measure_{function}_runtime_{input_name}() returns NUMERIC AS $$

                    DECLARE
                    _timing1  timestamptz;
                    _start_ts timestamptz;
                    _end_ts   timestamptz;
                    _overhead numeric;     -- in ms
                    _timing   numeric;     -- in ms
                    runtime numeric;
                    BEGIN
                    _timing1  := clock_timestamp();
                    _start_ts := clock_timestamp();
                    _end_ts   := clock_timestamp();
                    -- take minimum duration as conservative estimate
                    _overhead := 1000 * extract(epoch FROM LEAST(_start_ts - _timing1
                                                                , _end_ts   - _start_ts));

                    _start_ts := clock_timestamp();
                    PERFORM * FROM pgml.{input_name}_{function}();  -- your query here, replacing the outer SELECT with PERFORM
                    _end_ts   := clock_timestamp();
                    
                    -- RAISE NOTICE 'Timing overhead in ms = %', _overhead;
                    runtime := 1000 * (extract(epoch FROM _end_ts - _start_ts)) - _overhead;

                    RETURN runtime;
                    END;
                    $$ LANGUAGE plpgsql;
                """).format(input_name = sql.SQL(self.input_name)
                            , function = sql.SQL(function))
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()
    
    def create_acc_measure_function(self, function, score):

        conn, cur = self.create_postgres_connection()

        scoring_function = self.input_name + '_' + function

        if self.task == 'classification':

            query = sql.SQL(""" with counts as (select sum(case when round(p.prediction) = t.class and t.class = 1 then 1 else 0 end) as tp
                                        , sum(case when round(p.prediction) = 0 and t.class = 1 then 1 else 0 end) as fn
                                        , sum(case when round(p.prediction) = 1 and t.class = 0 then 1 else 0 end) as fp
                                        , sum(case when round(p.prediction) = t.class then 1 else 0 end)::decimal / count(p.*) as acc
                                        from pgml.{scoring_function}() p
                                        left join pgml.{experiment_name}_test t
                                        on p.row_id = t.row_id
                                        )

                                        , scores as (select case when tp+fn > 0 then tp::decimal / (tp+fn) else 0 end as recall
                                                    , case when tp+fp > 0 then tp::decimal / (tp+fp) else 0 end as precision
                                                    ,  case when tp+fn > 0 and tp+fp > 0 then (2*(tp::decimal/(tp+fn) * tp::decimal/(tp+fp))) / (tp::decimal/(tp+fn) + tp::decimal/(tp+fp)) else 0 end as f1
                                                    , acc
                                                    from counts 
                                        )  """).format(scoring_function = sql.SQL(scoring_function)
                                                        , experiment_name = sql.SQL(self.experiment_name))
        elif self.task == 'regression':

            query = sql.SQL(""" with scores as(select sqrt(sum((ln(p.prediction + 1) - ln(t.class + 1))^2)/count(p.*)) as rmsle
                                        from pgml.{scoring_function}() p
                                        left join pgml.{experiment_name}_test t
                                            on p.row_id = t.row_id
                                        )""").format(scoring_function = sql.SQL(scoring_function)
                                                        , experiment_name = sql.SQL(self.experiment_name))

        function_statement = sql.SQL(""" 
                                        DROP FUNCTION IF EXISTS pgml.measure_{score}_{scoring_function};
                                        CREATE OR REPLACE FUNCTION pgml.measure_{score}_{scoring_function}(OUT score NUMERIC) RETURNS NUMERIC AS
                                        $$
                                        {query}

                                        select {score}
                                        from scores;

                                        $$ LANGUAGE SQL;
                                    """).format(scoring_function = sql.SQL(scoring_function)
                                                , experiment_name = sql.SQL(self.experiment_name)
                                                , score = sql.SQL(score)
                                                , query = query)

        ### Create function
        cur.execute(function_statement)
        conn.commit()

    def generate_report_function_pg(self):

        conn, cur = self.create_postgres_connection()

        if self.model.__class__.__name__ in ('MLPRegressor', 'MLPClassifier'):
            coef_statement = 'nn_matrix_' + self.experiment_name
        elif self.model.__class__.__name__ in ('LogisticRegression', 'LinearRegression'):
            coef_statement = self.input_name + '_coefficients'

        if self.task == 'classification':
            out_statement = """, OUT acc NUMERIC
                                , OUT prec NUMERIC
                                , OUT recall NUMERIC
                                , OUT f1 NUMERIC"""
            select_statement_model = sql.SQL(""" , pgml.measure_acc_{input_name}_score() as acc
                                        , pgml.measure_precision_{input_name}_score() as precision
                                        , pgml.measure_recall_{input_name}_score() as recall
                                        , pgml.measure_f1_{input_name}_score() as f1""").format(input_name = sql.SQL(self.input_name))
            select_statement_index = sql.SQL(""", pgml.measure_acc_{input_name}_score_kv() as acc
                                        , pgml.measure_precision_{input_name}_score_kv() as precision
                                        , pgml.measure_recall_{input_name}_score_kv() as recall
                                        , pgml.measure_f1_{input_name}_score_kv() as f1""").format(input_name = sql.SQL(self.input_name))
            select_statement_pgml = sql.SQL(""", pgml.measure_acc_{input_name}_score_pgml() as acc
                                        , pgml.measure_precision_{input_name}_score_pgml() as precision
                                        , pgml.measure_recall_{input_name}_score_pgml() as recall
                                        , pgml.measure_f1_{input_name}_score_pgml() as f1""").format(input_name = sql.SQL(self.input_name))
            
        elif self.task == 'regression':
            out_statement = """, OUT rmsle NUMERIC"""
            select_statement_model = sql.SQL(""" , pgml.measure_rmsle_{input_name}_score() as rmsle""").format(input_name = sql.SQL(self.input_name))
            select_statement_index = sql.SQL(""", pgml.measure_rmsle_{input_name}_score_kv() as rmsle""").format(input_name = sql.SQL(self.input_name))
            select_statement_pgml = sql.SQL(""", pgml.measure_rmsle_{input_name}_score_pgml() as rmsle""").format(input_name = sql.SQL(self.input_name))
        
        #### Sizes
        current = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)
        
        with open(path + '/' + self.input_name + '.joblib', "wb") as File:
            dump(self.pipeline, File)
        model_size = getsize(path + '/' + self.input_name + '.joblib') 
        
        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS pgml.{input_name}_report;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_report(
                                                                                        OUT method TEXT
                                                                                        , OUT size NUMERIC
                                                                                        {out_statement}
                                                                                    ) returns SETOF record AS
                                        $$
                                        select 'model' as method
                                                , pg_total_relation_size('pgml.{coef_statement}') as size
                                                {select_statement_model}

                                        union all

                                        select 'index' as method
                                                , pg_total_relation_size('pgml.{input_name}_kv')
                                                {select_statement_index}
                                        
                                        union all

                                        select 'pgml' as method
                                                , {model_size} as size
                                                {select_statement_pgml};

                                        
                                        $$ LANGUAGE SQL STABLE;
                                        """
                                     ).format(input_name = sql.SQL(self.input_name)
                                              , out_statement = sql.SQL(out_statement)
                                              , select_statement_model = select_statement_model
                                              , select_statement_index = select_statement_index
                                              , select_statement_pgml  = select_statement_pgml
                                              , coef_statement = sql.SQL(coef_statement)
                                              , model_size = sql.SQL(str(model_size))  
                                            )
        ### Create function
        cur.execute(function_statement)
        conn.commit()
        

    def create_coef_table(self):

        conn, cur = self.create_postgres_connection()
        
        table_statement = sql.SQL("""
                                DROP TABLE IF EXISTS pgml.{input_name}_coefficients CASCADE;
                                CREATE TABLE pgml.{input_name}_coefficients
                                (col_id INTEGER
                                , val NUMERIC
                                , PRIMARY KEY(col_id)
                                ); 
                            """).format(input_name = sql.SQL(self.input_name))
        ### Create function
        cur.execute(table_statement)
        conn.commit()
        
        ##### Insert coefficients and bias in the nn table

        insert_statement = 'INSERT INTO pgml.' + self.input_name + '_coefficients (col_id, val) VALUES '
        for idc, c in enumerate(self.pipeline[-1].coef_):
            if self.task == 'regression':
                tup = (idc, c)
                insert_statement += str(tup) + ','
            elif self.task == 'classification':
                for idx, x in enumerate(c):
                    tup = (idx, x)
                    insert_statement += str(tup) + ','
        
        if self.task == 'regression':
            intercept = (-1, self.pipeline[-1].intercept_)
        elif self.task == 'classification':
            intercept = (-1, self.pipeline[-1].intercept_[0])
        
        insert_statement += str(intercept) + ';'

        ### Create function
        cur.execute(insert_statement)
        conn.commit()

    def create_scorer_function_lr(self):

        conn, cur = self.create_postgres_connection()

        ### Creates a function to transform preprocessed arrays into predictions

        if self.task == 'classification':
            select_statement = '1/(1 + EXP(-(coef.val + intercept.val))) as prediction' 
        elif self.task == 'regression':
            select_statement = 'coef.val + intercept.val as prediction' 

        function_statement = sql.SQL("""DROP FUNCTION IF EXISTS pgml.{input_name}_score;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_score(OUT row_id INTEGER, OUT prediction NUMERIC)
                                        RETURNS SETOF record 
                                        AS $func$

                                        with pre as (select * from pgml.{experiment_name}_preprocessed),

                                        intercept as (select val from pgml.{input_name}_coefficients where col_id = -1),

                                        coef as (select pre.row_id, sum(pre.val * b.val) as val
                                                from pre
                                                left join pgml.{input_name}_coefficients b
                                                on pre.col_id = b.col_id
                                                group by 1)

                                        select coef.row_id, {select_statement}
                                        from coef, intercept
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    """).format(experiment_name = sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , select_statement = sql.SQL(select_statement)
                                                )       

        ### Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score')
        if self.task == 'classification':
            self.create_acc_measure_function('score', 'acc')
            self.create_acc_measure_function('score', 'f1')
            self.create_acc_measure_function('score', 'precision')
            self.create_acc_measure_function('score', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score', 'rmsle')

    def create_nn_table(self):

        conn, cur = self.create_postgres_connection()

        function_statement = 'DROP TABLE IF EXISTS pgml.nn_matrix_' + self.experiment_name + '; '
        function_statement += """ CREATE TABLE pgml.nn_matrix_""" + self.experiment_name
        function_statement += """ (id  INTEGER,
                                row INTEGER,
                                col INTEGER,
                                val NUMERIC,
                                PRIMARY KEY(id, row, col)
                                ); 
                            """
        cur.execute(function_statement)
        conn.commit()
        
        ##### Insert coefficients and bias in the nn table

        insert_statement = """ INSERT INTO pgml.nn_matrix_""" + self.experiment_name + """ (id, row, col, val) VALUES """
        for idc, c in enumerate(self.pipeline[-1].coefs_):
            for idx, weights in enumerate(c):
                for idw, w in enumerate(weights):
                    tup = (idc, idx, idw, w)
                    insert_statement += str(tup) + ','
        for idc, i in enumerate(self.pipeline[-1].intercepts_):
            idc += len(self.pipeline[-1].coefs_)
            for idx, bias in enumerate(i):
                tup = (idc, idx, 0, bias)
                if idc < len(self.pipeline[-1].intercepts_) + len(self.pipeline[-1].coefs_) - 1 and idx < len(i):
                    insert_statement += str(tup) + ','
                else:
                    insert_statement += str(tup) + ';'
        
        cur.execute(insert_statement)
        conn.commit()
        
        # return insert_statement

    def create_nn_scorer_function(self):

        conn, cur = self.create_postgres_connection()
        ### Creates a function to transform preprocessed arrays into predictions

        if self.task == 'classification':
            select_statement = '1/(1 + EXP(-(m1.val + nn.val))) as prediction'
        elif self.task == 'regression':
            select_statement = 'm1.val + nn.val as prediction'

        function_statement = sql.SQL("""DROP FUNCTION IF EXISTS pgml.{input_name}_score;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_score(OUT row_id INT, OUT prediction NUMERIC) returns SETOF RECORD AS $func$
                                        
                                        WITH input_weights as
                                        (
                                        SELECT m1.row_id, m2.col, sum(m1.val * m2.val) AS val
                                        FROM   pgml.{experiment_name}_preprocessed m1
                                        join (select * from pgml.nn_matrix_{experiment_name} where id=0) m2
                                            on m1.col_id = m2.row
                                        GROUP BY m1.row_id, m2.col
                                        )
                                        ,

                                        activation as
                                        (
                                        select iw.row_id, iw.col, 1/(1 + EXP(-(iw.val + nn.val))) as val
                                        from input_weights iw
                                        join (select * from pgml.nn_matrix_{experiment_name} where id=2) nn
                                            on iw.col = nn.row
                                        )
                                        ,

                                        output_weights as
                                        (
                                        select m1.row_id, sum(m1.val * nn2.val) as val
                                        from activation m1
                                        join (select * from pgml.nn_matrix_{experiment_name} where id=1) nn2
                                            on m1.col = nn2.row
                                        group by 1
                                        )
                                                                        
                                        select m1.row_id, {select_statement}
                                        from output_weights m1, pgml.nn_matrix_{experiment_name} nn
                                        where nn.id = 3;
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    
                                    """).format(experiment_name = sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , select_statement = sql.SQL(select_statement)
                                                )
        
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score')
        if self.task == 'classification':
            self.create_acc_measure_function('score', 'acc')
            self.create_acc_measure_function('score', 'f1')
            self.create_acc_measure_function('score', 'precision')
            self.create_acc_measure_function('score', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score', 'rmsle')
    
    def train_model_in_pgml(self):

        conn, cur = self.create_postgres_connection()

        if self.task == 'regression':
            hyper_statement = ''
        elif self.task == 'classification':
            hyper_statement = """,hyperparams => '{
                                                "max_iter": 100000
                                            }'"""

        function_statement = sql.SQL(
                                        """
                                            SELECT pgml.train(
                                            project_name => '{experiment_name}'::text, 
                                            task => '{task}'::text, 
                                            relation_name => 'pgml.{experiment_name}_train'::text,
                                            y_column_name => 'target'::text,
                                            test_size => 0.01,
                                            algorithm => 'linear'
                                            {hyper_statement}
                                        );
                                    
                                        """
                                    ).format(task = sql.SQL(self.task)
                                             , experiment_name = sql.SQL(self.experiment_name)
                                             , hyper_statement = sql.SQL(hyper_statement))
        
        cur.execute(function_statement)
        conn.commit()
    
    def deploy_pgml_trained_model(self):
        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL( """SELECT * FROM pgml.deploy(
                                    '{experiment_name}'::text,
                                    strategy => 'most_recent'
                                );""").format(experiment_name = sql.SQL(self.experiment_name))
        
        cur.execute(function_statement)
        conn.commit()
    
    def create_scorer_function_pgml(self):

        function_statement = sql.SQL(""" 
                                        DROP FUNCTION IF EXISTS pgml.{input_name}_score_pgml;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_score_pgml(OUT row_id INT, OUT prediction NUMERIC) returns SETOF RECORD AS $func$
                                        WITH predictions AS (
                                        SELECT row_number() over () as row_id
                                                , pgml.predict_batch('{experiment_name}'::text, d.m) AS prediction
                                        FROM pgml.{experiment_name}_encoded d
                                    )
                                    SELECT row_id - 1 as row_id, prediction FROM predictions
                                    $func$ LANGUAGE SQL STABLE;
                                    
                                    """).format(input_name = sql.SQL(self.input_name)
                                                , experiment_name = sql.SQL(self.experiment_name))
        
        conn, cur = self.create_postgres_connection()

        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score_pgml')
        if self.task == 'classification':
            self.create_acc_measure_function('score_pgml', 'acc')
            self.create_acc_measure_function('score_pgml', 'f1')
            self.create_acc_measure_function('score_pgml', 'precision')
            self.create_acc_measure_function('score_pgml', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score_pgml', 'rmsle')

    def create_utils(self):

        self.adapt_psycopg2()
        self.create_aux_functions()
    
    def create_test_artifacts(self):

        self.create_test_table()
        self.insert_test_tuples()
    
    def create_train_artifacts(self):

        self.create_train_table()
        self.insert_train_tuples()
    
    def create_arbes_solution(self, limit_factor):

        kv_table_name = (self.experiment_name + '_' + self.model.__class__.__name__).lower() + '_' + 'kv'
        index_name = (self.experiment_name + '_' + self.model.__class__.__name__).lower() + '_' + 'index'

        self.create_kv_table(kv_table_name)
        self.insert_tuples_in_kv_table(kv_table_name)
        self.create_index(index_name, kv_table_name)
        self.create_preproccesing_function(limit_factor)
        # self.create_preproccesing_function_kv()
        self.create_scoring_function_kv()

        print('Index representation was created')
    
    def create_model_postgres_representation(self, limit_factor):
        
        
        self.create_imputation_set(limit_factor)
        self.create_encode_scale_set()
        self.create_preprocessing_pipeline()

        if self.model.__class__.__name__ in ('MLPRegressor', 'MLPClassifier'):
            self.create_nn_table()
            self.create_nn_scorer_function()
        elif self.model.__class__.__name__ in ('LinearRegression', 'LogisticRegression'):
            self.create_coef_table()
            self.create_scorer_function_lr()

        print('Model representation was created')
    
    def create_pgml_artifacts(self):

        self.train_model_in_pgml()
        self.deploy_pgml_trained_model()
        self.create_scorer_function_pgml()
        print('PGML model artifacts were created')
    
    def create_standalone_index(self):

        # Populate index
        index_time = 0
        model_index = Trie(self.task)
        st = time.time()
        tuples = self.get_kv_tuples('standalone')
        for idx, i in enumerate(tuples):
            key = i[:len(self.solution)]
            value = round(i[-1])
            model_index.insert(key, value)
        index_time = time.time() - st
        
        print('index populated')

        return model_index, index_time
    
    def perform_index_inference(self):

        inference_runtimes = np.zeros(self.experiment_test_instances, dtype=float)
        preprocessing_runtimes = np.zeros(self.experiment_test_instances, dtype=float)
        y_pred_trie = np.zeros_like(inference_runtimes)

        experiment = Experiment(self.experiment_name, self.path, self.task)
        experiment.prepare_dataset()
        model_index, time_to_populate_index = self.create_standalone_index()

        for idx, i in enumerate(experiment.X_test):
            instance = i[self.solution]
            start = time.time()
            preprocessed_instance = self.encoder.transform_single(instance, self.solution)
            preprocessing_runtime = time.time() - start
            preprocessing_runtimes[idx] = preprocessing_runtime
            st = time.time()
            y_pred_trie[idx] = round(model_index.query(preprocessed_instance))
            inference_runtimes[idx] = time.time() - st
        
        index_avg_prep_runtimes = preprocessing_runtimes.mean()
        index_avg_scoring_runtimes = inference_runtimes.mean()

        print('index inference done')

        #### Sizes
        current = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + self.input_name + '_' + 'index.joblib', "wb") as File:
            dump(model_index, File)
        index_size = getsize(path + '/' + self.input_name + '_' + 'index.joblib') 

        #### Error

        if self.task == 'classification':
            index_accuracy = accuracy_score(experiment.y_test, y_pred_trie)
            index_f1 = f1_score(experiment.y_test, y_pred_trie)
            index_recall = recall_score(experiment.y_test, y_pred_trie)
            index_precision = precision_score(experiment.y_test, y_pred_trie)

            return index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_accuracy, index_f1, index_recall, index_precision, time_to_populate_index
        elif self.task == 'regression':
            index_error = mean_squared_log_error(experiment.y_test, y_pred_trie, squared=False)
        
            return index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_error, time_to_populate_index
    
    def perform_model_inference(self):
        inference_runtimes = np.zeros(self.experiment_test_instances, dtype=float)
        preprocessing_runtimes = np.zeros(self.experiment_test_instances, dtype=float)

        experiment = Experiment(self.experiment_name, self.path, self.task)
        experiment.prepare_dataset()
        for idx, i in enumerate(experiment.X_test):
            instance = i.reshape((1, i.size))
            start = time.time()
            transformed_instance = self.pipeline[:-1].transform(instance)
            end = time.time() - start
            preprocessing_runtimes[idx] = end
            start = time.time()
            self.pipeline[-1].predict(transformed_instance)
            end = time.time() - start
            inference_runtimes[idx] = end
        
        model_avg_prep_runtimes = preprocessing_runtimes.mean()
        model_avg_scoring_runtimes = inference_runtimes.mean()

        print('model inference done')

        #### Sizes
        current = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + self.input_name + '.joblib', "wb") as File:
            dump(self.pipeline, File)
        model_size = getsize(path + '/' + self.input_name + '.joblib') 

        #### Error

        y_pred = self.pipeline.predict(experiment.X_test)

        if self.task == 'regression':
            model_error = mean_squared_log_error(experiment.y_test, y_pred, squared=False)

            return model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_error
        elif self.task == 'classification':
            model_accuracy = accuracy_score(experiment.y_test, y_pred)
            model_f1 = f1_score(experiment.y_test, y_pred)
            model_recall = recall_score(experiment.y_test, y_pred)
            model_precision = precision_score(experiment.y_test, y_pred)

            return model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_accuracy, model_f1, model_recall, model_precision
        
    def create_report(self):
        
        if self.task == 'regression':
            model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_error = self.perform_model_inference()
            index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_error, index_time = self.perform_index_inference()
            summary = [[self.experiment_name, self.model_name, self.experiment_training_instances, self.experiment_input_dimensions, self.experiment_transformed_input_size, self.training_preprocessing_runtime, self.training_time, self.encoding_time, self.solution_time, index_time, self.max_path_length, self.experiment_test_instances, model_error, index_error, model_avg_prep_runtimes, model_avg_scoring_runtimes, model_avg_prep_runtimes + model_avg_scoring_runtimes, index_avg_prep_runtimes, index_avg_scoring_runtimes, index_avg_prep_runtimes + index_avg_scoring_runtimes, model_size, index_size]]
            columns = ['name', 'model', 'training_instances', 'original_dimensions', 'input_dimensions', 'training_preprocessing_runtime', 'training_runtime', 'encoding_runtime', 'solution_runtime', 'index_runtime', 'max_path_length', 'testing_instances', 'model_error', 'index_error', 'model_avg_preprocessing_runtime', 'model_avg_scoring_runtime', 'model_end_to_end_runtime', 'index_avg_preprocessing_runtime', 'index_avg_scoring_runtime', 'index_end_to_end_runtime', 'model_size', 'index_size']
        elif self.task =='classification':
            model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_accuracy, model_f1, model_recall, model_precision = self.perform_model_inference()
            index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_accuracy, index_f1, index_recall, index_precision, index_time = self.perform_index_inference()
            summary = [[self.experiment_name, self.model_name, self.experiment_training_instances, self.experiment_input_dimensions, self.experiment_transformed_input_size, self.training_preprocessing_runtime, self.training_time, self.encoding_time, self.solution_time, index_time, self.max_path_length, self.experiment_test_instances, model_accuracy, model_f1, model_recall, model_precision, index_accuracy, index_f1, index_recall, index_precision, model_avg_prep_runtimes, model_avg_scoring_runtimes, model_avg_prep_runtimes + model_avg_scoring_runtimes , index_avg_prep_runtimes, index_avg_scoring_runtimes, index_avg_prep_runtimes + index_avg_scoring_runtimes, model_size, index_size]]
            columns = ['name', 'model', 'training_instances', 'original_dimensions', 'input_dimensions', 'training_preprocessing_runtime', 'training_runtime', 'encoding_runtime', 'solution_runtime', 'index_runtime', 'max_path_length', 'testing_instances', 'model_accuracy', 'model_f1', 'model_recall', 'model_precision', 'index_accuracy', 'index_f1', 'index_recall', 'index_precision', 'model_avg_preprocessing_runtime', 'model_avg_scoring_runtime', 'model_end_to_end_runtime', 'index_avg_preprocessing_runtime', 'index_avg_scoring_runtime', 'index_end_to_end_runtime', 'model_size', 'index_size']

        summary_df = pd.DataFrame(summary, columns=columns)
        current = os.path.realpath(__file__)
        path = os.path.join(os.path.dirname(current), 'experiments/output')
        Path(path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, self.input_name)

        if self.is_balanced:
            path += '_balanced'
        if self.populate_paths:
            path += '_pp'
        
        # summary_df.to_csv(path + '_standalone.csv', index=False)

        return summary_df
    
    def create_report_pg(self, limit_factor):

        DEC2FLOAT = new_type(
        DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
        register_type(DEC2FLOAT)

        conn, cur = self.create_postgres_connection()

        size_effectiveness_query = sql.SQL(""" select * from pgml.{input_name}_report()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))

        cur.execute(size_effectiveness_query)

        result = cur.fetchall()

        results = []
        for row in result:
            results.append(list(row))
        
        #### Get performance numbers for model representation:

        impute_time_query = sql.SQL(""" analyze; select * from pgml.measure_impute_set_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))
        encode_scale_time_query = sql.SQL(""" analyze; select * from pgml.measure_encode_scale_set_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))
        scoring_time_query = sql.SQL(""" analyze; select * from pgml.measure_score_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))

        queries = [impute_time_query, encode_scale_time_query, scoring_time_query]
        for q in queries:
            cur.execute(q)
            r = cur.fetchone()
            results[0].append(r[0])
        
        #### Get performance numbers for Index:
        
        translate_time_query = sql.SQL(""" analyze; select * from pgml.measure_translate_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))
        scoring_time_query = sql.SQL(""" analyze; select * from pgml.measure_score_kv_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))

        results[1].append(0)
        queries = [translate_time_query, scoring_time_query]
        for q in queries:
            cur.execute(q)
            r = cur.fetchone()
            results[1].append(r[0])
        
        #### Get performance numbers for pgml representation:

        impute_time_query = sql.SQL(""" analyze; select * from pgml.measure_impute_set_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))
        encode_scale_time_query = sql.SQL(""" analyze; select * from pgml.measure_encode_scale_set_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))
        scoring_time_query = sql.SQL(""" analyze; select * from pgml.measure_score_pgml_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name), limit_factor = sql.SQL(str(limit_factor)))

        queries = [impute_time_query, encode_scale_time_query, scoring_time_query]
        for q in queries:
            cur.execute(q)
            r = cur.fetchone()
            results[2].append(r[0])
        
        if self.task == 'classification':
            columns = ['Solution', 'Size (B)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Imputation Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)']
        elif self.task =='regression':
            columns = ['Solution', 'Size (B)', 'RMSLE', 'Imputation Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)']

        summary_df = pd.DataFrame(results, columns=columns)
        summary_df['End-to-End Latency (ms)'] = summary_df.apply(lambda x: x['Imputation Latency (ms)'] + x['Encode Scale Latency (ms)'] + x['Score Latency (ms)'], axis=1)
        summary_df['Experiment'] = self.experiment_name
        summary_df['Algorithm'] = self.model_name
        summary_df['Batch Size (Records)'] = limit_factor

        return summary_df

    


        




