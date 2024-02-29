import os
from pathlib import Path
from optimizer import Problem, Optimizer
from encoder import Encoder
import pandas as pd
import numpy as np
from database_connect import connect
from psycopg2 import sql
from psycopg2.extensions import register_adapter, AsIs, register_type, new_type, DECIMAL
import re
from itertools import product
import time 
from sklearn.metrics import accuracy_score, mean_squared_log_error, recall_score, precision_score, f1_score
from joblib import dump
from copy import deepcopy
from os.path import getsize
from inference_trie import Trie
from scipy import sparse
from io import StringIO
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm
from numpy import percentile

class Transpiler:
    """This is the parent class of the transpiler. It contains some standard methods to interact with a postgres instance, such as create a postgres connection, insert test and train data
        , and a fit function that trains the pipeline that is passed in the constructor.
    """
    def __init__(self, X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb=False, pipeline=None, inferdb_columns=None, feature_names=None, section=None, featurizer_query=None) -> None:
        """Constructor method for the transpiler parent class. Responsible for the training of the passed pipeline.

        Args:
            X_train (np.array or pd.DataFrame): array or dataframe containing the train data
            X_test (np.array or pd.DataFrame): array or dataframe containing the test data
            y_train (np.array): array containing the train true values
            y_test (np.array): array containing the test true values
            experiment_name (string): name of the experiment
            task (string): 
                - 'classification' for binary classification
                - 'regression' for regression
                - 'multi-class' for multi-label classification
            has_featurizer (bool): boolean indicator for featurizer at the 0 position (pipeline[0]) of the pipeline steps.
            featurize_inferdb (bool, optional): If true, featurizer should implement method 'featurize_for_inferdb' which transforms the input data and returns a set
                                                of defined features for InferDB . Defaults to False.
            pipeline (Scikitlearn Pipeline, optional): Pipeline to train input data. Transforms input data into predictions. Defaults to None.
            inferdb_columns (list, optional): List containing a subset of columns for InferDB. Defaults to None.
            feature_names (list, optional): Ordered list containing the names for each column in the input data. Defaults to None.
            section (string, optional): Schema prefix for postgres. Defaults to None.
            featurizer_query (psycopg2.Composed, optional): SQL composition containing the query to to transform the input data into a featurized relation in postgres. Defaults to None.
        """
        self.experiment_name = self.text_normalizer(experiment_name).lower()
        self.model = pipeline[-1]
        self.task = task
        self.model_name = (self.model.__class__.__name__).lower()
        self.input_name = self.experiment_name + '_' + self.model_name
        self.has_featurizer = has_featurizer
        self.featurize_inferdb = featurize_inferdb
        self.pipeline = pipeline 
        self.X_train = X_train
        self.X_test = X_test
        if self.task == 'regression':
            self.y_train = np.log(y_train)
        else:
            self.y_train = y_train
        self.y_test = y_test

        self.experiment_feature_names = feature_names
        self.inferdb_columns = inferdb_columns

        self.section = section

        self.featurizer_query = featurizer_query
       
        self.fit()


    def create_postgres_connection(self):
        """Method to create a postgres connection

        Returns:
            conn, cur: db connection and cursor
        """
        conn = connect(self.section)
        cur = conn.cursor()

        return conn, cur

    def addapt_numpy_float64(numpy_float64):
        """Adapts numpy dtype float64 to postgres dtypes

        Args:
            numpy_float64 (numpy float64 dtype): Numpy's float64 dtype

        Returns:
            _type_: psycopg adapter
        """
        return AsIs(numpy_float64)
    
    def addapt_numpy_int64(numpy_int64):
        """Adapts numpy dtype int64 to postgres dtypes

        Args:
            numpy_int64 (numpy int64 dtype): Numpy's int64 dtype

        Returns:
            _type_: psycopg adapter
        """
        return AsIs(numpy_int64)
    
    def adapt_psycopg2(self):
        """Method to adapt int64 and float64 numpy dtypes to postgres dtypes
        """
        register_adapter(np.float64, self.addapt_numpy_float64)
        register_adapter(np.int64, self.addapt_numpy_int64)

    def text_normalizer(self, text):
        """Method to clean strings 

        Args:
            text (string): string to clean

        Returns:
            string: cleaned string
        """

        rep = {"-": "_", ".": "_", "?":"", "/":"", "(":"_", ")":"_", "&":"_", "#":"x"} # define desired replacements here

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pattern = re.compile("|".join(rep.keys()))

        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    
    def create_aux_functions(self):
        """Method to crate auxiliary functions in postgres that are needed in the preprocessing pipelines.
        """

        conn, cur = self.create_postgres_connection()

        parent = Path(__file__).resolve().parents[1]
        
        geo_distance = os.path.join(parent, 'sql_scripts/geo_distance.sql')
        prefix_search = os.path.join(parent, 'sql_scripts/prefix_search.sql')
        crazy_exp = os.path.join(parent, 'sql_scripts/crazy_exp.sql')

        gd = open(geo_distance, 'r')
        ps = open(prefix_search, 'r')
        exp = open(crazy_exp, 'r')

        gdr = gd.read()
        psr = ps.read()
        expr = exp.read()

        sql_gd = sql.SQL(gdr)
        sql_ps = sql.SQL(psr)
        sql_exp = sql.SQL(expr)

        cur.execute(sql_gd)
        conn.commit()
    
        cur.execute(sql_ps)
        conn.commit()

        cur.execute(sql_exp)
        conn.commit()
    
    def evaluate_runtime(self, q):
        """Extracts a query runtime from postgres' explain analyze

        Args:
            q (string): query to evaluate

        Returns:
            float: runtime in ms
        """        """"""


        conn, cur = self.create_postgres_connection()

        query = sql.SQL(
                        """
                        analyze;
                        explain analyze  
                        {q}
                        """
                    ).format(q = q)

        cur.execute(query)
        analyze_fetched = cur.fetchall()
        runtime = float(analyze_fetched[-1][0].split(':')[1].strip()[:-3])

        return runtime
    
    def create_measure_function(self, function):
        """creates function in postgres to measure the runtime of a query

        Args:
            function (string): function name to evaluate
        """        """"""

        conn, cur = self.create_postgres_connection()
    
        function_statement = sql.SQL("""
                    DROP FUNCTION IF EXISTS {section}.measure_{function}_runtime_{input_name} ;
                    CREATE OR REPLACE FUNCTION {section}.measure_{function}_runtime_{input_name}() returns NUMERIC AS $$

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
                    PERFORM * FROM {section}.{input_name}_{function}();  -- your query here, replacing the outer SELECT with PERFORM
                    _end_ts   := clock_timestamp();
                    
                    -- RAISE NOTICE 'Timing overhead in ms = %', _overhead;
                    runtime := 1000 * (extract(epoch FROM _end_ts - _start_ts)) - _overhead;

                    RETURN runtime;
                    END;
                    $$ LANGUAGE plpgsql;
                """).format(input_name = sql.SQL(self.input_name)
                            , function = sql.SQL(function)
                            , section = sql.SQL(self.section))
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()
    
    def create_acc_measure_function(self, function, score):
        """creates a postgres function to measure the effectiveness of a scoring function

        Args:
            function (string): function name
            score (string): score; 'f1', 'recall', 'precision', 'acc', 'rmsle'
        """        """"""

        conn, cur = self.create_postgres_connection()

        scoring_function = self.input_name + '_' + function

        if self.task == 'classification':

            query = sql.SQL(""" with counts as (select sum(case when round(p.prediction) = t.target and t.target = 1 then 1 else 0 end) as tp
                                        , sum(case when round(p.prediction) = 0 and t.target = 1 then 1 else 0 end) as fn
                                        , sum(case when round(p.prediction) = 1 and t.target = 0 then 1 else 0 end) as fp
                                        , sum(case when round(p.prediction) = t.target then 1 else 0 end)::decimal / count(p.*) as acc
                                        from {section}.{input_name}_scored p
                                        left join {section}.{experiment_name}_test t
                                        on p.row_id = t.row_id
                                        )

                                        , scores as (select case when tp+fn > 0 then tp::decimal / (tp+fn) else 0 end as recall
                                                    , case when tp+fp > 0 then tp::decimal / (tp+fp) else 0 end as precision
                                                    ,  case when tp+fn > 0 and tp+fp > 0 then (2*(tp::decimal/(tp+fn) * tp::decimal/(tp+fp))) / (tp::decimal/(tp+fn) + tp::decimal/(tp+fp)) else 0 end as f1
                                                    , acc
                                                    from counts 
                                        )  """).format(scoring_function = sql.SQL(scoring_function)
                                                        , experiment_name = sql.SQL(self.experiment_name)
                                                        , section = sql.SQL(self.section)
                                                        , input_name = sql.SQL(self.input_name)
                                                        )
        elif self.task == 'multi-class':

            unique_labels = np.unique(self.y_train)

            count_statement = 'WITH '
            from_statement = ''
            tp_select_statement = '('
            fn_select_statement = '('
            fp_select_statement = '('
            acc_select_statement = '('

            for idl, l in enumerate(unique_labels):

                count_query = sql.SQL("""
                                        counts_{l} as (select sum(case when round(p.prediction) = t.target then 1 else 0 end) as tp
                                            , sum(case when round(p.prediction) != t.target and t.target = {l} then 1 else 0 end) as fn
                                            , sum(case when round(p.prediction) = {l} and t.target != {l} then 1 else 0 end) as fp
                                            , sum(case when round(p.prediction) = t.target then 1 else 0 end)::decimal / count(p.*) as acc
                                            from {section}.{input_name}_scored p
                                            left join {section}.{experiment_name}_test t
                                            on p.row_id = t.row_id
                                            ),
                                        """).format(scoring_function = sql.SQL(scoring_function)
                                                    , experiment_name = sql.SQL(self.experiment_name)
                                                    , section = sql.SQL(self.section)
                                                    , l = sql.SQL(str(l))
                                                    , input_name = sql.SQL(self.input_name)
                                                    )
                count_statement += count_query.as_string(conn)

                if idl < unique_labels.shape[0] - 1:
                    from_statement += 'counts_' + str(l) + ' as c' + str(l) + ', '
                    tp_select_statement += 'c' + str(l) + '.tp + '
                    fn_select_statement += 'c' + str(l) + '.fn + '
                    fp_select_statement += 'c' + str(l) + '.fp + '
                    acc_select_statement += 'c' + str(l) + '.acc + '
                else:
                    from_statement += 'counts_' + str(l) + ' as c' + str(l)
                    tp_select_statement += 'c' + str(l) + '.tp ) as tp'
                    fn_select_statement += 'c' + str(l) + '.fn ) as fn'
                    fp_select_statement += 'c' + str(l) + '.fp ) as fp'
                    acc_select_statement += 'c' + str(l) + '.acc )::decimal / ' + str(unique_labels.shape[0]) + ' as acc'
            
            
            counts_query = sql.SQL("""
                    counts as (
                                  select 
                                            {tp_select_statement},
                                            {fn_select_statement},
                                            {fp_select_statement},
                                            {acc_select_statement}
                                    from {from_statement}
                    )
                """).format(tp_select_statement = sql.SQL(tp_select_statement)
                            , fn_select_statement = sql.SQL(fn_select_statement)
                            , fp_select_statement = sql.SQL(fp_select_statement)
                            , from_statement = sql.SQL(from_statement)
                            , acc_select_statement = sql.SQL(acc_select_statement)
                            )
            
            count_statement += counts_query.as_string(conn)
                                          

            count_statement += """, scores as (select case when tp+fn > 0 then tp::decimal / (tp+fn) else 0 end as recall
                                                , case when tp+fp > 0 then tp::decimal / (tp+fp) else 0 end as precision
                                                ,  case when tp+fn > 0 and tp+fp > 0 then (2*(tp::decimal/(tp+fn) * tp::decimal/(tp+fp))) / (tp::decimal/(tp+fn) + tp::decimal/(tp+fp)) else 0 end as f1
                                                , acc
                                                from counts
                                    )  """

            query = count_statement
            
        elif self.task == 'regression':

            query = sql.SQL(""" with scores as(select sqrt(sum((ln(p.prediction + 1) - ln(t.target + 1))^2)/count(p.*)) as rmsle
                                        from {section}.{input_name}_scored p
                                        left join {section}.{experiment_name}_test t
                                            on p.row_id = t.row_id
                                        )""").format(scoring_function = sql.SQL(scoring_function)
                                                        , experiment_name = sql.SQL(self.experiment_name)
                                                        , section = sql.SQL(self.section)
                                                        , input_name = sql.SQL(self.input_name)
                                                    )

        function_statement = sql.SQL(""" 
                                        DROP FUNCTION IF EXISTS {section}.measure_{score}_{scoring_function};
                                        CREATE OR REPLACE FUNCTION {section}.measure_{score}_{scoring_function}(OUT score NUMERIC) RETURNS NUMERIC AS
                                        $$
                                        {query}

                                        select {score}
                                        from scores;

                                        $$ LANGUAGE SQL;
                                    """).format(scoring_function = sql.SQL(scoring_function)
                                                , experiment_name = sql.SQL(self.experiment_name)
                                                , score = sql.SQL(score)
                                                , query = query if self.task in ('regression', 'classification') else sql.SQL(query)
                                                , section = sql.SQL(self.section)
                                                )

        ### Create function
        cur.execute(function_statement)
        conn.commit()
    
    def create_test_table(self):
        """Method to create a new postgres relation for the test data
        """

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.text_normalizer(self.experiment_name + '_test').lower()
        initial_statement = 'DROP TABLE IF EXISTS ' + str(self.section) + '.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE ' + str(self.section) + '.' + table_name + '(ROW_ID INTEGER,'
        df_test = pd.DataFrame(self.X_test, columns=self.experiment_feature_names)
        df_test['target'] = self.y_test

        for idf, feature in enumerate(list(df_test)):

            if feature == 'target' and self.task == 'regression':
                feature_definition = self.text_normalizer(feature).lower() + ' REAL'
            elif feature == 'target' and self.task in ('classification', 'multi-class'):
                feature_definition = self.text_normalizer(feature).lower() + ' INTEGER'
            elif is_numeric_dtype(df_test[feature]):
                feature_definition = self.text_normalizer(feature).lower() + ' NUMERIC'
            elif re.search("datetime", feature):
                feature_definition = self.text_normalizer(feature).lower() + ' TIMESTAMP'
            elif is_string_dtype(df_test[feature]):
                feature_definition = self.text_normalizer(feature).lower() + ' TEXT'

            if idf < len(list(df_test)) - 1:
                initial_statement = initial_statement + feature_definition + ', '
            else:
                initial_statement = initial_statement + feature_definition + ', PRIMARY KEY(ROW_ID))'

        ### Create table
        cur.execute(initial_statement)
        conn.commit()
    
    def create_train_table(self):
        """Method to create a new postgres relation for the train data
        """

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.text_normalizer(self.input_name + '_train').lower()
        initial_statement = 'DROP TABLE IF EXISTS ' + str(self.section) + '.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE ' + str(self.section) + '.' + table_name + '('
        for idf in range(self.preprocessing_output_shape[1]):
            
            initial_statement += 'f_' + str(idf) + ' REAL, '
        
        if self.task == 'regression':
            initial_statement += 'target double precision)'
        else:
            initial_statement += 'target INTEGER)'

        ### Create table
        cur.execute(initial_statement)
        conn.commit()

    def drop_train_table(self):
        """Method to drop the train data
        """

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.text_normalizer(self.input_name + '_train').lower()
        initial_statement = 'DROP TABLE IF EXISTS ' + str(self.section) + '.' + table_name + ' CASCADE;'

        ### Drop table
        cur.execute(initial_statement)
        conn.commit()

    def clean_up(self):

        conn, cur = self.create_postgres_connection()
        ### Drops all tables after the experiment is done
        
        initial_statement = sql.SQL("""
                                DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_imputed CASCADE;
                                DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_featurized CASCADE;
                                DROP TABLE IF EXISTS {section}.{input_name}_test CASCADE;
                                DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_scored CASCADE;
                                DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_encoded CASCADE;
                                DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_encoded_relational CASCADE;
                                DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_preprocessed CASCADE;
                            """).format(section=sql.SQL(self.section),
                                        input_name=sql.SQL(self.input_name)
                                        )

        ### Drop table
        cur.execute(initial_statement)
        conn.commit()


    def insert_test_tuples(self):
        """Inserts the test tuples into the corresponding postgres relation
        """

        conn, cur = self.create_postgres_connection()
        ### Inserts tuples into test table
        table_name = self.text_normalizer(self.experiment_name + '_test').lower()
        ids = (np.arange(self.experiment_test_instances, dtype=np.intc)).reshape((self.experiment_test_instances, 1))
        test_input = np.hstack((ids, self.X_test, self.y_test.reshape((self.experiment_test_instances, 1))))
        ### Insert test data

        buffer = StringIO()
        df = pd.DataFrame(test_input)
        df[0] = df[0].astype('int')
        if self.task in ('classification', 'multi-class'):
            df[test_input.shape[1]-1] = df[test_input.shape[1]-1].astype('int')
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY {section}.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name),
                                                                                                        section = sql.SQL(self.section)), buffer)
       
        conn.commit()
        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")
    
    def insert_train_tuples(self):
        """Inserts the train tupes into the corresponding train relation
        """

        conn, cur = self.create_postgres_connection()
        ### Inserts tuples into test table
        table_name = self.text_normalizer(self.input_name + '_train').lower()
        
        try:
            train_input = np.hstack((self.x_trans, self.y_train.reshape((self.y_train.shape[0], 1))))
        except ValueError:
            train_input = sparse.hstack((self.x_trans, self.y_train.reshape((self.y_train.shape[0], 1))))
            
            train_input = np.asarray(train_input.todense())

        buffer = StringIO()
        df = pd.DataFrame(train_input)
        if self.task in ('classification', 'multi-class'):
            df[train_input.shape[1]-1] = df[train_input.shape[1]-1].astype('int')
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY {section}.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name),
                                                                                                        section=sql.SQL(self.section)), buffer)
        conn.commit()

        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")
    
    def create_utils(self):
        """creates adapters and auxiliary functions in postgres
        """

        self.adapt_psycopg2()
        self.create_aux_functions()
    
    def create_test_artifacts(self):
        """creates test relation and inserts test tuples
        """

        self.create_test_table()
        self.insert_test_tuples()
    
    def create_train_artifacts(self):
        """creates train relation and inserts train tuples
        """

        self.create_train_table()
        self.insert_train_tuples()
    
    def fit(self):
        """Trains the pipeline
        """
 
        if self.has_featurizer:

            self.x_featurized = self.pipeline[0].fit_transform(self.X_train, self.y_train)
            self.feature_names_featurizer = self.pipeline[0].get_feature_names()
        else:
            self.x_featurized = self.X_train
        
        if self.featurize_inferdb:
            self.x_featurized_inferdb = self.pipeline[0].transform_for_inferdb(self.X_train)
            self.x_test_inferdb = self.pipeline[0].transform_for_inferdb(self.X_test)
        elif self.inferdb_columns and isinstance(self.X_train, np.ndarray):
            self.x_featurized_inferdb = self.X_train[:, self.inferdb_columns]
            self.x_test_inferdb = self.X_test[:, self.inferdb_columns]
        else:
            self.x_featurized_inferdb = self.X_train
            self.x_test_inferdb = self.X_test

        st = time.time()
        self.x_trans = self.pipeline[:-1].fit_transform(self.X_train, self.y_train)
        self.training_preprocessing_runtime = time.time() - st

        st = time.time()
        self.pipeline[-1].fit(self.x_trans, self.y_train)
        self.training_time = self.training_preprocessing_runtime + (time.time() - st)
        
        y_train_pred = self.pipeline[-1].predict(self.x_trans) 

        self.x_test_trans = self.pipeline[:-1].transform(self.X_test)

        if self.task == 'regression':
            y_train_pred = np.exp(y_train_pred)
            ##### Required for overflow with underfitting models such as linear regression.
            y_train_pred = np.where((y_train_pred / self.y_train.max()) > 10000, self.y_train.max() * 10000, y_train_pred)

            
        print('Training Complete')
        
        
        self.experiment_training_instances = self.X_train.shape[0]
        self.experiment_input_dimensions = self.x_featurized.shape[1]
        self.experiment_test_instances = self.X_test.shape[0]
        self.experiment_transformed_input_size = self.x_trans.shape[1]
        self.y_train_pred = y_train_pred
        self.preprocessing_output_shape = self.x_trans.shape
    
    def translate_imputer(self, limit_factor = -1, source_table = 'test', source_table_feature_names = None):
        """Translate a pipeline's simple imputer to a postgres sql query

        Args:
            limit_factor (int, optional): maximum number of records to process. If -1 process all records in relation. Defaults to -1.
            source_table (str, optional): table to process the data from. Defaults to 'test'.
            source_table_feature_names (list, optional): list of names (in order) of the source table. Defaults to None.
        """    

        conn, cur = self.create_postgres_connection()

        impute_array = self.pipeline.named_steps['imputer'].statistics_

        if not source_table_feature_names:
            normalized_feature_names = [self.text_normalizer(feature.lower()) for feature in self.experiment_feature_names]
        else:
            normalized_feature_names = [self.text_normalizer(feature.lower()) for feature in source_table_feature_names]
        
        if source_table == 'test':
            table_name = self.experiment_name + '_' + source_table
            return_type = self.experiment_name + '_' + source_table
        elif source_table == 'featurized':
            table_name = self.experiment_name + '_' + source_table
            return_type = self.experiment_name + '_' + source_table
        else:
            table_name = self.input_name + '_' + source_table
            return_type = self.input_name + '_' + source_table
        function_name = self.experiment_name + '_' + self.model_name + '_impute_set'

        
        selection_statement = 'SELECT ROW_ID, '
        for idf, feature_name in enumerate(normalized_feature_names):
    
            imputation_value = impute_array[idf]

            # imputation_statement = 'CASE WHEN ' + feature_name + """='NaN' or """  + feature_name + ' IS NULL THEN '  + str(imputation_value) + ' ELSE ' + feature_name + ' END'
            imputation_statement = 'CASE WHEN ' + feature_name + ' IS NULL THEN '  + str(imputation_value) + ' ELSE ' + feature_name + ' END'

                
            if idf < self.experiment_input_dimensions - 1:
                selection_statement += imputation_statement + ', '
            else:
                if source_table == 'test':
                    selection_statement += imputation_statement + ', ' + 'target'
                else:
                    selection_statement += imputation_statement
        
        if limit_factor > 0:
            source_table = sql.SQL("""{section}.{table_name} ORDER BY ROW_ID ASC LIMIT {limit_factor}""").format(table_name = sql.SQL(table_name)
                                                                                                            , limit_factor = sql.SQL(str(limit_factor))
                                                                                                            , section = sql.SQL(self.section)
                                                                                                            )
        else:
            source_table = sql.SQL("""{section}.{table_name} ORDER BY ROW_ID ASC""").format(table_name = sql.SQL(table_name) 
                                                                                            , section = sql.SQL(self.section))
            
        self.create_measure_function('impute_set')

        self.imputation_query = sql.SQL("""{selection_statement} FROM {source_table}""").format(source_table = source_table, selection_statement = sql.SQL(selection_statement))

        function_statement = sql.SQL(
                            """
                                DROP FUNCTION IF EXISTS {section}.{function_name} CASCADE;
                                CREATE OR REPLACE FUNCTION {section}.{function_name}() RETURNS SETOF {section}.{return_type} AS
                                $func$
                                    {selection_statement} FROM {source_table};
                                $func$
                                LANGUAGE SQL STABLE PARALLEL SAFE;
                            """).format(function_name = sql.Identifier(function_name)
                                        , return_type = sql.Identifier(return_type)
                                        , source_table = source_table
                                        , selection_statement = sql.SQL(selection_statement)
                                        , section = sql.SQL(self.section)
                                        )
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_imputed;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_imputed AS 
                                        
                                        SELECT * FROM  {section}.{function_name}();
                                        
                                    """).format(input_name = sql.SQL(self.input_name)
                                                , limit_factor = sql.SQL(str(limit_factor))
                                                , function_name = sql.Identifier(function_name)
                                                , section = sql.SQL(self.section)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

    
    def translate_column_transfomer(self, limit_factor = -1, source_table = 'imputed', source_table_feature_names=None):
        """translates a pipeline's column transformer into a postgres SQL query

        Args:
            limit_factor (int, optional): maximum number of records to process. If -1 process all records in relation. Defaults to -1.
            source_table (str, optional): table to process the data from. Defaults to 'test'.
            source_table_feature_names (list, optional): list of names (in order) of the source table. Defaults to None.
        """        """"""

        conn, cur = self.create_postgres_connection()

        transformers = [(i[0], i[1], i[2]) for i in self.pipeline.named_steps['column_transformer'].transformers_]

        function_name = self.experiment_name + '_' + self.model_name + '_encode_scale_set'
        return_type = 'NUMERIC[]'
        if source_table == 'test':
            table_name = self.experiment_name + '_' + source_table
        elif source_table == 'featurized':
            table_name = self.experiment_name + '_' + source_table
        else:
            table_name = self.input_name + '_' + source_table

        if limit_factor > 0:
            source_table = sql.SQL("""{section}.{table_name} ORDER BY ROW_ID ASC LIMIT {limit_factor}""").format(table_name = sql.SQL(table_name)
                                                                                                            , limit_factor = sql.SQL(str(limit_factor))
                                                                                                            , section = sql.SQL(self.section)
                                                                                                            )
        else:
            source_table = sql.SQL("""{section}.{table_name} ORDER BY ROW_ID ASC""").format(table_name = sql.SQL(table_name)
                                                                                            , section = sql.SQL(self.section)
                                                                                            )

        select_statement = 'SELECT ARRAY['
        select_statement_relational = 'row_id, '
        number_of_features = 0
        for idt, transformation in enumerate(transformers):

            if transformation[0] == 'num' and not transformation[2]:
                continue
            elif transformation[0] == 'num' and transformation[2]:

                robust_scaler = transformation[1].named_steps['scaler']
                centers = robust_scaler.center_
                scale = robust_scaler.scale_

                if source_table_feature_names:
                    feature_names = source_table_feature_names
                else:
                    feature_names = [self.text_normalizer(self.experiment_feature_names[i].lower()) for i in transformation[2]]

                for idf, feature in enumerate(feature_names):
                    
                    if len(transformers) > 1:
                        select_statement += '(' + feature + '-(' + str(centers[idf]) + '))' + '/' + str(scale[idf]) + ', '
                        select_statement_relational += '(' + feature + '-(' + str(centers[idf]) + '))' + '/' + str(scale[idf]) + ' as f_' + str(idf) + ', '
                    elif len(transformers) == 1 and idf < len(feature_names) - 1:
                        select_statement += '(' + feature + '-(' + str(centers[idf]) + '))' + '/' + str(scale[idf]) + ', '
                        select_statement_relational += '(' + feature + '-(' + str(centers[idf]) + '))' + '/' + str(scale[idf]) + ' as f_' + str(idf) + ', '
                    elif len(transformers) == 1 and idf == len(feature_names) - 1: 
                        select_statement += '(' + feature + '-(' + str(centers[idf]) + '))' + '/' + str(scale[idf]) + ']'
                        select_statement_relational += '(' + feature + '-(' + str(centers[idf]) + '))' + '/' + str(scale[idf]) + ' as f_' + str(idf)
                    
                    number_of_features += 1
                        
            elif transformation[0] == 'cat':

                encoder = transformation[1].named_steps['encoder']
                categories = encoder.categories_
                
                feature_names = [self.text_normalizer(self.experiment_feature_names[i].lower()) for i in transformation[2]]
                for idf, feature in enumerate(feature_names): 
                    for idc, category in enumerate(categories[idf]):
                        if idf == len(feature_names) - 1 and idc == len(categories[idf]) - 1:
                            select_statement += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END] '
                            select_statement_relational += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END as f_' + str(number_of_features)
                        else:
                            select_statement += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END, '
                            select_statement_relational += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END as f_' + str(number_of_features) + ', ' 
                        number_of_features += 1
            
            elif transformation[0] == 'remainder':

                if source_table_feature_names:
                    feature_names = source_table_feature_names
                else:
                    feature_names = [self.text_normalizer(self.experiment_feature_names[i].lower()) for i in transformation[2]]

                for idf, feature in enumerate(feature_names):
                    if idf < len(feature_names) - 1:
                        select_statement += feature + ', '
                        select_statement_relational += feature + ' as f_' + str(number_of_features) + ', '
                    elif idf == len(feature_names) - 1: 
                        select_statement += feature + ']'
                        select_statement_relational += feature + ' as f_' + str(number_of_features)
                    
                    number_of_features += 1
            
        function_statement_vector = sql.SQL(
                        """
                            DROP FUNCTION IF EXISTS {section}.{function_name} CASCADE;
                            CREATE OR REPLACE FUNCTION {section}.{function_name}() RETURNS SETOF {return_type} AS
                            $func$
                                {select_statement} FROM {source_table};
                            $func$
                            LANGUAGE SQL STABLE PARALLEL SAFE;
                        """).format(function_name = sql.Identifier(function_name)
                                    , return_type = sql.SQL(return_type)
                                    , source_table = source_table
                                    , select_statement = sql.SQL(select_statement)
                                    , section = sql.SQL(self.section)
                                    )
        
        function_statement_relational = sql.SQL(
                        """
                            DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_encoded_relational;
                            CREATE MATERIALIZED VIEW {section}.{input_name}_encoded_relational AS 

                            SELECT {select_statement_relational}
                            FROM {source_table};
                        """).format(input_name = sql.SQL(self.input_name)
                                    , experiment_name = sql.SQL(self.experiment_name)
                                    , select_statement_relational = sql.SQL(select_statement_relational)
                                    , section = sql.SQL(self.section)
                                    , source_table = source_table
                                    )
        
        ### Create function
        cur.execute(function_statement_vector)
        conn.commit()

        cur.execute(function_statement_relational)
        conn.commit()

        self.create_measure_function('encode_scale_set')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_encoded;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_encoded AS 
                                        
                                        SELECT m.* as m FROM  {section}.{function_name}() as m;
                                        
                                    """).format(input_name=sql.SQL(self.input_name)
                                                , function_name = sql.Identifier(function_name)
                                                , section = sql.SQL(self.section)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

    def create_featurizer(self):
        """creates SQL featurizer query in postgres
        """        

        conn, cur = self.create_postgres_connection()

        table_creation_query = sql.SQL("""
                                            DROP TABLE IF EXISTS {section}.{experiment_name}_featurized CASCADE;
                                            CREATE TABLE {section}.{experiment_name}_featurized AS
                                       
                                            {featurizer_query}
                                        """).format(featurizer_query = self.featurizer_query,
                                                    section = sql.SQL(self.section),
                                                    experiment_name = sql.SQL(self.experiment_name)
                                                    )

        cur.execute(table_creation_query)

        conn.commit()

class InferDB(Transpiler):
    """Child class for InferDB

    Args:
        Transpiler (Transpiler): Transpiler
    """

    def __init__(self, X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, populate_paths=False, featurize_inferdb=False, pipeline=None, inferdb_columns=None, feature_names=None, section='pgml'):
        """Constructor method for InferDB

        Args:
            X_train (np.array or pd.DataFrame): array or dataframe containing the train data
            X_test (np.array or pd.DataFrame): array or dataframe containing the test data
            y_train (np.array): array containing the train true values
            y_test (np.array): array containing the test true values
            experiment_name (string): name of the experiment
            task (string): 
                - 'classification' for binary classification
                - 'regression' for regression
                - 'multi-class' for multi-label classification
            has_featurizer (bool): boolean indicator for featurizer at the 0 position (pipeline[0]) of the pipeline steps.
            featurize_inferdb (bool, optional): If true, featurizer should implement method 'featurize_for_inferdb' which transforms the input data and returns a set
                                                of defined features for InferDB . Defaults to False.
            pipeline (Scikitlearn Pipeline, optional): Pipeline to train input data. Transforms input data into predictions. Defaults to None.
            inferdb_columns (list, optional): List containing a subset of columns for InferDB. Defaults to None.
            feature_names (list, optional): Ordered list containing the names for each column in the input data. Defaults to None.
            section (string, optional): Schema prefix for postgres. Defaults to None.
        """

        super().__init__(X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb, pipeline, inferdb_columns, feature_names, section)
        self.populate_paths = populate_paths
    
    def get_kv_tuples(self, cat_mask, with_pred=True):
        """Creates kv tuples

        Args:
            cat_mask (list): list containing the indices for categorical features in the input data
            with_pred (bool, optional): if True aggregates predictions from pipeline. If false aggregates true values from y_train. Defaults to True.

        Returns:
            array: array containing the kv pairs
        """

        encoder = Encoder(self.task)

        if with_pred:
            encoder.fit(self.x_featurized_inferdb, self.y_train_pred, cat_mask)
        else:
            encoder.fit(self.x_featurized_inferdb, self.y_train, cat_mask)

        start = time.time()
        encoded_training_set = encoder.transform_dataset(self.x_featurized_inferdb, [i for i in range(self.x_featurized_inferdb.shape[1])])
        end = time.time() - start

        if with_pred: 
            my_problem = Problem(encoded_training_set, self.y_train_pred, encoder.num_bins, self.task, 1)
        else:
            my_problem = Problem(encoded_training_set, self.y_train, encoder.num_bins, self.task, 1)

        self.encoding_time = end
        
        start = time.time()
        my_problem.set_costs()
        print('Costs set')
        my_optimizer = Optimizer(my_problem, 1)
        my_optimizer.greedy_search()
        end = time.time() - start
        self.solution_time = end
        encoded_training_set = encoded_training_set[:, my_optimizer.greedy_solution]
        print('Greedy Search Complete, index:' + str(my_optimizer.greedy_solution))

        if self.populate_paths:
            training_tuples = [tuple(x) for x in encoded_training_set]
            bin_set = []
            for feature in my_optimizer.greedy_solution:
                bin_set.append([i for i in range(self.encoder.num_bins[feature]+1)])
            combinations_set = product(*bin_set)
            encoded_pipeline = deepcopy(self.pipeline)
            encoded_pipeline[-1].fit(encoded_training_set, self.y_train_pred)
            diff_set = set(combinations_set) - set(training_tuples)

            y_pred_diff = encoded_pipeline[-1].predict(np.array(list(diff_set)))

        if with_pred:
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
        else:
            training_output = []
            for idt, t in enumerate(encoded_training_set):
                key = ''
                for idf, feature in enumerate(t):
                    if idf < len(t) - 1:
                        key = key + str(int(feature)) + '.'
                    else:
                        key = key + str(int(feature))
                    # key = key + str(feature)
                training_output.append((key, self.y_train[idt]))

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
        if self.task == 'multi-class':

            k_v_aggregates = k_v_frame.groupby(['key', 'value'], as_index=False).size()
            idx = k_v_aggregates.groupby(['key'])['size'].idxmax()
            agg_df = k_v_aggregates.loc[idx]
            agg_df.drop(columns='size', inplace=True)
            tuples = agg_df.to_numpy()
        else:
            k_v_aggregates = k_v_frame.groupby(['key']).mean()
            k_v_aggregates.reset_index(drop=False, inplace=True)
            tuples = k_v_aggregates.to_numpy()

        self.solution = my_optimizer.greedy_solution
        self.encoder = encoder
        self.max_path_length = len(self.solution)

        return tuples

    def create_kv_table(self, table_name):
        """Creates new relation in postgres for the kv pairs

        Args:
            table_name (string): name of the relation
        """
        conn, cur = self.create_postgres_connection()
        ## Creates a table where key, predictions will be stored
        cur.execute(sql.SQL('DROP TABLE IF EXISTS {section}.{table_name} CASCADE; CREATE TABLE {section}.{table_name} (key TEXT NOT NULL, value NUMERIC NOT NULL)').format(table_name=sql.Identifier(table_name)
                                                                                                                                                                           , section = sql.SQL(self.section)
                                                                                                                                                                           ))
        conn.commit()

    def insert_tuples_in_kv_table(self,table_name, cat_mask=[] , with_pred=True):
        """Inserts kv tuples in the kv relation

        Args:
            table_name (string): kv relation name
            cat_mask (list, optional): list containing the indices of categorical features in the input data. Defaults to [].
            with_pred (bool, optional): if True aggregates predictions from pipeline. If false aggregates true values from y_train.. Defaults to True.
        """
        conn, cur = self.create_postgres_connection()
        tuples = self.get_kv_tuples(cat_mask=cat_mask, with_pred=with_pred)

        buffer = StringIO()
        df = pd.DataFrame(tuples)
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY {section}.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name)
                                                                                                        , section = sql.SQL(self.section)
                                                                                                        ), buffer)
        conn.commit()
        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")

    def create_index(self, index_name, table_name):
        """Creates an index in Postgres on top of the kv table

        Args:
            index_name (string): Index name
            table_name (string): table name to build the index on top of
        """
        conn, cur = self.create_postgres_connection()
        ## Creates an index for the keys using spgist
        cur.execute(sql.SQL('DROP INDEX IF EXISTS {index_name} CASCADE;CREATE INDEX {index_name} ON {section}.{table_name} using spgist(key); analyze;').format(index_name = sql.Identifier(index_name)
                                                                                                                                                                , table_name = sql.Identifier(table_name)
                                                                                                                                                                , section = sql.SQL(self.section)
                                                                                                                                                                ))
        conn.commit()
    
    def create_scoring_function_kv(self):
        """Creates scorer function for InferDB. This function creates a prediction for all datapoints in the test relation
        """        
        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_score_kv CASCADE;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_score_kv(OUT row_id INTEGER, OUT prediction NUMERIC) RETURNS SETOF record AS 
                                        $func$
                                        select k1.ROW_ID, case when k2.value is not null then k2.value else {section}.prefix_search(k1.key, '{input_name}_kv') end as prediction
                                        from {section}.{input_name}_translated k1
                                        left join {section}.{input_name}_kv k2
                                        on k1.key = k2.key
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    """).format(input_name = sql.SQL(self.input_name),
                                                name = sql.SQL(self.experiment_name)
                                                , section = sql.SQL(self.section)
                                                )
        
        ## Create function
        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_scored CASCADE;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_scored AS 
                                        
                                        SELECT * FROM  {section}.{input_name}_score_kv();
                                        
                                    """).format(
                                                input_name = sql.SQL(self.input_name)
                                                , section = sql.SQL(self.section)
                                                )
        cur.execute(persist_statement)
        conn.commit()

        self.create_measure_function('score_kv')

        if self.task in ('classification', 'multi-class'):

            self.create_acc_measure_function('score_kv', 'acc')
            self.create_acc_measure_function('score_kv', 'f1')
            self.create_acc_measure_function('score_kv', 'precision')
            self.create_acc_measure_function('score_kv', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score_kv', 'rmsle')
    
    def create_preproccesing_function(self, cat_mask = [], limit_factor=-1, featurize_query=None, table='test'):
        """Creates SQL query that contains the logic to transform values in the input data to keys in the embedded space

        Args:
            cat_mask (list, optional): list of indices for categorical features in the input data. Defaults to [].
            limit_factor (int, optional): number of records to process. If -1 all records are processed. Defaults to -1.
            featurize_query (string, optional): SQL query to featurize input data. Defaults to None.
            table (str, optional): name of the source table for preprocess the data from. Defaults to 'test'.
        """        


        conn, cur = self.create_postgres_connection()

        ### Creates a function to transform raw values into keys
        if featurize_query:
            feature_list = [(i, self.text_normalizer(self.experiment_feature_names[i])) for i in self.solution] 
        elif table == 'featurized' and self.inferdb_columns:
            feature_list = [(i, self.text_normalizer(self.inferdb_columns[i])) for i in self.solution]
        else:
            feature_list = [(i, self.text_normalizer(self.experiment_feature_names[i])) for i in self.solution]

        select_statement = """ SELECT ROW_ID, """
        for idf, feature in enumerate(feature_list):
            feature_index = feature[0]
            feature_name = feature[1]
            if feature_index not in cat_mask:
                ranges_list = list(self.encoder.bin_ranges[feature_index])
                for idr, range in enumerate(ranges_list):
                    if idf < len(feature_list) - 1:
                        if idr == 0:
                            if len(ranges_list) > 1:
                                select_statement += ' CASE WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """.'"""
                            else:
                                select_statement += ' CASE WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """.' ELSE """ + """'""" + str(idr + 1) + """.' END || """
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """.'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """.' ELSE """ + """'""" + str(idr + 1) + """.' END || """
                    else:
                        if idr == 0 and len(ranges_list) == 1:
                            select_statement += ' CASE WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """' ELSE """ + """'""" + str(idr + 1) + """' END """
                        elif idr == 0:
                            select_statement += ' CASE WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + ' <= ' + str(range) + ' THEN ' + """'""" + str(idr) + """' ELSE """ + """'""" + str(idr + 1) + """' END """
            else:
                ranges_list = self.encoder.embeddings[feature_index]
                embeddings = [[str(j) for j in i] for i in ranges_list]
                for idr, range in enumerate(embeddings):
                    range = [str(i) for i in range]
                    if idf < len(feature_list) - 1:
                        if idr == 0:
                            select_statement += ' CASE WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.' ELSE """ + """'""" + str(idr + 1) + """.' END || """ 
                    else:
                        if idr == 0:
                            select_statement += ' CASE WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """' ELSE """ + """'""" + str(idr + 1) + """' END """ 

        if featurize_query:
            trunc_f_query = featurize_query.as_string('conn').replace('SELECT', '')
            function_statement = sql.SQL("""
                    DROP FUNCTION IF EXISTS {section}.measure_featurizer_kv_runtime_{input_name} ;
                    CREATE OR REPLACE FUNCTION {section}.measure_featurizer_kv_runtime_{input_name}() returns NUMERIC AS $$

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
                    PERFORM {featurize_query};  -- your query here, replacing the outer SELECT with PERFORM
                    _end_ts   := clock_timestamp();
                    
                    -- RAISE NOTICE 'Timing overhead in ms = %', _overhead;
                    runtime := 1000 * (extract(epoch FROM _end_ts - _start_ts)) - _overhead;

                    RETURN runtime;
                    END;
                    $$ LANGUAGE plpgsql;
                """).format(input_name = sql.SQL(self.input_name)
                            , featurize_query = sql.SQL(trunc_f_query)
                            , section = sql.SQL(self.section)
                            )
            
            cur.execute(function_statement)
            conn.commit()

            persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{name}_featurized_kv;
                                        CREATE MATERIALIZED VIEW {section}.{name}_featurized_kv AS 
                                        
                                        {featurize_query};
                                        
                                    """).format(name=sql.SQL(self.experiment_name)
                                                , featurize_query = featurize_query
                                                , section = sql.SQL(self.section)
                                                )
        
            cur.execute(persist_statement)
            conn.commit()

            source_table = sql.SQL("{section}.{experiment_name}_featurized_kv").format(experiment_name = sql.SQL(self.experiment_name)
                                                                                       , section = sql.SQL(self.section))
        else:
            if limit_factor > 0:
                source_table = sql.SQL("{section}.{experiment_name}_{table} ORDER BY ROW_ID ASC LIMIT {limit_factor}").format(experiment_name = sql.SQL(self.experiment_name)
                                                                                                                    , limit_factor = sql.SQL(str(limit_factor))
                                                                                                                    , section = sql.SQL(self.section)
                                                                                                                    , table = sql.SQL(table)
                                                                                                                    )
            else:
                source_table = sql.SQL("{section}.{experiment_name}_{table} ORDER BY ROW_ID ASC").format(experiment_name = sql.SQL(self.experiment_name)
                                                                                                                    , limit_factor = sql.SQL(str(limit_factor))
                                                                                                                    , section = sql.SQL(self.section)
                                                                                                                    , table = sql.SQL(table)
                                                                                                                    )


        function_statement = sql.SQL(""" DROP FUNCTION IF EXISTS {section}.{input_name}_translate CASCADE ;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_translate(OUT ROW_ID INTEGER, OUT key TEXT) returns SETOF record AS 
                                        $func$
                                        
                                        {select_statement} 
                                        FROM {source_table}
                                        ;

                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    """).format(input_name = sql.SQL(self.input_name)
                                                , source_table = source_table
                                                , select_statement = sql.SQL(select_statement)
                                                , section = sql.SQL(self.section)
                                                )
                                                

        # print(function_statement)
        ## Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('translate')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_translated;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_translated AS 
                                        
                                        SELECT * FROM  {section}.{input_name}_translate();
                                        
                                    """).format(name=sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , limit_factor = sql.SQL(str(limit_factor))
                                                , section = sql.SQL(self.section)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()
    
    def create_solution(self, cat_mask=[], limit_factor=-1, featurize_query=None, with_pred=True, source_table='test'):
        """Creates kv relation and inserts kv tuples. Creates preprocessing and scoring functions.

        Args:
            cat_mask (list, optional): list of indices for categorical features in the input data. Defaults to [].
            limit_factor (int, optional): number of records to process. if -1 all records in the input data are processed. Defaults to -1.
            featurize_query (_type_, optional): _description_. Defaults to None.
            with_pred (bool, optional): If true aggregates predictions from pipeline. If false aggregates true values from y_train. Defaults to True.
        """

        kv_table_name = (self.input_name).lower() + '_' + 'kv'
        index_name = (self.input_name).lower() + '_' + 'index'

        self.create_kv_table(kv_table_name)
        self.insert_tuples_in_kv_table(kv_table_name, with_pred=with_pred)
        self.create_index(index_name, kv_table_name)
        self.create_preproccesing_function(cat_mask, limit_factor, featurize_query, source_table)
        self.create_scoring_function_kv()

        print('Index representation was created')

    def create_report_function_pg(self):
        """Creates function in postgres to summarize end-to-end performance of InferDB solution
        """        

        conn, cur = self.create_postgres_connection()

        if self.model.__class__.__name__ in ('MLPRegressor', 'MLPClassifier'):
            coef_statement = 'nn_matrix_' + self.experiment_name
        elif self.model.__class__.__name__ in ('LogisticRegression', 'LinearRegression'):
            coef_statement = self.input_name + '_coefficients'
        
        if self.has_featurizer:
            featurizer_statement = sql.SQL("""{section}.measure_featurizer_kv_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name)
                                                                                                                , section = sql.SQL(self.section)
                                                                                                                )
        else: 
            featurizer_statement = sql.SQL("""0""")
        
        query = sql.SQL(""" select max(row_id) from {section}.{input_name}_translated """).format(input_name = sql.SQL(self.input_name)
                                                                                                    , section = sql.SQL(self.section)
                                                                                                    )

        cur.execute(query)

        batch_size = cur.fetchall()[0][0] + 1

        if self.task in ('classification', 'multi-class'):
            out_statement = """, OUT acc NUMERIC
                                , OUT prec NUMERIC
                                , OUT recall NUMERIC
                                , OUT f1 NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            
            select_statement = sql.SQL(""", {section}.measure_acc_{input_name}_score_kv() as acc
                                                , {section}.measure_precision_{input_name}_score_kv() as precision
                                                , {section}.measure_recall_{input_name}_score_kv() as recall
                                                , {section}.measure_f1_{input_name}_score_kv() as f1
                                                , {featurizer_statement}
                                                , {section}.measure_translate_runtime_{input_name}()
                                                , {section}.measure_score_kv_runtime_{input_name}()
                                                , {batch_size} 
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , section = sql.SQL(self.section)
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        )
        
        
        elif self.task == 'regression':
            out_statement = """, OUT rmsle NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            select_statement = sql.SQL(""", {section}.measure_rmsle_{input_name}_score_kv() as rmsle
                                                , {featurizer_statement}
                                                , {section}.measure_translate_runtime_{input_name}()
                                                , {section}.measure_score_kv_runtime_{input_name}()
                                                , {batch_size} 
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , section = sql.SQL(self.section)
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        )

        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_inferdb_report;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_inferdb_report(
                                                                                        OUT method TEXT
                                                                                        , OUT size NUMERIC
                                                                                        {out_statement}
                                                                                    ) returns SETOF record AS
                                        $$

                                        select 'InferDB' as method
                                            , pg_total_relation_size('{section}.{input_name}_kv')
                                            {select_statement}

                                        
                                        $$ LANGUAGE SQL STABLE;
                                        """
                                        ).format(input_name = sql.SQL(self.input_name)
                                                , out_statement = sql.SQL(out_statement)
                                                , select_statement  = select_statement
                                                , section = sql.SQL(self.section) 
                                            )
        ### Create function
        cur.execute(function_statement)
        conn.commit()

    def create_report_pg(self, cat_mask=[], batch_size=-1, featurize_query=None, with_pred=True, iterations=5, source_table='test'):
        """Creates summary dataframe containing performance numbers for InferDB solution

        Args:
            cat_mask (list, optional): list containing indices for categorical features in input data. Defaults to [].
            batch_size (int, optional): batch size to process. If -1 all records in the input data are processed. Defaults to -1.
            featurize_query (_type_, optional): _description_. Defaults to None.
            with_pred (bool, optional): If true aggregates pipeline's predictions. If false, aggregates y_train values. Defaults to True.

        Returns:
            DataFrame: pandas DataFrame containing performance numbers for InferDB solution
        """        

        DEC2FLOAT = new_type(
        DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
        register_type(DEC2FLOAT)

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.experiment_name + '_' + model_name

        self.create_solution(cat_mask, batch_size, featurize_query, with_pred, source_table)
        self.create_report_function_pg()

        conn, cur = self.create_postgres_connection()

        query = sql.SQL(""" select * from {section}.{input_name}_inferdb_report()""").format(input_name = sql.SQL(input_name)
                                                                                             , section = sql.SQL(self.section)
                                                                                             )
        df = pd.DataFrame()

        for i in range(iterations):
            cur.execute(query)

            result = cur.fetchall()

            results = []
            for row in result:
                results.append(list(row))
            
            if self.task == 'regression':
                columns = ['Solution', 'Size (B)', 'RMSLE', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
                # columns = ['Solution', 'Size (B)', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
            else:
                columns = ['Solution', 'Size (B)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
                # columns = ['Solution', 'Size (B)', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']

            summary_df = pd.DataFrame(results, columns=columns)
            summary_df['End-to-End Latency (ms)'] = summary_df.apply(lambda x: x['Impute Featurize Latency (ms)'] + x['Encode Scale Latency (ms)'] + x['Score Latency (ms)'], axis=1)
            summary_df['Experiment'] = self.experiment_name
            summary_df['Algorithm'] = model_name
            summary_df['Iteration'] = i
        
            df = pd.concat((df, summary_df))

        return df

class SQLmodel(Transpiler):
    """Child class for SQL transpiler

    Args:
        Transpiler (Transpiler): _description_
    """

    def __init__(self, X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb=False, pipeline=None, inferdb_columns=None, feature_names=None, section='pgml', featurize_query=None):
        """Constructor method for SQL transpiler

        Args:
            X_train (np.array or pd.DataFrame): array or dataframe containing the train data
            X_test (np.array or pd.DataFrame): array or dataframe containing the test data
            y_train (np.array): array containing the train true values
            y_test (np.array): array containing the test true values
            experiment_name (string): name of the experiment
            task (string): 
                - 'classification' for binary classification
                - 'regression' for regression
                - 'multi-class' for multi-label classification
            has_featurizer (bool): boolean indicator for featurizer at the 0 position (pipeline[0]) of the pipeline steps.
            featurize_inferdb (bool, optional): If true, featurizer should implement method 'featurize_for_inferdb' which transforms the input data and returns a set
                                                of defined features for InferDB . Defaults to False.
            pipeline (Scikitlearn Pipeline, optional): Pipeline to train input data. Transforms input data into predictions. Defaults to None.
            inferdb_columns (list, optional): List containing a subset of columns for InferDB. Defaults to None.
            feature_names (list, optional): Ordered list containing the names for each column in the input data. Defaults to None.
            section (string, optional): Schema prefix for postgres. Defaults to 'pgml'.
            featurize_query (Composition, optional): SQL query to featurize the input data. Defaults to None.
        """

        super().__init__(X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb, pipeline, inferdb_columns, feature_names, section, featurize_query)

        self.create_aux_functions()

    def create_preprocessing_pipeline(self, source_table = 'encoded'):
        """Creates a preprocessing function in postgres

        Args:
            source_table (str, optional): name of the relation to process the data from. Defaults to 'encoded'.
        """        

        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_preprocess CASCADE;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_preprocess(OUT row_id INTEGER, OUT col_id INTEGER, OUT val NUMERIC) RETURNS SETOF record AS
                                        $func$
                                        with d as (select row_number() over () as id, m.* from {section}.{input_name}_{source_table} as m)

                                        SELECT m.id - 1 as row_id, u.ord - 1 as col_id, u.val
                                        FROM   d m,
                                            LATERAL unnest(m.m) WITH ORDINALITY AS u(val, ord)
                                        where u.val != 0
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;

                                    """).format(input_name = sql.SQL(self.input_name)
                                                , source_table = sql.SQL(source_table)
                                                , section = sql.SQL(self.section)
                                                )
        ### Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('preprocess')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_preprocessed;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_preprocessed AS 
                                        
                                        SELECT * FROM  {section}.{input_name}_preprocess();
                                        
                                    """).format(
                                                input_name = sql.SQL(self.input_name)
                                                , section = sql.SQL(self.section)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()
    
    def create_coef_table(self):
        """Creates a relation containing linear/logistic regression coefficients
        """        

        conn, cur = self.create_postgres_connection()

        if self.task in ('regression', 'classification'):
        
            table_statement = sql.SQL("""
                                    DROP TABLE IF EXISTS {section}.{input_name}_coefficients CASCADE;
                                    CREATE TABLE {section}.{input_name}_coefficients
                                    (col_id INTEGER
                                    , val NUMERIC
                                    , intercept NUMERIC
                                    , PRIMARY KEY(col_id)
                                    ); 
                                """).format(input_name = sql.SQL(self.input_name)
                                            , section = sql.SQL(self.section)
                                            )
        elif self.task == 'multi-class':

            table_statement = sql.SQL("""
                                    DROP TABLE IF EXISTS {section}.{input_name}_coefficients CASCADE;
                                    CREATE TABLE {section}.{input_name}_coefficients
                                    (
                                    class_id INTEGER
                                    , col_id INTEGER
                                    , val NUMERIC
                                    , intercept NUMERIC
                                    , PRIMARY KEY(class_id, col_id)
                                    ); 
                                """).format(input_name = sql.SQL(self.input_name)
                                            , section = sql.SQL(self.section)
                                            )



        ### Create function
        cur.execute(table_statement)
        conn.commit()
        
        ##### Insert coefficients and bias in the nn table

        if self.task in ('regression', 'classification'):
            insert_statement = 'INSERT INTO ' + str(self.section) + '.' + self.input_name + '_coefficients (col_id, val, intercept) VALUES '
        else:
            insert_statement = 'INSERT INTO ' + str(self.section) + '.' + self.input_name + '_coefficients (class_id, col_id, val, intercept) VALUES '
        for idc, c in enumerate(self.pipeline[-1].coef_):

            if self.task == 'regression':
                tup = (idc, c, self.pipeline[-1].intercept_)
                if idc < len(self.pipeline[-1].coef_) - 1:
                    insert_statement += str(tup) + ', '
                else:
                    insert_statement += str(tup)
            elif self.task == 'classification':
                for idx, x in enumerate(c):
                    tup = (idx, x, self.pipeline[-1].intercept_[0])
                    if idx < len(c) - 1:
                        insert_statement += str(tup) + ', '
                    else:
                        insert_statement += str(tup)
            elif self.task == 'multi-class':
                for idx, x in enumerate(c):
                    tup = (idc, idx, x, self.pipeline[-1].intercept_[idc])
                    if idx == len(c) - 1 and idc == len(self.pipeline[-1].coef_) - 1:
                        insert_statement += str(tup)
                    else:
                        insert_statement += str(tup) + ','

        ### Create function
        cur.execute(insert_statement)
        conn.commit()

    def create_scorer_function_lr(self):
        """Creates a function in postgres that scores all datapoints in the input data using a linear model
        """        

        conn, cur = self.create_postgres_connection()

        ### Creates a function to transform preprocessed arrays into predictions

        if self.task == 'classification':
            select_statement = '1/(1 + pgml.crazy_exp(-(sum(pre.val * b.val) + b.intercept))) as prediction'
        elif self.task == 'regression':
            select_statement = 'pgml.crazy_exp(sum(pre.val * b.val) + b.intercept) as prediction'
        
        if self.task in ('regression', 'classification'):

            function_statement = sql.SQL("""DROP FUNCTION IF EXISTS {section}.{input_name}_score CASCADE;
                                            CREATE OR REPLACE FUNCTION {section}.{input_name}_score(OUT row_id INTEGER, OUT prediction NUMERIC)
                                            RETURNS SETOF record 
                                            AS $func$

                                            with pre as (select * from {section}.{input_name}_preprocessed)

                                            select pre.row_id, {select_statement}
                                            from pre
                                            left join {section}.{input_name}_coefficients b
                                            on pre.col_id = b.col_id
                                            group by 1, intercept

                                            $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                        """).format(experiment_name = sql.SQL(self.experiment_name)
                                                    , input_name = sql.SQL(self.input_name)
                                                    , select_statement = sql.SQL(select_statement)
                                                    , section = sql.SQL(self.section)
                                                    )     
        elif self.task == 'multi-class':

              function_statement = sql.SQL("""DROP FUNCTION IF EXISTS {section}.{input_name}_score CASCADE;
                                            CREATE OR REPLACE FUNCTION {section}.{input_name}_score(OUT row_id INTEGER, OUT prediction NUMERIC)
                                            RETURNS SETOF record 
                                            AS $func$

                                            with pre as (select * from {section}.{input_name}_preprocessed),

                                            coef as (select pre.row_id, b.class_id, 1/(1 + pgml.crazy_exp(-(sum(pre.val * b.val) + b.intercept))) as pred
                                                    from pre
                                                    left join {section}.{input_name}_coefficients b
                                                    on pre.col_id = b.col_id
                                                    group by 1, 2, b.intercept),
                                           
                                           rank as(

                                            select row_id, class_id, RANK () OVER ( PARTITION BY row_id
                                                                                            ORDER BY pred DESC
                                                                                        ) rank
                                            from coef
                                           )
            

                                            select row_id, class_id as prediction
                                            from rank
                                            where rank=1;
                                            $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                        """).format(experiment_name = sql.SQL(self.experiment_name)
                                                    , input_name = sql.SQL(self.input_name)
                                                    , section = sql.SQL(self.section)
                                                    ) 

        ### Create function
        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_scored CASCADE;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_scored AS 
                                        
                                        SELECT * FROM  {section}.{input_name}_score();
                                        
                                    """).format(
                                                input_name = sql.SQL(self.input_name)
                                                , section = sql.SQL(self.section)
                                                )
        cur.execute(persist_statement)
        conn.commit()

        self.create_measure_function('score')
        if self.task in ('classification', 'multi-class'):
            self.create_acc_measure_function('score', 'acc')
            self.create_acc_measure_function('score', 'f1')
            self.create_acc_measure_function('score', 'precision')
            self.create_acc_measure_function('score', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score', 'rmsle')

    def create_nn_table(self):
        """Creates a relation in postgres holding the layers and weights of a neural network
        """        

        conn, cur = self.create_postgres_connection()

        function_statement = 'DROP TABLE IF EXISTS ' + str(self.section) + '.' + 'nn_matrix_' + self.experiment_name + '; '
        function_statement += """ CREATE TABLE """ + self.section + '.' + """nn_matrix_""" + self.experiment_name
        function_statement += """ (id  INTEGER,
                                row INTEGER,
                                col INTEGER,
                                val NUMERIC,
                                bias NUMERIC,
                                PRIMARY KEY(row, col, id)
                                ); 
                            """
        cur.execute(function_statement)
        conn.commit()
        
        ##### Insert coefficients and bias in the nn table

        insert_statement = """ INSERT INTO """ + self.section + """.nn_matrix_""" + self.experiment_name + """ (id, row, col, val, bias) VALUES """
        for idc, c in enumerate(self.pipeline[-1].coefs_):
            for idx, weights in enumerate(c):
                for idw, w in enumerate(weights):
                    tup = (idc, idx, idw, w, self.pipeline[-1].intercepts_[idc][idw])

                    if idc == len (self.pipeline[-1].coefs_) - 1 and idx == len(c) - 1 and idw == len(weights) - 1:
                        insert_statement += str(tup)
                    else:
                        insert_statement += str(tup) + ','
        
        cur.execute(insert_statement)
        conn.commit()
        
        # return insert_statement

    def create_nn_scorer_function(self):
        """creates a function in postgres that scores all datapoints in the input data using a neural network model relation
        """        

        conn, cur = self.create_postgres_connection()
        ### Creates a function to transform preprocessed arrays into predictions

        if self.task in ('classification', 'multi-class'):
            select_statement = 'GREATEST(0, sum(m1.val * nn2.val) + bias) as prediction'
        elif self.task == 'regression':
            select_statement = 'pgml.crazy_exp(GREATEST(0,sum(m1.val * nn2.val) + bias)) as prediction'

        if self.task in ('classification', 'regression'):

            output_statement = sql.SQL("""
                                            select m1.row_id, {select_statement}
                                            from input_weights m1
                                            join (select * from {section}.nn_matrix_{experiment_name} where id=1) nn2
                                                on m1.col = nn2.row
                                            group by 1, bias
                                        """
                                    ).format(experiment_name = sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , select_statement = sql.SQL(select_statement)
                                                , section = sql.SQL(self.section)
                                                )
        else:
            output_statement = sql.SQL("""
                                        , output_weights as
                                        (
                                        select m1.row_id, nn2.col, {select_statement}
                                        from input_weights m1
                                        join (select * from {section}.nn_matrix_{experiment_name} where id=1) nn2
                                            on m1.col = nn2.row
                                        group by 1, 2, bias
                                        )
                                     
                                        , rank as(

                                            select row_id, col as class_id, RANK () OVER ( PARTITION BY row_id
                                                                                            ORDER BY prediction DESC
                                                                                        ) rank
                                            from output_weights
                                           )
                                                                        
                                        select row_id, class_id as prediction
                                        from rank
                                        where rank=1;
                                        """
                                    ).format(experiment_name = sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , select_statement = sql.SQL(select_statement)
                                                , section = sql.SQL(self.section)
                                                )


        function_statement = sql.SQL("""DROP FUNCTION IF EXISTS {section}.{input_name}_score CASCADE;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_score(OUT row_id INT, OUT prediction NUMERIC) returns SETOF RECORD AS $func$
                                        
                                        WITH input_weights as
                                        (
                                        SELECT m1.row_id, m2.col, GREATEST(0, sum(m1.val * m2.val) + bias) AS val
                                        FROM   {section}.{input_name}_preprocessed m1
                                        join (select * from {section}.nn_matrix_{experiment_name} where id=0) m2
                                            on m1.col_id = m2.row
                                        GROUP BY m1.row_id, m2.col, bias
                                        )
                                     
                                        {output_statement}
                                     
                                     $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    
                                    """).format(experiment_name = sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , select_statement = sql.SQL(select_statement)
                                                , section = sql.SQL(self.section)
                                                , output_statement = output_statement
                                                )
        
        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_scored CASCADE;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_scored AS 
                                        
                                        SELECT * FROM  {section}.{input_name}_score();
                                        
                                    """).format(
                                                input_name = sql.SQL(self.input_name)
                                                , section = sql.SQL(self.section)
                                                )
        cur.execute(persist_statement)
        conn.commit()

        self.create_measure_function('score')
        if self.task in ('classification', 'multi-class'):
            self.create_acc_measure_function('score', 'acc')
            self.create_acc_measure_function('score', 'f1')
            self.create_acc_measure_function('score', 'precision')
            self.create_acc_measure_function('score', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score', 'rmsle')
    
    def create_sql_representation(self):
        """Creates SQL artifacts for a given model
        """        

        if self.model.__class__.__name__ in ('MLPRegressor', 'MLPClassifier'):
            self.create_nn_table()
            self.create_nn_scorer_function()
        elif self.model.__class__.__name__ in ('LinearRegression', 'LogisticRegression'):
            self.create_coef_table()
            self.create_scorer_function_lr()

        print('Model representation was created')
    
    def create_report_function_pg(self):
        """Creates a function in postgres to summarize the model's performance
        """        

        conn, cur = self.create_postgres_connection()

        if self.model.__class__.__name__ in ('MLPRegressor', 'MLPClassifier'):
            coef_statement = 'nn_matrix_' + self.experiment_name
        elif self.model.__class__.__name__ in ('LogisticRegression', 'LinearRegression'):
            coef_statement = self.input_name + '_coefficients'
        
        if self.has_featurizer:
            featurizer_runtime = self.evaluate_runtime(self.featurizer_query)
            featurizer_statement = sql.SQL("""{featurizer_runtime}""").format(featurizer_runtime=sql.SQL(str(featurizer_runtime)))
        elif 'imputer' in list(self.pipeline.named_steps.keys()): 
            featurizer_statement = sql.SQL("""{section}.measure_impute_set_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name)
                                                                                                             , section = sql.SQL(self.section))
        else:
            featurizer_statement = sql.SQL("""0""")

        query = sql.SQL(""" select max(row_id) from {section}.{input_name}_preprocessed """).format(input_name = sql.SQL(self.input_name)
                                                                                                    , section = sql.SQL(self.section)
                                                                                                    )

        cur.execute(query)

        batch_size = cur.fetchall()[0][0] + 1

        if self.task in ('classification', 'multi-class'):
            out_statement = """, OUT acc NUMERIC
                                , OUT prec NUMERIC
                                , OUT recall NUMERIC
                                , OUT f1 NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            select_statement = sql.SQL(""", {section}.measure_acc_{input_name}_score() as acc
                                                , {section}.measure_precision_{input_name}_score() as precision
                                                , {section}.measure_recall_{input_name}_score() as recall
                                                , {section}.measure_f1_{input_name}_score() as f1
                                                , {featurizer_statement}
                                                , {section}.measure_encode_scale_set_runtime_{input_name}()
                                                , {section}.measure_score_runtime_{input_name}()
                                                , {batch_size}
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        , section = sql.SQL(self.section)
                                                        )
        
        
        elif self.task == 'regression':

            out_statement = """, OUT rmsle NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            select_statement = sql.SQL(""", {section}.measure_rmsle_{input_name}_score() as rmsle
                                                , {featurizer_statement}
                                                , {section}.measure_encode_scale_set_runtime_{input_name}()
                                                , {section}.measure_score_runtime_{input_name}()
                                                , {batch_size}
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        , section = sql.SQL(self.section)
                                                        )

        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_model_report;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_model_report(
                                                                                        OUT method TEXT
                                                                                        , OUT size NUMERIC
                                                                                        {out_statement}
                                                                                    ) returns SETOF record AS
                                        $$

                                        select 'SQL' as method
                                            , pg_total_relation_size('{section}.{coef_statement}') as size
                                            {select_statement}

                                        
                                        $$ LANGUAGE SQL STABLE;
                                        """
                                        ).format(input_name = sql.SQL(self.input_name)
                                                , out_statement = sql.SQL(out_statement)
                                                , select_statement  = select_statement
                                                , coef_statement = sql.SQL(coef_statement) 
                                                , section = sql.SQL(self.section)
                                            )
        ### Create function
        cur.execute(function_statement)
        conn.commit()

    def create_report_pg(self, iterations=5):
        """Creates SQL artifacts for a model and executes performance function

        Returns:
            DataFrame: Pandas DataFrame containing a summary of the performance numbers for the model
        """        

        DEC2FLOAT = new_type(
        DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
        register_type(DEC2FLOAT)

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.experiment_name + '_' + model_name

        self.create_sql_representation()
        self.create_report_function_pg()

        conn, cur = self.create_postgres_connection()

        query = sql.SQL(""" select * from {section}.{input_name}_model_report()""").format(input_name = sql.SQL(input_name)
                                                                                           , section = sql.SQL(self.section)
                                                                                           )

        df = pd.DataFrame()
        for i in range(iterations):
            cur.execute(query)

            result = cur.fetchall()

            results = []
            for row in result:
                results.append(list(row))
            
            if self.task == 'regression':
                columns = ['Solution', 'Size (B)', 'RMSLE', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
                # columns = ['Solution', 'Size (B)', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
            else:
                columns = ['Solution', 'Size (B)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
                # columns = ['Solution', 'Size (B)', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']

            summary_df = pd.DataFrame(results, columns=columns)
            summary_df['End-to-End Latency (ms)'] = summary_df.apply(lambda x: x['Impute Featurize Latency (ms)'] + x['Encode Scale Latency (ms)'] + x['Score Latency (ms)'], axis=1)
            summary_df['Experiment'] = self.experiment_name
            summary_df['Algorithm'] = model_name
            summary_df['Iteration'] = i
        
            df = pd.concat((df, summary_df))

        return df

class PGML(Transpiler):
    """Child class for the PGML transpiler 

    Args:
        Transpiler (Transpiler): _description_
    """

    def __init__(self, X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb=False, pipeline=None, inferdb_columns=None, feature_names=None, section='pgml', featurizer_query=None):
        """Constructor method for PGML transpiler

        Args:
            X_train (np.array or pd.DataFrame): array or dataframe containing the train data
            X_test (np.array or pd.DataFrame): array or dataframe containing the test data
            y_train (np.array): array containing the train true values
            y_test (np.array): array containing the test true values
            experiment_name (string): name of the experiment
            task (string): 
                - 'classification' for binary classification
                - 'regression' for regression
                - 'multi-class' for multi-label classification
            has_featurizer (bool): boolean indicator for featurizer at the 0 position (pipeline[0]) of the pipeline steps.
            featurize_inferdb (bool, optional): If true, featurizer should implement method 'featurize_for_inferdb' which transforms the input data and returns a set
                                                of defined features for InferDB . Defaults to False.
            pipeline (Scikitlearn Pipeline, optional): Pipeline to train input data. Transforms input data into predictions. Defaults to None.
            inferdb_columns (list, optional): List containing a subset of columns for InferDB. Defaults to None.
            feature_names (list, optional): Ordered list containing the names for each column in the input data. Defaults to None.
            section (string, optional): Schema prefix for postgres. Defaults to 'pgml'.
            featurize_query (Composition, optional): SQL query to featurize the input data. Defaults to None.
        """

        super().__init__(X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb, pipeline, inferdb_columns, feature_names, section, featurizer_query)
    
    def train_model_in_pgml(self, model, model_parameters=None):
        """Trains a model in postgres using PGML

        Args:
            model (str): model name
            model_parameters (str, optional): String containing model's parameters in PGML format. Defaults to None.
        """        

        conn, cur = self.create_postgres_connection()

        task = sql.SQL(self.task) if self.task in ('regression', 'classification') else sql.SQL('classification')

        if model_parameters:
        
            function_statement = sql.SQL(
                                            """
                                                SELECT {section}.train(
                                                project_name => '{input_name}'::text, 
                                                task => '{task}'::text, 
                                                relation_name => '{section}.{input_name}_train'::text,
                                                y_column_name => 'target'::text,
                                                algorithm => {model},
                                                hyperparams => {model_parameters}
                                            );
                                        
                                            """
                                        ).format(task = task
                                                , experiment_name = sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , model = sql.SQL(model)
                                                , model_parameters = sql.SQL(model_parameters)
                                                , section = sql.SQL(self.section)
                                                )
        else:
            function_statement = sql.SQL(
                                            """
                                                SELECT {section}.train(
                                                project_name => '{input_name}'::text, 
                                                task => '{task}'::text, 
                                                relation_name => '{section}.{input_name}_train'::text,
                                                y_column_name => 'target'::text,
                                                algorithm => {model}
                                            );
                                        
                                            """
                                        ).format(task = task
                                                , experiment_name = sql.SQL(self.experiment_name)
                                                , input_name = sql.SQL(self.input_name)
                                                , model = sql.SQL(model)
                                                , section = sql.SQL(self.section)
                                                )

        
        cur.execute(function_statement)
        conn.commit()
    
    def deploy_pgml_trained_model(self):
        """Deploys a trained model in postgres using PGML
        """        
        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL( """SELECT * FROM {section}.deploy(
                                    '{input_name}'::text,
                                    strategy => 'most_recent'
                                );""").format(input_name = sql.SQL(self.input_name)
                                              , section = sql.SQL(self.section)
                                              )
        
        cur.execute(function_statement)
        conn.commit()
    
    def create_scorer_function_pgml(self, source_table):
        """Creates a function in postgres that scores all points in the input data using a PGML deployed model 

        Args:
            source_table (str): name of the relation to score the records from
        """        

        if self.task == 'regression':
            prediction_statement = 'pgml.crazy_exp(prediction)'
        else:
            prediction_statement = 'prediction'

        function_statement = sql.SQL(""" 
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_score_pgml CASCADE;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_score_pgml(OUT row_id INT, OUT prediction NUMERIC) returns SETOF RECORD AS $func$
                                        WITH predictions AS (
                                        SELECT row_number() over () as row_id
                                                , {section}.predict_batch('{input_name}'::text, d.m) AS prediction
                                        FROM {section}.{input_name}_{source_table} d
                                    )
                                    SELECT row_id - 1 as row_id, {prediction_statement} FROM predictions
                                    $func$ LANGUAGE SQL STABLE;
                                    
                                    """).format(input_name = sql.SQL(self.input_name)
                                                , experiment_name = sql.SQL(self.experiment_name)
                                                , source_table = sql.SQL(source_table)
                                                , section = sql.SQL(self.section)
                                                , prediction_statement = sql.SQL(prediction_statement)
                                                )
        
        conn, cur = self.create_postgres_connection()

        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS {section}.{input_name}_scored CASCADE;
                                        CREATE MATERIALIZED VIEW {section}.{input_name}_scored AS 
                                        
                                        SELECT * FROM  {section}.{input_name}_score_pgml();
                                        
                                    """).format(
                                                input_name = sql.SQL(self.input_name)
                                                , section = sql.SQL(self.section)
                                                )
        cur.execute(persist_statement)
        conn.commit()

        self.create_measure_function('score_pgml')
        if self.task in ('classification', 'multi-class'):
            self.create_acc_measure_function('score_pgml', 'acc')
            self.create_acc_measure_function('score_pgml', 'f1')
            self.create_acc_measure_function('score_pgml', 'precision')
            self.create_acc_measure_function('score_pgml', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score_pgml', 'rmsle')
    
    def create_solution(self, model, model_parameters, source_table):
        """Creates PGML artifacts

        Args:
            model (str): model name
            model_parameters (str): string containing model parameters in PGML format
            source_table (str): name of the relation to create the scores from
        """        

        self.train_model_in_pgml(model, model_parameters)
        self.drop_train_table()
        self.deploy_pgml_trained_model()
        self.create_scorer_function_pgml(source_table)
        print('PGML model artifacts were created')
    
    def create_report_function_pg(self):
        """creates a function in postgres that summarizes the performance of a PGML solution
        """        

        conn, cur = self.create_postgres_connection()

        
        if self.has_featurizer:
            featurizer_runtime = self.evaluate_runtime(self.featurizer_query)
            featurizer_statement = sql.SQL("""{featurizer_runtime}""").format(featurizer_runtime=sql.SQL(str(featurizer_runtime)))
        elif 'imputer' in list(self.pipeline.named_steps.keys()): 
            featurizer_statement = sql.SQL("""{section}.measure_impute_set_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name)
                                                                                                             , section = sql.SQL(self.section)
                                                                                                             )
        else:
            featurizer_statement = sql.SQL("""0""")

        query = sql.SQL(""" select count(*) from {section}.{input_name}_encoded """).format(input_name = sql.SQL(self.input_name)
                                                                                            , section = sql.SQL(self.section)
                                                                                            )

        cur.execute(query)

        batch_size = cur.fetchall()[0][0]

        if self.task in ('classification', 'multi-class'):
            out_statement = """, OUT acc NUMERIC
                                , OUT prec NUMERIC
                                , OUT recall NUMERIC
                                , OUT f1 NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            select_statement_pgml = sql.SQL(""", {section}.measure_acc_{input_name}_score_pgml() as acc
                                                , {section}.measure_precision_{input_name}_score_pgml() as precision
                                                , {section}.measure_recall_{input_name}_score_pgml() as recall
                                                , {section}.measure_f1_{input_name}_score_pgml() as f1
                                                , {featurizer_statement}
                                                , {section}.measure_encode_scale_set_runtime_{input_name}()
                                                , {section}.measure_score_pgml_runtime_{input_name}()
                                                , {batch_size}
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        , section = sql.SQL(self.section)
                                                        )
        
        
        elif self.task == 'regression':
            out_statement = """, OUT rmsle NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            select_statement_pgml = sql.SQL(""", {section}.measure_rmsle_{input_name}_score_pgml() as rmsle
                                                , {featurizer_statement}
                                                , {section}.measure_encode_scale_set_runtime_{input_name}()
                                                , {section}.measure_score_pgml_runtime_{input_name}()
                                                , {batch_size}
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        , section = sql.SQL(self.section)
                                                        )
        
        #### Sizes
        current = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + self.input_name + '.joblib', "wb") as File:
            dump(self.pipeline, File)
        model_size = getsize(path + '/' + self.input_name + '.joblib') 

        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_pgml_report;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_pgml_report(
                                                                                        OUT method TEXT
                                                                                        , OUT size NUMERIC
                                                                                        {out_statement}
                                                                                    ) returns SETOF record AS
                                        $$

                                        select 'PGML' as method
                                                , {model_size} as size
                                                {select_statement_pgml};

                                        
                                        $$ LANGUAGE SQL STABLE;
                                        """
                                        ).format(input_name = sql.SQL(self.input_name)
                                                , out_statement = sql.SQL(out_statement)
                                                , select_statement_pgml  = select_statement_pgml
                                                , model_size = sql.SQL(str(model_size))  
                                                , section = sql.SQL(self.section)
                                            )
        ### Create function
        cur.execute(function_statement)
        conn.commit()

    def create_report_pg(self, model, model_parameters, source_table='encoded', featurize_query=None, iterations=5):
        """creates a summary of the performance numbers of a deployed PGML solution

        Args:
            model (str): model name
            model_parameters (str): string containing model parameters in the PGML format
            source_table (str, optional): name of the relation to score the records from. Defaults to 'encoded'.
            featurize_query (str, optional): SQL query to featuroze the input data. Defaults to None.

        Returns:
            _type_: _description_
        """        
        

        DEC2FLOAT = new_type(
        DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
        register_type(DEC2FLOAT)

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.experiment_name + '_' + model_name

        self.create_solution(model, model_parameters, source_table)
        self.create_report_function_pg()

        conn, cur = self.create_postgres_connection()

        query = sql.SQL(""" select * from {section}.{input_name}_pgml_report() """).format(input_name = sql.SQL(input_name), section=sql.SQL(self.section))
        
        df = pd.DataFrame()
        for i in range(iterations):

            cur.execute(query)

            result = cur.fetchall()

            results = []
            for row in result:
                results.append(list(row))
            
            if self.task == 'regression':
                columns = ['Solution', 'Size (B)', 'RMSLE', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']

                # columns = ['Solution', 'Size (B)', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
            else:
                columns = ['Solution', 'Size (B)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']

                # columns = ['Solution', 'Size (B)', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']

            summary_df = pd.DataFrame(results, columns=columns)
            summary_df['End-to-End Latency (ms)'] = summary_df.apply(lambda x: x['Impute Featurize Latency (ms)'] + x['Encode Scale Latency (ms)'] + x['Score Latency (ms)'], axis=1)
            summary_df['Experiment'] = self.experiment_name
            summary_df['Algorithm'] = model_name
            summary_df['Iteration'] = i
        
            df = pd.concat((df, summary_df))

        return df

class MADLIB(Transpiler):

    def __init__(self, X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb=False, pipeline=None, inferdb_columns=None, feature_names=None, section='madlib'):

        super().__init__(X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb, pipeline, inferdb_columns, feature_names, section)
    
    def create_train_table_mlp_knn_madlib(self):

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.text_normalizer(self.input_name + '_madlib_train').lower()
        initial_statement = 'DROP TABLE IF EXISTS madlib.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE madlib.' + table_name + ' AS '
            
        initial_statement += 'SELECT row_number() over () as id, ARRAY[ '
        for idf in range(self.preprocessing_output_shape[1]):
            
            if idf < self.preprocessing_output_shape[1] - 1:
                initial_statement += 'f_' + str(idf) + ', '
            else:
                initial_statement += 'f_' + str(idf) + '] as features, '
            
        initial_statement += 'target FROM madlib.' + self.text_normalizer(self.input_name + '_train').lower()

        ### Create table
        cur.execute(initial_statement)
        conn.commit()
    
    def create_test_table_mlp_knn_madlib(self, source_table):

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.text_normalizer(self.input_name + '_madlib_test').lower()
        initial_statement = 'DROP TABLE IF EXISTS madlib.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE madlib.' + table_name + ' AS '
            
        initial_statement += 'SELECT row_number() over () as id, m as features '
            
        initial_statement += 'FROM madlib.' + self.text_normalizer(self.input_name + '_' + source_table).lower()

        ### Create table
        cur.execute(initial_statement)
        conn.commit()

    def train_mlp_in_madlib(self, model_params, layer_size):

        conn, cur = self.create_postgres_connection()

        if self.task in ('classification', 'multi-class'):
            function = 'classification'
        else:
            function = 'regression'
        train_statement = sql.SQL(
                                        """

                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib, madlib.{input_name}_madlib_summary, madlib.{input_name}_madlib_standardization;
                                            SELECT madlib.mlp_{function}(
                                                                'madlib.{input_name}_madlib_train',      -- Source table
                                                                'madlib.{input_name}_madlib',      -- Destination table
                                                                'features',     -- Input features
                                                                'target',     -- Label
                                                                ARRAY[{layer_size}],         -- Number of units per layer
                                                                '{model_params}',
                                                                'sigmoid',           -- Activation function
                                                                NULL,                    -- Default weight (1)
                                                                FALSE,                   -- No warm start
                                                                FALSE                    -- Not verbose
                                                            );
                                        """
                                    ).format(model_params = sql.SQL(model_params)
                                             , input_name = sql.SQL(self.input_name)
                                             , layer_size = sql.SQL(str(layer_size))
                                             , function = sql.SQL(function)
                                             )

        cur.execute(train_statement)
        conn.commit()
                                            
        predict_statement= sql.SQL( """

                                    DROP TABLE IF EXISTS madlib.{input_name}_madlib_predictions;
                                    SELECT madlib.mlp_predict(
                                                                    'madlib.{input_name}_madlib',                     -- Model table
                                                                    'madlib.{input_name}_madlib_test',         -- Test data table
                                                                    'id',                                      -- Id column in test table
                                                                    'madlib.{input_name}_madlib_predictions',         -- Output table for predictions
                                                                    'response'                                 -- Output classes, not probabilities
                                                                );
                                    """
                                 ).format(input_name = sql.SQL(self.input_name)
                                             
                                             )
        
        cur.execute(predict_statement)
        conn.commit()

        rand = str(round(time.time()))[-4:]
        self.madlib_runtime_query = sql.SQL("""
                                                SELECT madlib.mlp_predict(
                                                                    'madlib.{input_name}_madlib',                     -- Model table
                                                                    'madlib.{input_name}_madlib_test',         -- Test data table
                                                                    'id',                                      -- Id column in test table
                                                                    'madlib.{input_name}_madlib_predictions_{rand}',         -- Output table for predictions
                                                                    'response'                                 -- Output classes, not probabilities
                                                                );
                                            """).format(input_name = sql.SQL(self.input_name)
                                                , rand = sql.SQL(rand)
                                             
                                             )
    
    def train_dt_in_madlib(self):

        conn, cur = self.create_postgres_connection()

        select_statement = ""
        for idf, f in enumerate(self.experiment_feature_names):
            
            select_statement += f + ' as f_' + str(idf) + ', '
            
        select_statement += 'row_id as id, target'

        train_statement = sql.SQL(
                                        """
                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib_train;
                                            CREATE TABLE madlib.{input_name}_madlib_train AS

                                            SELECT row_number() over () as id, mt.*
                                            FROM madlib.{input_name}_train as mt;


                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib, madlib.{input_name}_madlib_summary;
                                            SELECT madlib.tree_train('madlib.{input_name}_madlib_train',         -- source table
                                                                    'madlib.{input_name}_madlib',    -- output model table
                                                                    'id',              -- id column
                                                                    'target',           -- response
                                                                    '*',   -- features
                                                                    'id, target',        -- exclude columns
                                                                    'gini',            -- split criterion
                                                                    NULL::text,        -- no grouping
                                                                    NULL::text       -- no weights, all observations treated equally
                                                                    --,NULL::text,                 -- max depth
                                                                    --NULL::text,                 -- min split
                                                                    --NULL::text,                 -- min bucket
                                                                    --NULL::text                 -- number of bins per continuous variable
                                                                    );
                                            """
                                ).format(
                                             input_name = sql.SQL(self.input_name)
                                             , experiment_name = sql.SQL(self.experiment_name)
                                             , select_statement = sql.SQL(select_statement)
                                             )
        
        cur.execute(train_statement)
        conn.commit()
        
        predict_statement = sql.SQL(

                                        """
                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib_test;
                                            CREATE TABLE madlib.{input_name}_madlib_test AS

                                            SELECT {select_statement}
                                            FROM madlib.{experiment_name}_test as mt;
                                            
                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib_predictions;
                                            SELECT madlib.tree_predict('madlib.{input_name}_madlib',          -- tree model
                                                                    'madlib.{input_name}_madlib_test',               -- new data table
                                                                    'madlib.{input_name}_madlib_predictions',    -- output table
                                                                    'response'); 

                                        """
                                ).format(
                                             input_name = sql.SQL(self.input_name)
                                             , experiment_name = sql.SQL(self.experiment_name)
                                             , select_statement = sql.SQL(select_statement)
                                             )
                                            

        cur.execute(predict_statement)
        conn.commit()

        rand = str(round(time.time()))[-4:]
        self.madlib_runtime_query = sql.SQL("""
                                                SELECT madlib.tree_predict('madlib.{input_name}_madlib',          -- tree model
                                                                    'madlib.{input_name}_madlib_test',               -- new data table
                                                                    'madlib.{input_name}_madlib_predictions_{rand}',    -- output table
                                                                    'response'); 

                                        """
                                ).format(
                                             input_name = sql.SQL(self.input_name)
                                             , rand = sql.SQL(rand)
                            
                                             )
                                                
    
    def train_lr_in_madlib(self):

        conn, cur = self.create_postgres_connection()

        select_statement = ""
        training_features = """ARRAY[1, """
        for idf, f in enumerate(self.experiment_feature_names):
            
            select_statement += f + ' as f_' + str(idf) + ', '
            if idf < len(self.experiment_feature_names) - 1:
                training_features += 'f_' + str(idf) + ', '
            else:
                training_features += 'f_' + str(idf) + ']'
            
        select_statement += 'row_id as id, target'

        if self.task in ('classification', 'multi-class'):

            train_function = sql.SQL("""
                                    madlib.logregr_train( 'madlib.{input_name}_train',  -- Source table
                                                            'madlib.{input_name}_madlib',         -- Output table
                                                            'target',                             -- Dependent variable
                                                            '{training_features}',                  -- Feature vector
                                                            NULL,                                   -- Grouping
                                                            10000,                                     -- Max iterations
                                                            'irls'                                  -- Optimizer to use
                                                            );

                                """
                            ).format(input_name = sql.SQL(self.input_name)
                                    , training_features = sql.SQL(training_features)
                                    )
            
            predict_function = sql.SQL("""round(madlib.logregr_predict_prob(coef, {training_features}))""").format(training_features = sql.SQL(training_features))
        else:

            train_function = sql.SQL(
                                    """
                                    madlib.linregr_train( 'madlib.{input_name}_train',  -- Source table
                                                            'madlib.{input_name}_madlib',         -- Output table
                                                            'target',                             -- Dependent variable
                                                            '{training_features}',                  -- Feature vector
                                                            );

                                """
                            ).format(input_name = sql.SQL(self.input_name)
                                    , training_features = sql.SQL(training_features)
                                    )
            
            predict_function = sql.SQL("""
                                       madlib.linregr_predict(coef, {training_features})
                                        """
                                       ).format(training_features = sql.SQL(training_features))

        train_statement = sql.SQL(
                                        """

                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib, madlib.{input_name}_madlib_summary;
                                            SELECT {train_function}
                                        """
                                ).format(
                                             input_name = sql.SQL(self.input_name)
                                             , train_function = train_function
                                             
                                             )
        cur.execute(train_statement)
        conn.commit()
        predict_statement = sql.SQL(
                                    """
                                        DROP TABLE IF EXISTS madlib.{input_name}_madlib_predictions;
                                            
                                        CREATE TABLE madlib.{input_name}_madlib_predictions AS
                                        SELECT p.row_id as id, {predict_function} as estimated_target
                                        FROM {section}.{input_name}_encoded_relational p, {section}.{input_name}_madlib m; 
                                    """
                                ).format(
                                             input_name = sql.SQL(self.input_name)
                                             , section = sql.SQL(self.section)
                                             , predict_function = predict_function
                                             )
        
        cur.execute(predict_statement)
        conn.commit()

        self.madlib_runtime_query = sql.SQL("""SELECT p.row_id as id, {predict_function} as estimated_target
                                        FROM {section}.{input_name}_encoded_relational p, {section}.{input_name}_madlib m; 
                                    """
                                ).format(
                                             input_name = sql.SQL(self.input_name)
                                             , section = sql.SQL(self.section)
                                             , predict_function = predict_function
                                             )

    def train_knn_in_madlib(self):

        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL(
                                        """
                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib;
                                            SELECT * FROM madlib.knn(
                                                            'madlib.{input_name}_madlib_train',      -- Table of training data
                                                            'features',                -- Col name of training data
                                                            'id',                  -- Col name of id in train data
                                                            'target',               -- Training labels
                                                            'madlib.{input_name}_madlib_test',       -- Table of test data
                                                            'features',                -- Col name of test data
                                                            'id',                  -- Col name of id in test data
                                                            'madlib.{input_name}_madlib',  -- Output table
                                                            5,                    -- Number of nearest neighbors
                                                            False,                 -- True to list nearest-neighbors by id
                                                            'madlib.squared_dist_norm2', -- Distance function
                                                            False,
                                                            'kd_tree',
                                                            'depth=3, leaf_nodes=2'
                                                            );

                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib_predictions;
                                            
                                            CREATE TABLE madlib.{input_name}_madlib_predictions AS
                                            SELECT id, prediction as estimated_target
                                            FROM madlib.{input_name}_madlib;
                                            
                                        """
                                    ).format(
                                             input_name = sql.SQL(self.input_name)
                                             )
        
        cur.execute(function_statement)
        conn.commit()

        rand = str(round(time.time()))[-4:]
        self.madlib_runtime_query = sql.SQL("""
                                            SELECT * FROM madlib.knn(
                                                            'madlib.{input_name}_madlib_train',      -- Table of training data
                                                            'features',                -- Col name of training data
                                                            'id',                  -- Col name of id in train data
                                                            'target',               -- Training labels
                                                            'madlib.{input_name}_madlib_test',       -- Table of test data
                                                            'features',                -- Col name of test data
                                                            'id',                  -- Col name of id in test data
                                                            'madlib.{input_name}_madlib_{rand}',  -- Output table
                                                            5,                    -- Number of nearest neighbors
                                                            False,                 -- True to list nearest-neighbors by id
                                                            'madlib.squared_dist_norm2', -- Distance function
                                                            False,
                                                            'kd_tree',
                                                            'depth=3, leaf_nodes=2'
                                                            );
                                            """).format(
                                             input_name = sql.SQL(self.input_name)
                                             , rand = sql.SQL(rand)
                                             )
    
    def train_xgb_in_madlib(self, model_params):

        conn, cur = self.create_postgres_connection()

        select_statement = ""
        for idf, f in enumerate(self.experiment_feature_names):
            
            select_statement += f + ' as f_' + str(idf) + ', '
            
        select_statement += 'row_id as id, target'

        function_statement = sql.SQL(
                                        """
                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib_train;
                                            CREATE TABLE madlib.{input_name}_madlib_train AS

                                            SELECT row_number() over () as id, mt.*
                                            FROM madlib.{input_name}_train as mt;


                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib, madlib.{input_name}_madlib_summary;
                                            SELECT madlib.xgboost(
                                                        'madlib.{input_name}_madlib_train',  -- Training table
                                                        'madlib.{input_name}_madlib',        -- output model
                                                        'id',       -- Id column
                                                        'target',      -- Class label column
                                                        '*',        -- Independent variables
                                                        NULL,       -- Columns to exclude from features
                                                        $$
                                                            {model_params}
                                                        $$         -- XGBoost grid search parameters
                                                        , NULL
                                                        , 0.9
                                                        
                                            );
                                            
                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib_test;
                                            CREATE TABLE madlib.{input_name}_madlib_test AS

                                            SELECT {select_statement}
                                            FROM madlib.{experiment_name}_test as mt;
                                            
                                            DROP TABLE IF EXISTS madlib.{input_name}_madlib_predictions, madlib.{input_name}_madlib_output, madlib.{input_name}_madlib_output_metrics, madlib.{input_name}_madlib_output_roc_curve;
                                            SELECT madlib.xgboost_predict(
                                                                            'madlib.{input_name}_madlib_test', --test table
                                                                            'madlib.{input_name}_madlib',          -- xgb model
                                                                            'madlib.{input_name}_madlib_output',    -- output table
                                                                            'id',
                                                                            'target',
                                                                            1
                                                                            ); 
                                            
                                            CREATE TABLE  madlib.{input_name}_madlib_predictions AS

                                            SELECT id, target_predicted::INTEGER as estimated_target
                                            FROM    madlib.{input_name}_madlib_output;                  
                                        """
                                    ).format(
                                             input_name = sql.SQL(self.input_name)
                                             , experiment_name = sql.SQL(self.experiment_name)
                                             , select_statement = sql.SQL(select_statement)
                                             , model_params = sql.SQL(model_params)
                                             )
        
        cur.execute(function_statement)
        conn.commit()
    
    def create_scorer_function_madlib(self):
        
        select_statement = ""

        if self.model_name in ('mlpclassifier', 'kneighborsclassifier'):
            select_statement += 'id - 1 as row_id'
        else:
            select_statement += 'id as row_id'

        function_statement = sql.SQL(""" 
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_score_madlib;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_score_madlib(OUT row_id INT, OUT prediction REAL) returns SETOF RECORD AS $func$
                                        
                                        SELECT {select_statement}, estimated_target as prediction 
                                        FROM {section}.{input_name}_madlib_predictions
                                        $func$ LANGUAGE SQL STABLE;
                                    
                                    """).format(input_name = sql.SQL(self.input_name)
                                                , section = sql.SQL(self.section)
                                                , select_statement = sql.SQL(select_statement)
                                                )
        
        conn, cur = self.create_postgres_connection()

        cur.execute(function_statement)
        conn.commit()

        if self.task in ('classification', 'multi-class'):
            self.create_acc_measure_function('score_madlib', 'acc')
            self.create_acc_measure_function('score_madlib', 'f1')
            self.create_acc_measure_function('score_madlib', 'precision')
            self.create_acc_measure_function('score_madlib', 'recall')
        elif self.task == 'regression':
            self.create_acc_measure_function('score_madlib', 'rmsle')
    
    def create_mlp_solution(self, model_params, layer_size, source_table='encoded'):

        self.create_train_table_mlp_knn_madlib()
        self.create_test_table_mlp_knn_madlib(source_table)
        self.train_mlp_in_madlib(model_params, layer_size)
        self.create_scorer_function_madlib()
        self.create_report_function_madlib()


        # elif self.model.__class__.__name__ in ('LinearRegression', 'LogisticRegression'):
        #     self.create_coef_table()
        #     self.create_scorer_function_lr()
        print('MADLIB model artifacts were created')
    
    def create_dt_solution(self):

        self.train_dt_in_madlib()
        self.create_scorer_function_madlib()
        self.create_report_function_madlib()

        print('MADLIB model artifacts were created')
    
    def create_lr_solution(self):

        self.train_lr_in_madlib()
        self.create_scorer_function_madlib()
        self.create_report_function_madlib()

        print('MADLIB model artifacts were created')
    
    def create_knn_solution(self, source_table='encoded'):

        self.create_train_table_mlp_knn_madlib()
        self.create_test_table_mlp_knn_madlib(source_table)
        self.train_knn_in_madlib()
        self.create_scorer_function_madlib()
        self.create_report_function_madlib()

        print('MADLIB model artifacts were created')
    
    def create_report_function_madlib(self):

        conn, cur = self.create_postgres_connection()

        if self.has_featurizer:
            featurizer_statement = sql.SQL("""{section}.measure_featurizer_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name)
                                                                                                             , section = sql.SQL(self.section)
                                                                                                             )
        elif 'imputer' in list(self.pipeline.named_steps.keys()): 
            featurizer_statement = sql.SQL("""{section}.measure_impute_set_runtime_{input_name}()""").format(input_name = sql.SQL(self.input_name)
                                                                                                             , section = sql.SQL(self.section)
                                                                                                             )
        else:
            featurizer_statement = sql.SQL("""0""")

        query = sql.SQL(""" select count(*) from {section}.{input_name}_encoded """).format(input_name = sql.SQL(self.input_name)
                                                                                            , section = sql.SQL(self.section)
                                                                                            )

        cur.execute(query)

        batch_size = cur.fetchall()[0][0]

        madlib_runtime = self.evaluate_runtime(self.madlib_runtime_query)

        if self.task in ('classification', 'multi-class'):
            out_statement = """, OUT acc NUMERIC
                                , OUT prec NUMERIC
                                , OUT recall NUMERIC
                                , OUT f1 NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            select_statement_madlib = sql.SQL(""", {section}.measure_acc_{input_name}_score_madlib() as acc
                                                , {section}.measure_precision_{input_name}_score_madlib() as precision
                                                , {section}.measure_recall_{input_name}_score_madlib() as recall
                                                , {section}.measure_f1_{input_name}_score_madlib() as f1
                                                , {featurizer_statement}
                                                , {section}.measure_encode_scale_set_runtime_{input_name}()
                                                , {madlib_runtime}
                                                , {batch_size}
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        , section = sql.SQL(self.section)
                                                        , madlib_runtime = sql.SQL(str(madlib_runtime))
                                                        )
        
        
        elif self.task == 'regression':
            out_statement = """, OUT rmsle NUMERIC
                                , OUT impute_runtime NUMERIC
                                , OUT encode_runtime NUMERIC
                                , OUT score_runtime NUMERIC
                                , OUT batch_size NUMERIC
                            """
            select_statement_madlib = sql.SQL(""", {section}.measure_rmsle_{input_name}_score_madlib() as rmsle
                                                , {featurizer_statement}
                                                , {section}.measure_encode_scale_set_runtime_{input_name}()
                                                , {madlib_runtime}
                                                , {batch_size}
                                            """).format(input_name = sql.SQL(self.input_name)
                                                        , featurizer_statement = featurizer_statement
                                                        , batch_size = sql.SQL(str(batch_size))
                                                        , section = sql.SQL(self.section)
                                                        , madlib_runtime = sql.SQL(str(madlib_runtime))
                                                        )
        
        #### Sizes
        current = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True) 


        if self.model_name in ('mlpregressor', 'mlpclassifier'):
            model_size = sql.SQL("""pg_total_relation_size('madlib.{input_name}_madlib') + pg_total_relation_size('madlib.{input_name}_madlib_summary') + pg_total_relation_size('madlib.{input_name}_madlib_standardization')"""
                                 ).format(input_name = sql.SQL(self.input_name))
        elif self.model_name in ('decisiontreeregressor', 'decisiontreeclassifier'):
            model_size = sql.SQL("""pg_total_relation_size('madlib.{input_name}_madlib') + pg_total_relation_size('madlib.{input_name}_madlib_summary')"""
                                 ).format(input_name = sql.SQL(self.input_name))
        elif self.model_name in ('linearregression', 'logisticregression'):
            model_size = sql.SQL("""pg_total_relation_size('madlib.{input_name}_madlib') + pg_total_relation_size('madlib.{input_name}_madlib_summary')""").format(input_name = sql.SQL(self.input_name))
        else:
            model_size = sql.SQL("""pg_total_relation_size('madlib.{input_name}_madlib')"""
                                ).format(input_name = sql.SQL(self.input_name))


        function_statement = sql.SQL("""
                                        DROP FUNCTION IF EXISTS {section}.{input_name}_madlib_report;
                                        CREATE OR REPLACE FUNCTION {section}.{input_name}_madlib_report(
                                                                                        OUT method TEXT
                                                                                        , OUT size NUMERIC
                                                                                        {out_statement}
                                                                                    ) returns SETOF record AS
                                        $$

                                        select 'MADlib' as method
                                                , ({model_size}) as size
                                                {select_statement_madlib};

                                        
                                        $$ LANGUAGE SQL STABLE;
                                        """
                                        ).format(input_name = sql.SQL(self.input_name)
                                                , out_statement = sql.SQL(out_statement)
                                                , select_statement_madlib  = select_statement_madlib
                                                , model_size = model_size 
                                                , section = sql.SQL(self.section)
                                            )
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()

    def create_report_pg(self, featurize_query=None):
        

        DEC2FLOAT = new_type(
        DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
        register_type(DEC2FLOAT)

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.experiment_name + '_' + model_name

        # self.create_solution()
        # self.create_report_function_madlib()

        conn, cur = self.create_postgres_connection()

        query = sql.SQL(""" select * from {section}.{input_name}_madlib_report() """).format(input_name = sql.SQL(input_name)
                                                                                           , section = sql.SQL(self.section)
                                                                                           )
        
        df = pd.DataFrame()
        for i in range(5):
            cur.execute(query)

            result = cur.fetchall()

            results = []
            for row in result:
                results.append(list(row))
            
            if self.task == 'regression':
                columns = ['Solution', 'Size (B)', 'RMSLE', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']
            else:
                columns = ['Solution', 'Size (B)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)', 'Batch Size (Records)']

            summary_df = pd.DataFrame(results, columns=columns)
            summary_df['End-to-End Latency (ms)'] = summary_df.apply(lambda x: x['Impute Featurize Latency (ms)'] + x['Encode Scale Latency (ms)'] + x['Score Latency (ms)'], axis=1)
            summary_df['Experiment'] = self.experiment_name
            summary_df['Algorithm'] = model_name
            summary_df['Iteration'] = i

            df = pd.concat((df, summary_df))

        return df

class Standalone(Transpiler):
    """Child class for InferDB's standalone implementation

    Args:
        Transpiler (Transpiler): _description_
    """

    def __init__(self, X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb=False, pipeline=None, inferdb_columns=None, feature_names=None):
        """Constructor method for InferDB's standalone implementation

        Args:
            X_train (np.array or pd.DataFrame): array or dataframe containing the train data
            X_test (np.array or pd.DataFrame): array or dataframe containing the test data
            y_train (np.array): array containing the train true values
            y_test (np.array): array containing the test true values
            experiment_name (string): name of the experiment
            task (string): 
                - 'classification' for binary classification
                - 'regression' for regression
                - 'multi-class' for multi-label classification
            has_featurizer (bool): boolean indicator for featurizer at the 0 position (pipeline[0]) of the pipeline steps.
            featurize_inferdb (bool, optional): If true, featurizer should implement method 'featurize_for_inferdb' which transforms the input data and returns a set
                                                of defined features for InferDB . Defaults to False.
            pipeline (Scikitlearn Pipeline, optional): Pipeline to train input data. Transforms input data into predictions. Defaults to None.
            inferdb_columns (list, optional): List containing a subset of columns for InferDB. Defaults to None.
            feature_names (list, optional): Ordered list containing the names for each column in the input data. Defaults to None.
        """        

        super().__init__(X_train, X_test, y_train, y_test, experiment_name, task, has_featurizer, featurize_inferdb, pipeline, inferdb_columns, feature_names)

    def get_kv_tuples(self, cat_mask, with_pred=True, balance_ratio=1):
        """Creates kv tuples. Keys--> embedding, Values--> predictions

        Args:
            cat_mask (list): list containing the indices of the categorical features in the input data
            with_pred (bool, optional): If True, aggregate model's predictions. If False aggregate y_train true values. Defaults to True.
            balance_ratio (int, optional): _description_. Defaults to 1.

        Returns:
            nd-array: nd-array containing kv tuples. value is the last element of each array and keys correspond to the n-1 elements of the array, where n is the size of the second dimension of the array.
        """        

        encoder = Encoder(self.task)

        if with_pred:
            encoder.fit(self.x_featurized_inferdb, self.y_train_pred, cat_mask)
        else:
            encoder.fit(self.x_featurized_inferdb, self.y_train, cat_mask)

        start = time.time()
        encoded_training_set = encoder.transform_dataset(self.x_featurized_inferdb, [i for i in range(self.x_featurized_inferdb.shape[1])])
        end = time.time() - start

        if with_pred: 
            my_problem = Problem(encoded_training_set, self.y_train_pred, encoder.num_bins, self.task, 1)
        else:
            my_problem = Problem(encoded_training_set, self.y_train, encoder.num_bins, self.task, 1)

        self.encoding_time = end
        
        start = time.time()
        my_problem.set_costs()
        print('Costs set')
        my_optimizer = Optimizer(my_problem, 1)
        my_optimizer.greedy_search()
        end = time.time() - start
        self.solution_time = end
        encoded_training_set = encoded_training_set[:, my_optimizer.greedy_solution]
        print('Greedy Search Complete, index:' + str(my_optimizer.greedy_solution))

        encoded_df= pd.DataFrame(encoded_training_set)
        target_variable_number = encoded_df.shape[1]

        if with_pred:
            encoded_df[target_variable_number] = self.y_train_pred
        else:
            encoded_df[target_variable_number] = self.y_train

        if self.task == 'multi-class':
            groupby_list = [i for i in range(len(my_optimizer.greedy_solution))]
            extended_groupby_list = deepcopy(groupby_list)
            extended_groupby_list.extend([target_variable_number])
            size_df = encoded_df.groupby(extended_groupby_list, as_index=False).size()
            idx = size_df.groupby(groupby_list)['size'].idxmax()
            agg_df = size_df.loc[idx]
            agg_df.drop(columns='size', inplace=True)
        else:
            agg_df = encoded_df.groupby([i for i in range(len(my_optimizer.greedy_solution))], as_index=False)[target_variable_number].mean()
            if balance_ratio > 1:
                agg_df[target_variable_number] = agg_df.apply(lambda x: min(x[target_variable_number] * balance_ratio, 1), axis = 1)

        tuples = agg_df.to_numpy()
        self.solution = my_optimizer.greedy_solution
        self.encoder = encoder
        self.max_path_length = len(self.solution)

        return tuples
    
    def create_standalone_index(self, cat_mask, with_pred=True, balance_ratio=1):
        """Populates a Trie instance

        Args:
            cat_mask (list): list containing the indices of the categorical features in the input data
            with_pred (bool, optional): If true aggregates model's predictions. If False aggregates y_train true values. Defaults to True.
            balance_ratio (int, optional): _description_. Defaults to 1.

        Returns:
            Trie, float: Trie instance and time to build the instance
        """        

        # Populate index
        index_time = 0
        model_index = Trie(self.task)
        st = time.time()
        tuples = self.get_kv_tuples(cat_mask, with_pred, balance_ratio)
        for i in tuples:
            key = i[:len(self.solution)]
            value = round(i[-1])
            model_index.insert(key, value)
        index_time = time.time() - st
        
        print('index populated')

        return model_index, index_time
    
    def perform_index_inference(self, cat_mask, with_pred=True, balance_ratio=1):
        """Performs inference with an Standalone InferDB implementation

        Args:
            cat_mask (list): list containing the indices of the categorical features in the input data
            with_pred (bool, optional): If True aggregates model's predictions. If False aggregates y_train tru values. Defaults to True.
            balance_ratio (int, optional): _description_. Defaults to 1.

        Returns:
            float, float, int, float, float: index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_error, time_to_populate_index
            float, float, int, float, float, float, float, float: index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_accuracy, index_f1, index_recall, index_precision, time_to_populate_index
        """        

        model_index, time_to_populate_index = self.create_standalone_index(cat_mask, with_pred, balance_ratio)

        if isinstance(self.X_test, pd.DataFrame):
            self.X_test.reset_index(inplace=True, drop=True)
            sample = self.X_test.sample(n=10, random_state = 50)
            sample.reset_index(drop=True, inplace=True)
        else:
            rng = np.random.default_rng()
            sample = rng.choice(self.X_test, 10, replace=False)

        inference_runtimes = np.zeros(sample.shape[0], dtype=float)
        preprocessing_runtimes = np.zeros(sample.shape[0], dtype=float)
        if self.featurize_inferdb:
            for index, row in sample.iterrows():
                start = time.time()
                x_featurized = np.array(self.pipeline[0].transform_for_inferdb(row))[0]
                instance = x_featurized[self.solution]
                preprocessed_instance = self.encoder.transform_single(instance, self.solution)
                preprocessing_runtime = time.time() - start
                preprocessing_runtimes[index] = preprocessing_runtime
                st = time.time()
                model_index.query(preprocessed_instance)
                inference_runtimes[index] = time.time() - st
        else:
            if self.inferdb_columns:
                subset = sample[:, self.inferdb_columns]
            else:
                subset = sample
            
            if isinstance(subset, pd.DataFrame):
                subset = subset.to_numpy()

            for index, row in tqdm(enumerate(subset)):
                start = time.time()
                instance = row[self.solution]
                preprocessed_instance = self.encoder.transform_single(instance, self.solution)
                preprocessing_runtime = time.time() - start
                preprocessing_runtimes[index] = preprocessing_runtime
                st = time.time()
                model_index.query(preprocessed_instance)
                inference_runtimes[index] = time.time() - st
            
        self.y_pred_trie = np.zeros_like(self.y_test, dtype=float)

        if isinstance(self.x_test_inferdb, pd.DataFrame):
            self.x_test_inferdb = self.x_test_inferdb.to_numpy()

        for index, row in tqdm(enumerate(self.x_test_inferdb)):
            instance = row[self.solution]
            preprocessed_instance = self.encoder.transform_single(instance, self.solution)
            self.y_pred_trie[index] = model_index.query(preprocessed_instance)
            
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
            index_accuracy = accuracy_score(self.y_test, self.y_pred_trie)
            index_f1 = f1_score(self.y_test, self.y_pred_trie, average='macro')
            index_recall = recall_score(self.y_test, self.y_pred_trie, average='macro')
            index_precision = precision_score(self.y_test, self.y_pred_trie, average='macro')

            return index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_accuracy, index_f1, index_recall, index_precision, time_to_populate_index
        if self.task == 'multi-class':
            index_accuracy = accuracy_score(self.y_test, self.y_pred_trie)
            index_f1 = f1_score(self.y_test, self.y_pred_trie, average='macro')
            index_recall = recall_score(self.y_test, self.y_pred_trie, average='macro')
            index_precision = precision_score(self.y_test, self.y_pred_trie, average='macro')

            return index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_accuracy, index_f1, index_recall, index_precision, time_to_populate_index
        elif self.task == 'regression':
            index_error = mean_squared_log_error(self.y_test, self.y_pred_trie, squared=False)
        
            return index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_error, time_to_populate_index
    
    def perform_model_inference(self):
        """Performs inference with a trained pipeline

        Returns:
            float, float, int, float: model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_error
            float, float, int, float, float, float, float: model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_accuracy, model_f1, model_recall, model_precision
        """        
    
        if isinstance(self.X_test, pd.DataFrame):

            self.X_test.reset_index(inplace=True, drop=True)
            sample = self.X_test.sample(n=10, random_state = 50, replace=False)
            sample.reset_index(inplace=True, drop=True)
            inference_runtimes = np.zeros(sample.shape[0], dtype=float)
            preprocessing_runtimes = np.zeros(sample.shape[0], dtype=float)
            
            for index, row in sample.iterrows():
                row = row.to_frame().transpose()
                start = time.time()
                transformed_instance = self.pipeline[:-1].transform(row)
                end = time.time() - start
                preprocessing_runtimes[index] = end
                self.experiment_transformed_input_size = transformed_instance.shape[1]
                start = time.time()
                self.pipeline[-1].predict(transformed_instance)
                end = time.time() - start
                inference_runtimes[index] = end
        elif isinstance(self.X_test, np.ndarray):
            
            rng = np.random.default_rng()
            sample = rng.choice(self.X_test, 10, replace=False)
            inference_runtimes = np.zeros(sample.shape[0], dtype=float)
            preprocessing_runtimes = np.zeros(sample.shape[0], dtype=float)
            for index, row in enumerate(sample):
                start = time.time()
                transformed_instance = self.pipeline[:-1].transform(row.reshape((1, row.shape[0])))
                end = time.time() - start
                preprocessing_runtimes[index] = end
                self.experiment_transformed_input_size = transformed_instance.shape[1]
                start = time.time()
                self.pipeline[-1].predict(transformed_instance)
                end = time.time() - start
                inference_runtimes[index] = end
        
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
        
        y_pred = self.pipeline[-1].predict(self.x_test_trans)
        
        if self.task == 'regression':
            model_error = mean_squared_log_error(self.y_test, np.exp(y_pred), squared=False)
            return model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_error
        elif self.task == 'classification':
            model_accuracy = accuracy_score(self.y_test, y_pred)
            model_f1 = f1_score(self.y_test, y_pred, average='macro')
            model_recall = recall_score(self.y_test, y_pred, average='macro')
            model_precision = precision_score(self.y_test, y_pred, average='macro')
            return model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_accuracy, model_f1, model_recall, model_precision
        elif self.task == 'multi-class':
            model_accuracy = accuracy_score(self.y_test, y_pred)
            model_f1 = f1_score(self.y_test, y_pred, average='macro')
            model_recall = recall_score(self.y_test, y_pred, average='macro')
            model_precision = precision_score(self.y_test, y_pred, average='macro')
            return model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_accuracy, model_f1, model_recall, model_precision
    
    def create_report(self, cat_mask, with_pred=True, balance_ratio=1):
        """Creates a summary dataframe containing performance numbers for an InferDB instance

        Args:
            cat_mask (list): list containing indices for categorical features in the input data
            with_pred (bool, optional): If True aggregates predictions from a model. If False, aggregates true values from y_train. Defaults to True.
            balance_ratio (int, optional): _description_. Defaults to 1.

        Returns:
            DataFrame: DataFrame containing a summary of performance numbers for InferDB standalone implementation
        """        

        if self.task == 'regression':
            model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_error = self.perform_model_inference()
            index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_error, index_time = self.perform_index_inference(cat_mask, with_pred, balance_ratio)
            summary = [[self.experiment_name, self.model_name, self.experiment_training_instances, self.experiment_input_dimensions, self.experiment_transformed_input_size, self.training_preprocessing_runtime, self.training_time, self.encoding_time, self.solution_time, index_time, self.max_path_length, self.experiment_test_instances, model_error, index_error, model_avg_prep_runtimes, model_avg_scoring_runtimes, model_avg_prep_runtimes + model_avg_scoring_runtimes, index_avg_prep_runtimes, index_avg_scoring_runtimes, index_avg_prep_runtimes + index_avg_scoring_runtimes, model_size, index_size]]
            columns = ['name', 'model', 'training_instances', 'original_dimensions', 'input_dimensions', 'training_preprocessing_runtime', 'training_runtime', 'encoding_runtime', 'solution_runtime', 'index_runtime', 'max_path_length', 'testing_instances', 'model_error', 'index_error', 'model_avg_preprocessing_runtime', 'model_avg_scoring_runtime', 'model_end_to_end_runtime', 'index_avg_preprocessing_runtime', 'index_avg_scoring_runtime', 'index_end_to_end_runtime', 'model_size', 'index_size']
        elif self.task in ('classification', 'multi-class'):
            model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_accuracy, model_f1, model_recall, model_precision = self.perform_model_inference()
            index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_accuracy, index_f1, index_recall, index_precision, index_time = self.perform_index_inference(cat_mask, with_pred, balance_ratio)
            summary = [[self.experiment_name, self.model_name, self.experiment_training_instances, self.experiment_input_dimensions, self.experiment_transformed_input_size, self.training_preprocessing_runtime, self.training_time, self.encoding_time, self.solution_time, index_time, self.max_path_length, self.experiment_test_instances, model_accuracy, model_f1, model_recall, model_precision, index_accuracy, index_f1, index_recall, index_precision, model_avg_prep_runtimes, model_avg_scoring_runtimes, model_avg_prep_runtimes + model_avg_scoring_runtimes , index_avg_prep_runtimes, index_avg_scoring_runtimes, index_avg_prep_runtimes + index_avg_scoring_runtimes, model_size, index_size]]
            columns = ['name', 'model', 'training_instances', 'original_dimensions', 'input_dimensions', 'training_preprocessing_runtime', 'training_runtime', 'encoding_runtime', 'solution_runtime', 'index_runtime', 'max_path_length', 'testing_instances', 'model_accuracy', 'model_f1', 'model_recall', 'model_precision', 'index_accuracy', 'index_f1', 'index_recall', 'index_precision', 'model_avg_preprocessing_runtime', 'model_avg_scoring_runtime', 'model_end_to_end_runtime', 'index_avg_preprocessing_runtime', 'index_avg_scoring_runtime', 'index_end_to_end_runtime', 'model_size', 'index_size']

        summary_df = pd.DataFrame(summary, columns=columns)

        return summary_df


