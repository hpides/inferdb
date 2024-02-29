import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from pathlib import Path
from nyc_rides_featurizer import NYC_Featurizer
import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[3]
src_folder = os.path.join(project_folder,'src')
sys.path.append(str(src_folder))
from pickle import load, dump
from psycopg2 import sql
from psycopg2.extensions import register_adapter, AsIs, register_type, new_type, DECIMAL
import re
from database_connect import connect
from encoder import Encoder
from optimizer import Problem, Optimizer
from scipy import sparse
from io import StringIO
from os.path import getsize
from inference_trie import Trie
import time



class preprocessing_experiment:
    def __init__(self, name, featurizer, model) -> None:
        self.name = name
        self.model = model
        self.featurizer = featurizer

        numerical_transformer = Pipeline(
                    steps=
                            [
                                ('scaler', StandardScaler())
                            ]
                    )
        categorical_transformer = Pipeline(
            steps=
                    [
                        ('encoder', OneHotEncoder(handle_unknown='ignore'))
                    ]
            )

        column_transformer = ColumnTransformer(
                                                transformers=[
                                                                ('num', numerical_transformer, self.featurizer.num_features)
                                                                , ('cat', categorical_transformer, self.featurizer.cat_features)
                                                            ]
                                                , remainder='passthrough'

                                            )
        self.pipeline = Pipeline(
                                    steps=
                                            [   ('featurizer', self.featurizer)
                                                , ('column_transformer', column_transformer)
                                                , ('clf', self.model)
                                            ]
                            )
    
    def create_postgres_connection(self):
        conn = connect()
        cur = conn.cursor()

        return conn, cur

    def text_normalizer(self, text):

        rep = {"-": "_", ".": "_", "?":"", "/":"", "(":"_", ")":"_", "&":"_"} # define desired replacements here

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pattern = re.compile("|".join(rep.keys()))

        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    
    def text_normalizer_select_statement(self, text):

        rep = {"-inf": "'-infinity'::numeric", "inf": "'infinity'::numeric"} # define desired replacements here

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pattern = re.compile("|".join(rep.keys()))

        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    def train_pipeline(self, path_to_data, target_feature, persist):

        df = pd.read_csv(path_to_data)

        self.training_features = [i for i in list(df) if i != target_feature]
        X = df[self.training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        start = time.time()
        trained_pipeline = self.pipeline.fit(X_train, y_train)
        end =  time.time() - start

        transformed_df = self.pipeline[0].transform(X_train.head(5))
        transformed_features = list(transformed_df)

        self.training_time = end 

        start = time.time()
        self.pipeline[:-1].transform(X_train)
        end = time.time() - start

        self.preproc_training_time = end

        if not persist:
            return trained_pipeline
        else:
            self.persist_pipeline(trained_pipeline)
    
    def persist_pipeline(self, trained_pipeline):

        #### Sizes
        project_folder = Path(__file__).resolve().parents[3]
        path = os.path.join(project_folder, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        if self.featurizer.depth == 'deep':
            path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_complex_pipeline.joblib')
        else:
            path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_simple_pipeline.joblib')

        with open(path, "wb") as File:
            dump(trained_pipeline, File)
    
    def create_train_table(self, trained_pipeline, path_to_data, target_feature):

        conn, cur = self.create_postgres_connection()

        df = pd.read_csv(path_to_data)

        X = df[self.training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        transformed_df = trained_pipeline[:-1].transform(X_train.head(5))

        ### Creates a table where test data will be stored
        table_name = self.name + '_train'
        initial_statement = 'DROP TABLE IF EXISTS pgml.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE pgml.' + table_name + '('
        for idf in range(transformed_df.shape[1]):
            
            initial_statement += 'f_' + str(idf) + ' REAL, '
            
        initial_statement += 'target REAL)'

        ### Create table
        cur.execute(initial_statement)
        conn.commit()
    
    def create_test_table(self, datetime_list, string_list, numeric_list):

        conn, cur = self.create_postgres_connection()
        ### Creates a table where test data will be stored
        table_name = self.name + '_test'
        initial_statement = 'DROP TABLE IF EXISTS pgml.' + table_name + ' CASCADE; '
        initial_statement += 'CREATE TABLE pgml.' + table_name + '(ROW_ID INTEGER, '
        
        for idf, feature in enumerate(self.training_features):
            
            if feature in datetime_list:
                initial_statement +=  self.text_normalizer(feature).lower() + ' TIMESTAMP, '
            elif feature in string_list:
                initial_statement += feature + ' TEXT, '
            elif feature in numeric_list:
                initial_statement += feature + ' REAL, '

        initial_statement += 'target REAL)'

        ### Create table
        cur.execute(initial_statement)
        conn.commit()
    
    def insert_test_tuples(self, path_to_data, target_feature):

        conn, cur = self.create_postgres_connection()
        table_name = self.name + '_test'
        insert_statement = 'INSERT INTO pgml.{} VALUES ('
        for feature in self.training_features:

            insert_statement += "%s" + ','
        
        #### Extra value for the target
        insert_statement += "%s" + ')'
        
        df = pd.read_csv(path_to_data)

        X = df[self.training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        ids = (np.arange(X_test.shape[0], dtype=np.intc)).reshape((X_test.shape[0], 1))
        test_input = np.hstack((X_test.to_numpy(), y_test.reshape((y_test.shape[0], 1))))
        test_input = np.hstack((ids, test_input))

        # tuples = [tuple(t) for t in test_input]
        # args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", tuple(x)).decode('utf-8') for x in test_input) 
        # cur.execute("INSERT INTO pgml.nyc_rides_test VALUES " + args_str)

        buffer = StringIO()
        df = pd.DataFrame(test_input)
        df[0] = df[0].astype('int')
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY pgml.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name)), buffer)

        # sql_insert_query = sql.SQL(insert_statement).format(sql.Identifier(table_name))
        # print(test_input.shape)
        # cur.executemany(sql_insert_query, test_input)
        conn.commit()
        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")
    
    def insert_train_tuples(self, trained_pipeline, path_to_data, target_feature):

        conn, cur = self.create_postgres_connection()

        df = pd.read_csv(path_to_data)

        X = df[self.training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        transformed_df = trained_pipeline[:-1].transform(X_train)
        

        table_name = self.name + '_train'
        insert_statement = 'INSERT INTO pgml.{} VALUES ('
        for feature in range(transformed_df.shape[1]):

            insert_statement += "%s" + ','
        
        #### Extra value for the target
        insert_statement += "%s" + ")"

        try:
            train_input = np.hstack((transformed_df, y_train.reshape((y_train.shape[0], 1))))
        except ValueError:
            train_input = sparse.hstack((transformed_df, y_train.reshape((y_train.shape[0], 1))))
            
            train_input = np.asarray(train_input.todense())

        buffer = StringIO()
        df = pd.DataFrame(train_input)
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert("COPY pgml.nyc_rides_train FROM STDIN WITH CSV DELIMITER ';'", buffer)
        conn.commit()

        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")
    
    def geo_distance(self):
        
        conn, cur = self.create_postgres_connection()

        func = sql.SQL("""
                        /*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
                        /*::                                                                         :*/
                        /*::  This routine calculates the distance between two points (given the     :*/
                        /*::  latitude/longitude of those points). It is being used to calculate     :*/
                        /*::  the distance between two locations using GeoDataSource(TM) Products    :*/
                        /*::                                                                         :*/
                        /*::  Definitions:                                                           :*/
                        /*::    South latitudes are negative, east longitudes are positive           :*/
                        /*::                                                                         :*/
                        /*::  Passed to function:                                                    :*/
                        /*::    lat1, lon1 = Latitude and Longitude of point 1 (in decimal degrees)  :*/
                        /*::    lat2, lon2 = Latitude and Longitude of point 2 (in decimal degrees)  :*/
                        /*::    unit = the unit you desire for results                               :*/
                        /*::           where: 'M' is statute miles (default)                         :*/
                        /*::                  'K' is kilometers                                      :*/
                        /*::                  'N' is nautical miles                                  :*/
                        /*::  Worldwide cities and other features databases with latitude longitude  :*/
                        /*::  are available at https://www.geodatasource.com                         :*/
                        /*::                                                                         :*/
                        /*::  For enquiries, please contact sales@geodatasource.com                  :*/
                        /*::                                                                         :*/
                        /*::  Official Web site: https://www.geodatasource.com                       :*/
                        /*::                                                                         :*/
                        /*::  Thanks to Kirill Bekus for contributing the source code.               :*/
                        /*::                                                                         :*/
                        /*::         GeoDataSource.com (C) All Rights Reserved 2022                  :*/
                        /*::                                                                         :*/
                        /*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/

                        CREATE OR REPLACE FUNCTION pgml.calculate_distance(lat1 float, lon1 float, lat2 float, lon2 float, units varchar)
                        RETURNS float AS $dist$
                            DECLARE
                                dist float = 0;
                                radlat1 float;
                                radlat2 float;
                                theta float;
                                radtheta float;
                            BEGIN
                                IF lat1 = lat2 AND lon1 = lon2
                                    THEN RETURN dist;
                                ELSE
                                    radlat1 = pi() * lat1 / 180;
                                    radlat2 = pi() * lat2 / 180;
                                    theta = lon1 - lon2;
                                    radtheta = pi() * theta / 180;
                                    dist = sin(radlat1) * sin(radlat2) + cos(radlat1) * cos(radlat2) * cos(radtheta);

                                    IF dist > 1 THEN dist = 1; END IF;

                                    dist = acos(dist);
                                    dist = dist * 180 / pi();
                                    dist = dist * 60 * 1.1515;

                                    IF units = 'K' THEN dist = dist * 1.609344; END IF;
                                    IF units = 'N' THEN dist = dist * 0.8684; END IF;

                                    RETURN dist;
                                END IF;
                            END;
                        $dist$ LANGUAGE plpgsql PARALLEL SAFE;
                        
                        """
                    )
        cur.execute(func)
        conn.commit()
                       
    
    def create_featurizer(self, pickup_encoder=None, dropoff_encoder=None, cluster_encoder=None, freq_mapper=None, batch_size=None):

        conn, cur = self.create_postgres_connection()

        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        if pickup_encoder and dropoff_encoder and cluster_encoder and freq_mapper:
            pickup_cluster = 'CASE '
            dropoff_cluster = 'CASE '
            number_of_pickup_clusters = len(pickup_encoder[0])
            number_of_dropoff_clusters = len(dropoff_encoder[0])

            for idc in range(number_of_pickup_clusters):
                if idc < number_of_pickup_clusters - 1:
                    pickup_cluster += ' WHEN pickup_latitude BETWEEN ' + str(pickup_encoder[0][idc][0]) + ' and ' + str(pickup_encoder[0][idc][1]) + ' AND pickup_longitude BETWEEN ' + str(pickup_encoder[1][idc][0]) + ' and ' + str(pickup_encoder[1][idc][1]) + ' THEN ' + str(idc)
                else:
                    pickup_cluster += ' WHEN pickup_latitude BETWEEN ' + str(pickup_encoder[0][idc][0]) + ' and ' + str(pickup_encoder[0][idc][1]) + ' AND pickup_longitude BETWEEN ' + str(pickup_encoder[1][idc][0]) + ' and ' + str(pickup_encoder[1][idc][1]) + ' THEN ' + str(idc) + ' END '
            
            for idc in range(number_of_dropoff_clusters):
                if idc < number_of_dropoff_clusters - 1:
                    dropoff_cluster += ' WHEN dropoff_latitude BETWEEN ' + str(dropoff_encoder[0][idc][0]) + ' and ' + str(dropoff_encoder[0][idc][1]) + ' AND dropoff_longitude BETWEEN ' + str(dropoff_encoder[1][idc][0]) + ' and ' + str(dropoff_encoder[1][idc][1]) + ' THEN ' + str(idc)
                else:
                    dropoff_cluster += ' WHEN dropoff_latitude BETWEEN ' + str(dropoff_encoder[0][idc][0]) + ' and ' + str(dropoff_encoder[0][idc][1]) + ' AND dropoff_longitude BETWEEN ' + str(dropoff_encoder[1][idc][0]) + ' and ' + str(dropoff_encoder[1][idc][1]) + ' THEN ' + str(idc) + ' END ' 
        
            pickup_cluster = self.text_normalizer_select_statement(pickup_cluster)
            dropoff_cluster = self.text_normalizer_select_statement(dropoff_cluster)

            subquery = sql.SQL("""
                                        (
                                            SELECT
                                                row_id
                                                , passenger_count
                                                , TRIM(to_char(pickup_datetime, 'Day')) as day
                                                , extract(hour from pickup_datetime) as hour
                                                , {pickup_cluster} as pickup_cluster
                                                , {dropoff_cluster} as dropoff_cluster
                                                , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                                , extract(month from pickup_datetime) as month
                                                , CASE WHEN extract(hour from pickup_datetime) BETWEEN 6 and 11 THEN 'morning'
                                                    WHEN extract(hour from pickup_datetime) BETWEEN 12 and 15 THEN 'afternoon'
                                                    WHEN extract(hour from pickup_datetime) BETWEEN 16 and 21 THEN 'evening'
                                                    ELSE 'late_night' END as pickup_hour_of_day
                                                , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                                            FROM (SELECT * FROM pgml.nyc_rides_test ORDER BY ROW_ID ASC LIMIT {limit_factor}) zt
                                        ) as s
                                        
                                        """
                                    ).format(pickup_cluster = sql.SQL(pickup_cluster)
                                                , dropoff_cluster = sql.SQL(dropoff_cluster)
                                                , limit_factor = sql.SQL(str(batch_size))
                                                )
        

            clusters_cluster = 'CASE '
            number_clusters = len(cluster_encoder[0])
            for idc in range(number_clusters):
                if idc < number_clusters - 1:
                    clusters_cluster += ' WHEN s.pickup_cluster IN ' + str(tuple(cluster_encoder[0][idc])) + ' AND s.dropoff_cluster IN ' + str(tuple(cluster_encoder[1][idc])) + ' THEN ' + str(idc)
                else:
                    clusters_cluster += ' WHEN s.pickup_cluster IN ' + str(tuple(cluster_encoder[0][idc])) + ' AND s.dropoff_cluster IN ' + str(tuple(cluster_encoder[1][idc])) + ' THEN ' + str(idc) + ' END '
            
            clusters_cluster = self.text_normalizer_select_statement(clusters_cluster)


            featurizer_statement = sql.SQL("""
                                                DROP FUNCTION IF EXISTS pgml.{input_name}_featurizer(OUT row_id INTEGER, OUT passenger_count INTEGER, OUT day TEXT, OUT hour INTEGER
                                                , OUT is_weekend INTEGER, OUT month INTEGER, OUT pickup_hour_of_day TEXT, OUT distance REAL, OUT route_freq INTEGER, OUT freq_dist REAL, OUT clusters_cluster TEXT) CASCADE;
                                                CREATE OR REPLACE FUNCTION pgml.{input_name}_featurizer(OUT row_id INTEGER, OUT passenger_count INTEGER, OUT day TEXT, OUT hour INTEGER
                                                , OUT is_weekend INTEGER, OUT month INTEGER, OUT pickup_hour_of_day TEXT, OUT distance REAL, OUT route_freq INTEGER, OUT freq_dist REAL, OUT clusters_cluster TEXT) returns SETOF record AS
                                                $func$
                                                
                                                SELECT
                                                    s.row_id
                                                    , s.passenger_count
                                                    , s.day
                                                    , s.hour
                                                    , s.is_weekend
                                                    , s.month
                                                    , s.pickup_hour_of_day
                                                    , s.distance
                                                    , COALESCE(rq.count, 0) as route_freq
                                                    , s.distance * COALESCE(rq.count, 0) as freq_dist
                                                    , {clusters_cluster} as clusters_cluster
                                                FROM {subquery}
                                                LEFT JOIN pgml.route_freq as rq
                                                ON
                                                    s.pickup_cluster || '_' || s.dropoff_cluster = rq.cluster_combination
                                                    AND
                                                    s.day = rq.day
                                                    AND
                                                    s.hour = rq.hour
                                                ORDER BY ROW_ID ASC

                                                $func$ LANGUAGE SQL STABLE;
                                        
                                            """
                                        ).format(subquery = subquery
                                                    , input_name = sql.SQL(input_name)
                                                    , clusters_cluster = sql.SQL(clusters_cluster)
                                                    )
            def create_route_freq_table():

                conn, cur = self.create_postgres_connection()

                cur.execute("DROP TABLE IF EXISTS pgml.route_freq CASCADE; CREATE TABLE pgml.route_freq(cluster_combination TEXT, day TEXT, hour INTEGER, count REAL, PRIMARY KEY(cluster_combination, day, hour))")
                conn.commit()
            
            def insert_route_freq_tuples(route_freq_mapper):
                conn, cur = self.create_postgres_connection()

                tuples = [(d[0], d[1], d[2], route_freq_mapper.get(d)['count']) for d in route_freq_mapper]
                # args_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x).decode('utf-8') for x in tuples) 

                buffer = StringIO()
                df = pd.DataFrame(tuples)
                df.to_csv(buffer, index=False, header=False, sep=';')
                buffer.seek(0)

                cur.copy_expert(sql.SQL("COPY pgml.route_freq FROM STDIN WITH CSV DELIMITER ';'"), buffer)

                # cur.execute("INSERT INTO pgml.route_freq VALUES " + args_str)
                conn.commit()
            
            create_route_freq_table()
            insert_route_freq_tuples(freq_mapper)
        
        else:

            featurizer_statement = sql.SQL("""
                                                DROP FUNCTION IF EXISTS pgml.{input_name}_featurizer(OUT row_id INTEGER, OUT passenger_count INTEGER, OUT day TEXT, OUT hour INTEGER,
                                                OUT is_weekend INTEGER, OUT month INTEGER) CASCADE;
                                                CREATE OR REPLACE FUNCTION pgml.{input_name}_featurizer(OUT row_id INTEGER, OUT passenger_count INTEGER, OUT day TEXT, OUT hour INTEGER,
                                                OUT is_weekend INTEGER, OUT month INTEGER, OUT distance REAL) returns SETOF record AS
                                                $func$
                                                    SELECT 
                                                        row_id
                                                        , passenger_count
                                                        , TRIM(to_char(pickup_datetime, 'Day')) as day
                                                        , extract(hour from pickup_datetime) as hour
                                                        , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                                        , extract(month from pickup_datetime) as month
                                                        , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                                                    FROM (SELECT * FROM pgml.nyc_rides_test ORDER BY ROW_ID ASC LIMIT {limit_factor}) zt
                                                $func$ LANGUAGE SQL STABLE;
                                            """).format(input_name = sql.SQL(input_name)
                                                        , limit_factor = sql.SQL(str(batch_size)))

        cur.execute(featurizer_statement)
        conn.commit()

        self.create_measure_function('featurizer')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_featurized;
                                        CREATE MATERIALIZED VIEW pgml.{name}_featurized AS 
                                        
                                        SELECT * FROM  pgml.{input_name}_featurizer();
                                        
                                    """).format(name=sql.SQL(self.name)
                                                , input_name = sql.SQL(input_name)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()
    
    def create_kv_featurizer(self, batch_size=None):

        conn, cur = self.create_postgres_connection()

        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        featurizer_statement = sql.SQL("""
                                            DROP FUNCTION IF EXISTS pgml.{input_name}_featurizer_kv(OUT row_id INTEGER, OUT passenger_count INTEGER, OUT day TEXT, OUT hour INTEGER,
                                            OUT is_weekend INTEGER, OUT month INTEGER, OUT pickup_hour_of_day TEXT, OUT distance REAL) CASCADE;
                                            CREATE OR REPLACE FUNCTION pgml.{input_name}_featurizer_kv(OUT row_id INTEGER, OUT passenger_count INTEGER, OUT day TEXT, OUT hour INTEGER,
                                            OUT is_weekend INTEGER, OUT month INTEGER, OUT distance REAL) returns SETOF record AS
                                            $func$
                                                SELECT 
                                                    row_id
                                                    , passenger_count
                                                    , TRIM(to_char(pickup_datetime, 'Day')) as day
                                                    , extract(hour from pickup_datetime) as hour
                                                    , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                                    , extract(month from pickup_datetime) as month
                                                    , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                                                FROM (SELECT * FROM pgml.nyc_rides_test ORDER BY ROW_ID ASC LIMIT {limit_factor}) zt
                                            $func$ LANGUAGE SQL STABLE;
                                        """).format(input_name = sql.SQL(input_name)
                                                    , limit_factor = sql.SQL(str(batch_size)))

        cur.execute(featurizer_statement)
        conn.commit()

        self.create_measure_function('featurizer_kv')

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_featurized_kv;
                                        CREATE MATERIALIZED VIEW pgml.{name}_featurized_kv AS 
                                        
                                        SELECT * FROM  pgml.{input_name}_featurizer_kv();
                                        
                                    """).format(name=sql.SQL(self.name)
                                                , input_name = sql.SQL(input_name)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

    def create_encode_scale_set(self, trained_pipeline):

        conn, cur = self.create_postgres_connection()

        num_mask = trained_pipeline.named_steps['column_transformer'].transformers_[0][2]
        cat_mask = trained_pipeline.named_steps['column_transformer'].transformers_[1][2]
        model_name = (trained_pipeline[-1].__class__.__name__).lower()
        function_name = self.name + '_' + model_name + '_encode_scale_set'
        table_name = 'nyc_rides_featurizer'

        numerical_feature_names = self.featurizer.num_features
        categorical_feature_names = self.featurizer.cat_features

        scaler = trained_pipeline.named_steps['column_transformer'].transformers_[0][1].named_steps['scaler']
        means = scaler.mean_
        vars = np.sqrt(scaler.var_)

        select_statement = 'SELECT ARRAY['
        for idf, feature in enumerate(numerical_feature_names):
            if vars[idf] > 0:
                if cat_mask:
                    select_statement += '(' + feature + '-(' + str(means[idf]) + '))' + '/' + str(vars[idf]) + ', '
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
            unique_categories = trained_pipeline.named_steps['column_transformer'].transformers_[1][1].named_steps['encoder'].categories_[idf]
            for idc, category in enumerate(unique_categories):
                if idf == len(categorical_feature_names) - 1 and idc == len(unique_categories) - 1:
                    select_statement += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END]'
                else:
                    select_statement += ' CASE WHEN ' + feature + '=' + """'""" + str(category) + """'""" + ' THEN 1 ELSE 0 END, '

        function_statement = sql.SQL(
                            """
                                --DROP FUNCTION IF EXISTS pgml.{function_name} CASCADE;
                                CREATE OR REPLACE FUNCTION pgml.{function_name}() RETURNS SETOF REAL[] AS
                                $func$
                                    {select_statement} as m FROM pgml.{name}_featurized;
                                $func$
                                LANGUAGE SQL STABLE;
                            """).format(function_name = sql.Identifier(function_name)
                                        , name = sql.SQL(self.name)
                                        , select_statement = sql.SQL(select_statement))
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_scaled;
                                        CREATE MATERIALIZED VIEW pgml.{name}_scaled AS 
                                        
                                        SELECT m.* FROM pgml.{function_name}() as m;
                                        
                                    """).format(function_name=sql.SQL(function_name)
                                                , name = sql.SQL(self.name)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

        self.create_measure_function('encode_scale_set')
    
    def create_preprocessing_pipeline(self):

        conn, cur = self.create_postgres_connection()

        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        function_statement = sql.SQL("""
                                        --DROP FUNCTION IF EXISTS pgml.{input_name}_preprocess CASCADE;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_preprocess(OUT row_id INTEGER, OUT col_id INTEGER, OUT val NUMERIC) RETURNS SETOF record AS
                                        $func$
                                        with d as (select row_number() over () as id, m.* from pgml.{name}_scaled as m)

                                        SELECT m.id - 1 as row_id, u.ord - 1 as col_id, u.val
                                        FROM   d m,
                                            LATERAL unnest(m.m) WITH ORDINALITY AS u(val, ord)
                                        where u.val != 0
                                        $func$ LANGUAGE SQL STABLE;

                                    """).format(input_name = sql.SQL(input_name)
                                                , name = sql.SQL(self.name)
                                                )
        ### Create function
        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_preprocessed;
                                        CREATE MATERIALIZED VIEW pgml.{name}_preprocessed AS 
                                        
                                        SELECT * FROM  pgml.{input_name}_preprocess();
                                        
                                    """).format(name=sql.SQL(self.name)
                                                , input_name = sql.SQL(input_name)
                                                )
        
        cur.execute(persist_statement)
        conn.commit()

        self.create_measure_function('preprocess')
    
    def create_coef_table(self, trained_pipeline):

        conn, cur = self.create_postgres_connection()
        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name
        
        table_statement = sql.SQL("""
                                DROP TABLE IF EXISTS pgml.{input_name}_coefficients CASCADE;
                                CREATE TABLE pgml.{input_name}_coefficients
                                (col_id INTEGER
                                , val NUMERIC
                                , PRIMARY KEY(col_id)
                                ); 
                            """).format(input_name = sql.SQL(input_name))
        ### Create function
        cur.execute(table_statement)
        conn.commit()
        
        ##### Insert coefficients and bias in the nn table

        insert_statement = 'INSERT INTO pgml.' + input_name + '_coefficients (col_id, val) VALUES '
        for idc, c in enumerate(trained_pipeline[-1].coef_):
            tup = (idc, c)
            insert_statement += str(tup) + ','
        
        intercept = (-1, trained_pipeline[-1].intercept_)
        insert_statement += str(intercept) + ';'

        ### Create function
        cur.execute(insert_statement)
        conn.commit()
    
    def create_scorer_function_lr(self):

        conn, cur = self.create_postgres_connection()

        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        ### Creates a function to transform preprocessed arrays into predictions

        
        select_statement = 'coef.val + intercept.val as prediction' 

        function_statement = sql.SQL("""--DROP FUNCTION IF EXISTS pgml.{input_name}_score;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_score(OUT row_id INTEGER, OUT prediction NUMERIC)
                                        RETURNS SETOF record 
                                        AS $func$

                                        with pre as (select * from pgml.{name}_preprocessed),

                                        intercept as (select val from pgml.{input_name}_coefficients where col_id = -1),

                                        coef as (select pre.row_id, sum(pre.val * b.val) as val
                                                from pre
                                                left join pgml.{input_name}_coefficients b
                                                on pre.col_id = b.col_id
                                                group by 1)

                                        select coef.row_id, {select_statement}
                                        from coef, intercept
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    """).format(input_name = sql.SQL(input_name)
                                                , select_statement = sql.SQL(select_statement)
                                                , name = sql.SQL(self.name)
                                                )       

        ### Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score')
        self.create_acc_measure_function('score')
    
    def create_nn_table(self, trained_pipeline):

        conn, cur = self.create_postgres_connection()

        function_statement = 'DROP TABLE IF EXISTS pgml.nn_matrix_' + self.name + '; '
        function_statement += """ CREATE TABLE pgml.nn_matrix_""" + self.name
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

        insert_statement = """ INSERT INTO pgml.nn_matrix_""" + self.name + """ (id, row, col, val) VALUES """
        for idc, c in enumerate(trained_pipeline[-1].coefs_):
            for idx, weights in enumerate(c):
                for idw, w in enumerate(weights):
                    tup = (idc, idx, idw, w)
                    insert_statement += str(tup) + ','
        for idc, i in enumerate(trained_pipeline[-1].intercepts_):
            idc += len(trained_pipeline[-1].coefs_)
            for idx, bias in enumerate(i):
                tup = (idc, idx, 0, bias)
                if idc < len(trained_pipeline[-1].intercepts_) + len(trained_pipeline[-1].coefs_) - 1 and idx < len(i):
                    insert_statement += str(tup) + ','
                else:
                    insert_statement += str(tup) + ';'
        
        cur.execute(insert_statement)
        conn.commit()
        
        # return insert_statement

    def create_nn_scorer_function(self):

        conn, cur = self.create_postgres_connection()
        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name
        ### Creates a function to transform preprocessed arrays into predictions

        
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
                                    
                                    """).format(experiment_name = sql.SQL(self.name)
                                                , input_name = sql.SQL(input_name)
                                                , select_statement = sql.SQL(select_statement)
                                                )
        
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score')
        self.create_acc_measure_function('score')

    def create_acc_measure_function(self, scoring_function):

        conn, cur = self.create_postgres_connection()

        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        query = sql.SQL(""" with scores as(select sqrt(sum((ln(GREATEST(p.prediction, 0) + 1) - ln(t.target + 1))^2)/count(p.*)) as rmsle
                                        from pgml.{input_name}_{scoring_function}() p
                                        left join pgml.{experiment_name}_test t
                                            on p.row_id = t.row_id
                                        )""").format(input_name = sql.SQL(input_name)
                                                    , experiment_name = sql.SQL(self.name)
                                                    , scoring_function = sql.SQL(scoring_function))

        function_statement = sql.SQL(""" 
                                        --DROP FUNCTION IF EXISTS pgml.measure_acc_{input_name}_{scoring_function};
                                        CREATE OR REPLACE FUNCTION pgml.measure_acc_{input_name}_{scoring_function}(OUT score NUMERIC) RETURNS NUMERIC AS
                                        $$
                                        {query}

                                        select rmsle
                                        from scores;

                                        $$ LANGUAGE SQL;
                                    """).format(input_name = sql.SQL(input_name)
                                                , experiment_name = sql.SQL(self.name)
                                                , scoring_function = sql.SQL(scoring_function)
                                                , query = query)

        ### Create function
        cur.execute(function_statement)
        conn.commit()

    ############################################### KV SOLUTION

    def get_kv_tuples(self, trained_pipeline, purpose, path_to_data, target_feature, persist):

        df = pd.read_csv(path_to_data)
        encoder = Encoder('regression')
        
        X = df[self.training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        featurizer = NYC_Featurizer('shallow')
        featurizer.fit(X_train, y_train)
        features_names = list(featurizer.transform(X_train.head(5)))
        x_train_featurized = np.array(featurizer.transform(X_train))
        y_train_pred = trained_pipeline.predict(X_train)

        cat_mask = [idf for idf, i in enumerate(features_names) if i in featurizer.cat_features]

        self.experiment_training_instances = X_train.shape[0]
        self.experiment_input_dimensions = x_train_featurized.shape[1] 
        self.experiment_test_instances = X_test.shape[0]

        encoder.fit(x_train_featurized, y_train_pred, cat_mask)
        start = time.time()
        encoded_training_set = encoder.transform_dataset(x_train_featurized, [i for i in range(x_train_featurized.shape[1])])
        end = time.time() - start
        my_problem = Problem(encoded_training_set, y_train, encoder.num_bins, 'regression', 1)

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
        
        y_train_pred = np.where(y_train_pred < 0, 0, y_train_pred)

        if purpose != 'standalone':
            
            training_output = []
            for idt, t in enumerate(encoded_training_set):
                key = ''
                for idf, feature in enumerate(t):
                    if idf < len(t) - 1:
                        key = key + str(int(feature)) + '.'
                    else:
                        key = key + str(int(feature))
                training_output.append((key, y_train_pred[idt]))
            
            k_v_frame = pd.DataFrame(training_output, columns=['key', 'value'])
            k_v_aggregates = k_v_frame.groupby(['key']).mean()
            k_v_aggregates.reset_index(drop=False, inplace=True)
            tuples = k_v_aggregates.to_numpy()
        else:

            encoded_df= pd.DataFrame(encoded_training_set)
            target_variable_number = encoded_df.shape[1]
            encoded_df[target_variable_number] = y_train_pred
            agg_df = encoded_df.groupby([i for i in range(len(my_optimizer.greedy_solution))], as_index=False)[target_variable_number].mean()
            tuples = agg_df.to_numpy()

        if not persist: 
            return tuples, encoder, features_names, my_optimizer.greedy_solution
        else:
            self.persist_encoder_kv_tuples(tuples, encoder, features_names, my_optimizer.greedy_solution)

    def persist_encoder_kv_tuples(self, tuples, encoder, features_names, solution):

        project_folder = Path(__file__).resolve().parents[3]
        path = os.path.join(project_folder, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        if self.featurizer.depth == 'deep':
            tuples_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_complex_tuples.joblib')
            encoder_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_complex_encoder.joblib')
            features_names_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_complex_features_names.joblib')
            solution_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_complex_solution.joblib')
        else:
            tuples_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_simple_tuples.joblib')
            encoder_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_simple_encoder.joblib')
            features_names_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_simple_features_names.joblib')
            solution_path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_simple_solution.joblib')

        with open(tuples_path, "wb") as File:
            dump(tuples, File)
        
        with open(encoder_path, "wb") as File:
            dump(encoder, File)
        
        with open(features_names_path, "wb") as File:
            dump(features_names, File)
        
        with open(solution_path, "wb") as File:
            dump(solution, File)
    
    def create_kv_table(self):
        conn, cur = self.create_postgres_connection()

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        table_name = self.name + '_' + model_name + '_kv'
        ## Creates a table where key, predictions will be stored
        cur.execute(sql.SQL('DROP TABLE IF EXISTS pgml.{table_name} CASCADE; CREATE TABLE pgml.{table_name} (key TEXT NOT NULL, value NUMERIC NOT NULL)').format(table_name=sql.Identifier(table_name)))
        conn.commit()

    def insert_tuples_in_kv_table(self, tuples):
        conn, cur = self.create_postgres_connection()
        model_name = (self.pipeline[-1].__class__.__name__).lower()
        table_name = self.name + '_' + model_name + '_kv'
        ## Inserts keys and predictions in an already created table
        # tuples = self.get_kv_tuples('db')
        # sql_insert_query = sql.SQL("INSERT INTO pgml.{} (key, value) VALUES (%s,%s)").format(sql.Identifier(table_name))

        # tuples = [tuple(t) for t in tuples]
        # args_str = ','.join(cur.mogrify("(%s,%s)", tuple(x)).decode('utf-8') for x in tuples) 
        # cur.execute(sql.SQL("INSERT INTO pgml.{table_name} VALUES {args_str}").format(table_name=sql.Identifier(table_name), args_str = sql.SQL(args_str))) 

        buffer = StringIO()
        df = pd.DataFrame(tuples)
        df.to_csv(buffer, index=False, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert(sql.SQL("COPY pgml.{table_name} FROM STDIN WITH CSV DELIMITER ';'").format(table_name=sql.SQL(table_name)), buffer)

        # cur.executemany(sql_insert_query, tuples)
        conn.commit()
        print(cur.rowcount, "Records inserted successfully into " + table_name + " table")

    def create_index(self):
        conn, cur = self.create_postgres_connection()
        model_name = (self.pipeline[-1].__class__.__name__).lower()
        table_name = self.name + '_' + model_name + '_kv'
        index_name = self.name + '_' + (self.pipeline[-1].__class__.__name__).lower() + '_' + 'index'
        ## Creates an index for the keys using spgist
        cur.execute(sql.SQL('DROP INDEX IF EXISTS {index_name} CASCADE;CREATE INDEX {index_name} ON pgml.{table_name} using spgist(key); analyze;').format(index_name = sql.Identifier(index_name), table_name = sql.Identifier(table_name)))
        conn.commit()
    
    def create_preproccesing_function(self, encoder, features_names, solution_indices):

        conn, cur = self.create_postgres_connection()

        model_name = (self.model.__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        ### Creates a function to transform raw values into keys

        feature_list = [features_names[i] for i in solution_indices]

        select_statement = """ SELECT ROW_ID, """
        for idf, feature in enumerate(feature_list):
            feature_index = solution_indices[idf]
            feature_name = feature
            if feature not in self.featurizer.cat_features:
                ranges_list = list(encoder.bin_ranges[feature_index])
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
                ranges_list = encoder.embeddings[feature_index]
                embeddings = [[str(j) for j in i] for i in ranges_list]
                for idr, range in enumerate(embeddings):
                    range = [str(i) for i in range]
                    if idf < len(feature_list) - 1:
                        if idr == 0:
                            select_statement += 'CASE WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """.' ELSE """ + """'""" + str(idr + 1) + """.' END || """ 
                    else:
                        if idr == 0:
                            select_statement += 'CASE WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """'"""
                        elif idr < len(ranges_list) - 1:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """'"""
                        else:
                            select_statement += ' WHEN ' + feature_name + '::TEXT= ANY(ARRAY' + str(range) + ') THEN ' + """'""" + str(idr) + """' ELSE """ + """'""" + str(idr + 1) + """' END """ 


        function_statement = sql.SQL(""" --DROP FUNCTION IF EXISTS pgml.{input_name}_translate CASCADE;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_translate(OUT ROW_ID INTEGER, OUT key TEXT) returns SETOF record AS 
                                        $func$
                                        
                                        {select_statement} 
                                        FROM pgml.{name}_featurized_kv
                                        ORDER BY ROW_ID ASC;

                                        $func$ LANGUAGE SQL STABLE;
                                    """).format(input_name = sql.SQL(input_name)
                                                , name = sql.SQL(self.name)
                                                , select_statement = sql.SQL(select_statement)
                                                )

        # print(function_statement)
        ## Create function
        cur.execute(function_statement)
        conn.commit()

        persist_statement = sql.SQL(""" 
                                        DROP MATERIALIZED VIEW IF EXISTS pgml.{name}_translated;
                                        CREATE MATERIALIZED VIEW pgml.{name}_translated AS 
                                        
                                        SELECT * FROM  pgml.{input_name}_translate();
                                        
                                    """).format(name=sql.SQL(self.name),
                                                input_name = sql.SQL(input_name))
        
        cur.execute(persist_statement)
        conn.commit()

        

        self.create_measure_function('translate')
    
    def create_scoring_function_kv(self):
        conn, cur = self.create_postgres_connection()

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        function_statement = sql.SQL("""
                                        --DROP FUNCTION IF EXISTS pgml.{input_name}_score_kv CASCADE;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_score_kv(OUT row_id INTEGER, OUT prediction NUMERIC) RETURNS SETOF record AS 
                                        $func$
                                        select k1.ROW_ID, case when k2.value is not null then k2.value else pgml.prefix_search(k1.key, '{input_name}_kv') end as prediction
                                        from pgml.{name}_translated k1
                                        left join pgml.{input_name}_kv k2
                                        on k1.key = k2.key
                                        $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    """).format(input_name = sql.SQL(input_name)
                                                , name = sql.SQL(self.name))
        
        ## Create function
        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score_kv')
        self.create_acc_measure_function('score_kv')
    
    def create_prefix_search_udf(self):

        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL(""" CREATE OR REPLACE FUNCTION pgml.prefix_search(searched_key TEXT, search_table TEXT) returns NUMERIC AS 
                                            $$

                                            DECLARE 
                                            r INTEGER;
                                            prediction NUMERIC;
                                            not_found BOOLEAN;
                                            BEGIN
                                                r := 0;
                                                not_found := TRUE;
                                                WHILE not_found LOOP
                                                    r := r + 1;
                                                    execute format('SELECT count(*)=0 from pgml.%I where key ^@ left(%L, length(%L) - (%s*2))', search_table, searched_key, searched_key, r) INTO not_found;
                                                END LOOP;
                                                
                                                execute format('SELECT avg(value) from pgml.%I where key ^@ left(%L, length(%L) - (%s*2))', search_table, searched_key, searched_key, r) INTO prediction;
                                                
                                                RETURN prediction;
                                            END;
                                            $$
                                            LANGUAGE plpgsql STABLE PARALLEL SAFE;   
                                        """
                                     )
        cur.execute(function_statement)
        conn.commit()
    
    def train_model_in_pgml(self):

        conn, cur = self.create_postgres_connection()

        function_statement = sql.SQL(
                                        """
                                            SELECT pgml.train(
                                            project_name => '{experiment_name}'::text, 
                                            task => 'regression'::text, 
                                            relation_name => 'pgml.{experiment_name}_train'::text,
                                            y_column_name => 'target'::text,
                                            test_size => 0.1,
                                            algorithm => 'linear'
                                            
                                        );
                                    
                                        """
                                    ).format(
                                             experiment_name = sql.SQL(self.name)
                                             )
        
        cur.execute(function_statement)
        conn.commit()
    
    def deploy_pgml_trained_model(self):
        conn, cur = self.create_postgres_connection()

        

        function_statement = sql.SQL( """SELECT * FROM pgml.deploy(
                                    '{experiment_name}'::text,
                                    strategy => 'most_recent'
                                );""").format(experiment_name = sql.SQL(self.name))
        
        cur.execute(function_statement)
        conn.commit()
    
    def create_scorer_function_pgml(self):

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        function_statement = sql.SQL(""" 
                                        --DROP FUNCTION IF EXISTS pgml.{input_name}_score_pgml CASCADE;
                                        CREATE OR REPLACE FUNCTION pgml.{input_name}_score_pgml(OUT row_id INT, OUT prediction NUMERIC) returns SETOF RECORD AS $func$
                                        WITH predictions AS (
                                        SELECT row_number() over () as row_id
                                                , pgml.predict_batch('{experiment_name}'::text, d.m) AS prediction
                                        FROM pgml.{experiment_name}_scaled d
                                    )
                                    SELECT row_id - 1 as row_id, prediction 
                                    FROM predictions
                                    $func$ LANGUAGE SQL STABLE PARALLEL SAFE;
                                    
                                    """).format(input_name = sql.SQL(input_name)
                                                , experiment_name = sql.SQL(self.name)
                                                )
        
        conn, cur = self.create_postgres_connection()

        cur.execute(function_statement)
        conn.commit()

        self.create_measure_function('score_pgml')
        self.create_acc_measure_function('score_pgml')
    
    def create_measure_function(self, function):

        conn, cur = self.create_postgres_connection()

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name
    
        function_statement = sql.SQL("""
                    DROP FUNCTION IF EXISTS pgml.measure_{function}_runtime_{input_name} CASCADE;
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
                """).format(input_name = sql.SQL(input_name)
                            , function = sql.SQL(function))
        
        ### Create function
        cur.execute(function_statement)
        conn.commit()
    
    def generate_report_function_pg(self):

        conn, cur = self.create_postgres_connection()
        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        if self.model.__class__.__name__ in ('MLPRegressor', 'MLPClassifier'):
            coef_statement = 'nn_matrix_' + self.name
        elif self.model.__class__.__name__ in ('LogisticRegression', 'LinearRegression'):
            coef_statement = input_name + '_coefficients'

        out_statement = """, OUT rmsle NUMERIC"""
        select_statement_model = sql.SQL(""" , pgml.measure_acc_{input_name}_score() as rmsle""").format(input_name = sql.SQL(input_name))
        select_statement_index = sql.SQL(""", pgml.measure_acc_{input_name}_score_kv() as rmsle""").format(input_name = sql.SQL(input_name))
        select_statement_pgml = sql.SQL(""", pgml.measure_acc_{input_name}_score_pgml() as rmsle""").format(input_name = sql.SQL(input_name))
        
        #### Sizes

        project_folder = Path(__file__).resolve().parents[3]
        path = os.path.join(project_folder, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)
        
        with open(path + '/' + input_name + '.joblib', "wb") as File:
            dump(self.pipeline[-1], File)
        model_size = getsize(path + '/' + input_name + '.joblib') 
        
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
                                     ).format(input_name = sql.SQL(input_name)
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
    
    def create_report_pg(self, batch_size):

        DEC2FLOAT = new_type(
        DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
        register_type(DEC2FLOAT)

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        conn, cur = self.create_postgres_connection()

        size_effectiveness_query = sql.SQL(""" select * from pgml.{input_name}_report()""").format(input_name = sql.SQL(input_name))

        cur.execute(size_effectiveness_query)

        result = cur.fetchall()

        results = []
        for row in result:
            results.append(list(row))
        
        #### Get performance numbers for model representation:

        impute_time_query = sql.SQL(""" analyze; select * from pgml.measure_featurizer_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))
        encode_scale_time_query = sql.SQL(""" analyze; select * from pgml.measure_encode_scale_set_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))
        scoring_time_query = sql.SQL(""" analyze; select * from pgml.measure_score_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))

        queries = [impute_time_query, encode_scale_time_query, scoring_time_query]
        for q in queries:
            cur.execute(q)
            r = cur.fetchone()
            results[0].append(r[0])
        
        #### Get performance numbers for Index:
        impute_time_query = sql.SQL(""" analyze; select * from pgml.measure_featurizer_kv_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))
        translate_time_query = sql.SQL(""" analyze; select * from pgml.measure_translate_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))
        scoring_time_query = sql.SQL(""" analyze; select * from pgml.measure_score_kv_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))

        queries = [impute_time_query, translate_time_query, scoring_time_query]
        for q in queries:
            cur.execute(q)
            r = cur.fetchone()
            results[1].append(r[0])
        
        #### Get performance numbers for pgml representation:

        impute_time_query = sql.SQL(""" analyze; select * from pgml.measure_featurizer_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))
        encode_scale_time_query = sql.SQL(""" analyze; select * from pgml.measure_encode_scale_set_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))
        scoring_time_query = sql.SQL(""" analyze; select * from pgml.measure_score_pgml_runtime_{input_name}()""").format(input_name = sql.SQL(input_name))

        queries = [impute_time_query, encode_scale_time_query, scoring_time_query]
        for q in queries:
            cur.execute(q)
            r = cur.fetchone()
            results[2].append(r[0])
        
        
        columns = ['Solution', 'Size (B)', 'RMSLE', 'Impute Featurize Latency (ms)', 'Encode Scale Latency (ms)', 'Score Latency (ms)']

        summary_df = pd.DataFrame(results, columns=columns)
        summary_df['End-to-End Latency (ms)'] = summary_df.apply(lambda x: x['Impute Featurize Latency (ms)'] + x['Encode Scale Latency (ms)'] + x['Score Latency (ms)'], axis=1)
        summary_df['Experiment'] = self.name
        summary_df['Algorithm'] = model_name
        summary_df['Batch Size (Records)'] = batch_size

        return summary_df

    def create_standalone_index(self, trained_pipeline, path_to_data, target_feature):

        # Populate index
        index_time = 0
        model_index = Trie('regression')
        st = time.time()
        tuples, encoder, features_names, solution  = self.get_kv_tuples(trained_pipeline, 'standalone', path_to_data, target_feature, False)
        for idx, i in enumerate(tuples):
            key = i[:len(solution)]
            value = round(i[-1])
            model_index.insert(key, value)
        index_time = time.time() - st
        
        print('index populated')

        return model_index, index_time, encoder, solution
    
    def perform_index_inference(self, trained_pipeline, path_to_data, target_feature):

        df = pd.read_csv(path_to_data)
        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        self.training_features = [i for i in list(df) if i != target_feature]
        X = df[self.training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)
        

        model_index, time_to_populate_index, encoder, solution = self.create_standalone_index(trained_pipeline, path_to_data, target_feature)
        self.max_path_length = len(solution)

        X_test.reset_index(inplace=True, drop=True)
        sample = X_test.sample(frac=0.01, random_state = 50)
        indices = sample.index.values.tolist()
        sample.reset_index(inplace=True, drop=True)
        inference_runtimes = np.zeros(sample.shape[0], dtype=float)
        preprocessing_runtimes = np.zeros(sample.shape[0], dtype=float)
        y_pred_trie = np.zeros_like(inference_runtimes)

        featurizer = NYC_Featurizer('shallow')
        featurizer.fit(X_train, y_train)

        for index, row in sample.iterrows():
            start = time.time()
            x_featurized = np.array(featurizer.transform(row))[0]
            instance = x_featurized[solution]
            preprocessed_instance = encoder.transform_single(instance, solution)
            preprocessing_runtime = time.time() - start
            preprocessing_runtimes[index] = preprocessing_runtime
            st = time.time()
            y_pred_trie[index] = round(model_index.query(preprocessed_instance))
            inference_runtimes[index] = time.time() - st
        
        index_avg_prep_runtimes = preprocessing_runtimes.mean()
        index_avg_scoring_runtimes = inference_runtimes.mean()

        print('index inference done')

        #### Sizes
        project_folder = Path(__file__).resolve().parents[3]
        path = os.path.join(project_folder, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + input_name + '_' + 'index.joblib', "wb") as File:
            dump(model_index, File)
        index_size = getsize(path + '/' + input_name + '_' + 'index.joblib') 

        #### Error

        index_error = mean_squared_log_error(y_test[indices], y_pred_trie, squared=False)
        
        return index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_error, time_to_populate_index
    
    def perform_model_inference(self, trained_pipeline, path_to_data, target_feature):
        df = pd.read_csv(path_to_data)
        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name

        self.training_features = [i for i in list(df) if i != target_feature]
        X = df[self.training_features]
        y = df[target_feature].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

        sample = X_test.sample(frac=0.01, random_state = 50)
        sample.reset_index(inplace=True, drop=True)
        inference_runtimes = np.zeros(sample.shape[0], dtype=float)
        preprocessing_runtimes = np.zeros(sample.shape[0], dtype=float)
        for index, row in sample.iterrows():
        # for idx, i in enumerate(X_test):
            start = time.time()
            # instance = i.reshape((1, i.size))
            transformed_instance = trained_pipeline[:-1].transform(row)
            end = time.time() - start
            preprocessing_runtimes[index] = end
            self.experiment_transformed_input_size = transformed_instance.shape[1]
            start = time.time()
            trained_pipeline[-1].predict(transformed_instance)
            end = time.time() - start
            inference_runtimes[index] = end
        
        model_avg_prep_runtimes = preprocessing_runtimes.mean()
        model_avg_scoring_runtimes = inference_runtimes.mean()

        print('model inference done')

        #### Sizes
        project_folder = Path(__file__).resolve().parents[3]
        path = os.path.join(project_folder, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + '/' + input_name + '.joblib', "wb") as File:
            dump(self.pipeline, File)
        model_size = getsize(path + '/' + input_name + '.joblib') 

        #### Error

        y_pred = trained_pipeline.predict(X_test)
        y_pred = np.where(y_pred < 0, 0, y_pred)

        model_error = mean_squared_log_error(y_test, y_pred, squared=False)

        return model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_error
        
        
    def create_report(self, trained_pipeline, path_to_data, target_feature):

        model_name = (self.pipeline[-1].__class__.__name__).lower()
        input_name = self.name + '_' + model_name
        
        model_avg_prep_runtimes, model_avg_scoring_runtimes, model_size, model_error = self.perform_model_inference(trained_pipeline, path_to_data, target_feature)
        index_avg_prep_runtimes, index_avg_scoring_runtimes, index_size, index_error, index_time = self.perform_index_inference(trained_pipeline, path_to_data, target_feature)
        summary = [[self.name, (self.pipeline[-1].__class__.__name__).lower(), self.experiment_training_instances, self.experiment_input_dimensions, self.experiment_transformed_input_size, self.preproc_training_time ,self.training_time, self.encoding_time, self.solution_time, index_time, self.max_path_length, self.experiment_test_instances, model_error, index_error, model_avg_prep_runtimes, model_avg_scoring_runtimes, model_avg_prep_runtimes + model_avg_scoring_runtimes, index_avg_prep_runtimes, index_avg_scoring_runtimes, index_avg_prep_runtimes + index_avg_scoring_runtimes, model_size, index_size]]
        columns = ['name', 'model', 'training_instances', 'original_dimensions', 'input_dimensions', 'training_preprocessing_runtime' ,'training_runtime', 'encoding_runtime', 'solution_runtime', 'index_runtime', 'max_path_length', 'testing_instances', 'model_error', 'index_error', 'model_avg_preprocessing_runtime', 'model_avg_scoring_runtime', 'model_end_to_end_runtime', 'index_avg_preprocessing_runtime', 'index_avg_scoring_runtime', 'index_end_to_end_runtime', 'model_size', 'index_size']
    
        summary_df = pd.DataFrame(summary, columns=columns)
        project_folder = Path(__file__).resolve().parents[3]
        path = os.path.join(project_folder, 'experiments', 'output')
        Path(path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, input_name)

        return summary_df
    
    def persist_experiment(self):

        #### Sizes
        project_folder = Path(__file__).resolve().parents[3]
        path = os.path.join(project_folder, 'objects')
        Path(path).mkdir(parents=True, exist_ok=True)

        if self.featurizer.depth == 'deep':
            path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_complex_experiment.joblib')
        else:
            path = os.path.join(path, self.name + '_' + (self.model.__class__.__name__).lower() + '_simple_experiment.joblib')

        with open(path, "wb") as File:
            dump(self, File)
