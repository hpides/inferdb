import os
from pathlib import Path 
import sys
project_folder = Path(__file__).resolve().parents[2]
src_folder = os.path.join(project_folder, 'src')
featurizer_folder = os.path.join(project_folder, 'experiments', 'microbenchmarks', 'preprocessing')
sys.path.append(str(src_folder))
sys.path.append(str(featurizer_folder))
from transpiler import SQLmodel, InferDB, PGML
from sklearn.neural_network import MLPRegressor
import pandas as pd
from nyc_rides_featurizer import NYC_Featurizer
import re
from psycopg2 import sql
from io import StringIO

def nyc_rides_nn(batch_sizes):

    inferdb = InferDB('nyc_rides', MLPRegressor(max_iter=10000, activation='logistic'), 'regression', 1, False, True, 'trip_duration', NYC_Featurizer('deep'))
    sqlmodel = SQLmodel('nyc_rides', MLPRegressor(max_iter=10000, activation='logistic'), 'regression', True, 'trip_duration', NYC_Featurizer('deep'))
    pgml = PGML('nyc_rides', MLPRegressor(max_iter=10000, activation='logistic'), 'regression', True, 'trip_duration', NYC_Featurizer('deep'))

    df = pd.DataFrame()
    
    def text_normalizer_select_statement(text):

        rep = {"-inf": "'-infinity'::numeric", "inf": "'infinity'::numeric"} # define desired replacements here

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pattern = re.compile("|".join(rep.keys()))

        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    
    def create_featurizer(pickup_encoder=sqlmodel.featurizer.pickup_encoder.splits, dropoff_encoder=sqlmodel.featurizer.dropoff_encoder.splits, cluster_encoder=sqlmodel.featurizer.clusters_encoder.splits, freq_mapper=sqlmodel.featurizer.route_freq_mappers, batch_size=100):


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
        
            pickup_cluster = text_normalizer_select_statement(pickup_cluster)
            dropoff_cluster = text_normalizer_select_statement(dropoff_cluster)

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

                                            FROM pgml.nyc_rides_test ORDER BY ROW_ID ASC LIMIT {limit_factor}
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
            
            clusters_cluster = text_normalizer_select_statement(clusters_cluster)


            featurizer_statement = sql.SQL("""SELECT
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
                                        
                                            """
                                        ).format(subquery = subquery
                                                    , clusters_cluster = sql.SQL(clusters_cluster)
                                                    )
            return_type_statement = sql.SQL("""DROP TYPE IF EXISTS pgml.nyc_rides_featurizer CASCADE; 
                                            CREATE TYPE pgml.nyc_rides_featurizer AS (
                                                    row_id INTEGER, 
                                                    passenger_count INTEGER, 
                                                    day TEXT, 
                                                    hour INTEGER,
                                                    is_weekend INTEGER, 
                                                    month INTEGER, 
                                                    pickup_hour_of_day TEXT, 
                                                    distance REAL, 
                                                    route_freq INTEGER, 
                                                    freq_dist REAL, 
                                                    clusters_cluster TEXT
                                                ); 
                                                
                                            """)
            def create_route_freq_table():

                conn, cur = sqlmodel.create_postgres_connection()

                cur.execute("DROP TABLE IF EXISTS pgml.route_freq CASCADE; CREATE TABLE pgml.route_freq(cluster_combination TEXT, day TEXT, hour INTEGER, count REAL, PRIMARY KEY(cluster_combination, day, hour))")
                conn.commit()
            
            def insert_route_freq_tuples(route_freq_mapper):
                conn, cur = sqlmodel.create_postgres_connection()

                tuples = [(d[0], d[1], d[2], route_freq_mapper.get(d)['count']) for d in route_freq_mapper]

                buffer = StringIO()
                df = pd.DataFrame(tuples)
                df.to_csv(buffer, index=False, header=False, sep=';')
                buffer.seek(0)

                cur.copy_expert(sql.SQL("COPY pgml.route_freq FROM STDIN WITH CSV DELIMITER ';'"), buffer)

                conn.commit()
            
            create_route_freq_table()
            insert_route_freq_tuples(freq_mapper)
        
        else:

            featurizer_statement = sql.SQL("""SELECT 
                                                        row_id
                                                        , passenger_count
                                                        , TRIM(to_char(pickup_datetime, 'Day')) as day
                                                        , extract(hour from pickup_datetime) as hour
                                                        , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                                        , extract(month from pickup_datetime) as month
                                                        , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                                                    FROM pgml.nyc_rides_test ORDER BY ROW_ID ASC LIMIT {limit_factor}
                                            """).format(limit_factor = sql.SQL(str(batch_size)))
            
            return_type_statement = sql.SQL("""DROP TYPE IF EXISTS pgml.nyc_rides_featurizer CASCADE; 
                                            CREATE TYPE pgml.nyc_rides_featurizer AS (
                                                    row_id INTEGER, 
                                                    passenger_count INTEGER, 
                                                    day TEXT, 
                                                    hour INTEGER,
                                                    is_weekend INTEGER, 
                                                    month INTEGER, 
                                                    distance REAL
                                                ); 
                                                
                                            """)

        return featurizer_statement, return_type_statement
    
    for i in range(5):
        for b in batch_sizes:
            featurize_query = sql.SQL(""" SELECT row_id
                                                , passenger_count
                                                , TRIM(to_char(pickup_datetime, 'Day')) as day
                                                , extract(hour from pickup_datetime) as hour
                                                , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                                , extract(month from pickup_datetime)::INTEGER as month
                                                , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

                                            FROM pgml.nyc_rides_test ORDER BY ROW_ID ASC LIMIT {limit}
                                """
                              ).format(limit = sql.SQL(str(b)))
            inferdb_df = inferdb.create_report_pg(b, NYC_Featurizer('shallow'), featurize_query)
            inferdb_df['Iteration'] = i
            sql_featurizer, return_type_sql = create_featurizer(batch_size=b)
            sqlmodel_df = sqlmodel.create_report_pg(b, sql_featurizer, return_type_sql)
            sqlmodel_df['Iteration'] = i
            pgml_df = pgml.create_report_pg(b, sql_featurizer)
            pgml_df['Iteration'] = i

            df = pd.concat([df, inferdb_df, sqlmodel_df, pgml_df])
            df = pd.concat([df, inferdb_df, sqlmodel_df])

    experiment_folder = Path(__file__).resolve().parents[1]

    path = os.path.join(experiment_folder, 'output')
    path = os.path.join(path, inferdb.experiment_name + '_' + inferdb.model_name)
    df.to_csv(path + '_pg.csv', index=False)

if __name__ == "__main__":
    nyc_rides_nn()