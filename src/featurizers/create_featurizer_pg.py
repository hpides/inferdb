import sys
import os
from pathlib import Path
import pandas as pd
import re
from psycopg2 import sql
from io import StringIO
from nyc_rides_featurizer import NYC_Featurizer 
from database_connect import connect

class NYC_Featurizer_pg:

    def __init__(self, fitted_featurizer:NYC_Featurizer) -> None:
        self.cluster_centers = fitted_featurizer.kmeans.cluster_centers_
        self.outlier_boundaries = fitted_featurizer.outlier_boundaries
        self.outlier_training_features = fitted_featurizer.outlier_training_features
        self.outlier_imputers = fitted_featurizer.outlier_imputers
        self.pca_components = fitted_featurizer.pca.components_
        self.pca_means = fitted_featurizer.pca.mean_
        self.cluster_mappers = fitted_featurizer.cluster_mappers
        self.section = 'pgml'
    
    def create_postgres_connection(self):
        conn = connect(self.section)
        cur = conn.cursor()

        return conn, cur
    
    def create_aux_functions(self):

        conn, cur = self.create_postgres_connection()

        parent = Path(__file__).resolve().parents[2]
        
        geo_distance = os.path.join(parent, 'sql_scripts/geo_distance.sql')
        prefix_search = os.path.join(parent, 'sql_scripts/prefix_search.sql')

        gd = open(geo_distance, 'r')
        ps = open(prefix_search, 'r')

        gdr = gd.read()
        psr = ps.read()

        sql_gd = sql.SQL(gdr)
        sql_ps = sql.SQL(psr)

        cur.execute(sql_gd)
        conn.commit()
    
        cur.execute(sql_ps)
        conn.commit()
    
    def text_normalizer_select_statement(text):

        rep = {"-inf": "'-infinity'::numeric", "inf": "'infinity'::numeric"} # define desired replacements here

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pattern = re.compile("|".join(rep.keys()))

        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    
    def push_cluster_table(self):

        conn, cur = self.create_postgres_connection()

        cur.execute("""DROP TABLE IF EXISTS pgml.nyc_rides_clusters CASCADE; 
                    CREATE TABLE pgml.nyc_rides_clusters(cluster_id INTEGER, latitude REAL, longitude REAL, PRIMARY KEY(cluster_id))"""
                    )
        conn.commit()

        buffer = StringIO()
        df = pd.DataFrame(self.cluster_centers)
        df.to_csv(buffer, index=True, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert("COPY pgml.nyc_rides_clusters FROM STDIN WITH CSV DELIMITER ';'", buffer)
        conn.commit()
    
    def push_mappers_table(self):

        conn, cur = self.create_postgres_connection()

        cur.execute("""DROP TABLE IF EXISTS pgml.nyc_rides_mappers CASCADE; 
                    CREATE TABLE pgml.nyc_rides_mappers(pickup_cluster INTEGER, dropoff_cluster INTEGER, avg_distance REAL, avg_travel_time REAL, avg_cnt_of_steps REAL, cnt REAL, avg_trip_duration REAL, avg_speed REAL, PRIMARY KEY(pickup_cluster, dropoff_cluster))"""
                    )
        conn.commit()

        buffer = StringIO()
        self.cluster_mappers.to_csv(buffer, index=True, header=False, sep=';')
        buffer.seek(0)

        cur.copy_expert("COPY pgml.nyc_rides_mappers FROM STDIN WITH CSV DELIMITER ';'", buffer)
        conn.commit()

    def create_clusters_query(self, limit_factor=-1):

        if limit_factor > 0:
            source_table = sql.SQL("""pgml.nyc_rides_test WHERE ROW_ID < {limit_factor} ORDER BY ROW_ID ASC""").format(limit_factor = sql.SQL(str(limit_factor)))
        else:
            source_table = sql.SQL("""pgml.nyc_rides_test ORDER BY ROW_ID ASC""")

        query  = sql.SQL(""" 
                    distances_2_clusters as (
                                                    select row_id, cluster_id, SQRT((cl.latitude - t.pickup_latitude)^2 + (cl.longitude - t.pickup_longitude)^2) as pickup_distance,
                                                                            SQRT((cl.latitude - t.dropoff_latitude)^2 + (cl.longitude - t.dropoff_longitude)^2) as dropoff_distance
                                                    from pgml.nyc_rides_clusters cl, (select * from {source_table}) t
                                                    ),
                    rank as (
                                select row_id, cluster_id, row_number () over (partition by row_id order by pickup_distance asc) as pickup_rank, row_number () over (partition by row_id order by dropoff_distance asc) as dropoff_rank
                                from distances_2_clusters
                    ),

                    pickup_clusters as (
                    
                                select row_id, cluster_id as pickup_cluster
                                from rank
                                where pickup_rank = 1
                    ),

                    dropoff_clusters as (
                    
                                select row_id, cluster_id as dropoff_cluster
                                from rank
                                where dropoff_rank = 1
                    ),

                    clusters as (
                    
                                select pu.row_id, pickup_cluster, dropoff_cluster
                                from pickup_clusters pu
                                left join dropoff_clusters dr
                                on pu.row_id = dr.row_id
                    )
                
                    """).format(source_table=source_table)

        return query.as_string('conn')

    def create_pca_query(self, limit_factor=-1):
        
        k00 = self.pca_components[0][0]
        k01 = self.pca_components[0][1]
        k10 = self.pca_components[1][0]
        k11 = self.pca_components[1][1]

        avg_latitude = self.pca_means[0]
        avg_longitude = self.pca_means[1]
        
        query = sql.SQL("""
                            pca_components as ( 
                                    select row_id, (pickup_latitude - {avg_latitude}) * {k00} + (pickup_longitude - {avg_longitude}) * {k01} as pickup_pca0, (pickup_latitude - {avg_latitude}) * {k10} + (pickup_longitude - {avg_longitude}) * {k11} as pickup_pca1,
                                                    (dropoff_latitude - {avg_latitude}) * {k00} + (dropoff_longitude - {avg_longitude}) * {k01} as dropoff_pca0, (dropoff_latitude - {avg_latitude}) * {k10} + (dropoff_longitude - {avg_longitude}) * {k11} as dropoff_pca1
                                    from pgml.nyc_rides_test
                        
                            ),
                        
                            pca as (
                                    select row_id, pickup_pca0, pickup_pca1, dropoff_pca0, dropoff_pca1, 
                                    ABS(dropoff_pca1 - pickup_pca1) + ABS(dropoff_pca0 - pickup_pca0) as pca_manhattan
                                    from pca_components)
                        """).format(k00 = sql.SQL(str(k00)),
                                    k01 = sql.SQL(str(k01)),
                                    k10 = sql.SQL(str(k10)),
                                    k11 = sql.SQL(str(k11)),
                                    avg_latitude = sql.SQL(str(avg_latitude)),
                                    avg_longitude = sql.SQL(str(avg_longitude))
                                    )
        return query.as_string('conn')


    def create_imputation_query(self):

        imputation_query = """ imputation_query as (SELECT row_id, """

        for idf, feature in enumerate(self.outlier_imputers):

            training_features = self.outlier_training_features[feature]
            intercept = self.outlier_imputers[feature][0]
            lower_boundary = self.outlier_boundaries[feature][0]
            upper_boundary = self.outlier_boundaries[feature][1]

            imputation_query += sql.SQL(""" CASE WHEN {feature} > {lower_boundary} 
                                        AND {feature} < {upper_boundary} 
                                        THEN {feature} 
                                        ELSE ({intercept} + """).format(feature = sql.SQL(feature.lower()),
                                                                        intercept = sql.SQL(str(intercept)),
                                                                        lower_boundary = sql.SQL(str(lower_boundary)),
                                                                        upper_boundary = sql.SQL(str(upper_boundary))
                                                                        ).as_string('conn')

            for idt, training_feature in enumerate(training_features):

                coef = self.outlier_imputers[feature][1][idt]
                

                if idt < len(training_features) - 1:

                    imputation_query += sql.SQL(""" {training_feature} * {coef} + """).format(training_feature = sql.SQL(training_feature.lower()),
                                                                                            coef = sql.SQL(str(coef))
                                                                                            ).as_string('conn')
                else:
                    if idf < len(self.outlier_imputers) - 1:
                        imputation_query += sql.SQL(""" {training_feature} * {coef}) END as {feature}, """).format(training_feature = sql.SQL(training_feature.lower()),
                                                                                                coef = sql.SQL(str(coef)),
                                                                                                feature = sql.SQL(feature)
                                                                                                ).as_string('conn')
                    else:
                        imputation_query += sql.SQL(""" {training_feature} * {coef}) END as {feature} """).format(training_feature = sql.SQL(training_feature.lower()),
                                                                                                coef = sql.SQL(str(coef)),
                                                                                                feature = sql.SQL(feature)
                                                                                                ).as_string('conn')
        imputation_query += " FROM basic_datetime_extractions)"
                        
        return imputation_query
    
    
    def create_featurizer_query(self, limit_factor=-1):

        cluster_query = self.create_clusters_query()
        pca_query = self.create_pca_query()
        imputation_query = self.create_imputation_query()

        if limit_factor > 0:
            source_table = sql.SQL("""pgml.nyc_rides_test WHERE ROW_ID < {limit_factor} ORDER BY ROW_ID ASC""").format(limit_factor = sql.SQL(str(limit_factor)))
        else:
            source_table = sql.SQL("""pgml.nyc_rides_test ORDER BY ROW_ID ASC""")

    
        query = sql.SQL(""" 
                                WITH 
                                    {pca_query}
                        
                                , basic_datetime_extractions as (    
                                    SELECT row_id
                                            , extract(isodow from pickup_datetime) - 1 as pickup_weekday
                                            , extract(week from pickup_datetime) as pickup_weekofyear
                                            , extract(hour from pickup_datetime) as pickup_hour
                                            , extract(minute from pickup_datetime) as pickup_minute
                                            , CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
                                            , pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as geo_distance
                                            , srccounty
                                            , dstcounty
                                            , vendor_id
                                            , passenger_count 
                                            , pickup_longitude
                                            , pickup_latitude
                                            , dropoff_longitude
                                            , dropoff_latitude
                                            , distance
                                            , duration
                                            , motorway
                                            , trunk
                                            , primarie
                                            , secondary
                                            , tertiary
                                            , unclassified
                                            , residential
                                            , ntrafficsignals
                                            , ncrossing
                                            , nstop
                                            , nintersection
                                    FROM {source_table})
                        

                        SELECT 
                                bde.row_id
                                , bde.pickup_weekday
                                , bde.is_weekend
                                , bde.pickup_weekofyear
                                , bde.pickup_hour
                                , bde.pickup_minute
                                , bde.srccounty
                                , bde.dstcounty
                                , bde.vendor_id
                                , bde.passenger_count
                                , bde.pickup_longitude
                                , bde.pickup_latitude
                                , bde.dropoff_longitude
                                , bde.dropoff_latitude
                                , bde.distance
                                , bde.duration
                                , bde.motorway
                                , bde.trunk
                                , bde.primarie
                                , bde.secondary
                                , bde.tertiary
                                , bde.unclassified
                                , bde.residential
                                , bde.ntrafficsignals
                                , bde.ncrossing
                                , bde.nstop
                                , bde.nintersection
                                , pca.pickup_pca0
                                , pca.pickup_pca1
                                , pca.dropoff_pca0
                                , pca.dropoff_pca1
                                , pca.pca_manhattan
                                , bde.geo_distance
                        FROM basic_datetime_extractions bde
                        LEFT JOIN pca 
                            ON bde.row_id = pca.row_id;
                        """
                        ).format(
                                 pca_query = sql.SQL(pca_query),
                                 
                                 source_table = source_table
                                 )
        
        return query
    
    
    def push_featurizer_to_pg(self):

        self.create_aux_functions()

        conn, cur = self.create_postgres_connection()

        self.push_cluster_table()
        self.push_mappers_table()

        featurizer_query = self.create_featurizer_query()

        table_creation_query = sql.SQL("""
                                            DROP TABLE IF EXISTS pgml.nyc_rides_featurized CASCADE;
                                            CREATE TABLE pgml.nyc_rides_featurized AS
                                       
                                            {featurizer_query}
                                        """).format(featurizer_query = featurizer_query)

        cur.execute(table_creation_query)

        conn.commit()

