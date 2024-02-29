with featurized_data as
(
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
	, CASE  WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (13, 8) THEN 0 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (13, 8) THEN 1 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (13, 8) THEN 2 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (13, 8) THEN 3 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (13, 8) THEN 4 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (18, 12) THEN 5 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (18, 12) THEN 6 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (18, 12) THEN 7 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (18, 12) THEN 8 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (18, 12) THEN 9 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (7, 2, 14) THEN 10 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (7, 2, 14) THEN 11 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (7, 2, 14) THEN 12 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (7, 2, 14) THEN 13 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (7, 2, 14) THEN 14 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 15 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 16 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 17 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 18 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (17, 6, 1, 19, 16, 11) THEN 19 WHEN s.pickup_cluster IN (4, 3, 16, 14, 13, 9) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 20 WHEN s.pickup_cluster IN (21, 24, 10, 19, 12, 18) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 21 WHEN s.pickup_cluster IN (8, 7, 2) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 22 WHEN s.pickup_cluster IN (11, 5, 6, 1, 17, 15, 22) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 23 WHEN s.pickup_cluster IN (0, 23, 20) AND s.dropoff_cluster IN (23, 3, 5, 22, 9, 15, 10, 24, 4, 21, 0, 20) THEN 24 END  as clusters_cluster
	FROM 
	(
		SELECT
		row_id
		, passenger_count
		, TRIM(to_char(pickup_datetime, 'Day')) as day
		, extract(hour from pickup_datetime) as hour
		, CASE  WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 0 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 1 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 2 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 3 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN '-infinity'::numeric and -74.00587844848633 THEN 4 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 5 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 6 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 7 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 8 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -74.00587844848633 and -73.9825553894043 THEN 9 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 10 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 11 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 12 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 13 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.9825553894043 and -73.95190048217773 THEN 14 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 15 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 16 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 17 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 18 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.95190048217773 and -73.89102554321289 THEN 19 WHEN pickup_latitude BETWEEN '-infinity'::numeric and 40.70807075500488 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 20 WHEN pickup_latitude BETWEEN 40.70807075500488 and 40.72068977355957 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 21 WHEN pickup_latitude BETWEEN 40.72068977355957 and 40.76872444152832 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 22 WHEN pickup_latitude BETWEEN 40.76872444152832 and 40.77431678771973 AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 23 WHEN pickup_latitude BETWEEN 40.77431678771973 and 'infinity'::numeric AND pickup_longitude BETWEEN -73.89102554321289 and 'infinity'::numeric THEN 24 END  as pickup_cluster
		, CASE  WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 0 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 1 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 2 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 3 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN '-infinity'::numeric and -74.00753402709961 THEN 4 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 5 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 6 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 7 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 8 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -74.00753402709961 and -73.97815322875977 THEN 9 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 10 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 11 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 12 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 13 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.97815322875977 and -73.94842910766602 THEN 14 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 15 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 16 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 17 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 18 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.94842910766602 and -73.92588424682617 THEN 19 WHEN dropoff_latitude BETWEEN '-infinity'::numeric and 40.70149040222168 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 20 WHEN dropoff_latitude BETWEEN 40.70149040222168 and 40.72253227233887 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 21 WHEN dropoff_latitude BETWEEN 40.72253227233887 and 40.77517890930176 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 22 WHEN dropoff_latitude BETWEEN 40.77517890930176 and 40.79763603210449 AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 23 WHEN dropoff_latitude BETWEEN 40.79763603210449 and 'infinity'::numeric AND dropoff_longitude BETWEEN -73.92588424682617 and 'infinity'::numeric THEN 24 END  as dropoff_cluster
		, CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
		, extract(month from pickup_datetime) as month
		, CASE WHEN extract(hour from pickup_datetime) BETWEEN 6 and 11 THEN 'morning'
		WHEN extract(hour from pickup_datetime) BETWEEN 12 and 15 THEN 'afternoon'
		WHEN extract(hour from pickup_datetime) BETWEEN 16 and 21 THEN 'evening'
		ELSE 'late_night' END as pickup_hour_of_day
		, pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

		FROM (SELECT * FROM pgml.nyc_rides_test) zt
	) as s
	LEFT JOIN pgml.route_freq as rq
	ON
	s.pickup_cluster || '_' || s.dropoff_cluster = rq.cluster_combination
	AND
	s.day = rq.day
	AND
	s.hour = rq.hour
	where passenger_count > 6
),

encode_scaled as
(
	
	SELECT row_id,ARRAY[((passenger_count-(1.6628402817959693))/1.3122565670353086)::real, ((distance-(3.4471878326333023))/4.314937461985763)::real, 
				 ((hour-(13.60859126695753))/6.401555533353099)::real, ((route_freq-(356.7411458861792))/417.43107981115435)::real, 
				 ((freq_dist-(650.1091490302825))/820.8054068657466)::real,  
				 (CASE WHEN day='Friday' THEN 1 ELSE 0 END)::real,  (CASE WHEN day='Monday' THEN 1 ELSE 0 END)::Real,  
				 CASE WHEN day='Saturday' THEN 1 ELSE 0 END,  CASE WHEN day='Sunday' THEN 1 ELSE 0 END,  
				 CASE WHEN day='Thursday' THEN 1 ELSE 0 END,  CASE WHEN day='Tuesday' THEN 1 ELSE 0 END,  
				 CASE WHEN day='Wednesday' THEN 1 ELSE 0 END,  CASE WHEN is_weekend='0' THEN 1 ELSE 0 END,  
				 CASE WHEN is_weekend='1' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='0' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='1' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='2' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='3' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='4' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='5' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='6' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='7' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='8' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='9' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='10' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='11' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='12' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='13' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='14' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='15' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='16' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='17' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='18' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='19' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='20' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='21' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='22' THEN 1 ELSE 0 END,  
				 CASE WHEN clusters_cluster='23' THEN 1 ELSE 0 END,  CASE WHEN clusters_cluster='24' THEN 1 ELSE 0 END,  
				 CASE WHEN month='1' THEN 1 ELSE 0 END,  CASE WHEN month='2' THEN 1 ELSE 0 END,  CASE WHEN month='3' THEN 1 ELSE 0 END,  
				 CASE WHEN month='4' THEN 1 ELSE 0 END,  CASE WHEN month='5' THEN 1 ELSE 0 END,  CASE WHEN month='6' THEN 1 ELSE 0 END,  
				 CASE WHEN pickup_hour_of_day='afternoon' THEN 1 ELSE 0 END,  CASE WHEN pickup_hour_of_day='evening' THEN 1 ELSE 0 END,  
				 CASE WHEN pickup_hour_of_day='late_night' THEN 1 ELSE 0 END,  CASE WHEN pickup_hour_of_day='morning' THEN 1 ELSE 0 END] as m 
	FROM featurized_data
),


preprocessed_data
as(

    SELECT m.row_id, u.ord - 1 as col_id, u.val
    FROM   encode_scaled m,
        LATERAL unnest(m.m) WITH ORDINALITY AS u(val, ord)
    where u.val != 0
),

intercept as (select val from pgml.nyc_rides_linearregression_coefficients where col_id = -1),

coef as (select pre.row_id, sum(pre.val * b.val) as val
        from preprocessed_data
        left join pgml.nyc_rides_linearregression_coefficients b
        on pre.col_id = b.col_id
        group by 1)

select coef.row_id, coef.val + intercept.val as prediction
from coef, intercept



