---- First Featurize
analyze;
with featurized_data as
(

	SELECT 
	row_id
	, passenger_count
	, TRIM(to_char(pickup_datetime, 'Day')) as day
	, extract(hour from pickup_datetime) as hour
	, CASE WHEN extract(dow from pickup_datetime) in (0, 6) THEN 1 else 0 END as is_weekend
	, extract(month from pickup_datetime) as month
	, pgml.calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, 'K') as distance

	FROM pgml.nyc_rides_test where passenger_count > 6

),

translated_data as
(
	SELECT row_id, CASE WHEN hour <= 2.5 THEN '0.' WHEN hour <= 9.5 AND hour > 2.5 THEN '1.' WHEN hour <= 11.5 AND hour > 9.5 THEN '2.' ELSE '3.' END || 
					CASE WHEN month::TEXT= ANY(ARRAY['1']) THEN '0.' WHEN month::TEXT= ANY(ARRAY['2']) THEN '1.' WHEN month::TEXT= ANY(ARRAY['3']) THEN '2.' WHEN month::TEXT= ANY(ARRAY['4']) THEN '3.' WHEN month::TEXT= ANY(ARRAY['5']) THEN '4.' WHEN month::TEXT= ANY(ARRAY['6']) THEN '5.' ELSE '6.' END || 
					CASE WHEN day::TEXT= ANY(ARRAY['Monday']) THEN '0.' WHEN day::TEXT= ANY(ARRAY['Sunday']) THEN '1.' WHEN day::TEXT= ANY(ARRAY['Saturday']) THEN '2.' WHEN day::TEXT= ANY(ARRAY['Wednesday']) THEN '3.' WHEN day::TEXT= ANY(ARRAY['Friday']) THEN '4.' WHEN day::TEXT= ANY(ARRAY['Tuesday']) THEN '5.' WHEN day::TEXT= ANY(ARRAY['Thursday']) THEN '6.' ELSE '7.' END || 
					CASE WHEN distance <= 0.7610131204128265 THEN '0' WHEN distance <= 1.0520523190498352 AND distance > 0.7610131204128265 THEN '1' WHEN distance <= 1.2976540923118591 AND distance > 1.0520523190498352 THEN '2' WHEN distance <= 1.5451794266700745 AND distance > 1.2976540923118591 THEN '3' WHEN distance <= 1.7953275442123413 AND distance > 1.5451794266700745 THEN '4' WHEN distance <= 2.1136789321899414 AND distance > 1.7953275442123413 THEN '5' WHEN distance <= 2.4617480039596558 AND distance > 2.1136789321899414 THEN '6' WHEN distance <= 2.791672468185425 AND distance > 2.4617480039596558 THEN '7' WHEN distance <= 3.507714867591858 AND distance > 2.791672468185425 THEN '8' WHEN distance <= 4.474145889282227 AND distance > 3.507714867591858 THEN '9' WHEN distance <= 5.80251407623291 AND distance > 4.474145889282227 THEN '10' ELSE '11' END as key  
	FROM featurized_data
),

predictions as
(
	select k1.row_id, case when k2.value is not null then k2.value else pgml.prefix_search(k1.key, 'nyc_rides_mlpregressor_kv') end as prediction
	from translated_data k1
	left join pgml.nyc_rides_mlpregressor_kv k2
	on k1.key = k2.key
	where k1.key in (select key from translated_data)
)

select * from predictions

select day, hour, avg(kv.prediction) as avg_prediction
from featurized_data d
join predictions kv
on d.row_id = kv.row_id
group by 1, 2