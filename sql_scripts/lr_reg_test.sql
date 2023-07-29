with start_time as(
select current_timestamp as st
),

preprocessed_data as (

select row_number() over () as row_id, s.*  from nyc_rides_linearregression_scale() s

),

numbered_data as (

select row_number() over () as row_id, d.* from nyc_rides_test d
	
)

select sqrt(sum((ln(lr.prediction + 1) - ln(d.class + 1))^2)/count(*)) as res
from numbered_data d
	, (select s.row_id, score_lr_nyc_rides(s.s) as prediction from preprocessed_data s) lr
where lr.row_id = d.row_id

