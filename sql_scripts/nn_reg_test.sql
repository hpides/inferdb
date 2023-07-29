with start_time as(
select current_timestamp as st
),

preprocessed_data as (

select row_number() over () as row_id, s.*  from nyc_rides_mlpregressor_scale() s

),

numbered_data as (

select row_number() over () as row_id, d.* from nyc_rides_test d
	
)

select sqrt(sum((ln(nn.prediction + 1) - ln(d.class + 1))^2)/count(*)) as res
from preprocessed_data s
	, numbered_data d
	, (select row_id, round(prediction) as prediction from score_nn_nyc_rides()) nn
where s.row_id = d.row_id and s.row_id = nn.row_id

