with preprocessed_data as (

select row_number() over () as row_id, s.*  from adult_logisticregression_scale() s

),

numbered_data as (

select row_number() over () as row_id, d.* from adult_test d
	
)

select (count(*) - sum(abs(d.class - lr.prediction)))/count(*) as err
from numbered_data d
	, (select s.row_id, round(score_lr_adult(s.s)) as prediction from preprocessed_data s) lr
where lr.row_id = d.row_id