-------------------------- REGRESSION ------------------------
select sqrt(sum((ln(kv.value + 1) - ln(nyc.tv + 1))^2)/count(*)) as res
from (select preprocess_nyc_rides_mlpregressor(nyc) as key, nyc.class as tv from nyc_rides_test nyc) nyc
	, nyc_rides_mlpregressor_kv kv
where nyc.key = kv.key

------------ single preprocessing latency

with ins as (select * from nyc_rides_test nyc limit 1)
,
ins_key as (select preprocess_nyc_rides_mlpregressor(nyc) as key, nyc.class as tv 
from ins nyc)

select kk.*, kv.value
from ins_key kk, nyc_rides_mlpregressor_kv kv
where kk.key = kv.key

----------- Batch Processing -----
with ins as (select * from nyc_rides_test nyc)
,
ins_key as (select preprocess_nyc_rides_mlpregressor(nyc) as key, nyc.class as tv 
from ins nyc)

select kk.*, kv.value
from ins_key kk, nyc_rides_mlpregressor_kv kv
where kk.key = kv.key
----------------------

select sqrt(sum((ln(kv.value + 1) - ln(nyc.tv + 1))^2)/count(*)) as res
from (select preprocess_nyc_rides_linearregression(nyc) as key, nyc.class as tv from nyc_rides_test nyc) nyc
	, nyc_rides_linearregression_kv kv
where nyc.key = kv.key

-----------------------
------------ single preprocessing latency

with ins as (select * from nyc_rides_test nyc limit 1)
,
ins_key as (select preprocess_nyc_rides_linearregression(nyc) as key, nyc.class as tv 
from ins nyc)

select kk.*, kv.value
from ins_key kk, nyc_rides_linearregression_kv kv
where kk.key = kv.key

----------- Batch Processing -----
with ins as (select * from nyc_rides_test nyc)
,
ins_key as (select preprocess_nyc_rides_linearregression(nyc) as key, nyc.class as tv 
from ins nyc)

select kk.*, kv.value
from ins_key kk, nyc_rides_linearregression_kv kv
where kk.key = kv.key

-------------------------- CLASSIFICATION ------------------------


select (count(*) - sum(abs(kv.value - ai.tv)))/count(*) as err
from (select preprocess_click_prediction_small_logisticregression(ai) as key, ai.class as tv from click_prediction_small_test ai) ai
	, click_prediction_small_logisticregression_kv kv
where ai.key = kv.key

select (count(*) - sum(abs(kv.value - ai.tv)))/count(*) as err
from (select preprocess_click_prediction_small_mlpclassifier(ai) as key, ai.class as tv from click_prediction_small_test ai) ai
	, click_prediction_small_mlpclassifier_kv kv
where ai.key = kv.key

------------ single preprocessing latency

with ins as (select * from nyc_rides_test nyc limit 1)
,
ins_key as (select preprocess_nyc_rides_mlpregressor(nyc) as key, nyc.class as tv 
from ins nyc)

select kk.*, kv.value
from ins_key kk, nyc_rides_mlpregressor_kv kv
where kk.key = kv.key

----------- Batch Processing -----
with ins as (select * from nyc_rides_test nyc)
,
ins_key as (select preprocess_nyc_rides_mlpregressor(nyc) as key, nyc.class as tv 
from ins nyc)

select kk.*, kv.value
from ins_key kk, nyc_rides_mlpregressor_kv kv
where kk.key = kv.key
----------------------

