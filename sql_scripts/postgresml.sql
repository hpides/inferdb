SELECT pgml.train(
    project_name => 'preprocessed_adult_model'::text, 
    task => 'classification'::text, 
    relation_name => 'pgml.adult_train'::text,
    y_column_name => 'class'::text, 
    preprocess => '{
		"row_id": {"impute": "mean", "scale": "standard"},
        "age": {"impute": "mean", "scale": "standard"},
        "workclass": {"encode": "target_mean", "scale": "standard"},
        "fnlwgt": {"impute": "mean", "scale": "standard"},
        "education": {"encode": "target_mean", "scale": "standard"},
		"education_num" : {"impute": "mean", "scale": "standard"},
		"marital_status" : {"encode": "target_mean", "scale": "standard"},
		"occupation" : {"encode": "target_mean", "scale": "standard"},
		"relationship" : {"encode": "target_mean", "scale": "standard"},
		"race" : {"encode": "target_mean", "scale": "standard"},
		"sex" : {"encode": "target_mean", "scale": "standard"},
		"capital_gain" : {"impute": "mean", "scale": "standard"},
		"capital_loss" : {"impute": "mean", "scale": "standard"},
		"hours_per_week" : {"impute": "mean", "scale": "standard"},
		"native_country" : {"encode": "target_mean", "scale": "standard"}
    }'::jsonb,
	test_size => 0.01
);

SELECT * FROM pgml.overview;

select * from pgml.trained_models


SELECT * FROM pgml.train('Breast Cancer Detection', 'classification', 'pgml.adult_train', 'class');

select * from pgml.creditcard_train
select * from pgml.breast_cancer


SELECT pgml.train(
    project_name => 'credit_card'::text, 
    task => 'classification'::text, 
    relation_name => 'pgml.creditcard_train'::text,
    y_column_name => 'class'::text,
	test_size => 0.01
);

WITH predictions AS (
	SELECT pgml.predict_batch(
		'credit_card'::text,
		d
	) AS prediction
	FROM pgml.creditcard_logisticregression_encode_scale_set(150000) d
)
SELECT prediction FROM predictions




select pgml.predict('preprocessed_bc', (17.99::real,10.38::real,122.8::real,1001::real,0.1184::real,0.2776::real,0.3001::real,0.1471::real,0.2419::real,0.07871::real,1.095::real,0.9053::real,8.589::real,153.4::real,0.006399::real,0.04904::real,0.05373::real,0.01587::real,0.03003::real,0.006193::real,25.38::real,17.33::real,184.6::real,2019::real,0.1622::real,0.6656::real,0.7119::real,	0.2654::real,0.4601::real,	0.118::real))

select * FROM pgml.digits

select pgml.predict('preprocessed_adult_model',(0::integer, 56::bigint,0.1::real,33115.0::bigint,0.1::real,9::bigint,0.1::real,0.1::real,0.1::real,0.1::real,0.1::real,0.0::float8,0.0::float8,40::float8,0.1::real))

select d.* as prep_tuples from pgml.creditcard_logisticregression_encode_scale_set(150000) d

