DROP FUNCTION IF EXISTS pgml.crazy_exp(numeric) CASCADE;
CREATE OR REPLACE FUNCTION pgml.crazy_exp(numeric) RETURNS numeric AS $$
-- exp(6000) and above will throw 'argument for function "Exp" too big
-- For such cases, return zero because [insert explanation here]
SELECT CASE WHEN $1 < 0 THEN 0 WHEN $1 < 6000 THEN exp($1) ELSE exp(5999.99::numeric) END;
$$ LANGUAGE 'sql' IMMUTABLE;

DROP FUNCTION IF EXISTS pgml.crazy_exp(double precision) CASCADE;
CREATE OR REPLACE FUNCTION pgml.crazy_exp(double precision) RETURNS double precision AS $$
-- exp(6000) and above will throw 'argument for function "Exp" too big
-- For such cases, return zero because [insert explanation here]
SELECT CASE WHEN $1 < 0 THEN 0 WHEN $1 < 700 THEN exp($1) ELSE exp(700::double precision) END;
$$ LANGUAGE 'sql' IMMUTABLE;

DROP FUNCTION IF EXISTS pgml.crazy_exp(real) CASCADE;
CREATE OR REPLACE FUNCTION pgml.crazy_exp(real) RETURNS double precision AS $$
-- exp(6000) and above will throw 'argument for function "Exp" too big
-- For such cases, return zero because [insert explanation here]
SELECT CASE WHEN $1 < 0 THEN 0 WHEN $1 < 700 THEN exp($1) ELSE exp(700::double precision) END;
$$ LANGUAGE 'sql' IMMUTABLE;