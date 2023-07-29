CREATE OR REPLACE FUNCTION crazy_exp(numeric) RETURNS numeric AS $$
-- exp(6000) and above will throw 'argument for function "Exp" too big
-- For such cases, return zero because [insert explanation here]
SELECT CASE WHEN $1 < 6000 THEN exp($1) ELSE exp(5999.99::numeric) END;
$$ LANGUAGE 'sql' IMMUTABLE;