DROP FUNCTION IF EXISTS dot_product CASCADE;CREATE OR REPLACE FUNCTION dot_product(arr_q NUMERIC[], arr_e NUMERIC[]) returns NUMERIC AS $$
    SELECT (
                SELECT sum(s) 
                FROM unnest(ax.ar) s
            ) AS array_dot_prod 
    FROM (
        SELECT (
                    SELECT array_agg(e.el1 * e.el2) 
                    FROM unnest(t.ar1, t.ar2) e(el1, el2)
                ) ar
        FROM (SELECT arr_q as ar1,  arr_e as ar2) t
    ) ax;
$$ LANGUAGE SQL PARALLEL SAFE;