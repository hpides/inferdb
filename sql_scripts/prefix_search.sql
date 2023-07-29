CREATE OR REPLACE FUNCTION pgml.prefix_search(searched_key TEXT, search_table TEXT) returns NUMERIC AS 
$$

DECLARE 
   r INTEGER;
   prediction NUMERIC;
   not_found BOOLEAN;
BEGIN
    r := 0;
	not_found := TRUE;
    WHILE not_found LOOP
		r := r + 1;
        execute format('SELECT count(*)=0 from pgml.%I where key ^@ left(%L, length(%L) - (%s*2))', search_table, searched_key, searched_key, r) INTO not_found;
    END LOOP;
	
	execute format('SELECT avg(value) from pgml.%I where key ^@ left(%L, length(%L) - (%s*2))', search_table, searched_key, searched_key, r) INTO prediction;
	
	RETURN prediction;
END;
$$
LANGUAGE plpgsql STABLE PARALLEL SAFE;