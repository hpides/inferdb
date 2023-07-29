-- text array needs to have same values in each dimension. 
CREATE OR REPLACE FUNCTION pgml.linear_search_text(searched_value TEXT, arr TEXT[][]) returns INT AS $$
DECLARE

	not_found BOOL := TRUE;
	s INT := 0;
	
BEGIN

	WHILE not_found LOOP			
		not_found := (1- (searched_value = ANY(arr[s+1:s+1]))::int)::bool;
		s := s + 1;
	END LOOP;
	
	RETURN s-1;
	
END;  $$
LANGUAGE plpgsql PARALLEL SAFE;
