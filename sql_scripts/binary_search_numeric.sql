CREATE OR REPLACE FUNCTION pgml.binary_search_numeric(searched_value REAL, arr REAL[]) returns INT AS $$

DECLARE
	not_found BOOL := TRUE;
	l INT := 1;
	len INT := array_length(arr, 1);
	r INT := len;
	m_ INT := floor((r+l)/2);
	idx INT := 0;

BEGIN
				-- left extreme
	idx :=  (searched_value <= arr[1])::int * idx
				-- right extreme
				+ (searched_value > arr[len])::int * len;
			
	not_found := (1-(	-- left extreme termination condition
							(searched_value <= arr[1])::int
							-- right extreme termination condition
							+ (searched_value > arr[len])::int
						)
					 )::bool;

	WHILE not_found LOOP		
				
		idx :=  -- left side
				(searched_value > arr[m_-1] and searched_value <= arr[m_] and arr[m_-1] is not null)::int * (m_-1)
				-- right side
				+ (searched_value <= arr[m_+1] and searched_value > arr[m_] and arr[m_+1] is not null)::int * (m_);
			
		not_found := (1-(
							-- left side termination condition
							+ (searched_value > arr[m_-1] and searched_value <= arr[m_] and arr[m_-1] is not null)::int
							-- right side termination condition
							+ (searched_value <= arr[m_+1] and searched_value > arr[m_] and arr[m_+1] is not null)::int
						)
					 )::bool;
		r := r*(searched_value > arr[m_])::int + ((searched_value < arr[m_])::int * (m_));
		l := l*(searched_value < arr[m_])::int + ((searched_value > arr[m_])::int * (m_+1));
		m_ := floor((r+l)/2);
	END LOOP;
	RETURN idx;
END; $$
LANGUAGE plpgsql PARALLEL SAFE;
