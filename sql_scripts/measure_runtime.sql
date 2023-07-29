DROP FUNCTION IF EXISTS measure_runtime; CREATE OR REPLACE FUNCTION measure_runtime(func TEXT) returns NUMERIC AS $$

DECLARE
   _timing1  timestamptz;
   _start_ts timestamptz;
   _end_ts   timestamptz;
   _overhead numeric;     -- in ms
   _timing   numeric;     -- in ms
   runtime numeric;
BEGIN
   _timing1  := clock_timestamp();
   _start_ts := clock_timestamp();
   _end_ts   := clock_timestamp();
   -- take minimum duration as conservative estimate
   _overhead := 1000 * extract(epoch FROM LEAST(_start_ts - _timing1
                                              , _end_ts   - _start_ts));

   _start_ts := clock_timestamp();
   PERFORM * FROM format('%s', func);  -- your query here, replacing the outer SELECT with PERFORM
   _end_ts   := clock_timestamp();
   
-- RAISE NOTICE 'Timing overhead in ms = %', _overhead;
   runtime := 1000 * (extract(epoch FROM _end_ts - _start_ts)) - _overhead;

   RETURN runtime;
END;
$$ LANGUAGE plpgsql;