#!/bin/bash

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

cd ./inferdb/experiments/postgres

python ./creditcard/creditcard_pgml.py $iterations $paper_models
python ./hits/hits_pgml.py $iterations $paper_models
python ./mnist/mnist_pgml.py $iterations $paper_models
python ./nyc_rides/nyc_rides_pgml.py $iterations $paper_models
python ./pollution/pollution_pgml.py $iterations $paper_models
python ./rice/rice_pgml.py $iterations $paper_models

cd /app/inferdb/experiments/plots/postgres

python performance_size_tables.py

cd /app/inferdb/experiments/plots/latex/performance_size

pdflatex -output-directory /app/inferdb/experiments/plots/latex/output performance_size_table.tex

echo "Done!"