#!/bin/bash

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

cd ./inferdb/experiments/standalone

python credit_card_new_pipeline.py $iterations $paper_models
python hits.py $iterations $paper_models
python mnist.py $iterations $paper_models
python nyc_rides_complex.py $iterations $paper_models
python pm25.py $iterations $paper_models
python rice.py $iterations $paper_models

cd /app/inferdb/experiments/plots/standalone

python effectiveness_tables.py
python training_tables.py

cd /app/inferdb/experiments/plots/latex/training_table

pdflatex -output-directory /app/inferdb/experiments/plots/latex/output training_table.tex

cd /app/inferdb/experiments/plots/latex/effectiveness_tables

pdflatex -output-directory /app/inferdb/experiments/plots/latex/output effectiveness_tables.tex

echo "Done!"