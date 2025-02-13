#!/bin/bash
L=50000
max_time=25000


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 demes_file"
    exit 1
fi
demes_file=$1
deme=$(basename "$demes_file" .${demes_file##*.})

echo "Running method on simulations for $deme"
../src/run_trasp -input $demes_file -output results/"$deme"_sim -mode simulated -L $L -min_time 1 -max_time $max_time -num_timepoints 200 -delta 100 -log_time

echo "Running method on inferred trees for $deme"
python simulate_human_chrom.py -input $demes_file -output trees/inferred_from_sims/"$deme" 
../src/run_trasp -input trees/inferred_from_sims/"$deme" -output results/"$deme"_inferred -metadata trees/inferred_from_sims/"$deme"/metadata.csv  -mode inferred -L $L -min_time 1 -max_time $max_time -num_timepoints 200 -delta 100 -log_time
