#!/bin/bash
# L=20000

pops="all"
regions="None"
n=0
output="test"
max_time=20000

# Parse command-line arguments
while getopts "p:r:n:o:t:" opt; do
  case $opt in
    p) pops=$OPTARG ;;
    r) regions=$OPTARG ;;
    n) n=$OPTARG ;;
    o) output=$OPTARG ;;
    t) max_time=$OPTARG ;;
    \?) echo "Usage: $0 -p pops -n n -o output -t max_time" >&2
        exit 1 ;;
  esac
done

python simplify_trees.py -input trees/human/wohns -output trees/human/"$output" -populations $pops -individuals_per_pop $n -regions $regions
../src/run_trasp -input trees/human/"$output" -output results/"$output" -metadata trees/human/"$output"/metadata.csv -mode inferred -min_time 20 -max_time $max_time -num_timepoints 200 -delta 100 -log_time