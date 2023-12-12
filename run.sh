#!/bin/sh

# Output file
file='results/csv/gen_out.csv'

# Header
tr '\n' ',' <config/header.txt >$file

# Grid
M='M 7 11 15'
K_e='K_e 1 2 3 4'
N_bc='N_bc 100 300 500 1000 2000'
seed='seed 0 1 2 3 4 5 6 7 8 9'

# shellcheck disable=SC1083
# shellcheck disable=SC2086
parallel --header : \
    python main.py @config/defaults.txt \
    --M={M} --K-o={K_e} --K-e={K_e} --N-bc={N_bc} --seed={seed} \
    ::: $M ::: $K_e ::: $N_bc ::: $seed \
    >>$file
