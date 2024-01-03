#!/bin/sh

# Output file
file='results/03-01-24.csv'

# Header
tr '\n' ',' <config/header.txt >$file

# Grid
M='M 7 11 15'
K_e='K_e 1 2 3 4'
N_bc='N_bc 100 300 500 1000 2000'
T0='T0 0.1 0.03 0.02 0.01 0.005'
Tf='Tf 0.001 0.0003 0.0002 0.0001 0.00005'
seed='seed 0 1 2 3 4 5 6 7 8 9'

# shellcheck disable=SC1083
# shellcheck disable=SC2086
parallel -j40 --header : \
    python main.py @config/defaults.txt \
    --M={M} --K-o={K_e} --K-e={K_e} --N-bc={N_bc} --T0={T0} --Tf={Tf} --seed={seed} \
    ::: $M ::: $K_e ::: $N_bc :::+ $T0 :::+ $Tf ::: $seed \
    >>$file
