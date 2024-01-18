#!/bin/bash

# Output file
file="results/$1.csv"

# Header
tr '\n' ',' <config/header.txt >"$file"
'\n' >>"$file"

# Fixed parameters
N_tr=500
N_te=500
alpha=0.9999
L=1

# Grid
M='M 7 11 15'
K_o='K_o 1 2 3 4'
K_e='K_e 1 2 3 4'
N_bc='N_bc 100 300 500 1000 2000'
T0='T0 0.01 0.003 0.002 0.001 0.0005'
Tf='Tf 0.001 0.0003 0.0002 0.0001 0.00005'
error='error 0 0.1 0.2 0.3'
repetition=50

# shellcheck disable=SC1083
parallel -j35 --header : \
    python main.py \
    --N-tr $N_tr --N-te $N_te SA --model-o SRMP --model-e SRMP --alpha $alpha --L $L \
    --M {M} --K-o {K_o} --K-e {K_e} --N-bc {N_bc} --T0 {T0} --Tf {Tf} --error {error} \
    ::: repetition $(seq $repetition) \
    ::: "$M" ::: "$K_o" ::: "$K_e" ::: "$N_bc" :::+ "$T0" :::+ "$Tf" ::: "$error" \
    >>"$file"
