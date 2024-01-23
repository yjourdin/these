#!/bin/bash

# Output file
file="results/$1.csv"

# Header
tr '\n' ',' <config/header.txt >"$file"
echo >>"$file"

# Fixed parameters
N_tr=500
N_te=500
alpha=0.9999
L=1

# Grid
M='M 3 4 5 6'
K_o='K_o 1 2 3'
K_e='K_e 1 2 3'
N_bc='N_bc 100 200 300 400 500 600 700 800 900 1000'
T0='T0 0.01 0.005 0.003 0.0025 0.002 0.0016 0.0014 0.00125 0.0011 0.001'
Tf='Tf 0.001 0.0005 0.0003 0.00025 0.0002 0.00016 0.00014 0.000125 0.00011 0.0001'
error='error 0'
repetition=10

# shellcheck disable=SC1083
# shellcheck disable=SC2086
parallel -j35 --header : \
    python main.py \
    --N-tr $N_tr --N-te $N_te SA --model-o RMP --model-e RMP --alpha $alpha --L $L \
    --M {M} --K-o {K_o} --K-e {K_e} --N-bc {N_bc} --T0 {T0} --Tf {Tf} --error {error} \
    ::: repetition $(seq $repetition) \
    ::: $M ::: $K_o :::+ $K_e ::: $N_bc :::+ $T0 :::+ $Tf ::: $error \
    >>"$file"
