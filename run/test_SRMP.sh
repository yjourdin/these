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
M='M 11'
K_o='K_o 1'
K_e='K_e 1'
N_bc='N_bc 300'
T0='T0 0.1 0.03 0.02 0.01 0.005'
Tf='Tf 0.001 0.0003 0.0002 0.0001 0.00005'
error='error 0'
seed='seed 3135 5056 5444 6191 6348 9987 10821 13187 16139 17084 18379 20730 22164 22875 23369 24727 24834 28086 29672 29970 32097 35499 35680 38188 39000 40171 40967 41348 41497 41672 43615 44016 49009 50159 50742 50769 50911 53406 53537 54185 55604 55654 55883 57969 58357 58432 59950 60881 63193 64102'

# shellcheck disable=SC1083
# shellcheck disable=SC2086
parallel -j10 --header : \
    python main.py \
    --N-tr $N_tr --N-te $N_te SA --model-o SRMP --model-e SRMP --alpha $alpha --L $L \
    --M {M} --K-o {K_o} --K-e {K_e} --N-bc {N_bc} --T0 {T0} --Tf {Tf} --error {error} --seed {seed} \
    ::: $seed \
    ::: $M ::: $K_o :::+ $K_e ::: $N_bc :::+ $T0 :::+ $Tf ::: $error \
    >>"$file"
