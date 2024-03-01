#!/bin/bash

# Name of the experiment
name=$1

# Directory of the experiment
dir=results/"$name"

# Create experiment directories
mkdir "$dir"
mkdir "$dir"/A_train
mkdir "$dir"/A_test
mkdir "$dir"/Mo
mkdir "$dir"/Me
mkdir "$dir"/D

# Headers
echo "A,D,ke,Time,Iterations,Objective" >"$dir/train_results.csv"
echo "A,Mo,Me,Fitness,Kendall's tau" >"$dir/test_results.csv"

# Fixed parameters
N_exp=50
N_tr=500
N_te=5000
alpha=0.99

# Grid
M='7'
Ko='1'
Ke='1'
N_bc='100 200 300 400 500 600 700 800 900 1000 2000'
T0='0.01 0.005 0.003 0.0025 0.002 0.0016 0.0014 0.00125 0.0011 0.001 0.0005'
Tf='0.001 0.0005 0.0003 0.00025 0.0002 0.00016 0.00014 0.000125 0.00011 0.0001 0.00005'
error='0'
i=$(seq $N_exp)

# Number of cores
J=0

# Main

# Create A_train
parallel -j $J --header : \
    "python -m performance_table \
    $N_tr {M} -o $dir/A_train/No_{i}_M_{M}" \
    ::: M $M ::: i $i
echo "A_train created"

# Create Mo
parallel -j $J --header : \
    "python -m srmp \
    {K} {M} -o $dir/Mo/No_{i}_M_{M}_K_{K}" \
    ::: K $Ko ::: M $M ::: i $i
echo "Mo created"

# Create D
parallel -j $J --header : \
    "python -m preference_structure \
    $dir/A_train/No_{i}_M_{M} $dir/Mo/No_{i}_M_{M}_K_{K} {N} -e {error} -o $dir/D/No_{i}_M_{M}_K_{K}_N_{N}_E_{error}" \
    ::: N $N_bc ::: error $error ::: K $Ko ::: M $M ::: i $i
echo "D created"

# Create Me
parallel -j $J --header : \
    "python -m sa \
    SRMP {K} $dir/A_train/No_{i}_M_{M} $dir/D/No_{i}_M_{M}_K_{Ko}_N_{N}_e_{error} --T0 {T0} --alpha $alpha --Tf {Tf} -o $dir/Me/No_{i}_M_{M}_Ko_{Ko}_N_{N}_e_{error}_Ke_{Ke} -r $dir/train_results.csv" \
    ::: N $N_bc :::+ T0 $T0 :::+ Tf $Tf ::: error $error ::: Ke $Ke ::: Ko $Ko ::: M $M ::: i $i
echo "Me created"

# Create A_test
parallel -j $J --header : \
    "python -m performance_table \
    $N_te {M} -o $dir/A_test/No_{i}_M_{M}" \
    ::: M $M ::: i $i
echo "A_test created"

# Compute tests
parallel -j $J --header : \
    "python test.py \
    dir/A_train/No_{i}_M_{M} $dir/Mo/No_{i}_M_{M}_K_{K} $dir/Me/No_{i}_M_{M}_Ko_{Ko}_N_{N}_e_{error}_Ke_{Ke} -r $dir/test_results.csv" \
    ::: N $N_bc :::+ T0 $T0 :::+ Tf $Tf ::: error $error ::: Ke $Ke ::: Ko $Ko ::: M $M ::: i $i
