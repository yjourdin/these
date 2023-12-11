#!/bin/sh

echo "Time,Train accuracy,Test accuracy,Kendall's tau,General seed,A train seed,Model seed,D train seed,Learn seed,A test seed,Initial model seed,SA seed,N_tr,N_te,M,K_o,K_e,N_bc,Method,Model,Gamma,Non dictator,Lexicographic order,Profiles number,Max profiles number,T0,Alpha,L,Tf,Max time,Max iter,Max iter non improving,Seed,A train seed,Model seed,D train seed,Learn seed,A test seed" >evo_out.csv

for m in 7 11 15; do
    for ke in 1 2 3 4; do
        for nbc in 100 300 500 1000 2000; do
            for seed in 0 1 2 3 4 5 6 7 8 9; do
                python main.py @config.txt --M=$m --K-o=$ke --K-e=$ke --N-bc=$nbc --seed=$seed >>evo_out.csv &
            done
        done
    done
done
