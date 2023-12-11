#!/bin/sh

for m in 3 5 7; do
    for ke in 1 2 3; do
        for nbc in 10 20 30 40 50 60 70 80 90 100; do
            python main.py @config.txt --m=$m --k-e=$ke --n-bc=$nbc >>out.txt
        done
    done
done
