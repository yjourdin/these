#!/bin/sh
jupyter nbconvert "$1" --to slides --no-input "$2"
