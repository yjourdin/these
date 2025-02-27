#!/bin/bash
# shellcheck disable=SC2086
nohup $1 >${2-nohup.out} 2>&1 &
echo "$!,$1" >>~/save_pid.txt
