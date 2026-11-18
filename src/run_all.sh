#!/bin/bash
# Factorial design: 4 configurations x 8 models (30 tuning trials, 10 seeds). Run from the repository root.
cd "$(dirname "$0")/.."
mkdir -p results/main/logs
python src/run_config.py P1R1 phys 1 > results/main/logs/P1R1.out 2>&1 &
python src/run_config.py P1R0 phys 0 > results/main/logs/P1R0.out 2>&1 &
wait
python src/run_config.py P0R1 base 1 > results/main/logs/P0R1.out 2>&1 &
python src/run_config.py P0R0 base 0 > results/main/logs/P0R0.out 2>&1 &
wait
