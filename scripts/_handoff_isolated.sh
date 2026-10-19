#!/usr/bin/env bash
# One-off: wait for the current sarcasm run, then replace the long-lived chain with the isolated one.
cd "$(dirname "$0")/.."
until [ -f results/runs/sarcasm__muril__base__s87/metrics.json ]; do sleep 3; done
pkill -f run_all_training.sh
pkill -f "bin/python scripts/run_experiments"
sleep 5
[ -f results/runs/mc7__muril__base__s13/metrics.json ] || rm -rf results/runs/mc7__muril__base__s13
nohup scripts/run_all_isolated.sh > results/logs/run_all.log 2>&1 &
echo "switched at $(date)"
