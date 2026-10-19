#!/usr/bin/env bash
# One-off: after the main chain finishes, run the diagnostic ablations (one process per run).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until grep -q "=== ALL DONE" results/logs/run_all.log; do sleep 30; done
{
  echo "=== GROUP diagnostics_fusion_control $(date -u +%FT%TZ)"
  $PY scripts/run_experiments.py --group diagnostics_fusion_control --seeds 13 --only "sarcasm__muril__shuffled_labels__s13"
  for s in 13 42 87; do
    $PY scripts/run_experiments.py --group diagnostics_fusion_control --seeds $s --only "mc7__muril__fusion_shuffled__s$s"
  done
  echo "=== GROUP diagnostics $(date -u +%FT%TZ)"
  $PY scripts/run_experiments.py --group diagnostics --seeds 13
  echo "=== DIAGNOSTICS DONE $(date -u +%FT%TZ)"
} >> results/logs/run_all.log 2>&1
