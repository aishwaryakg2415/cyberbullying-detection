#!/usr/bin/env bash
# One-off: after the diagnostics, re-run the two mc2 fusion runs lost to the external-eval bug.
cd "$(dirname "$0")/.."
until grep -q "=== DIAGNOSTICS DONE" results/logs/run_all.log; do sleep 30; done
{
  echo "=== GROUP mc2_fusion_rerun $(date -u +%FT%TZ)"
  for s in 13 42; do
    .venv/bin/python scripts/run_experiments.py --group mc2_models --seeds $s --only "mc2__muril__fusion__s$s"
  done
  echo "=== RERUN DONE $(date -u +%FT%TZ)"
} >> results/logs/run_all.log 2>&1
