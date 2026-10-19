#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "=== RERUN DONE" results/logs/run_all.log; do sleep 30; done
{
  echo "=== GROUP diagnostics (memory-safe retry) $(date -u +%FT%TZ)"
  .venv/bin/python scripts/run_experiments.py --group diagnostics --seeds 13
  echo "=== FULLFT DONE $(date -u +%FT%TZ)"
} >> results/logs/run_all.log 2>&1
