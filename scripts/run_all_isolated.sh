#!/usr/bin/env bash
# Full training chain, ONE PROCESS PER RUN so GPU/unified memory is returned to the OS
# between runs (a long-lived process let MPS wired memory grow to ~9.4 GB on a 16 GB Mac).
# Completed runs are skipped. Windows: run `python scripts/run_experiments.py --only <id>` per id.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for g in sarcasm_component mc7_backbones mc7_sarcasm early_conda mc2_models; do
  echo "=== GROUP $g $(date -u +%FT%TZ)"
  for id in $($PY scripts/run_experiments.py --group "$g" --list); do
    $PY scripts/run_experiments.py --group "$g" --only "$id"
  done
done
echo "=== ALL DONE $(date -u +%FT%TZ)"
