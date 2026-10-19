#!/usr/bin/env bash
# Full training chain (macOS/Linux). Windows: see README "PowerShell" section.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
# Ordered by importance: core 7-class results first, binary task last.
for g in sarcasm_component mc7_backbones mc7_sarcasm early_conda mc2_models; do
  echo "=== GROUP $g $(date -u +%FT%TZ)"
  $PY scripts/run_experiments.py --group "$g"
done
echo "=== ALL DONE $(date -u +%FT%TZ)"
