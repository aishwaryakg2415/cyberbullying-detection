# Reproducibility checklist

| Item | Status | Where |
|---|---|---|
| Code for data download, preparation, training, evaluation, figures, paper | ✅ | `scripts/`, `src/` |
| Exact dataset versions (pinned revisions + SHA-256) | ✅ | `data/dataset_manifest.json` |
| Exact split membership (IDs) | ✅ | `data/splits/<task>/*_ids.json` |
| Preprocessing fully specified | ✅ | `src/data/text.py`, `RESEARCH_METHODOLOGY.md` §2 |
| Pretrained model revisions pinned | ✅ | `configs/experiments.yaml` |
| All hyperparameters reported | ✅ | `configs/experiments.yaml`, Table *hparams* |
| Hyperparameter search described | ✅ | Transformers: none (shared setting). TF-IDF: C grid on validation |
| Random seeds | ✅ | 13, 42, 87 (transformers); split seed 20261017 |
| Number of runs and variance | ✅ | 3 seeds, mean ± s.d.; ensemble bootstrap CI |
| Statistical tests specified | ✅ | paired bootstrap (2,000), exact McNemar; uncorrected, treated as descriptive |
| Per-item predictions released (no text) | ✅ | `results/runs/*/predictions_*.jsonl` |
| Metrics recomputable from predictions | ✅ | `scripts/verify_results.py` |
| Hardware and software versions | ✅ | `results/runs/*/environment.json`, `results/runs/_device/device_report.json`, `requirements.txt` |
| Runtime and memory | ✅ | `metrics.json` (`train_seconds`, `history[].memory`), `results/benchmarks/inference.json` |
| Experiment registry | ✅ | `results/experiment_registry.{jsonl,csv}` |
| Deviations from standard practice disclosed | ✅ | frozen token-embedding table; shared hyperparameters; CONDA re-split |
| Failed / abandoned runs disclosed | ✅ | `results/FINAL_EXPERIMENT_REPORT.md` §Failures |
| Data licences respected (no redistribution) | ✅ | `.gitignore`, `data/README.md` |
| Model checkpoints | ⚠️ partially | Two fp16 checkpoints are kept locally (not in git); other runs keep predictions only (disk limits) |
| Fully deterministic GPU training | ⚠️ no | MPS kernels are not bit-deterministic; seed-to-seed variance is reported instead |

**One-command reproduction (macOS/Linux):** `scripts/run_all_training.sh` after the data steps in `README.md`.
On Windows, run the `run_experiments.py --group …` commands listed in the README in order.
