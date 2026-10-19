# SACENet — Sarcasm-Aware Explainable Detection of Cyberbullying in Hinglish

A reproducible research codebase for classifying **cyberbullying types in Hinglish (Hindi-English code-mixed)
comments**. It tests whether **sarcasm supervision** helps, checks **SHAP explanations** for faithfulness, and
evaluates a **causal early-warning** procedure on a public conversational proxy dataset.

> **Status (honest summary).** All results come from runs whose per-item predictions are saved in
> `results/runs/` and are re-verified by `scripts/verify_results.py`. Measured numbers are in
> `results/FINAL_EXPERIMENT_REPORT.md` and the paper draft in `paper/`. Early detection was evaluated **only on an
> English game-chat proxy (CONDA)**, not on Hinglish cyberbullying. See *Limitations*.

The original prototype is preserved unchanged in `legacy/`: notebooks, the LLM sarcasm engine, the hard-coded
"results", and the original README and requirements. `reports/PROJECT_AUDIT.md` explains why it was replaced.

## Repository layout

```
configs/experiments.yaml      every experiment (models, pinned revisions, seeds, hyperparameters)
src/data/                     sources, loaders, cleaning, duplicate/leakage checks, splitting
src/models/                   TF-IDF baselines; transformer with optional multitask head / fusion
src/training/                 training loop (MPS/CUDA/CPU), baseline runner, task definitions
src/evaluation/               metrics, CIs, paired tests, calibration, early-detection metrics, aggregation
src/explainability/           SHAP, occlusion, deletion-faithfulness, stability
src/inference/                checkpoint loader / predictor
src/reporting/                figures, diagrams, tables
scripts/                      CLI entry points (below)
app/streamlit_app.py          demo app
data/                         manifest, dataset card, split ID lists (raw text is NOT committed)
results/                      runs, analysis, figures, tables, registry, reports
paper/                        manuscript (md + tex + pdf), bibliography, captions, checklists
tests/                        pytest suite
legacy/                       original code, untouched
```

## 1. Setup

Python 3.11 is recommended. A GPU is optional: CUDA and Apple MPS are detected automatically and smoke-tested, with
CPU as the fallback.

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# NVIDIA GPU: first install the CUDA build of torch from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

**macOS / Linux (bash)**
```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt        # or: uv venv --python 3.11 && uv pip install -r requirements.txt
```

No API keys, Kaggle account or credentials are needed (`.env.example` lists the optional variables).

## 2. Data

```powershell
python scripts/download_datasets.py --user-dir ..   # public sources; --user-dir only for the two user-supplied CSVs
python scripts/verify_datasets.py                    # SHA-256 + counts vs data/dataset_manifest.json
python scripts/prepare_data.py                       # clean, de-duplicate, split, leakage report
```
See `data/README.md` for sources, licences and the manual steps that HASOC/TRAC would need.

## 3. Models and experiments

```powershell
python scripts/download_models.py                    # pretrained encoders at pinned revisions (~3.5 GB)
python -m pytest -q                                  # tests (data, metrics, early detection, inference, SHAP, app)
python scripts/run_experiments.py --smoke            # 1-epoch end-to-end sanity run (not a result)
python scripts/run_experiments.py --list             # all experiment ids
python scripts/run_experiments.py --group baselines
python scripts/run_experiments.py --group sarcasm_component
python scripts/run_experiments.py --group mc7_backbones
python scripts/run_experiments.py --group mc7_sarcasm
python scripts/run_experiments.py --group early_conda
python scripts/run_experiments.py --group mc2_models
python scripts/run_early_detection.py                # conversation-level early-warning evaluation (proxy)
```
On macOS/Linux, `scripts/run_all_isolated.sh` runs all training groups in sequence, one process per run (recommended:
this returns GPU memory to the OS between runs). Completed runs are skipped
when re-run. Every run is logged in `results/experiment_registry.{jsonl,csv}` with its config, seed, device, timing,
metrics and output folder.

Approximate cost on an Apple M1 Pro (MPS): ~10–14 min per MuRIL/XLM-R run on MC7 (batch 32, ≤ 5 epochs); the full
plan took ~13 h including re-runs. Memory: multitask runs pushed a 16 GB Mac to memory-pressure "warning", and an
unfrozen-embedding run was infeasible. On 8 GB machines (untested), try `batch_size: 16` with `grad_accum: 2` per experiment,
and run `scripts/resource_watchdog.py` alongside training.

## 4. Analysis, explainability, figures, tables, paper

```powershell
python scripts/analyze_results.py                    # aggregates seeds; recomputes metrics from predictions
python scripts/run_explainability.py --device cpu    # SHAP + faithfulness + stability (needs the MC7 MuRIL s13 checkpoint)
python scripts/benchmark_inference.py                # latency / size (run when the GPU is idle)
python scripts/generate_paper_assets.py              # results/figures/**, results/tables/*, caption files
python scripts/build_paper.py                        # paper/*.md + *.tex (+ PDF if tectonic/pdflatex is installed)
python scripts/verify_results.py                     # end-to-end consistency checks
```
To compile the PDF on Windows, install Tectonic (`scoop install tectonic` or `choco install tectonic`) or MiKTeX.

## 5. Demo app

```powershell
streamlit run app/streamlit_app.py
```
The app uses the trained MC7 MuRIL checkpoint (and the sarcasm checkpoint for an informational score). It shows
calibrated probabilities, optional SHAP attributions, and a clearly labelled conversation *prototype*. It does not store
submitted text, and it shows an error, never fabricated output, if a checkpoint is missing.

## Limitations (read before use)

* MC-Hinglish1.0 is small (8.4k), Roman-script Twitter text with no reported inter-annotator agreement; 18% of its
  official test items have near-duplicates in the training split.
* Sarcasm supervision comes from a different corpus (2017 tweets). MC-Hinglish has no gold sarcasm labels.
* Early detection is validated only on a proxy (English in-game chat), so it is not validated for Hinglish cyberbullying.
* Hyperparameters are shared across models, not tuned per model; the token-embedding table is frozen for compute reasons.
* Models can reproduce biases about religion, gender and dialect. They are research artefacts to *assist* human
  moderators, not to make automated decisions about people.

## Licences and data use

Code: licence to be chosen by the repository owner. Data are **not redistributed**: MC-Hinglish1.0
(CC-BY-NC-SA-4.0, non-commercial), Swami et al. (GPL-3.0), CONDA (no licence stated), TweetEval/OLID (OLID terms).
The provenance of the two user-supplied sarcasm CSVs is undocumented. They are used only in a supplementary test
and should not be published without clarifying their origin.

## Citation

If you use this code, cite the datasets and models listed in `paper/references.bib`, in particular
Rahman-Laskar et al. (2024) for MC-Hinglish1.0 and Swami et al. (2018) for the sarcasm corpus.
