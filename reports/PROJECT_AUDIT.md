# SACENet — Project Audit

*Audit date: 2026-10-17. Auditor: Claude Code (AI assistant), on behalf of the project owner.*
*Scope: GitHub repository `aishwaryakg2415/cyberbullying-detection` at commit `c3249b3` (single
"Initial commit"), plus four CSV files the owner placed in the parent working directory.*

## 1. Summary

The original repository describes a complete, IEEE-ready system, but **none of it had been executed**:
both notebooks contain no outputs, the two main entry points crash on start-up, and several results
shown in the README and code are hard-coded rather than measured. The core research idea is still
viable: a MuRIL classifier on MC-Hinglish1.0, a sarcasm component, SHAP explanations, and early
detection. However, most of the scientific scaffolding (valid splits, genuine sarcasm supervision,
real sequential data, honest metrics) had to be rebuilt.

Status key: ✅ working · ⚠️ partially working / methodologically unsound · ❌ broken · 📝 proposed only

| Component (original file) | Status | Finding |
|---|---|---|
| `train.py` entry point | ❌ | Calls `CyberbullyingDataset(use_sample_corpus=True, balance=True, augment=True)`; `augment` is not a parameter → `TypeError` before training starts. |
| `evaluate.py` entry point | ❌ | Same `augment=True` crash. |
| Early-detection results (`evaluate.py:403-406`) | ❌ **fabricated** | `f1_vals = [0.68, 0.74, 0.81, 0.84, 0.85]  # representative values` — hard-coded numbers are plotted and saved to `eval_results.json` as if measured. |
| README "Expected Results" table | ❌ **unsupported** | Accuracy/F1 figures (~0.72/0.81/0.87) were never produced by any code in the repository. |
| README/notebook model description | ⚠️ | Says "Twitter-RoBERTa", "multi-platform (Twitter, Instagram, Formspring, YouTube, Reddit)", HatEval/Founta; the code actually uses MuRIL on MC-Hinglish1.0 only. No multi-platform data exists in the repo. |
| Train/val split (`dataset.py:get_splits`) | ⚠️ **leakage** | Minority class is up-sampled with replacement *before* `train_test_split`, so identical copies of the same tweet land in both train and validation, inflating validation scores and corrupting early stopping. |
| Binary label mapping | ⚠️ | 7 MC-Hinglish classes collapsed to binary; justified by dataset semantics, but `fillna(1)` silently maps any unknown label to "cyberbullying". |
| Sarcasm labels for MC-Hinglish (`_assign_sarcasm`) | ⚠️ **circular** | `sarcastic = 1` if label is `mockery`, or if a regex cue fires, or if an exact-text lookup hits. The lookup has **0** overlapping texts with MC-Hinglish (verified), so the "sarcasm" field is just `mockery` + regex. Evaluating "sarcastic subset" performance with it measures the regex, not sarcasm. |
| Local sarcasm model (`config.LOCAL_SARCASM_MODEL`) | ❌ | `cardiffnlp/xlm-roberta-base-tweet-sentiment-multilingual` is a **sentiment** model (negative/neutral/positive). `LocalSarcasmDetector` looks for an `irony` label that does not exist, falls back to `label_1`/0.5 — i.e. the "sarcasm score" is meaningless. |
| LLM sarcasm engine (Gemini/OpenAI) | 📝 | Requires paid/registered API keys; never run; prompts include made-up "examples". Not used in this rebuild (no keys, non-reproducible, and LLM labels are not ground truth). |
| Sarcasm-aware fusion (`fuse_scores`) | ⚠️ | Hand-set additive boost `p + 0.18 * s` (threshold 0.55, sarcasm threshold 0.45) with no tuning or validation. Re-evaluated in this rebuild as an ablation with a *trained* sarcasm model. |
| SHAP explainer (`model.py`) | ⚠️ | Plausible use of `shap.Explainer` with a text masker, but never run; the narrative text asserts causation ("predicted … because of tokens"). |
| Early detection (`early_detect_conversation`) | 📝 | Concatenates first-K messages of a hand-written synthetic conversation; no conversation dataset, no labels, no metric. |
| Kaggle notebook | ⚠️ | Hard dependency on `/kaggle/input` and `/kaggle/working`; violates the "no Kaggle" requirement. |
| `requirements.txt` | ⚠️ | Unpinned; includes heavy unused packages (openai, anthropic, google-generativeai, lime, rouge-score, wordcloud, plotly). |
| Tests | ❌ | None. |

## 2. Data found locally

| File (parent directory) | Verified identity | Notes |
|---|---|---|
| `Train_data_MC-Hinglish1.0 (1).csv`, `Test_data_MC-Hinglish1.0 (1).csv` | ✅ **Byte-identical** to the official Hugging Face release `sahinurlaskar/MC-Hinglish1.0` @ `a8ae581` (SHA-256 `9bf055ee…`, `fc327650…`). | 7,400 + 1,000 rows (header excluded); 7 labels; CC-BY-NC-SA-4.0. |
| `human_labelled_features.csv` | ⚠️ **Provenance unknown.** Could not be matched to any public release (searched GitHub, HF, papers). | 8,885 rows; `human_labelled` ∈ {non-sarcastic 4,768; sarcastic 3,737; non_sarcastic 377; error 3}; contains Devanagari and some Marathi. AI labels in the same file agree with the human labels on only ~53 % of rows (AI marks 86 % as sarcastic). 0 texts overlap MC-Hinglish. |
| `ai_labelled_features.csv` | ⚠️ Provenance unknown; labels are LLM-generated. | 12,809 rows; machine labels only → **not ground truth**; not used for training or evaluation. |
| `jbo` | Empty file (0 bytes). | Left untouched. |

## 3. Environment

| Item | Finding |
|---|---|
| Machine | Apple M1 Pro, **16 GB** unified memory. The brief mentions an 8 GB laptop; all experiments ran on this 16 GB machine. Running them with 8 GB was **not tested**, and the multitask runs reached macOS memory-pressure "warning" even at 16 GB. |
| OS | macOS 27.0.1 (arm64). |
| GPU | Apple M1 Pro integrated GPU (16 cores, Metal 4). **No CUDA.** PyTorch acceleration is via the **MPS** backend; availability is checked at runtime with a matmul + backward smoke test (`src/utils/device.py`). |
| Disk | **Only ~16 GB free (97 % full)** at audit time. This constrains how many fine-tuned checkpoints can be kept: only two are retained, both in fp16 (7-class MuRIL seed 13 and sarcasm MuRIL seed 13). All other runs keep predictions and metrics only. |
| Python | System Python 3.9.6 (too old for current PyTorch/transformers). A project venv with **Python 3.11** was created with `uv` (`.venv/`). |
| Tooling | `git`, `uv`, `brew`, `node` available. No LaTeX compiler, Graphviz or pandoc at audit time. |
| Network | GitHub, Hugging Face Hub (models + datasets) reachable; no credentials required for any source used. |
| Git | Repository was not cloned locally; it was cloned into `cyberbullying-detection/`. Work is on branch `sacenet-research-rebuild`; nothing pushed. Original modules moved (with `git mv`) to `legacy/` — nothing deleted. |

## 4. Prioritised implementation plan (executed in this order)

1. **Reproducible environment**: venv + pinned `requirements.txt`; device detection with smoke test (CUDA → MPS → CPU).
2. **Datasets**: verify MC-Hinglish1.0 (done: hash match); acquire a *human-annotated* Hinglish sarcasm corpus (Swami et al., 2018 — public, GPL-3.0); acquire a sequential dataset with turn-level labels (CONDA — the only openly downloadable candidate found; English game chat, used strictly as a proxy for the early-detection mechanism); English external test (TweetEval-offensive); document HASOC/TRAC/Bohra as requiring manual access.
3. **Data pipeline**: unified schema, code-mix-preserving cleaning, exact/normalised/near-duplicate detection, group-aware stratified validation split carved from the official training set only, leakage report, training-only class weights. No oversampling.
4. **Baselines**: majority, TF-IDF (word+char) + LR, TF-IDF + linear SVM.
5. **Transformers**: MuRIL, XLM-R base, HingRoBERTa (Hinglish-specific, L3Cube) — 3 seeds each on the 7-class task.
6. **Sarcasm-aware variants** (all trained on genuine sarcasm labels): (a) multitask MuRIL with an auxiliary sarcasm head, (b) the same with *shuffled* sarcasm labels (control: isolates the sarcasm signal from the extra-data/regularisation effect), (c) feature fusion with a separately trained sarcasm model, (d) the original additive boost rule re-evaluated with the trained sarcasm model.
7. **Early detection** on CONDA: causal prefix-based alarms, detection delay, false-alarm rate, ERDE.
8. **Explainability**: SHAP Partition explainer on the best MuRIL; deletion-based faithfulness check vs random tokens; stability across SHAP seeds; runtime overhead.
9. **Figures, tables, paper (two versions), docs, Streamlit demo, tests, final report.**

## 5. Decisions that need the owner's input (work continued without blocking)

* **Provenance of `human_labelled_features.csv` / `ai_labelled_features.csv`.** If you know the source
  (paper, URL, licence, annotation procedure), add it to `data/DATASET_CARD.md`. Until then, these are
  used only in one clearly labelled *supplementary* cross-corpus sarcasm test.
* **HASOC / TRAC access.** Both require registering with the organisers (see `data/README.md`). They are
  not needed for the current results, but would enable the external Hinglish evaluation the brief asked for.

## 6. Post-audit outcome (added at the end of the project)

All planned stages were executed. Final status: see `results/FINAL_EXPERIMENT_REPORT.md`. Deviations from this plan
(configuration re-tuning, the collapsed learning rate, the infeasible unfrozen-embedding diagnostic, and the CONDA
re-split) are disclosed there and in `RESEARCH_METHODOLOGY.md`.
