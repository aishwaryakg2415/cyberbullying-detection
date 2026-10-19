# Research methodology

This document records *how* every reported number is produced and *why* each protocol decision was made.
Measured outcomes live in `results/FINAL_EXPERIMENT_REPORT.md`; data facts live in `data/DATASET_CARD.md`.

## 1. Research questions

* **RQ1** How do classical and transformer models (multilingual vs. Hinglish-specific) compare on MC-Hinglish1.0 under leakage-controlled splits?
* **RQ2** Does sarcasm supervision from a separate human-annotated Hinglish corpus help, and is any gain due to the sarcasm *labels* rather than extra data or regularisation?
* **RQ3** Are SHAP explanations of the model faithful and stable?
* **RQ4** In a proxy setting, how early and with how many false alarms can a causal turn-level detector flag toxic conversations?

## 2. Data protocol

* **Sources**: official, revision-pinned, hash-verified (`data/dataset_manifest.json`). No Kaggle.
* **Labels are never merged across constructs.** Cyberbullying type (MC-Hinglish), sarcasm (Swami),
  offensiveness (OLID) and in-game toxicity (CONDA) are separate fields. The only derived label is MC2
  (`not_cyberbullying` vs. the six cyberbullying types), justified by the dataset's own definitions.
* **Cleaning** (`src/data/text.py`): keep case, punctuation, script and code-switching. Emoji → CLDR names, URL /
  @mention placeholders, elongations capped at 3. Swami model input has all `#` tokens removed (retrieval artefact).
* **Duplicates**: exact, normalised and near-duplicate (char 3–5-gram TF-IDF cosine ≥ 0.9).
* **Splits**: MC-Hinglish keeps the official test split. Validation is a stratified 10% of the official train split,
  grouped by near-duplicate cluster. Swami uses a stratified, grouped 70/10/20 split. CONDA is pooled and re-split
  70/10/20 by match id. All split ID lists are committed (`data/splits/`). Split seed: 20261017.
* **Leakage checks**: exact / normalised / near-duplicate overlap between every pair of splits (`data_statistics.json`).
  Official-test items with a near-duplicate in training are flagged, and MC results are also reported on the
  de-duplicated test subset.
* **Imbalance**: no over- or under-sampling. Class-weighted cross-entropy with inverse-frequency weights computed on the
  training split only (also `class_weight="balanced"` for the linear baselines).

## 3. Models

| Family | Details |
|---|---|
| Majority | most frequent training class |
| TF-IDF + LR / SVM | word 1–2-grams + char_wb 2–5-grams, sublinear TF. C ∈ {0.1, 0.3, 1, 3, 10} chosen on validation macro-F1. The SVM uses Platt scaling with 5-fold CV *within train* |
| MuRIL / XLM-R base / HingRoBERTa | [CLS] pooling → dropout 0.1 → linear head. All transformer layers fine-tuned; the token-embedding table is frozen (see §5) |
| MTL | shared encoder, sarcasm head trained on Swami train; $\mathcal{L}=\mathrm{CE}_w^{CB}+0.5\,\mathrm{CE}_w^{sarc}$. One auxiliary batch per main batch (the auxiliary set is cycled) |
| MTL-shuffled (control) | identical to MTL, but sarcasm labels permuted once with a fixed seed (class ratio kept) |
| Fusion | separately trained sarcasm MuRIL (Swami, seed 13) is frozen. Its P(sarcastic) and 768-d pooled embedding (precomputed) pass through Linear(769→64)+GELU and are concatenated with h before the head |
| Legacy boost | the original repository's rule $p'=\min(1,p+0.18s)$ if $s\ge0.45$, threshold 0.55, re-evaluated on MC2 with the trained sarcasm model's $s$ |

**Hyperparameters are shared, not tuned per model** (encoder lr 3e-5, head lr 1e-3, batch 32, ≤5 epochs, early
stopping patience 2, warm-up 10%, weight decay 0.01, clip 1.0, max_len 128).

**Tuning history (disclosed).** The first configuration (lr 3e-5 for encoder *and* heads) under-fitted. MC7 MuRIL
seed 13 had validation loss 1.41 after 5 epochs and a fitted temperature of 0.23 (very under-confident logits).
Its test macro-F1 (0.662) was far below its validation macro-F1 (0.774). Seeing that *test* number triggered a
pilot; this is test-set influence on the decision to tune and is acknowledged as such. The pilot itself used
validation data only (test predictions were never computed): encoder lr ∈ {3e-5, 5e-5} with head lr 1e-3, seed 13.
5e-5/1e-3 was best on MC7 validation macro-F1 (0.781 vs 0.758) and NLL (0.644 vs 0.688) and was adopted first,
but it **collapsed to the majority class on the sarcasm task** (seeds 13 and 42; flat training loss ln 2). A further
validation-only pilot showed 3e-5/1e-3 to be stable on sarcasm (validation macro-F1 0.761). The final rule is: among
settings that do not collapse on any task's validation data, take the best, i.e. **3e-5/1e-3 for every model and
task**. Runs from the collapsed setting are kept in `results/runs_superseded/config_v2_collapsed/`. Any final run
whose selected checkpoint predicts a single class is flagged `collapsed` and reported, not dropped. All v1 runs are kept in `results/runs_superseded/` and excluded
from results (`results/tuning/pilot_summary.json`). This keeps comparisons fair under a fixed compute
budget, at the cost of possibly under-estimating models that would prefer different settings. The MTL weight
λ = 0.5 was fixed a priori, not tuned on test.

## 4. Evaluation

* **Primary metric**: macro-F1 (MC classes are near-balanced except `not_cyberbullying`; sarcasm and CONDA are imbalanced).
* Also reported: accuracy, weighted-F1, per-class P/R/F1, confusion matrix, ROC-AUC (binary or macro one-vs-rest),
  PR-AUC (average precision), Brier score, ECE (15 equal-width bins, top label), NLL, and ECE after temperature scaling
  (T fitted on validation logits).
* **Model selection**: the epoch with the best validation macro-F1. The test set is used exactly once per run.
* **Uncertainty**: 3 seeds (13, 42, 87) for every transformer configuration, reported as mean ± sample s.d. Seed-ensemble
  (mean probability) macro-F1 has a 95% percentile-bootstrap CI (2,000 resamples).
* **Comparisons**: paired bootstrap on the ensemble predictions (2,000 resamples; two-sided p as twice the smaller tail),
  exact McNemar test on per-item correctness, and the count of seeds with a positive difference. No multiple-comparison
  correction is applied, so **p-values are descriptive**. We do not claim significance from any single test.
* **Verification**: `scripts/verify_results.py` recomputes every metric from the saved per-item predictions and checks
  prediction IDs against the committed split ID lists.

## 5. Compute constraints and their consequences

* Hardware: Apple M1 Pro (16 GB), PyTorch MPS backend. No CUDA. fp32 training, because fp16 autocast on MPS was
  measured to be slower (`results/runs/_device/device_report.json`).
* The optimiser step over MuRIL's 151M-parameter token-embedding table took 0.74 s of a 0.92 s step on MPS. The
  token-embedding table is therefore frozen for **all** transformer models (position embeddings and all 12 layers
  remain trainable). This is a deviation from full fine-tuning, applied uniformly.
* Disk: < 6 GB free at the start, so only two fine-tuned checkpoints are retained (fp16): MC7 MuRIL seed 13 (SHAP and
  demo) and the sarcasm MuRIL seed 13 (fusion and demo). All other runs keep per-item predictions, metrics, configs
  and logs.

## 6. Explainability protocol

* SHAP Partition explainer with the tokenizer-based Text masker, explaining the predicted class probability
  (500 evaluations for qualitative examples, 300 for the faithfulness sample, 1,000 as the stability reference).
* Subword values are summed to whitespace words.
* **Faithfulness**: deletion test on 100 random test comments (≥ 4 words): drop in P(predicted class) after removing
  the top-k SHAP words vs. k random words (10 draws), k ∈ {1, 2, 3}, with a bootstrap CI on the difference.
* **Stability**: Spearman correlation and top-3 overlap of word rankings at 100/250/500 evaluations vs. 1,000, and
  between SHAP and leave-one-word-out occlusion.
* **Overhead**: seconds per explanation vs. a single forward pass on the same device.
* Examples are selected deterministically (most confident items per outcome type, ≤ 18 words).

## 7. Early-detection protocol (proxy)

* Turn scorer $f$ trained on CONDA train turns (toxic = E ∪ I): TF-IDF LR and DistilRoBERTa, each with and without the
  two preceding turns as context (causal).
* Conversation alarm at the first $t$ with $\max_{s\le t} p_s \ge \tau$ (or EMA). τ maximises validation-conversation
  F1, plus a second operating point with validation false-alarm rate ≤ 10%.
* Metrics on test conversations: precision/recall/F1 of alarms, false-alarm rate, delay relative to the first toxic
  turn, ERDE$_3$/ERDE$_5$, and prefix curves (decision after k turns vs. the eventual label).
* **Scope**: this tests the mechanics of causal early warning on public sequential data. It is **not** evidence about
  Hinglish cyberbullying.

## 8. What would change the conclusions

Different hyperparameters per model, full (unfrozen-embedding) fine-tuning, more seeds, a sarcasm corpus from
the same period and topics as MC-Hinglish, or MC-Hinglish texts with gold sarcasm labels.
