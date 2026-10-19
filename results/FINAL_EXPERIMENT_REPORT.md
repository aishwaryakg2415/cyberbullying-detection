# SACENet — Final Experiment Report

*Rendered from saved results by `scripts/build_paper.py` on 2026-10-18. Every number traces to per-item predictions
in `results/runs/` (re-checked by `scripts/verify_results.py`: 396/396 checks passed over 57 runs).*

## 1. What was done

1. **Audit** (`reports/PROJECT_AUDIT.md`). The original code had never run and contained hard-coded "results". It is
   preserved in `legacy/`.
2. **Data.** All sources were verified by SHA-256 (`data/dataset_manifest.json`): MC-Hinglish1.0 (primary), Swami et al.
   sarcasm, CONDA (early-detection proxy), TweetEval-Offensive (English stress test), and the user-supplied sarcasm file
   (supplementary). HASOC and TRAC need registration and are documented in `data/README.md`.
3. **Pipeline.** Code-mix-preserving cleaning; exact, normalised and near-duplicate detection; group-aware splits; leakage
   report; training-only class weights. 32 tests pass.
4. **Experiments.** 54 completed runs: baselines, 3 transformer backbones × 3 seeds, sarcasm component,
   MTL / shuffled control / fusion / fusion control, binary task, CONDA scorers, and diagnostics.
5. **Explainability.** SHAP with a deletion-faithfulness test, stability checks and timing.
6. **Assets.** 19 figures (PNG+PDF+data), 18 tables (CSV+LaTeX+Markdown), and the paper
   (Markdown, LaTeX, PDF).
7. **Demo app** (`app/streamlit_app.py`), tested headless.

## 2. Headline results

### MC7 (seven cyberbullying types), official test split
**MC7 (7-way cyberbullying type): test-set results (mean ± s.d. over seeds; ensemble = mean of seed probabilities with bootstrap 95% CI; dedup = test items without near-duplicates in training).**

| Model | Seeds | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (macro OvR) | Macro-F1 (dedup test) | Ens. macro-F1 [95% CI] | ECE |
|---|---|---|---|---|---|---|---|---|
| HingRoBERTa | 3 | 0.741 ± 0.039 | 0.721 ± 0.040 | 0.733 ± 0.037 | 0.936 ± 0.009 | 0.708 ± 0.048 | 0.734 [0.708, 0.758] | 0.112 ± 0.028 |
| Majority class | 1 | 0.200 | 0.048 | 0.067 | 0.500 | 0.050 | 0.048 [0.043, 0.052] | 0.002 |
| MuRIL | 3 | 0.660 ± 0.010 | 0.632 ± 0.016 | 0.646 ± 0.015 | 0.899 ± 0.006 | 0.628 ± 0.017 | 0.642 [0.615, 0.669] | 0.094 ± 0.005 |
| MuRIL + sarcasm fusion | 3 | 0.676 ± 0.009 | 0.653 ± 0.009 | 0.666 ± 0.007 | 0.902 ± 0.004 | 0.650 ± 0.007 | 0.649 [0.621, 0.674] | 0.071 ± 0.001 |
| MuRIL + fusion (shuffled-sarcasm features) | 3 | 0.672 ± 0.006 | 0.648 ± 0.005 | 0.661 ± 0.006 | 0.901 ± 0.005 | 0.640 ± 0.002 | 0.651 [0.623, 0.676] | 0.082 ± 0.006 |
| MuRIL + sarcasm MTL | 3 | 0.653 ± 0.007 | 0.631 ± 0.009 | 0.643 ± 0.007 | 0.898 ± 0.003 | 0.632 ± 0.009 | 0.648 [0.620, 0.675] | 0.063 ± 0.027 |
| MuRIL + shuffled-sarcasm MTL | 3 | 0.666 ± 0.016 | 0.645 ± 0.020 | 0.657 ± 0.020 | 0.899 ± 0.003 | 0.643 ± 0.035 | 0.651 [0.623, 0.677] | 0.070 ± 0.026 |
| TF-IDF + LR | 1 | 0.743 | 0.727 | 0.735 | 0.925 | 0.703 | 0.727 [0.699, 0.754] | 0.057 |
| TF-IDF + SVM | 1 | 0.762 | 0.743 | 0.751 | 0.923 | 0.722 | 0.743 [0.717, 0.771] | 0.145 |
| XLM-R base | 3 | 0.649 ± 0.016 | 0.630 ± 0.018 | 0.638 ± 0.015 | 0.893 ± 0.019 | 0.628 ± 0.018 | 0.653 [0.626, 0.678] | 0.152 ± 0.009 |


### Validation vs. test, and near-duplicate subsets
**MC7 macro-F1 by evaluation subset: validation (in-distribution), full official test, test items with a near-duplicate in training, and the de-duplicated remainder (mean ± s.d. over seeds).**

| Model | Full test (1000) | Near-dup items (182) | De-duplicated (818) | Validation |
|---|---|---|---|---|
| HingRoBERTa | 0.721 ± 0.040 | 0.734 ± 0.015 | 0.708 ± 0.048 | 0.793 ± 0.008 |
| MuRIL | 0.632 ± 0.016 | 0.625 ± 0.010 | 0.628 ± 0.017 | 0.761 ± 0.004 |
| MuRIL + sarcasm fusion | 0.653 ± 0.009 | 0.630 ± 0.019 | 0.650 ± 0.007 | 0.755 ± 0.004 |
| MuRIL + fusion (shuffled-sarcasm features) | 0.648 ± 0.005 | 0.648 ± 0.026 | 0.640 ± 0.002 | 0.752 ± 0.003 |
| MuRIL + sarcasm MTL | 0.631 ± 0.009 | 0.601 ± 0.040 | 0.632 ± 0.009 | 0.754 ± 0.006 |
| MuRIL + shuffled-sarcasm MTL | 0.645 ± 0.020 | 0.613 ± 0.039 | 0.643 ± 0.035 | 0.754 ± 0.005 |
| TF-IDF + LR | 0.727 | 0.785 | 0.703 | 0.773 |
| TF-IDF + SVM | 0.743 | 0.790 | 0.722 | 0.763 |
| XLM-R base | 0.630 ± 0.018 | 0.595 ± 0.021 | 0.628 ± 0.018 | 0.769 ± 0.007 |


### Distribution-shift diagnostic
**Domain-classifier test of distribution shift (TF-IDF LR, 5-fold CV ROC-AUC; 0.5 = indistinguishable).**

| Comparison | ROC-AUC |
|---|---|
| official train vs official test | 0.884 |
| official train vs our validation (control) | 0.578 |
| class 'abusive': train vs test (vs val) | 0.941 (0.579) |
| class 'age': train vs test (vs val) | 0.921 (0.689) |
| class 'gender': train vs test (vs val) | 0.923 (0.637) |
| class 'mockery': train vs test (vs val) | 0.879 (0.561) |
| class 'not_cyberbullying': train vs test (vs val) | 0.960 (0.539) |
| class 'offensive': train vs test (vs val) | 0.911 (0.641) |
| class 'religion': train vs test (vs val) | 0.928 (0.699) |


### Paired comparisons
**Paired comparisons on the test set (B minus A). Bootstrap: 2,000 paired resamples of the seed-ensemble predictions; McNemar: exact test on per-item correctness. No multiple-comparison correction was applied; treat p-values as descriptive.**

| Comparison (B vs A) | Delta macro-F1 (ensemble) | 95% CI | p (bootstrap) | p (McNemar) | Seeds with Delta>0 |
|---|---|---|---|---|---|
| XLM-R base vs MuRIL | 0.011 | [-0.017, +0.038] | 0.460 | 0.771 | 1/3 |
| HingRoBERTa vs MuRIL | 0.091 | [+0.065, +0.118] | 0.000 | 0.000 | 3/3 |
| MuRIL + sarcasm MTL vs MuRIL | 0.006 | [-0.014, +0.026] | 0.552 | 0.744 | 2/3 |
| MuRIL + shuffled-sarcasm MTL vs MuRIL | 0.008 | [-0.010, +0.026] | 0.361 | 0.410 | 3/3 |
| MuRIL + sarcasm fusion vs MuRIL | 0.007 | [-0.013, +0.026] | 0.495 | 0.657 | 3/3 |
| TF-IDF + SVM vs MuRIL | 0.101 | [+0.072, +0.130] | 0.000 | 0.000 | 1/1 |
| TF-IDF + LR vs MuRIL | 0.085 | [+0.057, +0.113] | 0.000 | 0.000 | 1/1 |
| MuRIL + sarcasm MTL vs MuRIL + shuffled-sarcasm MTL | -0.002 | [-0.022, +0.017] | 0.821 | 0.741 | 1/3 |
| MuRIL + sarcasm fusion vs MuRIL + fusion (shuffled-sarcasm features) | -0.001 | [-0.019, +0.016] | 0.837 | 0.804 | 2/3 |
| XLM-R base vs MuRIL | 0.019 | [-0.006, +0.046] | 0.150 | 0.422 | 1/3 |
| MuRIL + sarcasm MTL vs MuRIL | -0.001 | [-0.021, +0.020] | 0.924 | 0.568 | 1/3 |
| MuRIL + sarcasm fusion vs MuRIL | 0.015 | [-0.004, +0.034] | 0.124 | 0.188 | 3/3 |
| TF-IDF + SVM vs MuRIL | 0.029 | [+0.001, +0.060] | 0.045 | 0.078 | 1/1 |
| MuRIL vs TF-IDF + LR | 0.067 | [+0.019, +0.115] | 0.003 | 0.326 | 1/1 |


### Binary task (MC2)
**MC2 (binary cyberbullying): test-set results (mean ± s.d. over seeds; ensemble = mean of seed probabilities with bootstrap 95% CI; dedup = test items without near-duplicates in training).**

| Model | Seeds | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | Macro-F1 (dedup test) | Ens. macro-F1 [95% CI] | ECE |
|---|---|---|---|---|---|---|---|---|
| Majority class | 1 | 0.800 | 0.444 | 0.711 | 0.500 | 0.441 | 0.444 [0.437, 0.452] | 0.002 |
| MuRIL | 3 | 0.912 ± 0.009 | 0.859 ± 0.006 | 0.911 ± 0.006 | 0.947 ± 0.006 | 0.867 ± 0.005 | 0.858 [0.830, 0.885] | 0.061 ± 0.017 |
| MuRIL + sarcasm fusion | 3 | 0.920 ± 0.008 | 0.872 ± 0.007 | 0.919 ± 0.006 | 0.939 ± 0.009 | 0.884 ± 0.009 | 0.873 [0.845, 0.898] | 0.053 ± 0.022 |
| MuRIL + sarcasm MTL | 3 | 0.904 ± 0.013 | 0.853 ± 0.016 | 0.905 ± 0.012 | 0.934 ± 0.009 | 0.860 ± 0.018 | 0.857 [0.828, 0.882] | 0.075 ± 0.011 |
| TF-IDF + LR | 1 | 0.925 | 0.879 | 0.924 | 0.960 | 0.876 | 0.879 [0.852, 0.904] | 0.025 |
| TF-IDF + SVM | 1 | 0.929 | 0.887 | 0.928 | 0.965 | 0.885 | 0.887 [0.859, 0.912] | 0.031 |
| XLM-R base | 3 | 0.910 ± 0.012 | 0.863 ± 0.017 | 0.911 ± 0.011 | 0.941 ± 0.006 | 0.871 ± 0.015 | 0.876 [0.850, 0.900] | 0.073 ± 0.009 |


### Prototype sarcasm-boost rule (MC2)
**Original SACENet additive sarcasm boost (p + 0.18*s if s >= 0.45; threshold 0.55) re-evaluated on MC2 with a trained sarcasm model, compared with the unboosted MuRIL classifier.**

| Seed | Setting | Macro-F1 | Recall (CB) | False-positive rate | Boosted items |
|---|---|---|---|---|---|
| 13 | base@0.50 | 0.854 | 0.940 | 0.230 | – |
| 13 | base@0.55 | 0.852 | 0.939 | 0.230 | – |
| 13 | boost@0.55 | 0.853 | 0.941 | 0.235 | 429 |
| 42 | base@0.50 | 0.859 | 0.931 | 0.190 | – |
| 42 | base@0.55 | 0.858 | 0.930 | 0.190 | – |
| 42 | boost@0.55 | 0.860 | 0.932 | 0.190 | 429 |
| 87 | base@0.50 | 0.866 | 0.981 | 0.315 | – |
| 87 | base@0.55 | 0.864 | 0.980 | 0.315 | – |
| 87 | boost@0.55 | 0.864 | 0.980 | 0.315 | 429 |


### Sarcasm detection (Swami test, hashtags removed)
**Sarcasm detection (Swami test, hashtags removed): test-set results (mean ± s.d. over seeds; ensemble = mean of seed probabilities with bootstrap 95% CI; dedup = test items without near-duplicates in training).**

| Model | Seeds | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | Macro-F1 (dedup test) | Ens. macro-F1 [95% CI] | ECE |
|---|---|---|---|---|---|---|---|---|
| Majority class | 1 | 0.904 | 0.475 | 0.858 | 0.500 | – | 0.475 [0.470, 0.480] | 0.000 |
| MuRIL | 3 | 0.912 ± 0.010 | 0.781 ± 0.002 | 0.918 ± 0.006 | 0.949 ± 0.008 | – | 0.806 [0.766, 0.842] | 0.046 ± 0.003 |
| MuRIL (shuffled training labels) | 1 | 0.904 | 0.475 | 0.858 | 0.433 | – | 0.475 [0.470, 0.480] | 0.376 |
| TF-IDF + LR | 1 | 0.912 | 0.739 | 0.911 | 0.935 | – | 0.739 [0.693, 0.783] | 0.041 |
| TF-IDF + SVM | 1 | 0.914 | 0.724 | 0.909 | 0.932 | – | 0.724 [0.675, 0.773] | 0.024 |


### External tests
**Cross-dataset evaluation without any re-training or threshold tuning. Label constructs differ from the training task (offensive language; differently sourced sarcasm), so these numbers measure transfer, not in-domain performance.**

| Test set | Model (trained on) | N | Macro-F1 (mean) | ROC-AUC (mean) | Runs |
|---|---|---|---|---|---|
| TweetEval-Offensive (en) | Majority class (MC2) | 860 | 0.218 | 0.500 | 1 |
| TweetEval-Offensive (en) | MuRIL (MC2) | 860 | 0.275 | 0.623 | 3 |
| TweetEval-Offensive (en) | MuRIL + sarcasm MTL (MC2) | 860 | 0.253 | 0.645 | 3 |
| TweetEval-Offensive (en) | TF-IDF + LR (MC2) | 860 | 0.235 | 0.544 | 1 |
| TweetEval-Offensive (en) | TF-IDF + SVM (MC2) | 860 | 0.253 | 0.552 | 1 |
| TweetEval-Offensive (en) | XLM-R base (MC2) | 860 | 0.317 | 0.678 | 3 |
| User sarcasm file (hi-en) | Majority class (Swami) | 8,455 | 0.363 | 0.500 | 1 |
| User sarcasm file (hi-en) | MuRIL (Swami) | 8,455 | 0.520 | 0.537 | 3 |
| User sarcasm file (hi-en) | TF-IDF + LR (Swami) | 8,455 | 0.448 | 0.443 | 1 |
| User sarcasm file (hi-en) | TF-IDF + SVM (Swami) | 8,455 | 0.433 | 0.450 | 1 |


### Explainability
**SHAP faithfulness, stability and overhead (MuRIL, MC7).**

| Measure | Value |
|---|---|
| deletion drop, top-1 vs random | 0.421 vs 0.092 (diff +0.329, 95% CI [+0.272, +0.385]; n=100) |
| deletion drop, top-2 vs random | 0.527 vs 0.164 (diff +0.363, 95% CI [+0.301, +0.425]; n=100) |
| deletion drop, top-3 vs random | 0.558 vs 0.209 (diff +0.349, 95% CI [+0.295, +0.403]; n=100) |
| rank stability: budget 100 vs 1000 | Spearman 0.950; top-3 overlap 0.83 (n=12) |
| rank stability: budget 250 vs 1000 | Spearman 0.986; top-3 overlap 0.94 (n=12) |
| rank stability: budget 500 vs 1000 | Spearman 0.997; top-3 overlap 0.97 (n=12) |
| SHAP vs leave-one-word-out occlusion | Spearman 0.273; top-3 overlap 0.57 (n=23) |
| time per explanation (cpu, 500 evals) | 0.87 s vs 23 ms inference (x38) |


### Early detection (CONDA proxy, English game chat)
**Early-warning proxy evaluation on CONDA test conversations (running-max policy; thresholds chosen on validation).**

| Turn scorer | F1 | Precision | Recall | False-alarm rate | Mean delay | ERDE_3 | ERDE_5 | Recall @FAR<=0.10(val) | FAR @FAR<=0.10(val) |
|---|---|---|---|---|---|---|---|---|---|
| conda__distilroberta__base__s13 | 0.913 | 0.916 | 0.909 | 0.051 | 0.010 | 0.161 | 0.097 | 0.909 | 0.051 |
| conda__distilroberta__base__s42 | 0.910 | 0.930 | 0.890 | 0.041 | 0.071 | 0.165 | 0.101 | 0.890 | 0.041 |
| conda__distilroberta__base__s87 | 0.914 | 0.918 | 0.911 | 0.050 | 0.008 | 0.161 | 0.096 | 0.911 | 0.050 |
| conda__tfidf_lr | 0.912 | 0.928 | 0.897 | 0.043 | 0.022 | 0.165 | 0.098 | 0.897 | 0.043 |
| conda_ctx__distilroberta__base__s13 | 0.906 | 0.948 | 0.868 | 0.029 | 0.148 | 0.171 | 0.109 | 0.868 | 0.029 |
| conda_ctx__distilroberta__base__s42 | 0.910 | 0.926 | 0.894 | 0.044 | 0.070 | 0.165 | 0.101 | 0.894 | 0.044 |
| conda_ctx__distilroberta__base__s87 | 0.910 | 0.927 | 0.894 | 0.043 | 0.058 | 0.165 | 0.101 | 0.894 | 0.043 |
| conda_ctx__tfidf_lr | 0.853 | 0.824 | 0.884 | 0.116 | -0.044 | 0.179 | 0.118 | 0.849 | 0.083 |


### Runtime and size
**Model size, inference latency and training time (Apple M1 Pro).**

| name | params_m | trainable_params_m | size_mb | latency_ms_bs1_cpu | latency_ms_bs1_mps | throughput_per_s_bs32_mps | train_minutes_mean |
|---|---|---|---|---|---|---|---|
| MuRIL | 237.562 | 86.047 | 453.113 | 23.749 | 11.307 | 272.041 | 13.761 |
| XLM-R base | 278.049 | 86.047 | 530.336 | 30.242 | 10.828 | 333.086 | 13.327 |
| HingRoBERTa | 278.049 | 86.047 | 530.336 | 29.936 | 10.479 | 346.571 | 13.193 |
| DistilRoBERTa | 82.124 | 43.520 | 156.639 | 15.972 | 6.095 | 512.006 | – |
| TF-IDF + LR | 0.784 | – | 9.973 | 0.571 | – | – | 0.458 |
| TF-IDF + SVM | 0.784 | – | 33.913 | 2.798 | – | – | 0.811 |


## 3. Findings in one paragraph each

- **RQ1.** On the official test split, character-n-gram TF-IDF models (0.743) and
  HingRoBERTa (0.721) outperform MuRIL (0.632)
  and XLM-R (0.630). On validation all are close. The official test split is
  distribution-shifted (domain-classifier AUC 0.884 vs. 0.578 control).
- **RQ2.** Multitask sarcasm supervision shows no reliable difference from the base model (Δ macro-F1 = +0.006, 95% CI [-0.014, +0.026]), and a shuffled-label control performs as well as real sarcasm labels (Δ = -0.002), so the sarcasm labels themselves add nothing measurable. Feature fusion with a separately trained sarcasm model shows no reliable difference from the base model (Δ = +0.007, 95% CI [-0.013, +0.026]). A Hinglish-pretrained encoder (HingRoBERTa) is the only transformer that approaches the TF-IDF baselines. A fusion control whose features come from a sarcasm model trained on permuted labels scores 0.648 ± 0.005; real vs. control features: no reliable difference (Δ = -0.001, 95% CI [-0.019, +0.016]). Any fusion effect is therefore not attributable to sarcasm knowledge.
- **RQ3.** SHAP is faithful under deletion (deleting the top-attributed word lowers the predicted-class probability by 0.421 vs. 0.092 for a random word), stable at ≥ 100 evaluations, and weakly aligned with
  occlusion (Spearman 0.273).
- **RQ4 (proxy only).** The causal alarm flags toxic conversations with alarm F1 0.913 at a 0.051 false-alarm rate, but almost always at the first toxic turn rather than before it. This does not validate Hinglish early detection.
- **Diagnostic.** A diagnostic run with the token-embedding table unfrozen could not be completed: two attempts (batch 32, and batch 16 with 2-step gradient accumulation) drove the 16 GB machine to critical memory pressure and were stopped. Whether freezing contributes to the gap therefore remains open.

## 4. Failures, superseded runs and deviations (full disclosure)

- **Config v1** (lr 3e-5 for encoder and head) under-fitted. MC7 MuRIL s13 reached val 0.774 / test 0.662 with temperature
  0.23. Its test result was seen and prompted a validation-only pilot (`results/tuning/pilot_summary.json`). The 5 runs are
  kept in `results/runs_superseded/`.
- **Config v2** (encoder lr 5e-5) collapsed to the majority class on the sarcasm task (2 runs, kept in
  `results/runs_superseded/config_v2_collapsed/`). The final config (3e-5 / 1e-3) is stable on all tasks' validation sets.
  Collapsed runs among final results: 1.
- **Frozen token-embedding table** for all models, for compute reasons (MPS optimiser cost). The unfrozen diagnostic was
  attempted twice and abandoned at critical memory pressure (`results/runs_superseded/not_feasible/README.json`).
- **CONDA** official split is utterance-level, so it was re-split by match. Conversations are short and partially incomplete.
- **Swami sarcasm corpus**: retrieval hashtags leaked labels and were removed from the model input.
- **Watchdog pauses**: training was paused once for ~17 s by an over-strict swap rule (then corrected), and the unfrozen
  diagnostic was paused twice at critical memory pressure (`results/logs/watchdog.log`).
- **Bug-induced re-runs**: `mc2__muril__fusion__s13` crashed in its external evaluation (fusion features do not exist for
  OLID texts); `s42` was stopped pre-emptively. Both were re-run after the fix (external evaluation is skipped for fusion runs).
- **Collapse flag**: `sarcasm__muril__shuffled_labels__s13` (the fusion-control feature extractor) predicts the majority class,
  as expected for a model trained on permuted labels.
- **Not done**: unfrozen-embedding diagnostic (memory); HASOC/TRAC evaluation (access required); Hinglish conversation-level early detection (no data); human
  evaluation of explanations; per-model hyperparameter search.

## 5. Generated assets
- Figures: `results/figures/**` (captions: `paper/FIGURE_CAPTIONS.md`)
- Tables: `results/tables/*` (captions: `paper/TABLE_CAPTIONS.md`)
- Paper: `paper/SACENet_Research_Paper_Draft.{md,tex}`, `paper/build/SACENet_Research_Paper_Draft.pdf`; v1 in `paper/SACENet_Research_Paper_Draft_v1.md`
- Registry: `results/experiment_registry.{jsonl,csv}`; verification: `results/verification_report.json`

## 6. Recommended next steps
1. Resolve `paper/Author_Review_Notes.md` §A (provenance of user files, literature currency).
2. Request HASOC/TRAC access and add them as external tests.
3. Run full fine-tuning with a per-model learning-rate search as a robustness appendix.
4. Annotate sarcasm on a sample of MC-Hinglish to test the sarcasm hypothesis directly.
