**Sarcasm detection (Swami test, hashtags removed): test-set results (mean ± s.d. over seeds; ensemble = mean of seed probabilities with bootstrap 95% CI; dedup = test items without near-duplicates in training).**

| Model | Seeds | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | Macro-F1 (dedup test) | Ens. macro-F1 [95% CI] | ECE |
|---|---|---|---|---|---|---|---|---|
| Majority class | 1 | 0.904 | 0.475 | 0.858 | 0.500 | – | 0.475 [0.470, 0.480] | 0.000 |
| MuRIL | 3 | 0.912 ± 0.010 | 0.781 ± 0.002 | 0.918 ± 0.006 | 0.949 ± 0.008 | – | 0.806 [0.766, 0.842] | 0.046 ± 0.003 |
| MuRIL (shuffled training labels) | 1 | 0.904 | 0.475 | 0.858 | 0.433 | – | 0.475 [0.470, 0.480] | 0.376 |
| TF-IDF + LR | 1 | 0.912 | 0.739 | 0.911 | 0.935 | – | 0.739 [0.693, 0.783] | 0.041 |
| TF-IDF + SVM | 1 | 0.914 | 0.724 | 0.909 | 0.932 | – | 0.724 [0.675, 0.773] | 0.024 |
