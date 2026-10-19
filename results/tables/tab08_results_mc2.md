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
