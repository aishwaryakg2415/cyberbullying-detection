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
