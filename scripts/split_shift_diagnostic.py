"""Is the official MC-Hinglish test split drawn from the same distribution as train?

A "domain classifier" (TF-IDF word+char LR) is trained to tell train texts from test texts with
5-fold stratified CV; ROC-AUC ~0.5 means indistinguishable. Control: train vs. our validation split
(carved from the same official train split), which should be ~0.5. Output: results/analysis/split_shift.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.models.baselines import TfidfFeatures  # noqa: E402
from src.utils.io import PROCESSED_DIR, RESULTS_DIR, load_jsonl, save_json  # noqa: E402


def auc(a, b, seed=0):
    texts = [r["text"] for r in a] + [r["text"] for r in b]
    y = np.array([0] * len(a) + [1] * len(b))
    X = TfidfFeatures().fit_transform(texts)
    p = cross_val_predict(LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    tr = load_jsonl(PROCESSED_DIR / "mc7" / "train.jsonl")
    va = load_jsonl(PROCESSED_DIR / "mc7" / "val.jsonl")
    te = load_jsonl(PROCESSED_DIR / "mc7" / "test.jsonl")
    rng = np.random.default_rng(0)
    out = {"note": "ROC-AUC of a TF-IDF LR domain classifier, 5-fold CV. 0.5 = splits indistinguishable."}
    sub = [tr[i] for i in rng.choice(len(tr), len(te), replace=False)]   # equal sizes
    out["train_vs_test"] = auc(sub, te)
    out["train_vs_val (control)"] = auc([tr[i] for i in rng.choice(len(tr), len(va), replace=False)], va)
    per = {}
    for lab in sorted({r["cb_type"] for r in te}):
        a = [r for r in tr if r["cb_type"] == lab]
        b = [r for r in te if r["cb_type"] == lab]
        a = [a[i] for i in rng.choice(len(a), min(len(a), 4 * len(b)), replace=False)]
        per[lab] = {"train_vs_test": auc(a, b), "n_test": len(b)}
        av = [r for r in va if r["cb_type"] == lab]
        per[lab]["train_vs_val"] = auc([x for x in a if x not in av][: 4 * len(av)], av)
    out["per_class"] = per
    save_json(out, RESULTS_DIR / "analysis" / "split_shift.json")
    print(out)


if __name__ == "__main__":
    main()
