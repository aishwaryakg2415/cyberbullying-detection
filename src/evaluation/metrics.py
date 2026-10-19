"""Classification metrics, uncertainty estimates, calibration and paired tests.

All functions take plain arrays so they can be re-run on saved predictions
(scripts/verify_results.py recomputes every reported metric from the
predictions files).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
    """Top-label ECE with equal-width bins; also returns the reliability table."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - c)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": float(c), "acc": float(acc)})
    return {"ece": float(ece), "bins": rows}


def brier_score(y: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score: mean over items of sum_k (p_k - 1[y=k])^2."""
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def classification_metrics(y_true: Sequence[int], probs: np.ndarray, label_names: list[str],
                           y_pred: Optional[Sequence[int]] = None) -> dict:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(1) if y_pred is None else np.asarray(y_pred, dtype=int)
    K = len(label_names)
    P, R, F, S = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": macro_f1(y, pred),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_precision": float(P.mean()),
        "macro_recall": float(R.mean()),
        "per_class": {label_names[k]: {"precision": float(P[k]), "recall": float(R[k]), "f1": float(F[k]),
                                       "support": int(S[k])} for k in range(K)},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(K))).tolist(),
        "labels": label_names,
    }
    present = np.unique(y)
    if probs.shape[1] == 2:
        if len(present) == 2:
            out["roc_auc"] = float(roc_auc_score(y, probs[:, 1]))
            out["pr_auc"] = float(average_precision_score(y, probs[:, 1]))
        out["f1_positive"] = float(F[1])
    elif len(present) == K:
        out["roc_auc_macro_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        onehot = np.eye(K)[y]
        out["pr_auc_macro"] = float(np.mean([average_precision_score(onehot[:, k], probs[:, k]) for k in range(K)]))
    # Probability-quality metrics are only meaningful for real probabilities.
    if np.allclose(probs.sum(1), 1, atol=1e-4):
        ece = expected_calibration_error(y, probs)
        out["ece"] = ece["ece"]
        out["reliability"] = ece["bins"]
        out["brier"] = brier_score(y, probs)
        out["nll"] = nll(y, probs)
    return out


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, fn: Callable = macro_f1, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(y[idx], pred[idx])
    return {"point": float(fn(y, pred)), "lo": float(np.quantile(stats, alpha / 2)),
            "hi": float(np.quantile(stats, 1 - alpha / 2)), "n_boot": n_boot}


def paired_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, fn: Callable = macro_f1,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference fn(b) - fn(a) with 95% CI and a two-sided bootstrap p-value.

    p is estimated as 2*min(P(delta<=0), P(delta>=0)) over resamples
    (percentile method; see Koehn 2004 for the NLP usage of paired bootstrap).
    """
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = fn(y[idx], b[idx]) - fn(y[idx], a[idx])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(fn(y, b) - fn(y, a)), "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)), "p_value": float(min(1.0, p)), "n_boot": n_boot}


def mcnemar_exact(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on per-item correctness."""
    from scipy.stats import binomtest

    ca, cb = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    n01 = int((ca & ~cb).sum())   # a right, b wrong
    n10 = int((~ca & cb).sum())   # a wrong, b right
    p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017) fitted by minimising NLL on held-out logits."""
    from scipy.optimize import minimize_scalar

    logits, y = np.asarray(logits, float), np.asarray(y, int)

    def f(t):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()

    res = minimize_scalar(f, bounds=(0.05, 20), method="bounded")
    return float(res.x)


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
