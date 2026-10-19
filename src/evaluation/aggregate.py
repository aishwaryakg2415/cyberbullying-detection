"""Collect run artefacts, recompute metrics from saved predictions, aggregate seeds."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..training.data import TASKS
from ..utils.io import PROCESSED_DIR, RUNS_DIR, load_json, load_jsonl
from .metrics import classification_metrics

PRETTY = {
    "majority": "Majority class", "tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM",
    ("muril", "base"): "MuRIL", ("xlmr", "base"): "XLM-R base", ("hingroberta", "base"): "HingRoBERTa",
    ("muril", "mtl"): "MuRIL + sarcasm MTL", ("muril", "mtl_shuffled"): "MuRIL + shuffled-sarcasm MTL",
    ("muril", "fusion"): "MuRIL + sarcasm fusion", ("muril", "full_ft"): "MuRIL (embeddings unfrozen)",
    ("muril", "fusion_shuffled"): "MuRIL + fusion (shuffled-sarcasm features)",
    ("muril", "shuffled_labels"): "MuRIL (shuffled training labels)", ("distilroberta", "base"): "DistilRoBERTa",
}


def group_key(cfg: dict) -> str:
    if cfg.get("kind") == "baseline":
        return f"{cfg['task']}|{cfg['model']}"
    return f"{cfg['task']}|{cfg['model_key']}|{cfg['variant']}"


def pretty(gk: str) -> str:
    parts = gk.split("|")
    if len(parts) == 2:
        return PRETTY.get(parts[1], parts[1])
    return PRETTY.get((parts[1], parts[2]), f"{parts[1]} {parts[2]}")


def collect_runs() -> dict[str, list[dict]]:
    """group_key -> list of {exp_id, cfg, metrics, dir} for completed, non-smoke runs."""
    out = defaultdict(list)
    for d in sorted(RUNS_DIR.iterdir()):
        if d.name.startswith("_") or not (d / "metrics.json").exists():
            continue
        cfg = load_json(d / "config.json")
        if cfg.get("skip_test"):   # validation-only tuning runs are not results
            continue
        out[group_key(cfg)].append({"exp_id": d.name, "cfg": cfg, "metrics": load_json(d / "metrics.json"), "dir": d})
    return dict(out)


def load_preds(run_dir: Path, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = load_jsonl(run_dir / f"predictions_{split}.jsonl")
    return [r["id"] for r in rows], np.array([r["y"] for r in rows]), np.array([r["prob"] for r in rows])


def ensemble(runs: list[dict], split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Mean probability over seeds (rows aligned by id)."""
    ids0, y0, P = load_preds(runs[0]["dir"], split)
    acc = P.copy()
    for r in runs[1:]:
        ids, y, Pr = load_preds(r["dir"], split)
        idx = {k: i for i, k in enumerate(ids)}
        acc += Pr[[idx[k] for k in ids0]]
    return ids0, y0, acc / len(runs)


def test_meta(task: str) -> dict[str, dict]:
    return {r["id"]: r for r in load_jsonl(PROCESSED_DIR / task / "test.jsonl")}


def seed_summary(runs: list[dict], split: str, labels: list[str], subset: set | None = None) -> dict:
    """Mean/std over seeds of metrics recomputed from predictions (optionally on an id subset)."""
    vals = defaultdict(list)
    for r in runs:
        ids, y, P = load_preds(r["dir"], split)
        if subset is not None:
            m = np.array([i in subset for i in ids])
            y, P = y[m], P[m]
        mm = classification_metrics(y, P, labels)
        for k in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier", "roc_auc", "pr_auc",
                  "roc_auc_macro_ovr", "pr_auc_macro", "f1_positive"):
            if k in mm:
                vals[k].append(mm[k])
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n_seeds": len(v), "values": v} for k, v in vals.items()}


def labels_of(task: str) -> list[str]:
    return TASKS[task]["labels"]
