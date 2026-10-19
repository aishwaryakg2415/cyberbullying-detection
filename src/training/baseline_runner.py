"""Fit/evaluate a classical baseline and save the same artefacts as transformer runs."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from ..evaluation.metrics import classification_metrics
from ..models.baselines import BASELINES
from ..utils.env import memory_snapshot, package_versions, set_seed
from ..utils.io import load_json, save_json, save_jsonl
from .data import TASKS, load_split


def run_baseline(cfg: dict, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.json").exists():
        return load_json(run_dir / "metrics.json")
    save_json(cfg, run_dir / "config.json")
    save_json({"device": {"selected": "cpu"}, "packages": package_versions()}, run_dir / "environment.json")
    set_seed(cfg["seed"])
    task = cfg["task"]
    labels = TASKS[task]["labels"]
    data = {s: load_split(task, s) for s in ("train", "val", "test")}
    m = BASELINES[cfg["model"]]()
    t0 = time.perf_counter()
    _, Xtr, ytr, _ = data["train"]
    _, Xva, yva, _ = data["val"]
    if cfg["model"] == "majority":
        m.fit(Xtr, ytr, n_classes=len(labels))
    else:
        m.fit(Xtr, ytr, Xva, yva, n_classes=len(labels), seed=cfg["seed"])
    fit_s = time.perf_counter() - t0
    out = {"exp_id": cfg["exp_id"], "task": task, "labels": labels, "train_seconds": fit_s,
           "selected": getattr(m, "selected", {}), "device": "cpu"}
    for split in ("val", "test"):
        ids, X, y, _ = data[split]
        P = m.predict_proba(X)
        # Ensure full class coverage (majority prior already has K columns).
        if P.shape[1] != len(labels):
            full = np.zeros((len(X), len(labels)))
            full[:, m.model.classes_] = P
            P = full
        out[split] = classification_metrics(y, P, labels)
        save_jsonl([{"id": i, "y": int(t), "prob": p.tolist()} for i, t, p in zip(ids, y, P)],
                   run_dir / f"predictions_{split}.jsonl")
    for extra in cfg.get("extra_eval", []) or []:
        ids, X, y, _ = load_split(extra, "test")
        P = m.predict_proba(X)
        save_jsonl([{"id": i, "y": int(t), "prob": p.tolist()} for i, t, p in zip(ids, y, P)],
                   run_dir / f"predictions_{extra}.jsonl")
    # Size + latency (single text, CPU)
    blob = pickle.dumps(m)
    out["model_bytes"] = len(blob)
    _, Xte, _, _ = data["test"]
    lat = []
    for t in Xte[:200]:
        s = time.perf_counter()
        m.predict_proba([t])
        lat.append((time.perf_counter() - s) * 1000)
    out["latency_ms_cpu_bs1"] = {"mean": float(np.mean(lat)), "p50": float(np.median(lat)),
                                 "p95": float(np.percentile(lat, 95))}
    out["memory_end"] = memory_snapshot()
    save_json(out, run_dir / "metrics.json")
    if cfg.get("keep_model"):
        (run_dir / "model.pkl").write_bytes(blob)
    return out
