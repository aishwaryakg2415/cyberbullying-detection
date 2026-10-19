"""Append-only experiment registry (results/experiment_registry.jsonl).

One line per state change (started / completed / failed). `latest()` collapses
to the most recent entry per experiment id; scripts/run_experiments.py also
exports results/experiment_registry.csv.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .io import RESULTS_DIR

REGISTRY = RESULTS_DIR / "experiment_registry.jsonl"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def log(entry: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"logged_at": now(), **entry}, ensure_ascii=False) + "\n")


def latest() -> dict[str, dict]:
    if not REGISTRY.exists():
        return {}
    out: dict[str, dict] = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            out[e["exp_id"]] = {**out.get(e["exp_id"], {}), **e}
    return out


def export_csv(path: Path | None = None) -> Path:
    import pandas as pd

    rows = list(latest().values())
    path = path or RESULTS_DIR / "experiment_registry.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
