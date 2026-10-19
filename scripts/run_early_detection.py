"""Conversation-level early-detection evaluation on CONDA (proxy study).

Requires the turn-level scorer runs from `run_experiments.py --group early_conda`.
For each scorer: thresholds are chosen on VALIDATION conversations, then applied
once to TEST conversations. Output: results/early_detection/*.json

Usage:  python scripts/run_early_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.early import build_conversations, choose_tau, evaluate_alarms, prefix_curve  # noqa: E402
from src.utils.io import PROCESSED_DIR, RESULTS_DIR, RUNS_DIR, load_jsonl, save_json  # noqa: E402

OUT = RESULTS_DIR / "early_detection"


def scores_of(run: Path, split: str) -> dict[str, float]:
    return {r["id"]: r["prob"][1] for r in load_jsonl(run / f"predictions_{split}.jsonl")}


def main() -> int:
    rows = {s: load_jsonl(PROCESSED_DIR / "conda" / f"{s}.jsonl") for s in ("val", "test")}
    runs = sorted(p for p in RUNS_DIR.glob("conda*") if (p / "metrics.json").exists())
    if not runs:
        print("No CONDA scorer runs found; run: python scripts/run_experiments.py --group early_conda")
        return 1
    summary = {}
    for run in runs:
        res = {"scorer": run.name, "policies": {}}
        for policy in ("max", "ema"):
            val = build_conversations(rows["val"], scores_of(run, "val"))
            test = build_conversations(rows["test"], scores_of(run, "test"))
            tau_f1 = choose_tau(val, policy)
            tau_far = choose_tau(val, policy, max_far=0.10)
            res["policies"][policy] = {
                "tau_maxF1_val": tau_f1, "tau_far10_val": tau_far,
                "test_at_maxF1": evaluate_alarms(test, tau_f1, policy),
                "test_at_far10": evaluate_alarms(test, tau_far, policy),
                "val_at_maxF1": evaluate_alarms(val, tau_f1, policy),
                "prefix_curve_test": prefix_curve(test, tau_f1, policy, max_k=10),
            }
        save_json(res, OUT / f"{run.name}.json")
        m = res["policies"]["max"]["test_at_maxF1"]
        summary[run.name] = {k: m[k] for k in ("f1", "precision", "recall", "false_alarm_rate", "delay_mean",
                                               "delay_median", "erde_3", "erde_5", "alarms_before_onset")}
        print(f"{run.name:45s} F1={m['f1']:.3f} FAR={m['false_alarm_rate']:.3f} delay={m['delay_mean']:.2f} "
              f"ERDE5={m['erde_5']:.3f}")
    save_json(summary, OUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
