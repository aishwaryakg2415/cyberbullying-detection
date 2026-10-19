"""SHAP explanations + faithfulness + stability for the retained MuRIL (MC7, seed 13) model.

Usage:  python scripts/run_explainability.py [--device cpu] [--n-faith 100]
Requires: results/runs/mc7__muril__base__s13/checkpoint/model and results/analysis/analysis.json
Outputs: results/explainability/{examples,faithfulness,stability,overhead}.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability.explain import (deletion_test, occlusion_words, rank_agreement, shap_explain,  # noqa: E402
                                        to_words)
from src.inference.predictor import Predictor  # noqa: E402
from src.utils.io import PROCESSED_DIR, RESULTS_DIR, RUNS_DIR, load_json, load_jsonl, save_json  # noqa: E402

OUT = RESULTS_DIR / "explainability"
MODEL = RUNS_DIR / "mc7__muril__base__s13" / "checkpoint" / "model"


def pick_examples(ea: dict, meta: dict, max_words: int = 18) -> list[dict]:
    """Deterministic selection: per-class confident correct, FPs, FNs, wrong-type, ambiguous."""
    short = lambda e: len(meta[e["id"]]["text"].split()) <= max_words  # noqa: E731
    sel, seen = [], set()

    def add(e, kind):
        if e["id"] not in seen and short(e):
            seen.add(e["id"])
            sel.append({**e, "kind": kind})
            return True
        return False

    by_class = {}
    for e in ea["examples"].get("correct", []):
        by_class.setdefault(e["y"], []).append(e)
    # analysis keeps only the top 25 correct; fall back to any class present
    for c, es in sorted(by_class.items()):
        for e in es:
            if add(e, "correct"):
                break
    for kind, n in (("false_positive", 4), ("false_negative", 4), ("wrong_type", 4)):
        k = 0
        for e in ea["examples"].get(kind, []):
            if k < n and add(e, kind):
                k += 1
    k = 0
    for e in ea["ambiguous"]:
        if k < 4 and add({**e, "conf": None}, "ambiguous"):
            k += 1
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu avoids contending with GPU training")
    ap.add_argument("--n-faith", type=int, default=100)
    ap.add_argument("--max-evals", type=int, default=500)
    args = ap.parse_args()
    if not MODEL.exists():
        print(f"missing {MODEL}; train mc7__muril__base__s13 first")
        return 1
    pred = Predictor(MODEL, device=args.device, calibrated=False)
    an = load_json(RESULTS_DIR / "analysis" / "analysis.json")
    ea = an["error_analysis_mc7"]
    meta = {r["id"]: r for r in load_jsonl(PROCESSED_DIR / "mc7" / "test.jsonl")}

    # 1) Qualitative examples (full budget) + occlusion + deletion
    ex = pick_examples(ea, meta)
    texts = [meta[e["id"]]["text"] for e in ex]
    t0 = time.perf_counter()
    sh = shap_explain(pred, texts, max_evals=args.max_evals)
    shap_secs = time.perf_counter() - t0
    out_ex = []
    for e, s in zip(ex, sh):
        words, wv = to_words(s["tokens"], s["values"])
        ow, od, _ = occlusion_words(pred, s["text"], s["target"])
        agree = rank_agreement(wv, od) if len(ow) == len(words) else {"spearman": None, "top3_overlap": None,
                                                                      "note": "word segmentation mismatch"}
        dele = deletion_test(pred, s["text"], words, wv, s["target"])
        out_ex.append({**e, **s, "words": words, "word_values": wv.tolist(), "occlusion": od.tolist(),
                       "shap_vs_occlusion": agree, "deletion": dele, "y_true": e["y"]})
    save_json(out_ex, OUT / "examples.json")

    # 2) Quantitative faithfulness on a random test sample (fixed seed), reduced budget
    rng = np.random.default_rng(0)
    ids = [i for i in meta if len(meta[i]["text"].split()) >= 4]
    sample = [ids[i] for i in rng.choice(len(ids), min(args.n_faith, len(ids)), replace=False)]
    fs = shap_explain(pred, [meta[i]["text"] for i in sample], max_evals=300)
    rows = []
    for i, s in zip(sample, fs):
        words, wv = to_words(s["tokens"], s["values"])
        rows.append({"id": i, "target": s["target_label"], "seconds": s["seconds"],
                     **deletion_test(pred, s["text"], words, wv, s["target"], seed=1)})
    summ = {}
    for k in (1, 2, 3):
        top = [r[f"drop_top{k}"] for r in rows if f"drop_top{k}" in r]
        rnd = [r[f"drop_rand{k}"] for r in rows if f"drop_rand{k}" in r]
        diff = np.array(top) - np.array(rnd)
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        summ[f"k={k}"] = {"n": len(top), "mean_drop_top": float(np.mean(top)), "mean_drop_random": float(np.mean(rnd)),
                          "mean_diff": float(diff.mean()), "diff_ci95": [float(np.quantile(boot, .025)),
                                                                         float(np.quantile(boot, .975))],
                          "share_top_gt_random": float((diff > 0).mean())}
    save_json({"budget_max_evals": 300, "summary": summ, "rows": rows}, OUT / "faithfulness.json")

    # 3) Stability across evaluation budgets (first 12 examples): 100/250 vs 1000
    stab = []
    for s in sh[:12]:
        ref = shap_explain(pred, [s["text"]], [s["target"]], max_evals=1000)[0]
        _, rv = to_words(ref["tokens"], ref["values"])
        row = {"text_len_words": len(s["text"].split())}
        for b in (100, 250, 500):
            cur = s if b == args.max_evals else shap_explain(pred, [s["text"]], [s["target"]], max_evals=b)[0]
            _, cv = to_words(cur["tokens"], cur["values"])
            row[f"budget_{b}"] = rank_agreement(cv, rv) if len(cv) == len(rv) else None
        stab.append(row)
    agg = {}
    for b in (100, 250, 500):
        sp = [r[f"budget_{b}"]["spearman"] for r in stab if r.get(f"budget_{b}") and r[f"budget_{b}"]["spearman"] is not None]
        ov = [r[f"budget_{b}"]["top3_overlap"] for r in stab if r.get(f"budget_{b}")]
        agg[f"budget_{b}_vs_1000"] = {"spearman_mean": float(np.mean(sp)) if sp else None,
                                      "top3_overlap_mean": float(np.mean(ov)) if ov else None, "n": len(sp)}
    occ = [e["shap_vs_occlusion"] for e in out_ex if e["shap_vs_occlusion"].get("spearman") is not None]
    save_json({"budget_vs_1000": agg, "rows": stab,
               "shap_vs_occlusion": {"spearman_mean": float(np.mean([o["spearman"] for o in occ])),
                                     "top3_overlap_mean": float(np.mean([o["top3_overlap"] for o in occ])),
                                     "n": len(occ)}}, OUT / "stability.json")

    # 4) Overhead: SHAP seconds per text vs plain inference latency (same device)
    lat = []
    for t in texts[:20]:
        s0 = time.perf_counter()
        pred.predict_proba([t], clean=False)
        lat.append(time.perf_counter() - s0)
    save_json({"device": args.device, "shap_max_evals": args.max_evals,
               "shap_seconds_per_text_mean": float(np.mean([s["seconds"] for s in sh])),
               "shap_seconds_per_text_budget300_mean": float(np.mean([r["seconds"] for r in rows])),
               "inference_seconds_per_text_mean": float(np.mean(lat)),
               "overhead_factor": float(np.mean([s["seconds"] for s in sh]) / np.mean(lat)),
               "total_qualitative_seconds": shap_secs}, OUT / "overhead.json")
    print("faithfulness:", summ)
    print("stability:", agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
