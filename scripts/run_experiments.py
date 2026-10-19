"""Run experiment groups from configs/experiments.yaml and record them in the registry.

Examples (PowerShell or bash):
    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --smoke                 # 1-epoch, 64-example sanity run
    python scripts/run_experiments.py --group baselines
    python scripts/run_experiments.py --group mc7_backbones --seeds 13
    python scripts/run_experiments.py --only "mc7__muril__base__s13"

Completed runs (metrics.json present) are skipped; interrupted transformer runs
resume from their last epoch checkpoint.
"""
from __future__ import annotations

import argparse
import fnmatch
import logging
import sys
import time
import traceback
from pathlib import Path

import os

import yaml

# Use only the locally cached, revision-pinned weights (scripts/download_models.py).
# Without this, transformers may fetch an auto-converted safetensors file from a
# Hub PR branch instead of the pinned revision.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import registry  # noqa: E402
from src.utils.env import hardware_summary  # noqa: E402
from src.utils.io import ROOT, RUNS_DIR  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
BASELINE_MODELS = {"majority", "tfidf_lr", "tfidf_svm"}


def expand(group: str, seeds: list[int]) -> list[dict]:
    runs = []
    for spec in CFG["groups"][group]:
        if "models" in spec:   # classical baselines (deterministic given data; one seed)
            for m in spec["models"]:
                runs.append({"exp_id": f"{spec['task']}__{m}", "kind": "baseline", "task": spec["task"],
                             "model": m, "seed": 0, "extra_eval": spec.get("extra_eval", []), "group": group})
            continue
        mdl = CFG["models"][spec["model"]]
        for s in seeds:
            cfg = {**CFG["train_defaults"], "kind": "transformer", "group": group,
                   "exp_id": f"{spec['task']}__{spec['model']}__{spec['variant']}__s{s}",
                   "task": spec["task"], "model_key": spec["model"], "model_name": mdl["name"],
                   "revision": mdl["revision"], "variant": spec["variant"], "seed": s,
                   "extra_eval": spec.get("extra_eval", []),
                   "keep_checkpoint": spec.get("keep_seed") == s,
                   "export_features": spec.get("export_features") if spec.get("export_seed") == s else None}
            for k in ("epochs", "max_len", "batch_size", "lr", "head_lr", "skip_test", "patience",
                      "freeze_word_embeddings", "shuffle_train_labels", "keep_checkpoint", "grad_accum",
                      "eval_batch_size"):   # per-spec overrides
                if k in spec:
                    cfg[k] = spec[k]
            if spec.get("aux"):
                cfg["aux"] = dict(spec["aux"])
            if spec.get("fusion_from"):
                cfg["fusion"] = {"from": spec["fusion_from"],
                                 "features": str(RUNS_DIR / spec["fusion_from"] / "features_mc7.npz")}
            runs.append(cfg)
    return runs


def run_one(cfg: dict) -> dict:
    run_dir = RUNS_DIR / cfg["exp_id"]
    if (run_dir / "metrics.json").exists():
        print(f"[done] {cfg['exp_id']}")
        return {"status": "completed"}
    registry.log({"exp_id": cfg["exp_id"], "status": "started", "task": cfg["task"], "group": cfg["group"],
                  "model": cfg.get("model_name", cfg.get("model")), "variant": cfg.get("variant", "baseline"),
                  "seed": cfg["seed"], "hardware": hardware_summary(), "run_dir": str(run_dir.relative_to(ROOT)),
                  "split_ids": f"data/splits/{cfg['task']}/", "started_at": registry.now()})
    t0 = time.perf_counter()
    try:
        if cfg["kind"] == "baseline":
            from src.training.baseline_runner import run_baseline
            out = run_baseline(cfg, run_dir)
        else:
            from src.training.trainer import train_run
            out = train_run(cfg, run_dir)
        test = out.get("test", {})
        registry.log({"exp_id": cfg["exp_id"], "status": "completed", "ended_at": registry.now(),
                      "wall_seconds": round(time.perf_counter() - t0, 1), "device": out.get("device"),
                      "test_macro_f1": round(test["macro_f1"], 4) if test else None,
                      "test_accuracy": round(test["accuracy"], 4) if test else None,
                      "val_macro_f1": round(out["val"]["macro_f1"], 4), "collapsed": out.get("collapsed"),
                      "outputs": str(run_dir.relative_to(ROOT))})
        print(f"[ok]   {cfg['exp_id']}: " + (f"test macro-F1={test['macro_f1']:.4f}" if test else
                                                 f"val macro-F1={out['val']['macro_f1']:.4f} (tuning run; test not evaluated)"))
        return {"status": "completed"}
    except Exception as e:
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb)
        registry.log({"exp_id": cfg["exp_id"], "status": "failed", "ended_at": registry.now(),
                      "error": f"{type(e).__name__}: {e}"})
        print(f"[FAIL] {cfg['exp_id']}: {e}\n{tb}")
        return {"status": "failed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="*", default=[])
    ap.add_argument("--only", default=None, help="glob over exp ids")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run (not a result)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    seeds = args.seeds or CFG["seeds"]

    if args.smoke:
        mdl = CFG["models"]["muril"]
        cfg = {**CFG["train_defaults"], "kind": "transformer", "group": "smoke", "exp_id": "_smoke__mc7__muril",
               "task": "mc7", "model_name": mdl["name"], "revision": mdl["revision"], "variant": "smoke",
               "seed": 0, "epochs": 1, "smoke_n": 64, "aux": {"task": "sarcasm", "lambda": 0.5}}
        import shutil
        shutil.rmtree(RUNS_DIR / cfg["exp_id"], ignore_errors=True)
        return 0 if run_one(cfg)["status"] == "completed" else 1

    groups = args.group or list(CFG["groups"])
    runs = [r for g in groups for r in expand(g, seeds)]
    if args.only:
        runs = [r for r in runs if fnmatch.fnmatch(r["exp_id"], args.only)]
    if args.list:
        for r in runs:
            print(r["exp_id"])
        return 0
    failed = sum(run_one(r)["status"] == "failed" for r in runs)
    registry.export_csv()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
