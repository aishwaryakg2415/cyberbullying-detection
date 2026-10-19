"""End-to-end consistency checks for every reported number.

1. raw data hashes/counts match data/dataset_manifest.json
2. each run's prediction IDs == committed split IDs; gold labels == processed labels
3. metrics recomputed from predictions == metrics.json (accuracy, macro-F1)
4. transformer runs record the device actually used
5. every result figure has its underlying data file
6. paper values were rendered from the current analysis.json; count remaining PENDING markers

Usage:  python scripts/verify_results.py      (exit code 1 if any check fails)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.training.data import TASKS, encode_label  # noqa: E402
from src.utils.io import (FIG_DIR, PAPER_DIR, PROCESSED_DIR, RESULTS_DIR, ROOT, RUNS_DIR, SPLITS_DIR,  # noqa: E402
                          load_json, load_jsonl, save_json)


def main() -> int:
    rep = {"checks": [], "failures": []}

    def check(name, ok, detail=""):
        rep["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            rep["failures"].append(name + (f": {detail}" if detail else ""))

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_datasets.py")], capture_output=True, text=True)
    check("dataset hashes and counts match manifest", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])

    gold_cache = {}
    n_runs = 0
    for d in sorted(RUNS_DIR.iterdir()):
        if d.name.startswith("_") or not (d / "metrics.json").exists():
            continue
        n_runs += 1
        cfg, m = load_json(d / "config.json"), load_json(d / "metrics.json")
        task = cfg["task"]
        for split in (("val",) if cfg.get("skip_test") else ("val", "test")):
            pf = d / f"predictions_{split}.jsonl"
            if not pf.exists():
                check(f"{d.name}/{split} predictions exist", False)
                continue
            preds = load_jsonl(pf)
            ids = [p["id"] for p in preds]
            ref_ids = load_json(SPLITS_DIR / task / f"{split}_ids.json")
            check(f"{d.name}/{split}: prediction ids == split ids", sorted(ids) == sorted(ref_ids),
                  f"{len(ids)} vs {len(ref_ids)}")
            key = (task, split)
            if key not in gold_cache:
                field = TASKS[task]["field"]
                gold_cache[key] = {r["id"]: encode_label(task, r[field]) for r in load_jsonl(PROCESSED_DIR / task / f"{split}.jsonl")}
            gold = gold_cache[key]
            check(f"{d.name}/{split}: gold labels match processed data", all(gold[p["id"]] == p["y"] for p in preds))
            y = np.array([p["y"] for p in preds])
            P = np.array([p["prob"] for p in preds])
            mm = classification_metrics(y, P, TASKS[task]["labels"])
            ok = abs(mm["macro_f1"] - m[split]["macro_f1"]) < 1e-9 and abs(mm["accuracy"] - m[split]["accuracy"]) < 1e-9
            check(f"{d.name}/{split}: metrics recomputed from predictions", ok,
                  f"{mm['macro_f1']:.6f} vs {m[split]['macro_f1']:.6f}")
        if cfg.get("kind") == "transformer":
            env = load_json(d / "environment.json")
            check(f"{d.name}: device recorded", env["device"]["selected"] == m["device"],
                  f"{env['device']['selected']} / {m['device']}")
    rep["n_runs"] = n_runs

    for png in FIG_DIR.rglob("*.png"):
        if png.parent.name == "diagrams":
            continue
        has = any((png.with_suffix(s)).exists() for s in (".csv", ".json"))
        check(f"figure data present: {png.relative_to(ROOT)}", has)

    vu = PAPER_DIR / "generated" / "values_used.json"
    an = RESULTS_DIR / "analysis" / "analysis.json"
    if vu.exists() and an.exists():
        check("paper rendered after latest analysis", vu.stat().st_mtime >= an.stat().st_mtime,
              "re-run scripts/build_paper.py")
        md = (PAPER_DIR / "SACENet_Research_Paper_Draft.md").read_text(encoding="utf-8")
        rep["paper_pending_markers"] = md.count("[PENDING")
        check("paper has no PENDING markers", md.count("[PENDING") == 0, f"{md.count('[PENDING')} remaining")
        # spot-check: MuRIL MC7 macro-F1 in paper equals analysis
        a = load_json(an)
        if "mc7|muril|base" in a.get("mc7_test", {}):
            val = f"{a['mc7_test']['mc7|muril|base']['seed_mean']['macro_f1']['mean']:.3f}"
            check("paper contains MuRIL MC7 macro-F1 from analysis", val in md, val)
    save_json(rep, RESULTS_DIR / "verification_report.json")
    n_ok = sum(c["ok"] for c in rep["checks"])
    print(f"{n_ok}/{len(rep['checks'])} checks passed over {n_runs} runs")
    for f in rep["failures"]:
        print("  FAIL:", f)
    return 0 if not rep["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
