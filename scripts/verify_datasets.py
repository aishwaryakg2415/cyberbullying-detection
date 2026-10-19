"""Verify raw files against data/dataset_manifest.json (hashes, row counts, labels).

Usage:  python scripts/verify_datasets.py [--write-manifest]

--write-manifest regenerates data/dataset_manifest.json from the files on disk
(use once after the first download; afterwards run without the flag to verify).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import loaders  # noqa: E402
from src.data.sources import SOURCES, UNAVAILABLE  # noqa: E402
from src.utils.io import DATA_DIR, RAW_DIR, load_json, save_json, sha256_file  # noqa: E402

MANIFEST = DATA_DIR / "dataset_manifest.json"


def count_records(key: str) -> dict:
    d = RAW_DIR / key
    if key == "mc_hinglish":
        recs = loaders.load_mc_hinglish(d)
        return {"records": len(recs), "by_split": dict(Counter(r.orig_split for r in recs)),
                "labels": dict(Counter(r.cb_type for r in recs))}
    if key == "swami_sarcasm":
        recs = loaders.load_swami_sarcasm(d)
        return {"records": len(recs), "labels": dict(Counter(r.label_orig for r in recs))}
    if key == "conda":
        recs = loaders.load_conda(d)
        return {"records": len(recs), "by_split": dict(Counter(r.orig_split for r in recs)),
                "labels": dict(Counter(r.toxic_intent for r in recs)),
                "conversations": len({r.conversation_id for r in recs})}
    if key == "tweeteval_offensive":
        recs = loaders.load_tweeteval_offensive(d)
        return {"records": len(recs), "labels": dict(Counter(r.label_orig for r in recs))}
    if key == "user_sarcasm_human":
        p = d / "human_labelled_features.csv"
        if not p.exists():
            return {"records": None}
        recs = loaders.load_user_sarcasm_human(p)
        return {"records": len(recs), "labels": dict(Counter(r.sarcasm for r in recs))}
    if key == "user_sarcasm_ai":
        p = d / "ai_labelled_features.csv"
        if not p.exists():
            return {"records": None}
        import csv
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        return {"records": len(rows), "labels": dict(Counter(r["ai_labelled"] for r in rows))}
    return {}


def _strkeys(d):
    """JSON object keys are strings; normalise so freshly built counts compare equal to the manifest."""
    if isinstance(d, dict):
        return {str(k): _strkeys(v) for k, v in d.items()}
    return d


def build() -> dict:
    man = {"generated_by": "scripts/verify_datasets.py --write-manifest", "datasets": {}, "unavailable": UNAVAILABLE}
    for key, src in SOURCES.items():
        files = {}
        for fname in src["files"]:
            p = RAW_DIR / key / fname
            files[fname] = {"sha256": sha256_file(p), "bytes": p.stat().st_size} if p.exists() else None
        entry = {k: v for k, v in src.items() if k != "files"}
        entry["urls"] = src["files"]
        entry["files"] = files
        entry["counts"] = _strkeys(count_records(key)) if any(files.values()) else None
        man["datasets"][key] = entry
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    current = build()
    if args.write_manifest or not MANIFEST.exists():
        save_json(current, MANIFEST)
        print(f"Manifest written: {MANIFEST}")
        return 0
    ref = load_json(MANIFEST)
    ok = True
    for key, entry in ref["datasets"].items():
        cur = current["datasets"].get(key, {})
        for fname, meta in (entry.get("files") or {}).items():
            got = (cur.get("files") or {}).get(fname)
            if meta is None:
                print(f"[n/a ] {key}/{fname}: not part of reference manifest")
            elif got is None:
                print(f"[MISS] {key}/{fname}")
                ok = False
            elif got["sha256"] != meta["sha256"]:
                print(f"[HASH] {key}/{fname}: {got['sha256'][:12]} != {meta['sha256'][:12]}")
                ok = False
            else:
                print(f"[ok  ] {key}/{fname} sha256={meta['sha256'][:12]}")
        if entry.get("counts") and cur.get("counts") and entry["counts"] != cur["counts"]:
            print(f"[CNT ] {key}: counts differ from manifest")
            ok = False
        elif entry.get("counts"):
            print(f"[ok  ] {key}: records={entry['counts'].get('records')}")
    print("VERIFIED" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
