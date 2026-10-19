"""Download every openly available dataset from its official, pinned location.

Usage:
    python scripts/download_datasets.py                     # all public sources
    python scripts/download_datasets.py --user-dir ..       # also copy user-supplied sarcasm CSVs
    python scripts/download_datasets.py --only mc_hinglish

Nothing here requires Kaggle, credentials or API keys. Files are written to
data/raw/<source>/ and their SHA-256 hashes are recorded in
data/raw/download_log.json (consumed by verify_datasets.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sources import SOURCES  # noqa: E402
from src.utils.io import RAW_DIR, load_json, save_json, sha256_file  # noqa: E402

# Local file names the project owner used for the user-supplied files.
USER_FILES = {
    "user_sarcasm_human": ["human_labelled_features.csv"],
    "user_sarcasm_ai": ["ai_labelled_features.csv"],
}
# The MC-Hinglish files can also be copied from a local download (identical bytes).
LOCAL_ALIASES = {
    ("mc_hinglish", "train.csv"): ["Train_data_MC-Hinglish1.0.csv", "Train_data_MC-Hinglish1.0 (1).csv"],
    ("mc_hinglish", "test.csv"): ["Test_data_MC-Hinglish1.0.csv", "Test_data_MC-Hinglish1.0 (1).csv"],
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "sacenet-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of source keys")
    ap.add_argument("--user-dir", type=Path, default=None,
                    help="directory holding user-supplied CSVs (human/ai_labelled_features.csv)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    log_path = RAW_DIR / "download_log.json"
    log = load_json(log_path) if log_path.exists() else {}
    status = 0
    for key, src in SOURCES.items():
        if args.only and key not in args.only:
            continue
        entry = log.setdefault(key, {"files": {}})
        for fname, url in src["files"].items():
            dest = RAW_DIR / key / fname
            if dest.exists() and not args.force:
                print(f"[skip] {key}/{fname} already present")
                entry["files"].setdefault(fname, {"status": "present"})
            elif url is None:
                cands = USER_FILES.get(key, [])
                found = None
                if args.user_dir:
                    found = next((args.user_dir / c for c in cands if (args.user_dir / c).exists()), None)
                if not found:
                    print(f"[manual] {key}/{fname}: not downloadable; pass --user-dir with {cands}")
                    entry["files"][fname] = {"status": "missing (user-supplied, not public)"}
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found, dest)
                entry["files"][fname] = {"status": "copied", "from": str(found)}
                print(f"[copy] {found} -> {dest}")
            else:
                try:
                    fetch(url, dest)
                    entry["files"][fname] = {"status": "downloaded", "url": url}
                    print(f"[ok]   {key}/{fname}")
                except Exception as e:  # network or HTTP error: try local alias, else report
                    alias = None
                    if args.user_dir:
                        alias = next((args.user_dir / c for c in LOCAL_ALIASES.get((key, fname), [])
                                      if (args.user_dir / c).exists()), None)
                    if alias:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(alias, dest)
                        entry["files"][fname] = {"status": "copied (download failed)", "from": str(alias),
                                                 "error": str(e)}
                        print(f"[copy] {alias} -> {dest} (download failed: {e})")
                    else:
                        entry["files"][fname] = {"status": "failed", "url": url, "error": str(e)}
                        print(f"[FAIL] {key}/{fname}: {e}")
                        status = 1
                        continue
            if dest.exists():
                entry["files"][fname].update({
                    "sha256": sha256_file(dest), "bytes": dest.stat().st_size,
                    "retrieved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                })
        entry["revision"] = src.get("revision")
    save_json(log, log_path)
    print(f"Download log -> {log_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
