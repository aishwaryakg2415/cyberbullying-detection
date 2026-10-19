"""Download pretrained encoders at the revisions pinned in configs/experiments.yaml.

Usage:  python scripts/download_models.py [--models muril xlmr ...]
After this, experiments run with HF_HUB_OFFLINE=1 so no other weights are fetched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
# Only the weight format each checkpoint ships at its pinned revision.
PATTERNS = ["*.json", "*.txt", "*.model", "merges.txt", "sentencepiece.bpe.model"]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(cfg["models"]))
    args = ap.parse_args()
    for key in args.models:
        m = cfg["models"][key]
        weights = ["model.safetensors"] if key in ("xlmr", "distilroberta") else ["pytorch_model.bin"]
        p = snapshot_download(m["name"], revision=m["revision"], allow_patterns=PATTERNS + weights)
        print(f"[ok] {key}: {m['name']}@{m['revision'][:8]} -> {p}")


if __name__ == "__main__":
    main()
