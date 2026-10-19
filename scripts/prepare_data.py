"""Clean, validate, de-duplicate and split all datasets.

Usage:  python scripts/prepare_data.py
Writes data/processed/<task>/*.jsonl, data/splits/<task>/*_ids.json and
data/processed/data_statistics.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare import prepare_all  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    stats = prepare_all()
    summary = {
        "mc7": stats["mc_hinglish"]["mc7"]["splits"],
        "sarcasm": stats["swami_sarcasm"]["sarcasm"]["splits"],
        "conda_conversations": stats["conda"]["conversations"],
        "mc_leakage": stats["mc_hinglish"]["quality"]["leakage_after_split"],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:6000])
