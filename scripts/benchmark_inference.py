"""Inference latency / throughput / size for every model family (run when the GPU is idle).

Latency does not depend on the fine-tuned weight values, only on the architecture,
so each encoder is instantiated with its classification head and timed on real
MC-Hinglish test comments. Classical baselines use the numbers measured in their runs.

Usage:  python scripts/benchmark_inference.py
Output: results/benchmarks/inference.json
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.evaluation.aggregate import collect_runs, pretty  # noqa: E402
from src.models.transformer import SACENetClassifier  # noqa: E402
from src.utils.env import hardware_summary, package_versions  # noqa: E402
from src.utils.io import PROCESSED_DIR, RESULTS_DIR, ROOT, load_jsonl, save_json  # noqa: E402


def time_model(model, tok, texts, device, bs1_n=60, bs32_batches=6):
    model.to(device).eval()

    def run(batch):
        enc = tok(batch, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            model(enc["input_ids"], enc["attention_mask"])
        if device == "mps":
            torch.mps.synchronize()

    for t in texts[:5]:
        run([t])   # warm-up
    lat = []
    for t in texts[:bs1_n]:
        s = time.perf_counter()
        run([t])
        lat.append((time.perf_counter() - s) * 1000)
    s = time.perf_counter()
    n = 0
    for b in range(bs32_batches):
        batch = texts[b * 32:(b + 1) * 32]
        run(batch)
        n += len(batch)
    thr = n / (time.perf_counter() - s)
    return float(np.median(lat)), float(np.percentile(lat, 95)), float(thr)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
    texts = [r["text"] for r in load_jsonl(PROCESSED_DIR / "mc7" / "test.jsonl")]
    groups = collect_runs()
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    models = []
    for key in ("muril", "xlmr", "hingroberta", "distilroberta"):
        m = cfg["models"][key]
        tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
        model = SACENetClassifier(m["name"], {"main": 7}, revision=m["revision"], freeze_word_embeddings=True)
        n = model.n_parameters()
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row = {"name": {"muril": "MuRIL", "xlmr": "XLM-R base", "hingroberta": "HingRoBERTa",
                        "distilroberta": "DistilRoBERTa"}[key], "params_m": n / 1e6, "trainable_params_m": tr / 1e6,
               "size_mb": n * 2 / 2**20, "size_note": "fp16 state dict"}
        for d in devices:
            med, p95, thr = time_model(model, tok, texts, d)
            row[f"latency_ms_bs1_{d}"] = med
            row[f"latency_p95_ms_bs1_{d}"] = p95
            row[f"throughput_per_s_bs32_{d}"] = thr
        tm = [r["metrics"]["train_seconds"] for gk, rs in groups.items() if f"|{key}|base" in gk and gk.startswith("mc7")
              for r in rs]
        if tm:
            row["train_minutes_mean"] = float(np.mean(tm) / 60)
        models.append(row)
        del model
    for gk in ("mc7|tfidf_lr", "mc7|tfidf_svm"):
        if gk in groups:
            m = groups[gk][0]["metrics"]
            models.append({"name": pretty(gk), "params_m": m["selected"].get("n_features", 0) * 7 / 1e6,
                           "trainable_params_m": None, "size_mb": m["model_bytes"] / 2**20,
                           "size_note": "pickled vectoriser + model",
                           "latency_ms_bs1_cpu": m["latency_ms_cpu_bs1"]["p50"],
                           "train_minutes_mean": m["train_seconds"] / 60})
    save_json({"hardware": hardware_summary(), "packages": package_versions(), "devices": devices,
               "models": models}, RESULTS_DIR / "benchmarks" / "inference.json")
    for r in models:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
