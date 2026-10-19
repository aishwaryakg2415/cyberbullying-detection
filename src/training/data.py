"""Task definitions and tokenised datasets for transformer training."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..data.schema import MC_LABELS
from ..utils.io import PROCESSED_DIR, load_jsonl

TASKS = {
    "mc7": {"field": "cb_type", "labels": MC_LABELS},
    "mc2": {"field": "cb_binary", "labels": ["not_cyberbullying", "cyberbullying"]},
    "sarcasm": {"field": "sarcasm", "labels": ["non_sarcastic", "sarcastic"]},
    "usr_sarcasm": {"field": "sarcasm", "labels": ["non_sarcastic", "sarcastic"]},
    "olid_ext": {"field": "offensive", "labels": ["not_offensive", "offensive"]},
    # CONDA turn-level toxicity: E(xplicit) or I(mplicit) = toxic; A(ction)/O(ther) = non-toxic.
    "conda": {"field": "toxic_intent", "labels": ["non_toxic", "toxic"]},
    "conda_ctx": {"field": "toxic_intent", "labels": ["non_toxic", "toxic"]},   # + causal context
}


def encode_label(task: str, value) -> int:
    spec = TASKS[task]
    if spec["field"] == "cb_type":
        return spec["labels"].index(value)
    if spec["field"] == "toxic_intent":
        return int(value in ("E", "I"))
    return int(value)


def load_split(task: str, split: str) -> tuple[list[str], list[str], np.ndarray, list[dict]]:
    rows = load_jsonl(PROCESSED_DIR / task / f"{split}.jsonl")
    field = TASKS[task]["field"]
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    y = np.array([encode_label(task, r[field]) for r in rows], dtype=int)
    return ids, texts, y, rows


@dataclass
class Encoded:
    ids: list[str]
    input_ids: list[list[int]]
    labels: np.ndarray
    fusion: Optional[np.ndarray] = None


def encode(tokenizer, ids, texts, labels, max_len: int, fusion: Optional[np.ndarray] = None) -> Encoded:
    enc = tokenizer(texts, truncation=True, max_length=max_len, add_special_tokens=True)
    return Encoded(ids=list(ids), input_ids=enc["input_ids"], labels=np.asarray(labels), fusion=fusion)


def batches(data: Encoded, batch_size: int, pad_id: int, shuffle: bool, seed: int = 0,
            sort_by_length: bool = False):
    idx = list(range(len(data.ids)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    elif sort_by_length:
        idx.sort(key=lambda i: len(data.input_ids[i]))
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        L = max(len(data.input_ids[i]) for i in b)
        ii = torch.full((len(b), L), pad_id, dtype=torch.long)
        am = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            seq = data.input_ids[i]
            ii[r, :len(seq)] = torch.tensor(seq)
            am[r, :len(seq)] = 1
        out = {"input_ids": ii, "attention_mask": am, "labels": torch.tensor(data.labels[b]), "index": b}
        if data.fusion is not None:
            out["fusion"] = torch.tensor(data.fusion[b], dtype=torch.float32)
        yield out


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights from the TRAINING split, normalised to mean 1."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)
