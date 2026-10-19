"""Load a saved SACENet checkpoint and run batched inference."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ..data.text import clean_text
from ..evaluation.metrics import softmax
from ..models.transformer import SACENetClassifier
from ..utils.device import select_device
from ..utils.io import load_json

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class Predictor:
    """`predict_proba(texts)` -> (N, K) probabilities from the trained main head.

    `calibrated=True` applies the temperature fitted on the validation split.
    Fusion checkpoints are not supported here (they need sarcasm-model features).
    """

    def __init__(self, model_dir: str | Path, device: str = "auto", calibrated: bool = True):
        model_dir = Path(model_dir)
        self.cfg = load_json(model_dir / "sacenet_config.json")
        if self.cfg.get("fusion_dim"):
            raise ValueError("Predictor does not support fusion checkpoints")
        self.device = select_device(device).selected
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = SACENetClassifier(self.cfg["model_name"], self.cfg["heads"], self.cfg["dropout"],
                                       revision=self.cfg["revision"])
        state = torch.load(model_dir / "weights.pt", map_location="cpu")
        self.model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()})
        self.model.to(self.device).eval()
        self.labels = self.cfg["labels"]
        self.T = float(self.cfg.get("temperature", 1.0)) if calibrated else 1.0
        self.max_len = int(self.cfg["max_len"])

    @torch.no_grad()
    def logits(self, texts: list[str], batch_size: int = 32, clean: bool = True, task: str = "main") -> np.ndarray:
        texts = [clean_text(t) if clean else t for t in texts]
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[s:s + batch_size], truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            out.append(self.model(enc["input_ids"], enc["attention_mask"], task=task).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict_proba(self, texts: list[str], batch_size: int = 32, clean: bool = True) -> np.ndarray:
        return softmax(self.logits(texts, batch_size, clean), self.T)
