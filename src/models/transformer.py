"""Transformer classifiers used by SACENet.

One module covers the three architectural variants compared in the paper:

* single-task:   h = Enc(x)[CLS];            z = W_main h
* multitask:     shared encoder, two heads;  L = CE_main + lambda * CE_sarc
* feature-fusion: h' = [h ; phi(s)],         z = W_main h'
                  where s are features from a *separately trained* sarcasm
                  model (its P(sarcastic) and pooled embedding), precomputed
                  and frozen.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel


class SACENetClassifier(nn.Module):
    def __init__(self, encoder_name: str, heads: dict[str, int], dropout: float = 0.1,
                 fusion_dim: int = 0, fusion_proj: int = 64, revision: Optional[str] = None,
                 freeze_word_embeddings: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, revision=revision)
        if freeze_word_embeddings:
            # MuRIL's 197k-token (151M-param) and XLM-R's 250k-token input embedding tables
            # dominate optimiser cost on Apple MPS (0.74 s of a 0.92 s step, measured).
            # Freezing only the token-embedding table (position/type embeddings and all
            # transformer layers stay trainable) is applied identically to every model.
            self.encoder.get_input_embeddings().weight.requires_grad_(False)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fusion_dim = fusion_dim
        if fusion_dim:
            # Small projection so a 769-d sarcasm feature cannot swamp the 768-d text vector.
            self.fusion = nn.Sequential(nn.Linear(fusion_dim, fusion_proj), nn.GELU(), nn.Dropout(dropout))
            in_dim = hid + fusion_proj
        else:
            self.fusion = None
            in_dim = hid
        self.heads = nn.ModuleDict({
            t: nn.Linear(in_dim if t == "main" else hid, n) for t, n in heads.items()
        })

    def pooled(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, task: str = "main",
                fusion_feats: Optional[torch.Tensor] = None, return_pooled: bool = False):
        h = self.pooled(input_ids, attention_mask)
        z_in = self.dropout(h)
        if task == "main" and self.fusion is not None:
            if fusion_feats is None:
                raise ValueError("fusion model requires fusion_feats")
            z_in = torch.cat([z_in, self.fusion(fusion_feats)], dim=-1)
        logits = self.heads[task](z_in)
        return (logits, h) if return_pooled else logits

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
