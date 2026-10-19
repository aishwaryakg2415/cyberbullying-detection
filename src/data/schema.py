"""Unified record schema.

Every dataset is converted into `Record`s. Concepts that differ across
datasets (cyberbullying category, sarcasm, offensiveness, in-game toxicity)
live in *separate* fields and are never merged into a single label column.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Record:
    id: str                       # stable, dataset-prefixed identifier
    text: str                     # cleaned model input
    text_raw: str                 # original text as distributed
    source: str                   # dataset key (see dataset_manifest.json)
    language: str                 # declared language of the dataset (e.g. "hi-en")
    script: str                   # detected script: roman / devanagari / mixed / other
    label_orig: Optional[str]     # original label string, unmodified
    orig_split: Optional[str] = None
    split: Optional[str] = None   # train / val / test after our split procedure
    # Task-specific labels: only populated when genuinely annotated.
    cb_type: Optional[str] = None       # MC-Hinglish 7-way cyberbullying type
    cb_binary: Optional[int] = None     # derived: 1 if any cyberbullying type
    sarcasm: Optional[int] = None       # human sarcasm annotation (1/0)
    offensive: Optional[int] = None     # OLID/TweetEval offensive (1/0)
    toxic_intent: Optional[str] = None  # CONDA utterance intent (E/I/A/O)
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None
    timestamp: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# MC-Hinglish1.0 labels exactly as distributed.
MC_LABELS = ["not_cyberbullying", "age", "gender", "religion", "mockery", "abusive", "offensive"]
# Binary mapping. Justification: in MC-Hinglish1.0 every label other than
# `not_cyberbullying` is defined by the dataset authors as a *type* of
# cyberbullying (Rahman-Laskar et al., 2024), so the union is "cyberbullying".
MC_TO_BINARY = {lab: int(lab != "not_cyberbullying") for lab in MC_LABELS}

CONDA_TOXIC = {"E", "I"}   # Explicit / Implicit toxicity (Weld et al., 2021)
