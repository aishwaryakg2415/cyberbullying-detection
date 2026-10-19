"""Reproducible, stratified, group-aware splitting."""
from __future__ import annotations

import numpy as np


def stratified_group_split(labels: list, groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask selecting ~`frac` of items for the held-out side.

    Whole groups (e.g. near-duplicate clusters or conversations) are assigned to
    one side only. Stratification is by the label of each group's first member,
    which is exact for singleton groups (the vast majority here).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels)
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(np.zeros(len(labels)), labels, groups))
    mask = np.zeros(len(labels), dtype=bool)
    mask[held] = True
    return mask


def group_split_by_key(keys: list[str], fracs: dict[str, float], seed: int) -> dict[str, str]:
    """Assign each distinct key (e.g. match id) to a split by seeded shuffle."""
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    out, start = {}, 0
    names = list(fracs)
    for i, name in enumerate(names):
        end = len(uniq) if i == len(names) - 1 else start + int(round(fracs[name] * len(uniq)))
        for k in uniq[start:end]:
            out[k] = name
        start = end
    return out
