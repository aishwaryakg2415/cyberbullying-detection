"""Data-quality checks: missing values, duplicates, near-duplicates, leakage."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .schema import Record
from .text import normalised_key


def missing_report(records: list[Record]) -> dict:
    return {
        "n": len(records),
        "empty_text": sum(1 for r in records if not r.text.strip()),
        "missing_label": sum(1 for r in records if r.label_orig in (None, "")),
    }


def exact_duplicate_groups(records: list[Record], key: str = "raw") -> list[list[int]]:
    """Indices grouped by identical raw text (key='raw') or normalised key ('norm')."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        k = r.text_raw.strip() if key == "raw" else normalised_key(r.text_raw)
        if k:
            groups[k].append(i)
    return [g for g in groups.values() if len(g) > 1]


def near_duplicate_pairs(texts_a: list[str], texts_b: list[str] | None = None,
                         threshold: float = 0.9, max_pairs: int = 200000) -> list[tuple[int, int, float]]:
    """Character 3-5-gram TF-IDF cosine >= threshold.

    If `texts_b` is None, pairs within `texts_a` (i<j); otherwise cross pairs (i in a, j in b).
    Exact-cosine on sparse vectors; fine for corpora of ~10^4 short texts.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    a = [normalised_key(t) for t in texts_a]
    b = None if texts_b is None else [normalised_key(t) for t in texts_b]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    vec.fit(a + (b or []))
    A = vec.transform(a)
    B = A if b is None else vec.transform(b)
    out: list[tuple[int, int, float]] = []
    step = 512
    for s in range(0, A.shape[0], step):
        sims = (A[s:s + step] @ B.T).tocoo()
        for i, j, v in zip(sims.row, sims.col, sims.data):
            gi = s + i
            if v >= threshold and (b is not None or gi < j):
                out.append((int(gi), int(j), float(v)))
                if len(out) >= max_pairs:
                    return out
    return out


def connected_groups(n: int, pairs: Iterable[tuple[int, int]]) -> np.ndarray:
    """Union-find over pairs -> group id per index (singletons get their own id)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def label_conflicts(records: list[Record], groups: list[list[int]], field: str) -> list[dict]:
    """Duplicate groups whose members carry different labels for `field`."""
    out = []
    for g in groups:
        labs = Counter(getattr(records[i], field) for i in g)
        if len(labs) > 1:
            out.append({"ids": [records[i].id for i in g], "labels": dict(labs),
                        "text": records[g[0]].text_raw[:120]})
    return out


def leakage_report(splits: dict[str, list[Record]], near_threshold: float = 0.9) -> dict:
    """Exact / normalised / near-duplicate overlap between every pair of splits."""
    names = list(splits)
    rep = {}
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            A, B = splits[a], splits[b]
            raw_a = {r.text_raw.strip() for r in A}
            norm_a = {normalised_key(r.text_raw) for r in A}
            exact = sum(1 for r in B if r.text_raw.strip() in raw_a)
            norm = sum(1 for r in B if normalised_key(r.text_raw) in norm_a)
            near = near_duplicate_pairs([r.text_raw for r in A], [r.text_raw for r in B], near_threshold)
            rep[f"{a}__{b}"] = {
                "exact": exact, "normalised": norm,
                f"near_dup_pairs@{near_threshold}": len(near),
                f"{b}_items_with_near_dup_in_{a}": len({j for _, j, _ in near}),
            }
    return rep
