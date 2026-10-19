"""Conversation-level early-detection evaluation (causal, prefix-based).

Setting. A conversation is a sequence of turns u_1..u_n with turn-level labels.
It is *positive* if any turn is toxic; its onset o is the index of the first
toxic turn. A detector emits a score p_t after observing turn t using ONLY
u_1..u_t (no future turns). The alarm policy raises an alarm at the first t
where the aggregated score a_t >= tau; the decision is then frozen.

Metrics.
  * conversation-level precision / recall / F1 of the alarm
  * false-alarm rate on negative conversations
  * detection delay  d = t_alarm - o   (turns; negative => alarm before onset)
  * ERDE_o (Losada & Crestani, 2016): FP cost = positive rate, FN cost = 1,
    TP cost = 1 - 1/(1+exp(k-o)) with k = #turns read at alarm time
  * prefix curves: decision after reading only the first k turns, scored
    against (a) the *eventual* conversation label and (b) toxicity within the prefix.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Conversation:
    cid: str
    turn_ids: list[str]
    labels: np.ndarray      # 1 = toxic turn
    scores: np.ndarray      # causal turn scores p_t in [0,1]

    @property
    def positive(self) -> bool:
        return bool(self.labels.any())

    @property
    def onset(self) -> int | None:
        idx = np.flatnonzero(self.labels)
        return int(idx[0]) if len(idx) else None


def aggregate(scores: np.ndarray, policy: str = "max", alpha: float = 0.5) -> np.ndarray:
    """Causal aggregation: running max, or exponential moving average."""
    if policy == "max":
        return np.maximum.accumulate(scores)
    if policy == "ema":
        out, a = np.empty_like(scores), 0.0
        for t, s in enumerate(scores):
            a = s if t == 0 else alpha * s + (1 - alpha) * a
            out[t] = a
        return out
    if policy == "last":
        return scores.copy()
    raise ValueError(policy)


def alarm_index(conv: Conversation, tau: float, policy: str = "max") -> int | None:
    a = aggregate(conv.scores, policy)
    hit = np.flatnonzero(a >= tau)
    return int(hit[0]) if len(hit) else None


def evaluate_alarms(convs: list[Conversation], tau: float, policy: str = "max", erde_o=(1, 3, 5)) -> dict:
    tp = fp = fn = tn = 0
    delays, erde = [], {o: [] for o in erde_o}
    pos_rate = np.mean([c.positive for c in convs])
    pre_onset = 0
    for c in convs:
        t = alarm_index(c, tau, policy)
        if c.positive:
            if t is None:
                fn += 1
                for o in erde_o:
                    erde[o].append(1.0)
            else:
                tp += 1
                d = t - c.onset
                delays.append(d)
                pre_onset += int(d < 0)
                k = t + 1
                for o in erde_o:
                    erde[o].append(1 - 1 / (1 + math.exp(min(k - o, 50))))
        else:
            if t is None:
                tn += 1
                for o in erde_o:
                    erde[o].append(0.0)
            else:
                fp += 1
                for o in erde_o:
                    erde[o].append(float(pos_rate))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    d = np.array(delays) if delays else np.array([np.nan])
    return {
        "tau": tau, "policy": policy, "n_conversations": len(convs), "n_positive": tp + fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "delay_mean": float(np.nanmean(d)), "delay_median": float(np.nanmedian(d)),
        "delay_dist": {int(k): int(v) for k, v in zip(*np.unique(d[~np.isnan(d)], return_counts=True))} if delays else {},
        "alarms_before_onset": pre_onset,
        **{f"erde_{o}": float(np.mean(erde[o])) for o in erde_o},
    }


def prefix_curve(convs: list[Conversation], tau: float, policy: str = "max", max_k: int = 10) -> list[dict]:
    """Decision after only k turns, vs eventual label and vs within-prefix label."""
    rows = []
    for k in range(1, max_k + 1):
        y_ev, y_pre, yhat = [], [], []
        for c in convs:
            pre = c.scores[:k]
            yhat.append(int(aggregate(pre, policy)[-1] >= tau))
            y_ev.append(int(c.positive))
            y_pre.append(int(c.labels[:k].any()))
        rows.append({"k": k, **_prf(np.array(y_ev), np.array(yhat), "eventual"),
                     **_prf(np.array(y_pre), np.array(yhat), "prefix"),
                     "n_conv_longer_than_k": int(sum(len(c.labels) > k for c in convs))})
    return rows


def _prf(y, p, tag):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {f"{tag}_precision": pr, f"{tag}_recall": rc, f"{tag}_f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}


def choose_tau(convs: list[Conversation], policy: str = "max", max_far: float | None = None) -> float:
    """Threshold on VALIDATION conversations: maximise F1 (optionally subject to FAR <= max_far)."""
    best, best_tau = -1, 0.5
    for tau in np.linspace(0.05, 0.95, 91):
        r = evaluate_alarms(convs, float(tau), policy)
        if max_far is not None and r["false_alarm_rate"] > max_far:
            continue
        if r["f1"] > best:
            best, best_tau = r["f1"], float(tau)
    return best_tau


def build_conversations(rows: list[dict], scores: dict[str, float]) -> list[Conversation]:
    by = defaultdict(list)
    for r in rows:
        by[r["conversation_id"]].append(r)
    out = []
    for cid, rs in by.items():
        rs.sort(key=lambda r: r["turn_index"])
        out.append(Conversation(cid, [r["id"] for r in rs], np.array([r["meta"]["toxic"] for r in rs]),
                                np.array([scores[r["id"]] for r in rs], dtype=float)))
    return out
