"""Build processed task files, split ID lists and data-quality statistics.

Tasks produced (data/processed/<task>/{train,val,test}.jsonl):
  mc7          MC-Hinglish1.0 7-way cyberbullying type (official test kept intact)
  mc2          same records, derived binary label (cyberbullying vs not)
  sarcasm      Swami et al. (2018) sarcasm, stratified group-aware 70/10/20
  conda        CONDA utterances with conversation ids, grouped by match (proxy)
  olid_ext     TweetEval-offensive test (English external stress test)
  usr_sarcasm  user-supplied human sarcasm labels (supplementary external test)

Validation for MC-Hinglish is carved from the official training split only.
Near-duplicate clusters (char-n-gram cosine >= 0.9) are kept on one side of
the train/val boundary. Oversampling is never applied here; class weighting
is computed from the training split only, inside the trainer.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ..utils.io import PROCESSED_DIR, RAW_DIR, SPLITS_DIR, save_json, save_jsonl, sha256_text
from . import loaders
from .quality import (connected_groups, exact_duplicate_groups, label_conflicts, leakage_report,
                      missing_report, near_duplicate_pairs)
from .schema import Record
from .splits import group_split_by_key, stratified_group_split
from .text import normalised_key

logger = logging.getLogger(__name__)

SPLIT_SEED = 20261017
NEAR_DUP = 0.9


def _dist(records: list[Record], field: str) -> dict:
    return dict(sorted(Counter(getattr(r, field) for r in records).items(), key=lambda kv: str(kv[0])))


def _write_task(task: str, splits: dict[str, list[Record]], label_field: str) -> dict:
    tdir = PROCESSED_DIR / task
    info = {"task": task, "label_field": label_field, "splits": {}}
    for name, recs in splits.items():
        save_jsonl([r.to_dict() for r in recs], tdir / f"{name}.jsonl")
        ids = [r.id for r in recs]
        info["splits"][name] = {
            "n": len(recs), "label_dist": _dist(recs, label_field), "script_dist": _dist(recs, "script"),
            "ids_sha256": sha256_text("\n".join(ids)),
        }
        save_json(ids, SPLITS_DIR / task / f"{name}_ids.json")
    save_json(info, tdir / "task_info.json")
    return info


def _dup_summary(records: list[Record], field: str) -> dict:
    exact = exact_duplicate_groups(records, "raw")
    norm = exact_duplicate_groups(records, "norm")
    near = near_duplicate_pairs([r.text_raw for r in records], None, NEAR_DUP)
    return {
        "exact_dup_groups": len(exact), "exact_dup_rows_extra": sum(len(g) - 1 for g in exact),
        "normalised_dup_groups": len(norm), "normalised_dup_rows_extra": sum(len(g) - 1 for g in norm),
        f"near_dup_pairs@{NEAR_DUP}": len(near),
        "exact_dup_label_conflicts": label_conflicts(records, exact, field)[:50],
        "n_exact_dup_label_conflicts": len(label_conflicts(records, exact, field)),
        "_near_pairs": near,
    }


def prepare_mc(raw: Path) -> dict:
    recs = loaders.load_mc_hinglish(raw / "mc_hinglish")
    train = [r for r in recs if r.orig_split == "train"]
    test = [r for r in recs if r.orig_split == "test"]
    q = {"missing": missing_report(recs)}
    dtrain = _dup_summary(train, "cb_type")
    near = dtrain.pop("_near_pairs")
    q["train_duplicates"] = dtrain
    dtest = _dup_summary(test, "cb_type")
    dtest.pop("_near_pairs")
    q["test_duplicates"] = dtest

    # Group = connected component of normalised-exact OR near-duplicate pairs.
    pairs = [(i, j) for i, j, _ in near]
    for g in exact_duplicate_groups(train, "norm"):
        pairs += [(g[0], k) for k in g[1:]]
    groups = connected_groups(len(train), pairs)
    val_mask = stratified_group_split([r.cb_type for r in train], groups, 0.1, SPLIT_SEED)
    tr = [r for r, m in zip(train, val_mask) if not m]
    va = [r for r, m in zip(train, val_mask) if m]
    for r in tr:
        r.split = "train"
    for r in va:
        r.split = "val"
    for r in test:
        r.split = "test"

    # Official test is preserved. We additionally flag test items that have a
    # near-duplicate in train, so results can be reported on a de-duplicated subset.
    cross = near_duplicate_pairs([r.text_raw for r in train], [r.text_raw for r in test], NEAR_DUP)
    leaked_test = {j for _, j, _ in cross}
    norm_train = {normalised_key(r.text_raw) for r in train}
    for j, r in enumerate(test):
        r.meta["near_dup_in_train"] = bool(j in leaked_test or normalised_key(r.text_raw) in norm_train)
    q["leakage_after_split"] = leakage_report({"train": tr, "val": va, "test": test}, NEAR_DUP)
    q["test_items_flagged_near_dup_in_official_train"] = sum(r.meta["near_dup_in_train"] for r in test)

    info7 = _write_task("mc7", {"train": tr, "val": va, "test": test}, "cb_type")
    info2 = _write_task("mc2", {"train": tr, "val": va, "test": test}, "cb_binary")
    return {"quality": q, "mc7": info7, "mc2": info2,
            "length_words": _length_stats(recs), "script": _dist(recs, "script")}


def prepare_sarcasm(raw: Path) -> dict:
    recs = loaders.load_swami_sarcasm(raw / "swami_sarcasm")
    d = _dup_summary(recs, "sarcasm")
    near = d.pop("_near_pairs")
    pairs = [(i, j) for i, j, _ in near]
    for g in exact_duplicate_groups(recs, "norm"):
        pairs += [(g[0], k) for k in g[1:]]
    groups = connected_groups(len(recs), pairs)
    labels = [r.sarcasm for r in recs]
    test_mask = stratified_group_split(labels, groups, 0.2, SPLIT_SEED)
    rest_idx = np.where(~test_mask)[0]
    val_sub = stratified_group_split([labels[i] for i in rest_idx], groups[rest_idx], 0.125, SPLIT_SEED + 1)
    val_idx = set(rest_idx[val_sub].tolist())
    splits = {"train": [], "val": [], "test": []}
    for i, r in enumerate(recs):
        r.split = "test" if test_mask[i] else ("val" if i in val_idx else "train")
        splits[r.split].append(r)
    # Code-mixing statistics from the corpus' own token-level language tags.
    tags = loaders.swami_language_tags(raw / "swami_sarcasm")
    cmi = []
    for r in recs:
        t = tags.get(r.meta["tweet_id"], [])
        langs = [lt for _, lt in t if lt in ("en", "hi")]
        if langs:
            c = Counter(langs)
            cmi.append(100 * (1 - max(c.values()) / len(langs)))
    q = {"missing": missing_report(recs), "duplicates": d,
         "leakage_after_split": leakage_report(splits, NEAR_DUP),
         "cmi_mean": float(np.mean(cmi)) if cmi else None, "cmi_n": len(cmi),
         "token_lang_dist": dict(Counter(lt for v in tags.values() for _, lt in v))}
    info = _write_task("sarcasm", splits, "sarcasm")
    return {"quality": q, "sarcasm": info, "length_words": _length_stats(recs), "script": _dist(recs, "script")}


def prepare_conda(raw: Path) -> dict:
    recs = loaders.load_conda(raw / "conda")
    # Order utterances within each conversation by chat time then row id.
    by_conv: dict[str, list[Record]] = {}
    for r in recs:
        by_conv.setdefault(r.conversation_id, []).append(r)
    for conv in by_conv.values():
        conv.sort(key=lambda r: (r.timestamp if r.timestamp is not None else 0, r.meta["row_id"]))
        for k, r in enumerate(conv):
            r.turn_index = k
    # The official CONDA train/valid files were split at UTTERANCE level: 3,540
    # conversations and 1,772/1,778 valid-file matches also occur in the train
    # file (measured; see data/DATASET_CARD.md). That split is unusable for
    # conversation-level evaluation, so all annotated utterances are pooled and
    # re-split by MATCH id (70/10/20). No match, hence no conversation, crosses splits.
    official_shared = len({r.conversation_id for r in recs if r.orig_split == "train"}
                          & {r.conversation_id for r in recs if r.orig_split == "valid"})
    match_split = group_split_by_key([r.meta["match_id"] for r in recs], {"train": 0.7, "val": 0.1, "test": 0.2},
                                     SPLIT_SEED)
    splits = {"train": [], "val": [], "test": []}
    for r in recs:
        r.split = match_split[r.meta["match_id"]]
        splits[r.split].append(r)
    test = splits["test"]
    overlap = {
        "official_files_shared_conversations": official_shared,
        "match_ids_train_test": len({r.meta["match_id"] for r in splits["train"]} & {r.meta["match_id"] for r in test}),
        "match_ids_val_test": len({r.meta["match_id"] for r in splits["val"]} & {r.meta["match_id"] for r in test}),
        "conv_ids_train_val": len({r.conversation_id for r in splits["train"]} & {r.conversation_id for r in splits["val"]}),
    }
    conv_stats = {}
    for name, rs in splits.items():
        convs: dict[str, list[Record]] = {}
        for r in rs:
            convs.setdefault(r.conversation_id, []).append(r)
        lens = [len(v) for v in convs.values()]
        pos = [v for v in convs.values() if any(x.meta["toxic"] for x in v)]
        conv_stats[name] = {"conversations": len(convs), "utterances": len(rs),
                            "mean_len": float(np.mean(lens)), "median_len": float(np.median(lens)),
                            "max_len": int(max(lens)), "toxic_conversations": len(pos)}
    info = _write_task("conda", splits, "toxic_intent")
    # Causal-context variant: each turn is prefixed by the previous (at most) two
    # turns of the SAME conversation. Never uses future turns.
    import copy
    ctx_splits = {}
    for name, rs in splits.items():
        convs: dict[str, list[Record]] = {}
        for r in rs:
            convs.setdefault(r.conversation_id, []).append(r)
        out = []
        for conv in convs.values():
            conv.sort(key=lambda r: r.turn_index)
            for k, r in enumerate(conv):
                c = copy.deepcopy(r)
                prev = [p.text for p in conv[max(0, k - 2):k]]
                c.text = (" | ".join(prev) + " || " + r.text) if prev else r.text
                c.meta["context_turns"] = len(prev)
                out.append(c)
        ctx_splits[name] = out
    info_ctx = _write_task("conda_ctx", ctx_splits, "toxic_intent")
    return {"quality": {"missing": missing_report(recs), "split_overlap": overlap}, "conda": info,
            "conda_ctx": info_ctx, "conversations": conv_stats}


def prepare_external(raw: Path) -> dict:
    out = {}
    olid = loaders.load_tweeteval_offensive(raw / "tweeteval_offensive")
    for r in olid:
        r.split = "test"
    out["olid_ext"] = _write_task("olid_ext", {"test": olid}, "offensive")
    p = raw / "user_sarcasm_human" / "human_labelled_features.csv"
    if p.exists():
        usr = loaders.load_user_sarcasm_human(p)
        d = _dup_summary(usr, "sarcasm")
        d.pop("_near_pairs")
        # De-duplicate by normalised text (keep first) and drop items whose
        # labels conflict across duplicates, so the external test is clean.
        seen, keep = {}, []
        for r in usr:
            seen.setdefault(normalised_key(r.text_raw), []).append(r)
        for k, grp in seen.items():
            if len({g.sarcasm for g in grp}) == 1 and k:
                keep.append(grp[0])
        for r in keep:
            r.split = "test"
        ai = Counter((r.sarcasm, (r.meta.get("ai_label") or "").strip().lower()) for r in usr)
        out["usr_sarcasm"] = _write_task("usr_sarcasm", {"test": keep}, "sarcasm")
        out["usr_sarcasm"]["quality"] = {"raw_rows_with_valid_human_label": len(usr), "duplicates": d,
                                         "human_vs_ai_label_counts": {f"{a}|{b}": c for (a, b), c in ai.items()}}
    return out


def _length_stats(recs: list[Record]) -> dict:
    n = np.array([len(r.text_raw.split()) for r in recs])
    return {"mean": float(n.mean()), "median": float(np.median(n)), "p95": float(np.percentile(n, 95)),
            "max": int(n.max()), "min": int(n.min())}


def prepare_all(raw: Path = RAW_DIR) -> dict:
    stats = {"split_seed": SPLIT_SEED, "near_dup_threshold": NEAR_DUP}
    stats["mc_hinglish"] = prepare_mc(raw)
    stats["swami_sarcasm"] = prepare_sarcasm(raw)
    stats["conda"] = prepare_conda(raw)
    stats["external"] = prepare_external(raw)
    save_json(stats, PROCESSED_DIR / "data_statistics.json")
    return stats
