"""Aggregate completed runs into analysis JSON (consumed by figures, tables and the paper).

Every number is recomputed from the saved per-item predictions, never copied
from logs. Outputs: results/analysis/*.json

Usage:  python scripts/analyze_results.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

from src.data.text import normalised_key  # noqa: E402
from src.evaluation.aggregate import (collect_runs, ensemble, labels_of, load_preds, pretty, seed_summary,  # noqa: E402
                                      test_meta)
from src.evaluation.metrics import (bootstrap_ci, classification_metrics, macro_f1, mcnemar_exact,  # noqa: E402
                                    paired_bootstrap)
from src.utils.io import PROCESSED_DIR, RESULTS_DIR, RUNS_DIR, load_json, load_jsonl, save_json  # noqa: E402

OUT = RESULTS_DIR / "analysis"
N_BOOT = 2000

# Small, documented lexicon of very common Hindi/Hinglish profanity spellings, used ONLY to
# slice the error analysis ("does the model over-trigger on profanity in non-bullying text?").
PROFANITY = re.compile(r"\b(bc|mc|bsdk|bkl|chutiya\w*|chutiye|madarchod\w*|behenchod\w*|bhenchod\w*|bhosd\w*|"
                       r"gaand\w*|gand|lodu|lund|randi\w*|harami\w*|kamina\w*|saala|sala|saali|fuck\w*|shit)\b", re.I)


def group_table(groups, task, split="test", subset=None):
    labels = labels_of(task)
    rows = {}
    for gk, runs in sorted(groups.items()):
        if not gk.startswith(task + "|"):
            continue
        s = seed_summary(runs, split, labels, subset)
        ids, y, P = ensemble(runs, split)
        if subset is not None:
            m = np.array([i in subset for i in ids])
            y, P = y[m], P[m]
        ens = classification_metrics(y, P, labels)
        ci = bootstrap_ci(y, P.argmax(1), macro_f1, n_boot=N_BOOT, seed=0)
        mets = [r["metrics"] for r in runs]
        rows[gk] = {"name": pretty(gk), "n_runs": len(runs), "seeds": [r["cfg"]["seed"] for r in runs],
                    "seed_mean": s, "ensemble": {k: v for k, v in ens.items() if k != "reliability"},
                    "ensemble_macro_f1_ci95": ci,
                    "train_seconds_mean": float(np.mean([m.get("train_seconds", 0) for m in mets])),
                    "best_epoch": [m.get("best_epoch") for m in mets],
                    "n_parameters": mets[0].get("n_parameters"),
                    "n_trainable_parameters": mets[0].get("n_trainable_parameters"),
                    "temperature": [m.get("temperature") for m in mets]}
        cal = [m.get(f"{split}_calibrated") for m in mets if m.get(f"{split}_calibrated")]
        if cal and subset is None:
            rows[gk]["calibrated_ece_mean"] = float(np.mean([c["ece"] for c in cal]))
            rows[gk]["calibrated_brier_mean"] = float(np.mean([c["brier"] for c in cal]))
    return rows


def paired(groups, a, b, split="test"):
    """b vs a: per-seed paired bootstrap (matched seeds) + ensemble paired bootstrap + McNemar."""
    if a not in groups or b not in groups:
        return None
    ra = {r["cfg"]["seed"]: r for r in groups[a]}
    rb = {r["cfg"]["seed"]: r for r in groups[b]}
    per_seed = []
    for s in sorted(set(ra) & set(rb)) or [None]:
        A = ra[s] if s is not None else groups[a][0]
        B = rb[s] if s is not None else groups[b][0]
        ida, y, Pa = load_preds(A["dir"], split)
        idb, _, Pb = load_preds(B["dir"], split)
        idx = {k: i for i, k in enumerate(idb)}
        Pb = Pb[[idx[k] for k in ida]]
        pb = paired_bootstrap(y, Pa.argmax(1), Pb.argmax(1), n_boot=N_BOOT, seed=s or 0)
        per_seed.append({"seed": s, **pb})
    ida, y, Pa = ensemble(groups[a], split)
    idb, _, Pb = ensemble(groups[b], split)
    idx = {k: i for i, k in enumerate(idb)}
    Pb = Pb[[idx[k] for k in ida]]
    return {"a": a, "b": b, "a_name": pretty(a), "b_name": pretty(b), "per_seed": per_seed,
            "ensemble_bootstrap": paired_bootstrap(y, Pa.argmax(1), Pb.argmax(1), n_boot=N_BOOT),
            "ensemble_mcnemar": mcnemar_exact(y, Pa.argmax(1), Pb.argmax(1))}


def legacy_boost(groups):
    """Original SACENet rule p' = min(1, p + 0.18 * s) if s >= 0.45; decision p' >= 0.55,
    re-evaluated with a TRAINED sarcasm model's s (features exported by the sarcasm run)."""
    gk = "mc2|muril|base"
    feat = RUNS_DIR / "sarcasm__muril__base__s13" / "features_mc7.npz"
    if gk not in groups or not feat.exists():
        return None
    z = np.load(feat)
    s_of = dict(zip(z["ids"].tolist(), z["prob"].tolist()))
    out = {"rule": "p' = min(1, p + 0.18*s) if s >= 0.45 else p; predict 1 if p' >= 0.55", "per_seed": []}
    for r in groups[gk]:
        ids, y, P = load_preds(r["dir"], "test")
        p = P[:, 1]
        s = np.array([s_of[i] for i in ids])
        fused = np.where(s >= 0.45, np.minimum(1, p + 0.18 * s), p)
        res = {"seed": r["cfg"]["seed"]}
        for name, pred in (("base@0.50", p >= 0.5), ("base@0.55", p >= 0.55), ("boost@0.55", fused >= 0.55)):
            res[name] = {"macro_f1": macro_f1(y, pred.astype(int)),
                         "recall_cb": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                         "fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))}
        res["n_boosted"] = int((s >= 0.45).sum())
        res["boost_vs_base055"] = paired_bootstrap(y, (p >= 0.55).astype(int), (fused >= 0.55).astype(int),
                                                   n_boot=N_BOOT, seed=r["cfg"]["seed"])
        out["per_seed"].append(res)
    out["sarcasm_prob_on_mc_test"] = {"mean": float(np.mean([s_of[i] for i in ids])),
                                      "share_ge_0.45": float(np.mean([s_of[i] >= 0.45 for i in ids]))}
    return out


def external(groups):
    out = {}
    # (a) Binary MC models on English OLID offensive test: P(cyberbullying) vs offensive label.
    for gk, runs in groups.items():
        if gk.startswith("mc2|") and (runs[0]["dir"] / "predictions_olid_ext.jsonl").exists():
            vals = []
            for r in runs:
                ids, y, P = load_preds(r["dir"], "olid_ext")
                m = classification_metrics(y, P, ["not_offensive", "offensive"])
                vals.append({k: m[k] for k in ("macro_f1", "accuracy", "roc_auc", "pr_auc", "f1_positive")})
            out[f"olid_ext::{gk}"] = {"name": pretty(gk), "n": len(ids), "seeds": vals,
                                      "macro_f1_mean": float(np.mean([v["macro_f1"] for v in vals])),
                                      "roc_auc_mean": float(np.mean([v["roc_auc"] for v in vals]))}
    # (b) Sarcasm models (Swami-trained) on the user-supplied human-labelled sarcasm file.
    for gk, runs in groups.items():
        if gk.startswith("sarcasm|") and (runs[0]["dir"] / "predictions_usr_sarcasm.jsonl").exists():
            vals = []
            for r in runs:
                ids, y, P = load_preds(r["dir"], "usr_sarcasm")
                m = classification_metrics(y, P, ["non_sarcastic", "sarcastic"])
                vals.append({k: m.get(k) for k in ("macro_f1", "accuracy", "roc_auc", "pr_auc", "f1_positive")})
            out[f"usr_sarcasm::{gk}"] = {"name": pretty(gk), "n": len(ids), "seeds": vals,
                                         "macro_f1_mean": float(np.mean([v["macro_f1"] for v in vals])),
                                         "roc_auc_mean": float(np.mean([v["roc_auc"] for v in vals]))}
    return out


def error_analysis(groups, train_vocab: set):
    gk = "mc7|muril|base"
    if gk not in groups:
        return None
    labels = labels_of("mc7")
    meta = test_meta("mc7")
    ids, y, P = ensemble(groups[gk], "test")
    pred = P.argmax(1)
    texts = [meta[i]["text_raw"] for i in ids]

    def eng_ratio(t):
        w = re.findall(r"[a-z]+", t.lower())
        return np.mean([x in ENGLISH_STOP_WORDS for x in w]) if w else 0.0

    def oov(t):
        w = normalised_key(t).split()
        return np.mean([x not in train_vocab for x in w]) if w else 0.0

    feats = {
        "near_dup_in_train": [bool(meta[i]["meta"].get("near_dup_in_train")) for i in ids],
        "profanity": [bool(PROFANITY.search(t)) for t in texts],
        "len_words": [len(t.split()) for t in texts],
        "english_function_word_ratio": [eng_ratio(t) for t in texts],
        "oov_rate": [oov(t) for t in texts],
    }
    sar = RUNS_DIR / "sarcasm__muril__base__s13" / "features_mc7.npz"
    if sar.exists():
        z = np.load(sar)
        s_of = dict(zip(z["ids"].tolist(), z["prob"].tolist()))
        feats["pred_sarcasm_prob"] = [s_of[i] for i in ids]

    def sub(mask, name):
        m = np.asarray(mask)
        if m.sum() < 10:
            return {"slice": name, "n": int(m.sum())}
        return {"slice": name, "n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
                "macro_f1": macro_f1(y[m], pred[m])}

    L = np.array(feats["len_words"])
    E = np.array(feats["english_function_word_ratio"])
    O = np.array(feats["oov_rate"])
    prof = np.array(feats["profanity"])
    nd = np.array(feats["near_dup_in_train"])
    not_cb = y == 0
    slices = [
        sub(~nd, "test without near-duplicate in train"), sub(nd, "test WITH near-duplicate in train"),
        sub(L <= 8, "short (<=8 words)"), sub((L > 8) & (L <= 20), "medium (9-20 words)"), sub(L > 20, "long (>20 words)"),
        sub(E < 0.05, "low English function-word ratio (<5%)"), sub(E >= 0.15, "high English function-word ratio (>=15%)"),
        sub(O == 0, "no OOV words vs train vocabulary"), sub(O >= 0.3, "OOV rate >= 30% (spelling/transliteration variation)"),
        sub(prof, "contains profanity"), sub(~prof, "no profanity"),
    ]
    if "pred_sarcasm_prob" in feats:
        S = np.array(feats["pred_sarcasm_prob"])
        slices += [sub(S >= 0.5, "sarcasm model P>=0.5 (model-inferred, not gold)"),
                   sub(S < 0.5, "sarcasm model P<0.5")]
    # False positives on harmless text, split by profanity presence ("friendly" swearing).
    fp = {"not_cb_total": int(not_cb.sum()),
          "not_cb_predicted_cb": int((not_cb & (pred != 0)).sum()),
          "not_cb_with_profanity": int((not_cb & prof).sum()),
          "fp_rate_with_profanity": float(((pred != 0) & not_cb & prof).sum() / max((not_cb & prof).sum(), 1)),
          "fp_rate_without_profanity": float(((pred != 0) & not_cb & ~prof).sum() / max((not_cb & ~prof).sum(), 1)),
          "cb_total": int((~not_cb).sum()),
          "cb_predicted_not_cb": int((~not_cb & (pred == 0)).sum())}
    cm = np.array(classification_metrics(y, P, labels)["confusion_matrix"])
    off = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if i != j]
    off.sort(key=lambda t: -t[2])
    # Qualitative examples: ids only here (texts are looked up at figure time, truncated).
    examples = defaultdict(list)
    conf = P.max(1)
    for k, i in enumerate(ids):
        tag = "correct" if pred[k] == y[k] else ("false_positive" if y[k] == 0 else
                                                ("false_negative" if pred[k] == 0 else "wrong_type"))
        examples[tag].append({"id": i, "y": labels[y[k]], "pred": labels[pred[k]], "conf": float(conf[k])})
    for tag in examples:
        examples[tag].sort(key=lambda e: -e["conf"])
    ambiguous = sorted(({"id": i, "y": labels[y[k]], "pred": labels[pred[k]],
                         "margin": float(np.sort(P[k])[-1] - np.sort(P[k])[-2])} for k, i in enumerate(ids)),
                       key=lambda e: e["margin"])[:20]
    kept = {k: v[:25] for k, v in examples.items() if k != "correct"}
    per_class_correct = defaultdict(list)
    for e in examples.get("correct", []):          # already sorted by confidence
        if len(per_class_correct[e["y"]]) < 5:
            per_class_correct[e["y"]].append(e)
    kept["correct"] = [e for c in labels for e in per_class_correct.get(c, [])]
    return {"model": pretty(gk), "slices": slices, "false_positive_analysis": fp, "top_confusions": off[:12],
            "examples": kept, "ambiguous": ambiguous,
            "per_item_features": {"ids": ids, **{k: list(map(float, v)) for k, v in feats.items()}}}


def main() -> int:
    groups = collect_runs()
    if not groups:
        print("no completed runs")
        return 1
    meta = test_meta("mc7")
    dedup = {i for i, r in meta.items() if not r["meta"].get("near_dup_in_train")}
    train_vocab = {w for r in load_jsonl(PROCESSED_DIR / "mc7" / "train.jsonl") for w in normalised_key(r["text_raw"]).split()}
    res = {
        "runs": {gk: [r["exp_id"] for r in rs] for gk, rs in groups.items()},
        "mc7_test": group_table(groups, "mc7"),
        "mc7_test_dedup": group_table(groups, "mc7", subset=dedup),
        "mc7_val": group_table(groups, "mc7", "val"),
        "mc2_test": group_table(groups, "mc2"),
        "mc2_test_dedup": group_table(groups, "mc2", subset=dedup),
        "sarcasm_test": group_table(groups, "sarcasm"),
        "conda_test": {**group_table(groups, "conda"), **group_table(groups, "conda_ctx")},
        "n_test_dedup": len(dedup),
    }
    neardup = {i for i, r in meta.items() if r["meta"].get("near_dup_in_train")}
    res["mc7_test_neardup"] = group_table(groups, "mc7", subset=neardup)
    shift = RESULTS_DIR / "analysis" / "split_shift.json"
    if shift.exists():
        res["split_shift"] = load_json(shift)
    base = "mc7|muril|base"
    comps = {}
    for b in ("mc7|xlmr|base", "mc7|hingroberta|base", "mc7|muril|mtl", "mc7|muril|mtl_shuffled",
              "mc7|muril|fusion", "mc7|tfidf_svm", "mc7|tfidf_lr"):
        comps[f"{base} -> {b}"] = paired(groups, base, b)
    comps["mc7|muril|mtl_shuffled -> mc7|muril|mtl"] = paired(groups, "mc7|muril|mtl_shuffled", "mc7|muril|mtl")
    comps["mc7|muril|fusion_shuffled -> mc7|muril|fusion"] = paired(groups, "mc7|muril|fusion_shuffled", "mc7|muril|fusion")
    comps["mc7|muril|base -> mc7|muril|full_ft"] = paired(groups, "mc7|muril|base", "mc7|muril|full_ft")
    for b in ("mc2|xlmr|base", "mc2|muril|mtl", "mc2|muril|fusion", "mc2|tfidf_svm"):
        comps[f"mc2|muril|base -> {b}"] = paired(groups, "mc2|muril|base", b)
    comps["sarcasm|tfidf_lr -> sarcasm|muril|base"] = paired(groups, "sarcasm|tfidf_lr", "sarcasm|muril|base")
    res["paired_tests"] = {k: v for k, v in comps.items() if v}
    res["legacy_boost_ablation"] = legacy_boost(groups)
    res["external"] = external(groups)
    res["error_analysis_mc7"] = error_analysis(groups, train_vocab)
    # Auxiliary sarcasm head of MTL models evaluated on Swami test (diagnostic).
    res["mtl_aux_heads"] = {r["exp_id"]: {k: r["metrics"]["aux_test"][k] for k in ("macro_f1", "accuracy")}
                            for gk, rs in groups.items() for r in rs if "aux_test" in r["metrics"]}
    # Training curves (per run) for figures.
    res["histories"] = {r["exp_id"]: r["metrics"]["history"] for rs in groups.values() for r in rs
                        if "history" in r["metrics"]}
    save_json(res, OUT / "analysis.json")
    for t in ("mc7_test", "mc2_test", "sarcasm_test"):
        print(f"--- {t}")
        for gk, row in res[t].items():
            s = row["seed_mean"]["macro_f1"]
            print(f"  {row['name']:32s} macro-F1 {s['mean']:.4f} ± {s['std']:.4f} (n={s['n_seeds']})  "
                  f"ens {row['ensemble']['macro_f1']:.4f} [{row['ensemble_macro_f1_ci95']['lo']:.3f},"
                  f"{row['ensemble_macro_f1_ci95']['hi']:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
