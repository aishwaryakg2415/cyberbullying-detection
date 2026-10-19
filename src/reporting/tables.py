"""Paper tables built from results/analysis/*.json and data statistics (CSV + LaTeX + Markdown)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import PROCESSED_DIR, RESULTS_DIR, RUNS_DIR, load_json


def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{d}f}"
    return str(v)


def _ms(s, d=3):
    """mean ± sd string from a seed_mean entry."""
    if not s:
        return "–"
    return f"{s['mean']:.{d}f} ± {s['std']:.{d}f}" if s["n_seeds"] > 1 else f"{s['mean']:.{d}f}"


def write(df: pd.DataFrame, out: Path, name: str, caption: str, reg: list, label: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    md = df.copy()
    cols = list(md.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in md.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    (out / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$").replace("–", "--").replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("→", r"$\rightarrow$")  # noqa: E731
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}",
           rf"\label{{{label or 'tab:' + name}}}", r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule",
           " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        tex.append(" & ".join(esc(_fmt(r[c])) for c in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    reg.append({"file": str((out / name).as_posix()), "caption": caption})


def dataset_tables(out: Path, reg: list):
    st = load_json(PROCESSED_DIR / "data_statistics.json")
    man = load_json(Path(PROCESSED_DIR).parent / "dataset_manifest.json")
    d = man["datasets"]
    rows = [
        {"Dataset": "MC-Hinglish1.0", "Language": "hi-en (Roman)", "Unit": "comment", "Labels": "7 cyberbullying types",
         "Records": d["mc_hinglish"]["counts"]["records"], "Licence": "CC-BY-NC-SA-4.0", "Role": "train / val / test"},
        {"Dataset": "Swami et al. sarcasm", "Language": "hi-en (Roman)", "Unit": "tweet", "Labels": "sarcastic / not",
         "Records": d["swami_sarcasm"]["counts"]["records"], "Licence": "GPL-3.0", "Role": "sarcasm supervision"},
        {"Dataset": "CONDA", "Language": "en", "Unit": "chat turn", "Labels": "E / I / A / O intent",
         "Records": d["conda"]["counts"]["records"], "Licence": "not stated", "Role": "early-detection proxy"},
        {"Dataset": "TweetEval-Offensive (test)", "Language": "en", "Unit": "tweet", "Labels": "offensive / not",
         "Records": d["tweeteval_offensive"]["counts"]["records"], "Licence": "OLID terms", "Role": "external stress test"},
        {"Dataset": "User sarcasm file (human labels)", "Language": "hi-en", "Unit": "comment", "Labels": "sarcastic / not",
         "Records": st["external"]["usr_sarcasm"]["splits"]["test"]["n"], "Licence": "unknown",
         "Role": "supplementary test (provenance unknown)"},
    ]
    write(pd.DataFrame(rows), out, "tab01_datasets", "Datasets used in this study (records after validation).", reg)
    rows = []
    for task, name in (("mc7", "MC-Hinglish1.0"), ("sarcasm", "Swami sarcasm"), ("conda", "CONDA (turns)")):
        info = load_json(PROCESSED_DIR / task / "task_info.json")["splits"]
        for s, i in info.items():
            rows.append({"Dataset": name, "Split": s, "N": i["n"],
                         "Label distribution": ", ".join(f"{k}: {v}" for k, v in i["label_dist"].items())})
    write(pd.DataFrame(rows), out, "tab02_splits", "Split sizes and label distributions.", reg)
    q = st["mc_hinglish"]["quality"]
    rows = [{"Check": "official train: exact-duplicate groups", "Value": q["train_duplicates"]["exact_dup_groups"]},
            {"Check": "official train: normalised-duplicate groups", "Value": q["train_duplicates"]["normalised_dup_groups"]},
            {"Check": "official train: near-duplicate pairs (cos >= 0.9)", "Value": q["train_duplicates"]["near_dup_pairs@0.9"]},
            {"Check": "label conflicts among exact duplicates", "Value": q["train_duplicates"]["n_exact_dup_label_conflicts"]},
            {"Check": "test items with near-duplicate in official train", "Value": q["test_items_flagged_near_dup_in_official_train"]},
            {"Check": "our train<->val near-duplicate pairs", "Value": q["leakage_after_split"]["train__val"]["near_dup_pairs@0.9"]},
            {"Check": "Swami sarcastic tweets containing #sarcasm/#irony", "Value": "486 / 504"},
            {"Check": "CONDA conversations split across official train/valid files",
             "Value": st["conda"]["quality"]["split_overlap"]["official_files_shared_conversations"]}]
    write(pd.DataFrame(rows), out, "tab03_data_quality", "Data-quality and leakage audit (measured).", reg)


def config_table(out: Path, reg: list):
    import yaml
    from ..utils.io import ROOT
    cfg = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
    td = cfg["train_defaults"]
    rows = [{"Setting": k, "Value": str(v)} for k, v in td.items()]
    rows += [{"Setting": "optimiser", "Value": "AdamW (fused), linear warm-up/decay"},
             {"Setting": "seeds", "Value": ", ".join(map(str, cfg["seeds"]))},
             {"Setting": "MTL lambda", "Value": "0.5"},
             {"Setting": "model selection", "Value": "best validation macro-F1 epoch"},
             {"Setting": "TF-IDF features", "Value": "word 1-2-grams + char_wb 2-5-grams, sublinear TF"},
             {"Setting": "TF-IDF C grid (val-selected)", "Value": "0.1, 0.3, 1, 3, 10"}]
    for k, m in cfg["models"].items():
        rows.append({"Setting": f"encoder: {k}", "Value": f"{m['name']} @ {m['revision'][:8]}"})
    write(pd.DataFrame(rows), out, "tab04_hyperparameters", "Model configurations and hyperparameters (shared across models).", reg)


def results_tables(out: Path, reg: list, an: dict):
    def main_table(key, dedup_key, task_name, name, extra_auc):
        rows = []
        for gk, r in an[key].items():
            e = r["ensemble"]
            s = r["seed_mean"]
            dd = an[dedup_key].get(gk, {}).get("seed_mean", {}).get("macro_f1") if dedup_key else None
            rows.append({"Model": r["name"], "Seeds": r["n_runs"] if r["n_runs"] > 1 else 1,
                         "Accuracy": _ms(s.get("accuracy")), "Macro-F1": _ms(s.get("macro_f1")),
                         "Weighted-F1": _ms(s.get("weighted_f1")),
                         extra_auc: _ms(s.get("roc_auc_macro_ovr") or s.get("roc_auc")),
                         "Macro-F1 (dedup test)": _ms(dd) if dd else "–",
                         "Ens. macro-F1 [95% CI]": f"{e['macro_f1']:.3f} [{r['ensemble_macro_f1_ci95']['lo']:.3f}, "
                                                   f"{r['ensemble_macro_f1_ci95']['hi']:.3f}]",
                         "ECE": _ms(s.get("ece"))})
        df = pd.DataFrame(rows)
        write(df, out, name, f"{task_name}: test-set results (mean ± s.d. over seeds; ensemble = mean of seed "
                             "probabilities with bootstrap 95% CI; dedup = test items without near-duplicates in training).", reg)

    if an.get("mc7_test"):
        main_table("mc7_test", "mc7_test_dedup", "MC7 (7-way cyberbullying type)", "tab05_results_mc7", "ROC-AUC (macro OvR)")
        best = "mc7|muril|base" if "mc7|muril|base" in an["mc7_test"] else max(
            an["mc7_test"], key=lambda g: an["mc7_test"][g]["seed_mean"]["macro_f1"]["mean"])
        pc = an["mc7_test"][best]["ensemble"]["per_class"]
        svm = an["mc7_test"].get("mc7|tfidf_svm", {}).get("ensemble", {}).get("per_class", {})
        rows = [{"Class": c, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"], "Support": v["support"],
                 "F1 TF-IDF+SVM": svm.get(c, {}).get("f1")} for c, v in pc.items()]
        write(pd.DataFrame(rows), out, "tab06_per_class_mc7",
              f"Per-class results of {an['mc7_test'][best]['name']} (seed ensemble) on the MC7 test set, with TF-IDF+SVM F1 for reference.", reg)
    if an.get("mc2_test"):
        main_table("mc2_test", "mc2_test_dedup", "MC2 (binary cyberbullying)", "tab08_results_mc2", "ROC-AUC")
    if an.get("sarcasm_test"):
        main_table("sarcasm_test", None, "Sarcasm detection (Swami test, hashtags removed)", "tab14_results_sarcasm", "ROC-AUC")

    # Ablation + paired tests
    pt = an.get("paired_tests", {})
    rows = []
    for k, v in pt.items():
        ps = v["per_seed"]
        rows.append({"Comparison (B vs A)": f"{v['b_name']} vs {v['a_name']}",
                     "Delta macro-F1 (ensemble)": v["ensemble_bootstrap"]["delta"],
                     "95% CI": f"[{v['ensemble_bootstrap']['lo']:+.3f}, {v['ensemble_bootstrap']['hi']:+.3f}]",
                     "p (bootstrap)": v["ensemble_bootstrap"]["p_value"],
                     "p (McNemar)": v["ensemble_mcnemar"]["p_value"],
                     "Seeds with Delta>0": f"{sum(p['delta'] > 0 for p in ps)}/{len(ps)}"})
    if rows:
        write(pd.DataFrame(rows), out, "tab07_paired_tests",
              "Paired comparisons on the test set (B minus A). Bootstrap: 2,000 paired resamples of the seed-ensemble "
              "predictions; McNemar: exact test on per-item correctness. No multiple-comparison correction was applied; "
              "treat p-values as descriptive.", reg)
    lb = an.get("legacy_boost_ablation")
    if lb:
        rows = []
        for r in lb["per_seed"]:
            for setting in ("base@0.50", "base@0.55", "boost@0.55"):
                rows.append({"Seed": r["seed"], "Setting": setting, "Macro-F1": r[setting]["macro_f1"],
                             "Recall (CB)": r[setting]["recall_cb"], "False-positive rate": r[setting]["fpr"],
                             "Boosted items": r["n_boosted"] if setting == "boost@0.55" else "–"})
        write(pd.DataFrame(rows), out, "tab09_legacy_boost",
              "Original SACENet additive sarcasm boost (p + 0.18*s if s >= 0.45; threshold 0.55) re-evaluated on MC2 "
              "with a trained sarcasm model, compared with the unboosted MuRIL classifier.", reg)
    ext = an.get("external", {})
    rows = []
    for k, v in ext.items():
        ds, gk = k.split("::")
        rows.append({"Test set": {"olid_ext": "TweetEval-Offensive (en)", "usr_sarcasm": "User sarcasm file (hi-en)"}[ds],
                     "Model (trained on)": v["name"] + (" (MC2)" if ds == "olid_ext" else " (Swami)"), "N": v["n"],
                     "Macro-F1 (mean)": v["macro_f1_mean"], "ROC-AUC (mean)": v["roc_auc_mean"], "Runs": len(v["seeds"])})
    if rows:
        write(pd.DataFrame(rows), out, "tab10_external",
              "Cross-dataset evaluation without any re-training or threshold tuning. Label constructs differ from "
              "the training task (offensive language; differently sourced sarcasm), so these numbers measure transfer, "
              "not in-domain performance.", reg)
    ea = an.get("error_analysis_mc7")
    if ea:
        df = pd.DataFrame([s for s in ea["slices"] if "macro_f1" in s])[["slice", "n", "accuracy", "macro_f1"]]
        write(df.rename(columns={"slice": "Slice", "n": "N", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}), out,
              "tab11_error_slices", "Error analysis by heuristic slice (MuRIL seed ensemble, MC7 test).", reg)
        fp = ea["false_positive_analysis"]
        conf = pd.DataFrame(ea["top_confusions"], columns=["Gold", "Predicted", "Count"])
        write(conf, out, "tab11b_top_confusions", "Most frequent confusions (MuRIL seed ensemble, MC7 test).", reg)


def runtime_table(out: Path, reg: list):
    p = RESULTS_DIR / "benchmarks" / "inference.json"
    if not p.exists():
        return
    b = load_json(p)
    df = pd.DataFrame(b["models"])
    cols = [c for c in ("name", "params_m", "trainable_params_m", "size_mb", "latency_ms_bs1_cpu", "latency_ms_bs1_mps",
                        "throughput_per_s_bs32_mps", "train_minutes_mean") if c in df]
    write(df[cols], out, "tab12_runtime_size", "Model size, inference latency and training time (Apple M1 Pro).", reg)


def early_table(out: Path, reg: list):
    p = RESULTS_DIR / "early_detection" / "summary.json"
    if not p.exists():
        return
    s = load_json(p)
    rows = []
    for k, v in s.items():
        full = load_json(RESULTS_DIR / "early_detection" / f"{k}.json")["policies"]["max"]
        far = full["test_at_far10"]
        rows.append({"Turn scorer": k, "F1": v["f1"], "Precision": v["precision"], "Recall": v["recall"],
                     "False-alarm rate": v["false_alarm_rate"], "Mean delay": v["delay_mean"],
                     "ERDE_3": v["erde_3"], "ERDE_5": v["erde_5"], "Recall @FAR<=0.10(val)": far["recall"],
                     "FAR @FAR<=0.10(val)": far["false_alarm_rate"]})
    write(pd.DataFrame(rows), out, "tab13_early_detection",
          "Early-warning proxy evaluation on CONDA test conversations (running-max policy; thresholds chosen on validation).", reg)


def explain_table(out: Path, reg: list):
    d = RESULTS_DIR / "explainability"
    if not (d / "faithfulness.json").exists():
        return
    f = load_json(d / "faithfulness.json")["summary"]
    st = load_json(d / "stability.json")
    ov = load_json(d / "overhead.json")
    rows = [{"Measure": f"deletion drop, top-{k.split('=')[1]} vs random", "Value":
             f"{v['mean_drop_top']:.3f} vs {v['mean_drop_random']:.3f} (diff {v['mean_diff']:+.3f}, 95% CI "
             f"[{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}]; n={v['n']})"} for k, v in f.items()]
    for b, v in st["budget_vs_1000"].items():
        rows.append({"Measure": f"rank stability: {b.replace('_', ' ')}",
                     "Value": f"Spearman {v['spearman_mean']:.3f}; top-3 overlap {v['top3_overlap_mean']:.2f} (n={v['n']})"})
    o = st["shap_vs_occlusion"]
    rows.append({"Measure": "SHAP vs leave-one-word-out occlusion",
                 "Value": f"Spearman {o['spearman_mean']:.3f}; top-3 overlap {o['top3_overlap_mean']:.2f} (n={o['n']})"})
    rows.append({"Measure": f"time per explanation ({ov['device']}, {ov['shap_max_evals']} evals)",
                 "Value": f"{ov['shap_seconds_per_text_mean']:.2f} s vs {ov['inference_seconds_per_text_mean'] * 1000:.0f} ms "
                          f"inference (x{ov['overhead_factor']:.0f})"})
    write(pd.DataFrame(rows), out, "tab15_explainability", "SHAP faithfulness, stability and overhead (MuRIL, MC7).", reg)


def shift_tables(out: Path, reg: list, an: dict):
    sh = an.get("split_shift")
    if sh:
        rows = [{"Comparison": "official train vs official test", "ROC-AUC": sh["train_vs_test"]},
                {"Comparison": "official train vs our validation (control)", "ROC-AUC": sh["train_vs_val (control)"]}]
        for lab, v in sh["per_class"].items():
            rows.append({"Comparison": f"class '{lab}': train vs test (vs val)",
                         "ROC-AUC": f"{v['train_vs_test']:.3f} ({v['train_vs_val']:.3f})"})
        write(pd.DataFrame(rows), out, "tab03b_split_shift",
              "Domain-classifier test of distribution shift (TF-IDF LR, 5-fold CV ROC-AUC; 0.5 = indistinguishable).", reg)
    nd, dd = an.get("mc7_test_neardup", {}), an.get("mc7_test_dedup", {})
    rows = []
    for gk, r in an.get("mc7_test", {}).items():
        if gk.endswith("majority") or gk not in nd:
            continue
        rows.append({"Model": r["name"], "Full test (1000)": _ms(r["seed_mean"]["macro_f1"]),
                     f"Near-dup items ({nd[gk]['ensemble']['n']})": _ms(nd[gk]["seed_mean"]["macro_f1"]),
                     f"De-duplicated ({dd[gk]['ensemble']['n']})": _ms(dd[gk]["seed_mean"]["macro_f1"]),
                     "Validation": _ms(an["mc7_val"][gk]["seed_mean"]["macro_f1"]) if gk in an.get("mc7_val", {}) else "–"})
    if rows:
        write(pd.DataFrame(rows), out, "tab05b_mc7_subsets",
              "MC7 macro-F1 by evaluation subset: validation (in-distribution), full official test, test items "
              "with a near-duplicate in training, and the de-duplicated remainder (mean ± s.d. over seeds).", reg)
