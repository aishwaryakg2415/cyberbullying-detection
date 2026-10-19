"""Result figures. Each function reads saved artefacts only and skips (returns False)
when its inputs do not exist yet, so no empty or placeholder chart is ever written."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from ..evaluation.aggregate import ensemble, labels_of, load_preds, pretty
from ..utils.io import PROCESSED_DIR, RESULTS_DIR, load_json, load_jsonl
from .style import CLASSICAL, INK, INK2, MUTED, SERIES, div_cmap, save, seq_cmap

SHORT = {"not_cyberbullying": "not CB", "cyberbullying": "CB", "non_sarcastic": "non-sarc.", "sarcastic": "sarcastic"}


def dataset_distributions(out: Path, reg: list) -> bool:
    st = load_json(PROCESSED_DIR / "data_statistics.json")
    rows = []
    for split, info in st["mc_hinglish"]["mc7"]["splits"].items():
        for lab, n in info["label_dist"].items():
            rows.append({"dataset": "MC-Hinglish1.0", "split": split, "label": lab, "n": n})
    df = pd.DataFrame(rows)
    order = df.groupby("label")["n"].sum().sort_values(ascending=True).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    left = np.zeros(len(order))
    for k, split in enumerate(["train", "val", "test"]):
        v = np.array([df[(df.label == l) & (df.split == split)]["n"].sum() for l in order])
        ax.barh(order, v, left=left, color=SERIES[k], label=split, edgecolor="white", linewidth=1.5, height=0.65)
        left += v
    for i, t in enumerate(left):
        ax.text(t + 15, i, f"{int(t):,}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("comments")
    ax.set_title("MC-Hinglish1.0 classes by split", loc="left")
    ax.set_xlim(0, left.max() * 1.16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.grid(axis="y", visible=False)
    sw = st["swami_sarcasm"]["sarcasm"]["splits"]
    swd = pd.DataFrame([{"dataset": "Swami sarcasm", "split": s, "label": l, "n": n}
                        for s, i in sw.items() for l, n in i["label_dist"].items()])
    ax = axes[1]
    for k, lab in enumerate(["0", "1"]):
        v = [swd[(swd.split == s) & (swd.label == lab)]["n"].sum() for s in ["train", "val", "test"]]
        ax.bar(np.arange(3) + (k - 0.5) * 0.38, v, width=0.36, color=SERIES[k],
               label=["non-sarcastic", "sarcastic"][k])
    ax.set_xticks(range(3), ["train", "val", "test"])
    ax.set_title("Swami sarcasm", loc="left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    cs = st["conda"]["conversations"]
    ax = axes[2]
    tox = [cs[s]["toxic_conversations"] for s in ["train", "val", "test"]]
    non = [cs[s]["conversations"] - cs[s]["toxic_conversations"] for s in ["train", "val", "test"]]
    ax.bar(np.arange(3) - 0.19, non, width=0.36, color=SERIES[0], label="no toxic turn")
    ax.bar(np.arange(3) + 0.19, tox, width=0.36, color=SERIES[1], label="≥1 toxic turn")
    ax.set_xticks(range(3), ["train", "val", "test"])
    ax.set_title("CONDA conversations", loc="left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig07_dataset_distributions", pd.concat([df, swd]), reg,
         "Class distributions of the datasets after splitting. MC-Hinglish validation is carved from the official "
         "training split; the official test split is unchanged.")
    return True


def language_distribution(out: Path, reg: list, analysis: dict) -> bool:
    st = load_json(PROCESSED_DIR / "data_statistics.json")
    tl = st["swami_sarcasm"]["quality"]["token_lang_dist"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    labs = ["hi", "en", "rest"]
    vals = [tl.get(k, 0) for k in labs]
    ax.bar(["Hindi (romanised)", "English", "other / punct."], vals, color=SERIES[0], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v / sum(vals):.0%}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_ylabel("tokens")
    ax.set_title("Swami corpus: gold token language tags", loc="left")
    ax.grid(axis="x", visible=False)
    ea = analysis.get("error_analysis_mc7")
    ok = True
    data = {"swami_token_lang": tl}
    if ea:
        e = np.array(ea["per_item_features"]["english_function_word_ratio"])
        axes[1].hist(e, bins=np.linspace(0, 0.6, 25), color=SERIES[0], edgecolor="white", linewidth=1)
        axes[1].set_xlabel("share of English function words (heuristic)")
        axes[1].set_ylabel("MC-Hinglish test comments")
        axes[1].set_title("MC-Hinglish: English-function-word share", loc="left")
        axes[1].grid(axis="x", visible=False)
        data["mc_test_english_function_word_ratio"] = e.tolist()
    else:
        axes[1].remove()
        ok = False
    fig.tight_layout()
    save(fig, out, "fig08_language_distribution", data, reg,
         "Language composition. Left: gold token-level language tags released with the Swami et al. corpus. Right: "
         "share of English function words (scikit-learn English stop-word list) per MC-Hinglish test comment. This "
         "is a heuristic proxy for English content, since MC-Hinglish has no language tags. All MC-Hinglish texts are "
         "in Roman script.")
    return ok


def training_curves(out: Path, reg: list, analysis: dict) -> bool:
    H = analysis["histories"]
    keys = [k for k in ("mc7__muril__base__s13", "mc7__xlmr__base__s13", "mc7__hingroberta__base__s13") if k in H]
    if not keys:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    rows = []
    for c, k in enumerate(keys):
        h = H[k]
        ep = [r["epoch"] + 1 for r in h]
        name = pretty("mc7|" + k.split("__")[1] + "|base")
        axes[0].plot(ep, [r["train_loss"] for r in h], color=SERIES[c], ls="--", lw=1.6)
        axes[0].plot(ep, [r["val_loss"] for r in h], color=SERIES[c], label=name)
        axes[1].plot(ep, [r["val_macro_f1"] for r in h], color=SERIES[c], marker="o", ms=4, label=name)
        for r in h:
            rows.append({"run": k, "epoch": r["epoch"] + 1, "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                         "val_macro_f1": r["val_macro_f1"], "val_accuracy": r["val_accuracy"]})
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy (class-weighted for train)")
    axes[0].set_title("Loss (dashed = train, solid = validation)", loc="left")
    axes[0].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation macro-F1")
    axes[1].set_title("Validation macro-F1", loc="left")
    axes[1].legend()
    for a in axes:
        a.set_xticks(sorted({r["epoch"] for r in rows}))
    fig.tight_layout()
    save(fig, out, "fig09_training_curves_mc7", pd.DataFrame(rows), reg,
         "Training and validation loss and validation macro-F1 per epoch on MC7 (seed 13). The checkpoint with "
         "the highest validation macro-F1 is evaluated on test.")
    return True


def confusion(out: Path, reg: list, groups: dict) -> bool:
    gks = [g for g in ("mc7|muril|base", "mc7|tfidf_svm") if g in groups]
    if not gks:
        return False
    labels = labels_of("mc7")
    fig, axes = plt.subplots(1, len(gks), figsize=(3.9 * len(gks), 3.6))
    axes = np.atleast_1d(axes)
    data = {}
    for ax, gk in zip(axes, gks):
        ids, y, P = ensemble(groups[gk], "test")
        K = len(labels)
        cm = np.zeros((K, K), int)
        for t, p in zip(y, P.argmax(1)):
            cm[t, p] += 1
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=seq_cmap(), vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if rn[i, j] > 0.55 else INK)
        sl = [SHORT.get(l, l) for l in labels]
        ax.set_xticks(range(K), sl, rotation=40, ha="right")
        ax.set_yticks(range(K), sl)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.grid(False)
        n = len(groups[gk])
        ax.set_title(pretty(gk) + (f" ({n}-seed ensemble)" if n > 1 else ""), loc="left", fontsize=9)
        data[gk] = cm.tolist()
    fig.tight_layout()
    save(fig, out, "fig11_confusion_mc7", data, reg,
         "Confusion matrices on the official MC-Hinglish1.0 test set (counts; shading = row-normalised recall).")
    return True


def roc_pr(out: Path, reg: list, groups: dict) -> bool:
    panels = [("mc2", ["mc2|muril|base", "mc2|xlmr|base", "mc2|tfidf_svm"], "Binary cyberbullying (MC2)"),
              ("sarcasm", ["sarcasm|muril|base", "sarcasm|tfidf_lr"], "Sarcasm (Swami test)")]
    panels = [(t, [g for g in gs if g in groups], title) for t, gs, title in panels]
    panels = [p for p in panels if p[1]]
    if not panels:
        return False
    fig, axes = plt.subplots(2, len(panels), figsize=(3.7 * len(panels), 6.4), squeeze=False)
    rows = []
    for c, (task, gks, title) in enumerate(panels):
        for k, gk in enumerate(gks):
            ids, y, P = ensemble(groups[gk], "test")
            fpr, tpr, _ = roc_curve(y, P[:, 1])
            pr, rc, _ = precision_recall_curve(y, P[:, 1])
            from sklearn.metrics import auc, average_precision_score
            a, ap = auc(fpr, tpr), average_precision_score(y, P[:, 1])
            col = SERIES[k] if not gk.split("|")[1].startswith("tfidf") else CLASSICAL
            axes[0, c].plot(fpr, tpr, color=col, label=f"{pretty(gk)} (AUC {a:.3f})")
            axes[1, c].plot(rc, pr, color=col, label=f"{pretty(gk)} (AP {ap:.3f})")
            rows.append({"task": task, "model": pretty(gk), "roc_auc": a, "pr_auc": ap})
        axes[0, c].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        base = y.mean()
        axes[1, c].axhline(base, color=MUTED, lw=0.8, ls=":")
        axes[0, c].set(xlabel="false-positive rate", ylabel="true-positive rate")
        axes[1, c].set(xlabel="recall", ylabel="precision", ylim=(0, 1.02))
        axes[0, c].set_title(title + " — ROC", loc="left", fontsize=9)
        axes[1, c].set_title(title + " — PR", loc="left", fontsize=9)
        axes[0, c].legend(loc="lower right", fontsize=7)
        axes[1, c].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig12_roc_pr", pd.DataFrame(rows), reg,
         "ROC and precision–recall curves of seed-ensembled probabilities on the binary tasks. Dotted lines: chance "
         "(ROC) and positive-class prevalence (PR). The 7-class task uses macro one-vs-rest AUC, reported in Table 3.")
    return True


def model_comparison(out: Path, reg: list, analysis: dict) -> bool:
    rows = []
    for tab, label in (("mc7_test", "full test (n=1000)"), ("mc7_test_dedup", f"de-duplicated test (n={analysis['n_test_dedup']})")):
        for gk, r in analysis[tab].items():
            if gk.endswith("majority"):
                continue
            s = r["seed_mean"]["macro_f1"]
            rows.append({"model": r["name"], "gk": gk, "subset": label, "mean": s["mean"], "std": s["std"],
                         "n_seeds": s["n_seeds"], "ci_lo": r["ensemble_macro_f1_ci95"]["lo"],
                         "ci_hi": r["ensemble_macro_f1_ci95"]["hi"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return False
    order = (df[df.subset.str.startswith("full")].sort_values("mean")["model"].tolist())
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.0))
    for k, (sub, dd) in enumerate(df.groupby("subset", sort=False)):
        dd = dd.set_index("model").reindex(order)
        y = np.arange(len(order)) + (k - 0.5) * 0.38
        ax.barh(y, dd["mean"], height=0.36, color=SERIES[k], label=sub,
                xerr=dd["std"].where(dd["n_seeds"] > 1), error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
        for yy, m, sd in zip(y, dd["mean"], dd["std"].fillna(0)):
            ax.text(m + sd + 0.004, yy, f"{m:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("test macro-F1 (bar = mean over seeds; whisker = ±1 s.d.)")
    lo = max(0, df["mean"].min() - 0.08)
    ax.set_xlim(lo, min(1, (df["mean"] + df["std"].fillna(0)).max() + 0.05))
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("MC7: model comparison", loc="left")
    fig.tight_layout()
    save(fig, out, "fig14_model_comparison_mc7", df, reg,
         "Seven-way macro-F1 on the official test set and on its subset without near-duplicates of training items. "
         "Transformers: mean ± s.d. over 3 seeds. Baselines are deterministic (single run). The x-axis does not start at 0.")
    return True


def ablation(out: Path, reg: list, analysis: dict) -> bool:
    keys = [("mc7|muril|base", "MuRIL (no sarcasm)"), ("mc7|muril|mtl_shuffled", "+ MTL, shuffled labels (control)"),
            ("mc7|muril|mtl", "+ MTL, real sarcasm labels"),
            ("mc7|muril|fusion_shuffled", "+ fusion, shuffled-sarcasm features (control)"),
            ("mc7|muril|fusion", "+ fusion, sarcasm-model features")]
    t = analysis["mc7_test"]
    keys = [(g, n) for g, n in keys if g in t]
    if len(keys) < 2:
        return False
    rows = []
    for g, n in keys:
        s = t[g]["seed_mean"]["macro_f1"]
        rows.append({"variant": n, "gk": g, "mean": s["mean"], "std": s["std"], "values": s["values"],
                     "mockery_f1_ens": t[g]["ensemble"]["per_class"]["mockery"]["f1"]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["mean"], color=SERIES[0], height=0.55)
    for yy, (_, r) in zip(y, df.iterrows()):
        ax.scatter(r["values"], [yy] * len(r["values"]), color=INK, s=12, zorder=3)
        ax.text(r["mean"] + 0.012, yy + 0.18, f"{r['mean']:.3f} ± {r['std']:.3f}", fontsize=7, color=INK2)
    ax.set_yticks(y, df["variant"])
    ax.set_xlim(max(0, df["mean"].min() - 0.06), min(1, df["mean"].max() + 0.06))
    ax.set_xlabel("test macro-F1 (dots = individual seeds)")
    ax.set_title("Sarcasm-component ablation (MC7)", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.barh(y, df["mockery_f1_ens"], color=SERIES[1], height=0.55)
    ax.set_yticks(y, [""] * len(y))
    for yy, v in zip(y, df["mockery_f1_ens"]):
        ax.text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_xlim(max(0, df["mockery_f1_ens"].min() - 0.08), min(1, df["mockery_f1_ens"].max() + 0.06))
    ax.set_xlabel("F1 on 'mockery' (seed ensemble)")
    ax.set_title("Mockery class", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "fig15_ablation_sarcasm", df.drop(columns=["values"]).assign(values=df["values"].astype(str)), reg,
         "Ablation of the sarcasm components on MC7 (MuRIL). Controls: MTL with permuted sarcasm labels, and fusion "
         "with features from a sarcasm model trained on permuted labels. A gain over the matching control would be "
         "attributable to sarcasm knowledge; none is observed.")
    return True


def calibration_fig(out: Path, reg: list, groups: dict) -> bool:
    gk = "mc7|muril|base"
    if gk not in groups:
        return False
    run = sorted(groups[gk], key=lambda r: r["cfg"]["seed"])[0]
    m = run["metrics"]
    rows = []
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for k, (key, name) in enumerate((("test", "uncalibrated"), ("test_calibrated", f"temperature T={m['temperature']:.2f}"))):
        bins = [b for b in m[key]["reliability"] if b["n"]]
        ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], marker="o", ms=4, color=SERIES[k],
                label=f"{name} (ECE {m[key]['ece']:.3f})")
        rows += [{"setting": name, **b} for b in bins]
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=0.8)
    ax.set(xlabel="mean confidence (top label)", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"Reliability — MuRIL, MC7, seed {run['cfg']['seed']}", loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig19_calibration_mc7", pd.DataFrame(rows), reg,
         "Reliability diagram on the MC7 test set before and after temperature scaling (T fitted on validation).")
    return True


def shap_heatmaps(out: Path, reg: list) -> bool:
    p = RESULTS_DIR / "explainability" / "examples.json"
    if not p.exists():
        return False
    ex = load_json(p)
    show = []
    for kind in ("correct", "false_positive", "false_negative", "wrong_type", "ambiguous"):
        show += [e for e in ex if e["kind"] == kind][:2]
    show = show[:8]
    fig, axes = plt.subplots(len(show), 1, figsize=(7.4, 0.62 * len(show) + 0.6))
    axes = np.atleast_1d(axes)
    cmap = div_cmap()
    vmax = max(max(abs(v) for v in e["word_values"]) for e in show) or 1
    # Fix the layout first (colorbar steals space), then measure the monospace advance once.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal", fraction=0.03, pad=0.02, aspect=50)
    cb.set_label("SHAP value toward the explained (predicted) class; red = pushes toward, blue = away", fontsize=7)
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = axes[0].text(0, 0, "M" * 20, fontsize=7.5, family="DejaVu Sans Mono")
    bb = probe.get_window_extent(renderer)
    x0, x1 = axes[0].transAxes.inverted().transform([(bb.x0, 0), (bb.x1, 0)])[:, 0]
    cw = (x1 - x0) / 20          # width of one monospace character in axes units
    probe.remove()
    for ax, e in zip(axes, show):
        x = 0.0
        for w, v in zip(e["words"], e["word_values"]):
            w_disp = (w[:14] + "…") if len(w) > 15 else w
            if x + cw * (len(w_disp) + 1) > 0.97:
                ax.text(x, 0.35, "…", fontsize=8, va="center")
                break
            ax.text(x, 0.35, w_disp, fontsize=7.5, va="center", ha="left", family="DejaVu Sans Mono",
                    bbox=dict(boxstyle="round,pad=0.18", fc=cmap(0.5 + 0.5 * v / vmax), ec="none"))
            x += cw * (len(w_disp) + 1.4)     # word width + box padding + gap
        ax.text(0, 0.95, f"[{e['kind'].replace('_', ' ')}] gold: {e['y_true']} | explained class: {e['target_label']} "
                         f"(p={max(e['prob']):.2f})", fontsize=7, color=INK2, va="center")
    save(fig, out, "fig16_shap_token_attributions",
         [{k: e[k] for k in ("id", "kind", "y_true", "target_label", "words", "word_values", "prob")} for e in show], reg,
         "Word-level SHAP attributions (subword values summed) for selected MC7 test comments explained with respect "
         "to the predicted class. Attributions describe the model's sensitivity, not the author's intent. Texts are "
         "shown as distributed (CC-BY-NC-SA-4.0); offensive content is not censored, for scientific transparency.")
    return True


def faithfulness_fig(out: Path, reg: list) -> bool:
    p = RESULTS_DIR / "explainability" / "faithfulness.json"
    if not p.exists():
        return False
    f = load_json(p)["summary"]
    ks = sorted(f, key=lambda k: int(k.split("=")[1]))
    df = pd.DataFrame([{"k": int(k.split("=")[1]), **{a: f[k][a] for a in ("mean_drop_top", "mean_drop_random", "n")},
                        "lo": f[k]["diff_ci95"][0], "hi": f[k]["diff_ci95"][1]} for k in ks])
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    x = np.arange(len(df))
    ax.bar(x - 0.19, df["mean_drop_top"], 0.36, color=SERIES[0], label="delete top-k SHAP words")
    ax.bar(x + 0.19, df["mean_drop_random"], 0.36, color=CLASSICAL, label="delete k random words")
    ax.set_xticks(x, [f"k={k}" for k in df["k"]])
    ax.set_ylabel("mean drop in P(predicted class)")
    ax.set_title("Deletion test (faithfulness)", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig16b_shap_faithfulness", df, reg,
         f"Faithfulness check on {int(df['n'].iloc[0])} random MC7 test comments: drop in the predicted-class "
         "probability after deleting the k words with the highest SHAP value vs. k random words (10 draws).")
    return True


def error_slices(out: Path, reg: list, analysis: dict) -> bool:
    ea = analysis.get("error_analysis_mc7")
    if not ea:
        return False
    sl = [s for s in ea["slices"] if "macro_f1" in s]
    df = pd.DataFrame(sl)
    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(df) + 0.8))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["accuracy"], color=SERIES[0], height=0.6)
    for yy, (_, r) in zip(y, df.iterrows()):
        ax.text(r["accuracy"] + 0.005, yy, f"{r['accuracy']:.3f}  (n={r['n']})", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, df["slice"], fontsize=7.5)
    ax.set_xlim(max(0, df["accuracy"].min() - 0.1), 1.0)
    ax.set_xlabel("accuracy of MuRIL seed ensemble on slice")
    ax.set_title("Error analysis by slice (MC7 test)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "fig17_error_slices", df, reg,
         "Accuracy on test slices. Slices are defined by heuristics (profanity lexicon, English function-word share, "
         "out-of-vocabulary rate relative to training text) or by the sarcasm model's prediction. None is a gold "
         "annotation; small slices have wide uncertainty.")
    return True


def runtime_size(out: Path, reg: list) -> bool:
    p = RESULTS_DIR / "benchmarks" / "inference.json"
    if not p.exists():
        return False
    b = load_json(p)
    df = pd.DataFrame(b["models"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    order = df.sort_values("params_m")["name"].tolist()
    d = df.set_index("name").reindex(order)
    y = np.arange(len(order))
    axes[0].barh(y - 0.19, d["latency_ms_bs1_cpu"], 0.36, color=SERIES[0], label="CPU, batch 1")
    if "latency_ms_bs1_mps" in d:
        axes[0].barh(y + 0.19, d["latency_ms_bs1_mps"], 0.36, color=SERIES[1], label="MPS GPU, batch 1")
    axes[0].set_yticks(y, order)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("median latency per comment (ms, log scale)")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", visible=False)
    axes[1].barh(y, d["size_mb"], color=SERIES[0], height=0.55)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("serialized size (MB, log scale)")
    for yy, v in zip(y, d["size_mb"]):
        axes[1].text(v * 1.08, yy, f"{v:,.1f}", va="center", fontsize=7, color=INK2)
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "fig18_runtime_size", df, reg,
         "Inference latency (median of repeated single-comment predictions) and serialized model size, measured on "
         "an Apple M1 Pro (16 GB).")
    return True


def early_curves(out: Path, reg: list) -> bool:
    d = RESULTS_DIR / "early_detection"
    files = sorted(d.glob("conda*.json")) if d.exists() else []
    if not files:
        return False
    picks = {}
    for f in files:
        r = load_json(f)
        name = f.stem
        fam = "ctx" if name.startswith("conda_ctx") else "utt"
        mdl = "tfidf" if "tfidf" in name else "distilroberta"
        if mdl == "distilroberta" and not name.endswith("s13"):
            continue
        picks[f"{mdl}|{fam}"] = r
    labels = {"tfidf|utt": "TF-IDF LR, turn only", "tfidf|ctx": "TF-IDF LR, + 2 prior turns",
              "distilroberta|utt": "DistilRoBERTa, turn only", "distilroberta|ctx": "DistilRoBERTa, + 2 prior turns"}
    keys = [k for k in ("distilroberta|utt", "distilroberta|ctx", "tfidf|utt") if k in picks]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    rows = []
    for c, k in enumerate(keys):
        pc = picks[k]["policies"]["max"]["prefix_curve_test"]
        ks = [r["k"] for r in pc]
        axes[0].plot(ks, [r["eventual_recall"] for r in pc], color=SERIES[c], marker="o", ms=3, label=labels[k])
        axes[0].plot(ks, [r["eventual_precision"] for r in pc], color=SERIES[c], ls="--", lw=1.4)
        dd = picks[k]["policies"]["max"]["test_at_maxF1"]["delay_dist"]
        rows += [{"scorer": labels[k], **r} for r in pc]
        xs = sorted(int(x) for x in dd)
        tot = sum(dd.values())
        axes[1].plot(xs, np.cumsum([dd[str(x)] if str(x) in dd else dd.get(x, 0) for x in xs]) / tot,
                     color=SERIES[c], drawstyle="steps-post", label=labels[k])
    axes[0].set(xlabel="turns observed (k)", ylabel="vs. eventual conversation label", ylim=(0, 1.02))
    axes[0].set_title("Prefix decisions (solid = recall, dashed = precision)", loc="left", fontsize=8.5)
    axes[0].legend(fontsize=6.5, loc="lower right")
    axes[1].set(xlabel="detection delay (turns after first toxic turn)", ylabel="cumulative share of detected",
                ylim=(0, 1.02))
    axes[1].set_title("Delay of correct alarms", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig20_early_detection_proxy", pd.DataFrame(rows), reg,
         "Early-warning proxy study on CONDA test conversations (English game chat; not cyberbullying, not Hinglish). "
         "Threshold chosen on validation conversations. A negative delay means the alarm fired on a non-toxic turn "
         "before the first toxic turn.")
    return True
