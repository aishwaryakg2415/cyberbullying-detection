"""Regenerable system / method diagrams (matplotlib) + Mermaid sources.

Visual convention used in every diagram:
  solid box  = implemented and evaluated on the target data (MC-Hinglish1.0)
  dashed box = implemented but evaluated only on a proxy dataset, or not validated
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import AXIS, INK, INK2, MUTED, SERIES, save

FILL = {"data": "#eef4fc", "model": "#fdf0ea", "eval": "#e9f7f1", "out": "#f4f3f0"}


def _box(ax, x, y, w, h, text, kind="model", dashed=False, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=FILL[kind], ec=MUTED if dashed else INK2, lw=1.0,
                                ls=(0, (4, 3)) if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True,
            fontweight="bold" if bold else "normal")


def _arrow(ax, x1, y1, x2, y2, dashed=False, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0,
                                 color=MUTED if dashed else INK2, ls=(0, (4, 3)) if dashed else "-"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, fontsize=7, color=INK2, ha="center")


def _canvas(w=7.2, h=4.0, xlim=10, ylim=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.axis("off")
    return fig, ax


def _legend(ax, x, y):
    _box(ax, x, y, 1.6, 0.35, "evaluated on MC-Hinglish", "out", fs=6.5)
    _box(ax, x + 1.8, y, 1.6, 0.35, "proxy / not validated", "out", dashed=True, fs=6.5)


def system_architecture(out: Path, reg: list):
    fig, ax = _canvas(7.8, 4.6, 11, 6.6)
    W = 2.7
    _box(ax, 1.5, 5.7, W, 0.75, "Hinglish comment\n(Roman / Devanagari)", "data", fs=7.5)
    _box(ax, 1.5, 4.4, W, 0.75, "Normalisation\n(code-mix preserving)", "data", fs=7.5)
    _box(ax, 5.0, 4.4, W, 0.95, "MuRIL encoder\n(shared, fine-tuned)", "model", fs=8, bold=True)
    _box(ax, 8.9, 5.5, W, 0.75, "Cyberbullying-type head\n(7 classes; binary view)", "model", fs=7.5)
    _box(ax, 8.9, 4.2, W, 0.75, "Auxiliary sarcasm head\n(MTL; Swami et al. labels)", "model", fs=7.5)
    _box(ax, 5.0, 2.8, W, 0.75, "Separate sarcasm model\n(feature-fusion variant)", "model", fs=7.5)
    _box(ax, 8.9, 2.8, W, 0.75, "Temperature-scaled\nprobabilities", "eval", fs=7.5)
    _box(ax, 8.9, 1.5, W, 0.75, "SHAP token attributions\n+ faithfulness check", "eval", fs=7.5)
    _box(ax, 5.0, 1.5, W, 0.75, "Causal early warning\n(turn scores -> alarm)", "eval", dashed=True, fs=7.5)
    _box(ax, 1.5, 1.5, W, 0.75, "Conversation stream\n(CONDA proxy only)", "data", dashed=True, fs=7.5)
    _arrow(ax, 1.5, 5.32, 1.5, 4.78)
    _arrow(ax, 2.85, 4.4, 3.65, 4.4)
    _arrow(ax, 6.35, 4.6, 7.55, 5.4)
    _arrow(ax, 6.35, 4.3, 7.55, 4.2)
    _arrow(ax, 2.2, 4.02, 3.9, 3.18)
    _arrow(ax, 6.35, 2.95, 7.7, 5.12)
    ax.text(7.25, 3.55, "fusion\nfeatures", fontsize=6.5, color=INK2, ha="center")
    _arrow(ax, 10.35, 5.12, 10.35, 3.18)
    ax.text(10.45, 4.2, "", fontsize=6)
    _arrow(ax, 8.9, 2.42, 8.9, 1.88)
    _arrow(ax, 2.85, 1.5, 3.65, 1.5, dashed=True)
    _box(ax, 2.0, 0.35, 3.0, 0.4, "solid: evaluated on MC-Hinglish1.0", "out", fs=6.5)
    _box(ax, 5.6, 0.35, 3.6, 0.4, "dashed: proxy dataset only / not validated", "out", dashed=True, fs=6.5)
    save(fig, out, "fig01_system_architecture", registry=reg,
         caption="SACENet system overview. Solid components were trained and evaluated on MC-Hinglish1.0. "
                 "The dashed early-warning path was evaluated only on the English CONDA proxy dataset.")


def data_pipeline(out: Path, reg: list):
    fig, ax = _canvas(7.6, 2.8, 12, 3.4)
    steps = [("Official sources\n(pinned revisions)", "data"), ("SHA-256 + count\nverification", "data"),
             ("Unified schema\n(labels kept separate)", "data"), ("Cleaning\n(emoji→names, URL/@user,\nhashtags removed: Swami)", "data"),
             ("Exact / normalised /\nnear-dup detection", "eval"), ("Group-aware stratified\nsplits (+ ID lists)", "eval"),
             ("Leakage report\n(train↔val↔test)", "eval")]
    xs = [0.9 + i * 1.7 for i in range(len(steps))]
    for x, (t, k) in zip(xs, steps):
        _box(ax, x, 2.0, 1.5, 1.2, t, k, fs=7)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + 0.75, 2.0, b - 0.75, 2.0)
    ax.text(6, 0.6, "No oversampling. Class weights from the training split only. Official MC-Hinglish test kept intact; "
                    "near-duplicates of training items flagged.", ha="center", fontsize=7.5, color=INK2)
    save(fig, out, "fig02_data_pipeline", registry=reg,
         caption="Dataset acquisition and preprocessing pipeline (scripts/download_datasets.py → verify_datasets.py → prepare_data.py).")


def model_architecture(out: Path, reg: list):
    fig, ax = _canvas(7.2, 3.6, 10, 5)
    _box(ax, 1.3, 2.5, 2.0, 0.8, "Tokens x\n(WordPiece, ≤128)", "data")
    _box(ax, 4.0, 2.5, 2.4, 2.4, "Transformer encoder\n12 layers, d=768\n\n(token-embedding table\nfrozen; all layers\nfine-tuned)", "model")
    _box(ax, 6.6, 2.5, 1.4, 0.8, "h = [CLS]\n+ dropout", "model")
    _box(ax, 8.7, 3.4, 1.9, 0.8, "softmax(W h + b)\n7 types", "model")
    _box(ax, 8.7, 1.6, 1.9, 0.8, "softmax(W' h + b')\nsarcasm (MTL)", "model")
    _arrow(ax, 2.3, 2.5, 2.8, 2.5)
    _arrow(ax, 5.2, 2.5, 5.9, 2.5)
    _arrow(ax, 7.3, 2.7, 7.75, 3.3)
    _arrow(ax, 7.3, 2.3, 7.75, 1.7)
    ax.text(5, 0.35, r"$\mathcal{L} = \mathrm{CE}_w(y, \hat y) + \lambda\,\mathrm{CE}_{w'}(s, \hat s)$,  $\lambda=0.5$ (MTL only)",
            ha="center", fontsize=9, color=INK)
    save(fig, out, "fig03_model_architecture", registry=reg,
         caption="Classifier architecture shared by MuRIL, XLM-R and HingRoBERTa runs. The sarcasm head exists only "
                 "in the multitask variants; CE_w is class-weighted cross-entropy.")


def sarcasm_fusion(out: Path, reg: list):
    fig, ax = _canvas(7.8, 3.2, 10.4, 4.6)
    ax.text(2.4, 4.3, "(a) Multitask (MTL)", fontsize=9, fontweight="bold", ha="center")
    _box(ax, 1.2, 3.2, 1.8, 0.6, "MC-Hinglish batch", "data", fs=7.5)
    _box(ax, 1.2, 2.2, 1.8, 0.6, "Swami sarcasm\nbatch", "data", fs=7.5)
    _box(ax, 3.2, 2.7, 1.4, 1.4, "shared\nencoder", "model")
    _box(ax, 4.75, 3.4, 1.0, 0.5, "CB head", "model", fs=7)
    _box(ax, 4.75, 2.0, 1.0, 0.5, "sarc head", "model", fs=7)
    _arrow(ax, 2.1, 3.2, 2.5, 2.9)
    _arrow(ax, 2.1, 2.2, 2.5, 2.5)
    _arrow(ax, 3.9, 2.95, 4.25, 3.35)
    _arrow(ax, 3.9, 2.45, 4.25, 2.05)
    ax.text(2.6, 1.2, "control: identical, but sarcasm\nlabels randomly permuted", fontsize=7, color=INK2, ha="center")
    ax.text(8.0, 4.3, "(b) Feature fusion", fontsize=9, fontweight="bold", ha="center")
    _box(ax, 6.4, 3.2, 1.6, 0.6, "comment x", "data", fs=7.5)
    _box(ax, 6.4, 2.0, 1.6, 0.9, "frozen sarcasm\nMuRIL (trained\non Swami)", "model", fs=7)
    _box(ax, 8.2, 3.2, 1.3, 0.6, "MC encoder\nh", "model", fs=7)
    _box(ax, 8.2, 2.0, 1.3, 0.7, "φ([p_sarc; e])\n769→64", "model", fs=7)
    _box(ax, 9.75, 2.6, 0.95, 0.8, "[h ; φ]\n→ CB head", "model", fs=7)
    _arrow(ax, 7.2, 3.2, 7.55, 3.2)
    _arrow(ax, 7.2, 2.0, 7.55, 2.0)
    _arrow(ax, 8.85, 3.1, 9.27, 2.8)
    _arrow(ax, 8.85, 2.1, 9.27, 2.4)
    save(fig, out, "fig04_sarcasm_components", registry=reg,
         caption="Two ways of injecting sarcasm knowledge learned from the human-annotated Swami et al. (2018) corpus. "
                 "(a) multitask learning with a shuffled-label control; (b) feature fusion with a separately trained, "
                 "frozen sarcasm model (its P(sarcastic) and pooled embedding).")


def early_workflow(out: Path, reg: list):
    fig, ax = _canvas(7.6, 2.6, 12, 3.2)
    for k, x in enumerate([1.0, 2.6, 4.2]):
        _box(ax, x, 2.3, 1.3, 0.6, f"turn u{k + 1}", "data", fs=7.5)
        _box(ax, x, 1.2, 1.3, 0.6, f"p{k + 1} = f(u≤{k + 1})", "model", fs=7.5, dashed=True)
        _arrow(ax, x, 2.0, x, 1.5)
    ax.text(5.3, 2.3, "…", fontsize=14, color=INK2)
    _box(ax, 7.1, 1.2, 2.0, 0.7, "a_t = max_{s≤t} p_s\n(or EMA)", "model", fs=7.5, dashed=True)
    _box(ax, 9.6, 1.2, 1.8, 0.7, "alarm at first t\nwith a_t ≥ τ_val", "eval", fs=7.5, dashed=True)
    _arrow(ax, 4.85, 1.2, 6.1, 1.2)
    _arrow(ax, 8.1, 1.2, 8.7, 1.2)
    ax.text(6, 0.25, "Causal: no future turn is visible. τ chosen on validation conversations. Metrics: delay, false-alarm rate, ERDE.",
            ha="center", fontsize=7.5, color=INK2)
    save(fig, out, "fig05_early_detection_workflow", registry=reg,
         caption="Causal early-warning procedure. It was evaluated only on CONDA (English in-game chat) as a proxy, "
                 "because no openly available Hinglish conversation dataset has turn-level cyberbullying labels.")


def methodology(out: Path, reg: list):
    fig, ax = _canvas(7.6, 3.0, 12, 3.8)
    row1 = ["Audit legacy\ncode & data", "Verify & acquire\ndatasets", "Leakage-controlled\nsplits", "Baselines\n(majority, TF-IDF)"]
    row2 = ["Transformers\n×3 seeds", "Sarcasm variants\n(MTL / control / fusion)", "Calibration, CIs,\npaired tests", "SHAP + faithfulness;\nearly-warning proxy"]
    for i, t in enumerate(row1):
        _box(ax, 1.4 + i * 3.0, 2.9, 2.4, 0.8, t, "data" if i < 3 else "model", fs=7.5)
        if i:
            _arrow(ax, 1.4 + (i - 1) * 3.0 + 1.2, 2.9, 1.4 + i * 3.0 - 1.2, 2.9)
    for i, t in enumerate(row2):
        _box(ax, 1.4 + i * 3.0, 1.2, 2.4, 0.8, t, "model" if i < 2 else "eval", fs=7.5)
        if i:
            _arrow(ax, 1.4 + (i - 1) * 3.0 + 1.2, 1.2, 1.4 + i * 3.0 - 1.2, 1.2)
    _arrow(ax, 10.4, 2.5, 1.4, 1.6)
    save(fig, out, "fig06_methodology", registry=reg, caption="Experimental methodology.")


MERMAID = {
    "system_architecture.mmd": """flowchart LR
  A[Hinglish comment] --> B[Normalisation]
  B --> C[MuRIL encoder]
  C --> D[Cyberbullying-type head]
  C --> E[Aux sarcasm head - MTL]
  B --> F[Frozen sarcasm model] -->|fusion features| D
  D --> G[Temperature-scaled probabilities] --> H[SHAP + faithfulness]
  S[Conversation stream - CONDA proxy]:::proxy -.-> W[Causal early warning]:::proxy
  classDef proxy stroke-dasharray: 5 5
""",
    "early_detection.mmd": """flowchart LR
  u1[turn 1] --> p1[p1=f(u<=1)]
  u2[turn 2] --> p2[p2=f(u<=2)]
  ut[turn t] --> pt[pt=f(u<=t)]
  p1 & p2 & pt --> a[a_t = running max] --> al{a_t >= tau_val?} -->|yes| alarm[alarm at t]
""",
}


def all_diagrams(out: Path, reg: list, mermaid_dir: Path):
    system_architecture(out, reg)
    data_pipeline(out, reg)
    model_architecture(out, reg)
    sarcasm_fusion(out, reg)
    early_workflow(out, reg)
    methodology(out, reg)
    mermaid_dir.mkdir(parents=True, exist_ok=True)
    for k, v in MERMAID.items():
        (mermaid_dir / k).write_text(v)
