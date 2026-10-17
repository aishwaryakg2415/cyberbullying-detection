"""
=============================================================================
  Sarcasm-Aware Explainable Early Detection of Cyberbullying
  Module 7: SHAP Visualisation & Paper-Ready Figures
  IEEE Research Paper Implementation
=============================================================================
"""

import os
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud

from config import PLOTS_DIR, RESULTS_DIR, TOP_K_TOKENS

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(PLOTS_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# IEEE-style colour palette
COLORS = {
    "positive"    : "#C44E52",   # red — pushes toward cyberbullying
    "negative"    : "#4C72B0",   # blue — pushes away
    "neutral"     : "#AAAAAA",
    "sarcasm"     : "#DD8452",   # orange — sarcasm highlight
    "background"  : "#F7F7F7",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKEN ATTRIBUTION WATERFALL
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_waterfall(
    tokens      : List[str],
    shap_values : List[float],
    base_value  : float,
    prediction  : float,
    text        : str = "",
    title       : str = "SHAP Token Attribution",
    save_name   : str = "shap_waterfall",
    top_k       : int = TOP_K_TOKENS,
) -> str:
    """
    Plot a SHAP waterfall / force plot for token-level attributions.
    Positive bars (red) push towards cyberbullying;
    negative bars (blue) push against.
    """
    # Sort by absolute value and take top_k
    pairs     = list(zip(tokens, shap_values))
    pairs_s   = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    pairs_s   = sorted(pairs_s, key=lambda x: x[1])    # sort by value for waterfall

    tok_labels = [p[0] for p in pairs_s]
    vals       = [p[1] for p in pairs_s]
    colors     = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(4, len(tok_labels) * 0.42)))
    bars = ax.barh(tok_labels, vals, color=colors, edgecolor="white", height=0.7)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        x_pos = bar.get_width() + (0.002 if val >= 0 else -0.002)
        align = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", ha=align, fontsize=8)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value  →  Contribution to Cyberbullying Class", fontsize=11)
    ax.set_title(
        f"{title}\n"
        f"Base Value: {base_value:.4f}  |  Prediction: {prediction:.4f}\n"
        f'"{text[:80]}…"' if len(text) > 80 else f'"{text}"',
        fontsize=11, fontweight="bold", pad=12
    )

    # Legend
    pos_patch = mpatches.Patch(color=COLORS["positive"], label="Pushes → Cyberbullying")
    neg_patch = mpatches.Patch(color=COLORS["negative"], label="Pushes → Benign")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP waterfall saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. SARCASM VS BULLY QUADRANT PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_sarcasm_quadrant(
    bully_probs   : List[float],
    sarcasm_scores: List[float],
    labels        : List[int],
    save_name     : str = "sarcasm_quadrant",
) -> str:
    """
    Four-quadrant scatter plot:
    X-axis: sarcasm score, Y-axis: cyberbullying probability.
    Colour-coded by true label.
    Quadrants:
      Q1 (high sarcasm, high bully) = "Covert Bully" → highest risk
      Q2 (low sarcasm, high bully)  = "Overt Bully"
      Q3 (low sarcasm, low bully)   = "Safe"
      Q4 (high sarcasm, low bully)  = "Benign Sarcasm"
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    lbl_arr = np.array(labels)
    bp_arr  = np.array(bully_probs)
    ss_arr  = np.array(sarcasm_scores)

    # Colour by label
    c_map = {0: "#4C72B0", 1: "#C44E52"}
    for lbl, name in [(0, "Non-cyberbullying"), (1, "Cyberbullying")]:
        idx = lbl_arr == lbl
        ax.scatter(ss_arr[idx], bp_arr[idx], c=c_map[lbl],
                   alpha=0.7, s=60, label=name, edgecolors="white", lw=0.5)

    # Quadrant lines
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.axvline(0.5, color="black", ls="--", lw=1)

    # Quadrant labels
    kwargs = dict(fontsize=10, alpha=0.55, fontweight="bold")
    ax.text(0.75, 0.92, "Covert Bully\n(High Risk)",   transform=ax.transAxes,
            ha="center", color="#C44E52", **kwargs)
    ax.text(0.25, 0.92, "Overt Bully",                  transform=ax.transAxes,
            ha="center", color="#8B0000", **kwargs)
    ax.text(0.75, 0.08, "Benign Sarcasm",               transform=ax.transAxes,
            ha="center", color="#4C72B0", **kwargs)
    ax.text(0.25, 0.08, "Safe",                         transform=ax.transAxes,
            ha="center", color="#2c7bb6", **kwargs)

    ax.set_xlabel("Sarcasm Score (LLM Engine)", fontsize=12)
    ax.set_ylabel("Cyberbullying Probability (RoBERTa Classifier)", fontsize=12)
    ax.set_title(
        "Sarcasm–Cyberbullying Risk Quadrant Analysis\n"
        "(SACENet Decision Space)", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Quadrant plot saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. WORDCLOUD OF TOP BULLY TOKENS
# ─────────────────────────────────────────────────────────────────────────────

def plot_token_wordcloud(
    token_scores  : dict,   # {token: shap_value}
    title         : str = "Top Cyberbullying Tokens (SHAP)",
    save_name     : str = "token_wordcloud",
    mode          : str = "positive",   # "positive" | "negative"
) -> str:
    """Word cloud sized by |SHAP value|, coloured by direction."""
    filtered = {
        tok: abs(val)
        for tok, val in token_scores.items()
        if (val > 0 if mode == "positive" else val < 0)
        and len(tok.strip()) > 1
        and not tok.startswith("Ġ")   # remove BPE artefacts
    }
    if not filtered:
        logger.warning("No tokens to plot for wordcloud.")
        return ""

    def colour_fn(*args, **kwargs):
        return COLORS["positive"] if mode == "positive" else COLORS["negative"]

    wc = WordCloud(
        width=900, height=450, background_color="white",
        max_words=80, color_func=colour_fn,
        prefer_horizontal=0.85,
    ).generate_from_frequencies(filtered)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    direction = "→ Cyberbullying" if mode == "positive" else "→ Benign"
    ax.set_title(f"{title} ({direction})", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{save_name}_{mode}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Wordcloud saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE ARCHITECTURE DIAGRAM (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_architecture_diagram(save_name: str = "sacenet_architecture") -> str:
    """Generate a clean SACENet architecture diagram for the paper."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    def box(cx, cy, w, h, text, color="#4C72B0", fontsize=9, text_color="white"):
        rect = plt.Rectangle((cx - w/2, cy - h/2), w, h,
                              fc=color, ec="white", lw=2, zorder=3, alpha=0.9)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold", zorder=4, wrap=True,
                multialignment="center")

    def arrow(x1, y1, x2, y2, color="#555555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Input
    box(7, 7.3, 3, 0.7, "INPUT TEXT\n(Social Media Post)", "#2c3e50", fontsize=10)

    # Two branches
    arrow(7, 6.95, 4.5, 6.2)
    arrow(7, 6.95, 9.5, 6.2)

    # Left: Sarcasm branch
    box(4.5, 5.8, 3, 0.7, "LLM Sarcasm Engine\n(Gemini / GPT / BERT)", "#DD8452")
    arrow(4.5, 5.45, 4.5, 4.7)
    box(4.5, 4.35, 3, 0.6, "Sarcasm Score\n+ Indicators", "#e67e22")

    # Right: Classification branch
    box(9.5, 5.8, 3, 0.7, "Twitter-RoBERTa\nCyberbullying Classifier", "#4C72B0")
    arrow(9.5, 5.45, 9.5, 4.7)
    box(9.5, 4.35, 3, 0.6, "P(Cyberbullying)\n[0, 1]", "#2980b9")

    # Fusion
    arrow(4.5, 4.05, 7, 3.4)
    arrow(9.5, 4.05, 7, 3.4)
    box(7, 3.1, 3.5, 0.6, "Sarcasm-Aware Score Fusion\n(Dynamic Boost)", "#55A868")

    # Decision
    arrow(7, 2.8, 7, 2.1)
    box(7, 1.8, 3, 0.6, "Classification Decision\n(Cyberbullying / Benign)", "#2c3e50")

    # SHAP branch
    arrow(7, 1.5, 7, 0.9)
    box(7, 0.6, 3.5, 0.6, "SHAP Explainability\n(Token-Level Attributions)", "#8172B3")

    ax.set_title(
        "SACENet — Sarcasm-Aware Cyberbullying Early-detection Network\n"
        "Architecture Overview", fontsize=14, fontweight="bold", pad=15
    )
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Architecture diagram saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLATFORM-WISE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_platform_distribution(stats: dict, save_name: str = "platform_distribution") -> str:
    """Stacked bar chart of dataset distribution by platform."""
    platforms = list(stats.get("platform_distribution", {}).keys())
    counts    = list(stats.get("platform_distribution", {}).values())

    if not platforms:
        logger.warning("No platform data available.")
        return ""

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(platforms, counts, color=list(COLORS.values())[:len(platforms)],
                  edgecolor="white", width=0.55)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(cnt), ha="center", fontsize=10)

    ax.set_xlabel("Platform", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Multi-Platform Dataset Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Platform distribution plot saved: {path}")
    return path


if __name__ == "__main__":
    # Demo: architecture diagram
    path = plot_architecture_diagram()
    print(f"Architecture diagram: {path}")
