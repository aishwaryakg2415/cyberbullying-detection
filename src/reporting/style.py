"""Shared figure style (validated palette; see dataviz reference palette)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated 3-slot categorical (all-pairs)
CLASSICAL = "#898781"                                # baselines rendered in muted ink
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 110,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "text.color": INK, "pdf.fonttype": 42,
})


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", ["#ffffff"] + SEQ)


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div_br", [DIV_NEG, DIV_MID, DIV_POS])


def save(fig, out_dir: Path, name: str, data=None, registry: list | None = None, caption: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    if data is not None:
        if hasattr(data, "to_csv"):
            data.to_csv(out_dir / f"{name}.csv", index=False)
        else:
            (out_dir / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float))
    if registry is not None:
        registry.append({"file": str((out_dir / name).as_posix()), "caption": caption})
