"""Unified Matplotlib style for all plots (blue & orange palette)."""
from __future__ import annotations

PALETTE = {
    "blue": "#1f77b4",   # modern blue
    "orange": "#ff7f0e", # modern orange
}

def apply_mpl_style():
    import matplotlib as mpl
    # Base sizes
    mpl.rcParams.update({
        "figure.figsize": (7, 4),
        "savefig.dpi": 200,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "axes.prop_cycle": mpl.cycler(color=[PALETTE["blue"], PALETTE["orange"]]),
    })
