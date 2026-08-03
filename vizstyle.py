"""Shared matplotlib styling so every chart in the project reads as one system.

The categorical palette is validated for colour-vision deficiency and for
normal-vision separation (worst adjacent pair ΔE 9.1 CVD / 22.9 normal, both
above threshold). Two slots sit below 3:1 contrast on the surface, so charts
using them must carry visible labels or an accompanying table — the notebook
prints a DataFrame beside every figure for exactly that reason.

Series colours are assigned by *entity* and never cycled: `colour_for("random")`
returns the same hue no matter which chart it appears in or how many other
series are present.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_FAINT = "#8a8880"
GRID = "#e6e5e0"

# Fixed categorical order. Slot 1 blue, 2 orange, 3 aqua, 4 yellow.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Stable entity -> slot mapping, so an agent keeps its colour across figures.
_ASSIGNED = {
    "random": SERIES[3],
    "first": INK_FAINT,
    "ga": SERIES[0],
    "ga-flat": SERIES[1],
    "dqn": SERIES[2],
}


def colour_for(name: str) -> str:
    """Colour for a named agent or series, assigned once and reused."""
    if name not in _ASSIGNED:
        unused = [c for c in SERIES if c not in _ASSIGNED.values()]
        _ASSIGNED[name] = unused[0] if unused else INK_FAINT
    return _ASSIGNED[name]


def use_style():
    """Apply the project chart style. Call once per notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 120,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,          # grid behind the data, always
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_FAINT,
        "ytick.color": INK_FAINT,
        "xtick.labelcolor": INK_SOFT,
        "ytick.labelcolor": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SOFT,
        "lines.linewidth": 2.0,          # thin marks, not heavy
        "lines.markersize": 5,
        "font.size": 10,
        "figure.constrained_layout.use": True,
    })


def finish(ax, title=None, subtitle=None, xlabel=None, ylabel=None, legend=True):
    """Apply title/labels consistently. Subtitles carry the interpretation."""
    if title:
        ax.set_title(title)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=9.5, color=INK_SOFT, va="bottom",
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        handles = ax.get_legend_handles_labels()[0]
        # One series needs no legend box: the title already names it.
        if len(handles) > 1:
            ax.legend(loc="best")
    ax.grid(axis="x", visible=False)
    return ax


def new_figure(width=7.2, height=3.8):
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax
