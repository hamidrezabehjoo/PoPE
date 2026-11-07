#!/usr/bin/env python3
"""Shared publication style for the figures in the no-control manuscript."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe--Ito-derived semantic palette, shared with the TikZ flow diagram.
NAIVE = "#D55E00"       # vermillion: naive composed sampler
CORRECTED = "#009E73"   # teal: path-product / terminally corrected population
EXACT = "#222222"       # exact target or reference
GAUSSIAN = "#0072B2"    # Gaussian reference bed
ORANGE = "#E69F00"
PURPLE = "#8A63A8"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D9D9D9"


def use_style():
    """Set final-size typography; figures are not designed for later shrinking."""
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.7,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "lines.linewidth": 1.45,
        "lines.markersize": 4.2,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.24,
        "axes.grid": True,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.55,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    })


def panel_title(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", pad=4)


def threshold_band(ax, lo, hi, color=NAIVE):
    ax.axvspan(lo, hi, color=color, alpha=0.10, lw=0, zorder=0)
    ax.axvline(lo, color=color, ls=(0, (1.5, 2)), lw=0.9, zorder=1)
    ax.axvline(hi, color=color, ls=(0, (1.5, 2)), lw=0.9, zorder=1)


def save_figure(fig, outdir, stem):
    """Save one figure as vector PDF and high-resolution PNG."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        paths.append(path)
    return paths
