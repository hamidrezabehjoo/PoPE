#!/usr/bin/env python3
"""Figures for the controlled-sampler and early-stopping experiments."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})
COLORS = {
    "naive": "#7f7f7f",
    "terminal IS": "#d62728",
    "exact Doob": "#1f77b4",
    "approx Doob IS": "#2ca02c",
}
LABELS = {
    "naive": "naive",
    "terminal IS": "terminal IS",
    "exact Doob": "exact Doob",
    "approx Doob IS": r"approx. Doob IS ($\rho=0.8$)",
}


def controlled_samplers():
    cfg = pd.read_csv(OUT / "controlled_gaussian_cfg.csv")
    df = pd.read_csv(OUT / "controlled_2d.csv")
    gmm = df[(df.bed == "GMM") & df.w.isin([2.0, 3.0])]
    methods = ["naive", "terminal IS", "exact Doob", "approx Doob IS"]

    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.25))
    a = ax[0]
    a.plot(cfg.w, np.abs(cfg.mean_error), "o-", color=COLORS["exact Doob"],
           label="absolute mean error")
    a.plot(cfg.w, np.abs(cfg.variance_error), "s--", color="#9467bd",
           label="absolute variance error")
    a.axvline(1.9539, color="black", ls=":", lw=1.2, label=r"$w^*\approx1.95$")
    a.set_yscale("log")
    a.set_xlabel("guidance $w$")
    a.set_ylabel("absolute target-moment error")
    a.set_title("(a) Exact Gaussian Doob control")
    a.legend(frameon=False, loc="upper left")
    a.text(0.97, 0.06, "ESS/$N=1$ at every $w$", transform=a.transAxes,
           ha="right", va="bottom")

    x = np.arange(2)
    width = 0.19
    for panel, metric, ylabel, title in [
        (1, "mmd2", r"biased MMD$^2$", "(b) GMM: kernel discrepancy"),
        (2, "sliced_w1", "sliced $W_1$", "(c) GMM: sliced Wasserstein"),
    ]:
        a = ax[panel]
        for j, method in enumerate(methods):
            vals = [float(gmm[(gmm.w == w) & (gmm.method == method)][metric].iloc[0])
                    for w in [2.0, 3.0]]
            a.bar(x + (j - 1.5) * width, vals, width,
                  color=COLORS[method], label=LABELS[method])
        a.set_xticks(x, ["$w=2$", "$w=3$"])
        a.set_ylabel(ylabel)
        a.set_title(title)
        if metric == "mmd2":
            a.set_yscale("log")
        if panel == 2:
            a.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "controlled_samplers.pdf", bbox_inches="tight")
    fig.savefig(OUT / "controlled_samplers.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def early_stopping():
    early = pd.read_csv(OUT / "early_stopping_final.csv")
    high = pd.read_csv(OUT / "lambda_star_high_guidance_refined.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.25))

    a = ax[0]
    for vc, marker in zip([0.25, 0.05, 0.01, 0.001], ["o", "s", "^", "d"]):
        d = early[early.vc == vc].sort_values("t_min")
        a.plot(d.t_min, d.w_star, marker=marker, ms=4,
               label=rf"$v_c={vc:g}$")
    a.set_xlabel(r"early-stopping time $t_{\min}$")
    a.set_ylabel(r"critical scale $w^*(t_{\min})$")
    a.set_title("(a) Terminal-IS threshold")
    a.legend(frameon=False, ncol=2)

    a = ax[1]
    a.semilogy(high.w, high.lambda_star - 1, "o-", color="#1f77b4")
    a.set_xlabel("guidance $w$")
    a.set_ylabel(r"first-moment margin $\lambda^*(w)-1$")
    a.set_title("(b) Margin above the first moment")
    a.set_xticks(high.w)
    a.text(0.04, 0.08, "mean and variance axes agree",
           transform=a.transAxes, color="0.25")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "early_stopping.pdf", bbox_inches="tight")
    fig.savefig(OUT / "early_stopping.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    controlled_samplers()
    early_stopping()
    print("wrote controlled_samplers.pdf and early_stopping.pdf")
