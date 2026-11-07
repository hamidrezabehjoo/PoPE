#!/usr/bin/env python3
"""Rebuild the paper's GMM summary figures from the reported author-run table.

No new Monte Carlo runs are performed.  The MC markers and normalizer ratios are
the values reported in the paper's GMM table (N=6e4, M=2000, except the seed
counts documented there).  The Gaussian ESS reference is recomputed from the
exact discrete Gaussian path MGF.  The GMM critical-scale band is supplied from
the deterministic Feynman--Kac domain-refinement scan.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gmm2d import gaussian_path_mgf

C_G, C_M, C_EX = "#2166ac", "#b2182b", "#4393c3"

# Author-run values reported in the manuscript table.
WS = np.array([0.5, 1.0, 1.5, 1.75, 2.0, 3.0])
GAUSS_ESS_MC = np.array([0.902, 1.000, 0.781, 0.551, 0.159, 0.005])
GMM_ESS_MC = np.array([0.793, 1.000, 0.480, 0.200, 0.070, 0.008])
GMM_EW_MC = np.array([0.533, 1.000, 1.970, 2.730, 3.800, 12.900])
GMM_EW_EXACT = np.array([0.533, 1.000, 1.962, 2.746, 3.836, 14.280])
GMM_TV_NAIVE = np.array([0.210, 0.075, 0.123, 0.154, 0.174, 0.227])
GMM_TV_CORRECTED = np.array([0.092, 0.075, 0.087, 0.121, 0.154, 0.257])


def exact_gaussian_ess(wgrid, M=500):
    out = []
    for w in wgrid:
        e1x, _ = gaussian_path_mgf(2.25, 0.25, 0.8, w, 1.0, M=M)
        e1y, _ = gaussian_path_mgf(2.25, 0.25, 0.0, w, 1.0, M=M)
        e2x, okx = gaussian_path_mgf(2.25, 0.25, 0.8, w, 2.0, M=M)
        e2y, oky = gaussian_path_mgf(2.25, 0.25, 0.0, w, 2.0, M=M)
        ew2 = e2x * e2y if okx and oky else np.inf
        out.append((e1x * e1y) ** 2 / ew2 if np.isfinite(ew2) else 0.0)
    return np.asarray(out)


def threshold_band(ax, lo, hi):
    ax.axvspan(lo, hi, color=C_M, alpha=0.10, lw=0)
    ax.axvline(lo, color=C_M, ls=":", lw=1.0)
    ax.axvline(hi, color=C_M, ls=":", lw=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gmm-lo", type=float, required=True)
    ap.add_argument("--gmm-hi", type=float, required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    band_label = rf"$w^*_{{\rm GMM}}\in({args.gmm_lo:.2f},{args.gmm_hi:.2f})$"

    wgrid = np.linspace(0.45, 2.15, 35)
    ess_exact = exact_gaussian_ess(wgrid)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    ax.semilogy(wgrid, np.maximum(ess_exact, 1e-4), "-", color=C_EX, lw=2,
                label=r"Gaussian exact $(\mathbb{E}W)^2/\mathbb{E}[W^2]$")
    ax.semilogy(WS, GAUSS_ESS_MC, "o", color=C_G, ms=6, mfc="white", mew=1.6,
                label=r"Gaussian terminal IS ($N=6\times10^4$)")
    ax.semilogy(WS, GMM_ESS_MC, "s", color=C_M, ms=6, mfc="white", mew=1.6,
                label=r"GMM terminal IS ($N=6\times10^4$)")
    ax.axvline(1.9539, color=C_G, ls=":", lw=1.3)
    threshold_band(ax, args.gmm_lo, args.gmm_hi)
    ax.text(args.gmm_lo, 1.55, band_label, color=C_M, fontsize=8.5,
            ha="right", va="top", rotation=90)
    ax.text(1.9539, 1.55, r" $w^*_{\rm Gaussian}=1.95$", color=C_G,
            fontsize=8.5, ha="left", va="top", rotation=90)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("terminal ESS/$N$")
    ax.set_ylim(1e-3, 2.2)
    ax.set_xlim(0.4, 3.1)
    ax.legend(fontsize=7.3, loc="lower left", framealpha=0.95)
    ax.set_title("(a) Finite-$N$ ESS is not a threshold estimator", fontsize=9.5)

    ax = axes[1]
    ax.plot(WS, GMM_EW_MC, "s", color=C_M, ms=6, mfc="white", mew=1.6,
            label="GMM terminal IS")
    ax.plot(WS, GMM_EW_EXACT, "-", color=C_M, lw=1.7, alpha=0.75,
            label=r"GMM exact $\hat Z_0/\hat Z_1$")
    threshold_band(ax, args.gmm_lo, args.gmm_hi)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel(r"mean weight $\mathbb{E}[W]$")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 3.1)
    ax.legend(fontsize=7.8, loc="upper left")
    ax.set_title("(b) First-moment accuracy can outlive finite variance", fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"gmm2d_degeneracy.{ext}"),
                    dpi=150 if ext == "png" else None)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.axhspan(0.07, 0.09, color="0.85", alpha=0.65, lw=0,
               label="histogram floor (0.07--0.09)")
    ax.plot(WS, GMM_TV_NAIVE, "s--", color=C_M, lw=1.3, mfc="white", ms=6,
            label="GMM naive")
    ax.plot(WS, GMM_TV_CORRECTED, "s-", color=C_M, lw=1.9, ms=6,
            label="GMM terminal-weight corrected")
    threshold_band(ax, args.gmm_lo, args.gmm_hi)
    ax.text(args.gmm_hi + 0.03, 0.275, band_label, color=C_M, fontsize=9,
            ha="left", va="top")
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("TV to exact tempered target")
    ax.set_ylim(0.0, 0.30)
    ax.set_xlim(0.4, 3.1)
    ax.legend(fontsize=8.3, loc="upper left")
    ax.set_title("Exact target in law; finite-$N$ weights control accuracy", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"gmm2d_target_tv.{ext}"),
                    dpi=150 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
