#!/usr/bin/env python3
"""Regenerate all raster/vector figures in the no-control manuscript.

The plotting layout and semantic colors are shared across figures. Reported
numbers are read from the paper's saved outputs or recomputed with the existing
algorithms. The GMM density panels use the full N=60,000, M=2,000 settings and
cache each existing configuration so that styling passes do not rerun SMC.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.special import logsumexp

from paper_style import (CORRECTED, EXACT, GAUSSIAN, GRAY, LIGHT_GRAY, NAIVE,
                         ORANGE, PURPLE, panel_title, save_figure,
                         threshold_band, use_style)
import matplotlib.pyplot as plt

from dimension_scaling import (ETA, MU, V0, VC, ess1, kl_gauss,
                               naive_law_cfg, target_cfg)
from example1_gaussian import mixture_mom, naive_law_and_certificate, smc_example1, zhat
from gmm2d import (FINE, FXY, GMM_BED, gaussian_path_mgf, smc_cfg,
                   stable_seed, tempered_on, tv_to_target)
from gmm2d_paper_figures import (GAUSS_ESS_MC, GMM_ESS_MC, GMM_EW_EXACT,
                                 GMM_EW_MC, GMM_TV_CORRECTED, GMM_TV_NAIVE, WS)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output"
CACHE = DATA / "figure-cache"
FIGDIRS = [Path("/mnt/agents/output/figs"), DATA]
DIMS = np.array([1, 4, 16, 64, 256, 1024])

use_style()


def _normal_pdf(x, mean, var):
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


def gaussian_verification(refresh=False):
    """Figure 1: Gaussian data-end laws and exact normalizer chain."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "fig1.npz"
    if cache.exists() and not refresh:
        z = np.load(cache)
        x, wn, mc_t, mc_z = z["x"], z["wn"], z["mc_t"], z["mc_z"]
    else:
        r = smc_example1(N=60_000, M=2000, seed=0, return_samples=True)
        x, wn = r["x0"], r["wn"]
        mc_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mc_z = np.array([r["EW"], r["snaps"][0.25], r["snaps"][0.5],
                         r["snaps"][0.75], 1.0])
        np.savez(cache, x=x, wn=wn, mc_t=mc_t, mc_z=mc_z)

    h0, p0 = mixture_mom(0.0)
    mn, pn, _, _ = naive_law_and_certificate()
    xx = np.linspace(-3.2, 4.2, 800)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), layout="constrained")
    ax = axes[0]
    bins = np.linspace(-3.2, 4.2, 90)
    ax.hist(x, bins=bins, density=True, histtype="stepfilled", color=NAIVE,
            alpha=0.12, edgecolor=NAIVE, linewidth=0.65,
            label="Naive composed sampler")
    ax.hist(x, bins=bins, weights=wn, density=True, histtype="stepfilled",
            color=CORRECTED, alpha=0.14, edgecolor=CORRECTED, linewidth=0.65,
            label="Tilted path product")
    ax.plot(xx, _normal_pdf(xx, h0, p0), color=EXACT, lw=1.65,
            label=r"Exact target $m_0/\hat Z_0$")
    ax.plot(xx, _normal_pdf(xx, mn, pn), color=GRAY, lw=1.35, ls="--",
            label=r"Exact naive law $p^{\bar b}_0$")
    ax.set_xlim(-3.0, 4.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("Density")
    panel_title(ax, "a", "Data-end laws ($S=1$)")
    ax.legend(loc="upper left", fontsize=6.25)

    ax = axes[1]
    tt = np.linspace(0, 1, 300)
    ax.plot(tt, zhat(tt) / zhat(1.0), color=EXACT, lw=1.6,
            label=r"Exact $\hat Z_t/\hat Z_1$")
    ax.plot(mc_t, mc_z, "o", color=CORRECTED, ms=4.5, mfc="white",
            mew=1.25, label="Monte Carlo mean weight")
    ax.set_xlabel("Noising time $t$")
    ax.set_ylabel(r"$\hat Z_t/\hat Z_1$")
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(0.48, 1.025)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    panel_title(ax, "b", "Normalizer ratio along the path")
    ax.legend(loc="lower right")
    save_figure(fig, FIGDIRS, "gaussian_verification")
    plt.close(fig)


def early_stopping():
    """Figure 3: saved exact Gaussian early-stopping calculations."""
    early = pd.read_csv(DATA / "early_stopping_final.csv")
    high = pd.read_csv(DATA / "lambda_star_high_guidance_refined.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), layout="constrained")
    colors = [GAUSSIAN, ORANGE, CORRECTED, PURPLE]
    ax = axes[0]
    for vc, marker, color in zip([0.25, 0.05, 0.01, 0.001],
                                 ["o", "s", "^", "D"], colors):
        d = early[early.vc == vc].sort_values("t_min")
        ax.plot(d.t_min, d.w_star, marker=marker, ms=3.8, color=color,
                mfc="white", mew=1.0, label=rf"$v_c={vc:g}$")
    ax.set_xlabel(r"Early-stopping time $t_{\min}$")
    ax.set_ylabel(r"Critical scale $w^*(t_{\min})$")
    ax.set_xlim(-0.003, 0.103)
    ax.set_ylim(1.25, 2.24)
    panel_title(ax, "a", "Terminal-IS threshold")
    ax.legend(ncol=2, loc="lower right", columnspacing=1.1)

    ax = axes[1]
    ax.semilogy(high.w, high.lambda_star - 1, "o-", color=GAUSSIAN,
                mfc="white", mew=1.1)
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel(r"First-moment margin $\lambda^*(w)-1$")
    ax.set_xticks(high.w)
    ax.set_xlim(2, 53)
    panel_title(ax, "b", "Margin above the first moment")
    ax.text(0.04, 0.07, "Mean and variance axes agree",
            transform=ax.transAxes, color=GRAY, fontsize=6.8)
    save_figure(fig, FIGDIRS, "early_stopping")
    plt.close(fig)


def smc_cfg_1d(w, N=60_000, M=2000, seed=0, ess_times=None):
    """Existing one-dimensional CFG Euler--Maruyama diagnostic, with ESS traces."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    theta = 0.5 * w * (w - 1.0)
    p1 = (1 - w) / (V0 + 1.0) + w / (VC + 1.0)
    m1 = w * MU / (VC + 1.0) / p1
    x = m1 + np.sqrt(1.0 / p1) * rng.standard_normal(N)
    logw = np.zeros(N)
    ess = {1.0: 1.0}
    ess_set = {} if ess_times is None else {
        int(round((1 - float(t)) / dt)): float(t)
        for t in ess_times if float(t) < 1.0
    }
    for i in range(M):
        t = 1.0 - i * dt
        s0 = -x / (V0 + t)
        sc = -(x - MU) / (VC + t)
        logw += theta * (sc - s0) ** 2 * dt
        b = -((1 - w) * s0 + w * sc)
        x = x - b * dt + np.sqrt(dt) * rng.standard_normal(N)
        if i + 1 in ess_set:
            lw = logw - logsumexp(logw)
            ess[ess_set[i + 1]] = float(np.exp(-logsumexp(2 * lw))) / N
    lw = logw - logsumexp(logw)
    return {"x": x, "logw": logw,
            "ess": float(np.exp(-logsumexp(2 * lw))) / N,
            "ess_snap": ess}


def exact_ess_1d(w, M=500):
    e1, _ = gaussian_path_mgf(V0, VC, MU, w, 1.0, M=M)
    e2, ok = gaussian_path_mgf(V0, VC, MU, w, 2.0, M=M)
    return e1 ** 2 / e2 if ok else 0.0


def var_logw_1d(w, M=500):
    """Equation (varlogw), using the exact discrete Gaussian path covariance."""
    theta = 0.5 * w * (w - 1.0)
    if theta == 0:
        return 0.0
    dt = 1.0 / M
    ts = 1.0 - np.arange(M + 1) * dt
    v0t, vct = V0 + ts, VC + ts
    a = 1.0 / vct - 1.0 / v0t
    d = MU / vct
    p = (1 - w) / v0t + w / vct
    q = w * MU / vct
    p1 = (1 - w) / (V0 + 1.0) + w / (VC + 1.0)
    m = np.empty(M + 1)
    sig = np.empty(M + 1)
    m[0] = w * MU / (VC + 1.0) / p1
    sig[0] = 1.0 / p1
    logf = np.concatenate([[0.0], np.cumsum(np.log(np.abs(1 - p[:-1] * dt)))])
    for i in range(M):
        m[i + 1] = m[i] - (p[i] * m[i] - q[i]) * dt
        sig[i + 1] = (1 - p[i] * dt) ** 2 * sig[i] + dt
    ii = np.arange(M + 1)[:, None]
    jj = np.arange(M + 1)[None, :]
    mn = np.minimum(ii, jj)
    mx = np.maximum(ii, jj)
    cov = sig[mn] * np.exp(logf[mx] - logf[mn])
    kappa = theta * a[:-1] ** 2
    centered = m[:-1] - d[:-1] / a[:-1]
    integrand = (kappa[:, None] * kappa[None, :]
                 * (2 * cov[:-1, :-1] ** 2
                    + 4 * cov[:-1, :-1] * centered[:, None] * centered[None, :]))
    return float(integrand.sum() * dt ** 2)


def ess_degeneracy(refresh=False):
    """Figure 4: one-dimensional Gaussian trajectory and terminal ESS."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "fig4.npz"
    tgrid = np.linspace(0, 1, 41)
    if cache.exists() and not refresh:
        z = np.load(cache)
        traces = {key: z[key] for key in ("two", "w1.5", "w2", "w3")}
    else:
        two = smc_example1(N=60_000, M=2000, seed=0, ess_times=tgrid)
        traces = {"two": np.array([two["ess_snap"][float(t)] for t in tgrid])}
        for w, key in [(1.5, "w1.5"), (2.0, "w2"), (3.0, "w3")]:
            r = smc_cfg_1d(w, ess_times=tgrid)
            traces[key] = np.array([r["ess_snap"][float(t)] for t in tgrid])
            print(f"CFG w={w:g}: terminal ESS/N={r['ess']:.4f}", flush=True)
        np.savez(cache, t=tgrid, **traces)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), layout="constrained")
    ax = axes[0]
    specs = [("two", "Two experts, $\\beta=1/2$", GAUSSIAN),
             ("w1.5", "CFG $w=1.5$", ORANGE),
             ("w2", "CFG $w=2$", PURPLE),
             ("w3", "CFG $w=3$", NAIVE)]
    for key, label, color in specs:
        ax.plot(tgrid, traces[key], color=color, lw=1.45, label=label)
    ax.plot(0, 0.820, "*", color=GAUSSIAN, ms=7.0, clip_on=False)
    ax.plot(0, 0.863, "*", color=ORANGE, ms=7.0, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel(r"Noising time $t$ (sampling: $1\to0$)")
    ax.set_ylabel("ESS/$N$")
    panel_title(ax, "a", "Weight degeneracy along the trajectory")
    ax.legend(loc="upper left", ncol=2, fontsize=6.2, frameon=True,
              framealpha=0.90, edgecolor="none")

    ax = axes[1]
    wgrid = np.linspace(1.0, 5.0, 41)
    exact = np.array([exact_ess_1d(w) for w in wgrid])
    proxy = np.array([np.exp(-var_logw_1d(w)) for w in wgrid])
    ax.axvspan(1.9539, 5.05, color=NAIVE, alpha=0.055, lw=0)
    ax.plot(wgrid, exact, color=EXACT, lw=1.65,
            label=r"Exact $(\mathbb{E}W)^2/\mathbb{E}[W^2]$")
    ax.plot(wgrid, proxy, color=GRAY, lw=1.25, ls="--",
            label=r"Lognormal proxy $e^{-\mathrm{Var}\,\log W}$")
    mc_w = np.array([1.5, 2.0, 3.0, 5.0])
    mc_e = np.array([0.854, 0.312, 0.024, 0.006])
    ax.plot(mc_w, mc_e, "o", color=NAIVE, ms=4.5, mfc="white", mew=1.2,
            label=r"Monte Carlo, $N=6\times10^4$")
    ax.axvline(1.9539, color=GAUSSIAN, ls=(0, (1.5, 2)), lw=1.0)
    ax.annotate("$w^*\\approx1.95$\n$\\mathbb{E}[W^2]=\\infty$ beyond",
                xy=(1.9539, 0.40), xytext=(2.35, 0.72), fontsize=6.7,
                color=GRAY, ha="left",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=GRAY,
                                lw=0.7, shrinkA=3, shrinkB=3))
    ax.set_xlim(1.0, 5.05)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel("Terminal ESS/$N$")
    panel_title(ax, "b", "Critical guidance scale")
    ax.legend(loc="lower left", fontsize=6.15)
    save_figure(fig, FIGDIRS, "ess_degeneracy")
    plt.close(fig)


def dimension_scaling():
    """Figure 5: exact dimension-scaling curves, with no Monte Carlo."""
    product = {}
    meanfield = {}
    for w in (1.25, 1.5, 1.75):
        mn, vn = naive_law_cfg(V0, VC, MU, w)
        mt, vt = target_cfg(V0, VC, MU, w)
        kl1 = kl_gauss(mn, vn, mt, vt)
        _, e1 = ess1(V0, VC, MU, w)
        product[w] = (DIMS * kl1, -DIMS * np.log10(e1))
    for w in (1.25, 1.5):
        kls, essloss = [], []
        for d in DIMS:
            mud, vcd = MU / np.sqrt(d), V0 * (1 + ETA / np.sqrt(d))
            mn, vn = naive_law_cfg(V0, vcd, mud, w)
            mt, vt = target_cfg(V0, vcd, mud, w)
            _, e1 = ess1(V0, vcd, mud, w)
            kls.append(d * kl_gauss(mn, vn, mt, vt))
            essloss.append(-d * np.log10(e1))
        meanfield[w] = (np.array(kls), np.array(essloss))
    wstar = np.array([1.95, 6.18, 14.58, 31.40, 65.04, 132.32])

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), layout="constrained")
    colors = {1.25: GAUSSIAN, 1.5: ORANGE, 1.75: CORRECTED}
    right_axes = []
    for ax, values, title in [
            (axes[0], product, "Product bed"),
            (axes[1], meanfield, "Mean-field bed")]:
        twin = ax.twinx()
        right_axes.append(twin)
        twin.grid(False)
        for w, (kl, essloss) in values.items():
            ax.plot(DIMS, kl, color=colors[w], marker="o", ms=3.0,
                    label=rf"$w={w:g}$")
            twin.plot(DIMS, essloss, color=colors[w], ls="--", marker="s",
                      ms=2.8, mfc="white", mew=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        twin.set_yscale("log")
        ax.set_xlabel("Dimension $d$")
        panel_title(ax, "a" if ax is axes[0] else "b", title)
        ax.legend(loc="upper left", fontsize=6.1, handlelength=1.5)
    axes[0].set_ylabel(r"Naive KL $=d\,\mathrm{KL}_1$")
    axes[1].set_ylabel("Naive KL (total)")
    # Label the right axes explicitly without adding another legend entry.
    right_axes[0].set_ylabel(r"$-\log_{10}(\mathrm{ESS}/N)$")
    right_axes[1].set_ylabel(r"$-\log_{10}(\mathrm{ESS}/N)$")
    style_handles = [Line2D([0], [0], color=EXACT, lw=1.3, ls="-", label="KL"),
                     Line2D([0], [0], color=EXACT, lw=1.3, ls="--",
                            label=r"$-\log_{10}\mathrm{ESS}$")]
    axes[1].legend(handles=style_handles, loc="lower right", fontsize=5.9)

    ax = axes[2]
    ax.plot(DIMS, wstar, "o-", color=GAUSSIAN, mfc="white", mew=1.0,
            label=r"Mean-field $w^*(d)$")
    ax.plot(DIMS, 4.1 * np.sqrt(DIMS), "--", color=EXACT, lw=1.15,
            label=r"$4.1\sqrt{d}$")
    ax.axhline(1.9539, color=GRAY, ls=(0, (1.5, 2)), lw=1.0,
               label=r"Product bed: $1.95$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel(r"Critical guidance scale $w^*$")
    panel_title(ax, "c", "Threshold scaling")
    ax.legend(loc="upper left", fontsize=5.9)
    save_figure(fig, FIGDIRS, "dimension_scaling")
    plt.close(fig)


def exact_gaussian_ess_2d(wgrid, M=500):
    out = []
    for w in wgrid:
        e1x, _ = gaussian_path_mgf(2.25, 0.25, 0.8, w, 1.0, M=M)
        e1y, _ = gaussian_path_mgf(2.25, 0.25, 0.0, w, 1.0, M=M)
        e2x, okx = gaussian_path_mgf(2.25, 0.25, 0.8, w, 2.0, M=M)
        e2y, oky = gaussian_path_mgf(2.25, 0.25, 0.0, w, 2.0, M=M)
        ew2 = e2x * e2y if okx and oky else np.inf
        out.append((e1x * e1y) ** 2 / ew2 if np.isfinite(ew2) else 0.0)
    return np.asarray(out)


def gmm_summary():
    """Figures 6 and 7 from the reported full-run table and exact references."""
    lo, hi = 1.82, 1.85
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), layout="constrained")
    ax = axes[0]
    wgrid = np.linspace(0.45, 2.05, 34)
    ess_exact = exact_gaussian_ess_2d(wgrid)
    ax.semilogy(wgrid, np.maximum(ess_exact, 8e-4), "-", color=GAUSSIAN,
                lw=1.6, label=r"Gaussian exact")
    ax.semilogy(WS, GAUSS_ESS_MC, "o", color=GAUSSIAN, ms=4.2, mfc="white",
                mew=1.1, label=r"Gaussian terminal IS")
    ax.semilogy(WS, GMM_ESS_MC, "s", color=NAIVE, ms=4.2, mfc="white",
                mew=1.1, label=r"GMM terminal IS")
    ax.axvline(1.9539, color=GAUSSIAN, ls=(0, (1.5, 2)), lw=0.9)
    threshold_band(ax, lo, hi)
    ax.text(lo - 0.018, 1.70, r"$w^*_{\rm GMM}\in(1.82,1.85)$", color=NAIVE,
            fontsize=6.1, ha="right", va="top", rotation=90)
    ax.text(1.972, 1.70, r"$w^*_{\rm Gaussian}=1.95$", color=GAUSSIAN,
            fontsize=6.1, ha="left", va="top", rotation=90)
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel("Terminal ESS/$N$")
    ax.set_ylim(1e-3, 2.2)
    ax.set_xlim(0.4, 3.1)
    panel_title(ax, "a", "Finite-$N$ ESS is not a threshold estimator")
    ax.legend(loc="lower left", fontsize=6.15)

    ax = axes[1]
    ax.plot(WS, GMM_EW_EXACT, "-", color=EXACT, lw=1.55,
            label=r"Exact $\hat Z_0/\hat Z_1$")
    ax.plot(WS, GMM_EW_MC, "s", color=NAIVE, ms=4.2, mfc="white", mew=1.1,
            label="GMM terminal IS")
    threshold_band(ax, lo, hi)
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel(r"Mean weight $\mathbb{E}[W]$")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 3.1)
    ax.set_ylim(0.4, 20)
    panel_title(ax, "b", "First-moment accuracy outlives finite variance")
    ax.legend(loc="upper left", fontsize=6.4)
    save_figure(fig, FIGDIRS, "gmm2d_degeneracy")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.1), layout="constrained")
    ax.axhspan(0.07, 0.09, color=LIGHT_GRAY, alpha=0.75, lw=0,
               label="Histogram floor (0.07--0.09)")
    ax.plot(WS, GMM_TV_NAIVE, "s--", color=NAIVE, lw=1.25, mfc="white",
            mew=1.1, ms=4.4, label="Naive")
    ax.plot(WS, GMM_TV_CORRECTED, "s-", color=CORRECTED, lw=1.6, ms=4.4,
            label="Terminal-weight corrected")
    threshold_band(ax, lo, hi)
    ax.text(hi + 0.045, 0.282, r"$w^*_{\rm GMM}\in(1.82,1.85)$",
            color=NAIVE, fontsize=6.7, ha="left", va="top")
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel("Total variation to exact target")
    ax.set_ylim(0.0, 0.30)
    ax.set_xlim(0.4, 3.1)
    ax.set_xticks(WS)
    ax.legend(loc="upper left", fontsize=6.6)
    save_figure(fig, FIGDIRS, "gmm2d_target_tv")
    plt.close(fig)


def hdr_contours(ax, q2d, levels=(0.5, 0.9, 0.99)):
    q = q2d.ravel()
    order = np.argsort(q)[::-1]
    cumulative = np.cumsum(q[order]) / q.sum()
    thresholds = sorted(q[order][np.searchsorted(cumulative, p)] for p in levels)
    ax.contour(FINE, FINE, q2d.T, levels=thresholds, colors=EXACT,
               linewidths=[0.55, 0.85, 1.2])


def gmm_densities(refresh=False):
    """Figure 8 at the paper's full settings, using established full-driver seeds."""
    CACHE.mkdir(exist_ok=True)
    results = {}
    expected = {0.5: (0.210, 0.092), 1.5: (0.123, 0.087), 2.0: (0.174, 0.154)}
    for w in (0.5, 1.5, 2.0):
        cache = CACHE / f"fig8_w{w:g}.npz"
        if cache.exists() and not refresh:
            z = np.load(cache)
            results[w] = {key: z[key] for key in ("x", "wn", "tv_n", "tv_w")}
        else:
            r = smc_cfg(GMM_BED, w=w, N=60_000, M=2000,
                        seed=stable_seed("sweep", "M", w))
            tv_w = tv_to_target(r["x0"], r["wn"], GMM_BED, w)
            tv_n = tv_to_target(r["x0"], np.full(len(r["x0"]), 1 / len(r["x0"])),
                                GMM_BED, w)
            np.savez(cache, x=r["x0"], wn=r["wn"],
                                tv_n=tv_n, tv_w=tv_w)
            results[w] = {"x": r["x0"], "wn": r["wn"],
                          "tv_n": tv_n, "tv_w": tv_w}
        tv_n, tv_w = float(results[w]["tv_n"]), float(results[w]["tv_w"])
        print(f"GMM w={w:g}: naive TV={tv_n:.3f}, corrected TV={tv_w:.3f}",
              flush=True)
        if abs(tv_n - expected[w][0]) > 0.025 or abs(tv_w - expected[w][1]) > 0.025:
            raise RuntimeError(f"Full-run TV at w={w} is inconsistent with the manuscript")

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), layout="constrained")
    for j, w in enumerate((0.5, 1.5, 2.0)):
        r = results[w]
        q = tempered_on(FXY, GMM_BED, w, 0.0).reshape(len(FINE), len(FINE))
        for row in (0, 1):
            ax = axes[row, j]
            hdr_contours(ax, q)
            rng = np.random.default_rng(4)
            if row == 0:
                idx = rng.choice(len(r["x"]), min(8000, len(r["x"])), replace=False)
                color, tv = NAIVE, float(r["tv_n"])
            else:
                idx = rng.choice(len(r["x"]), min(8000, len(r["x"])),
                                 replace=False, p=r["wn"])
                color, tv = CORRECTED, float(r["tv_w"])
            ax.scatter(r["x"][idx, 0], r["x"][idx, 1], s=2.2, c=color,
                       alpha=0.24, edgecolors="none", rasterized=True)
            ax.plot(GMM_BED["mc"][:, 0], GMM_BED["mc"][:, 1], "+",
                    color=EXACT, ms=5.0, mew=1.0, zorder=5)
            ax.text(0.04, 0.96, f"TV = {tv:.3f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.7, color=EXACT,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                              alpha=0.78))
            ax.set_xlim(-3.4, 3.4)
            ax.set_ylim(-3.4, 3.4)
            ax.set_aspect("equal")
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])
            if row == 0:
                ax.set_title(f"({chr(97+j)}) $w={w:g}$", pad=4)
            else:
                ax.set_xlabel("$x_1$")
            if j == 0:
                ax.set_ylabel(("Naive\n" if row == 0 else "Corrected\n") + "$x_2$")
    save_figure(fig, FIGDIRS, "gmm2d_densities")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all",
                        help="comma-separated: gaussian,early,ess,dimension,gmm,densities")
    parser.add_argument("--refresh", action="store_true",
                        help="rerun cached full-size simulations")
    args = parser.parse_args()
    selected = set(args.only.split(",")) if args.only != "all" else {
        "gaussian", "early", "ess", "dimension", "gmm", "densities"}
    jobs = [("gaussian", gaussian_verification), ("early", early_stopping),
            ("ess", ess_degeneracy), ("dimension", dimension_scaling),
            ("gmm", gmm_summary), ("densities", gmm_densities)]
    for name, job in jobs:
        if name in selected:
            print(f"Building {name} figure(s)...", flush=True)
            job(refresh=args.refresh) if name in {"gaussian", "ess", "densities"} else job()
    print("Publication figures written to /mnt/agents/output/figs and repository output.")


if __name__ == "__main__":
    main()
