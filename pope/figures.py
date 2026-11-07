"""Build the six numerical manuscript figures from explicit paper data.

This module contains plotting only. Reported Monte Carlo values live in data/;
exact curves are recomputed from the closed-form and Riccati calculations.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .core import FINE, FXY, GMM_BED, gaussian_path_mgf, tempered_on
from .gaussian import (DIMS, ETA, MU, V0, VC, exact_ess_1d, ess1, kl_gauss,
                       mixture_mom, naive_law_and_certificate, naive_law_cfg,
                       target_cfg, var_logw_1d, zhat)
from .style import (CORRECTED, EXACT, GAUSSIAN, GRAY, LIGHT_GRAY, NAIVE, ORANGE,
                    PURPLE, panel_title, save_figure, threshold_band, use_style)

GMM_WSTAR = (1.82, 1.85)
GAUSSIAN_WSTAR = 1.9539


def _table(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def _normal_pdf(x, mean, var):
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


def gaussian_verification(data, out):
    saved = np.load(data / "gaussian_verification.npz")
    x, wn = saved["x"], saved["wn"]
    mc_t, mc_z = saved["mc_t"], saved["mc_z"]
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
    save_figure(fig, out, "gaussian_verification")
    plt.close(fig)


def early_stopping(data, out):
    early = _table(data / "early_stopping.csv")
    high = _table(data / "high_guidance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), layout="constrained")
    ax = axes[0]
    for vc, marker, color in zip([0.25, 0.05, 0.01, 0.001],
                                 ["o", "s", "^", "D"],
                                 [GAUSSIAN, ORANGE, CORRECTED, PURPLE]):
        rows = early[early["vc"] == vc]
        order = np.argsort(rows["t_min"])
        ax.plot(rows["t_min"][order], rows["w_star"][order], marker=marker,
                ms=3.8, color=color, mfc="white", mew=1.0,
                label=rf"$v_c={vc:g}$")
    ax.set_xlabel(r"Early-stopping time $t_{\min}$")
    ax.set_ylabel(r"Critical scale $w^*(t_{\min})$")
    ax.set_xlim(-0.003, 0.103)
    ax.set_ylim(1.25, 2.24)
    panel_title(ax, "a", "Terminal-IS threshold")
    ax.legend(ncol=2, loc="lower right", columnspacing=1.1)

    ax = axes[1]
    ax.semilogy(high["w"], high["lambda_star"] - 1, "o-", color=GAUSSIAN,
                mfc="white", mew=1.1)
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel(r"First-moment margin $\lambda^*(w)-1$")
    ax.set_xticks(high["w"])
    ax.set_xlim(2, 53)
    panel_title(ax, "b", "Margin above the first moment")
    ax.text(0.04, 0.07, "Mean and variance axes agree", transform=ax.transAxes,
            color=GRAY, fontsize=6.8)
    save_figure(fig, out, "early_stopping")
    plt.close(fig)


def ess_degeneracy(data, out):
    saved = np.load(data / "gaussian_ess_trajectory.npz")
    tgrid = saved["t"]
    summary = _table(data / "gaussian_cfg_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), layout="constrained")
    ax = axes[0]
    specs = [("two", "Two experts, $\\beta=1/2$", GAUSSIAN),
             ("w1.5", "CFG $w=1.5$", ORANGE),
             ("w2", "CFG $w=2$", PURPLE),
             ("w3", "CFG $w=3$", NAIVE)]
    for key, label, color in specs:
        ax.plot(tgrid, saved[key], color=color, lw=1.45, label=label)
    ax.plot(0, float(saved["ess_star_two"]), "*", color=GAUSSIAN, ms=7.0,
            clip_on=False)
    ax.plot(0, float(saved["ess_star_w1_5"]), "*", color=ORANGE, ms=7.0,
            clip_on=False)
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
    ax.axvspan(GAUSSIAN_WSTAR, 5.05, color=NAIVE, alpha=0.055, lw=0)
    ax.plot(wgrid, exact, color=EXACT, lw=1.65,
            label=r"Exact $(\mathbb{E}W)^2/\mathbb{E}[W^2]$")
    ax.plot(wgrid, proxy, color=GRAY, lw=1.25, ls="--",
            label=r"Lognormal proxy $e^{-\mathrm{Var}\,\log W}$")
    ax.plot(summary["w"], summary["ess_mc"], "o", color=NAIVE, ms=4.5,
            mfc="white", mew=1.2, label=r"Monte Carlo, $N=6\times10^4$")
    ax.axvline(GAUSSIAN_WSTAR, color=GAUSSIAN, ls=(0, (1.5, 2)), lw=1.0)
    ax.annotate("$w^*\\approx1.95$\n$\\mathbb{E}[W^2]=\\infty$ beyond",
                xy=(GAUSSIAN_WSTAR, 0.40), xytext=(2.35, 0.72), fontsize=6.7,
                color=GRAY, ha="left",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7,
                                shrinkA=3, shrinkB=3))
    ax.set_xlim(1.0, 5.05)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel("Terminal ESS/$N$")
    panel_title(ax, "b", "Critical guidance scale")
    ax.legend(loc="lower left", fontsize=6.15)
    save_figure(fig, out, "ess_degeneracy")
    plt.close(fig)


def dimension_scaling(data, out):
    product, meanfield = {}, {}
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
    thresholds = _table(data / "dimension_thresholds.csv")

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), layout="constrained")
    colors = {1.25: GAUSSIAN, 1.5: ORANGE, 1.75: CORRECTED}
    right_axes = []
    for ax, values, title in [(axes[0], product, "Product bed"),
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
    right_axes[0].set_ylabel(r"$-\log_{10}(\mathrm{ESS}/N)$")
    right_axes[1].set_ylabel(r"$-\log_{10}(\mathrm{ESS}/N)$")
    style_handles = [Line2D([0], [0], color=EXACT, lw=1.3, ls="-", label="KL"),
                     Line2D([0], [0], color=EXACT, lw=1.3, ls="--",
                            label=r"$-\log_{10}\mathrm{ESS}$")]
    axes[1].legend(handles=style_handles, loc="lower right", fontsize=5.9)

    ax = axes[2]
    ax.plot(thresholds["d"], thresholds["w_star"], "o-", color=GAUSSIAN,
            mfc="white", mew=1.0, label=r"Mean-field $w^*(d)$")
    ax.plot(DIMS, 4.1 * np.sqrt(DIMS), "--", color=EXACT, lw=1.15,
            label=r"$4.1\sqrt{d}$")
    ax.axhline(GAUSSIAN_WSTAR, color=GRAY, ls=(0, (1.5, 2)), lw=1.0,
               label=r"Product bed: $1.95$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel(r"Critical guidance scale $w^*$")
    panel_title(ax, "c", "Threshold scaling")
    ax.legend(loc="upper left", fontsize=5.9)
    save_figure(fig, out, "dimension_scaling")
    plt.close(fig)


def _exact_gaussian_ess_2d(wgrid, M=500):
    out = []
    for w in wgrid:
        e1x, _ = gaussian_path_mgf(2.25, 0.25, 0.8, w, 1.0, M=M)
        e1y, _ = gaussian_path_mgf(2.25, 0.25, 0.0, w, 1.0, M=M)
        e2x, okx = gaussian_path_mgf(2.25, 0.25, 0.8, w, 2.0, M=M)
        e2y, oky = gaussian_path_mgf(2.25, 0.25, 0.0, w, 2.0, M=M)
        ew2 = e2x * e2y if okx and oky else np.inf
        out.append((e1x * e1y) ** 2 / ew2 if np.isfinite(ew2) else 0.0)
    return np.asarray(out)


def gmm_summary(data, out):
    rows = _table(data / "gmm_summary.csv")
    ws = rows["w"]
    lo, hi = GMM_WSTAR
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), layout="constrained")
    ax = axes[0]
    wgrid = np.linspace(0.45, 2.05, 34)
    ess_exact = _exact_gaussian_ess_2d(wgrid)
    ax.semilogy(wgrid, np.maximum(ess_exact, 8e-4), "-", color=GAUSSIAN,
                lw=1.6, label="Gaussian exact")
    ax.semilogy(ws, rows["gaussian_ess_mc"], "o", color=GAUSSIAN, ms=4.2,
                mfc="white", mew=1.1, label="Gaussian terminal IS")
    ax.semilogy(ws, rows["gmm_ess_mean"], "s", color=NAIVE, ms=4.2,
                mfc="white", mew=1.1, label="GMM terminal IS")
    ax.axvline(GAUSSIAN_WSTAR, color=GAUSSIAN, ls=(0, (1.5, 2)), lw=0.9)
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
    ax.plot(ws, rows["gmm_ew_exact"], "-", color=EXACT, lw=1.55,
            label=r"Exact $\hat Z_0/\hat Z_1$")
    ax.plot(ws, rows["gmm_ew_mean"], "s", color=NAIVE, ms=4.2, mfc="white",
            mew=1.1, label="GMM terminal IS")
    threshold_band(ax, lo, hi)
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel(r"Mean weight $\mathbb{E}[W]$")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 3.1)
    ax.set_ylim(0.4, 20)
    panel_title(ax, "b", "First-moment accuracy outlives finite variance")
    ax.legend(loc="upper left", fontsize=6.4)
    save_figure(fig, out, "gmm2d_degeneracy")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.1), layout="constrained")
    ax.axhspan(0.07, 0.09, color=LIGHT_GRAY, alpha=0.75, lw=0,
               label="Histogram floor (0.07--0.09)")
    ax.plot(ws, rows["tv_naive"], "s--", color=NAIVE, lw=1.25, mfc="white",
            mew=1.1, ms=4.4, label="Naive")
    ax.plot(ws, rows["tv_corrected"], "s-", color=CORRECTED, lw=1.6, ms=4.4,
            label="Terminal-weight corrected")
    threshold_band(ax, lo, hi)
    ax.text(hi + 0.045, 0.282, r"$w^*_{\rm GMM}\in(1.82,1.85)$",
            color=NAIVE, fontsize=6.7, ha="left", va="top")
    ax.set_xlabel("Guidance weight $w$")
    ax.set_ylabel("Total variation to exact target")
    ax.set_ylim(0.0, 0.30)
    ax.set_xlim(0.4, 3.1)
    ax.set_xticks(ws)
    ax.legend(loc="upper left", fontsize=6.6)
    save_figure(fig, out, "gmm2d_target_tv")
    plt.close(fig)


def _hdr_contours(ax, q2d, levels=(0.5, 0.9, 0.99)):
    q = q2d.ravel()
    order = np.argsort(q)[::-1]
    cumulative = np.cumsum(q[order]) / q.sum()
    thresholds = sorted(q[order][np.searchsorted(cumulative, p)] for p in levels)
    ax.contour(FINE, FINE, q2d.T, levels=thresholds, colors=EXACT,
               linewidths=[0.55, 0.85, 1.2])


def gmm_densities(data, out):
    results = {}
    for w in (0.5, 1.5, 2.0):
        saved = np.load(data / f"gmm_density_w{w:g}.npz")
        results[w] = {key: saved[key] for key in ("x", "wn", "tv_n", "tv_w")}

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), layout="constrained")
    for j, w in enumerate((0.5, 1.5, 2.0)):
        result = results[w]
        q = tempered_on(FXY, GMM_BED, w, 0.0).reshape(len(FINE), len(FINE))
        for row in (0, 1):
            ax = axes[row, j]
            _hdr_contours(ax, q)
            rng = np.random.default_rng(4)
            if row == 0:
                idx = rng.choice(len(result["x"]), min(8000, len(result["x"])),
                                 replace=False)
                color, tv = NAIVE, float(result["tv_n"])
            else:
                idx = rng.choice(len(result["x"]), min(8000, len(result["x"])),
                                 replace=False, p=result["wn"])
                color, tv = CORRECTED, float(result["tv_w"])
            ax.scatter(result["x"][idx, 0], result["x"][idx, 1], s=2.2, c=color,
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
    save_figure(fig, out, "gmm2d_densities")
    plt.close(fig)


FIGURE_JOBS = {
    "gaussian": gaussian_verification,
    "early": early_stopping,
    "ess": ess_degeneracy,
    "dimension": dimension_scaling,
    "gmm": gmm_summary,
    "densities": gmm_densities,
}


def build_figures(data_dir, output_dir, only="all"):
    """Build selected figures; return the six stems that were requested."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    selected = list(FIGURE_JOBS) if only == "all" else [x.strip() for x in only.split(",")]
    unknown = set(selected) - set(FIGURE_JOBS)
    if unknown:
        raise ValueError(f"Unknown figure group(s): {', '.join(sorted(unknown))}")
    use_style()
    for name in selected:
        print(f"Building {name} ...", flush=True)
        FIGURE_JOBS[name](data_dir, output_dir)
    print(f"Figures written to {output_dir.resolve()}")
    return selected
