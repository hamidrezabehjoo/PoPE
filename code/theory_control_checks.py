#!/usr/bin/env python3
"""Theory-first controlled checks: SMC baselines and practical initialization.

All experiments use the existing Gaussian/GMM CFG beds.  No learned-control
or pretrained-model claim is made.  SMC uses the same naive proposal and path
potential as terminal IS, with systematic resampling whenever ESS/N < 1/2.
Standard-prior terminal IS includes p_1^{beta,S}/r_1 when ``corrected``;
standard-prior exact Doob includes q_1/r_1 when ``corrected``.  Omitting these
factors is reported separately as an initialization mismatch, not as exact IS.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from gmm2d import GAUSS_BED, GMM_BED, p0_pdf, pc_pdf, p0_score, pc_score, smc_cfg, stable_seed
from controlled_gmm import GridDoob2D, evaluate, write_csv
from controlled_gaussian import geometric_gaussian, vp_noise_end

OUT = Path(__file__).resolve().parents[1] / "output"


def normalize(logw):
    wn = np.exp(logw - logsumexp(logw))
    ess = float(1.0 / np.sum(wn * wn) / len(wn))
    return wn, ess


def systematic_resample(x, logw, rng):
    """Systematic resampling; returns resampled states and pre-resampling ESS."""
    wn, ess = normalize(logw)
    n = len(x)
    positions = (rng.random() + np.arange(n)) / n
    ind = np.searchsorted(np.cumsum(wn), positions, side="right")
    return x[ind], ess


def smc_resampling_cfg(bed, w, N=20_000, M=600, seed=0, threshold=0.5):
    """Feynman--Kac/SMC corrector with exact tempered initialization."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    th = 0.5 * w * (w - 1.0)
    # The same one-step exact tempered initialization as smc_cfg.
    # Keep this draw for seed compatibility with the reported runs.
    x = rng.normal(size=(N, 2))
    from gmm2d import gmm_sample
    x = gmm_sample(N, bed["m0"], bed["v0"], bed["w0"], 1.0, rng)
    log_init = w * (np.log(pc_pdf(x, bed, 1.0) + 1e-300)
                    - np.log(p0_pdf(x, bed, 1.0) + 1e-300))
    x, init_ess = systematic_resample(x, log_init, rng)
    logw = np.zeros(N)
    resample_times, pre_ess = [], []
    for i in range(M):
        t = 1.0 - i * dt
        s0, sc = p0_score(x, bed, t), pc_score(x, bed, t)
        d = sc - s0
        logw += th * np.einsum("nd,nd->n", d, d) * dt
        b = -((1.0 - w) * s0 + w * sc)
        x = x - b * dt + np.sqrt(dt) * rng.standard_normal(x.shape)
        if i + 1 < M:
            _, ess = normalize(logw)
            if ess < threshold:
                x, ess_before = systematic_resample(x, logw, rng)
                resample_times.append(t - dt)
                pre_ess.append(ess_before)
                logw = np.zeros(N)
    wn, ess = normalize(logw)
    return dict(x=x, wn=wn, ess=ess, log_weight_sd=float(np.std(logw)),
                mean_weight=float(np.exp(logw).mean()),
                n_resample=len(resample_times),
                min_pre_resample_ess=float(min(pre_ess)) if pre_ess else 1.0,
                initial_ess=init_ess)


def terminal_standard_prior(grid: GridDoob2D, N=20_000, M=600, seed=0,
                            corrected=True):
    """Naive CFG proposal from N(0,I), with optional exact noise-end factor."""
    rng = np.random.default_rng(seed)
    bed, w = grid.bed, grid.w
    dt = 1.0 / M
    x = rng.standard_normal((N, 2))
    logw = np.zeros(N)
    if corrected:
        logp0 = np.log(p0_pdf(x, bed, 1.0) + 1e-300)
        logpc = np.log(pc_pdf(x, bed, 1.0) + 1e-300)
        logr = -0.5 * np.sum(x * x, axis=1) - np.log(2.0 * np.pi)
        # p_1^{beta,S}/r_1 = m_1/(Z_1 r_1).
        logw += (1.0 - w) * logp0 + w * logpc - grid.logZ1 - logr
    th = 0.5 * w * (w - 1.0)
    for i in range(M):
        t = 1.0 - i * dt
        s0, sc = p0_score(x, bed, t), pc_score(x, bed, t)
        d = sc - s0
        logw += th * np.einsum("nd,nd->n", d, d) * dt
        b = -((1.0 - w) * s0 + w * sc)
        x = x - b * dt + np.sqrt(dt) * rng.standard_normal(x.shape)
    wn, ess = normalize(logw)
    return dict(x=x, wn=wn, ess=ess, log_weight_sd=float(np.std(logw)),
                logW=logw.copy(),
                mean_weight=float(np.exp(logsumexp(logw) - np.log(N))))


def doob_standard_prior(grid: GridDoob2D, N=20_000, M=600, seed=0,
                        corrected=True):
    """Exact grid Doob drift from N(0,I), with optional q_1/r_1 weighting."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    x = rng.standard_normal((N, 2))
    logw = np.zeros(N)
    if corrected:
        logq1 = np.log(grid.interp(grid.conv(1.0), x)) - grid.logZ0
        logr = -0.5 * np.sum(x * x, axis=1) - np.log(2.0 * np.pi)
        logw = logq1 - logr
    for i in range(M):
        t = 1.0 - i * dt
        _, score_q_grid = grid.conv_and_score(t)
        score_q = grid.interp(score_q_grid, x)
        x = x + score_q * dt + np.sqrt(dt) * rng.standard_normal(x.shape)
    wn, ess = normalize(logw)
    return dict(x=x, wn=wn, ess=ess, log_weight_sd=float(np.std(logw)),
                logW=logw.copy(),
                mean_weight=float(np.exp(logsumexp(logw) - np.log(N))))


def vp_doob_moment_propagation(v0, vc, mu, w, T, M=4000,
                               beta_min=0.1, beta_max=2.0,
                               standard_prior=True):
    """Exact mean/variance of the discrete affine VP Doob sampler.

    With standard_prior=False, initialize at the exact q_T.  Otherwise use
    N(0,1) without the q_T/N(0,1) correction.  The returned Gaussian KL is
    therefore a direct data-end initialization-error calculation, not an MC
    estimate.  The noising clock remains t:T->0.
    """
    _, h, P = geometric_gaussian([0.0, mu], [v0, vc], [1.0 - w, w], 0.0)
    aux = vp_noise_end(v0, vc, mu, w, T, beta_min, beta_max)
    alphaT = aux["alpha"]
    if standard_prior:
        mean, var = 0.0, 1.0
    else:
        mean = alphaT * h
        var = 1.0 + alphaT**2 * (P - 1.0)
    dt = T / M
    for i in range(M):
        t = T - i * dt
        beta = beta_min + (beta_max - beta_min) * t / T
        int_beta = beta_min * t + 0.5 * (beta_max - beta_min) * t*t / T
        alpha = np.exp(-0.5 * int_beta)
        vq = 1.0 + alpha**2 * (P - 1.0)
        mq = alpha * h
        # drift=A*x+c for reverse motion; one Euler step t->t-dt.
        A = -0.5 * beta + beta / vq
        c = -beta * mq / vq
        mean = mean - (A * mean + c) * dt
        var = var - 2.0 * A * var * dt + beta * dt
    kl = 0.5 * (var / P + (mean - h)**2 / P - 1.0
                + np.log(P / var))
    return dict(T=T, M=M, initialization="N(0,1)" if standard_prior else "q_T",
                mean=mean, variance=var, target_mean=h, target_variance=P,
                mean_error=mean-h, variance_error=var-P, data_end_gaussian_kl=kl)


def timed_call(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-smc", action="store_true")
    ap.add_argument("--skip-initialization", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    N, M, n = (4000, 160, 160) if args.quick else (20_000, 600, 320)
    diag_n = min(2000, N)

    if not args.skip_smc:
        rows = []
        for bed in [GAUSS_BED, GMM_BED]:
            w = 3.0
            grid = GridDoob2D(bed, w, n=n)
            runs = []
            naive, sec = timed_call(lambda: smc_cfg(
                bed, w, N=N, M=M, seed=stable_seed("theory-check-terminal", bed["name"], w),
                snapshots=()))
            base = dict(x=naive["x0"], wn=naive["wn"], ess=naive["ess"],
                        log_weight_sd=naive["sdlogW"], mean_weight=naive["EW"],
                        n_resample=0, min_pre_resample_ess=np.nan)
            runs.append(("terminal IS", base, sec))
            smc, sec = timed_call(lambda: smc_resampling_cfg(
                bed, w, N=N, M=M, seed=stable_seed("theory-check-smc", bed["name"], w)))
            runs.append(("adaptive SMC", smc, sec))
            exact, sec = timed_call(lambda: grid.sample(
                N=N, M=M, seed=stable_seed("theory-check-doob", bed["name"], w), rho=1.0))
            exact.update(n_resample=0, min_pre_resample_ess=np.nan)
            runs.append(("exact Doob", exact, sec))
            approx, sec = timed_call(lambda: grid.sample(
                N=N, M=M, seed=stable_seed("theory-check-approx", bed["name"], w), rho=0.8))
            approx.update(n_resample=0, min_pre_resample_ess=np.nan)
            runs.append(("approx Doob IS", approx, sec))
            for method, r, seconds in runs:
                metrics = evaluate(r, grid, diag_n=diag_n,
                                   diag_seed=stable_seed("theory-check-diagnostic", bed["name"], w))
                row = dict(bed=bed["name"], w=w, method=method, N=N, M=M, grid=n,
                           seconds=seconds, **{k: r.get(k, np.nan) for k in
                           ["ess", "log_weight_sd", "mean_weight", "n_resample",
                            "min_pre_resample_ess"]}, **metrics)
                rows.append(row)
                print(f"SMC {bed['name']} {method}: ESS={r['ess']:.4g}, "
                      f"MMD2={metrics['mmd2']:.6g}, SW1={metrics['sliced_w1']:.6g}, "
                      f"time={seconds:.2f}s, resamples={r.get('n_resample', 0)}", flush=True)
            write_csv(OUT / "smc_controlled_comparison.csv", rows)

    if not args.skip_initialization:
        rows = []
        for bed in [GAUSS_BED, GMM_BED]:
            w = 3.0
            grid = GridDoob2D(bed, w, n=n)
            configs = [
                ("terminal IS", "p_1^{beta,S}", True),
                ("terminal IS", "N(0,I)", True),
                ("terminal IS", "N(0,I)", False),
                ("exact Doob", "q_1", True),
                ("exact Doob", "N(0,I)", True),
                ("exact Doob", "N(0,I)", False),
            ]
            for method, init, corrected in configs:
                def run():
                    seed = stable_seed("shared-prior", bed["name"], method, init, corrected)
                    if method == "terminal IS" and init == "p_1^{beta,S}":
                        r = smc_cfg(bed, w, N=N, M=M, seed=seed, snapshots=())
                        return dict(x=r["x0"], wn=r["wn"], ess=r["ess"],
                                    log_weight_sd=r["sdlogW"], mean_weight=r["EW"])
                    if method == "terminal IS":
                        return terminal_standard_prior(grid, N, M, seed, corrected)
                    if init == "q_1":
                        return grid.sample(N=N, M=M, seed=seed, rho=1.0)
                    return doob_standard_prior(grid, N, M, seed, corrected)
                r, seconds = timed_call(run)
                metrics = evaluate(r, grid, diag_n=diag_n,
                                   diag_seed=stable_seed("shared-prior-diagnostic", bed["name"], w))
                row = dict(bed=bed["name"], w=w, method=method, initialization=init,
                           noise_end_corrected=corrected, N=N, M=M, grid=n,
                           seconds=seconds, **{k: r[k] for k in
                           ["ess", "log_weight_sd", "mean_weight"]}, **metrics)
                rows.append(row)
                print(f"INIT {bed['name']} {method} {init} corrected={corrected}: "
                      f"ESS={r['ess']:.4g}, MMD2={metrics['mmd2']:.6g}, "
                      f"SW1={metrics['sliced_w1']:.6g}", flush=True)
                write_csv(OUT / "shared_prior_initialization.csv", rows)

        vp_rows = []
        for T in [1.0, 2.0, 5.0, 10.0]:
            for standard in [False, True]:
                r = vp_doob_moment_propagation(2.25, 0.25, 0.8, 3.0, T,
                                               M=400 if args.quick else 4000,
                                               standard_prior=standard)
                vp_rows.append(r)
                print(f"VP T={T:g} {r['initialization']}: "
                      f"data-end Gaussian KL={r['data_end_gaussian_kl']:.6g}", flush=True)
        write_csv(OUT / "vp_shared_prior_data_end.csv", vp_rows)


if __name__ == "__main__":
    main()
