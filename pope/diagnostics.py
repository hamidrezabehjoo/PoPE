"""Adaptive-SMC, initialization, and distributional diagnostics.

The grid reference is used only to evaluate the exact tempered target and to
construct reference samples. It is not a controlled sampler.
"""
import csv
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .core import (GMM_BED, gmm_sample, p0_pdf, p0_score, pc_pdf, pc_score,
                   smc_cfg, stable_seed, tempered_on, tv_to_target)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(logw):
    wn = np.exp(logw - logsumexp(logw))
    ess = float(1.0 / np.sum(wn * wn) / len(wn))
    return wn, ess


def systematic_resample(x, logw, rng):
    wn, ess = normalize(logw)
    positions = (rng.random() + np.arange(len(x))) / len(x)
    indices = np.searchsorted(np.cumsum(wn), positions, side="right")
    return x[indices], ess


class ReferenceGrid:
    """Exact tempered target moments and reference draws on a uniform grid."""

    def __init__(self, bed, w, n=320, L=6.0):
        self.bed, self.w, self.n, self.L = bed, float(w), int(n), float(L)
        self.dx = 2.0 * L / n
        self.axis = -L + (np.arange(n) + 0.5) * self.dx
        self.xy = np.stack(np.meshgrid(self.axis, self.axis, indexing="ij"),
                           axis=-1).reshape(-1, 2)
        self.m0 = tempered_on(self.xy, bed, w, 0.0).reshape(n, n)
        if not np.all(np.isfinite(self.m0)) or self.m0.sum() <= 0:
            raise ValueError("Non-normalizable or numerically empty target")
        self.logZ0 = float(np.log(self.m0.sum()) + 2 * np.log(self.dx))
        m1 = tempered_on(self.xy, bed, w, 1.0)
        self.logZ1 = float(np.log(m1.sum()) + 2 * np.log(self.dx))
        self.target_prob = (self.m0 / self.m0.sum()).ravel()
        self.target_mean = self.target_prob @ self.xy
        centered = self.xy - self.target_mean
        self.target_cov = ((centered * self.target_prob[:, None]).T @ centered)

    def target_sample(self, N, seed=0):
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.n * self.n, size=N, p=self.target_prob)
        x = self.xy[indices].copy()
        return x + rng.uniform(-0.5 * self.dx, 0.5 * self.dx, size=x.shape)


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    weights = weights / weights.sum()
    positions = np.cumsum(weights) - 0.5 * weights
    return np.interp(quantiles, positions, values)


def sliced_wasserstein(x, wx, y, wy, n_slices=64, n_quantiles=1000):
    """Mean sliced 1-Wasserstein distance on fixed equally spaced directions."""
    angles = 2 * np.pi * np.arange(n_slices) / n_slices
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    quantiles = (np.arange(n_quantiles) + 0.5) / n_quantiles
    values = []
    for direction in directions:
        qx = weighted_quantile(x @ direction, wx, quantiles)
        qy = weighted_quantile(y @ direction, wy, quantiles)
        values.append(np.mean(np.abs(qx - qy)))
    return float(np.mean(values))


def rbf_mmd2(x, wx, y, wy, ell=1.0, block=500):
    """Fixed-bandwidth biased weighted RBF-kernel MMD squared."""
    wx, wy = wx / wx.sum(), wy / wy.sum()

    def kernel(a, b):
        d2 = (np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :]
              - 2 * a @ b.T)
        return np.exp(-0.5 * np.maximum(d2, 0.0) / ell**2)

    def gram(a, wa, b, wb):
        total = 0.0
        for i in range(0, len(a), block):
            for j in range(0, len(b), block):
                total += float(wa[i:i+block] @ kernel(a[i:i+block], b[j:j+block])
                               @ wb[j:j+block])
        return total

    return gram(x, wx, x, wx) + gram(y, wy, y, wy) - 2 * gram(x, wx, y, wy)


def evaluate(result, grid, diag_n=2000, diag_seed=0):
    """Moment, MMD, sliced-Wasserstein, and histogram-TV diagnostics."""
    x, wn = result["x"], result["wn"] / result["wn"].sum()
    mean = wn @ x
    var = wn @ ((x - mean)**2)
    target = grid.target_sample(diag_n, seed=diag_seed)
    rng = np.random.default_rng(diag_seed + 1)
    indices = rng.choice(len(x), size=min(diag_n, len(x)), replace=False)
    xs, ws = x[indices], wn[indices]
    ws /= ws.sum()
    wy = np.full(diag_n, 1.0 / diag_n)
    logq0 = np.log(tempered_on(x, grid.bed, grid.w, 0.0) + 1e-300) - grid.logZ0
    return dict(mean_x=float(mean[0]), mean_y=float(mean[1]),
                var_x=float(var[0]), var_y=float(var[1]),
                mean_error=float(np.linalg.norm(mean - grid.target_mean)),
                covariance_error=float(np.linalg.norm(
                    np.diag(var) - np.diag(grid.target_cov))),
                mmd2=rbf_mmd2(xs, ws, target, wy),
                sliced_w1=sliced_wasserstein(xs, ws, target, wy),
                cross_entropy=float(-wn @ logq0),
                tv=tv_to_target(x, wn, grid.bed, grid.w))


def smc_resampling_cfg(bed, w, N=20_000, M=600, seed=0, threshold=0.5):
    """Adaptive SMC with systematic resampling at incremental ESS/N < 0.5."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    theta = 0.5 * w * (w - 1.0)
    # Retain this draw for compatibility with the reported random streams.
    rng.normal(size=(N, 2))
    x = gmm_sample(N, bed["m0"], bed["v0"], bed["w0"], 1.0, rng)
    log_init = w * (np.log(pc_pdf(x, bed, 1.0) + 1e-300)
                    - np.log(p0_pdf(x, bed, 1.0) + 1e-300))
    x, init_ess = systematic_resample(x, log_init, rng)
    logw = np.zeros(N)
    score_calls = 0
    resample_times, pre_ess = [], []
    for i in range(M):
        t = 1.0 - i * dt
        s0 = p0_score(x, bed, t)
        score_calls += N
        sc = pc_score(x, bed, t)
        score_calls += N
        difference = sc - s0
        logw += theta * np.einsum("nd,nd->n", difference, difference) * dt
        drift = -((1.0 - w) * s0 + w * sc)
        x = x - drift * dt + np.sqrt(dt) * rng.standard_normal(x.shape)
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
                initial_ess=init_ess, score_calls=score_calls)


def terminal_standard_prior(grid, N=20_000, M=600, seed=0, corrected=True):
    """Naive CFG proposal from N(0,I), optionally with its exact initial ratio."""
    rng = np.random.default_rng(seed)
    bed, w = grid.bed, grid.w
    dt = 1.0 / M
    x = rng.standard_normal((N, 2))
    logw = np.zeros(N)
    if corrected:
        logp0 = np.log(p0_pdf(x, bed, 1.0) + 1e-300)
        logpc = np.log(pc_pdf(x, bed, 1.0) + 1e-300)
        logr = -0.5 * np.sum(x * x, axis=1) - np.log(2.0 * np.pi)
        logw += (1.0 - w) * logp0 + w * logpc - grid.logZ1 - logr
    theta = 0.5 * w * (w - 1.0)
    for i in range(M):
        t = 1.0 - i * dt
        s0, sc = p0_score(x, bed, t), pc_score(x, bed, t)
        difference = sc - s0
        logw += theta * np.einsum("nd,nd->n", difference, difference) * dt
        drift = -((1.0 - w) * s0 + w * sc)
        x = x - drift * dt + np.sqrt(dt) * rng.standard_normal(x.shape)
    wn, ess = normalize(logw)
    return dict(x=x, wn=wn, ess=ess, log_weight_sd=float(np.std(logw)),
                mean_weight=float(np.exp(logsumexp(logw) - np.log(N))))


def summarize(rows, group_keys, metric_keys):
    summaries = []
    groups = sorted({tuple(row[key] for key in group_keys) for row in rows})
    for group in groups:
        selected = [row for row in rows
                    if tuple(row[key] for key in group_keys) == group]
        out = dict(zip(group_keys, group))
        out["n_replicates"] = len(selected)
        for key in metric_keys:
            values = np.array([float(row[key]) for row in selected])
            out[key + "_mean"] = float(values.mean())
            out[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(out)
    return summaries


def timed_call(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def run_comparisons(output_dir, quick=False):
    """Regenerate the matched-budget SMC and shared-prior manuscript tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N, M, n = (2000, 100, 96) if quick else (20_000, 600, 320)
    diag_n = min(2000, N)
    grid = ReferenceGrid(GMM_BED, 3.0, n=n)
    rows = []
    for replicate in range(3):
        for method in ("terminal IS", "adaptive SMC"):
            seed = stable_seed("theory-replicate-all", method, replicate)

            def run():
                if method == "terminal IS":
                    result = smc_cfg(GMM_BED, 3.0, N=N, M=M, seed=seed, snapshots=())
                    return dict(x=result["x0"], wn=result["wn"], ess=result["ess"],
                                n_resample=0)
                return smc_resampling_cfg(GMM_BED, 3.0, N=N, M=M, seed=seed)

            result, seconds = timed_call(run)
            metrics = evaluate(result, grid, diag_n=diag_n,
                               diag_seed=stable_seed("theory-replicate-all-diagnostic", replicate))
            row = dict(replicate=replicate, method=method, N=N, M=M, grid=n,
                       seconds=seconds, ess=result["ess"],
                       n_resample=result.get("n_resample", 0), **metrics)
            rows.append(row)
            print("SMC", replicate, method, f"ESS={result['ess']:.5g}", flush=True)
            write_csv(output_dir / "smc_comparison_replicates.csv", rows)
    metrics = ["ess", "mmd2", "sliced_w1", "mean_error", "tv", "seconds", "n_resample"]
    write_csv(output_dir / "smc_comparison.csv", summarize(rows, ["method"], metrics))

    configs = [("p_1^{beta,S}", True), ("N(0,I)", True), ("N(0,I)", False)]
    init_rows = []
    for replicate in range(3):
        for initialization, corrected in configs:
            seed = stable_seed("shared-prior-replicate", "terminal IS",
                               initialization, corrected, replicate)
            if initialization == "p_1^{beta,S}":
                result0 = smc_cfg(GMM_BED, 3.0, N=N, M=M, seed=seed, snapshots=())
                result = dict(x=result0["x0"], wn=result0["wn"], ess=result0["ess"])
            else:
                result = terminal_standard_prior(grid, N, M, seed, corrected)
            metrics0 = evaluate(result, grid, diag_n=diag_n,
                                diag_seed=stable_seed("shared-prior-rep-diagnostic", replicate))
            row = dict(replicate=replicate, method="terminal IS",
                       initialization=initialization, noise_end_corrected=corrected,
                       N=N, M=M, grid=n, ess=result["ess"], **metrics0)
            init_rows.append(row)
            print("INIT", replicate, initialization, corrected,
                  f"ESS={result['ess']:.5g}", flush=True)
            write_csv(output_dir / "shared_prior_replicates.csv", init_rows)
    write_csv(output_dir / "shared_prior.csv",
              summarize(init_rows, ["method", "initialization", "noise_end_corrected"],
                        ["ess", "mmd2", "sliced_w1", "mean_error", "tv"]))
