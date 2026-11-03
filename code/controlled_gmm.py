#!/usr/bin/env python3
"""Grid-based exact and approximate Doob control for the 2D CFG test beds.

Under shared VE noising, q_t=K_t m_0/Z_0.  We represent m_0 on a periodic
finite grid, evaluate K_t m_0 by FFT Gaussian convolution, and use
-grad log q_t as the exact controlled reverse drift.  The sampler starts at
q_1 and reaches m_0/Z_0 with uniform weights.  Method C replaces the exact
control by rho*u* and evaluates its pathwise importance weight directly.

The grid is a numerical approximation to the continuous Doob function, not a
change of target.  Gaussian-bed tests and grid/time refinement are included to
quantify this approximation.  All diagnostics use the exact tempered density
on the data-end grid; MMD and sliced Wasserstein avoid the histogram-TV floor.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import logsumexp

from gmm2d import (GAUSS_BED, GMM_BED, p0_pdf, pc_pdf, p0_score, pc_score,
                   smc_cfg, stable_seed, tempered_on, tv_to_target)

OUT = Path(__file__).resolve().parents[1] / "output"


class GridDoob2D:
    """Doob function q_t=K_t m_0/Z_0 on a uniform periodic grid."""

    def __init__(self, bed, w, n=256, L=6.0):
        self.bed = bed
        self.w = float(w)
        self.n = int(n)
        self.L = float(L)
        self.dx = 2.0 * L / n
        self.axis = -L + (np.arange(n) + 0.5) * self.dx
        xy = np.stack(np.meshgrid(self.axis, self.axis, indexing="ij"),
                      axis=-1).reshape(-1, 2)
        self.xy = xy
        self.m0 = tempered_on(xy, bed, w, 0.0).reshape(n, n)
        if not np.all(np.isfinite(self.m0)) or self.m0.sum() <= 0:
            raise ValueError("Non-normalizable or numerically empty target")
        self.logZ0 = float(np.log(self.m0.sum()) + 2 * np.log(self.dx))
        self.m0_hat = np.fft.fft2(self.m0)
        freq = 2 * np.pi * np.fft.fftfreq(n, d=self.dx)
        self.kx, self.ky = np.meshgrid(freq, freq, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        m1 = tempered_on(xy, bed, w, 1.0)
        self.logZ1 = float(np.log(m1.sum()) + 2 * np.log(self.dx))
        self.target_prob = (self.m0 / self.m0.sum()).ravel()
        self.target_mean = self.target_prob @ xy
        d = xy - self.target_mean
        self.target_cov = (d * self.target_prob[:, None]).T @ d

    def conv(self, t):
        factor = np.exp(-0.5 * t * self.k2)
        conv = np.fft.ifft2(self.m0_hat * factor).real
        # FFT roundoff can create tiny negative values in remote tails.
        return np.maximum(conv, 1e-300)

    def conv_and_score(self, t):
        conv_hat = self.m0_hat * np.exp(-0.5 * t * self.k2)
        conv = np.maximum(np.fft.ifft2(conv_hat).real, 1e-300)
        gx = np.fft.ifft2(1j * self.kx * conv_hat).real
        gy = np.fft.ifft2(1j * self.ky * conv_hat).real
        return conv, np.stack([gx / conv, gy / conv], axis=-1)

    def interp(self, field, x):
        coords = (np.asarray(x) - self.axis[0]) / self.dx
        if field.ndim == 2:
            return map_coordinates(field, coords.T, order=1, mode="nearest")
        return np.stack([
            map_coordinates(field[..., j], coords.T, order=1, mode="nearest")
            for j in range(field.shape[-1])
        ], axis=-1)

    def grid_sample(self, density, N, rng):
        p = np.maximum(density.ravel(), 0.0)
        p = p / p.sum()
        ind = rng.choice(self.n * self.n, size=N, p=p)
        x = self.xy[ind].copy()
        x += rng.uniform(-0.5 * self.dx, 0.5 * self.dx, size=x.shape)
        return x

    def initial_sample(self, N, rng):
        return self.grid_sample(self.conv(1.0), N, rng)

    def target_sample(self, N, seed=0):
        return self.grid_sample(self.m0, N, np.random.default_rng(seed))

    def log_marginals(self, x, t):
        return (np.log(p0_pdf(x, self.bed, t) + 1e-300),
                np.log(pc_pdf(x, self.bed, t) + 1e-300))

    def sample(self, N=20_000, M=500, seed=0, rho=1.0):
        """Exact Doob (rho=1) or approximate-control IS (rho!=1)."""
        rng = np.random.default_rng(seed)
        dt = 1.0 / M
        x = self.initial_sample(N, rng)
        if rho == 1.0:
            logW = np.zeros(N)
        else:
            logp0, logpc = self.log_marginals(x, 1.0)
            logq1 = np.log(self.interp(self.conv(1.0), x)) - self.logZ0
            logW = ((1 - self.w) * logp0 + self.w * logpc - self.logZ1
                    - logq1)
        max_abs_score = 0.0
        for i in range(M):
            t = 1.0 - i * dt
            _, score_q_grid = self.conv_and_score(t)
            score_q = self.interp(score_q_grid, x)
            max_abs_score = max(max_abs_score, float(np.max(np.abs(score_q))))
            if rho == 1.0:
                drift = -score_q
                dB = np.sqrt(dt) * rng.standard_normal(x.shape)
                x = x - drift * dt + dB
            else:
                s0 = p0_score(x, self.bed, t)
                sc = pc_score(x, self.bed, t)
                score_m = (1 - self.w) * s0 + self.w * sc
                u_star = score_q - score_m
                u = rho * u_star
                V = -0.5 * self.w * (self.w - 1.0) * np.sum((sc - s0)**2, axis=1)
                dB = np.sqrt(dt) * rng.standard_normal(x.shape)
                logW += (-V * dt - np.sum(u * dB, axis=1)
                         - 0.5 * np.sum(u**2, axis=1) * dt)
                x = x - (-score_m - u) * dt + dB
        if rho == 1.0:
            wn = np.full(N, 1.0 / N)
            ess = 1.0
            sdlog = 0.0
            mean_weight = 1.0
        else:
            wn = np.exp(logW - logsumexp(logW))
            ess = float(np.exp(-logsumexp(2 * np.log(wn)))) / N
            sdlog = float(np.std(logW))
            mean_weight = float(np.exp(logsumexp(logW) - np.log(N)))
        outside = float(np.mean(np.any(np.abs(x) > self.L, axis=1)))
        return dict(x=x, wn=wn, ess=ess, log_weight_sd=sdlog,
                    mean_weight=mean_weight, outside=outside,
                    max_abs_score=max_abs_score)


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w = w / w.sum()
    positions = np.cumsum(w) - 0.5 * w
    return np.interp(quantiles, positions, v)


def sliced_wasserstein(x, wx, y, wy, n_slices=64, n_quantiles=1000):
    """Mean sliced 1-Wasserstein distance on fixed equally spaced directions."""
    angles = 2 * np.pi * np.arange(n_slices) / n_slices
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    values = []
    for d in directions:
        qx = weighted_quantile(x @ d, wx, q)
        qy = weighted_quantile(y @ d, wy, q)
        values.append(np.mean(np.abs(qx - qy)))
    return float(np.mean(values))


def rbf_mmd2(x, wx, y, wy, ell=1.0, block=500):
    """Weighted RBF-kernel MMD^2, using a fixed unit bandwidth."""
    wx = wx / wx.sum()
    wy = wy / wy.sum()

    def kernel(a, b):
        d2 = (np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :]
              - 2 * a @ b.T)
        return np.exp(-0.5 * np.maximum(d2, 0.0) / ell**2)

    def weighted_gram(a, wa, b, wb, diagonal=False):
        total = 0.0
        for i in range(0, len(a), block):
            for j in range(0, len(b), block):
                K = kernel(a[i:i+block], b[j:j+block])
                total += float(wa[i:i+block] @ K @ wb[j:j+block])
        return total

    return (weighted_gram(x, wx, x, wx) + weighted_gram(y, wy, y, wy)
            - 2 * weighted_gram(x, wx, y, wy))


def evaluate(result, grid: GridDoob2D, diag_n=3000, diag_seed=0):
    """Moment, tail-independent MMD/SW, cross-entropy and histogram-TV errors."""
    x, wn = result["x"], result["wn"]
    wn = wn / wn.sum()
    mean = wn @ x
    var = wn @ ((x - mean)**2)
    target = grid.target_sample(diag_n, seed=diag_seed)
    rng = np.random.default_rng(diag_seed + 1)
    ind = rng.choice(len(x), size=min(diag_n, len(x)), replace=False)
    xs, ws = x[ind], wn[ind]
    ws = ws / ws.sum()
    wy = np.full(diag_n, 1.0 / diag_n)
    mmd2 = rbf_mmd2(xs, ws, target, wy)
    sw = sliced_wasserstein(xs, ws, target, wy)
    logq0 = np.log(tempered_on(x, grid.bed, grid.w, 0.0) + 1e-300) - grid.logZ0
    cross_entropy = float(-wn @ logq0)
    tv = tv_to_target(x, wn, grid.bed, grid.w)
    return dict(mean_x=float(mean[0]), mean_y=float(mean[1]),
                var_x=float(var[0]), var_y=float(var[1]),
                mean_error=float(np.linalg.norm(mean - grid.target_mean)),
                covariance_error=float(np.linalg.norm(
                    np.diag(var) - np.diag(grid.target_cov))),
                mmd2=mmd2, sliced_w1=sw, cross_entropy=cross_entropy, tv=tv)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bed", choices=["gmm", "gaussian", "both"], default="both")
    ap.add_argument("--skip-naive", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    N, M, n = (4000, 160, 160) if args.quick else (20_000, 600, 320)
    beds = {"gmm": [GMM_BED], "gaussian": [GAUSS_BED],
            "both": [GAUSS_BED, GMM_BED]}[args.bed]
    rows = []
    for bed in beds:
        for w in [1.5, 2.0, 3.0]:
            grid = GridDoob2D(bed, w, n=n)
            print(f"\n=== {bed['name']} bed, w={w}, grid={n}, N={N}, M={M} ===")
            runs = []
            if not args.skip_naive:
                naive = smc_cfg(bed, w, N=N, M=M,
                                seed=stable_seed("naive-for-control", bed["name"], w),
                                snapshots=())
                runs.append(("naive", dict(x=naive["x0"],
                                           wn=np.full(N, 1.0 / N), ess=1.0,
                                           log_weight_sd=0.0, mean_weight=1.0,
                                           outside=np.nan, max_abs_score=np.nan)))
                runs.append(("terminal IS", dict(x=naive["x0"], wn=naive["wn"],
                                                 ess=naive["ess"],
                                                 log_weight_sd=naive["sdlogW"],
                                                 mean_weight=naive["EW"],
                                                 outside=np.nan,
                                                 max_abs_score=np.nan)))
            for method, rho in [("exact Doob", 1.0),
                                ("approx Doob IS", 0.8)]:
                r = grid.sample(N=N, M=M,
                                seed=stable_seed(method, bed["name"], w), rho=rho)
                runs.append((method, r))
            for method, r in runs:
                metrics = evaluate(r, grid, diag_n=min(2000, N),
                                   diag_seed=stable_seed("diagnostic", bed["name"], w))
                row = dict(bed=bed["name"], w=w, method=method, N=N, M=M,
                           grid=n, ess=r["ess"],
                           log_weight_sd=r["log_weight_sd"],
                           mean_weight=r["mean_weight"], **metrics)
                rows.append(row)
                print(f"{method:>16}: ESS/N={r['ess']:.4f}, "
                      f"mean error={metrics['mean_error']:.5f}, "
                      f"MMD^2={metrics['mmd2']:.6f}, "
                      f"SW1={metrics['sliced_w1']:.5f}, TV={metrics['tv']:.4f}")
            write_csv(OUT / "controlled_2d.csv", rows)


if __name__ == "__main__":
    main()
