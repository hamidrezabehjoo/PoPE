#!/usr/bin/env python3
"""Exact and approximate Doob control for one-dimensional Gaussian beds.

Conventions match the manuscript: VE noising p_t=N(mu,sigma^2+t), reverse
SDEs run t:1->0, and no tau=1-t substitution is used.  For a normalized
geometric target m_0/Z_0=N(h,P), shared noising gives

    q_t = K_t m_0/Z_0 = N(h, P+t).

Thus the exact controlled reverse drift is -grad log q_t=(x-h)/(P+t), the
noise-end law is q_1=N(h,P+1), and every particle has unit weight.  This
remains valid when uncontrolled terminal IS has infinite weight variance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from gmm2d import gaussian_path_mgf, stable_seed

OUT = Path(__file__).resolve().parents[1] / "output"


def geometric_gaussian(mus, variances, betas, t=0.0):
    """Return log mass, mean and variance of prod_k N(mu_k,v_k+t)^beta_k."""
    mus = np.asarray(mus, dtype=float)
    variances = np.asarray(variances, dtype=float) + t
    betas = np.asarray(betas, dtype=float)
    precision = float(np.sum(betas / variances))
    if precision <= 0:
        raise ValueError("The geometric Gaussian is not normalizable")
    var = 1.0 / precision
    mean = var * float(np.sum(betas * mus / variances))
    log_components = -0.5 * (np.log(2 * np.pi * variances) + mus**2 / variances)
    log_q_at_zero = -0.5 * np.log(2 * np.pi * var) - mean**2 / (2 * var)
    log_mass = float(betas @ log_components - log_q_at_zero)
    return log_mass, mean, var


def normal_logpdf(x, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def doob_sample(mus, variances, betas, N=60_000, M=2000, t_min=0.0, seed=0):
    """Uniform-weight exact Doob sampler for a Gaussian geometric target."""
    if not 0 <= t_min < 1:
        raise ValueError("t_min must satisfy 0 <= t_min < 1")
    rng = np.random.default_rng(seed)
    _, mean, var = geometric_gaussian(mus, variances, betas, 0.0)
    dt = (1.0 - t_min) / M
    x = mean + np.sqrt(var + 1.0) * rng.standard_normal(N)  # exact q_1
    for i in range(M):
        t = 1.0 - i * dt
        drift = (x - mean) / (var + t)  # -score q_t
        x = x - drift * dt + np.sqrt(dt) * rng.standard_normal(N)
    exact_var = var + t_min
    return dict(x=x, mean=float(x.mean()), variance=float(x.var()),
                exact_mean=mean, exact_variance=exact_var,
                mean_error=float(x.mean() - mean),
                variance_error=float(x.var() - exact_var), ess=1.0,
                log_weight_sd=0.0)


def approx_control_cfg(v0, vc, mu, w, rho=0.8, N=60_000, M=2000, seed=0):
    """Method C with u_hat=rho*u*, exact q_1 initialization, and path IS.

    The implementation evaluates equation (approxcontrolweight) directly.
    rho=1 is the exact Doob proposal (up to time discretization); rho<1 is an
    deliberately imperfect control used to verify the nonuniform weight.
    """
    rng = np.random.default_rng(seed)
    _, h, P = geometric_gaussian([0.0, mu], [v0, vc], [1.0 - w, w], 0.0)
    logZ1, h1, P1 = geometric_gaussian([0.0, mu], [v0, vc], [1.0 - w, w], 1.0)
    dt = 1.0 / M
    x = h + np.sqrt(P + 1.0) * rng.standard_normal(N)
    logW = (logZ1 + normal_logpdf(x, h1, P1)
            - normal_logpdf(x, h, P + 1.0))
    for i in range(M):
        t = 1.0 - i * dt
        s0 = -x / (v0 + t)
        sc = -(x - mu) / (vc + t)
        score_m = (1.0 - w) * s0 + w * sc
        _, ht, Pt = geometric_gaussian([0.0, mu], [v0, vc], [1.0 - w, w], t)
        score_q = -(x - h) / (P + t)
        u_star = score_q - score_m
        u = rho * u_star
        V = -0.5 * w * (w - 1.0) * (sc - s0) ** 2
        dB = np.sqrt(dt) * rng.standard_normal(N)
        logW += -V * dt - u * dB - 0.5 * u**2 * dt
        b_bar = -score_m
        x = x - (b_bar - u) * dt + dB
    log_norm = logsumexp(logW) - np.log(N)
    wn = np.exp(logW - logsumexp(logW))
    mean = float(wn @ x)
    var = float(wn @ (x - mean) ** 2)
    ess = float(np.exp(-logsumexp(2 * np.log(wn)))) / N
    _, h0, P0 = geometric_gaussian([0.0, mu], [v0, vc], [1.0 - w, w], 0.0)
    return dict(mean=mean, variance=var, exact_mean=h0, exact_variance=P0,
                mean_error=mean - h0, variance_error=var - P0, ess=ess,
                mean_weight=float(np.exp(log_norm)),
                log_weight_sd=float(np.std(logW)))


def mgf_finite(v0, vc, mu, w, lam, M, t_min):
    _, ok = gaussian_path_mgf(v0, vc, mu, w, lam, M=M, t_min=t_min)
    return bool(ok)


def lambda_star(v0, vc, mu, w, M=300, t_min=0.0, tol=2e-3):
    """Moment index for the one-dimensional CFG Gaussian path weight."""
    if w <= 1:
        return np.inf
    lo, hi = 0.0, 1.0
    while mgf_finite(v0, vc, mu, w, hi, M, t_min):
        lo = hi
        hi *= 2
        if hi > 1e4:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mgf_finite(v0, vc, mu, w, mid, M, t_min):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_w(v0, vc, mu, t_min=0.0, M=300, tol=2e-3):
    """Critical w located directly by the lambda=2 Cameron--Martin test."""
    lo, hi = 1.0, 2.0
    while mgf_finite(v0, vc, mu, hi, 2.0, M, t_min):
        lo = hi
        hi *= 2
        if hi > 1e4:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mgf_finite(v0, vc, mu, mid, 2.0, M, t_min):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vp_noise_end(v0, vc, mu, w, T, beta_min=0.1, beta_max=2.0):
    """Noise-end KLs to N(0,1) for a linear-beta VP noising process."""
    integrated_beta = T * 0.5 * (beta_min + beta_max)
    alpha = np.exp(-0.5 * integrated_beta)
    v0T = 1.0 + alpha**2 * (v0 - 1.0)
    vcT = 1.0 + alpha**2 * (vc - 1.0)
    mucT = alpha * mu
    _, geom_mean, geom_var = geometric_gaussian([0.0, mucT], [v0T, vcT],
                                                [1.0 - w, w], 0.0)
    _, target_mean, target_var = geometric_gaussian([0.0, mu], [v0, vc],
                                                    [1.0 - w, w], 0.0)
    doob_mean = alpha * target_mean
    doob_var = 1.0 + alpha**2 * (target_var - 1.0)

    def kl(m, v):
        return 0.5 * (v + m * m - 1.0 - np.log(v))

    return dict(T=T, alpha=alpha, geometric_kl=kl(geom_mean, geom_var),
                doob_kl=kl(doob_mean, doob_var))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-early-stopping", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    N, M = (5000, 300) if args.quick else (60_000, 2000)

    print("=== Two-expert Gaussian: exact uniform-weight Doob control ===")
    r = doob_sample([1.5, -1.0], [0.25, 1.44], [0.5, 0.5], N=N, M=M,
                    seed=stable_seed("two-expert-doob"))
    print(f"sample N({r['mean']:.5f},{r['variance']:.5f}); "
          f"exact N({r['exact_mean']:.5f},{r['exact_variance']:.5f}); ESS/N=1")
    write_csv(OUT / "controlled_gaussian_two_expert.csv",
              [dict(N=N, M=M, **{k: v for k, v in r.items() if k != "x"})])

    rows = []
    print("\n=== Gaussian CFG: exact Doob control below and above w* ===")
    for j, w in enumerate([1.5, 2.0, 3.0, 5.0]):
        r = doob_sample([0.0, 0.8], [2.25, 0.25], [1.0 - w, w], N=N, M=M,
                        seed=stable_seed("cfg-doob", w))
        rows.append(dict(w=w, method="exact Doob", N=N, M=M, **{
            k: v for k, v in r.items() if k != "x"}))
        print(f"w={w:>3}: mean {r['mean']:.6f} (exact {r['exact_mean']:.6f}), "
              f"var {r['variance']:.6f} (exact {r['exact_variance']:.6f}), "
              f"ESS/N={r['ess']:.3f}")
    write_csv(OUT / "controlled_gaussian_cfg.csv", rows)

    approx_rows = []
    print("\n=== Approximate-control IS (method C), rho=0.8 ===")
    for w in [2.0, 3.0]:
        r = approx_control_cfg(2.25, 0.25, 0.8, w, rho=0.8, N=N, M=M,
                               seed=stable_seed("approx-control", w))
        approx_rows.append(dict(w=w, rho=0.8, N=N, M=M, **r))
        print(f"w={w}: mean {r['mean']:.5f}, var {r['variance']:.5f}, "
              f"ESS/N={r['ess']:.4f}, sd(log W)={r['log_weight_sd']:.4f}")
    write_csv(OUT / "approx_control_gaussian_cfg.csv", approx_rows)

    print("\n=== Long-horizon VP noise-end approach to N(0,1), w=3 ===")
    vp_rows = [vp_noise_end(2.25, 0.25, 0.8, 3.0, T)
               for T in [1.0, 2.0, 5.0, 10.0]]
    for r in vp_rows:
        print(f"T={r['T']:>4}: alpha={r['alpha']:.3e}, "
              f"KL(geometric||N)={r['geometric_kl']:.3e}, "
              f"KL(q_1||N)={r['doob_kl']:.3e}")
    write_csv(OUT / "vp_noise_end.csv", vp_rows)

    if not args.skip_early_stopping:
        grid_M = 160 if args.quick else 300
        early_rows = []
        print(f"\n=== Early-stopping critical scales (lambda*=2, M={grid_M}) ===")
        for vc in [0.25, 0.05, 0.01, 0.001]:
            for tmin in [0.0, 0.001, 0.01, 0.05, 0.1]:
                # The 2D threshold is the smaller of the mean and variance axes.
                wx = critical_w(2.25, vc, 0.8, t_min=tmin, M=grid_M)
                wy = critical_w(2.25, vc, 0.0, t_min=tmin, M=grid_M)
                early_rows.append(dict(vc=vc, t_min=tmin, M=grid_M,
                                       w_star_mean_axis=wx,
                                       w_star_variance_axis=wy,
                                       w_star=min(wx, wy)))
                print(f"vc={vc:g}, t_min={tmin:g}: w*={min(wx, wy):.4f} "
                      f"(axes {wx:.4f}, {wy:.4f})")
        write_csv(OUT / "early_stopping.csv", early_rows)

        print("\n=== Check lambda*>1 away from the endpoint ===")
        margin_rows = []
        for w in [1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
            okx = mgf_finite(2.25, 0.25, 0.8, w, 1.01, grid_M, 0.0)
            oky = mgf_finite(2.25, 0.25, 0.0, w, 1.01, grid_M, 0.0)
            margin_rows.append(dict(w=w, lambda_test=1.01, M=grid_M,
                                    finite_mean_axis=okx,
                                    finite_variance_axis=oky,
                                    lambda_star_greater_than_1=okx and oky))
            print(f"w={w:>5}: E[W^1.01] finite = {okx and oky}")
        write_csv(OUT / "first_moment_margin.csv", margin_rows)
        high_rows = []
        for w in [5.0, 10.0, 20.0, 50.0]:
            lx = lambda_star(2.25, 0.25, 0.8, w, M=grid_M, tol=1e-3)
            ly = lambda_star(2.25, 0.25, 0.0, w, M=grid_M, tol=1e-3)
            high_rows.append(dict(w=w, M=grid_M, lambda_star_mean_axis=lx,
                                  lambda_star_variance_axis=ly,
                                  lambda_star=min(lx, ly)))
            print(f"w={w:>5}: lambda*={min(lx, ly):.4f}")
        write_csv(OUT / "lambda_star_high_guidance.csv", high_rows)


if __name__ == "__main__":
    main()
