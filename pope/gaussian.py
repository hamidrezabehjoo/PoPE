#!/usr/bin/env python3
"""
Example 1 (two Gaussian experts): closed forms, SMC verification, and the
KL certificate for the naive sampler.

Paper: "When Diffusion Composition Fails: Exact Composition through Path Measures",
Example (ex:gaussian) and the KL-certificate remark.

Setting: f = 0, g = 1 on [0,1] (VE noising), K = 2 experts, beta_1 = beta_2 =
1/2 (S = 1), p^(1)_0 = N(1.5, 0.5^2), p^(2)_0 = N(-1, 1.2^2), so
p^(k)_t = N(mu_k, sigma_k^2 + t).

Expected output (paper values):
  exact geometric mixture  m_0/Z_0  = N(1.1302, 0.4260)
  exact naive law                    = N(0.8321, 0.5225)
  Z_t/Z_1 at t = .75,.5,.25,0        = 0.92647, 0.83160, 0.70465, 0.52473
  KL certificate:  KL(naive_0 || m_0/Z_0) = 0.1155
                <= KL(P || Q) = int E_naive[V] dt - log(Z_1/Z_0)
                                = 0.7969 - 0.6449 = 0.1521
  SMC (N=6e4, M=2000): weighted data end ~ N(1.131, 0.423), ESS/N ~ 0.82,
  mean weights track Z_t/Z_1 at all intermediate times.
"""
import argparse

import numpy as np
from scipy.special import logsumexp

from .core import gaussian_path_mgf

V0, VC, MU = 2.25, 0.25, 0.8
ETA = VC / V0 - 1.0
DIMS = np.array([1, 4, 16, 64, 256, 1024])

MU1, MU2 = 1.5, -1.0
S1, S2 = 0.5, 1.2


def v1(t): return S1**2 + t
def v2(t): return S2**2 + t


def mixture_mom(t):
    """Mean and variance of the normalized geometric mixture p^{beta,S}_t."""
    A = 0.5 * (1.0 / v1(t) + 1.0 / v2(t))
    h = (MU1 / v1(t) + MU2 / v2(t)) / (1.0 / v1(t) + 1.0 / v2(t))
    return h, 1.0 / A


def zhat(t):
    """hat Z_t = int sqrt(p1 p2) dx (Bhattacharyya coefficient)."""
    a, b = v1(t), v2(t)
    return (np.sqrt(2 * np.sqrt(a * b) / (a + b))
            * np.exp(-(MU1 - MU2)**2 / (4 * (a + b))))


def Vt_mean(m, P, t):
    """E_{N(m,P)}[V_t], with V_t = (s1 - s2)^2 / 8 at S = 1, g = 1."""
    a = 1.0 / v1(t) - 1.0 / v2(t)            # s1 - s2 = a x + b
    b = -(MU1 / v1(t) - MU2 / v2(t))
    return (a * a * (P + m * m) + 2 * a * b * m + b * b) / 8.0


def naive_law_and_certificate(n_steps=400_000):
    """Moment ODEs of the naive composed sampler (linear drift => Gaussian
    marginals), plus the two certificate integrals.

    Reverse SDE dX = b~(X,t) dt + dW run t: 1 -> 0 with
    b~(x,t) = A(t) x + c(t). Moment equations in the parameter t:
        dm/dt = A m + c,   dP/dt = 2 A P - g^2   (g = 1).
    """
    ts = np.linspace(1.0, 0.0, n_steps + 1)
    dt = ts[1] - ts[0]
    m, P = mixture_mom(1.0)                 # init: exact p^{beta,S}_1
    I1 = 0.0                                # int E_naive[V] dt
    for t in ts[:-1]:
        A = 0.5 * (1.0 / v1(t) + 1.0 / v2(t))
        c = -0.5 * (MU1 / v1(t) + MU2 / v2(t))
        I1 += Vt_mean(m, P, t) * (-dt)
        m += (A * m + c) * dt
        P += (2 * A * P - 1.0) * dt
    # int E_{geomix}[V] dt should equal log(Z_1/Z_0) (normalizer identity)
    I2 = sum(Vt_mean(*mixture_mom(t), t) * (-dt) for t in ts[:-1])
    return m, P, I1, I2


def smc_example1(N=60_000, M=2000, seed=0, return_samples=False,
                 ess_times=None):
    """SMC verification: naive composed sampler + FK path weights (killing,
    S = 1). Init from the exact p^{beta,S}_1. No resampling.

    Optional outputs support the publication figures without changing the
    simulation: ``return_samples`` returns the terminal cloud and weights, and
    ``ess_times`` records normalized ESS at the requested noising times.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    h1, P1 = mixture_mom(1.0)
    x = h1 + np.sqrt(P1) * rng.standard_normal(N)
    logW = np.zeros(N)
    snaps = {}
    snap_set = {int(round((1 - t) / dt)): t for t in (0.75, 0.5, 0.25)}
    ess_snap = {1.0: 1.0}
    ess_snap_set = {} if ess_times is None else {
        int(round((1 - float(t)) / dt)): float(t)
        for t in ess_times if float(t) < 1.0
    }
    for i in range(M):
        t = 1.0 - i * dt
        sc1 = -(x - MU1) / v1(t)            # scores s_k = grad log p^(k)_t
        sc2 = -(x - MU2) / v2(t)
        logW += -((sc1 - sc2) ** 2) / 8.0 * dt          # e^{-int V}, V = (Ds)^2/8
        b = -0.5 * (sc1 + sc2)                          # composed drift
        x = x - b * dt + np.sqrt(dt) * rng.standard_normal(N)
        if i + 1 in snap_set:
            snaps[snap_set[i + 1]] = float(np.exp(logsumexp(logW) - np.log(N)))
        if i + 1 in ess_snap_set:
            lwi = logW - logsumexp(logW)
            ess_snap[ess_snap_set[i + 1]] = float(
                np.exp(-logsumexp(2 * lwi))) / N
    lwT = logW - logsumexp(logW)
    wn = np.exp(lwT)
    wmean = float(wn @ x)
    wvar = float(wn @ (x - wmean) ** 2)
    ess = float(np.exp(-logsumexp(2 * lwT))) / N
    EW = float(np.exp(logsumexp(logW) - np.log(N)))
    result = dict(umean=float(x.mean()), uvar=float(x.var()),
                  wmean=wmean, wvar=wvar, ess=ess, EW=EW, snaps=snaps,
                  ess_snap=ess_snap)
    if return_samples:
        result.update(x0=x.copy(), logW=logW.copy(), wn=wn.copy())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small SMC run (N=4e3, M=300)")
    args = ap.parse_args()

    print("=== Example 1: two Gaussian experts, closed forms ===")
    h0, P0 = mixture_mom(0.0)
    print(f"exact geometric mixture m_0/Z_0 = N({h0:.4f}, {P0:.4f})   "
          f"[paper: N(1.1302, 0.4260)]")
    m, P, I1, I2 = naive_law_and_certificate()
    print(f"exact naive law                 = N({m:.4f}, {P:.4f})   "
          f"[paper: N(0.8321, 0.5225)]")

    print("\n=== normalizer chain hat Z_t / hat Z_1 ===")
    Z1 = zhat(1.0)
    for t in (0.75, 0.5, 0.25, 0.0):
        print(f"  t = {t:.2f}: {zhat(t) / Z1:.5f}", end="")
    print("   [paper: 0.92647, 0.83160, 0.70465, 0.52473]")

    print("\n=== KL certificate (naive sampler) ===")
    kl_marg = 0.5 * (P / P0 - 1 + (h0 - m) ** 2 / P0 + np.log(P0 / P))
    log_ratio = np.log(Z1 / zhat(0.0))
    print(f"KL(naive_0 || m_0/Z_0)        = {kl_marg:.6f}   [paper: 0.1155]")
    print(f"int E_naive[V] dt             = {I1:.6f}   [paper: 0.7969]")
    print(f"log(Z_1/Z_0)                  = {log_ratio:.6f}   [paper: 0.6449]")
    print(f"  check: int E_geomix[V] dt   = {I2:.6f} (normalizer identity)")
    print(f"KL(P || Q) = I1 - log(Z1/Z0)  = {I1 - log_ratio:.6f}   [paper: 0.1521]")
    print(f"bound holds: {kl_marg:.4f} <= {I1 - log_ratio:.4f}: "
          f"{kl_marg <= I1 - log_ratio + 1e-12}")

    print("\n=== SMC verification (naive + FK weights, killing case S=1) ===")
    N, M = (4_000, 300) if args.quick else (60_000, 2_000)
    r = smc_example1(N=N, M=M, seed=0)
    print(f"N={N}, M={M}")
    print(f"naive (uniform) data end: N({r['umean']:.3f}, {r['uvar']:.3f})  "
          f"[exact N(0.832, 0.523)]")
    print(f"weighted data end:        N({r['wmean']:.3f}, {r['wvar']:.3f})  "
          f"[exact N(1.130, 0.426)]")
    print(f"terminal ESS/N = {r['ess']:.3f}   [paper: ~0.820]")
    print(f"E[W] = {r['EW']:.5f}   [exact Z_0/Z_1 = {zhat(0.0) / Z1:.5f}]")
    for t, v in sorted(r['snaps'].items(), reverse=True):
        print(f"  mean weight at t={t:.2f}: MC {v:.5f} vs exact {zhat(t) / Z1:.5f}")


if __name__ == "__main__":
    main()


# ---------------------- CFG Gaussian degeneracy calculations -----------------

def smc_cfg_1d(w, N=60_000, M=2000, seed=0, ess_times=None):
    """One-dimensional CFG terminal-IS run on the paper's Gaussian bed."""
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
            "ew": float(np.exp(logsumexp(logw) - np.log(N))),
            "var_logw": float(np.var(logw)), "ess_snap": ess}


def exact_ess_1d(w, M=500):
    e1, _ = gaussian_path_mgf(V0, VC, MU, w, 1.0, M=M)
    e2, ok = gaussian_path_mgf(V0, VC, MU, w, 2.0, M=M)
    return e1 ** 2 / e2 if ok else 0.0


def var_logw_1d(w, M=500):
    """Exact discrete Gaussian path covariance calculation of Var(log W)."""
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


def mgf_finite(v0, vc, mu, w, lam, M, t_min=0.0):
    _, ok = gaussian_path_mgf(v0, vc, mu, w, lam, M=M, t_min=t_min)
    return bool(ok)


def lambda_star_1d(w, v0=V0, vc=VC, mu=MU, M=1000, t_min=0.0, tol=5e-4):
    """Moment index from the Cameron--Martin feasibility boundary."""
    if w <= 1.0:
        return np.inf
    lo, hi = 0.0, 1.0
    while mgf_finite(v0, vc, mu, w, hi, M, t_min):
        lo, hi = hi, 2.0 * hi
        if hi > 1e4:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mgf_finite(v0, vc, mu, w, mid, M, t_min) else (lo, mid)
    return lo


def critical_w(v0, vc, mu, t_min=0.0, M=1200, tol=5e-5):
    """Locate the second-moment boundary directly at lambda=2."""
    lo, hi = 1.0, 2.0
    while mgf_finite(v0, vc, mu, hi, 2.0, M, t_min):
        lo, hi = hi, 2.0 * hi
        if hi > 1e4:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mgf_finite(v0, vc, mu, mid, 2.0, M, t_min) else (lo, mid)
    return 0.5 * (lo + hi)


def naive_law_cfg(v0, vc, mu, w):
    """Continuum data-end moments of the naive composed Gaussian sampler."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        mean, var = y
        a = (1 - w) / (v0 + t) + w / (vc + t)
        c = -w * mu / (vc + t)
        return [a * mean + c, 2 * a * var - 1.0]

    p1 = (1 - w) / (v0 + 1.0) + w / (vc + 1.0)
    sol = solve_ivp(rhs, [1.0, 0.0],
                    [(w * mu / (vc + 1.0)) / p1, 1.0 / p1],
                    rtol=1e-12, atol=1e-14)
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def target_cfg(v0, vc, mu, w):
    precision = (1 - w) / v0 + w / vc
    return (w * mu / vc) / precision, 1.0 / precision


def kl_gauss(m1, v1, m2, v2):
    return 0.5 * (v1 / v2 - 1 + (m2 - m1) ** 2 / v2 + np.log(v2 / v1))


def ess1(v0, vc, mu, w, M=600):
    e1, _ = gaussian_path_mgf(v0, vc, mu, w, 1.0, M)
    e2, ok = gaussian_path_mgf(v0, vc, mu, w, 2.0, M)
    return e1, e1**2 / e2 if ok else 0.0


def wstar_bed(v0, vc, mu, M=300, tol=5e-3):
    """Critical guidance on one Gaussian coordinate bed."""
    lo, hi = 1.01, 3.0
    if lambda_star_1d(hi, v0, vc, mu, M=M) > 2:
        previous = hi
        for candidate in [5, 8, 12, 20, 40, 80, 160]:
            if lambda_star_1d(candidate, v0, vc, mu, M=M) < 2:
                lo, hi = previous, candidate
                break
            previous = candidate
        else:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if lambda_star_1d(mid, v0, vc, mu, M=M) > 2 else (lo, mid)
    return 0.5 * (lo + hi)


def geometric_gaussian(mus, variances, betas, t=0.0):
    """Log mass, mean, and variance of a geometric Gaussian product."""
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
    return float(betas @ log_components - log_q_at_zero), mean, var


def vp_noise_end(v0, vc, mu, w, T, beta_min=0.1, beta_max=2.0):
    """Noise-end KLs for the long-horizon VP check in the Gaussian example."""
    integrated_beta = T * 0.5 * (beta_min + beta_max)
    alpha = np.exp(-0.5 * integrated_beta)
    v0T = 1.0 + alpha**2 * (v0 - 1.0)
    vcT = 1.0 + alpha**2 * (vc - 1.0)
    mucT = alpha * mu
    _, geometric_mean, geometric_var = geometric_gaussian(
        [0.0, mucT], [v0T, vcT], [1.0 - w, w])
    _, target_mean, target_var = geometric_gaussian(
        [0.0, mu], [v0, vc], [1.0 - w, w])
    exact_mean = alpha * target_mean
    exact_var = 1.0 + alpha**2 * (target_var - 1.0)

    def kl_standard(mean, var):
        return 0.5 * (var + mean * mean - 1.0 - np.log(var))

    return dict(T=T, alpha=alpha,
                geometric_kl=kl_standard(geometric_mean, geometric_var),
                exact_kl=kl_standard(exact_mean, exact_var))
