#!/usr/bin/env python3
"""
Example 1 (two Gaussian experts): closed forms, SMC verification, and the
KL certificate for the naive sampler.

Paper: "When Diffusion Composition Fails: Exact Products of Path Measures",
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


def smc_example1(N=60_000, M=2000, seed=0):
    """SMC verification: naive composed sampler + FK path weights (killing,
    S = 1). Init from the exact p^{beta,S}_1. No resampling."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    h1, P1 = mixture_mom(1.0)
    x = h1 + np.sqrt(P1) * rng.standard_normal(N)
    logW = np.zeros(N)
    snaps = {}
    snap_set = {int(round((1 - t) / dt)): t for t in (0.75, 0.5, 0.25)}
    for i in range(M):
        t = 1.0 - i * dt
        sc1 = -(x - MU1) / v1(t)            # scores s_k = grad log p^(k)_t
        sc2 = -(x - MU2) / v2(t)
        logW += -((sc1 - sc2) ** 2) / 8.0 * dt          # e^{-int V}, V = (Ds)^2/8
        b = -0.5 * (sc1 + sc2)                          # composed drift
        x = x - b * dt + np.sqrt(dt) * rng.standard_normal(N)
        if i + 1 in snap_set:
            snaps[snap_set[i + 1]] = float(np.exp(logsumexp(logW) - np.log(N)))
    lwT = logW - logsumexp(logW)
    wn = np.exp(lwT)
    wmean = float(wn @ x)
    wvar = float(wn @ (x - wmean) ** 2)
    ess = float(np.exp(-logsumexp(2 * lwT))) / N
    EW = float(np.exp(logsumexp(logW) - np.log(N)))
    return dict(umean=float(x.mean()), uvar=float(x.var()),
                wmean=wmean, wvar=wvar, ess=ess, EW=EW, snaps=snaps)


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
