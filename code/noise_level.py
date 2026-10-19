#!/usr/bin/env python3
"""
Noise-level (nu-expert) theorem: SMC check of the data-end laws.

Paper: the proposed theorem in the degeneracy/related discussion --
expert k run with noise nu_t > 0 and unchanged marginals,
  dX = [ f - 1/2 (g_t^2 + nu_t^2) grad log p^(k)_t ] dt + nu_t dW,
has a time reversal that is a *forward* diffusion with noise nu and drift
f_k^nu = f + 1/2 (nu^2 - g^2) grad log p^(k)_t. The path product of the
nu-experts is then covered by the forward-composition theorem with
  U^nu_t = (nu^2 - g^2)^2 / (8 nu^2) Var_alpha[grad log p^(k)_t]
         = ((nu^2 - g^2)/(nu^2 + g^2))^2  V~^{b,nu}_t ,
  V~^{b,nu}_t = (g^2 + nu^2)^2 / (8 nu^2) Var_alpha[grad log p^(k)_t].
The data end is the exact geometric mixture iff nu = g (Anderson's
reversal); the probability-flow limit nu -> 0 is singular
(U^nu ~ g^4 / (8 nu^2) -> inf).

Check on Example 1 (f=0, g=1, VE; experts N(1.5, 0.25) and N(-1, 1.44);
beta = 1/2 each): FK-weighted SMC of the nu-reverse composition, killing
weight exp(-int V~^{b,nu} dt). Expected data ends (paper):
  nu=0.25: N(1.649, 0.231)   nu=0.5: N(1.262, 0.378)   nu=1: N(1.130, 0.426)
  nu=1.5:  N(1.162, 0.417)   nu=3:   N(1.252, 0.395)
(nu = g = 1 recovers the exact geometric mixture N(1.1302, 0.4260).)
"""
import argparse

import numpy as np
from scipy.special import logsumexp

MU1, MU2 = 1.5, -1.0
V1_0, V2_0 = 0.25, 1.44            # data-end variances (sigma_1^2, sigma_2^2)


def smc_nu(nu, N=300_000, M=2000, seed=42):
    """FK-weighted SMC of the nu-reverse composition (Example-1 experts,
    beta=1/2, S=1). Init: exact tempered noise end (geometric mixture at
    t=1, Gaussian). No resampling."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    v1 = lambda t: V1_0 + t
    v2 = lambda t: V2_0 + t
    Pi1 = 0.5 / v1(1.0) + 0.5 / v2(1.0)                # tempered precision
    h1 = (0.5 * MU1 / v1(1.0) + 0.5 * MU2 / v2(1.0)) / Pi1
    x = h1 + np.sqrt(1.0 / Pi1) * rng.standard_normal(N)
    logW = np.zeros(N)
    c_kill = (1.0 + nu**2) ** 2 / (8.0 * nu**2) * 0.25  # (g^2+nu^2)^2/(8 nu^2) * Var coef 1/4
    for i in range(M):
        t = 1.0 - i * dt
        sc1 = -(x - MU1) / v1(t)
        sc2 = -(x - MU2) / v2(t)
        d = sc1 - sc2
        logW += -c_kill * d * d * dt                    # e^{-int V~^{b,nu} dt}
        b = -0.5 * (1.0 + nu**2) * 0.5 * (sc1 + sc2)    # composed nu-drift
        x = x - b * dt + nu * np.sqrt(dt) * rng.standard_normal(N)
    lwT = logW - logsumexp(logW)
    wn = np.exp(lwT)
    m = float(wn @ x)
    v = float(wn @ (x - m) ** 2)
    return m, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="N=3e4, M=300")
    args = ap.parse_args()
    N, M = (30_000, 300) if args.quick else (300_000, 2_000)

    paper = {0.25: (1.649, 0.231), 0.5: (1.262, 0.378), 1.0: (1.130, 0.426),
             1.5: (1.162, 0.417), 3.0: (1.252, 0.395)}
    print(f"=== nu-expert data ends by FK-weighted SMC (N={N}, M={M}) ===")
    for nu, (mp, vp) in paper.items():
        m, v = smc_nu(nu, N=N, M=M)
        tag = "  <- exact geometric mixture (nu = g)" if nu == 1.0 else ""
        print(f"  nu = {nu:>4}: N({m:.3f}, {v:.3f})   "
              f"[paper N({mp:.3f}, {vp:.3f})]{tag}")
    print("\nsingularity: U^nu ~ g^4/(8 nu^2) as nu -> 0; "
          "data end exact iff nu = g = 1.")


if __name__ == "__main__":
    main()
