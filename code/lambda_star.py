#!/usr/bin/env python3
"""
Moment/Pareto index lambda*(w) for the CFG Gaussian bed, and the exact
asymptotic ESS/N row.

Paper: degeneracy section -- lambda*(w) := sup{ lam : E W^lam < inf },
computed from the discrete Cameron-Martin condition of the exact quadratic
path action (gaussian_path_mgf). The 2D bed factorizes across axes: the MGF
is finite iff it is finite on both the mean axis (mu=0.8) and the variance
axis (mu=0.0).

Bed: p0 = N(0, 2.25 I_2), pc = N((0.8,0), 0.25 I_2) (Figure fig:ess).

Expected output (paper values):
  w            1.25   1.5    2      3      5
  lambda*(w)   5.77   3.19   1.94   1.37   1.13
  w* (level set lambda* = 2) ~ 1.95

  exact ESS/N = (E W)^2 / E W^2 at w = 0.5, 1, 1.5, 1.75, 2:
  0.903, 1.000, 0.765, 0.392, 0 (Table tab:ess / tab:gmm2d).
"""
import argparse

import numpy as np

from gmm2d import gaussian_path_mgf

V0, VC, MU = 2.25, 0.25, 0.8


def mgf_finite_2d(w, lam, M):
    _, okx = gaussian_path_mgf(V0, VC, MU, w, lam, M)
    _, oky = gaussian_path_mgf(V0, VC, 0.0, w, lam, M)
    return okx and oky


def lambda_star(w, M=1000, tol=5e-4):
    """sup{ lam : E W^lam < inf } by bisection on the Cameron-Martin
    feasibility condition. For w <= 1 the path weight is bounded by 1
    (killing regime) and lambda* = inf."""
    if w <= 1.0:
        return np.inf
    lo, hi = 0.0, 1.0
    while mgf_finite_2d(w, hi, M):
        lo = hi
        hi *= 2.0
        if hi > 1e4:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mgf_finite_2d(w, mid, M):
            lo = mid
        else:
            hi = mid
    return lo


def exact_ess(w, M=500):
    """(E W)^2 / E W^2 for the 2D Gaussian bed (product of the two axes)."""
    e1x, _ = gaussian_path_mgf(V0, VC, MU, w, 1.0, M)
    e1y, _ = gaussian_path_mgf(V0, VC, 0.0, w, 1.0, M)
    e2x, okx = gaussian_path_mgf(V0, VC, MU, w, 2.0, M)
    e2y, oky = gaussian_path_mgf(V0, VC, 0.0, w, 2.0, M)
    EW1 = e1x * e1y
    EW2 = e2x * e2y if (okx and oky) else np.inf
    return EW1, EW2, (EW1**2 / EW2 if np.isfinite(EW2) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="coarse grid M=400, skip w* root-find")
    args = ap.parse_args()
    M = 400 if args.quick else 1000

    print(f"=== lambda*(w) = sup{{lam : E W^lam < inf}}  (Cameron-Martin, M={M}) ===")
    ws = [1.25, 1.5, 2.0, 3.0, 5.0]
    vals = {}
    for w in ws:
        vals[w] = lambda_star(w, M=M)
        print(f"  w = {w:>4}: lambda* = {vals[w]:.4f}")
    print("paper:  w = 1.25, 1.5, 2, 3, 5  ->  5.77, 3.19, 1.94, 1.37, 1.13")

    if not args.quick:
        print("\n=== w* as the level set lambda*(w*) = 2 ===")
        lo, hi = 1.5, 2.0                      # lambda*(1.5)>2>lambda*(2)
        while hi - lo > 2e-3:
            mid = 0.5 * (lo + hi)
            if lambda_star(mid, M=M, tol=1e-3) > 2.0:
                lo = mid
            else:
                hi = mid
        print(f"  w* = {0.5 * (lo + hi):.4f}   [paper: ~1.95]")

    print("\n=== exact asymptotic ESS/N = (E W)^2 / E W^2 ===")
    for w in [0.5, 1.0, 1.5, 1.75, 2.0]:
        EW1, EW2, ess = exact_ess(w, M=M)
        EW2s = f"{EW2:.4f}" if np.isfinite(EW2) else "inf"
        print(f"  w = {w:>4}: E W = {EW1:.4f}, E W^2 = {EW2s}, ESS/N = {ess:.3f}")
    print("paper ESS/N row: 0.903, 1.000, 0.765, 0.392, 0")


if __name__ == "__main__":
    main()
