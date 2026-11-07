#!/usr/bin/env python3
"""Deterministic Feynman--Kac moments for the two-dimensional CFG beds.

Solves the conditional moment PDE in log form on [-L,L]^2,

  h_t = a . grad(h) + (1/2) Delta h + (1/2)|grad(h)|^2
        + lambda * theta * |s^c_t-s^0_t|^2,
  a = (1-w)s^0 + w s^c,   theta = w(w-1)/2,   h(.,0)=0.

The terminal moment is obtained by quadrature against the exact tempered
noise-end density.  A bounded-grid value is always finite: moment divergence is
diagnosed by growth as L increases, not by overflow or by one finite grid value.

Examples
--------
python3 gmm2d_exact_moments.py --bed gaussian --w 1.5 --lam 2 --L 4 6
python3 gmm2d_exact_moments.py --bed gmm --w 1.5 --lam 2.7 --L 4 6 8 10
python3 gmm2d_exact_moments.py --bed gmm --w 1.75 --lam 2 --L 4 6 8 10

The paper scans use dx=0.1 and dt_scale=0.25.  Runtime is roughly 12, 23, 41,
and 73 seconds per solve at L=4, 6, 8, and 10, respectively, on one CPU core.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.special import logsumexp

from .core import GAUSS_BED, GMM_BED, gaussian_path_mgf, p0_pdf, pc_pdf


def score_grid(x, y, means, variances, weights, t):
    """Exact noised-mixture score on the tensor grid x times y."""
    xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    v = variances + t
    d2 = ((xy * xy).sum(1)[:, None] - 2.0 * (xy @ means.T)
          + (means * means).sum(1)[None, :])
    logp = (-0.5 * d2 / v[None, :] - np.log(2.0 * np.pi * v[None, :])
            + np.log(weights)[None, :])
    post = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    score = -((post @ (1.0 / v))[:, None] * xy - post @ (means / v[:, None]))
    n = len(x)
    return score[:, 0].reshape(n, n), score[:, 1].reshape(n, n)


def grid_size(L, dx):
    """Odd centered grid with spacing as close as possible to dx."""
    n = int(round(2.0 * L / dx))
    return n + 1 if n % 2 == 0 else n


def solve_logmoment(bed, w, lam, L, dx=0.1, safety=0.18, dt_scale=0.25,
                    verbose=False):
    """Return log E[W^lambda] on one bounded grid.

    Neumann-like artificial boundaries are used for h.  The explicit step is
    safety*dx^2*dt_scale, adjusted upward only enough to give an integer number
    of steps on [0,1].  Reducing dt_scale stabilizes the stiff small-noise end.
    """
    n = grid_size(L, dx)
    x = np.linspace(-L, L, n)
    actual_dx = x[1] - x[0]
    xy = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    theta = 0.5 * w * (w - 1.0)
    dt = safety * actual_dx * actual_dx * dt_scale
    steps = int(np.ceil(1.0 / dt))
    dt = 1.0 / steps
    h = np.zeros((n, n))

    for j in range(steps):
        t = j * dt
        s0x, s0y = score_grid(x, x, bed["m0"], bed["v0"], bed["w0"], t)
        scx, scy = score_grid(x, x, bed["mc"], bed["vc"], bed["wc"], t)
        ax = (1.0 - w) * s0x + w * scx
        ay = (1.0 - w) * s0y + w * scy
        reaction = lam * theta * ((scx - s0x) ** 2 + (scy - s0y) ** 2)

        lap = np.zeros_like(h)
        hx = np.zeros_like(h)
        hy = np.zeros_like(h)
        lap[1:-1, 1:-1] = (
            h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[1:-1, :-2]
            - 4.0 * h[1:-1, 1:-1]) / actual_dx**2
        hx[1:-1, 1:-1] = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2.0 * actual_dx)
        hy[1:-1, 1:-1] = (h[1:-1, 2:] - h[1:-1, :-2]) / (2.0 * actual_dx)
        hn = h + dt * (ax * hx + ay * hy
                       + 0.5 * (lap + hx * hx + hy * hy) + reaction)
        hn[0, :] = hn[1, :]
        hn[-1, :] = hn[-2, :]
        hn[:, 0] = hn[:, 1]
        hn[:, -1] = hn[:, -2]
        h = hn

        if j % 25 == 0 or j == steps - 1:
            hmax = float(np.max(np.abs(h)))
            if not np.isfinite(hmax) or hmax > 1e8:
                return dict(logM=np.inf, hmax=hmax, n=n, L=L, steps=steps,
                            status="numerical_blowup", t=t)
        if verbose and (j % max(1, steps // 10) == 0 or j == steps - 1):
            print(f"t={t + dt:.5f}/{steps} max|h|={np.max(np.abs(h)):.4g}",
                  flush=True)

    logm = ((1.0 - w) * np.log(p0_pdf(xy, bed, 1.0) + 1e-300)
            + w * np.log(pc_pdf(xy, bed, 1.0) + 1e-300))
    loginit = logm - (logsumexp(logm) + 2.0 * np.log(actual_dx))
    # The cell measure is essential: omitting it introduces -2 log(dx).
    logM = 2.0 * np.log(actual_dx) + logsumexp(loginit + h.ravel())
    return dict(logM=float(logM), hmax=float(np.max(np.abs(h))), n=n, L=L,
                steps=steps, status="finite", t=1.0)


def gaussian_reference(w, lam, M=2000):
    """Exact discrete-Gaussian path moment on the moment-matched 2D bed."""
    mx, okx = gaussian_path_mgf(2.25, 0.25, 0.8, w, lam, M=M)
    my, oky = gaussian_path_mgf(2.25, 0.25, 0.0, w, lam, M=M)
    if not (okx and oky) or not np.isfinite(mx * my):
        return math.inf
    return math.log(mx * my)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bed", choices=("gmm", "gaussian"), default="gmm")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--L", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    ap.add_argument("--dx", type=float, default=0.1)
    ap.add_argument("--dt-scale", type=float, default=0.25)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", help="optional CSV output path")
    args = ap.parse_args()

    bed = GMM_BED if args.bed == "gmm" else GAUSS_BED
    rows = []
    for L in args.L:
        result = solve_logmoment(bed, args.w, args.lam, L=L, dx=args.dx,
                                 dt_scale=args.dt_scale, verbose=args.verbose)
        row = dict(bed=args.bed, w=args.w, lam=args.lam, dx=args.dx,
                   dt_scale=args.dt_scale, **result)
        if args.bed == "gaussian":
            row["gaussian_path_logM"] = gaussian_reference(args.w, args.lam)
        rows.append(row)
        print(row, flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        fields = list(rows[0])
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
