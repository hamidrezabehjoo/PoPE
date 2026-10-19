#!/usr/bin/env python3
"""
Finite-N scaling of the terminal ESS/N (data for Figure fig:gmm2dscale(a)).

Below w* the terminal ESS/N plateaus as N grows; above w* it collapses --
with heavy seed-to-seed scatter near the threshold (the lambda* in (2,4)
regime), hence multiple seeds per (w, N).

GMM bed:    w in {1.5, 1.6, 1.75}, N in {3e4, 6e4, 1e5, 3e5}
            (3 seeds for N <= 1e5; seed 0 only at N = 3e5)
Gaussian controls (on the correct sides of w* ~ 1.95):
            w = 1.75 and w = 2.0 at N in {6e4, 2.4e5}, seed 0

Reference values (authors' runs, seed 0):
  GMM w=1.5: 0.55 (6e4), 0.47 (3e5)          -- plateau
  GMM w=1.6: 0.40 (3e4) -> 0.37 (1e5)        -- slow decay
  GMM w=1.75: 0.15 (3e4), 0.22 (6e4), 0.03 (1e5)  -- spiky collapse
  Gaussian w=1.75: ~0.5 across N (below w*); w=2.0: finite-N artifacts only
"""
import argparse
import csv
import os

from gmm2d import GAUSS_BED, GMM_BED, smc_cfg, stable_seed

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="N in {3e3, 6e3}, M=300, 1 seed")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.quick:
        plan = [(GMM_BED, w, N, [0]) for w in (1.5, 1.75) for N in (3_000, 6_000)]
        M = 300
    else:
        plan = [(GMM_BED, w, N, ([0] if N == 300_000 else range(3)))
                for w in (1.5, 1.6, 1.75)
                for N in (30_000, 60_000, 100_000, 300_000)]
        plan += [(GAUSS_BED, w, N, [0])
                 for w in (1.75, 2.0) for N in (60_000, 240_000)]
        M = 2000

    path = os.path.join(OUTDIR, "gmm2d_scaling.csv")
    with open(path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["bed", "w", "N", "seed", "ESS_over_N", "EW_MC", "M"])
        for bed, w, N, seeds in plan:
            for s in seeds:
                r = smc_cfg(bed, w, N=N, M=M, seed=stable_seed("ns", bed["name"], w, N, s))
                print(f"{bed['name']:>8} w={w:<5} N={N:<7} seed={s}: "
                      f"ESS/N={r['ess']:.4f}  E W={r['EW']:.3f}", flush=True)
                wtr.writerow([bed["name"], w, N, s, f"{r['ess']:.5f}",
                              f"{r['EW']:.5f}", M])
    print("wrote", path)


if __name__ == "__main__":
    main()
