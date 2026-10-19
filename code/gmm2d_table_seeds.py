#!/usr/bin/env python3
"""
GMM table with seed statistics (Table tab:gmm2d of the paper, revised form).

The terminal ESS/N of the GMM bed is heavy-tailed near and above the
critical scale w*_GMM ~ 1.55 (the lambda* in (2,4) regime), so single-seed
values are meaningless: this driver reports mean +/- s.d., median and range
over seeds, alongside E[W] against the quadrature value of Z_0/Z_1.

Default settings reproduce the revised table: N=6e4, M=2000; w = 0.5
(4 seeds), 1.5 / 1.75 / 2.0 (8 seeds each); w = 1 is exact (ESS = E W = 1).
Optionally (--n1e5) spot-check w = 1.75 at N = 1e5 (collapsed draws).

Reference values (authors' runs):
  w=0.5 : ESS/N 0.793 +- 0.001,  E W 0.533 +- 0.001  (quad 0.533)
  w=1.5 : median 0.552, 7/8 seeds in [0.494, 0.578], one collapse 0.004;
          E W 1.971 +- 0.048 (quad 1.962)
  w=1.75: ESS/N 0.196 +- 0.036, range [0.144, 0.262];
          E W 2.728 +- 0.028 (quad 2.746); at N=1e5: seed range 0.005-0.15
  w=2.0 : ESS/N 0.065 +- 0.033;  E W 3.795 +- 0.067 (quad 3.836)
"""
import argparse
import csv
import os

import numpy as np

from gmm2d import GMM_BED, Zratio_grid, smc_cfg, stable_seed

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "output")


def run_grid(ws_seeds, N, M):
    rows = []
    for w, seeds in ws_seeds:
        ess, ew = [], []
        for s in seeds:
            r = smc_cfg(GMM_BED, w, N=N, M=M, seed=stable_seed("tab", w, s))
            ess.append(r["ess"]); ew.append(r["EW"])
            print(f"  w={w:<5} seed={s}: ESS/N={r['ess']:.4f}  E W={r['EW']:.3f}",
                  flush=True)
        z = Zratio_grid(GMM_BED, w)
        rows.append(dict(w=w, seeds=list(seeds), ess=ess, ew=ew, z01=z))
        e = np.array(ess); u = np.array(ew)
        print(f"w={w:<5} ESS/N mean={e.mean():.3f} sd={e.std():.3f} "
              f"median={np.median(e):.3f} range=[{e.min():.3f},{e.max():.3f}] | "
              f"E W mean={u.mean():.3f} sd={u.std():.3f} (quad {z:.3f})\n",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="N=6e3, M=300, 2 seeds per w")
    ap.add_argument("--n1e5", action="store_true",
                    help="add the w=1.75, N=1e5 collapse spot-check (3 seeds)")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    if args.quick:
        N, M = 6_000, 300
        ws_seeds = [(1.5, range(2)), (1.75, range(2))]
    else:
        N, M = 60_000, 2_000
        ws_seeds = [(0.5, range(4)), (1.5, range(8)),
                    (1.75, range(8)), (2.0, range(8))]

    print(f"=== GMM bed, N={N}, M={M} (w=1 is exact: ESS/N = E W = 1) ===")
    rows = run_grid(ws_seeds, N, M)

    if args.n1e5 and not args.quick:
        print("=== collapse spot-check: w=1.75, N=1e5, seeds 0-2 ===")
        rows += run_grid([(1.75, range(3))], 100_000, M)

    path = os.path.join(OUTDIR, "gmm2d_table_seeds.csv")
    with open(path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["w", "seed", "ESS_over_N", "EW_MC", "EW_quadrature",
                      "N", "M"])
        for row in rows:
            for s, e, u in zip(row["seeds"], row["ess"], row["ew"]):
                wtr.writerow([row["w"], s, f"{e:.5f}", f"{u:.5f}",
                              f"{row['z01']:.5f}", N, M])
    print("wrote", path)


if __name__ == "__main__":
    main()
