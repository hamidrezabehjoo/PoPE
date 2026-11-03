#!/usr/bin/env python3
"""Refinement and replicate checks for the second-stage controlled numerics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from controlled_gaussian import critical_w, lambda_star
from controlled_gmm import GridDoob2D, evaluate
from gmm2d import GMM_BED, stable_seed

OUT = Path(__file__).resolve().parents[1] / "output"


def write_csv(name, rows):
    if rows:
        with open(OUT / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-gmm", action="store_true")
    ap.add_argument("--skip-gaussian", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if not args.skip_gaussian:
        rows = []
        for w in [5.0, 10.0, 20.0, 50.0]:
            M = 160 if args.quick else 400
            lx = lambda_star(2.25, 0.25, 0.8, w, M=M, tol=1e-4)
            ly = lambda_star(2.25, 0.25, 0.0, w, M=M, tol=1e-4)
            rows.append(dict(w=w, M=M, lambda_star_mean_axis=lx,
                             lambda_star_variance_axis=ly,
                             lambda_star=min(lx, ly)))
            print(f"high guidance: w={w}, lambda*={min(lx, ly):.6f}")
        write_csv("lambda_star_high_guidance_refined.csv", rows)

        rows = []
        for vc, tmin in [(0.25, 0.0), (0.25, 0.1), (0.01, 0.0),
                         (0.01, 0.1), (0.001, 0.0), (0.001, 0.001),
                         (0.001, 0.01), (0.001, 0.1)]:
            M = 300 if args.quick else 1200
            value = critical_w(2.25, vc, 0.8, t_min=tmin, M=M, tol=5e-5)
            rows.append(dict(vc=vc, t_min=tmin, M=M, w_star=value))
            print(f"early stopping: vc={vc}, t_min={tmin}, w*={value:.6f}")
        write_csv("early_stopping_refined.csv", rows)

    if not args.skip_gmm:
        rows = []
        configs = [(128, 120, 3000)] if args.quick else [
            (192, 300, 20_000), (320, 600, 20_000), (448, 900, 20_000)]
        for n, M, N in configs:
            grid = GridDoob2D(GMM_BED, 3.0, n=n)
            r = grid.sample(N=N, M=M, rho=1.0,
                            seed=stable_seed("gmm-refinement", n, M))
            metrics = evaluate(r, grid, diag_n=min(2000, N),
                               diag_seed=stable_seed("gmm-refinement-diagnostic"))
            rows.append(dict(n=n, M=M, N=N, ess=r["ess"], outside=r["outside"],
                             max_abs_score=r["max_abs_score"], **metrics))
            print(f"GMM refinement n={n}, M={M}: {metrics}")
        write_csv("control_gmm_refinement.csv", rows)

        rows = []
        replicate_seeds = [0] if args.quick else [0, 1, 2]
        n, M, N = (128, 120, 3000) if args.quick else (320, 600, 20_000)
        for seed in replicate_seeds:
            grid = GridDoob2D(GMM_BED, 3.0, n=n)
            r = grid.sample(N=N, M=M, rho=1.0,
                            seed=stable_seed("gmm-doob-replicate", seed))
            metrics = evaluate(r, grid, diag_n=min(2000, N),
                               diag_seed=stable_seed("gmm-doob-replicate-diagnostic", seed))
            rows.append(dict(seed=seed, n=n, M=M, N=N, ess=r["ess"],
                             outside=r["outside"], **metrics))
            print(f"GMM replicate {seed}: {metrics}")
        write_csv("controlled_gmm_replicates.csv", rows)


if __name__ == "__main__":
    main()
