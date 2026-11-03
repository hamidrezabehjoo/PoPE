#!/usr/bin/env python3
"""Three-seed GMM summaries reported in the theory-first manuscript.

This driver reproduces the matched-budget adaptive-SMC comparison and the
shared-prior initialization table.  It writes both per-replicate CSVs and
mean/sample-standard-deviation summaries.
"""
from __future__ import annotations

import argparse

import numpy as np

from gmm2d import GMM_BED, smc_cfg, stable_seed
from controlled_gmm import GridDoob2D, evaluate, write_csv
from theory_control_checks import (OUT, timed_call, smc_resampling_cfg,
                                   terminal_standard_prior,
                                   doob_standard_prior)


def summarize(rows, group_keys, metric_keys):
    summaries = []
    groups = sorted({tuple(r[k] for k in group_keys) for r in rows})
    for group in groups:
        selected = [r for r in rows if tuple(r[k] for k in group_keys) == group]
        out = dict(zip(group_keys, group))
        out["n_replicates"] = len(selected)
        for key in metric_keys:
            values = np.array([float(r[key]) for r in selected])
            out[key + "_mean"] = float(values.mean())
            out[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(out)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N, M, n = (4000, 160, 160) if args.quick else (20_000, 600, 320)
    diag_n = min(2000, N)
    OUT.mkdir(exist_ok=True)
    grid = GridDoob2D(GMM_BED, 3.0, n=n)

    rows = []
    for rep in range(3):
        for method in ["terminal IS", "adaptive SMC", "exact Doob", "approx Doob IS"]:
            seed = stable_seed("theory-replicate-all", method, rep)

            def run():
                if method == "terminal IS":
                    r0 = smc_cfg(GMM_BED, 3.0, N=N, M=M, seed=seed, snapshots=())
                    return dict(x=r0["x0"], wn=r0["wn"], ess=r0["ess"],
                                log_weight_sd=r0["sdlogW"], n_resample=0)
                if method == "adaptive SMC":
                    return smc_resampling_cfg(GMM_BED, 3.0, N=N, M=M, seed=seed)
                return grid.sample(N=N, M=M, seed=seed,
                                   rho=1.0 if method == "exact Doob" else 0.8)

            r, seconds = timed_call(run)
            metrics = evaluate(r, grid, diag_n=diag_n,
                               diag_seed=stable_seed("theory-replicate-all-diagnostic", rep))
            row = dict(replicate=rep, method=method, N=N, M=M, grid=n,
                       seconds=seconds, ess=r["ess"],
                       n_resample=r.get("n_resample", 0), **metrics)
            rows.append(row)
            print("SMC", rep, method, f"ESS={r['ess']:.5g}",
                  f"MMD2={metrics['mmd2']:.6g}", f"SW1={metrics['sliced_w1']:.6g}",
                  flush=True)
            write_csv(OUT / "smc_controlled_comparison_replicated.csv", rows)
    metrics = ["ess", "mmd2", "sliced_w1", "mean_error", "tv", "seconds", "n_resample"]
    write_csv(OUT / "smc_controlled_summary.csv",
              summarize(rows, ["method"], metrics))

    configs = [
        ("terminal IS", "p_1^{beta,S}", True),
        ("terminal IS", "N(0,I)", True),
        ("terminal IS", "N(0,I)", False),
        ("exact Doob", "q_1", True),
        ("exact Doob", "N(0,I)", True),
        ("exact Doob", "N(0,I)", False),
    ]
    init_rows = []
    for rep in range(3):
        for method, init, corrected in configs:
            seed = stable_seed("shared-prior-replicate", method, init, corrected, rep)
            if method == "terminal IS" and init == "p_1^{beta,S}":
                r0 = smc_cfg(GMM_BED, 3.0, N=N, M=M, seed=seed, snapshots=())
                r = dict(x=r0["x0"], wn=r0["wn"], ess=r0["ess"])
            elif method == "terminal IS":
                r = terminal_standard_prior(grid, N, M, seed, corrected)
            elif init == "q_1":
                r = grid.sample(N=N, M=M, seed=seed, rho=1.0)
            else:
                r = doob_standard_prior(grid, N, M, seed, corrected)
            metrics0 = evaluate(r, grid, diag_n=diag_n,
                                diag_seed=stable_seed("shared-prior-rep-diagnostic", rep))
            row = dict(replicate=rep, method=method, initialization=init,
                       noise_end_corrected=corrected, N=N, M=M, grid=n,
                       ess=r["ess"], **metrics0)
            init_rows.append(row)
            print("INIT", rep, method, init, corrected, f"ESS={r['ess']:.5g}",
                  f"MMD2={metrics0['mmd2']:.6g}", f"SW1={metrics0['sliced_w1']:.6g}",
                  flush=True)
            write_csv(OUT / "shared_prior_initialization_replicated.csv", init_rows)
    write_csv(OUT / "shared_prior_initialization_summary.csv",
              summarize(init_rows, ["method", "initialization", "noise_end_corrected"],
                        ["ess", "mmd2", "sliced_w1", "mean_error", "tv"]))


if __name__ == "__main__":
    main()
