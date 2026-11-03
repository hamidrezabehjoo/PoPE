#!/usr/bin/env python3
"""Secondary w=1.5 finite-moment projection sanity check."""
import argparse
import json

import numpy as np

from controlled_gmm import GridDoob2D, write_csv
from gmm2d import GMM_BED, stable_seed
from learned_doob import (OUT, diagnostic_row, generate_naive_labels,
                          grid_oracle_diagnostic, learned_sample, psis_k,
                          rollout_objective, save_json, split_labels,
                          smc_sample, terminal_is_sample, train_value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paths", type=int, default=60_000)
    ap.add_argument("--train-steps", type=int, default=8000)
    ap.add_argument("--eval-N", type=int, default=20_000)
    ap.add_argument("--M", type=int, default=600)
    ap.add_argument("--grid-n", type=int, default=320)
    args = ap.parse_args()
    w = 1.5
    grid = GridDoob2D(GMM_BED, w, n=args.grid_n)
    labels = generate_naive_labels(
        grid, args.paths, M=args.M, seed=stable_seed("labels", w, args.seed))
    train_idx, _ = split_labels(labels, args.seed + 101)
    k = psis_k(labels["terminal_logw"][train_idx])
    scaling = []
    for n in (2000, 5000, 10_000, 20_000, 40_000, 60_000):
        if n > args.paths:
            continue
        lw = labels["terminal_logw"][train_idx][:n]
        lw = lw - np.max(lw)
        wn = np.exp(lw) / np.exp(lw).sum()
        scaling.append(dict(N=n, ess=float(1.0 / np.sum(wn**2) / n)))
    gate = bool(k < 0.5)  # ESS scaling is printed and inspected separately.
    print(f"w=1.5 moment gate: PSIS k={k:.4g}, scaling={scaling}, pass={gate}", flush=True)
    if not gate:
        save_json(OUT / f"sanity_w1.5_seed{args.seed}_gate.json",
                  dict(psis_k=k, scaling=scaling, gate=gate,
                       decision="direct MSE not licensed by the pre-registered gate"))
        return
    rows, oracle, training = [], {}, {}
    eval_seed = stable_seed("learned-evaluation", w, args.seed)
    for penalty in (0.0, 1e-4):
        name = "MSE" if penalty == 0 else "MSE + gradient penalty"
        model, info = train_value(
            labels, seed=args.seed, family="mse", max_steps=args.train_steps,
            grad_penalty=penalty)
        C, se = rollout_objective(model, grid, N=2048, M=args.M,
                                  seed=stable_seed("sanity-C", w, args.seed, penalty))
        r = learned_sample(model, grid, N=args.eval_N, M=args.M, seed=eval_seed)
        rows.append(diagnostic_row(name, r, grid, eval_seed, w, args.M,
                                   stage="sanity"))
        oracle[name] = grid_oracle_diagnostic(model, grid)
        training[name] = dict(info=info, rollout_C=C, rollout_C_se=se)
    terminal = terminal_is_sample(grid, args.eval_N, args.M, eval_seed)
    rows.append(diagnostic_row("terminal IS", terminal, grid, eval_seed, w, args.M))
    smc = smc_sample(grid, args.eval_N, args.M, eval_seed)
    rows.append(diagnostic_row("adaptive SMC", smc, grid, eval_seed, w, args.M,
                               raw_weights=False))
    write_csv(OUT / f"sanity_w1.5_seed{args.seed}.csv", rows)
    save_json(OUT / f"sanity_w1.5_seed{args.seed}.json",
              dict(w=w, seed=args.seed, psis_k=k, scaling=scaling, gate=gate,
                   training=training, oracle=oracle, rows=rows))
    for row in rows:
        print(row["method"], "ESS", row["ess"], "MMD2", row["mmd2"],
              "SW1", row["sliced_w1"], "TV", row["tv"], flush=True)


if __name__ == "__main__":
    main()
