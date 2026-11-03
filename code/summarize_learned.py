#!/usr/bin/env python3
"""Aggregate learned-control replicates and apply the pre-registered rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output" / "learned_doob"
METRICS = ["ess", "mmd2", "sliced_w1", "tv", "mean_weight_ratio",
           "raw_second_ratio", "psis_k", "max_weight_share", "mean_error",
           "sampling_seconds"]


def load_metrics(w=3.0, seeds=(0, 1, 2)):
    frames = []
    for seed in seeds:
        path = OUT / f"metrics_w{w:g}_seed{seed}.csv"
        frame = pd.read_csv(path)
        frame["training_seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    rows = []
    for method, group in df.groupby("method", sort=False):
        row = {"method": method, "n_seeds": group["training_seed"].nunique()}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def success_assessment(df, learned="two-stage learned + IS", baseline="terminal IS"):
    means = df.groupby("method")[METRICS].mean()
    l, b = means.loc[learned], means.loc[baseline]
    reductions = {m: float(1.0 - l[m] / b[m]) for m in ("mmd2", "sliced_w1", "tv")}
    ess_ratio = float(l["ess"] / b["ess"])
    per_seed = []
    for seed, group in df.groupby("training_seed"):
        glm = group.set_index("method")
        per_seed.append(dict(
            seed=int(seed),
            ess_ratio=float(glm.loc[learned, "ess"] / glm.loc[baseline, "ess"]),
            **{f"{m}_reduction": float(1.0 - glm.loc[learned, m] / glm.loc[baseline, m])
               for m in ("mmd2", "sliced_w1", "tv")}))
    loo = []
    for seed in sorted(df["training_seed"].unique()):
        sub = df[df["training_seed"] != seed].groupby("method")[METRICS].mean()
        ll, bb = sub.loc[learned], sub.loc[baseline]
        loo.append(dict(omitted_seed=int(seed), ess_ratio=float(ll["ess"] / bb["ess"]),
                        reductions={m: float(1.0 - ll[m] / bb[m])
                                    for m in ("mmd2", "sliced_w1", "tv")}))
    return dict(
        ess_ratio=ess_ratio,
        ess_pass=bool(ess_ratio >= 2.0),
        target_reductions=reductions,
        target_pass=bool(sum(v >= 0.30 for v in reductions.values()) >= 2),
        per_seed=per_seed,
        leave_one_seed_out=loo,
        mean_weight_ratio=float(l["mean_weight_ratio"]),
        psis_k_mean=float(l["psis_k"]),
        raw_second_ratio_mean=float(l["raw_second_ratio"]),
        core_pass=bool(ess_ratio >= 2.0 and
                       sum(v >= 0.30 for v in reductions.values()) >= 2),
        note="Core statistical gates only; refinement, bias, SMC cost, and frozen stress are assessed separately.")


def load_refinement(w=3.0, seeds=(0, 1, 2)):
    frames = []
    for seed in seeds:
        frame = pd.read_csv(OUT / f"refinement_w{w:g}_seed{seed}.csv")
        frame["training_seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def refinement_assessment(df, M=1200):
    sub = df[df["M"] == M].copy()
    result = success_assessment(sub)
    result["M"] = M
    return result


def main():
    df = load_metrics()
    summary = summarize(df)
    summary.to_csv(OUT / "learned_w3_summary.csv", index=False)
    assessment = success_assessment(df)
    (OUT / "success_assessment.json").write_text(json.dumps(assessment, indent=2))
    cols = ["method", "ess", "mmd2", "sliced_w1", "tv", "mean_weight_ratio",
            "raw_second_ratio", "psis_k", "sampling_seconds"]
    df[cols + ["training_seed"]].to_csv(OUT / "learned_w3_per_seed.csv", index=False)
    print(summary[["method", "ess_mean", "mmd2_mean", "sliced_w1_mean",
                   "tv_mean", "mean_weight_ratio_mean", "psis_k_mean"]].to_string(index=False))
    print(json.dumps(assessment, indent=2))
    paths = [OUT / f"refinement_w3_seed{s}.csv" for s in (0, 1, 2)]
    if all(p.exists() for p in paths):
        ref = load_refinement()
        summarize(ref).to_csv(OUT / "refinement_summary_all_M.csv", index=False)
        for M in sorted(ref["M"].unique()):
            summarize(ref[ref["M"] == M]).to_csv(
                OUT / f"refinement_summary_M{M}.csv", index=False)
            ra = refinement_assessment(ref, int(M))
            (OUT / f"refinement_assessment_M{M}.json").write_text(
                json.dumps(ra, indent=2))
            print(f"REFINEMENT M={M}", json.dumps(ra, indent=2))


if __name__ == "__main__":
    main()
