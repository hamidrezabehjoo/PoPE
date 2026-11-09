"""Canonical regeneration drivers for the paper's numerical data.

Every driver writes to a caller-selected results directory. None modifies the
bundled paper data in data/.
"""
import csv
from pathlib import Path

import numpy as np

from .core import (GAUSS_BED, GMM_BED, Zratio_grid, gaussian_path_mgf, smc_cfg,
                   stable_seed, tv_to_target)
from .diagnostics import (ReferenceGrid, evaluate, smc_resampling_cfg, summarize,
                          timed_call, write_csv)
from .surrogate import controlled_cfg, tempered_grid
from .gaussian import (DIMS, ETA, MU, V0, VC, critical_w, exact_ess_1d,
                       lambda_star_1d, smc_cfg_1d, smc_example1, var_logw_1d,
                       vp_noise_end, wstar_bed)
from .gmm_moments import solve_logmoment


def _exact_cfg_1d(w, M=500):
    ew, _ = gaussian_path_mgf(V0, VC, MU, w, 1.0, M=M)
    return float(ew), exact_ess_1d(w, M=M), var_logw_1d(w, M=M)


def _exact_gaussian_ess_2d(w, M=500):
    e1x, _ = gaussian_path_mgf(2.25, 0.25, 0.8, w, 1.0, M=M)
    e1y, _ = gaussian_path_mgf(2.25, 0.25, 0.0, w, 1.0, M=M)
    e2x, okx = gaussian_path_mgf(2.25, 0.25, 0.8, w, 2.0, M=M)
    e2y, oky = gaussian_path_mgf(2.25, 0.25, 0.0, w, 2.0, M=M)
    ew2 = e2x * e2y if okx and oky else np.inf
    return (e1x * e1y) ** 2 / ew2 if np.isfinite(ew2) else 0.0


def run_gaussian(output_dir, quick=False):
    """Regenerate the Gaussian verification and terminal-IS degeneracy data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N, M = (4000, 300) if quick else (60_000, 2000)
    tgrid = np.linspace(0, 1, 41)
    print(f"Two-expert Gaussian verification: N={N}, M={M}", flush=True)
    result = smc_example1(N=N, M=M, seed=0, return_samples=True, ess_times=tgrid)
    mc_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    mc_z = np.array([result["EW"], result["snaps"][0.25],
                     result["snaps"][0.5], result["snaps"][0.75], 1.0])
    np.savez_compressed(output_dir / "gaussian_verification.npz",
                        x=result["x0"], wn=result["wn"], mc_t=mc_t, mc_z=mc_z,
                        N=N, M=M)
    traces = {"two": np.array([result["ess_snap"][float(t)] for t in tgrid])}

    summary = []
    for w in (1.5, 2.0, 3.0, 5.0):
        print(f"CFG Gaussian w={w:g}: N={N}, M={M}", flush=True)
        run = smc_cfg_1d(w, N=N, M=M, seed=0,
                         ess_times=tgrid if w <= 3 else None)
        if w <= 3:
            traces[f"w{w:g}"] = np.array([run["ess_snap"][float(t)] for t in tgrid])
        ew_exact, ess_exact, var_exact = _exact_cfg_1d(w)
        summary.append(dict(w=w, var_logw_exact=var_exact,
                            var_logw_mc=run["var_logw"], ess_exact=ess_exact,
                            ess_mc=run["ess"], ew_exact=ew_exact, ew_mc=run["ew"],
                            N=N, M=M))
    np.savez_compressed(output_dir / "gaussian_ess_trajectory.npz", t=tgrid,
                        ess_star_two=0.820, ess_star_w1_5=exact_ess_1d(1.5),
                        **traces)
    write_csv(output_dir / "gaussian_cfg_summary.csv", summary)
    write_csv(output_dir / "vp_noise_end.csv",
              [vp_noise_end(V0, VC, MU, 3.0, T) for T in (1.0, 2.0, 5.0, 10.0)])


def run_gaussian_thresholds(output_dir, quick=False):
    """Regenerate early-stopping, moment-index, and dimension thresholds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    early_rows = []
    grid_m = 200 if quick else 1200
    for vc in (0.25, 0.05, 0.01, 0.001):
        for t_min in (0.0, 0.001, 0.01, 0.05, 0.1):
            value = critical_w(V0, vc, MU, t_min=t_min, M=grid_m)
            early_rows.append(dict(vc=vc, t_min=t_min, M=grid_m, w_star=value))
            print(f"early stopping vc={vc:g}, t_min={t_min:g}: w*={value:.6f}",
                  flush=True)
    write_csv(output_dir / "early_stopping.csv", early_rows)

    high_rows = []
    high_m = 200 if quick else 400
    for w in (5.0, 10.0, 20.0, 50.0):
        lx = lambda_star_1d(w, M=high_m, tol=1e-4)
        ly = lambda_star_1d(w, mu=0.0, M=high_m, tol=1e-4)
        high_rows.append(dict(w=w, M=high_m, lambda_star_mean_axis=lx,
                              lambda_star_variance_axis=ly, lambda_star=min(lx, ly)))
        print(f"high guidance w={w:g}: lambda*={min(lx, ly):.6f}", flush=True)
    write_csv(output_dir / "high_guidance.csv", high_rows)

    index_rows = []
    index_m = 300 if quick else 1000
    for w in (1.25, 1.5, 2.0, 3.0, 5.0):
        lx = lambda_star_1d(w, M=index_m)
        ly = lambda_star_1d(w, mu=0.0, M=index_m)
        index_rows.append(dict(w=w, M=index_m, lambda_star_mean_axis=lx,
                               lambda_star_variance_axis=ly, lambda_star=min(lx, ly)))
    write_csv(output_dir / "gaussian_moment_indices.csv", index_rows)

    dimension_rows = []
    dimension_m = 120 if quick else 300
    for d in DIMS:
        mud = MU / np.sqrt(d)
        vcd = V0 * (1 + ETA / np.sqrt(d))
        wx = wstar_bed(V0, vcd, mud, M=dimension_m)
        wy = wstar_bed(V0, vcd, 0.0, M=dimension_m)
        value = min(wx, wy)
        dimension_rows.append(dict(d=int(d), w_star=value, M=dimension_m))
        print(f"mean-field dimension d={d}: w*={value:.4f}", flush=True)
    write_csv(output_dir / "dimension_thresholds.csv", dimension_rows)


def run_gmm(output_dir, quick=False):
    """Regenerate the Gaussian/GMM sweep and the three density-panel runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N, M = (3000, 200) if quick else (60_000, 2000)
    ws = (0.5, 1.5, 2.0) if quick else (0.5, 1.0, 1.5, 1.75, 2.0, 3.0)
    rows = []
    for w in ws:
        gaussian = smc_cfg(GAUSS_BED, w, N=N, M=M,
                           seed=stable_seed("sweep", "G", w), snapshots=())
        n_seeds = 1 if quick else {0.5: 4, 1.0: 1, 1.5: 8,
                                   1.75: 8, 2.0: 8, 3.0: 1}[w]
        replicate_runs = []
        if n_seeds > 1:
            for seed_index in range(n_seeds):
                print(f"GMM w={w:g}, seed={seed_index}", flush=True)
                replicate_runs.append(smc_cfg(
                    GMM_BED, w, N=N, M=M, seed=stable_seed("tab", w, seed_index),
                    snapshots=()))
        print(f"GMM representative density/TV run, w={w:g}", flush=True)
        representative = smc_cfg(GMM_BED, w, N=N, M=M,
                                 seed=stable_seed("sweep", "M", w), snapshots=())
        if not replicate_runs:
            replicate_runs = [representative]
        ess = np.array([run["ess"] for run in replicate_runs])
        ew = np.array([run["EW"] for run in replicate_runs])
        tv_w = tv_to_target(representative["x0"], representative["wn"], GMM_BED, w)
        uniform = np.full(N, 1.0 / N)
        tv_n = tv_to_target(representative["x0"], uniform, GMM_BED, w)
        if w in (0.5, 1.5, 2.0):
            np.savez_compressed(output_dir / f"gmm_density_w{w:g}.npz",
                                x=representative["x0"], wn=representative["wn"],
                                tv_n=tv_n, tv_w=tv_w, N=N, M=M)
        rows.append(dict(w=w, gaussian_ess_exact=_exact_gaussian_ess_2d(w),
                         gaussian_ess_mc=gaussian["ess"],
                         gmm_ess_mean=float(ess.mean()),
                         gmm_ess_sd=float(ess.std(ddof=1)) if len(ess) > 1 else 0.0,
                         gmm_ew_mean=float(ew.mean()),
                         gmm_ew_sd=float(ew.std(ddof=1)) if len(ew) > 1 else 0.0,
                         gmm_ew_exact=Zratio_grid(GMM_BED, w),
                         tv_naive=tv_n, tv_corrected=tv_w,
                         gmm_n_seeds=n_seeds, N=N, M=M))
        write_csv(output_dir / "gmm_summary.csv", rows)


def run_accuracy(output_dir, quick=False):
    """Matched-score-call target-accuracy comparison on the GMM bed at w=3.

    Every method uses common seed labels, N particles, and M reverse steps. The
    timer covers initialization and sampling only; reference-grid construction
    and distributional evaluation are excluded. Score calls are counted at the
    expert-score call sites in the samplers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N, M = (2000, 100) if quick else (20_000, 600)
    n_seeds, grid_n, init_grid_n = (2, 96, 101) if quick else (8, 320, 701)
    w = 3.0
    grid = ReferenceGrid(GMM_BED, w, n=grid_n)
    noise_grid = tempered_grid(GMM_BED, w, 1.0, n=init_grid_n)
    diag_n = min(2000, N)
    methods = ("terminal IS", "surrogate control", "adaptive SMC")
    rows = []
    for replicate in range(n_seeds):
        for method in methods:
            # The same integer seed is used for every method. Different proposal
            # and initialization mechanisms need not consume identical streams.
            seed = replicate

            def run():
                if method == "terminal IS":
                    return controlled_cfg(GMM_BED, w, "naive", N=N, M=M,
                                          seed=seed, noise_grid=noise_grid)
                if method == "surrogate control":
                    return controlled_cfg(GMM_BED, w, "ctrl_q1", N=N, M=M,
                                          seed=seed, noise_grid=noise_grid)
                return smc_resampling_cfg(GMM_BED, w, N=N, M=M, seed=seed)

            result, seconds = timed_call(run)
            wn = result["wn"]
            if (not np.all(np.isfinite(wn)) or np.any(wn < 0)
                    or not np.isclose(wn.sum(), 1.0, rtol=0.0, atol=1e-12)):
                raise RuntimeError("Invalid normalized sampler weights")
            if result["score_calls"] != 2 * N * M:
                raise RuntimeError("Unexpected expert-score-call count")
            metrics = evaluate(
                result, grid, diag_n=diag_n,
                diag_seed=stable_seed("matched-accuracy-diagnostic", replicate))
            row = dict(replicate=replicate, seed=seed, method=method, w=w,
                       N=N, M=M, grid=grid_n, init_grid=init_grid_n,
                       seconds=seconds, score_calls=result["score_calls"],
                       ess=result["ess"],
                       n_resample=result.get("n_resample", 0), **metrics)
            rows.append(row)
            print(f"accuracy w=3, {method}, seed={seed}: "
                  f"MMD2={metrics['mmd2']:.6g}, SW1={metrics['sliced_w1']:.6g}, "
                  f"TV={metrics['tv']:.6g}, ESS/N={result['ess']:.6g}, "
                  f"calls={result['score_calls']}, time={seconds:.3f}s",
                  flush=True)
            write_csv(output_dir / "matched_accuracy_replicates.csv", rows)

    metric_keys = ["mmd2", "sliced_w1", "tv", "score_calls", "seconds",
                   "ess", "n_resample"]
    summary = summarize(rows, ["method"], metric_keys)
    order = {method: i for i, method in enumerate(methods)}
    summary.sort(key=lambda row: order[row["method"]])
    write_csv(output_dir / "matched_accuracy.csv", summary)

    def estimate(key, row):
        return f"{row[key + '_mean']:.6g} +/- {row[key + '_sd']:.3g}"

    table_rows = []
    for row in summary:
        table_rows.append([
            row["method"], estimate("mmd2", row), estimate("sliced_w1", row),
            estimate("tv", row), f"{row['score_calls_mean']:.0f}",
            estimate("seconds", row), estimate("ess", row),
        ])
    headers = ["Method", "MMD^2", "sliced W1", "TV", "score calls",
               "time [s]", "ESS/N"]
    markdown = ["| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    markdown += ["| " + " | ".join(values) + " |" for values in table_rows]
    markdown += ["", "Entries are mean +/- sample standard deviation over "
                 f"{n_seeds} common seeds; N={N}, M={M}, w=3.", ""]
    (output_dir / "matched_accuracy_table.md").write_text("\n".join(markdown))

    latex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Method & MMD$^2$ & sliced $W_1$ & TV & score calls & time [s] & ESS/N \\\\",
             "\\midrule"]
    for values in table_rows:
        escaped = [values[0]] + [
            value.replace("+/-", "$\\pm$") if "+/-" in value else value
            for value in values[1:]
        ]
        latex.append(" & ".join(escaped) + " \\\\")
    latex += ["\\bottomrule", "\\end{tabular}", ""]
    (output_dir / "matched_accuracy_table.tex").write_text("\n".join(latex))


def run_gmm_moments(output_dir, quick=False):
    """Regenerate the bounded-domain log-moment scans in the GMM table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        plan = [(1.5, 2.8, (3.0, 4.0), 0.2)]
    else:
        plan = [(1.5, 2.8, (4.0, 6.0, 8.0, 10.0), 0.1),
                (1.5, 2.9, (4.0, 6.0, 8.0, 10.0), 0.1),
                (1.75, 2.0, (4.0, 6.0, 8.0, 10.0), 0.1),
                (1.82, 2.0, (4.0, 6.0, 8.0, 10.0), 0.1),
                (1.85, 2.0, (4.0, 6.0, 8.0), 0.1)]
    rows = []
    for w, lam, domains, dx in plan:
        for L in domains:
            result = solve_logmoment(GMM_BED, w, lam, L=L, dx=dx, dt_scale=0.25)
            row = {"w": w, "lambda": lam, "dx": dx, "L": L, "n": result["n"],
                   "dt_scale": 0.25, "logM": result["logM"], "hmax": result["hmax"],
                   "solver_status": result["status"]}
            rows.append(row)
            print(row, flush=True)
            write_csv(output_dir / "gmm_exact_moments.csv", rows)
