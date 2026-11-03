#!/usr/bin/env python3
"""
Fast smoke tests (< 2 min). Run:  python test_smoke.py
"""
import numpy as np

from gmm2d import GAUSS_BED, GMM_BED, gaussian_path_mgf, smc_cfg
from example1_gaussian import naive_law_and_certificate, zhat, mixture_mom
from renyi_debruijn import check
from lambda_star import lambda_star
from controlled_gaussian import approx_control_cfg, doob_sample
from controlled_gmm import GridDoob2D
from theory_control_checks import (smc_resampling_cfg,
                                   vp_doob_moment_propagation)
from pretrained_cfg import time_grid

FAIL = []


def expect(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL.append(name)


# --- Example 1 closed forms ---
m, P, I1, I2 = naive_law_and_certificate(n_steps=60_000)
expect("naive law mean 0.8321", abs(m - 0.8321) < 2e-3)
expect("naive law var 0.5225", abs(P - 0.5225) < 2e-3)
expect("Z0/Z1 = 0.52473", abs(zhat(0.0) / zhat(1.0) - 0.52473) < 1e-4)
expect("int E_naive[V] = 0.7969", abs(I1 - 0.7969) < 3e-3)
h0, P0 = mixture_mom(0.0)
kl = 0.5 * (P / P0 - 1 + (h0 - m) ** 2 / P0 + np.log(P0 / P))
expect("KL marginal = 0.1155", abs(kl - 0.1155) < 3e-3)
expect("KL bound holds", kl <= I1 - np.log(zhat(1.0) / zhat(0.0)) + 1e-9)
expect("normalizer identity", abs(I2 - np.log(zhat(1.0) / zhat(0.0))) < 3e-3)

# --- smc_cfg at w=1: weights exactly 1 ---
r = smc_cfg(GMM_BED, 1.0, N=4_000, M=200, seed=0)
expect("w=1: ESS/N = 1", abs(r["ess"] - 1.0) < 1e-12)
expect("w=1: E W = 1", abs(r["EW"] - 1.0) < 1e-12)

# --- Cameron-Martin feasibility signs ---
_, ok1 = gaussian_path_mgf(2.25, 0.25, 0.8, 1.5, 1.0, M=300)
_, ok2 = gaussian_path_mgf(2.25, 0.25, 0.8, 5.0, 2.0, M=300)
expect("E W finite at w=1.5", ok1)
expect("E W^2 infinite at w=5", not ok2)

# --- lambda* bracket at w=2 (coarse grid, loose bounds) ---
ls = lambda_star(2.0, M=300, tol=5e-3)
expect("lambda*(2) in [1.5, 2.5]", 1.5 <= ls <= 2.5)

# --- Renyi--de Bruijn (finite-difference level) ---
expect("Renyi--de Bruijn VE", check("VE", betas=(0.5,), ts=(0.5,)) < 1e-6)

# --- exact Gaussian Doob moments and uniform weights ---
d = doob_sample([1.5, -1.0], [0.25, 1.44], [0.5, 0.5],
                N=6_000, M=200, seed=11)
expect("Gaussian Doob mean", abs(d["mean"] - d["exact_mean"]) < 0.02)
expect("Gaussian Doob variance", abs(d["variance"] - d["exact_variance"]) < 0.02)
expect("Gaussian Doob ESS/N", d["ess"] == 1.0)

# --- exact-control path weights become constant under time refinement ---
a = approx_control_cfg(2.25, 0.25, 0.8, 3.0, rho=1.0, N=1200, M=80, seed=4)
b = approx_control_cfg(2.25, 0.25, 0.8, 3.0, rho=1.0, N=1200, M=320, seed=4)
expect("exact-control weight refinement", b["log_weight_sd"] < 0.6 * a["log_weight_sd"])

# --- early-stopping MGF preserves the original full-horizon call ---
old = gaussian_path_mgf(2.25, 0.25, 0.8, 1.5, 1.0, M=300)
new = gaussian_path_mgf(2.25, 0.25, 0.8, 1.5, 1.0, M=300, t_min=0.0)
expect("early-stopping MGF backward compatibility", old == new)

# --- gridded Gaussian Doob sampler reaches the exact tempered target ---
grid = GridDoob2D(GAUSS_BED, 2.0, n=64)
g = grid.sample(N=2500, M=120, seed=8)
expect("grid Doob Gaussian mean", abs(g["x"][:, 0].mean() - 0.8470588) < 0.04)
expect("grid Doob Gaussian weights", g["ess"] == 1.0)

# --- pretrained pilot must descend rather than sort its time grid ---
expect("pretrained descending time grid",
       np.array_equal(time_grid(500, 12)[[0, 1, -2, -1]], [500, 458, 42, 0]))

# --- adaptive SMC reduces to uniform weights at w=1 ---
s = smc_resampling_cfg(GMM_BED, 1.0, N=2000, M=100, seed=3)
expect("adaptive SMC w=1 weights", abs(s["ess"] - 1.0) < 1e-12)
expect("adaptive SMC w=1 no path resampling", s["n_resample"] == 0)

# --- VP moment recursion isolates standard-prior data-end error ---
vp_exact = vp_doob_moment_propagation(2.25, 0.25, 0.8, 3.0, 1.0,
                                      M=400, standard_prior=False)
vp_prior = vp_doob_moment_propagation(2.25, 0.25, 0.8, 3.0, 1.0,
                                      M=400, standard_prior=True)
expect("VP exact-qT initialization reaches target",
       vp_exact["data_end_gaussian_kl"] < 1e-5)
expect("VP standard-prior initialization has nonzero error",
       vp_prior["data_end_gaussian_kl"] > 1e-3)

print("\n%d failure(s)" % len(FAIL) if FAIL else "\nall tests passed")
raise SystemExit(1 if FAIL else 0)
