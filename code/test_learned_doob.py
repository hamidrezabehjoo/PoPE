#!/usr/bin/env python3
"""Focused correctness tests for learned_doob.py (small, CPU-only)."""
import math

import numpy as np
import torch

from controlled_gmm import GridDoob2D
from gmm2d import GMM_BED, p0_score, pc_score
from learned_doob import (PositiveValueNet, fine_tune_control,
                          generate_naive_labels, psis_k, time_probabilities)


def test_boundary_and_time_law():
    model = PositiveValueNet(width=16, blocks=1)
    x = torch.randn(8, 2)
    t0 = torch.zeros(8)
    f, u = model.control(x, t0, create_graph=False)
    assert torch.all(f == 0) and torch.all(u == 0)
    times = np.linspace(1, 0, 31, dtype=np.float32)
    nz, p = time_probabilities(times)
    assert len(nz) == 30 and np.isclose(p.sum(), 1.0) and np.all(p > 0)
    # Beta(1/2,1) contributes more near t=0 than near t=1.
    assert p[-1] > p[0]


def test_suffix_labels():
    grid = GridDoob2D(GMM_BED, 3.0, n=64)
    labels = generate_naive_labels(grid, n_paths=128, M=20,
                                   n_snapshots=5, seed=123)
    assert labels["x"].shape == (128, 5, 2)
    assert np.all(labels["logw"] >= 0)
    assert np.all(labels["logw"][:, -1] == 0)
    assert np.allclose(labels["logw"][:, 0], labels["terminal_logw"])
    # Remaining-path weights decrease to one at the data end in this bed.
    assert np.all(np.median(labels["logw"], axis=0)[1:] >= -1e-6)


def test_exact_doob_weight_collapse():
    """Full weight formula, not the bypass used by grid.sample(rho=1)."""
    grid = GridDoob2D(GMM_BED, 3.0, n=96)
    N, M = 64, 400
    rng = np.random.default_rng(99)
    x = grid.initial_sample(N, rng)  # q_1
    logp0, logpc = grid.log_marginals(x, 1.0)
    logq1 = np.log(grid.interp(grid.conv(1.0), x)) - grid.logZ0
    logw = ((1 - grid.w) * logp0 + grid.w * logpc - grid.logZ1 - logq1)
    dt = 1.0 / M
    for i in range(M):
        t = 1.0 - i * dt
        _, score_q_grid = grid.conv_and_score(t)
        score_q = grid.interp(score_q_grid, x)
        s0, sc = p0_score(x, grid.bed, t), pc_score(x, grid.bed, t)
        score_m = (1 - grid.w) * s0 + grid.w * sc
        u = score_q - score_m
        V = -0.5 * grid.w * (grid.w - 1.0) * np.sum((sc - s0)**2, axis=1)
        xi = math.sqrt(dt) * rng.standard_normal(x.shape)
        logw += -V * dt - np.sum(u * xi, axis=1) - 0.5 * np.sum(u*u, axis=1) * dt
        x = x + (score_m + u) * dt + xi
    sd = float(np.std(logw))
    print(f"exact-control discrete log-weight sd={sd:.5g}")
    # Continuous-time weights are exactly constant; Euler/grid error is small.
    assert sd < 0.20


def test_policy_gradient_smoke():
    grid = GridDoob2D(GMM_BED, 3.0, n=64)
    model = PositiveValueNet(width=16, blocks=1)
    model.head.weight.data.normal_(0, 0.01)
    tuned, info = fine_tune_control(model, grid, seed=0, updates=1, batch=8,
                                    policy_steps=4, val_N=64, M=10,
                                    val_every=1)
    assert np.isfinite(info["history"][0]["val_C"])
    for p in tuned.parameters():
        assert torch.all(torch.isfinite(p))


def test_psis():
    lw = np.random.default_rng(0).normal(size=1000)
    assert np.isfinite(psis_k(lw))


def main():
    test_boundary_and_time_law()
    test_suffix_labels()
    test_exact_doob_weight_collapse()
    test_policy_gradient_smoke()
    test_psis()
    print("All learned-Doob tests passed.")


if __name__ == "__main__":
    main()
