#!/usr/bin/env python3
"""Pretrained class-conditional MNIST CFG disagreement and PSIS pilot.

This script uses the pretrained MNIST checkpoint from
https://github.com/byrkbrk/diffusion-model (tested at commit
8437d9e63cc07e4e518b6769bf85d3d9e0886aca).  Clone that repository separately
and pass its path via --repo; no third-party code or checkpoint is redistributed
here.  The model supplies epsilon_c and epsilon_0 with the same network.  The
score difference is -(epsilon_c-epsilon_0)/sqrt(1-ab_t).

We subsample the original 500-step VP schedule with ancestral posterior jumps.
The accumulated action and terminal log weight are the continuous-time
left-endpoint approximations

  D = integral ||s_c-s_0||^2 dt,
  log W = w(w-1)D/2.

This is a small diagnostic pilot, not a pretrained-model quality benchmark or
an exact sampler validation.  PSIS k-hat is computed by ArviZ 0.22.0.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from gmm2d import stable_seed

OUT = Path(__file__).resolve().parents[1] / "output"


def psis_k(log_weights):
    import arviz as az
    _, k = az.psislw(np.asarray(log_weights), reff=1.0, normalize=True)
    return float(k)


def time_grid(T, n_steps):
    """Descending training-time indices, including both T and zero."""
    return np.linspace(T, 0, n_steps + 1).round().astype(int)


def load_model(repo, device):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    from models import ContextUnet
    checkpoint_path = repo / "checkpoints" / "pretrained_mnist_checkpoint_49.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ContextUnet(in_channels=1, height=28, width=28,
                        n_feat=64, n_cfeat=10, n_downs=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    return model, checkpoint


def ancestral_jump(x, eps, t, s, ab, z):
    """Posterior-variance ancestral jump from training step t to s<t."""
    ab_t, ab_s = ab[t], ab[s]
    x0 = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
    sigma2 = (1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)
    direction_var = torch.clamp(1 - ab_s - sigma2, min=0.0)
    return (torch.sqrt(ab_s) * x0 + torch.sqrt(direction_var) * eps
            + torch.sqrt(sigma2) * z)


@torch.no_grad()
def run(repo, ws=(1.5, 2.0, 3.0, 5.0), N=256, n_steps=12,
        device="cpu", seed=0):
    model, checkpoint = load_model(repo, device)
    T = int(checkpoint["timesteps"])
    beta = torch.linspace(checkpoint["beta1"], checkpoint["beta2"], T + 1,
                          device=device)
    ab = torch.cumprod(1 - beta, dim=0)
    ab[0] = 1.0
    torch.manual_seed(seed)
    ws = np.asarray(ws, dtype=float)
    nw = len(ws)
    base_context = torch.nn.functional.one_hot(torch.arange(N) % 10, 10).float()
    context = base_context.repeat(nw, 1).to(device)
    zero_context = torch.zeros_like(context)
    w_tensor = torch.tensor(np.repeat(ws, N), dtype=torch.float32,
                            device=device)[:, None, None, None]
    x0 = torch.randn(N, 1, 28, 28, device=device)
    x = x0.repeat(nw, 1, 1, 1)
    logW = torch.zeros(nw * N, device=device)
    disagreement = torch.zeros_like(logW)
    times = time_grid(T, n_steps)
    for t, s in zip(times[:-1], times[1:]):
        time_tensor = torch.full((1, 1, 1, 1), t / T, device=device)
        eps_c = model(x, time_tensor, context)
        eps_0 = model(x, time_tensor, zero_context)
        eps_cfg = eps_0 + w_tensor * (eps_c - eps_0)
        score_diff2 = ((eps_c - eps_0).flatten(1)**2).sum(1) / (1 - ab[t])
        dt = float(t - s) / T
        disagreement += score_diff2 * dt
        logW += 0.5 * w_tensor.flatten() * (w_tensor.flatten() - 1) * score_diff2 * dt
        z0 = torch.randn(N, 1, 28, 28, device=device) if s > 0 else torch.zeros_like(x0)
        z = z0.repeat(nw, 1, 1, 1)
        x = ancestral_jump(x, eps_cfg, int(t), int(s), ab, z)
        print(f"step {t:3d}->{s:3d}", flush=True)
    return ws, disagreement.reshape(nw, N).cpu().numpy(), logW.reshape(nw, N).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="path to a clone of byrkbrk/diffusion-model")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--w", type=float, nargs="+", default=[1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ws, D, logW = run(args.repo, args.w, N=args.N, n_steps=args.steps,
                      device=args.device,
                      seed=stable_seed("pretrained-mnist-cfg", args.N, args.steps))
    rows = []
    for w, d, lw in zip(ws, D, logW):
        normalized = np.exp(lw - np.max(lw))
        normalized /= normalized.sum()
        ess = float(1.0 / np.sum(normalized**2) / len(lw))
        k = psis_k(lw)
        row = dict(w=w, N=len(lw), steps=args.steps,
                   disagreement_mean=float(d.mean()),
                   disagreement_sd=float(d.std()),
                   disagreement_cv=float(d.std() / d.mean()),
                   disagreement_per_pixel=float(d.mean() / 784),
                   log_weight_sd=float(lw.std()), ess=ess, psis_k=k)
        rows.append(row)
        print(row)
    stem = f"pretrained_cfg{args.suffix}"
    with open(OUT / f"{stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / f"{stem}.npz", w=ws, disagreement=D, logW=logW)


if __name__ == "__main__":
    main()
