#!/usr/bin/env python3
"""Learned approximate Doob control on the exact two-dimensional CFG bed.

Stage 1 fits a positive value function from naive-path *suffix* partial
weights.  Stage 2 optionally fine-tunes the control with a randomized,
leave-one-out-baselined policy-gradient estimator of the direct control
objective.  Frozen evaluation always uses the full approximate-Doob IS weight.

No exact Doob/grid control is used for training.  The grid is used only to
sample the exact tempered noise-end marginal and for target diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from gmm2d import (GMM_BED, Zratio_grid, p0_pdf, pc_pdf, p0_score, pc_score,
                   smc_cfg, stable_seed, tempered_on)
from controlled_gmm import GridDoob2D, evaluate, write_csv
from theory_control_checks import smc_resampling_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "learned_doob"
DEVICE = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.fc2(self.act(self.fc1(self.act(x))))
        return x + y


class PositiveValueNet(nn.Module):
    """f_theta(t,x)=t*softplus(a_theta(t,x)); u=grad_x f."""

    def __init__(self, width=128, blocks=4):
        super().__init__()
        self.inp = nn.Linear(3, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("x_mean", torch.zeros(2))
        self.register_buffer("x_std", torch.ones(2))
        self.register_buffer("t_mean", torch.tensor(0.5))
        self.register_buffer("t_std", torch.tensor(0.3))

    def set_normalization(self, x_mean, x_std, t_mean, t_std):
        self.x_mean.copy_(torch.as_tensor(x_mean, dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(x_std, dtype=torch.float32))
        self.t_mean.copy_(torch.as_tensor(float(t_mean), dtype=torch.float32))
        self.t_std.copy_(torch.as_tensor(float(t_std), dtype=torch.float32))

    def raw_output(self, x, t):
        xn = (x - self.x_mean) / self.x_std
        tn = (t - self.t_mean) / self.t_std
        z = torch.cat([xn, tn[:, None]], dim=1)
        z = self.inp(z)
        for block in self.blocks:
            z = block(z)
        return self.head(z).squeeze(1)

    def forward(self, x, t):
        return t * nn.functional.softplus(self.raw_output(x, t))

    def control(self, x, t, create_graph=False):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            f = self(x, t)
            u = torch.autograd.grad(f.sum(), x, create_graph=create_graph)[0]
        return f, u


def model_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PositiveValueNet(ckpt.get("width", 128), ckpt.get("blocks", 4))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def save_model(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model_dict(model), **metadata), path)


def sample_tempered(grid, n, rng, t=1.0):
    density = tempered_on(grid.xy, grid.bed, grid.w, t).reshape(grid.n, grid.n)
    return grid.grid_sample(density, n, rng)


def generate_naive_labels(grid, n_paths=60_000, M=600, n_snapshots=31,
                          seed=0):
    """Generate naive paths and remaining-path (suffix) partial-weight labels.

    At stored time t, log W_t is the potential integral from the current state
    down to the data end.  It equals total log weight minus the prefix log
    weight accumulated from t=1 to the stored state.  In particular W_0=1.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    snap_steps = np.unique(np.linspace(0, M, n_snapshots).round().astype(int))
    times = 1.0 - snap_steps / M
    ns = len(snap_steps)
    xs = np.empty((n_paths, ns, 2), dtype=np.float32)
    prefixes = np.empty((n_paths, ns), dtype=np.float32)
    x = sample_tempered(grid, n_paths, rng, 1.0)
    logw = np.zeros(n_paths)
    th = 0.5 * grid.w * (grid.w - 1.0)
    dt = 1.0 / M
    snap_lookup = {int(s): j for j, s in enumerate(snap_steps)}
    for i in range(M + 1):
        if i in snap_lookup:
            j = snap_lookup[i]
            xs[:, j] = x
            prefixes[:, j] = logw
        if i == M:
            break
        t = 1.0 - i * dt
        s0, sc = p0_score(x, grid.bed, t), pc_score(x, grid.bed, t)
        d = sc - s0
        logw += th * np.einsum("nd,nd->n", d, d) * dt
        score_m = (1.0 - grid.w) * s0 + grid.w * sc
        x = x + score_m * dt + math.sqrt(dt) * rng.standard_normal(x.shape)
    suffix = logw[:, None] - prefixes
    suffix[:, -1] = 0.0
    assert np.allclose(times[-1], 0.0) and np.all(suffix[:, -1] == 0.0)
    result = dict(x=xs, logw=suffix.astype(np.float32), times=times.astype(np.float32),
                  terminal_logw=logw.astype(np.float32), M=M, seed=seed,
                  seconds=time.perf_counter() - start)
    print(f"labels w={grid.w} seed={seed}: {n_paths} paths, M={M}, "
          f"{result['seconds']:.1f}s, terminal logW sd={np.std(logw):.3f}", flush=True)
    return result


def psis_k(logw):
    """PSIS Pareto-k diagnostic using the current ArviZ stats accessor."""
    import xarray as xr
    import arviz_stats  # noqa: F401 (registers the accessor)
    da = xr.DataArray(np.asarray(logw, dtype=np.float64), dims=("sample",))
    _, k = da.azstats.psislw(dim="sample")
    return float(k.item())


def split_labels(labels, seed):
    n = len(labels["x"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nval = max(1, int(round(0.1 * n)))
    return np.sort(perm[nval:]), np.sort(perm[:nval])


def time_probabilities(times):
    """50% uniform nonzero snapshots + 50% discretized Beta(1/2,1)."""
    nz = np.flatnonzero(times > 0)
    ts = times[nz].astype(float)
    # Symmetric bin boundaries around each descending grid point.
    boundaries = np.empty(len(ts) + 1)
    boundaries[1:-1] = 0.5 * (ts[:-1] + ts[1:])
    boundaries[0], boundaries[-1] = 1.0, 0.0
    from scipy.stats import beta
    cdf = beta.cdf(boundaries, 0.5, 1.0)
    prob = np.abs(np.diff(cdf))
    prob = np.maximum(prob, 0.0)
    prob /= prob.sum()
    prob = 0.5 * np.full(len(ts), 1.0 / len(ts)) + 0.5 * prob
    return nz, prob


def per_time_scales(labels, train_idx):
    y = labels["logw"][train_idx]
    # Robust scales for optimization only; f itself remains on the log-W scale.
    med = np.median(y, axis=0)
    q75, q25 = np.percentile(y, [75, 25], axis=0)
    log_scale = np.maximum((q75 - q25) / 1.349, 0.15)
    value_scale = np.maximum(np.exp(med), 1.0)
    value_scale[-1] = 1.0
    return log_scale.astype(np.float32), value_scale.astype(np.float32)


def snapshot_batch(labels, trajectories, time_idx, batch, rng):
    nz, prob = time_probabilities(labels["times"])
    paths = rng.choice(trajectories, size=batch, replace=True)
    js = nz[rng.choice(len(nz), size=batch, p=prob)]
    x = labels["x"][paths, js]
    logw = labels["logw"][paths, js]
    t = labels["times"][js]
    return (torch.from_numpy(x), torch.from_numpy(t),
            torch.from_numpy(logw), torch.from_numpy(js))


def value_loss(model, labels, train_idx, log_scale, value_scale, batch,
               rng, family="logmse", grad_penalty=0.0):
    x, t, logw, js = snapshot_batch(labels, train_idx, None, batch, rng)
    create = bool(grad_penalty)
    if create:
        f, u = model.control(x, t, create_graph=True)
    else:
        f = model(x, t)
        u = None
    ls = torch.from_numpy(log_scale)[js]
    vs = torch.from_numpy(value_scale)[js]
    if family == "logmse":
        loss = torch.mean(((f - logw) / ls) ** 2)
    elif family == "loghuber":
        loss = torch.mean(nn.functional.huber_loss(f / ls, logw / ls, delta=1.0))
    elif family == "huber":
        pred = torch.exp(torch.clamp(f, max=25.0)) / vs
        target = torch.exp(torch.clamp(logw, max=25.0)) / vs
        loss = torch.mean(nn.functional.huber_loss(pred, target, delta=3.0))
    elif family == "mse":
        pred = torch.exp(torch.clamp(f, max=25.0)) / vs
        target = torch.exp(torch.clamp(logw, max=25.0)) / vs
        loss = torch.mean((pred - target) ** 2)
    else:
        raise ValueError(family)
    if grad_penalty:
        loss = loss + grad_penalty * torch.mean(u ** 2)
    return loss


def validation_loss(model, labels, val_idx, log_scale, value_scale, family,
                    grad_penalty, n=4096, seed=0):
    model.eval()
    with torch.no_grad():
        # No gradient penalty is needed in the scalar validation loss.
        loss = value_loss(model, labels, val_idx, log_scale, value_scale, n,
                          np.random.default_rng(seed), family, 0.0)
    model.train()
    return float(loss)


def train_value(labels, seed=0, family="logmse", max_steps=12_000,
                batch=512, lr=3e-4, grad_penalty=0.0, val_every=250,
                patience=6, min_steps=2000, out_path=None):
    start = time.perf_counter()
    torch.manual_seed(seed)
    train_idx, val_idx = split_labels(labels, seed + 101)
    log_scale, value_scale = per_time_scales(labels, train_idx)
    model = PositiveValueNet()
    xtrain = labels["x"][train_idx].reshape(-1, 2)
    model.set_normalization(xtrain.mean(0), np.maximum(xtrain.std(0), 1e-3),
                            labels["times"].mean(), labels["times"].std())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 202)
    best, best_val, stale, history = None, np.inf, 0, []
    for step in range(1, max_steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / max_steps))
        loss = value_loss(model, labels, train_idx, log_scale, value_scale,
                          batch, rng, family, grad_penalty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % val_every == 0 or step == max_steps:
            vl = validation_loss(model, labels, val_idx, log_scale, value_scale,
                                 family, grad_penalty, seed=seed + 303)
            history.append(dict(step=step, train_loss=float(loss.detach()),
                                val_loss=vl, lr=opt.param_groups[0]["lr"]))
            print(f"stage1 seed={seed} {family} step={step} val={vl:.5g}", flush=True)
            if vl < best_val - 1e-5:
                best_val, best, stale = vl, model_dict(model), 0
            else:
                stale += 1
            if step >= min_steps and stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    seconds = time.perf_counter() - start
    metadata = dict(seed=seed, family=family, w=None, steps=step,
                    best_val_loss=best_val, train_seconds=seconds,
                    grad_penalty=grad_penalty, M=int(labels["M"]),
                    normalization="training trajectories only",
                    label_definition="remaining-path suffix partial weight")
    if out_path:
        save_model(out_path, model, **metadata)
    return model, dict(metadata=metadata, history=history,
                       terminal_psis_k=psis_k(labels["terminal_logw"][train_idx]),
                       peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def control_np(model, x, t):
    xt = torch.as_tensor(x, dtype=torch.float32)
    tt = torch.full((len(x),), float(t), dtype=torch.float32)
    _, u = model.control(xt, tt, create_graph=False)
    return u.detach().numpy().astype(np.float64)


def rollout_objective(model, grid, N=2048, M=600, seed=0, standard_prior=False):
    """Independent rollout estimate of the u-dependent part of C."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((N, 2)) if standard_prior
         else sample_tempered(grid, N, rng, 1.0))
    dt = 1.0 / M
    total = np.zeros(N)
    for i in range(M):
        t = 1.0 - i * dt
        u = control_np(model, x, t)
        s0, sc = p0_score(x, grid.bed, t), pc_score(x, grid.bed, t)
        d = sc - s0
        V = -0.5 * grid.w * (grid.w - 1.0) * np.sum(d * d, axis=1)
        total += (0.5 * np.sum(u * u, axis=1) + V) * dt
        score_m = (1.0 - grid.w) * s0 + grid.w * sc
        x = x + (score_m + u) * dt + math.sqrt(dt) * rng.standard_normal(x.shape)
    return float(total.mean()), float(total.std(ddof=1) / math.sqrt(N))


def fine_tune_control(model, grid, seed=0, updates=120, batch=64,
                      policy_steps=32, lr=2e-5, val_every=10, val_N=2048,
                      patience=4, M=600, out_path=None):
    """Direct-C policy gradient with time subsampling and LOO baselines.

    For a reverse Euler transition, d log p_theta = xi dot d u_theta.
    At selected step j the estimator uses the detached remaining cost G_j:
      grad [ .5||u_j||^2 dt + (G_j-b_j) xi_j dot u_j ].
    Time indices are uniform without replacement and rescaled by M/K.  The
    baseline at (b,j) is the mean of G_j over the *other* trajectories, hence
    independent of trajectory b.  This avoids backpropagation through time.
    """
    start = time.perf_counter()
    torch.manual_seed(seed + 404)
    model = copy.deepcopy(model).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 505)
    dt = 1.0 / M
    best, best_val, stale, history = None, np.inf, 0, []
    for update in range(1, updates + 1):
        x = sample_tempered(grid, batch, rng, 1.0)
        xs = np.empty((batch, M, 2), dtype=np.float32)
        xis = np.empty((batch, M, 2), dtype=np.float32)
        costs = np.empty((batch, M), dtype=np.float64)
        for j in range(M):
            t = 1.0 - j * dt
            u = control_np(model, x, t)
            s0, sc = p0_score(x, grid.bed, t), pc_score(x, grid.bed, t)
            d = sc - s0
            V = -0.5 * grid.w * (grid.w - 1.0) * np.sum(d * d, axis=1)
            costs[:, j] = (0.5 * np.sum(u * u, axis=1) + V) * dt
            xi = math.sqrt(dt) * rng.standard_normal(x.shape)
            xs[:, j], xis[:, j] = x, xi
            score_m = (1.0 - grid.w) * s0 + grid.w * sc
            x = x + (score_m + u) * dt + xi
        returns = np.flip(np.cumsum(np.flip(costs, axis=1), axis=1), axis=1)
        js = np.sort(rng.choice(M, size=min(policy_steps, M), replace=False))
        xb = torch.from_numpy(xs[:, js].reshape(-1, 2))
        tb = torch.from_numpy(np.repeat((1.0 - js / M)[None, :], batch, axis=0).reshape(-1).astype(np.float32))
        xib = torch.from_numpy(xis[:, js].reshape(-1, 2))
        G = returns[:, js]
        baseline = (G.sum(axis=0, keepdims=True) - G) / max(batch - 1, 1)
        adv = torch.from_numpy((G - baseline).reshape(-1).astype(np.float32))
        _, u = model.control(xb, tb, create_graph=True)
        direct = 0.5 * torch.sum(u * u, dim=1) * dt
        score = adv * torch.sum(u * xib, dim=1)
        loss = (M / len(js)) * torch.mean(direct + score)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        opt.step()
        if update == 1 or update % val_every == 0 or update == updates:
            vc, vse = rollout_objective(
                model, grid, N=val_N, M=M,
                seed=stable_seed("finetune-validation", grid.w, seed, update))
            history.append(dict(update=update, train_objective=float(costs.sum(1).mean()),
                                val_C=vc, val_C_se=vse, grad_norm=grad_norm))
            print(f"stage2 seed={seed} update={update} C={vc:.5g} +/- {vse:.3g}", flush=True)
            if vc < best_val:
                best_val, best, stale = vc, model_dict(model), 0
            else:
                stale += 1
            if stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    seconds = time.perf_counter() - start
    metadata = dict(finetune_seed=seed, finetune_updates=update,
                    finetune_best_C=best_val, finetune_seconds=seconds,
                    finetune_estimator="randomized policy gradient; LOO baseline",
                    finetune_policy_steps=policy_steps, finetune_batch=batch)
    if out_path:
        save_model(out_path, model, **metadata)
    return model, dict(metadata=metadata, history=history,
                       peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def learned_sample(model, grid, N=20_000, M=600, seed=0,
                   initialization="tempered"):
    """Frozen learned-control rollout with full approximate-Doob IS weights."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    if initialization == "tempered":
        x = sample_tempered(grid, N, rng, 1.0)
        logw = np.zeros(N)
    elif initialization == "standard":
        x = rng.standard_normal((N, 2))
        logp0 = np.log(p0_pdf(x, grid.bed, 1.0) + 1e-300)
        logpc = np.log(pc_pdf(x, grid.bed, 1.0) + 1e-300)
        logr = -0.5 * np.sum(x * x, axis=1) - math.log(2.0 * math.pi)
        logw = ((1.0 - grid.w) * logp0 + grid.w * logpc
                - grid.logZ1 - logr)
    else:
        raise ValueError(initialization)
    dt = 1.0 / M
    th = 0.5 * grid.w * (grid.w - 1.0)
    for i in range(M):
        t = 1.0 - i * dt
        u = control_np(model, x, t)
        s0, sc = p0_score(x, grid.bed, t), pc_score(x, grid.bed, t)
        d = sc - s0
        xi = math.sqrt(dt) * rng.standard_normal(x.shape)
        logw += (th * np.sum(d * d, axis=1) * dt
                 - np.sum(u * xi, axis=1)
                 - 0.5 * np.sum(u * u, axis=1) * dt)
        score_m = (1.0 - grid.w) * s0 + grid.w * sc
        x = x + (score_m + u) * dt + xi
    return dict(x=x, logw=logw, seconds=time.perf_counter() - start,
                model_evaluations=N * M)


def terminal_is_sample(grid, N, M, seed, initialization="tempered"):
    if initialization == "standard":
        from theory_control_checks import terminal_standard_prior
        r = terminal_standard_prior(grid, N=N, M=M, seed=seed, corrected=True)
        return dict(x=r["x"], logw=r["logW"],
                    seconds=np.nan, model_evaluations=0)
    start = time.perf_counter()
    r = smc_cfg(grid.bed, grid.w, N=N, M=M, seed=seed, snapshots=())
    # smc_cfg returns raw log weights after its exact-tempered SIR initialization.
    return dict(x=r["x0"], logw=r["logW"], seconds=time.perf_counter() - start,
                model_evaluations=0)


def smc_sample(grid, N, M, seed):
    start = time.perf_counter()
    r = smc_resampling_cfg(grid.bed, grid.w, N=N, M=M, seed=seed)
    return dict(x=r["x"], wn=r["wn"], seconds=time.perf_counter() - start,
                n_resample=r["n_resample"], model_evaluations=0)


def weight_scaling(logw, zratio):
    rows = []
    for n in (2000, 5000, 10_000, 20_000):
        if n > len(logw):
            continue
        lw = logw[:n]
        mean = float(np.exp(logsumexp(lw) - math.log(n)))
        second = float(np.exp(logsumexp(2 * lw) - math.log(n)))
        wn = np.exp(lw - logsumexp(lw))
        rows.append(dict(N=n, ess=float(1.0 / np.sum(wn * wn) / n),
                         mean_raw=mean, mean_ratio=mean / zratio,
                         second_raw=second, second_ratio=second / zratio**2))
    return rows


def diagnostic_row(method, result, grid, seed, w, M, stage="",
                   initialization="tempered", raw_weights=True):
    if "logw" in result and raw_weights:
        logw = result["logw"]
        wn = np.exp(logw - logsumexp(logw))
        mean_raw = float(np.exp(logsumexp(logw) - math.log(len(logw))))
        second_raw = float(np.exp(logsumexp(2 * logw) - math.log(len(logw))))
        k = psis_k(logw)
    else:
        wn = result["wn"]
        logw = np.log(wn + 1e-300)
        mean_raw, second_raw, k = np.nan, np.nan, np.nan
    r = dict(x=result["x"], wn=wn)
    metrics = evaluate(r, grid, diag_n=min(2000, len(wn)),
                       diag_seed=stable_seed("learned-diagnostic", w, seed))
    zratio = math.exp(grid.logZ0 - grid.logZ1)
    row = dict(method=method, stage=stage, w=w, seed=seed, M=M, N=len(wn),
               initialization=initialization,
               ess=float(1.0 / np.sum(wn * wn) / len(wn)),
               raw_second_ratio=second_raw / zratio**2,
               mean_weight_ratio=mean_raw / zratio,
               max_weight_share=float(np.max(wn)), psis_k=k,
               log_weight_sd=float(np.std(logw)),
               sampling_seconds=result.get("seconds", np.nan),
               model_evaluations=result.get("model_evaluations", 0),
               n_resample=result.get("n_resample", 0), **metrics)
    if "logw" in result and raw_weights:
        row["weight_scaling_json"] = json.dumps(weight_scaling(logw, zratio))
    else:
        row["weight_scaling_json"] = ""
    return row


def grid_oracle_diagnostic(model, grid, n=41,
                           times=(1.0, 0.75, 0.5, 0.25, 0.1)):
    """Exact-grid value/control RMSE, strictly as a post-training diagnostic."""
    # Restrict to the central diagnostic domain; remote FFT-grid tail scores
    # are numerically meaningless and can overflow an RMSE even when unused.
    axis = np.linspace(-2.0, 2.0, n)
    xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    xt = torch.from_numpy(xy.astype(np.float32))
    rows = []
    for t in times:
        conv, score_q_grid = grid.conv_and_score(t)
        conv_x = grid.interp(conv, xy)
        score_q = grid.interp(score_q_grid, xy)
        s0, sc = p0_score(xy, grid.bed, t), pc_score(xy, grid.bed, t)
        score_m = (1.0 - grid.w) * s0 + grid.w * sc
        m = tempered_on(xy, grid.bed, grid.w, t)
        logv = np.log(np.maximum(conv_x, 1e-300)) - np.log(np.maximum(m, 1e-300))
        u_star = score_q - score_m
        tt = torch.full((len(xy),), float(t), dtype=torch.float32)
        f, u = model.control(xt, tt, create_graph=False)
        keep = (m > 1e-8) & (conv_x > 1e-8)
        ferr = (f.detach().numpy() - logv)[keep].astype(np.float64)
        uerr = (u.detach().numpy() - u_star)[keep].astype(np.float64)
        uref = u_star[keep].astype(np.float64)
        rows.append(dict(t=t, n_grid_points=int(keep.sum()),
                         value_log_rmse=float(np.sqrt(np.mean(ferr**2))),
                         value_log_mae=float(np.mean(np.abs(ferr))),
                         control_rmse=float(np.sqrt(np.mean(uerr**2))),
                         control_relative_rmse=float(np.sqrt(np.mean(uerr**2)) /
                                                     (np.sqrt(np.mean(uref**2)) + 1e-12))))
    return rows


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def run_seed(w=3.0, seed=0, families=("logmse", "huber"), paths=60_000,
             train_steps=12_000, eval_N=20_000, M=600, grid_n=320,
             finetune_updates=120, selection_N=2048, skip_finetune=False,
             grad_penalty=0.0):
    """Train both Stage-1 families on shared labels, select by rollout C,
    fine-tune the winner, and evaluate against matched terminal IS and SMC."""
    OUT.mkdir(parents=True, exist_ok=True)
    grid = GridDoob2D(GMM_BED, w, n=grid_n)
    labels = generate_naive_labels(grid, paths, M=M, seed=stable_seed("labels", w, seed))
    candidates = []
    for family in families:
        path = OUT / f"value_w{w:g}_seed{seed}_{family}.pt"
        model, info = train_value(
            labels, seed=seed, family=family, max_steps=train_steps,
            grad_penalty=grad_penalty, out_path=path)
        info["metadata"]["w"] = w
        save_model(path, model, **info["metadata"])
        C, se = rollout_objective(
            model, grid, N=selection_N, M=M,
            seed=stable_seed("stage1-selection", w, seed))
        info["selection_C"], info["selection_C_se"] = C, se
        info["checkpoint"] = str(path)
        candidates.append((C, family, model, info))
        print(f"SELECT w={w} seed={seed} {family}: C={C:.6g} +/- {se:.3g}", flush=True)
    candidates.sort(key=lambda z: z[0])
    _, best_family, stage1, stage1_info = candidates[0]
    rows = []
    eval_seed = stable_seed("learned-evaluation", w, seed)
    r = learned_sample(stage1, grid, N=eval_N, M=M, seed=eval_seed)
    rows.append(diagnostic_row("learned value + IS", r, grid, eval_seed, w, M,
                               stage="stage1"))
    save_json(OUT / f"stage1_w{w:g}_seed{seed}.json",
              dict(best_family=best_family, candidates=[
                  dict(family=f, info=info) for _, f, _, info in candidates],
                   rows=rows))
    stage2, fine_info = None, None
    if not skip_finetune:
        stage2_path = OUT / f"control_w{w:g}_seed{seed}_finetuned.pt"
        stage2, fine_info = fine_tune_control(
            stage1, grid, seed=seed, updates=finetune_updates, M=M,
            out_path=stage2_path)
        fine_info["checkpoint"] = str(stage2_path)
        r = learned_sample(stage2, grid, N=eval_N, M=M, seed=eval_seed)
        rows.append(diagnostic_row("two-stage learned + IS", r, grid, eval_seed,
                                   w, M, stage="stage2"))
    terminal = terminal_is_sample(grid, eval_N, M, eval_seed)
    rows.append(diagnostic_row("terminal IS", terminal, grid, eval_seed, w, M))
    smc = smc_sample(grid, eval_N, M, eval_seed)
    rows.append(diagnostic_row("adaptive SMC", smc, grid, eval_seed, w, M,
                               raw_weights=False))
    write_csv(OUT / f"metrics_w{w:g}_seed{seed}.csv", rows)
    save_json(OUT / f"run_w{w:g}_seed{seed}.json",
              dict(w=w, seed=seed, paths=paths, M=M, eval_N=eval_N,
                   grid_n=grid_n, best_family=best_family,
                   candidates=[dict(family=f, info=info)
                               for _, f, _, info in candidates],
                   finetune=fine_info, rows=rows))
    return dict(stage1=stage1, stage2=stage2, grid=grid, rows=rows,
                best_family=best_family)


def refine_checkpoint(checkpoint, w=3.0, seed=0, eval_N=20_000,
                      grid_n=320, Ms=(300, 600, 1200)):
    model, _ = restore_model(checkpoint)
    grid = GridDoob2D(GMM_BED, w, n=grid_n)
    rows = []
    for M in Ms:
        eval_seed = stable_seed("learned-refinement", w, seed, M)
        r = learned_sample(model, grid, N=eval_N, M=M, seed=eval_seed)
        rows.append(diagnostic_row("two-stage learned + IS", r, grid, eval_seed,
                                   w, M, stage="refinement"))
        terminal = terminal_is_sample(grid, eval_N, M, eval_seed)
        rows.append(diagnostic_row("terminal IS", terminal, grid, eval_seed, w, M))
        write_csv(OUT / f"refinement_w{w:g}_seed{seed}.csv", rows)
    return rows


def stress_checkpoint(checkpoint, trained_w=3.0, seed=0, eval_N=20_000,
                      M=600, grid_n=320, test_w=2.0):
    model, _ = restore_model(checkpoint)
    rows = []
    for w, initialization in ((trained_w, "standard"), (test_w, "tempered")):
        grid = GridDoob2D(GMM_BED, w, n=grid_n)
        eval_seed = stable_seed("frozen-stress", trained_w, w, initialization, seed)
        r = learned_sample(model, grid, N=eval_N, M=M, seed=eval_seed,
                                   initialization=initialization)
        rows.append(diagnostic_row("frozen learned + IS", r, grid, eval_seed,
                                   w, M, stage="frozen stress",
                                   initialization=initialization))
        terminal = terminal_is_sample(grid, eval_N, M, eval_seed, initialization)
        rows.append(diagnostic_row("terminal IS", terminal, grid, eval_seed,
                                   w, M, initialization=initialization))
    write_csv(OUT / f"frozen_stress_w{trained_w:g}_seed{seed}.csv", rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--w", type=float, default=3.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--families", default="logmse,huber")
    run.add_argument("--paths", type=int, default=60_000)
    run.add_argument("--train-steps", type=int, default=12_000)
    run.add_argument("--eval-N", type=int, default=20_000)
    run.add_argument("--M", type=int, default=600)
    run.add_argument("--grid-n", type=int, default=320)
    run.add_argument("--finetune-updates", type=int, default=120)
    run.add_argument("--selection-N", type=int, default=2048)
    run.add_argument("--skip-finetune", action="store_true")
    run.add_argument("--grad-penalty", type=float, default=0.0)
    ref = sub.add_parser("refine")
    ref.add_argument("checkpoint")
    ref.add_argument("--w", type=float, default=3.0)
    ref.add_argument("--seed", type=int, default=0)
    ref.add_argument("--eval-N", type=int, default=20_000)
    ref.add_argument("--grid-n", type=int, default=320)
    stress = sub.add_parser("stress")
    stress.add_argument("checkpoint")
    stress.add_argument("--trained-w", type=float, default=3.0)
    stress.add_argument("--test-w", type=float, default=2.0)
    stress.add_argument("--seed", type=int, default=0)
    stress.add_argument("--eval-N", type=int, default=20_000)
    stress.add_argument("--M", type=int, default=600)
    stress.add_argument("--grid-n", type=int, default=320)
    args = ap.parse_args()
    if args.command == "run":
        run_seed(args.w, args.seed, tuple(args.families.split(",")), args.paths,
                 args.train_steps, args.eval_N, args.M, args.grid_n,
                 args.finetune_updates, args.selection_N, args.skip_finetune,
                 args.grad_penalty)
    elif args.command == "refine":
        refine_checkpoint(args.checkpoint, args.w, args.seed, args.eval_N,
                          args.grid_n)
    else:
        stress_checkpoint(args.checkpoint, args.trained_w, args.seed,
                          args.eval_N, args.M, args.grid_n, args.test_w)


if __name__ == "__main__":
    main()
