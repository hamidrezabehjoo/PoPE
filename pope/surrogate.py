"""Training-free Gaussian-surrogate controlled residual IS for VE/CFG.

The proposal and weight use the paper's single noising clock. Reverse simulation
runs from t=1 to t=0, with f=0 and g=1. No resampling or learned control is used.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .core import GAUSS_BED, GMM_BED
from .diagnostics import normalize, write_csv

__all__ = [
    "GaussianSurrogate", "TemperedGrid", "controlled_cfg", "initial_log_weight",
    "logp_score", "moment_matched_pair", "run_surrogate", "tempered_grid",
]

METHODS = ("naive", "ctrl_p1", "ctrl_q1")
GUIDANCE = (1.5, 2.0, 3.0, 5.0)


def logp_score(x, means, variances, weights, t):
    """Log density and score of an isotropic Gaussian mixture at time t."""
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    vt = variances + t
    d = x[:, None, :] - means[None, :, :]
    dim = x.shape[1]
    log_components = (-0.5 * np.einsum("ncd,ncd->nc", d, d) / vt[None, :]
                      - 0.5 * dim * np.log(2.0 * np.pi)
                      - 0.5 * dim * np.log(vt)[None, :]
                      + np.log(weights)[None, :])
    log_norm = logsumexp(log_components, axis=1)
    posterior = np.exp(log_components - log_norm[:, None])
    score = -np.einsum("nc,ncd->nd", posterior / vt[None, :], d)
    return log_norm, score


def moment_matched_pair(bed):
    """Return the isotropic Gaussian pair moment-matched to a CFG test bed."""
    matched = []
    for prefix in ("0", "c"):
        means = np.asarray(bed["m" + prefix], dtype=float)
        variances = np.asarray(bed["v" + prefix], dtype=float)
        weights = np.asarray(bed["w" + prefix], dtype=float)
        weights = weights / weights.sum()
        mean = weights @ means
        centered = means - mean
        second = variances[:, None] + centered * centered
        variance = weights @ second
        if not np.allclose(variance, variance[0], rtol=0.0, atol=1e-12):
            raise ValueError("The surrogate release expects an isotropic test bed")
        matched.append((mean, float(variance[0])))
    (mean0, v0), (meanc, vc) = matched
    if not np.allclose(mean0, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("The CFG surrogate expects a zero-mean reference")
    return v0, vc, meanc


@dataclass(frozen=True)
class GaussianSurrogate:
    """Exact Doob control for the moment-matched Gaussian CFG pair."""
    w: float
    v0: float
    vc: float
    c: np.ndarray

    @classmethod
    def from_bed(cls, bed, w):
        v0, vc, c = moment_matched_pair(bed)
        return cls(float(w), v0, vc, np.asarray(c, dtype=float))

    def __post_init__(self):
        if self.w < 0:
            raise ValueError("This CFG implementation requires w >= 0")
        # For w>1, precision is decreasing; for 0<=w<=1 it is positive.
        if self.precision(1.0) <= 0 or self.precision(0.0) <= 0:
            raise ValueError("The Gaussian surrogate product is non-normalizable")

    def precision(self, t):
        t = np.asarray(t)
        return (1.0 - self.w) / (self.v0 + t) + self.w / (self.vc + t)

    def mean(self, t):
        t = np.asarray(t)
        return (self.w * self.c / (self.vc + t)[..., None]) / self.precision(t)[..., None]

    @property
    def mu0(self):
        return self.mean(0.0)

    @property
    def q1_variance(self):
        return 1.0 / float(self.precision(0.0)) + 1.0

    def control(self, x, t):
        """Evaluate u_hat_t(x); the controlled motion drift adds u_hat."""
        x = np.asarray(x, dtype=float)
        return (-(x - self.mu0) / (1.0 / float(self.precision(0.0)) + t)
                + float(self.precision(t)) * (x - self.mean(t)))


@dataclass(frozen=True)
class TemperedGrid:
    """Exact tempered marginal on a fixed grid, used only for initialization."""
    points: np.ndarray
    probability: np.ndarray
    spacing: float

    def sample(self, N, rng):
        indices = rng.choice(len(self.probability), size=N, p=self.probability)
        x = self.points[indices].copy()
        return x + self.spacing * (rng.random((N, 2)) - 0.5)


def tempered_grid(bed, w, t, L=7.0, n=701):
    """Construct the normalized tempered marginal on [-L,L]^2."""
    axis = np.linspace(-L, L, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    log0, _ = logp_score(points, bed["m0"], bed["v0"], bed["w0"], t)
    logc, _ = logp_score(points, bed["mc"], bed["vc"], bed["wc"], t)
    logm = (1.0 - w) * log0 + w * logc
    probability = np.exp(logm - logsumexp(logm))
    return TemperedGrid(points, probability, float(axis[1] - axis[0]))


def initial_log_weight(x, bed, w, method, surrogate):
    """Return the initial residual factor; m_1/q_hat_1 differs from p_1/q_hat_1
    only by a w-dependent constant and hence gives identical normalized weights.
    """
    if method in ("naive", "ctrl_p1"):
        return np.zeros(len(x))
    if method != "ctrl_q1":
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    log0, _ = logp_score(x, bed["m0"], bed["v0"], bed["w0"], 1.0)
    logc, _ = logp_score(x, bed["mc"], bed["vc"], bed["wc"], 1.0)
    centered = x - surrogate.mu0
    logr = (-0.5 * np.einsum("nd,nd->n", centered, centered) / surrogate.q1_variance
            - 0.5 * x.shape[1] * np.log(2.0 * np.pi * surrogate.q1_variance))
    return (1.0 - w) * log0 + w * logc - logr


def controlled_cfg(bed, w, method, N=20_000, M=600, seed=0,
                   grid_L=7.0, grid_n=701, noise_grid=None):
    """Run naive terminal IS or Gaussian-surrogate controlled residual IS.

    ``ctrl_p1`` starts at the exact tempered noise end. ``ctrl_q1`` starts at the
    surrogate noise end and includes the initial ratio exactly once. All methods
    retain their full residual path weight and perform no resampling.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if N <= 0 or M <= 0:
        raise ValueError("N and M must be positive")
    rng = np.random.default_rng(seed)
    surrogate = GaussianSurrogate.from_bed(bed, w)
    if method in ("naive", "ctrl_p1"):
        if noise_grid is None:
            noise_grid = tempered_grid(bed, w, 1.0, L=grid_L, n=grid_n)
        x = noise_grid.sample(N, rng)
    else:
        x = (surrogate.mu0 + np.sqrt(surrogate.q1_variance)
             * rng.standard_normal((N, 2)))
    logw = initial_log_weight(x, bed, w, method, surrogate)
    dt = 1.0 / M
    score_calls = 0
    for i in range(M):
        t = 1.0 - i * dt
        _, score0 = logp_score(x, bed["m0"], bed["v0"], bed["w0"], t)
        score_calls += N
        _, scorec = logp_score(x, bed["mc"], bed["vc"], bed["wc"], t)
        score_calls += N
        difference = scorec - score0
        composed_score = (1.0 - w) * score0 + w * scorec
        potential = 0.5 * (1.0 - w) * w * np.einsum(
            "nd,nd->n", difference, difference)
        xi = np.sqrt(dt) * rng.standard_normal((N, 2))
        logw -= potential * dt
        motion_drift = composed_score
        if method != "naive":
            control = surrogate.control(x, t)
            motion_drift = motion_drift + control
            logw -= (np.einsum("nd,nd->n", control, xi)
                     + 0.5 * np.einsum("nd,nd->n", control, control) * dt)
        x = x + motion_drift * dt + xi
    wn, ess = normalize(logw)
    return dict(x=x, wn=wn, logW=logw, ess=ess,
                log_weight_sd=float(np.std(logw)),
                mean_weight=float(np.exp(logsumexp(logw) - np.log(N))),
                score_calls=score_calls)


def run_surrogate(output_dir, quick=False):
    """Regenerate the surrogate table and exact-Gaussian implementation check."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N, M, grid_n = (2000, 100, 101) if quick else (20_000, 600, 701)
    rows = []
    for w in GUIDANCE:
        noise_grid = tempered_grid(GMM_BED, w, 1.0, n=grid_n)
        for method in METHODS:
            for replicate in range(3):
                result = controlled_cfg(GMM_BED, w, method, N=N, M=M,
                                        seed=replicate, noise_grid=noise_grid)
                rows.append(dict(w=w, method=method, replicate=replicate,
                                 N=N, M=M, grid_n=grid_n, ess=result["ess"],
                                 log_weight_sd=result["log_weight_sd"]))
                print(f"surrogate w={w:g}, {method}, seed={replicate}: "
                      f"ESS/N={result['ess']:.6f}", flush=True)
                write_csv(output_dir / "surrogate_control_replicates.csv", rows)
    # The reported surrogate table uses the population s.d. over its three seeds,
    # matching the supplied experiment script (numpy.std's ddof=0 convention).
    summary = []
    for w in GUIDANCE:
        for method in METHODS:
            selected = [row for row in rows
                        if row["w"] == w and row["method"] == method]
            ess = np.array([row["ess"] for row in selected])
            log_sd = np.array([row["log_weight_sd"] for row in selected])
            summary.append(dict(w=w, method=method, n_replicates=len(selected),
                                ess_mean=float(ess.mean()), ess_sd=float(ess.std()),
                                log_weight_sd_mean=float(log_sd.mean()),
                                log_weight_sd_sd=float(log_sd.std()),
                                N=N, M=M, grid_n=grid_n))
    write_csv(output_dir / "surrogate_control.csv", summary)

    checks = []
    for w in GUIDANCE:
        result = controlled_cfg(GAUSS_BED, w, "ctrl_q1", N=N, M=M, seed=0)
        checks.append(dict(w=w, method="ctrl_q1", seed=0, N=N, M=M,
                           ess=result["ess"], log_weight_sd=result["log_weight_sd"]))
        print(f"exact Gaussian w={w:g}: ESS/N={result['ess']:.6f}", flush=True)
    write_csv(output_dir / "surrogate_gaussian_check.csv", checks)
