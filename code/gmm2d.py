#!/usr/bin/env python3
"""
Core module for the 2D test beds of
"When Diffusion Composition Fails: Exact Products of Path Measures"
(Behjoo & Chertkov) -- Section 4.4 (non-Gaussian GMM test bed), using the
SMC scheme of the CFG/degeneracy sections.

Setup (paper conventions): forward SDE f=0, g=1 on [0,1]; reference expert p0
and conditional expert pc share the forward drift; CFG path product
Q_w ~ (dP^c/dP^0)^w dP^0 with naive drift b~ = b0 + w(bc - b0); exact tempered
target q_w ~ (p0)^(1-w) (pc)^w; SMC weight
exp(+w(w-1)/2 * int ||grad log pc_t - grad log p0_t||^2 dt)
(the reference-as-expert case S=1, alpha_0 = 1-w < 0 for w > 1, hence the
*negative* -- branching -- potential and weights that grow).

Two beds, matched means/covariances, differing only in multimodality:
  Gaussian: p0 = N(0, 2.25 I2), pc = N((0.8,0), 0.25 I2)
  GMM:      p0 = 1/4 sum N((+-1.2,+-1.2), 0.81 I2)   (mean 0, var/axis 2.25)
            pc = 1/4 sum N((0.8+-0.4, +-0.4), 0.09 I2) (mean (0.8,0), var/axis 0.25)

Changelog vs the original development script (gmm2d_experiment.py):
  * docstring: "Section 6.4" corrected to Section 4.4;
  * salted hash() seeds replaced by deterministic stable_seed() (CRC32);
  * the "if N >= 100_000: break" early exit (which silently skipped the
    N = 3e5 scaling point) removed -- the loop lives in gmm2d_figures.py.

Dependencies: numpy, scipy, matplotlib.
"""
import math
import zlib

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "GAUSS_BED", "GMM_BED", "gmm_pdf", "gmm_score_fast", "gmm_sample",
    "p0_pdf", "pc_pdf", "p0_score", "pc_score", "tempered_on",
    "Zratio_grid", "tv_to_target", "smc_cfg", "gaussian_path_mgf",
    "stable_seed", "TVBINS", "TVEDGE", "TVCX", "TVCY", "TVXY", "FINE", "FXY",
]


def stable_seed(*key) -> int:
    """Deterministic 31-bit seed from a key (process-independent, unlike
    built-in hash(), which is salted for str/tuple)."""
    return zlib.crc32(repr(key).encode()) % 2**31


# ============================ GMM machinery ====================================
def gmm_pdf(X, means, vars_, weights):
    d = X[:, None, :] - means[None, :, :]
    expo = -0.5 * np.einsum('njd,njd->nj', d, d) / vars_[None, :]
    pk = np.exp(expo) / (2 * np.pi * vars_[None, :])
    return pk @ weights


def gmm_score_fast(X, means, vars_, weights, t):
    v = vars_ + t
    d2 = (X * X).sum(1)[:, None] - 2.0 * (X @ means.T) + (means * means).sum(1)[None, :]
    pk = np.exp(-0.5 * d2 / v[None, :]) / (2 * np.pi * v[None, :])
    post = pk * weights[None, :]
    post /= post.sum(axis=1, keepdims=True)
    return -((post @ (1.0 / v))[:, None] * X - post @ (means / v[:, None]))


def gmm_sample(N, means, vars_, weights, t, rng):
    v = vars_ + t
    idx = rng.choice(len(weights), size=N, p=weights)
    return means[idx] + np.sqrt(v[idx])[:, None] * rng.standard_normal((N, 2))


GAUSS_BED = dict(name='Gaussian',
                 m0=np.array([[0.0, 0.0]]), v0=np.array([2.25]), w0=np.array([1.0]),
                 mc=np.array([[0.8, 0.0]]), vc=np.array([0.25]), wc=np.array([1.0]))
GMM_BED = dict(name='GMM',
               m0=np.array([[-1.2, -1.2], [1.2, -1.2], [-1.2, 1.2], [1.2, 1.2]]),
               v0=np.array([0.81] * 4), w0=np.array([0.25] * 4),
               mc=np.array([[0.4, -0.4], [1.2, -0.4], [0.4, 0.4], [1.2, 0.4]]),
               vc=np.array([0.09] * 4), wc=np.array([0.25] * 4))


def p0_pdf(X, bed, t):   return gmm_pdf(X, bed['m0'], bed['v0'] + t, bed['w0'])
def pc_pdf(X, bed, t):   return gmm_pdf(X, bed['mc'], bed['vc'] + t, bed['wc'])
def p0_score(X, bed, t): return gmm_score_fast(X, bed['m0'], bed['v0'], bed['w0'], t)
def pc_score(X, bed, t): return gmm_score_fast(X, bed['mc'], bed['vc'], bed['wc'], t)


# ============================ grids / targets ==================================
TVBINS = 240
TVEDGE = np.linspace(-6, 6, TVBINS + 1)
TVCX = 0.5 * (TVEDGE[1:] + TVEDGE[:-1])
TVCY = 0.5 * (TVEDGE[1:] + TVEDGE[:-1])
TVXY = np.stack(np.meshgrid(TVCX, TVCY, indexing='ij'), axis=-1).reshape(-1, 2)
FINE = np.linspace(-6, 6, 600)
FXY = np.stack(np.meshgrid(FINE, FINE, indexing='ij'), axis=-1).reshape(-1, 2)


def tempered_on(xy, bed, w, t):
    """Unnormalized tempered marginal m_t(x) = p0_t^(1-w) pc_t^w."""
    return p0_pdf(xy, bed, t) ** (1 - w) * pc_pdf(xy, bed, t) ** w


def Zratio_grid(bed, w):
    """Quadrature value of hat Z_0 / hat Z_1 on the 600x600 grid."""
    m0 = tempered_on(FXY, bed, w, 0.0)
    m1 = tempered_on(FXY, bed, w, 1.0)
    return m0.sum() / m1.sum()


def tv_to_target(x, wn, bed, w):
    """Histogram TV of the (weighted) sample cloud to the exact tempered
    target q_w, on the fixed 240x240 grid (floor ~0.07-0.09)."""
    h, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=TVBINS,
                             range=((-6, 6), (-6, 6)), weights=wn)
    ph = h / h.sum()
    q = tempered_on(TVXY, bed, w, 0.0)
    qb = (q / q.sum()).reshape(TVBINS, TVBINS)
    return 0.5 * np.abs(ph - qb).sum()


# ============================ SMC (paper's CFG section) ========================
def smc_cfg(bed, w, N=60_000, M=2000, seed=0, snapshots=(0.75, 0.5, 0.25)):
    """Paper's scheme: exact tempered noise-end init (one resampling step),
    composed drift, left-endpoint path weights. No intermediate resampling.

    Returns dict with data-end cloud x0, normalized weights wn, terminal
    ESS/N, mean weight E[W], sd of log W, and ESS snapshots along the path.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / M
    th = 0.5 * w * (w - 1.0)
    x = gmm_sample(N, bed['m0'], bed['v0'], bed['w0'], 1.0, rng)      # r1 = p0_1
    log_init = w * (np.log(pc_pdf(x, bed, 1.0) + 1e-300)
                    - np.log(p0_pdf(x, bed, 1.0) + 1e-300))
    lw = log_init - logsumexp(log_init)
    x = x[rng.choice(N, size=N, p=np.exp(lw))]                       # -> p^{(w)}_1
    logW = np.zeros(N)
    ess_snap = {}
    snap_set = {int(round((1 - t) / dt)): t for t in snapshots}
    for i in range(M):
        t = 1.0 - i * dt
        s0 = p0_score(x, bed, t)
        sc = pc_score(x, bed, t)
        if th != 0.0:
            d = sc - s0
            logW += th * np.einsum('nd,nd->n', d, d) * dt
        b = -((1 - w) * s0 + w * sc)
        x = x - b * dt + np.sqrt(dt) * rng.standard_normal(x.shape)
        if i + 1 in snap_set:
            lwi = logW - logsumexp(logW)
            ess_snap[snap_set[i + 1]] = float(np.exp(-logsumexp(2 * lwi))) / N
    lwT = logW - logsumexp(logW)
    return dict(x0=x, logW=logW, wn=np.exp(lwT),
                ess=float(np.exp(-logsumexp(2 * lwT))) / N,
                EW=float(np.exp(logW).mean()), sdlogW=float(np.std(logW)),
                ess_snap=ess_snap)


# ================= exact Gaussian MGF (discrete reverse-Euler path) ============
def gaussian_path_mgf(v0, vc, mu, w, lam, M=500):
    """E[exp(lam * w(w-1)/2 * sum_i (A_i X_i - D_i)^2 dt)] for the discrete
    reverse-Euler path under b~ with X_0 ~ tempered noise end. Infinite when
    I - 2 lam th * |A|sqrt(dt) C |A|sqrt(dt) is not positive definite
    (Cameron-Martin threshold).

    Returns (value, is_finite). Per axis; the 2D Gaussian bed factorizes into
    the mean axis (mu=0.8) and the variance axis (mu=0.0).
    """
    dt = 1.0 / M
    ts = 1.0 - np.arange(M + 1) * dt
    v0t, vct = v0 + ts, vc + ts
    A = 1.0 / vct - 1.0 / v0t
    D = mu / vct
    P = (1 - w) / v0t + w / vct
    Q = w * mu / vct
    P1 = (1 - w) / (v0 + 1.0) + w / (vc + 1.0)
    m = np.empty(M + 1); m[0] = (w * mu / (vc + 1.0)) / P1
    Sig = np.empty(M + 1); Sig[0] = 1.0 / P1
    logF = np.concatenate([[0.0], np.cumsum(np.log(np.abs(1 - P[:-1] * dt)))])
    for i in range(M):
        m[i + 1] = m[i] - (P[i] * m[i] - Q[i]) * dt
        Sig[i + 1] = (1 - P[i] * dt) ** 2 * Sig[i] + dt
    ii = np.arange(M + 1)[:, None]
    jj = np.arange(M + 1)[None, :]
    mn = np.minimum(ii, jj)
    mx = np.maximum(ii, jj)
    C = Sig[mn] * np.exp(logF[mx] - logF[mn])
    th = 0.5 * w * (w - 1.0)
    const = lam * th * np.sum(D[:-1] ** 2 * dt)
    Sq = C[:-1, :-1]
    mq = m[:-1]
    B = lam * th * np.diag(A[:-1] ** 2 * dt)
    bvec = -2 * lam * th * A[:-1] * D[:-1] * dt
    Dd = np.abs(A[:-1]) * np.sqrt(dt)
    Mm = (Dd[:, None] * Sq) * Dd[None, :]
    try:
        np.linalg.cholesky(np.eye(M) - 2 * lam * th * Mm)
    except np.linalg.LinAlgError:
        return np.inf, False
    Ib2 = np.eye(M) - 2 * Sq @ B
    sign, logdet = np.linalg.slogdet(Ib2)
    if sign <= 0:
        return np.inf, False
    u = np.linalg.solve(Sq, mq)
    wv = u + bvec
    val = 0.5 * wv @ np.linalg.solve(Ib2, Sq @ wv) - 0.5 * mq @ u
    return math.exp(min(700.0, const + val - 0.5 * logdet)), True
