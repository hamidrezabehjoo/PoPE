# Experiment spec: real-data dihedral bed (planned Section 6.5)

Status: **planned** (not yet implemented). This document is the binding
specification for the real-data experiment; implement against it.

## Experiment matrix

| # | Name | Experts | Scores | Purpose |
|---|---|---|---|---|
| E1 | Helix pair | p0 = general-case Ramachandran, pc = DSSP-H | exact (grid) | main real-data bed |
| E2 | Strand pair | p0 same, pc = DSSP-E | exact (grid) | threshold contrast |
| E3 | Trained-net pair | same data | learned (condition-dropout MLP) | score-error study |
| E4 | Exact thresholds | E1/E2 fields | grid | lambda*(w), w* by torus PDE |
| E5 | KL certificate | E1 machinery | grid | certificate tightness |

## Data pipeline (`code/dihedral/data_prep.py`)

1. Top8000 chain list (`rlabduke/reference_data`, Richardson lab, Duke);
   extract per-residue (phi, psi) + DSSP (8->3: H,G,I->H; E,B->E; else C).
2. Filters: drop chain-terminal residues; drop any residue with mainchain
   B-factor > 30 (MolProbity convention); restrict to the MolProbity
   "general case" (exclude Gly, Pro, pre-Pro, Ile/Val). State all of this
   in the paper.
3. Grid: torus [-pi, pi)^2, 360x360 (1 deg). Empirical densities by
   wrapped-Gaussian kernel smoothing, bandwidth 5 deg nominal; sensitivity
   at {3, 8} deg in appendix; density floor eps = 1e-12 * max (documented).
4. Expected counts: general ~1.2M; H ~0.5M; E ~0.3M residues.

## Forward process / exact grid experts (tier 1)

- VE on the torus, f = 0, g = 1, t in [0, T], T = 10: wrapped heat kernel
  K_t with Fourier coefficients exp(-t|k|^2/2); noise end uniform to
  2 e^{-T/2} ~ 0.7%  => shared-prior assumption literally true, SMC init
  exact (uniform), no init-resampling approximation.
- p_t = K_t * p0 by FFT; exact scores s_t = grad p_t / p_t (spectral grad,
  bilinear interpolation to particles). No model in tier 1.
- Time slices: dt = 5e-3 (M = 2000), score fields cached or computed on
  the fly by FFT.

## SMC protocol (`code/dihedral/smc_dihedral.py`)

Mirrors `code/gmm2d.py::smc_cfg` with wrapped transitions:
  x <- wrap(x - b~ dt + sqrt(dt) xi),  b~ = -[(1-w) s0_t + w sc_t],
  logW += (w(w-1)/2) ||sc_t - s0_t||^2 dt   (left endpoint, no resampling).
Defaults: N = 6e4, M = 2000; 8 seeds per reported point (mean +/- s.d.,
median, range); w-sweep {0.25,...,3}; N-scaling {3e4, 6e4, 1e5, 3e5} at the
two w's bracketing the PDE threshold. Quadrature targets q_w and
Z_t(w) on the 360x360 grid; wrapped histogram TV; E W vs Z_0/Z_1.

## Exact thresholds by torus PDE (`code/dihedral/pde_lambda_star.py`)

E W^lam = int p_T^(w)(x) psi(x,T;lam) dx,
  -d_t psi = L_t psi + lam (w(w-1)/2) ||Ds_t||^2 psi,  psi(.,0) = 1,
  L_t = b~_t . grad + (1/2) Delta   (naive-composed generator).
Implicit Euler t: 0 -> T, matrix-free: Delta diagonal in Fourier,
b~.grad by spectral gradient x grid drift, potential pointwise;
BiCGSTAB preconditioned by (I - dt/2 Delta)^{-1}. Grid 180x180 default
(validate one point at 360x360). Blow-up: growth of log ||psi(.,T)||_inf
and the tilted-generator Rayleigh quotient; bisection in lam -> lambda*(w);
w* = level set lambda* = 2; w in {1.25, 1.5, 1.75, 2, 2.5, 3}, both pairs.

## Trained-net tier (`code/dihedral/train_score_net.py`)

Conditional score MLP, CFG-style: inputs (sin phi, cos phi, sin psi,
cos psi) + 64-dim Fourier time embedding + 16-dim condition embedding over
{H, E, C, 0} with dropout p = 0.15; 4x512 SiLU (~1M params). DSM loss
against the exact wrapped-kernel conditional score (few lattice wraps;
t log-uniform on [1e-3, 10]). Adam 2e-4, batch 4096, 100k steps.
Then the full SMC sweep with network scores; analyses: (a) ESS/N and TV vs
the exact grid target; (b) measured int g^2 E||s_hat - s||^2 dt vs the
|alpha_k|-amplified Girsanov bound; (c) threshold shift grid-vs-learned.

## Registered predictions (falsifiable)

1. w*_E < w*_H (strand conditional sits in a secondary basin of p0 with
   beta/PPII substructure -> larger disagreement mass; also smaller
   Chernoff overlap Z_0/Z_1). Report int E||Ds||^2 dt for both pairs.
2. ESS/N plateau/collapse at the PDE thresholds, both pairs.
3. PSIS: 1/k_hat from terminal weights tracks exact lambda*(w) (E4).
4. Failure of the ordering in (1) falsifies the multimodality intuition.

## KL certificate demonstration (E5)

KL(P||Q) = int (E_naive[V~_t] - E_geomix[V~_t]) dt with
E_geomix[V~_t] = d_t log Z_t by grid quadrature (no MC error) and
E_naive[V~_t] recorded per-particle along naive tier-1 runs. Report
certificate vs measured KL(naive_0 || q_w) and the Pinsker TV corollary
across w.

## Deliverables

Fig. A: ESS/N vs w (both pairs, seed error bars, PDE thresholds) +
       E W vs Z_0/Z_1.
Fig. B: N-scaling plateau/collapse; TV naive vs corrected.
Fig. C: Ramachandran HDR panels, q_w vs sampler clouds,
       w in {0.5, w*-eps, w*+eps}.
Fig. D: lambda*(w) (PDE, exact) vs 1/k_hat (PSIS), both pairs.
Table 4: w x {ESS/N mean+/-sd, E W, Z_0/Z_1, TV naive, TV corrected},
        both pairs (format of revised tab:gmm2d).
Table 5: KL certificate vs measured KL/TV.
Appendix: bandwidth sensitivity, DSSP mapping, refinement checks,
         RFdiffusion paragraph (zero-training; reinterpretation of the
         documented noise-scale quality/diversity tradeoff).

## Compute budget (1 GPU / 8 CPU cores)

data prep ~2-4 h CPU; tier-1 sweeps E1+E2 ~4-6 h CPU; N-scaling ~3 h;
PDE thresholds ~6-12 h CPU (1 h/GPU); training + net sweep ~3-4 h GPU.
Total ~2-3 days.

## Risks

E-subset sparsity (bandwidth sensitivity, possibly 2-deg grid); score
blow-up in empty regions (eps-floor, appendix sensitivity); PDE cost
(180^2 default, GPU fallback); DSSP label noise (robustness check);
scope creep (no RFdiffusion runs, no third bed, tier 2 = one paragraph
+ one panel).
