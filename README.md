# When Diffusion Composition Fails: Exact Composition through Path Measures

Code accompanying the paper

> **When Diffusion Composition Fails: Exact Composition through Path Measures**
> H. Behjoo and M. Chertkov

The paper studies diffusion-model composition as an exact product of
reverse-time *path measures*,
\(Q \propto \prod_k \big(\mathrm{d}P^{(k)}_\leftarrow / \mathrm{d}P^{0}_\leftarrow\big)^{\beta_k}\,
\mathrm{d}P^{0}_\leftarrow\), with the per-path Feynman–Kac weight driven by the
*disagreement potential* \(\widetilde V^b\). This repository reproduces every
numerical object in the paper: the Gaussian verification example, the weight
degeneracy theory (including the moment/Pareto index \(\lambda^*\)), the
noise-level (\(\nu\)-expert) theorem numerics, the Rényi–de Bruijn identity
check, and the two-dimensional non-Gaussian (GMM) test bed. The second-stage
numerics add exact and approximate Doob control, early-stopping and
high-guidance moment-index calculations, long-horizon VP noise-end checks,
grid/time refinements and replicates, and a small pretrained-MNIST CFG/PSIS
diagnostic pilot. The theory-first revision adds a matched-budget adaptive-SMC
baseline, three-seed controlled comparisons, standard-normal noise-end
initialization with exact density-ratio corrections, and an affine VP
calculation of the resulting data-end error. The learned-control follow-up adds
three-seed \(w=3\) positive-value regression and direct-control policy-gradient
fine-tuning, full approximate-Doob IS evaluation, time refinement, a \(w=1.5\)
finite-moment sanity check, and frozen initialization/guidance stress tests.

## Conventions (enforced throughout the code and the paper)

* Forward noising clock \(t: 0 \to 1\); reverse SDEs run on the **same clock**
  \(t: 1 \to 0\). There is **no** \(\tau = 1 - t\) substitution anywhere.
* Reverse drifts are written without a score symbol:
  \(b^{(k)} = f^{(k)} - g_t^2\,\nabla \log p^{(k)}_t\).
* Exponent sums are general, \(S = \sum_k \beta_k \neq 1\) allowed; CFG is the
  reference-as-expert case \(\alpha_0 = 1 - w\), \(\alpha_c = w\).

## Installation

```bash
python -m pip install -r requirements.txt   # numpy, scipy, matplotlib
# Learned-control follow-up (PyTorch, pandas, xarray, ArviZ):
python -m pip install -r requirements-learned.txt
```

The optional pretrained pilot additionally requires PyTorch and
`arviz==0.22.0` (`python -m pip install -r requirements-pretrained.txt`). Clone <https://github.com/byrkbrk/diffusion-model> separately,
use commit `8437d9e63cc07e4e518b6769bf85d3d9e0886aca`, and retain its
`checkpoints/pretrained_mnist_checkpoint_49.pth`. Neither that repository nor
its checkpoint is redistributed here.

Python >= 3.10. All scripts are deterministic (CRC32-based seeds; no salted
`hash()`), single-machine, and run on CPU (the pilot also accepts `--device`).

## Repository layout

```
code/
  gmm2d.py              core module: beds, scores, SMC, exact Gaussian path MGF,
                        quadrature targets (imported by the drivers below)
  example1_gaussian.py  Example 1: closed forms + SMC + KL certificate
  lambda_star.py        moment/Pareto index lambda*(w) and the exact ESS table
  noise_level.py        nu-expert theorem: SMC check of the data-end laws
  renyi_debruijn.py     Renyi--de Bruijn identity: numeric check (VE and OU)
  gmm2d_figures.py      full figure driver (degeneracy / scaling+TV / densities)
  gmm2d_table_seeds.py  GMM table with mean +/- s.d. over seeds
  gmm2d_scaling.py      finite-N scaling data (plateau vs collapse)
  controlled_gaussian.py exact/approximate Gaussian Doob control, early stopping,
                        high-guidance moment indices, and long-horizon VP checks
  controlled_gmm.py     FFT-grid Doob control, approximate control, MMD and sliced W1
  control_refinement.py refined thresholds, grid/time refinement, and GMM replicates
  controlled_figures.py second-stage manuscript figures from the CSV outputs
  theory_control_checks.py adaptive-SMC baseline, shared-prior terminal IS/Doob,
                        and exact affine VP data-end moment propagation
  theory_control_replicates.py three-seed GMM summaries reported in the manuscript
  pretrained_cfg.py     optional pretrained-MNIST disagreement/ESS/PSIS pilot
  learned_doob.py       positive-value learning, direct-C fine-tuning, full-IS evaluation
  run_w15_sanity.py     secondary finite-moment MSE sanity check
  summarize_learned.py  three-seed summaries and pre-registered success gates
  test_learned_doob.py  suffix-label, weight-sign, boundary, and policy-gradient tests
  test_smoke.py         fast sanity tests (< 2 min)
output/learned_doob/    learned checkpoints, raw metrics, summaries, and logs
```

## Reproduction map

| Paper object | Script | Command | Runtime |
|---|---|---|---|
| Example 1 (Gaussian verification figure/table) | `example1_gaussian.py` | `python example1_gaussian.py` | ~15 s |
| KL certificate numbers (0.1155 ≤ 0.1521 = 0.7969 − 0.6449) | `example1_gaussian.py` | same as above | — |
| \(\lambda^*(w)\) table (5.77 / 3.19 / 1.94 / 1.37 / 1.13) and exact ESS row | `lambda_star.py` | `python lambda_star.py` | ~5 min |
| \(\nu\)-expert data-end laws (N(1.649,0.231) … N(1.252,0.395)) | `noise_level.py` | `python noise_level.py` | ~3 min |
| Rényi–de Bruijn identity (corollary) | `renyi_debruijn.py` | `python renyi_debruijn.py` | < 5 s |
| `gmm2d_degeneracy.pdf`, `gmm2d_scaling_tv.pdf`, `gmm2d_densities.pdf`, sweep CSV | `gmm2d_figures.py` | `python gmm2d_figures.py` | ~50 min |
| GMM table (mean ± s.d. over 8 seeds) | `gmm2d_table_seeds.py` | `python gmm2d_table_seeds.py` | ~30 min |
| Finite-\(N\) scaling data (plateau vs collapse) | `gmm2d_scaling.py` | `python gmm2d_scaling.py` | ~2 h |
| Exact/approximate Gaussian control, VP horizon, early stopping | `controlled_gaussian.py` | `python controlled_gaussian.py` | ~20 min |
| Controlled two-dimensional Gaussian/GMM comparison | `controlled_gmm.py` | `python controlled_gmm.py` | ~30 min |
| Refined moment thresholds and controlled-GMM robustness checks | `control_refinement.py` | `python control_refinement.py` | ~20 min |
| Second-stage figures | `controlled_figures.py` | `python controlled_figures.py` | < 5 s |
| SMC baseline and shared-prior/VP data-end checks | `theory_control_checks.py` | `python theory_control_checks.py` | ~3 min |
| Three-seed SMC and shared-prior manuscript tables | `theory_control_replicates.py` | `python theory_control_replicates.py` | ~3 min |
| Learned \(w=3\), one training seed | `learned_doob.py` | `python learned_doob.py run --w 3 --seed 0 --paths 60000 --train-steps 8000 --eval-N 20000 --M 600` | ~10 min |
| Refine one frozen learned control | `learned_doob.py` | `python learned_doob.py refine output/learned_doob/control_w3_seed0_finetuned.pt --w 3 --seed 0` | ~8 min |
| \(w=1.5\) sanity check | `run_w15_sanity.py` | `python run_w15_sanity.py --seed 0` | ~10 min |
| Frozen prior/guidance stress | `learned_doob.py` | `python learned_doob.py stress output/learned_doob/control_w3_seed0_finetuned.pt --trained-w 3 --test-w 2 --seed 0` | ~5 min |
| Aggregate learned runs | `summarize_learned.py` | `python summarize_learned.py` | < 5 s |
| Pretrained CFG pilot | `pretrained_cfg.py` | `python pretrained_cfg.py --repo /path/to/diffusion-model --N 192 --steps 12 --w 1.5 2 3 5` | model-dependent |

The numerical drivers accept `--quick` for a small smoke-sized run. The
pretrained pilot instead exposes `--N`, `--steps`, and `--w` directly.

## Reproducibility notes

* **Deterministic seeds.** All drivers use `stable_seed(...)` (CRC32); the
  original development code used Python's salted `hash()`, which changes
  between processes.
* **Heavy tails near the threshold.** On the GMM bed the terminal ESS/N is
  seed-stable well below \(w^*_{\rm GMM}\approx 1.55\) (e.g. \(0.793\pm0.001\)
  at \(w=0.5\)) but becomes heavy-tailed near and above it: at \(w=1.5\) one
  seed in eight collapses (median 0.552, full range [0.004, 0.578] at
  \(N=6\times10^4\)); at \(w=1.75\) individual collapsed draws below 0.03
  appear by \(N=10^5\) while \(\mathbb E W\) remains accurate. This is the
  \(\lambda^*\in(2,4)\) regime of the paper — always report mean ± s.d. over
  seeds, never single-seed ESS values.
* **Original SMC checks.** These use exact tempered noise-end initialization
  (one importance-resampling step) and left-endpoint path weights, with no
  intermediate resampling. The separately reported adaptive-SMC baseline is
  described below.
* **Controlled samplers.** Exact Doob runs have uniform weights by
  construction. The approximate-control run uses \(\hat u=0.8u^*\); its path
  weight includes the control Girsanov factor. At exact control, the discrete
  path-weight standard deviation decreases under time refinement. The GMM
  implementation uses periodic FFT convolution and spectral scores; reported
  MMD values are fixed-size biased weighted statistics, not unbiased MMD
  estimates or asymptotic moment criteria.
* **Adaptive-SMC baseline.** The comparison uses the same naive proposal and
  left-endpoint Feynman–Kac potential as terminal IS, with systematic
  resampling when incremental ESS/N falls below 0.5. All methods have the
  same particle and time-step budgets. SMC's reported ESS is for the final
  incremental weights, not the full resampled genealogy. Wall times describe
  only this implementation.
* **Shared-prior checks.** Corrected terminal IS from \(N(0,I)\) includes
  \(p_1^{\beta,S}/r_1\); corrected exact Doob includes \(q_1/r_1\). Runs
  without these factors are intentionally uncorrected initialization-mismatch
  diagnostics. The VP data-end errors are exact discrete affine moment
  calculations, not Monte Carlo estimates.
* **Learned-control scope.** Training uses only naive-path suffix weights and
  exact component scores, never the exact grid Doob control. Stage 2 is a
  randomized policy-gradient estimator with time subsampling and a
  leave-one-out baseline, not full backpropagation through 600 transitions.
  Frozen evaluation retains the full potential and Girsanov weight. The
  two-dimensional results are a proof of concept, not a scalability claim;
  high PSIS tail estimates preclude a finite-variance-restoration claim.
* **Pretrained pilot limitations.** The reported pilot uses only 192
  trajectories per guidance value and 12 ancestral posterior jumps on the
  original 500-step schedule. It diagnoses score disagreement and weight
  tails for that discrete procedure; it is not an image-quality benchmark,
  an exact continuous-time validation, or a dimension-scaling study.

## Citation

```bibtex
@article{behjoo2026diffusion,
  title   = {When Diffusion Composition Fails: Exact Composition through Path Measures},
  author  = {Behjoo, H. and Chertkov, M.},
  year    = {2026},
  note    = {manuscript}
}
```

## License

MIT (see `LICENSE`). Please replace the copyright line with the author names.
