# When Diffusion Composition Fails: Exact Products of Path Measures

Code accompanying the paper

> **When Diffusion Composition Fails: Exact Products of Path Measures**
> S. Behjoo and M. Chertkov

The paper studies diffusion-model composition as an exact product of
reverse-time *path measures*,
\(Q \propto \prod_k \big(\mathrm{d}P^{(k)}_\leftarrow / \mathrm{d}P^{0}_\leftarrow\big)^{\beta_k}\,
\mathrm{d}P^{0}_\leftarrow\), with the per-path Feynman–Kac weight driven by the
*disagreement potential* \(\widetilde V^b\). This repository reproduces every
numerical object in the paper: the Gaussian verification example, the weight
degeneracy theory (including the moment/Pareto index \(\lambda^*\)), the
noise-level (\(\nu\)-expert) theorem numerics, the Rényi–de Bruijn identity
check, and the two-dimensional non-Gaussian (GMM) test bed.

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
```

Python >= 3.10. All scripts are deterministic (CRC32-based seeds; no salted
`hash()`), single-machine, CPU-only. Runtimes below are for a 2-core laptop.

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
  test_smoke.py         fast sanity tests (< 2 min)
output/                 all figures and CSVs land here
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

Every script accepts `--quick` for a small smoke-sized run.

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
* The SMC scheme uses the exact tempered noise-end initialization (one
  importance-resampling step) and left-endpoint path weights, exactly as
  described in the paper; there is **no** intermediate resampling.

## Citation

```bibtex
@article{behjoo2026diffusion,
  title   = {When Diffusion Composition Fails: Exact Products of Path Measures},
  author  = {Behjoo, S. and Chertkov, M.},
  year    = {2026},
  note    = {manuscript}
}
```

## License

MIT (see `LICENSE`). Please replace the copyright line with the author names.
