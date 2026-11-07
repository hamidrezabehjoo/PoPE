# When Diffusion Composition Fails

Reproduction code for **“When Diffusion Composition Fails: Exact Composition through Path Measures”** by Hamidreza Behjoo and Misha Chertkov.

This release reproduces the six numerical figures and the associated Gaussian, Gaussian-mixture, resampling, initialization, and optional MNIST diagnostics. The roadmap and three-flow schematic are TikZ artwork in the manuscript and require no numerical code.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python reproduce.py figures
```

This writes both PDF and PNG versions to `figures/`:

- `gaussian_verification`
- `early_stopping`
- `ess_degeneracy`
- `dimension_scaling`
- `gmm2d_degeneracy`
- `gmm2d_target_tv`
- `gmm2d_densities`

The command reads the canonical paper runs in `data/`; it does not rerun the expensive Monte Carlo experiments. Exact Gaussian curves are recomputed rather than stored as image data. To build a subset, use for example:

```bash
python reproduce.py figures --only gaussian,ess,dimension
```

## Regenerate the numerical data

Each suite writes to `results/<suite>/` and never overwrites `data/`. Full settings are the default; `--quick` is only a small functional test and does not reproduce the paper’s numerical accuracy.

```bash
# Gaussian terminal clouds, ESS trajectories, and CFG summary
python reproduce.py simulate gaussian

# Early stopping, high-guidance margins, moment indices, and dimension thresholds
python reproduce.py simulate thresholds

# Gaussian/GMM sweep and full-terminal clouds for the density panels
python reproduce.py simulate gmm

# Deterministic bounded-domain GMM moment scans
python reproduce.py simulate gmm-moments

# Three-seed adaptive-SMC and shared-prior tables
python reproduce.py simulate comparisons
```

To plot regenerated data rather than the bundled paper runs:

```bash
python reproduce.py figures \
  --data-dir results/gmm \
  --output-dir figures/regenerated-gmm \
  --only gmm,densities
```

The full Monte Carlo suites are CPU-intensive. Run them independently and retain their output directories. No cluster-specific scheduler, absolute path, pretrained model, or network access is required by these five suites.

For a custom deterministic moment scan, the solver also exposes its grid controls directly:

```bash
python -m pope.gmm_moments --bed gmm --w 1.82 --lam 2 \
  --L 4 6 8 10 --dx 0.1 --dt-scale 0.25 \
  --out results/custom-gmm-moments.csv
```

## Figure and table map

| Manuscript object | Figure group or regeneration suite |
|---|---|
| Gaussian data-end laws and normalizer chain | `gaussian` / `simulate gaussian` |
| Gaussian verification table | `simulate gaussian` |
| Early stopping and first-moment margin | `early` / `simulate thresholds` |
| ESS trajectories and terminal degeneracy | `ess` / `simulate gaussian` |
| Gaussian moment-index table | `simulate thresholds` |
| Dimension scaling | `dimension` / `simulate thresholds` |
| GMM degeneracy and mean-weight panels | `gmm` / `simulate gmm` |
| GMM target-TV panel | `gmm` / `simulate gmm` |
| GMM density panels | `densities` / `simulate gmm` |
| Deterministic GMM log-moment table | `simulate gmm-moments` |
| Matched-budget terminal-IS/adaptive-SMC table | `simulate comparisons` |
| Shared-prior initialization table | `simulate comparisons` |

The finite-N scaling statements in the text use the same `pope.core.smc_cfg` implementation with the particle counts and stable seeds specified in the manuscript. They are not presented as asymptotic moment estimates.

## Optional pretrained MNIST pilot

The appendix pilot is optional and is not needed for any figure. It requires PyTorch, ArviZ 0.22.0, and a separate local clone of `byrkbrk/diffusion-model` at commit `8437d9e63cc07e4e518b6769bf85d3d9e0886aca`, including its `checkpoints/pretrained_mnist_checkpoint_49.pth` file. Neither third-party code nor the checkpoint is redistributed here.

```bash
python -m pip install -r requirements-mnist.txt
python -m pope.mnist --repo /path/to/diffusion-model \
  --N 192 --steps 12 --w 1.5 2 3 5
```

The pilot is a finite-sample diagnostic for that discrete procedure, not an exact continuous-time validation, an image-quality benchmark, or proof of a moment divergence.

## Tests

Run the release tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

For an additional end-to-end functional check with small simulations:

```bash
python reproduce.py simulate gaussian --quick --output-dir results/test-gaussian
python reproduce.py figures --data-dir results/test-gaussian \
  --output-dir figures/test-gaussian --only gaussian,ess
```

Quick runs are tested for execution and invariants, not agreement with full-run Monte Carlo values.

## Conventions and reproducibility

- One noising clock is used throughout: forward time runs from 0 to 1 and reverse simulation from 1 to 0. There is no auxiliary time reversal.
- All random seeds are deterministic CRC32-based seeds, independent of Python’s salted `hash()`.
- `data/` contains the canonical reported numerical inputs; `results/` and `figures/` are generated and ignored by version control.
- Terminal IS uses exact tempered noise-end initialization unless a shared-prior experiment explicitly states otherwise. The standard-prior correction includes the required initial density ratio.
- Adaptive SMC uses the same naive proposal and left-endpoint potential, with systematic resampling when incremental ESS/N falls below 0.5. Its reported ESS is for final incremental weights.
- MMD is the fixed-unit-bandwidth biased weighted RBF statistic; sliced Wasserstein uses 64 fixed directions. These are finite-sample diagnostics.
- Positive finite-N ESS values beyond a second-moment threshold are finite-population artifacts, not asymptotic ESS values.

## Citation and license

See `CITATION.cff`. The code is released under the MIT license in `LICENSE`.
