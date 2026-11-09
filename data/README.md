# Paper data

These files are the canonical numerical inputs for the figures and tables in the
current manuscript. They are data, not caches: the plotting code never contains
a second hard-coded copy of the reported values.

- `gaussian_verification.npz`: terminal samples, normalized terminal weights,
  and mean-weight snapshots for the two-expert Gaussian example.
- `gaussian_ess_trajectory.npz`: ESS trajectories for the two-expert and CFG
  Gaussian runs.
- `gaussian_cfg_summary.csv`: exact and Monte Carlo CFG degeneracy diagnostics.
- `early_stopping.csv`, `high_guidance.csv`, `dimension_thresholds.csv`:
  deterministic Gaussian moment calculations.
- `gmm_summary.csv`: reported Gaussian/GMM sweep summary. GMM means and sample
  standard deviations use the seed counts in the final column; TV columns are
  the fixed full-run histogram diagnostics used in the figures.
- `gmm_density_w*.npz`: full-run terminal GMM clouds and weights for the three
  density columns.
- `gmm_exact_moments.csv`: deterministic bounded-domain log-moment scans.
- `smc_comparison.csv`, `shared_prior.csv`: three-seed summaries for the two
  initialization/resampling tables.
- `surrogate_control.csv`, `surrogate_control_replicates.csv`: three-seed
  Gaussian-surrogate controlled residual-IS ESS summaries and individual runs.
  Standard deviations in this table use the population (`ddof=0`) convention.
- `surrogate_gaussian_check.csv`: exact-Gaussian surrogate checks at seed 0.
- `mnist_pilot.csv`: the optional 192-trajectory, 12-jump pretrained diagnostic.

`python reproduce.py figures` reads these files and writes publication figures.
The `simulate` subcommands write regenerated files under `results/`; they never
overwrite this directory. Wall-time columns are implementation- and
machine-dependent. Positive finite-sample ESS values above a second-moment
threshold are finite-population artifacts, not asymptotic ESS estimates.
