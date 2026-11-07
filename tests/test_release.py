import unittest
from pathlib import Path

import numpy as np

from pope.core import GAUSS_BED, GMM_BED, gaussian_path_mgf, smc_cfg, stable_seed
from pope.diagnostics import ReferenceGrid, smc_resampling_cfg, weighted_quantile
from pope.gaussian import (critical_w, exact_ess_1d, mixture_mom,
                           naive_law_and_certificate, var_logw_1d, zhat)

ROOT = Path(__file__).resolve().parents[1]


class ClosedFormTests(unittest.TestCase):
    def test_gaussian_example(self):
        mean, var, integral_naive, integral_target = naive_law_and_certificate(n_steps=20_000)
        self.assertAlmostEqual(mean, 0.8321, places=3)
        self.assertAlmostEqual(var, 0.5225, places=3)
        self.assertAlmostEqual(zhat(0.0) / zhat(1.0), 0.52473, places=4)
        target_mean, target_var = mixture_mom(0.0)
        kl = 0.5 * (var / target_var - 1 + (target_mean - mean)**2 / target_var
                    + np.log(target_var / var))
        bound = integral_naive - np.log(zhat(1.0) / zhat(0.0))
        self.assertAlmostEqual(kl, 0.1155, places=3)
        self.assertLessEqual(kl, bound + 1e-8)
        self.assertAlmostEqual(integral_target,
                               np.log(zhat(1.0) / zhat(0.0)), places=3)

    def test_path_moment_feasibility(self):
        _, finite = gaussian_path_mgf(2.25, 0.25, 0.8, 1.5, 1.0, M=120)
        _, infinite = gaussian_path_mgf(2.25, 0.25, 0.8, 5.0, 2.0, M=120)
        self.assertTrue(finite)
        self.assertFalse(infinite)
        self.assertAlmostEqual(exact_ess_1d(1.5, M=200), 0.863, places=2)
        self.assertAlmostEqual(var_logw_1d(1.5, M=200), 0.0596, delta=0.001)

    def test_early_stopping_and_threshold(self):
        full = gaussian_path_mgf(2.25, 0.25, 0.8, 1.5, 1.0, M=120)
        stopped = gaussian_path_mgf(2.25, 0.25, 0.8, 1.5, 1.0, M=120, t_min=0.0)
        self.assertEqual(full, stopped)
        self.assertAlmostEqual(critical_w(2.25, 0.25, 0.8, M=150, tol=1e-3),
                               1.954, delta=0.03)


class SamplerTests(unittest.TestCase):
    def test_unit_guidance_is_uniform(self):
        result = smc_cfg(GMM_BED, 1.0, N=600, M=40, seed=0, snapshots=())
        self.assertEqual(result["ess"], 1.0)
        self.assertEqual(result["EW"], 1.0)
        np.testing.assert_allclose(result["wn"], 1.0 / 600)
        adaptive = smc_resampling_cfg(GMM_BED, 1.0, N=400, M=30, seed=3)
        self.assertAlmostEqual(adaptive["ess"], 1.0, places=12)
        self.assertEqual(adaptive["n_resample"], 0)

    def test_seeds_are_process_independent(self):
        self.assertEqual(stable_seed("sweep", "GMM", 1.5),
                         stable_seed("sweep", "GMM", 1.5))
        self.assertNotEqual(stable_seed("sweep", "GMM", 1.5),
                            stable_seed("sweep", "GMM", 2.0))

    def test_reference_grid_and_quantile(self):
        grid = ReferenceGrid(GAUSS_BED, 2.0, n=96)
        self.assertAlmostEqual(float(grid.target_mean[0]), 0.8470588, places=3)
        self.assertAlmostEqual(float(grid.target_mean[1]), 0.0, places=5)
        quantiles = weighted_quantile(np.array([0.0, 1.0]),
                                      np.array([0.5, 0.5]), np.array([0.25, 0.75]))
        np.testing.assert_allclose(quantiles, [0.0, 1.0])


class PaperDataTests(unittest.TestCase):
    def test_expected_files_and_rows(self):
        expected = ["early_stopping.csv", "high_guidance.csv",
                    "dimension_thresholds.csv", "gaussian_cfg_summary.csv",
                    "gmm_summary.csv", "gmm_exact_moments.csv",
                    "smc_comparison.csv", "shared_prior.csv", "mnist_pilot.csv"]
        for name in expected:
            self.assertTrue((ROOT / "data" / name).is_file(), name)
        gmm = np.genfromtxt(ROOT / "data" / "gmm_summary.csv", delimiter=",", names=True)
        np.testing.assert_allclose(gmm["w"], [0.5, 1.0, 1.5, 1.75, 2.0, 3.0])
        for w in (0.5, 1.5, 2.0):
            saved = np.load(ROOT / "data" / f"gmm_density_w{w:g}.npz")
            self.assertEqual(saved["x"].shape, (60_000, 2))
            self.assertAlmostEqual(float(saved["wn"].sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
