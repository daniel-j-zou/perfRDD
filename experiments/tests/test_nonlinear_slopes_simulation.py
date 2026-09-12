import unittest

import numpy as np

from experiments.scripts.nonlinear_slopes_simulation import (
    DEFAULT_DGP,
    estimate_threshold,
    generate_sample,
    population_truth,
)


class NonlinearSlopesSimulationTest(unittest.TestCase):
    """Small deterministic checks for the nonlinear extension diagnostic."""

    def test_quadratic_term_changes_the_population_target(self):
        full = population_truth(DEFAULT_DGP, include_quadratic=True)
        restricted = population_truth(DEFAULT_DGP, include_quadratic=False)
        self.assertLess(full["curvature"], 0.0)
        self.assertLess(restricted["curvature"], 0.0)
        self.assertGreater(abs(full["phi_star"] - restricted["phi_star"]), 0.5)

    def test_all_models_return_finite_estimates(self):
        sample = generate_sample(800, seed=20260914, dgp=DEFAULT_DGP)
        for model in ("alpha_only", "linear_slopes", "quadratic_slopes"):
            estimate = estimate_threshold(sample, model, DEFAULT_DGP)
            for key in (
                "phi_hat",
                "curvature_hat",
                "score_variance_hat",
                "variance_constant_hat",
                "variance_hat",
                "condition_number",
            ):
                self.assertTrue(np.isfinite(estimate[key]), (model, key))
            self.assertLess(estimate["curvature_hat"], 0.0)
            self.assertGreaterEqual(estimate["score_variance_hat"], 0.0)
            self.assertGreater(estimate["variance_hat"], 0.0)


if __name__ == "__main__":
    unittest.main()
