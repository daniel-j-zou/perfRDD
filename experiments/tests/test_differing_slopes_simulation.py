import unittest

import numpy as np

from experiments.scripts.differing_slopes_simulation import (
    DEFAULT_DGP,
    SCENARIOS,
    estimate_threshold,
    generate_sample,
    population_truth,
)


class DifferingSlopesSimulationTest(unittest.TestCase):
    """Small deterministic checks for the differing-slopes diagnostic."""

    def test_interaction_changes_the_population_policy_target(self):
        full = population_truth(DEFAULT_DGP, include_beta2=True)
        restricted = population_truth(DEFAULT_DGP, include_beta2=False)
        self.assertLess(full["curvature"], 0.0)
        self.assertLess(restricted["curvature"], 0.0)
        self.assertGreater(abs(full["phi_star"] - restricted["phi_star"]), 0.1)

    def test_estimate_and_variance_are_finite(self):
        sample = generate_sample(800, seed=20260908, dgp=DEFAULT_DGP)
        estimate = estimate_threshold(sample, DEFAULT_DGP, include_beta2=True)
        for key in (
            "phi_hat",
            "curvature_hat",
            "score_variance_hat",
            "variance_constant_hat",
            "variance_hat",
            "condition_number",
        ):
            self.assertTrue(np.isfinite(estimate[key]), key)
        self.assertLess(estimate["curvature_hat"], 0.0)
        self.assertGreaterEqual(estimate["score_variance_hat"], 0.0)
        self.assertGreater(estimate["variance_hat"], 0.0)

    def test_error_law_scenarios_preserve_sample_contract(self):
        for name, dgp in SCENARIOS.items():
            sample = generate_sample(120, seed=20260908, dgp=dgp)
            self.assertEqual(sample["X"].shape, (120, 2), name)
            self.assertEqual(sample["Y"].shape, (120,), name)
            self.assertTrue(np.isfinite(sample["Y"]).all(), name)


if __name__ == "__main__":
    unittest.main()
