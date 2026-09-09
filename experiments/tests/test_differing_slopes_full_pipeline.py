import unittest

import numpy as np

from experiments.scripts.differing_slopes_full_pipeline import (
    DEFAULT_DGP,
    SCENARIOS,
    _oracle_variance,
    _fit_variant,
    generate_sample,
    population_truth,
)


class DifferingSlopesFullPipelineTest(unittest.TestCase):
    def test_truth_scenarios_have_expected_curvature_status(self):
        baseline = population_truth(DEFAULT_DGP)
        nonlinear = population_truth(SCENARIOS["quadratic_misspecification"])
        boundary = population_truth(SCENARIOS["boundary_upper"])
        self.assertLess(baseline["curvature"], 0.0)
        self.assertLess(nonlinear["curvature"], 0.0)
        self.assertEqual(boundary["boundary"], 1.0)

    def test_generated_index_variants_are_finite(self):
        sample = generate_sample(600, 20260925, DEFAULT_DGP)
        for variant in ("full_gaussian_ols", "full_spline_ols", "crossfit5_gaussian"):
            phi, boundary, retained, _ = _fit_variant(
                sample, DEFAULT_DGP, 20260925, variant
            )
            self.assertTrue(np.isfinite(phi), variant)
            self.assertGreater(retained, 20.0, variant)
            self.assertIsInstance(boundary, bool)

    def test_cluster_variance_diagnostic_is_finite_and_larger(self):
        dgp = SCENARIOS["clustered"]
        sample = generate_sample(600, 20260925, dgp)
        phi, _, _, component = _fit_variant(sample, dgp, 20260925, "oracle")
        self.assertIsNotNone(component)
        iid, cluster = _oracle_variance(sample, dgp, component, phi)
        self.assertTrue(np.isfinite(iid))
        self.assertTrue(np.isfinite(cluster))
        self.assertGreater(cluster, iid)


if __name__ == "__main__":
    unittest.main()
