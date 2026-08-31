import unittest

import numpy as np
from scipy.stats import norm

from experiments.methods.spline_density import (
    fit_spline_density,
    spline_basis_dimension,
)


class SplineDensityTest(unittest.TestCase):
    def test_dimension_obeys_requested_rate_rule(self):
        self.assertEqual(spline_basis_dimension(1_000), 8)
        self.assertGreater(spline_basis_dimension(100_000), 8)
        with self.assertRaises(ValueError):
            spline_basis_dimension(1_000, exponent=0.25)

    def test_projection_recovers_standard_normal_density_and_survival(self):
        rng = np.random.default_rng(4)
        fit = fit_spline_density(
            rng.standard_normal(100_000), (-3.5, 3.5), n_basis=16
        )
        grid = np.linspace(-2.5, 2.5, 101)
        density_error = np.max(np.abs(fit.density(grid) - norm.pdf(grid)))
        survival_error = np.max(np.abs(fit.survival(grid) - norm.sf(grid)))
        self.assertLess(density_error, 0.015)
        self.assertLess(survival_error, 0.006)
        self.assertGreater(fit.support_fraction, 0.999)

    def test_density_is_zero_extended_off_support(self):
        fit = fit_spline_density(
            np.linspace(-2.0, 2.0, 1_000), (-3.0, 3.0), n_basis=10
        )
        np.testing.assert_array_equal(fit.density(np.array([-4.0, 4.0])), 0.0)
        self.assertAlmostEqual(float(fit.survival(np.array([4.0]))[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
