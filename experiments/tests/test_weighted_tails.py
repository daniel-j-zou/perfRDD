import unittest

import numpy as np
from scipy.stats import norm

from experiments.methods.weighted_tails import (
    density_influence,
    fit_weighted_tails,
    uj_score_terms,
    uj_utility,
)


class WeightedTailsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 200_000
        self.t = rng.standard_normal(n)
        self.x = np.column_stack((self.t, rng.standard_normal(n)))
        self.tails = fit_weighted_tails(self.t, self.x, (-4.0, 4.0), n_basis=24)

    def test_recovers_gaussian_density_and_tails(self):
        s = np.linspace(-2.0, 2.0, 9)
        np.testing.assert_allclose(self.tails.density(s), norm.pdf(s), atol=0.01)
        np.testing.assert_allclose(self.tails.survival(s), norm.sf(s), atol=0.01)
        # p_X(t) = (t phi(t), 0) and H_X(s) = (phi(s), 0) for this design.
        np.testing.assert_allclose(
            self.tails.weighted_density(s)[:, 0], s * norm.pdf(s), atol=0.01)
        np.testing.assert_allclose(
            self.tails.weighted_tail(s)[:, 0], norm.pdf(s), atol=0.01)
        np.testing.assert_allclose(self.tails.weighted_tail(s)[:, 1], 0.0, atol=0.01)
        beta = np.array([0.8, 0.25])
        np.testing.assert_allclose(
            self.tails.weighted_tail(s, beta), self.tails.weighted_tail(s) @ beta)

    def test_zero_extension_off_support(self):
        self.assertEqual(float(self.tails.survival(5.0)), 0.0)
        self.assertEqual(float(self.tails.density(-5.0)), 0.0)
        np.testing.assert_allclose(
            self.tails.survival(-5.0), self.tails.survival(-4.0))

    def test_density_influence_averages_to_the_score(self):
        rng = np.random.default_rng(4)
        eta = rng.standard_normal(3_000)
        keep = (np.abs(eta) < 1.28).astype(float)
        alpha = 0.1 + 0.9 * eta
        beta = np.array([0.8, 0.25])
        terms = uj_score_terms(-0.1, eta, keep, alpha, beta, self.tails)
        raw = density_influence(-0.1, eta, keep, alpha, beta, self.tails,
                                self.t, self.x, center=False)
        self.assertAlmostEqual(float(np.mean(raw)), float(np.mean(terms.score)), places=10)

    def test_utility_derivative_matches_score(self):
        rng = np.random.default_rng(5)
        eta = rng.standard_normal(2_000)
        keep = (np.abs(eta) < 1.28).astype(float)
        alpha = 0.1 + 0.9 * eta
        beta = np.array([0.8, 0.25])
        h = 1e-5
        numeric = (uj_utility(0.2 + h, eta, keep, alpha, beta, self.tails)
                   - uj_utility(0.2 - h, eta, keep, alpha, beta, self.tails)) / (2 * h)
        terms = uj_score_terms(0.2, eta, keep, alpha, beta, self.tails)
        self.assertAlmostEqual(numeric, float(np.mean(terms.score)), places=6)


if __name__ == "__main__":
    unittest.main()
