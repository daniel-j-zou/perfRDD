"""Checks for the score-redesign exploration helpers."""
import unittest

import numpy as np

from experiments.scripts.score_redesign_explore import _surrogate, as_cutoff_rule, best_threshold


class ScoreRedesignHelpers(unittest.TestCase):
    def test_best_threshold_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            score = rng.integers(0, 15, size=60).astype(float)     # many ties
            gain = rng.standard_normal(60)
            t, value = best_threshold(score, gain)
            candidates = [np.inf] + sorted(set(score))
            brute = max(np.mean(gain * (score >= c)) for c in candidates)
            self.assertAlmostEqual(value, brute, places=12)
            self.assertAlmostEqual(np.mean(gain * (score >= t)), value, places=12)

    def test_cutoff_rule_reproduces_assignments(self):
        rng = np.random.default_rng(2)
        n, p = 2000, 3
        X = rng.standard_normal((n, p))
        gamma = np.r_[0.4, rng.standard_normal(p)]
        eta = 1.7 * rng.standard_normal(n)
        sd_eta = eta.std()
        for sign in (1.0, -1.0):
            Qs = gamma[0] + X @ gamma[1:] + eta          # mirrored score; Q = sign * Qs
            Q = sign * Qs
            w = np.r_[0.6, rng.standard_normal(p)]
            w /= np.linalg.norm(w)
            phi = 0.3
            treat = np.column_stack((eta / sd_eta, X)) @ w >= phi
            scale = rng.uniform(0.5, 2.0, p)
            rule = as_cutoff_rule(w, phi, sd_eta, gamma, sign, scale, ["a", "b", "c"])
            cut = rule["cutoff_at_mean_covariates"] + X @ np.array(list(rule["cutoff_shift_per_sd"].values()))
            treat_q = Q >= cut if sign > 0 else Q <= cut
            agree = np.mean(treat == treat_q)
            self.assertGreater(agree, 0.999)

    def test_surrogate_gradient(self):
        rng = np.random.default_rng(3)
        Z = rng.standard_normal((500, 4))
        gain = rng.standard_normal(500)
        x = np.r_[rng.standard_normal(4), 0.2]
        f, g = _surrogate(x, Z, gain, 0.3)
        h = 1e-6
        num = np.array([(_surrogate(x + h * e, Z, gain, 0.3)[0] - _surrogate(x - h * e, Z, gain, 0.3)[0]) / (2 * h)
                        for e in np.eye(5)])
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
