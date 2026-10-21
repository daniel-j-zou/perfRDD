import unittest

import numpy as np

from experiments.methods.spline_density import (
    _basis_info,
    evaluate_basis_zero_outside,
    lebesgue_gram,
)
from experiments.methods.spline_size_cv import (
    density_lscv,
    density_size_grid,
    knot_grid,
    outcome_cv,
)


class SplineSizeCVTest(unittest.TestCase):
    def test_lscv_matches_brute_force_leave_one_out(self):
        rng = np.random.default_rng(0)
        n, support = 120, (-3.0, 3.0)
        x = rng.normal(size=(n, 2))
        t = x @ np.array([1.0, 0.5])
        scores = density_lscv(t, x, support, [8, 11])
        xt = np.column_stack((np.ones(n), x))
        for L, score in scores.items():
            info = _basis_info(L, support)
            G = lebesgue_gram(info)
            B = evaluate_basis_zero_outside(t, info)
            omega = np.linalg.solve(G, B.T @ xt / n)
            loo = []
            for i in range(n):
                keep = np.arange(n) != i
                om = np.linalg.solve(G, B[keep].T @ xt[keep] / (n - 1))
                loo.append(B[i] @ om @ xt[i])
            brute = float(np.trace(omega.T @ G @ omega)) - 2 * float(np.mean(loo))
            self.assertAlmostEqual(score, brute, places=10)

    def test_grids_stay_inside_the_rate_ranges(self):
        self.assertTrue(all(L >= 8 for L in density_size_grid(4000)))
        self.assertEqual(knot_grid(10), [3])
        with self.assertRaises(ValueError):
            density_size_grid(4000, exponents=(0.25,))
        with self.assertRaises(ValueError):
            knot_grid(4000, exponents=(0.14,))

    def test_outcome_cv_prefers_enough_knots(self):
        rng = np.random.default_rng(1)
        e = rng.uniform(-1, 1, 3000)
        y = np.sin(4 * e) + 0.1 * rng.normal(size=3000)

        def design(rows, k):
            info = _basis_info(k + 4, (-1.0, 1.0))
            return evaluate_basis_zero_outside(e[rows], info)

        scores = outcome_cv(3000, design, y, [1, 6], rng)
        self.assertLess(scores[6], scores[1])


if __name__ == "__main__":
    unittest.main()
