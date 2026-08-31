import unittest

import numpy as np

from experiments.scripts.hard_trim_gaussian_baseline import (
    L0,
    U0,
    generate_data,
    make_folds,
    population_truth,
    run_replication,
)


class GaussianHardTrimBaselineTest(unittest.TestCase):
    def test_population_truth(self):
        truth = population_truth()
        self.assertAlmostEqual(L0, -1.2815515655, places=9)
        self.assertAlmostEqual(U0, 1.2815515655, places=9)
        self.assertAlmostEqual(truth["hard_phi_star"], 0.7312916803, places=8)
        self.assertAlmostEqual(truth["untrimmed_phi_star"], 0.5, places=8)
        self.assertAlmostEqual(truth["hard_retention"], 0.8, places=12)
        self.assertAlmostEqual(truth["hard_treated_mass"], 0.4, places=12)
        self.assertLess(truth["hard_curvature"], 0.0)

    def test_folds_are_disjoint_and_cover_sample(self):
        folds = make_folds(1000, 7)
        joined = np.concatenate(list(folds.values()))
        self.assertEqual(len(joined), 1000)
        self.assertEqual(len(np.unique(joined)), 1000)
        np.testing.assert_array_equal(np.sort(joined), np.arange(1000))

    def test_generated_data_shapes(self):
        data = generate_data(800, 3)
        self.assertEqual(data.X.shape, (800, 3))
        self.assertEqual(data.Q.shape, (800,))
        self.assertTrue(set(np.unique(data.D)).issubset({0.0, 1.0}))

    def test_single_replication_is_finite(self):
        row = run_replication(1000, 0)
        for key in (
            "oracle_hard_phi",
            "feasible_hard_phi",
            "feasible_smooth_phi",
            "untrimmed_phi",
            "l_hat",
            "u_hat",
        ):
            self.assertTrue(np.isfinite(row[key]), key)
        self.assertLess(row["l_hat"], row["u_hat"])
        self.assertGreater(row["hard_retention"], 0.4)
        self.assertLess(row["hard_retention"], 1.0)


if __name__ == "__main__":
    unittest.main()
