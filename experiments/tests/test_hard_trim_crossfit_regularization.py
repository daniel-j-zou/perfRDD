import unittest

import numpy as np

from experiments.scripts.hard_trim_crossfit_regularization import (
    estimator_labels,
    make_crossfit_folds,
    run_replication,
    summarize,
)


class HardTrimCrossfitRegularizationTest(unittest.TestCase):
    def test_crossfit_folds_cover_sample_once(self):
        folds = make_crossfit_folds(1001, 13, 5)
        joined = np.concatenate(folds)
        self.assertEqual(len(joined), 1001)
        np.testing.assert_array_equal(np.sort(joined), np.arange(1001))

    def test_single_replication_and_summary(self):
        ridge = (0.0, 0.01)
        row = run_replication(1000, 2, ridge, 5)
        for label in estimator_labels(ridge):
            self.assertTrue(np.isfinite(row[f"{label}_phi"]), label)
            self.assertGreater(row[f"{label}_retention"], 0.5)
            self.assertLess(row[f"{label}_retention"], 1.0)
        result = summarize([row], ridge)
        self.assertEqual(result["1000"]["replications"], 1)
        self.assertEqual(
            set(result["1000"]["estimators"]), set(estimator_labels(ridge))
        )
        self.assertEqual(sum(row["theory_fold_counts"].values()), 1000)
        self.assertEqual(
            set(row["theory_first_stage_diagnostics"]),
            {"gamma_alpha_error", "gamma_g_error", "gamma_U_error",
             "gamma_l_error", "gamma_u_error", "l_hat", "u_hat"},
        )

    def test_spline_density_replication(self):
        row = run_replication(1000, 3, (0.0,), 2, "spline")
        for label in estimator_labels((0.0,), 2):
            self.assertTrue(np.isfinite(row[f"{label}_phi"]), label)
            self.assertGreaterEqual(row[f"{label}_density_basis"], 8)


if __name__ == "__main__":
    unittest.main()
