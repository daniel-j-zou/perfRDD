import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd_hard_trim import (
    _make_folds,
    perfrdd_hard_trim,
)


def _sample(n: int = 1200, seed: int = 4) -> RDDSample:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    gamma = np.ones(3) / np.sqrt(3.0)
    eta = rng.standard_normal(n)
    Q = X @ gamma + eta
    D = (Q > 0.0).astype(int)
    alpha = 2.0 + eta
    Y = D * alpha + 0.5 * eta ** 2 + X @ np.array([0.3, -0.2, 0.1])
    Y += rng.normal(0.0, 0.5, n)
    return RDDSample(
        Q=Q,
        X=X,
        Y=Y,
        threshold=0.0,
        name="hard_trim_test",
        feature_names=["x1", "x2", "x3"],
    )


class HardTrimMethodTest(unittest.TestCase):
    def test_crossfit_folds_partition_rows(self):
        folds = _make_folds(1001, 5, 17)
        joined = np.concatenate(folds)
        np.testing.assert_array_equal(np.sort(joined), np.arange(1001))

    def test_full_sample_and_crossfit_are_finite(self):
        sample = _sample()
        grid = np.linspace(-1.5, 1.5, 101)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = perfrdd_hard_trim(
                sample,
                root / "full",
                (-1.75, 1.75),
                c_values=(2.25,),
                phi_grid=grid,
            )
            crossfit = perfrdd_hard_trim(
                sample,
                root / "crossfit",
                (-1.75, 1.75),
                c_values=(2.25,),
                phi_grid=grid,
                crossfit_folds=5,
            )
        for result in (full, crossfit):
            self.assertTrue(np.isfinite(result["phi_star"]["2.25"]))
            self.assertGreater(result["hard_retention"], 0.6)
            self.assertLess(result["hard_retention"], 0.95)
            self.assertEqual(result["estimand"], "exact_hard_support_trimmed")
            self.assertEqual(
                set(result["phi_star"]), set(result["phi_star_near_grid_boundary"])
            )
            for fold in result["fold_diagnostics"]:
                self.assertTrue(np.isfinite(fold["design_condition_number"]))
                self.assertLessEqual(fold["design_rank"], fold["design_columns"])
                self.assertGreater(fold["lower_support_margin"], 0.0)
                self.assertGreater(fold["upper_support_margin"], 0.0)

    def test_support_must_contain_estimated_trim_window(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "strictly inside"):
                perfrdd_hard_trim(
                    _sample(),
                    Path(directory),
                    (-0.5, 0.5),
                    c_values=(2.25,),
                    phi_grid=np.linspace(-1.5, 1.5, 51),
                )

    def test_can_return_curves_without_writing_outputs(self):
        grid = np.linspace(-1.5, 1.5, 31)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            result = perfrdd_hard_trim(
                _sample(),
                output,
                (-1.75, 1.75),
                c_values=(0.0,),
                phi_grid=grid,
                write_outputs=False,
                return_curves=True,
            )
            self.assertFalse(output.exists())
        np.testing.assert_allclose(result["returned_phi_grid"], grid)
        self.assertEqual(len(result["returned_utility_curves"]["0.0"]), len(grid))


if __name__ == "__main__":
    unittest.main()
