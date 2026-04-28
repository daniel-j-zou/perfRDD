"""Unit tests for the RDDSample contract."""
from __future__ import annotations

import unittest

import numpy as np

from experiments._core.sample import RDDSample


def _toy_single(n: int = 50, threshold: float = 0.0):
    rng = np.random.default_rng(0)
    Q = rng.normal(size=n)
    X = rng.normal(size=(n, 3))
    Y = X.sum(axis=1) + (Q > threshold).astype(float) + rng.normal(size=n)
    return RDDSample(
        Q=Q, X=X, Y=Y,
        threshold=threshold,
        name="toy",
        feature_names=["x1", "x2", "x3"],
    )


def _toy_dual(n: int = 80, thr=(1.0, 2.0)):
    rng = np.random.default_rng(1)
    Q = rng.normal(loc=[1.0, 2.0], scale=1.0, size=(n, 2))
    X = rng.normal(size=(n, 2))
    Y = rng.normal(size=n)
    return RDDSample(
        Q=Q, X=X, Y=Y,
        threshold=thr,
        name="toy_dual",
        feature_names=["x1", "x2"],
    )


class TestRDDSampleSingle(unittest.TestCase):
    def test_shape_props(self):
        s = _toy_single()
        self.assertEqual(s.n, 50)
        self.assertEqual(s.p, 3)
        self.assertEqual(s.k, 1)

    def test_default_treatment_rule(self):
        s = _toy_single(threshold=0.0)
        np.testing.assert_array_equal(s.D, (s.Q > 0.0).astype(int))

    def test_custom_treatment_rule(self):
        s = _toy_single(threshold=0.0)
        # Override: treat if Q < threshold instead.
        s.treatment_rule = lambda q: (q < 0).astype(int)
        np.testing.assert_array_equal(s.D, (s.Q < 0.0).astype(int))

    def test_summary_keys(self):
        s = _toy_single()
        keys = set(s.summary().keys())
        self.assertEqual(keys, {"name", "n", "p", "k", "threshold", "n_treated", "n_control"})


class TestRDDSampleDual(unittest.TestCase):
    def test_shape(self):
        s = _toy_dual()
        self.assertEqual(s.k, 2)
        self.assertEqual(s.D.shape, (s.n,))

    def test_and_rule(self):
        s = _toy_dual(thr=(1.0, 2.0))
        expected = ((s.Q[:, 0] > 1.0) & (s.Q[:, 1] > 2.0)).astype(int)
        np.testing.assert_array_equal(s.D, expected)

    def test_threshold_length_mismatch_raises(self):
        # Q has 2 columns; threshold tuple of length 1 should fail at .D
        rng = np.random.default_rng(2)
        Q = rng.normal(size=(10, 2))
        X = rng.normal(size=(10, 1))
        Y = rng.normal(size=10)
        s = RDDSample(
            Q=Q, X=X, Y=Y, threshold=(0.0,),
            name="bad", feature_names=["x"],
        )
        with self.assertRaises(ValueError):
            _ = s.D


class TestRDDSampleValidation(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            RDDSample(
                Q=np.zeros(10), X=np.zeros((9, 1)), Y=np.zeros(10),
                threshold=0.0, name="bad", feature_names=["x"],
            )

    def test_feature_names_mismatch_raises(self):
        with self.assertRaises(ValueError):
            RDDSample(
                Q=np.zeros(10), X=np.zeros((10, 2)), Y=np.zeros(10),
                threshold=0.0, name="bad", feature_names=["x"],
            )


if __name__ == "__main__":
    unittest.main()
