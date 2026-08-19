import unittest

import numpy as np

from experiments.methods.perfrdd_smooth_trim import (
    _select_delta,
    _smooth_trim_weights,
    _symmetric_smooth_step,
)


class SmoothTrimGateTest(unittest.TestCase):
    def test_step_is_symmetric_and_flat_outside_transition(self):
        grid = np.linspace(-2.0, 2.0, 1001)
        h = _symmetric_smooth_step(grid)
        np.testing.assert_allclose(h + _symmetric_smooth_step(-grid), 1.0, atol=2e-15)
        self.assertTrue(np.all(h[grid <= -1.0] == 0.0))
        self.assertTrue(np.all(h[grid >= 1.0] == 1.0))
        self.assertTrue(np.all(np.diff(h) >= -1e-15))

    def test_gate_has_exact_core_and_compact_support(self):
        eta = np.array([-1.1, -1.0, -0.9, 0.0, 0.9, 1.0, 1.1])
        weights = _smooth_trim_weights(eta, -1.0, 1.0, 0.1)
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[-1], 0.0)
        self.assertEqual(weights[3], 1.0)
        self.assertTrue(np.all((weights >= 0.0) & (weights <= 1.0)))

    def test_default_delta_uses_a_separate_n_minus_one_third_rate(self):
        delta = _select_delta(-2.0, 3.0, 1000, None, 1.0, 1.0 / 3.0)
        self.assertAlmostEqual(delta, 0.5)

    def test_invalid_delta_is_rejected(self):
        with self.assertRaises(ValueError):
            _smooth_trim_weights(np.array([0.0]), -1.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            _select_delta(-1.0, 1.0, 100, 1.0, 1.0, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
