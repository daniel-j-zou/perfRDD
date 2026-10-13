import unittest

import numpy as np

from experiments.scripts.debiased_score_check import (
    debiased_one_step,
    generate,
    predictions,
    replicate,
    truth,
)


class DebiasedScoreTest(unittest.TestCase):
    def test_predictions_follow_the_fold_algebra(self):
        p = predictions()
        self.assertAlmostEqual(p["pi4_rot"], 43.675, delta=0.05)
        self.assertAlmostEqual(p["pi4_fixed"] / p["pi4_rot"], 4.0, places=10)
        # A half split costs a factor two for both estimators.
        self.assertAlmostEqual(p["dr2_one"], 2 * p["pi4_rot"], places=10)
        self.assertAlmostEqual(p["pi2_one"], 2 * p["pi4_rot"], places=10)
        # With a quarter evaluation fold the plug-in is better than the debiased score.
        self.assertLess(p["pi4f_one"], p["dr4f_one"])

    def test_one_step_is_finite_and_near_the_target(self):
        data = generate("baseline", 8000, 20261013)
        idx = np.random.default_rng(0).permutation(8000)
        phi_t, phi_dr, diag = debiased_one_step(data, idx[:4000], idx[4000:])
        target = truth("baseline")["phi_star"]
        self.assertTrue(np.isfinite(phi_t) and np.isfinite(phi_dr))
        self.assertLess(abs(phi_dr - target), 0.5)
        self.assertLess(diag["H_hat"], 0.0)
        # True first-stage loading is kappa * gamma_X = -0.0245 per coordinate.
        self.assertTrue(all(-0.15 < a < 0.1 for a in diag["A_hat"]))

    def test_replicate_returns_every_requested_estimator(self):
        out = replicate(("nonlinear", 6000, 7, ["pi4f_cf", "dr4f_cf"], [1, 3], 0, None))
        for key in ("pi4f_cf", "dr4f_cf_m1", "dr4f_cf_m3"):
            self.assertIn(key, out)
            self.assertTrue(np.isfinite(out[key]))


if __name__ == "__main__":
    unittest.main()
