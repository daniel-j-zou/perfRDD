import unittest

import numpy as np

from experiments.scripts.debiased_score_check import debiased_one_step
from experiments.scripts.ds_flexible_first_stage_check import (
    fit_first_stage,
    fold_estimate,
    generate,
)


class FlexibleFirstStageTest(unittest.TestCase):
    def test_linear_first_stage_matches_existing_plug_in(self):
        # The 'spread' design is the Gaussian baseline, and the linear first
        # stage must reproduce the shared-training plug-in of the debiased
        # script exactly.
        data = generate("spread", 4000, 11)
        perm = np.random.default_rng(3).permutation(4000)
        evalf, train = perm[:1000], perm[1000:]
        phi_t, _, _ = debiased_one_step(data, train, evalf, rep_mult=0)
        self.assertAlmostEqual(fold_estimate(data, train, evalf, "linear", 0), phi_t,
                               places=10)

    def test_spline_families_contain_the_linear_truth(self):
        # Without noise, Q = T is linear in X, so every spline family recovers
        # it exactly on the basis support.
        data = generate("single", 3000, 5)
        data.Q[:] = data.T
        idx = np.arange(3000)
        inside = np.all(np.abs(data.X) < 4.4, axis=1)
        for kind in ("additive", "index"):
            pred = fit_first_stage(data, idx, kind, 8)(data.X)
            self.assertLess(np.max(np.abs(pred - data.T)[inside]), 1e-8, kind)


if __name__ == "__main__":
    unittest.main()
