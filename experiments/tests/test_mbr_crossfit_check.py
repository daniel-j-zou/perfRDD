import unittest

import numpy as np

from experiments.scripts.mbr_crossfit_check import (
    ALPHA0,
    estimate,
    generate,
    theoretical_variance,
)


class MBRCrossfitCheckTest(unittest.TestCase):
    def test_theoretical_variance_values(self):
        self.assertAlmostEqual(theoretical_variance(1.0, "paper")["V"], 3.9475, places=3)
        self.assertAlmostEqual(theoretical_variance(1.0, "excluded")["V"], 3.0772, places=3)
        # Less truncation retains more observations and lowers the variance.
        self.assertGreater(theoretical_variance(0.8, "paper")["V"],
                           theoretical_variance(1.0, "paper")["V"])

    def test_stacked_and_plug_in_estimates_are_close_to_truth(self):
        rng = np.random.default_rng(5)
        for design in ("paper", "excluded"):
            data = generate(6000, rng, design)
            folds = np.array_split(rng.permutation(6000), 3)
            for plug_in in (False, True):
                alpha, dgamma = estimate(data, folds[0], folds[1], folds[2], 6, 1.0,
                                         plug_in=plug_in)
                self.assertLess(abs(alpha - ALPHA0), 0.4, (design, plug_in))
                self.assertEqual(dgamma.shape, (2,))


if __name__ == "__main__":
    unittest.main()
