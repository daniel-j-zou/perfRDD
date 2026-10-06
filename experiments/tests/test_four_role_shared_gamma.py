import unittest

import numpy as np

from experiments.scripts.four_role_shared_gamma_check import (
    DESIGNS,
    ROLES,
    predicted_variances,
    replicate,
)


class FourRoleSharedGammaTest(unittest.TestCase):
    def test_replicate_returns_fixed_and_rotated_thresholds(self):
        fixed, rotated = replicate((8000, 20261002, "four"))
        self.assertEqual(len(ROLES), 4)
        self.assertTrue(np.isfinite(fixed) and np.isfinite(rotated))
        # Population optimum is 0.7313; both estimates should be near it.
        self.assertLess(abs(fixed - 0.7313), 0.6)
        self.assertLess(abs(rotated - 0.7313), 0.3)

    def test_five_role_design_runs_and_is_not_an_exact_multiple(self):
        fixed, rotated = replicate((8000, 20261002, "five"))
        self.assertEqual(len(DESIGNS["five"]), 5)
        self.assertTrue(np.isfinite(fixed) and np.isfinite(rotated))
        four = predicted_variances("four")
        five = predicted_variances("five")
        self.assertAlmostEqual(four["fixed"] / four["rotated"], 4.0, places=10)
        self.assertAlmostEqual(five["rotated"], four["rotated"], places=10)
        self.assertGreater(five["fixed"], four["fixed"])
        self.assertLess(five["fixed"] / five["rotated"], 5.0)


if __name__ == "__main__":
    unittest.main()
