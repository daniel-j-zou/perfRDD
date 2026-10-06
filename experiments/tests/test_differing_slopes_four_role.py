import unittest

import numpy as np

from experiments.scripts.differing_slopes_four_role_check import ROLES, replicate
from experiments.scripts.differing_slopes_full_pipeline import DEFAULT_DGP, population_truth


class DifferingSlopesFourRoleTest(unittest.TestCase):
    def test_replicate_returns_three_thresholds_near_the_optimum(self):
        target = population_truth(DEFAULT_DGP)["phi_star"]
        fixed, rotated, full = replicate((4000, 20261005))
        self.assertEqual(len(ROLES), 4)
        self.assertTrue(np.all(np.isfinite([fixed, rotated, full])))
        self.assertLess(abs(fixed - target), 0.5)
        self.assertLess(abs(rotated - target), 0.25)
        self.assertLess(abs(full - target), 0.25)


if __name__ == "__main__":
    unittest.main()
