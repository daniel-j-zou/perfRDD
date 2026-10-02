import unittest

import numpy as np

from experiments.scripts.four_role_shared_gamma_check import ROLES, replicate


class FourRoleSharedGammaTest(unittest.TestCase):
    def test_replicate_returns_fixed_and_rotated_thresholds(self):
        fixed, rotated = replicate((8000, 20261002))
        self.assertEqual(len(ROLES), 4)
        self.assertTrue(np.isfinite(fixed) and np.isfinite(rotated))
        # Population optimum is 0.7313; both estimates should be near it.
        self.assertLess(abs(fixed - 0.7313), 0.6)
        self.assertLess(abs(rotated - 0.7313), 0.3)


if __name__ == "__main__":
    unittest.main()
