import unittest

import numpy as np

from experiments.scripts.four_role_rotation_average_check import replicate


class FourRoleRotationAverageTest(unittest.TestCase):
    def test_replicate_returns_four_rotations_and_the_pooled_estimate(self):
        draws = np.asarray(replicate((8000, 777_000)))
        self.assertEqual(draws.shape, (5,))
        self.assertTrue(np.all(np.isfinite(draws)))
        # The pooled-criterion estimate is close to the average of the rotations.
        self.assertLess(abs(draws[:4].mean() - draws[4]), 0.05)


if __name__ == "__main__":
    unittest.main()
