import unittest

import numpy as np

from experiments.scripts.eight_role_variance_decomposition import ROLES, decomposition


class EightRoleVarianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = decomposition()

    def test_location_invariance_cancels_intercept_loadings(self):
        q = self.result["loadings"]
        total = -q["A_alpha_0"] - q["A_g_0"] + q["a_U"] - q["b_l"] + q["b_u"]
        self.assertAlmostEqual(total, 0.0, places=8)
        # a_U = -U''(phi*) and A_g0 = U''(phi*).
        H = self.result["truth"]["hard_curvature"]
        self.assertAlmostEqual(q["a_U"], -H, places=8)
        self.assertAlmostEqual(q["A_g_0"], H, places=8)

    def test_influence_functions_are_centered(self):
        for name, mean in self.result["influence_means"].items():
            self.assertAlmostEqual(mean, 0.0, places=7, msg=name)

    def test_variance_totals(self):
        cov = np.asarray(self.result["score_covariance"])
        self.assertEqual(cov.shape, (len(ROLES), len(ROLES)))
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), -1e-10)
        tv = self.result["threshold_variance"]
        self.assertAlmostEqual(tv["fixed_eight_block"], 463.96, delta=0.05)
        self.assertAlmostEqual(tv["rotated_or_full_sample"], 43.675, delta=0.01)
        self.assertAlmostEqual(tv["ratio"], 10.623, delta=0.005)


if __name__ == "__main__":
    unittest.main()
