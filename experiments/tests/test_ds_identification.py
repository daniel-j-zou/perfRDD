"""Checks for the identification and curvature identities (this_week.tex, 2026-10-13)."""
import unittest

import numpy as np

from experiments.scripts import ds_identification_check as m


class TestDerivativeForms(unittest.TestCase):
    def test_forms_match_finite_difference(self):
        for d in m.designs():
            for phi in (-1.5, 0.2, 1.7):
                forms = m.dtau_forms(d, phi)
                num = m.fd(lambda y: m.tau(d, y), phi)
                self.assertAlmostEqual(forms["form_A"], num, places=6)
                self.assertAlmostEqual(forms["form_B"], num, places=6)
                if "gaussian_formula" in forms:
                    self.assertAlmostEqual(forms["gaussian_formula"], num, places=6)

    def test_aligned_designs_have_positive_slope(self):
        for d in m.designs()[:3]:
            for phi in np.linspace(d.phi_lo, d.phi_hi, 13):
                forms = m.dtau_forms(d, phi)
                self.assertGreater(forms["C_a"], 0)
                self.assertGreaterEqual(forms["C_b"], 0)
                self.assertGreaterEqual(forms["B_eps"], 0)


class TestTargets(unittest.TestCase):
    def test_four_role_design_target(self):
        from scipy import optimize
        d = m.designs()[0]
        star = optimize.brentq(lambda y: m.tau(d, y), -1, 1, xtol=1e-12)
        self.assertAlmostEqual(star, -0.1197699, places=6)

    def test_displayed_design_target(self):
        from scipy import optimize
        d = m.designs()[1]
        star = optimize.brentq(lambda y: m.tau(d, y), -1, 2, xtol=1e-12)
        self.assertAlmostEqual(star, 0.5451028, places=6)

    def test_bimodal_counterexample_is_not_monotone(self):
        out = m.bimodal_counterexample()
        self.assertLess(out["dtau_at_0"]["form_B"], 0)
        self.assertEqual(len(out["zeros"]), 3)


if __name__ == "__main__":
    unittest.main()
