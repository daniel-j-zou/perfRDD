"""Checks of the proposed proof simplifications (ds_simplification_checks.py, 2026-10-13)."""
import unittest

import numpy as np

from experiments.scripts import ds_simplification_checks as m


class TestGenericPopulation(unittest.TestCase):
    def test_matches_existing_module(self):
        v = m.check_validate()
        self.assertAlmostEqual(v["phi"][0], v["phi"][1], places=10)
        for key in ("A_dens_minus_Ag_Arho", "A_U_diff", "A_out_diff", "Abar_diff"):
            self.assertLess(v[key], 1e-12)

    def test_centred_and_raw_mixture_designs_agree(self):
        c = m.check_item4_center()
        self.assertAlmostEqual(c["phi_raw"], c["phi_centered"], places=10)
        self.assertLess(c["Abar_X_max_diff"], 1e-10)
        self.assertLess(abs(c["Abar_raw"][0]), 1e-10)
        self.assertLess(max(abs(x) for x in c["A_U_X_centered"]), 1e-10)


class TestConditioningRoute(unittest.TestCase):
    def test_population_maps(self):
        for design in (m.PAPER, m.MIXX, m.MIXX_RAW):
            r = m.check_item1_maps(m.Pop(design))
            self.assertLess(r["index_map_max_diff"], 1e-8)
            self.assertLess(r["eval_map_max_diff"], 1e-8)

    def test_finite_sieve_derivative(self):
        r = m.check_item1_sieve(m.Pop(m.MIXX), Ls=(12, 24))
        for c in r["cases"]:
            self.assertLess(c["fd_vs_population"], 1e-8)
            self.assertLess(c["fd_minus_taylor_minus_boundary"], 1e-8)
            self.assertLess(c["second_order_ratio"], 1.0)
            if c["basis"] == "full":
                self.assertGreater(c["boundary_term"], 1e-9)


class TestOutcomeBlock(unittest.TestCase):
    def test_stacked_equals_armwise(self):
        r = m.check_item3_ls()
        self.assertLess(r["max_fitted_diff"], 1e-9)
        self.assertEqual(r["stacked_rank"], r["stacked_cols"])

    def test_riesz_identity(self):
        r = m.check_item3_riesz(m.Pop(m.MIXX), n=400_000, n_dirs=4)
        self.assertLess(r["max_abs_z"], 4.0)
        self.assertTrue(r["r_zero_off_J"])


class TestInvariance(unittest.TestCase):
    def test_shift_of_X(self):
        r = m.check_item4_shift(n=8000)
        for row in r["rows"]:
            if row["case"] == "shared basis":
                self.assertLess(row["max_criterion_diff"], 1e-11)
                self.assertLess(row["phi_diff_over_sd"], 1e-4)


class TestPathwiseDerivative(unittest.TestCase):
    def test_law_of_X_direction(self):
        r = m.check_item_c_path()
        self.assertLess(r["abs_diff"], 1e-8)


if __name__ == "__main__":
    unittest.main()
