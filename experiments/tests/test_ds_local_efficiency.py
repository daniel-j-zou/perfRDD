"""Checks on the population side of the local-efficiency calculation."""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from experiments.scripts.ds_local_efficiency import DESIGNS, Population
from experiments.scripts.ds_local_efficiency_hetero import hetero_variances
from experiments.scripts.ds_local_efficiency_weight_diag import _gamma_qmle


class LocalEfficiencyPopulationTest(unittest.TestCase):

    def test_paper_design_target(self):
        res = Population(DESIGNS["paper_baseline"]).decomposition()
        self.assertAlmostEqual(res["phi_star"], 0.545103, places=5)
        self.assertAlmostEqual(res["H"], -0.1075, places=3)

    def test_intercept_loading_vanishes(self):
        # location invariance, for Gaussian and non-Gaussian eta
        for name in ("paper_baseline", "paper_t5", "paper_mix", "ds_linear"):
            res = Population(DESIGNS[name]).decomposition()
            self.assertLess(abs(res["Abar_intercept"]), 1e-12, name)

    def test_common_slope_values_reproduced(self):
        # beta2 = 0 reproduces the 2026-10-01 eight-role decomposition
        res = Population(replace(DESIGNS["paper_baseline"], beta2=(0.0, 0.0, 0.0))).decomposition()
        self.assertAlmostEqual(res["phi_star"], 0.731292, places=5)
        self.assertAlmostEqual(res["H"], -0.0720, places=4)
        self.assertAlmostEqual(res["b_l"], -0.0141, places=4)
        self.assertAlmostEqual(res["b_u"], 0.0621, places=4)
        self.assertAlmostEqual(res["variances"]["psi_U"], 0.0185, places=4)
        self.assertAlmostEqual(res["variances"]["psi_T"], 0.0363, places=4)

    def test_fisher_information(self):
        self.assertAlmostEqual(Population(DESIGNS["paper_baseline"]).I_eta, 1.0, places=6)
        self.assertAlmostEqual(Population(DESIGNS["paper_t5"]).I_eta, 1.25, places=4)

    def test_bound_ordering(self):
        for name in ("paper_baseline", "paper_t5", "paper_mix", "ds_linear"):
            res = Population(DESIGNS[name]).decomposition()
            self.assertLessEqual(res["V_rotated_eff"], res["V_rotated_stack_ls"] + 1e-9)
            self.assertLessEqual(res["V_rotated_stack_ls"], res["V_rotated_ols"] + 1e-9)

    def test_linear_design_has_no_gap(self):
        res = Population(DESIGNS["ds_linear"]).decomposition()
        self.assertTrue(np.allclose(res["J_Y"], 0.0, atol=1e-10))
        self.assertAlmostEqual(res["efficiency_gain"], 0.0, places=8)

    def test_hetero_reduces_to_homoskedastic(self):
        pop = Population(DESIGNS["paper_baseline"])
        res = pop.decomposition()
        het = hetero_variances(pop, 0.0)
        self.assertAlmostEqual(het["paper"], res["V_rotated_ols"], places=6)
        self.assertAlmostEqual(het["stacked"], res["V_rotated_eff"], places=6)
        self.assertAlmostEqual(het["bound"], res["V_rotated_eff"], places=6)
        self.assertAlmostEqual(het["weighted_ols"], res["V_rotated_ols"], places=6)

    def test_bound_below_every_estimator_under_heteroskedasticity(self):
        pop = Population(DESIGNS["paper_baseline"])
        for kappa in (0.5, 1.0, 1.5):
            het = hetero_variances(pop, kappa)
            for key in ("paper", "stacked", "weighted_ols"):
                self.assertLessEqual(het["bound"], het[key] + 1e-9, (kappa, key))

    def test_heteroskedastic_design_guard(self):
        with self.assertRaises(ValueError):
            Population(DESIGNS["paper_hetero"])

    def test_gamma_qmle_recovers_log_quadratic_variance(self):
        rng = np.random.default_rng(3)
        e = rng.uniform(-1.75, 1.75, 20000)
        B = np.column_stack((np.ones_like(e), e, e ** 2))
        theta = np.array([np.log(0.25), 1.0, 0.0])
        y = np.exp(B @ theta) * rng.standard_normal(len(e)) ** 2
        est = _gamma_qmle(B, y)
        self.assertTrue(np.allclose(est, theta, atol=0.05), est)


if __name__ == "__main__":
    unittest.main()
