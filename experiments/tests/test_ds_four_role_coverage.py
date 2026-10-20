"""Checks for the four-role coverage runner and the trim-window quadrature targets."""
import unittest

import numpy as np

from experiments.methods.trimmed_quadrature import trimmed_normal_mean
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts import ds_four_role_coverage as m
from experiments.scripts.ds_bias_decomposition import TRUE_TAILS


class TestTargets(unittest.TestCase):
    def test_coverage_design_targets(self):
        disp = m.population_summary(m.COVERAGE_DESIGNS["displayed"])
        self.assertAlmostEqual(disp["phi_star"], 0.545103, places=6)
        self.assertAlmostEqual(disp["V_rotated"], 31.09, places=2)
        nl = m.population_summary(m.COVERAGE_DESIGNS["nonlinear"])
        self.assertAlmostEqual(nl["phi_star"], 0.152143, places=6)

    def test_old_script_targets_use_the_trim_window(self):
        from experiments.scripts import differing_slopes_full_pipeline as fp
        from experiments.scripts import differing_slopes_nonlinear_outcome as nlo
        from experiments.scripts import differing_slopes_simulation as sim
        self.assertAlmostEqual(sim.population_truth(sim.DEFAULT_DGP)["phi_star"], -0.1197699, places=6)
        self.assertAlmostEqual(fp.population_truth(fp.DEFAULT_DGP)["phi_star"], -0.1197699, places=6)
        self.assertAlmostEqual(
            fp.population_truth(fp.SCENARIOS["quadratic_misspecification"])["phi_star"], 0.484416, places=5)
        self.assertAlmostEqual(nlo.population_truth(nlo.SCENARIOS["nonlinear"]), 0.152143, places=6)

    def test_trimmed_mean_matches_gauss_legendre(self):
        x, w = np.polynomial.legendre.leggauss(200)
        lo, hi = -1.2816, 1.2816
        e, we = 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w
        fun = lambda v: np.cos(v) + v ** 3
        gl = float(np.sum(we * fun(e) * np.exp(-0.5 * e ** 2) / np.sqrt(2 * np.pi)))
        self.assertAlmostEqual(trimmed_normal_mean(fun, lo, hi), gl, places=12)


class TestCriterion(unittest.TestCase):
    def test_true_tails_match_bias_decomposition(self):
        tails = m.TrueTails(m.COVERAGE_DESIGNS["displayed"])
        t = np.linspace(-3.5, 3.5, 41)
        b2 = np.array(m.COVERAGE_DESIGNS["displayed"].beta2)
        np.testing.assert_allclose(tails.survival(t), TRUE_TAILS.survival(t), atol=1e-14)
        np.testing.assert_allclose(tails.weighted_tail(t, b2), TRUE_TAILS.weighted_tail(t, b2), atol=1e-14)

    def test_shared_spline_pieces_equal_weighted_tails_criterion(self):
        d = m.COVERAGE_DESIGNS["nonlinear"]
        s = m.generate(d, 4000, 11)
        T = m._xt(s.X) @ np.r_[d.gamma0, d.gamma_x]
        tails = fit_weighted_tails(T[:1000], s.X[:1000], d.T_support)
        e = np.random.default_rng(3).normal(size=800)
        amc, b2 = 0.2 + 0.5 * e, np.array([0.7, -0.1])
        old = m._criterion(e, amc, b2, tails, 1000)
        new = m.SplineTailPieces(tails, e, 1000)
        for phi in np.linspace(-4, 4, 33):
            self.assertAlmostEqual(old(phi), new.value(phi, amc, b2), places=13)

    def test_linear_fit_recovers_linear_effect(self):
        from dataclasses import replace
        d = replace(m.COVERAGE_DESIGNS["nonlinear"], a_coef=(0.35, 0.9), b_coef=(0.2, 0.6))
        s = m.generate(d, 40000, 5)
        eta = s.Q - m._xt(s.X) @ np.r_[d.gamma0, d.gamma_x]
        fit = m.outcome_fit_linear(s, np.arange(len(s.Q)), eta, d.J)
        np.testing.assert_allclose(fit.beta2, d.beta2, atol=0.03)
        np.testing.assert_allclose(fit.a(np.array([-1.0, 1.0])), [-0.55, 1.25], atol=0.03)


class TestReplicate(unittest.TestCase):
    def test_replicate_returns_all_arms(self):
        for key in ("displayed", "nonlinear"):
            d = m.COVERAGE_DESIGNS[key]
            pop = m.population_summary(d)
            row = m.replicate((key, 4000, 0, 20261019, 3, pop))
            arms = {"avg", "pooled", "fixed", "oracle_avg", "oracle_pooled"}
            if m.LINEAR_ARM[key]:
                arms |= {"linear_avg", "linear_pooled"}
            self.assertEqual(set(row["est"]), arms)
            self.assertEqual(row["failures"], 0)
            self.assertTrue(all(len(v) == 3 for v in row["boot"].values()))
            self.assertLess(abs(row["est"]["avg"] - pop["phi_star"]), 0.5)
            self.assertLess(abs(row["est"]["oracle_avg"] - pop["phi_star"]), 0.1)
            summary = m.summarize([row, m.replicate((key, 4000, 1, 20261020, 3, pop))], 4000, pop)
            self.assertIn("coverage_percentile", summary["arms"]["avg"])
            table = m.report({"design": key, "population": pop, "cells": {"4000": summary}})
            self.assertIn("| 4,000 | rotated average |", table)

    def test_within_block_scheme_keeps_the_point_estimate(self):
        pop = m.population_summary(m.COVERAGE_DESIGNS["displayed"])
        task = ("displayed", 2000, 0, 7, 2, pop)
        resplit, within = m.replicate(task), m.replicate(task + ("within",))
        self.assertEqual(resplit["est"], within["est"])
        self.assertNotEqual(resplit["boot"]["avg"], within["boot"]["avg"])


if __name__ == "__main__":
    unittest.main()
