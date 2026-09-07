"""Fast tests for the pre-Monte-Carlo robustness smoke suite."""

import unittest

import numpy as np

from experiments.scripts.hard_trim_robustness_smoke import (
    density_smoke,
    estimate_once,
    make_sample,
    population_truth,
    run_smoke,
    short_bootstrap,
)


class HardTrimRobustnessSmokeTest(unittest.TestCase):
    def test_nongaussian_running_variables_are_finite(self):
        for law in ("t5", "skewed"):
            sample = make_sample(450, 4, running_law=law)
            full = estimate_once(sample)
            crossfit = estimate_once(sample, crossfit_folds=3)
            density = density_smoke(sample, (-8.0, 8.0))
        self.assertTrue(np.isfinite(full["phi"]))
        self.assertTrue(np.isfinite(crossfit["phi"]))
        self.assertTrue(density["support_fraction"] > 0.95)
        self.assertIn("survival_outside_unit_interval", density)

    def test_support_sensitivity_and_misspecification_do_not_crash(self):
        sample = make_sample(450, 8, running_law="t5")
        narrow = estimate_once(sample, (-2.5, 2.5))
        wide = estimate_once(sample, (-3.5, 3.5))
        misspecified = make_sample(
            450, 9, running_law="t5", misspecified_outcome=True
        )
        result = estimate_once(misspecified, crossfit_folds=3)
        self.assertTrue(np.isfinite(narrow["phi"]))
        self.assertTrue(np.isfinite(wide["phi"]))
        self.assertTrue(np.isfinite(result["phi"]))
        self.assertFalse(result["inference_available"])
        self.assertIn("grid_boundary", result)

    def test_short_bootstrap_is_finite_and_flagged_as_diagnostic(self):
        sample = make_sample(400, 10, running_law="t5")
        result = short_bootstrap(sample, reps=3)
        self.assertTrue(result["finite"])
        self.assertTrue(np.isfinite(result["sd"]))

    def test_smoke_runner_has_all_scenarios(self):
        result = run_smoke(n=350, reps=1, seed=13)
        self.assertEqual(
            set(result["scenarios"]),
            {
                "t5", "skewed", "misspecified_outcome",
                "support_sensitivity", "bootstrap_diagnostic",
            },
        )
        self.assertEqual(
            set(result["warnings"]),
            {"skewed_grid_boundary", "spline_survival_outside_unit_interval"},
        )

    def test_known_nongaussian_targets_use_their_trim_bounds(self):
        t5_target = population_truth("t5")["phi_star"]
        mixture_target = population_truth("mixture")["phi_star"]
        lognormal_target = population_truth("skewed")["phi_star"]
        self.assertAlmostEqual(t5_target, 0.5878, places=3)
        self.assertAlmostEqual(mixture_target, 0.7930, places=3)
        self.assertAlmostEqual(lognormal_target, 3.0, places=8)


if __name__ == "__main__":
    unittest.main()
