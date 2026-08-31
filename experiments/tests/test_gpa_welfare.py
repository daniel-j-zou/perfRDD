import unittest

import numpy as np
import pandas as pd

from experiments.datasets.gpa.welfare import build_welfare_outcomes


class GPAWelfareMenuTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "nextGPA": [0.4, np.nan, 0.2, np.nan],
            "gpacutoff": [1.5, 1.5, 1.6, 1.6],
            "fallreg_year2": [1, 0, 1, 0],
            "left_school": [0, 1, 1, 0],
            "credits_earned2": [4.0, 0.0, 2.0, 0.0],
            "goodstanding_year2": [1, 0, 1, 0],
        })

    def test_menu_is_complete_and_prespecified(self):
        outcomes = build_welfare_outcomes(self.frame)
        self.assertEqual(len(outcomes), 16)
        self.assertEqual(len(set(outcomes)), 16)
        for outcome in outcomes.values():
            self.assertEqual(len(outcome.values), len(self.frame))
            self.assertTrue(np.isfinite(outcome.values).all())

    def test_missing_gpa_assignment_uses_absolute_gpa_units(self):
        outcomes = build_welfare_outcomes(self.frame)
        actual = outcomes["composite_no_record_gpa_0p8"].values
        np.testing.assert_allclose(actual, [0.4, -0.7, 0.2, -0.8])

    def test_leave_penalty_is_distinct_from_missing_record(self):
        outcomes = build_welfare_outcomes(self.frame)
        base = outcomes["composite_no_record_gpa_0p8"].values
        penalized = outcomes["composite_a0p8_leave_penalty_0p25"].values
        np.testing.assert_allclose(penalized - base, [0.0, -0.25, -0.25, 0.0])
        # Row 2 has a recorded GPA and is still penalized because it is a leaver.
        self.assertTrue(np.isfinite(self.frame.loc[2, "nextGPA"]))
        self.assertEqual(penalized[2] - base[2], -0.25)

    def test_return_bonus_is_applied_by_status(self):
        outcomes = build_welfare_outcomes(self.frame)
        base = outcomes["composite_no_record_gpa_0p8"].values
        rewarded = outcomes["composite_a0p8_return_bonus_0p25"].values
        np.testing.assert_allclose(rewarded - base, [0.25, 0.0, 0.25, 0.0])


if __name__ == "__main__":
    unittest.main()
