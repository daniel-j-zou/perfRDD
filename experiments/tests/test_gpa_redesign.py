import unittest

import numpy as np
import pandas as pd

from experiments.datasets.gpa.redesign import composite_next_gpa


class GPACompositeOutcomeTest(unittest.TestCase):
    def test_observed_values_are_preserved_and_missing_values_share_units(self):
        frame = pd.DataFrame({
            "nextGPA": [0.4, np.nan, -0.2, np.nan],
            "gpacutoff": [1.5, 1.5, 1.6, 1.6],
        })
        actual = composite_next_gpa(frame, no_grade_absolute_gpa=0.9)
        np.testing.assert_allclose(actual, [0.4, -0.6, -0.2, -0.7])

    def test_assumed_absolute_gpa_is_validated(self):
        frame = pd.DataFrame({"nextGPA": [np.nan], "gpacutoff": [1.5]})
        with self.assertRaises(ValueError):
            composite_next_gpa(frame, -0.1)
        with self.assertRaises(ValueError):
            composite_next_gpa(frame, np.nan)

    def test_no_record_penalty_only_changes_missing_rows(self):
        frame = pd.DataFrame({
            "nextGPA": [0.4, np.nan],
            "gpacutoff": [1.5, 1.6],
        })
        actual = composite_next_gpa(
            frame,
            no_grade_absolute_gpa=0.0,
            no_grade_penalty=2.0,
        )
        np.testing.assert_allclose(actual, [0.4, -3.6])
        with self.assertRaises(ValueError):
            composite_next_gpa(frame, 0.0, no_grade_penalty=-0.1)


if __name__ == "__main__":
    unittest.main()
