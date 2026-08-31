"""Tests for the restricted taxi sample and diagnostic bootstrap helpers."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.datasets.taxi.adapter import prepare_haggag_paci_frame
from experiments.scripts.taxi_bootstrap_30k import (
    centered_interval,
    simultaneous_relative_band,
)


class TaxiBootstrapTest(unittest.TestCase):
    def test_paper_restrictions_keep_only_eligible_ride(self) -> None:
        frame = pd.DataFrame({
            "vendor_name": ["VTS"] * 5,
            "Payment_Type": ["Credit"] * 5,
            "Trip_Pickup_DateTime": [
                "2009-01-05 12:00:00",  # eligible weekday daytime
                "2009-01-05 20:00:00",  # excluded nighttime
                "2009-01-05 12:00:00",  # excluded toll
                "2009-11-05 12:00:00",  # excluded after October
                "2009-01-05 12:00:00",  # excluded non-meter grid
            ],
            "Fare_Amt": [14.9, 14.9, 14.9, 14.9, 15.0],
            "Tip_Amt": [2.0] * 5,
            "Tolls_Amt": [0.0, 0.0, 1.0, 0.0, 0.0],
            "surcharge": [0.0] * 5,
            "mta_tax": [np.nan] * 5,
            "Trip_Distance": [2.0] * 5,
            "Passenger_Count": [1.0] * 5,
        })
        restricted = prepare_haggag_paci_frame(frame)
        self.assertEqual(len(restricted), 1)
        self.assertAlmostEqual(float(restricted.iloc[0]["Fare_Amt"]), 14.9)

    def test_centered_interval_and_relative_band(self) -> None:
        estimate = np.array([2.0, 1.0, 0.0])
        curves = np.array([
            [2.1, 1.1, 0.0],
            [1.9, 0.9, 0.0],
            [2.2, 1.0, 0.0],
            [1.8, 1.0, 0.0],
        ])
        band = simultaneous_relative_band(estimate, curves, current_index=2)
        np.testing.assert_allclose(band["estimate"], estimate)
        self.assertGreater(float(band["critical_value"]), 0.0)

        lower, upper = centered_interval(1.0, np.array([0.8, 0.9, 1.1, 1.2]))
        self.assertLess(lower, 1.0)
        self.assertGreater(upper, 1.0)


if __name__ == "__main__":
    unittest.main()
