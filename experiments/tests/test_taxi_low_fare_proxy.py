"""Tests for the CMT-based low-fare menu-effect proxy."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.scripts.taxi_low_fare_proxy import fit_low_fare_proxy


class TaxiLowFareProxyTest(unittest.TestCase):
    def test_cell_fixed_effect_proxy_recovers_negative_contrast(self) -> None:
        rows = []
        for fare in (5.3, 5.7, 6.1):
            for vendor_cmt in (0.0, 1.0):
                for replicate in range(8):
                    x = np.array([
                        replicate - 3.5,
                        (replicate % 3) - 1.0,
                        0.2 * replicate,
                        -0.1 * replicate,
                    ])
                    tip = 1.0 + 0.2 * x[0] - 0.3 * vendor_cmt
                    rows.append({
                        "fare": fare,
                        "tip": tip,
                        "vendor_cmt": vendor_cmt,
                        **{f"x{j}": x[j] for j in range(4)},
                    })
        result = fit_low_fare_proxy(pd.DataFrame(rows))
        summary = result["summary"]
        self.assertAlmostEqual(
            summary["vts_weighted_proxy_dollars_per_trip"], -0.3, places=8
        )
        self.assertEqual(summary["vts_weighted_share_with_negative_proxy"], 1.0)
        self.assertIsNone(summary["point_crossing_fare"])


if __name__ == "__main__":
    unittest.main()
