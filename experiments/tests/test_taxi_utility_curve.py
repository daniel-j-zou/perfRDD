"""Unit tests for the taxi hard-trim utility-curve reconstruction."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.scripts.taxi_utility_curve import (
    evaluate_curve,
    reconstruct_components,
    summarize_curve,
)


class TaxiUtilityCurveTest(unittest.TestCase):
    def test_reconstructs_benefit_and_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utility_curve.csv"
            frame = pd.DataFrame({
                "phi": [5.0, 15.0, 25.0],
                "cost_0": [0.30, 0.15, 0.03],
                "cost_0.1": [0.20, 0.10, 0.02],
            })
            frame.to_csv(path, index=False)
            reconstructed = reconstruct_components(path)

        np.testing.assert_allclose(
            reconstructed["gross_tip_benefit"], [0.30, 0.15, 0.03]
        )
        np.testing.assert_allclose(
            reconstructed["treatment_exposure"], [1.0, 0.5, 0.1]
        )

    def test_summarizes_gain_relative_to_current_policy(self) -> None:
        components = pd.DataFrame({
            "phi": [5.0, 15.0, 25.0],
            "gross_tip_benefit": [0.30, 0.15, 0.03],
            "treatment_exposure": [1.0, 0.5, 0.1],
        })
        result = summarize_curve(
            evaluate_curve(components, 0.20),
            policy_minimum=5.0,
            policy_maximum=25.0,
        )
        self.assertEqual(result["phi_star"], 5.0)
        self.assertAlmostEqual(
            result["gain_over_current_cents_per_trimmed_trip"], 5.0
        )


if __name__ == "__main__":
    unittest.main()
