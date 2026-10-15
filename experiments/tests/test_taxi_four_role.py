"""Checks for the taxi four-role application script (no taxi data needed)."""
import unittest

import numpy as np
import pandas as pd

from experiments.scripts.taxi_four_role import (
    Data, boxcox, fare_cell, four_role, inv_boxcox, make_design, score_covariates)


def _synthetic(n, seed):
    """Differing-slopes design with Z independent of eta, scored on fare (lambda = 1)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    eta = 1.5 * rng.standard_normal(n)
    fare = 12.0 + 2.5 * Z[:, 0] + 1.0 * Z[:, 1] + eta
    D = (fare >= 15.0).astype(float)
    X = Z[:, :1]
    effect = -0.9 + 0.3 * eta + 0.6 * X[:, 0]
    Y = (0.5 * eta + 0.2 * eta ** 2 + 0.3 * Z[:, 0] - 0.2 * Z[:, 1]
         + D * effect + 0.5 * rng.standard_normal(n))
    return Data(boxcox(fare, 1.0), Z, X, Y, fare, 1.0)


class TaxiFourRole(unittest.TestCase):
    def test_boxcox_round_trip(self):
        x = np.array([2.5, 6.0, 15.0, 40.0])
        for lam in (0.0, 0.25, 0.5, 1.0):
            np.testing.assert_allclose(inv_boxcox(boxcox(x, lam), lam), x, rtol=1e-12)
            # monotone, so a cutoff on the transformed fare is a cutoff on the fare
            self.assertTrue(np.all(np.diff(boxcox(x, lam)) > 0))

    def test_fare_cell(self):
        fares = np.array([12.5, 12.9, 13.3, 12.9])
        self.assertEqual(fare_cell(12.6, fares), 12.9)
        self.assertEqual(fare_cell(12.9, fares), 12.9)
        self.assertEqual(fare_cell(14.0, fares), float("inf"))

    def test_score_covariates_centered_and_full_rank(self):
        rng = np.random.default_rng(4)
        n = 20_000
        dow = rng.integers(0, 7, n).astype(float)
        hour = np.where(dow >= 5, rng.uniform(6, 20, n), rng.uniform(6, 16, n))
        df = pd.DataFrame({"Trip_Distance": rng.lognormal(0.7, 0.7, n),
                           "Passenger_Count": rng.integers(1, 6, n),
                           "hour_of_day": hour, "day_of_week": dow})
        Z = score_covariates(df)
        np.testing.assert_allclose(Z.mean(0), 0.0, atol=1e-10)
        A = np.column_stack((np.ones(n), Z))
        self.assertEqual(np.linalg.matrix_rank(A), A.shape[1])

    def test_recovers_supported_optimum(self):
        data = _synthetic(60_000, seed=5)
        design = make_design(data, eps=0.10)
        # population target by Monte Carlo on a large independent draw, same window rule
        big = _synthetic(2_000_000, seed=6)
        g = np.linalg.lstsq(np.column_stack((np.ones(len(big.Q)), big.Z)), big.Q, rcond=None)[0]
        T = np.column_stack((np.ones(len(big.Q)), big.Z)) @ g
        eta = big.Q - T
        lo = max(design.phi0 - np.quantile(T, 0.9), design.J[0])
        hi = min(design.phi0 - np.quantile(T, 0.1), design.J[1])
        inside = (eta >= lo) & (eta <= hi)
        W = -0.9 + 0.3 * eta + 0.6 * big.X[:, 0]
        grid = np.linspace(*design.policy, 401)
        U = [np.mean(inside * W * (big.Q >= p)) for p in grid]
        target = grid[int(np.argmax(U))]
        est = four_role(data, design, seed=1)
        self.assertFalse(est["at_boundary"])
        self.assertLess(abs(est["phi_avg"] - target), 0.15)   # target grid step is 0.045
        self.assertLess(abs(est["phi_pooled"] - est["phi_avg"]), 0.1)
        self.assertAlmostEqual(est["beta2"][0], 0.6, delta=0.05)


if __name__ == "__main__":
    unittest.main()
