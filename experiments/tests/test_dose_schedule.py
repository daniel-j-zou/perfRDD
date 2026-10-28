"""Checks of experiments/methods/dose_schedule.py on the Model 2 design of the dose note.

S ~ U[-0.5, 1], Z ~ U[-1, 1], eta ~ U[-1/4, 1/4], Q = S + eta;
Y = S - 0.5 Z + sin(6 eta) + f(eta, A) + 0.2 A Z + N(0, 0.25^2), f = (1 + 0.5 e) a - 0.4 a^2;
cost c(a) = 0.2 a + a^2, so d*(q) = (0.8 + 0.5 E[eta | Q = q]) / 2.8.
"""
import unittest

import numpy as np

from experiments.methods import dose_schedule as ds


def draw(n, rng, schedule, score_modifier=0.0):
    S, Z, eta = rng.uniform(-0.5, 1, n), rng.uniform(-1, 1, n), rng.uniform(-0.25, 0.25, n)
    Q = S + eta
    A = schedule(Q)
    f = (1 + 0.5 * eta + score_modifier * S) * A - 0.4 * A ** 2
    Y = S - 0.5 * Z + np.sin(6 * eta) + f + 0.2 * A * Z + 0.25 * rng.standard_normal(n)
    return Q, np.column_stack([S, Z]), A, Y, Z[:, None]


def d_star(q):
    lo, hi = np.maximum(-0.25, q - 1), np.minimum(0.25, q + 0.5)
    return (0.8 + 0.5 * (lo + hi) / 2) / 2.8


class TestDoseSchedule(unittest.TestCase):
    def test_continuous_schedule_recovers_d_star(self):
        rng = np.random.default_rng(1)
        Q, X, A, Y, Z = draw(150_000, rng, lambda q: np.clip(0.5 - q, 0, None))
        prob = ds.DoseProblem(Q, X, A, Y, Z, ds.DoseBasis(0.0, 0.6, l=3), a_lo=0.0)
        est = ds.estimate(prob, n_bins=15, seed=0)
        mid = 0.5 * (est["edges"][:-1] + est["edges"][1:])
        dh = ds.d_hat(est["psi"], est["a_grid"], lambda a: 0.2 * a + a ** 2)
        inner = (mid > -0.5) & (mid < 1.0)
        self.assertLess(np.max(np.abs(dh - d_star(mid))[inner]), 0.06)
        # one-to-one schedule: the score-direction modifier is absorbed by f
        self.assertGreater(ds.modifier_test(prob)["r2_absorbed"], 0.99)

    def test_step_schedule_levels_and_modifier_test(self):
        steps = lambda q: np.select([q < 0.0, q < 0.5], [0.4, 0.2], 0.0)
        levels = np.array([0.2, 0.4])
        rng = np.random.default_rng(2)
        Q, X, A, Y, Z = draw(150_000, rng, steps)
        prob = ds.DoseProblem(Q, X, A, Y, Z, ds.DoseBasis(0.0, 0.4, levels=levels), a_lo=0.0)
        est = ds.estimate(prob, n_bins=10, seed=0)
        mid = 0.5 * (est["edges"][:-1] + est["edges"][1:])
        lo, hi = np.maximum(-0.25, mid - 1), np.minimum(0.25, mid + 0.5)
        m1 = (lo + hi) / 2
        for j, a in enumerate(est["a_grid"]):
            truth = (1 + 0.5 * m1) * a - 0.4 * a ** 2     # E[Z | Q] = 0
            self.assertLess(np.max(np.abs(est["psi"][:, j] - truth)), 0.03)
        t0 = ds.modifier_test(prob)
        self.assertLess(t0["r2_absorbed"], 0.97)
        self.assertLess(abs(t0["coef"]) / t0["se"], 3.5)
        # a modifier along the score direction is detected
        Q, X, A, Y, Z = draw(150_000, np.random.default_rng(3), steps, score_modifier=0.4)
        t1 = ds.modifier_test(ds.DoseProblem(Q, X, A, Y, Z, ds.DoseBasis(0.0, 0.4, levels=levels), a_lo=0.0))
        self.assertGreater(abs(t1["coef"]) / t1["se"], 5)


if __name__ == "__main__":
    unittest.main()
