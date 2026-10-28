import unittest

import numpy as np

from experiments.methods.two_score import (
    Pairs, Rule, TwoScoreProblem, estimate, prob_in_rule)


def _pairs(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 2))
    T = rng.standard_normal((n, 2)) @ np.array([[1.0, 0.3], [0.0, 0.9]])
    I = (np.abs(e[:, 0] + e[:, 1]) < 2.5).astype(float)
    u = I * (0.5 * e[:, 0] + 0.3 * e[:, 1] - 0.2)
    x = 0.4 * rng.standard_normal(n)
    return Pairs(e=e, u=u, I=I, T=T, x=x, n_u=n, n_x=n)


class TwoScoreTest(unittest.TestCase):
    def test_binned_surfaces_match_the_exact_pairwise_value(self):
        pr = _pairs()
        from scipy.interpolate import RegularGridInterpolator as R
        (n0, n1), S = pr.surface2()
        both = R((n0, n1), S)
        either = R((n0, n1), S[:, :1] + S[:1, :] - S)
        for phi in [(0.0, 0.0), (-1.0, 0.7), (1.2, -0.4)]:
            self.assertAlmostEqual(float(both([phi])[0]), pr.value(Rule("both", phi)), delta=2e-4)
            self.assertAlmostEqual(float(either([phi])[0]), pr.value(Rule("either", phi)), delta=2e-4)
        for lam, phi in [(0.5, 0.2), (0.2, -0.5)]:
            x, Sc = pr.curve1(lam)
            self.assertAlmostEqual(float(np.interp(phi, x, Sc)),
                                   pr.value(Rule("halfplane", (lam, phi))), delta=2e-4)

    def test_prob_in_rule_matches_brute_force(self):
        rng = np.random.default_rng(1)
        T = rng.standard_normal((4000, 2))
        e = rng.standard_normal((50, 2))
        for rule in (Rule("halfplane", (0.3, 0.2)), Rule("both", (0.1, -0.2)),
                     Rule("either", (0.5, 0.4))):
            brute = np.array([rule.treat_adj(T + ei).mean() for ei in e])
            self.assertLess(np.max(np.abs(prob_in_rule(rule, e, T) - brute)), 0.01)

    def test_estimate_recovers_the_population_optimum(self):
        # 'symmetric' design of two_score_population.py: best both-cutoffs rule (-0.496, -0.496)
        from experiments.scripts.two_score_sim import generate
        pop, X, eta, T, Q, Y = generate("symmetric", 32000, 11)
        d = pop.d
        prob = TwoScoreProblem(Q=Q, X=X, Y=Y, deployed=Rule("both", tuple(d.phi0)),
                               cost=d.cost, eps=d.eps)
        r = estimate(prob, {"both": {"kind": "both", "box": ((-3, 3), (-3, 3))}}, seed=3)
        self.assertEqual(r["both"]["edge_rotations"], 0)
        np.testing.assert_allclose(r["both"]["theta"], [-0.496, -0.496], atol=0.35)


if __name__ == "__main__":
    unittest.main()
