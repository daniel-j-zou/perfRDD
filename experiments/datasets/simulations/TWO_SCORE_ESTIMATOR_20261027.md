# Two-score threshold rules: a pairwise estimator, simulations and first data results (2026-10-27, Claude)

Exploratory. Multiple scores are future work in `PROJECT_SCOPE.md`. The author approved
this design on 2026-10-27: a pairwise value, a tensor-product alpha with an additive
check, weighted sums and both/either cutoffs, simulations first.

Code:
- `experiments/methods/two_score.py`: the estimator;
- `experiments/scripts/two_score_sim.py`: the simulations;
- `experiments/scripts/two_score_apply.py`: the data runs;
- `experiments/tests/test_two_score.py`: the tests.

Outputs are in the ignored `experiments/runs/two_score_sim/` and
`experiments/runs/two_score_apply/`.

## Estimator

It uses the paper's four roles (first stage, outcome, index, evaluation) and averages
the four rotated estimates.

1. **First stage.** OLS of each score on [1, X] gives T = Gamma X~ and eta = Q - T, both
   2-vectors.
2. **Trim.** I(eta) = 1{eps <= pi0(eta) <= 1 - eps}, where pi0(eta) is the share of
   index-fold units with T_j + eta in the deployed set. In one dimension this is the
   paper's quantile window.
3. **Outcome fit.** A stacked spline regression, per arm, of Y on B(eta) and X, fitted on
   units with pi0 in [eps/2, 1 - eps/2]. B is a cubic tensor-product basis with
   m = round(n_arm^{1/7}) interior knots per coordinate, or an additive basis.
4. **Value.** For a rule A:

       U_hat(A) = (n_u n_x)^{-1} sum_i sum_j I_i {alpha_hat(eta_i) - c + beta2_hat'X_j} 1{T_j + eta_i in A}.

   Here i runs over the evaluation fold and j over the index fold. No density of T is
   estimated. Over a grid of rules it is one FFT convolution of linearly binned masses
   followed by a survival transform: 2-D for both/either cutoffs, 1-D per weight for
   weighted averages. The binned values match the exact pairwise sums to 1e-5 once
   nodes are offset by half a cell.
5. **Maximization.** Grid argmax, then quadratic refinement; golden-section search over
   the weight.

**Change from the agreed plan: a smoothness penalty.** Plain least squares on the tensor
basis blew up, with alpha errors in the hundreds. The deployed rule leaves tensor cells
with data from one arm only. Each arm's spline coefficients now carry a second-difference
(P-spline) penalty: pen x (mean diagonal of that arm's Gram block), with pen = 0.03.

On the trim set, alpha RMSE across pen = 0 to 0.1 (symmetric and complementary designs):

| n | pen = 0 | pen = 0.1 |
|---|---|---|
| 16k | 1.5 to 12,000 | 0.10 to 0.15 |
| 64k | 0.16 to 3.9 | 0.06 to 0.09 |

The penalty is a tuning choice that no theory covers yet.

## Simulations (`two_score_sim.py`; 200 data sets at n = 16k, 100 at 64k)

The `symmetric` and `complementary` designs of `two_score_population.py`:
- deployed rule both cutoffs at (0, 0), eps = 0.10, c = 2.25;
- b(eta) = 0.5 eta1 - 0.3 eta2 + 0.2 eta1 eta2, so b is not additive;
- in `complementary`, alpha is not additive either.

Targets come from the population quadrature. Rules:
- `both` and `either`: theta = (phi1, phi2);
- `half`: lam Q1 + (1 - lam) Q2 >= phi, theta = (lam, phi);
- `half_fixed`: lam = 1/2.

The oracle uses the true eta, T, alpha, beta2 and pi0.

| design | rule | truth | oracle n Var 16k / 64k | tensor sqrt(n) bias 64k | tensor n Var 16k / 64k | additive sqrt(n) bias 64k |
|---|---|---|---|---|---|---|
| symmetric | both | (-0.496, -0.496) | (5.48, 5.69) / (5.6, 4.69) | (0.065, -2.472) | (202.53, 206.39) / (119.59, 157.77) | (-8.644, -10.079) |
| symmetric | either | (0.61, 0.61) | (4.71, 3.87) / (3.07, 4.48) | (0.229, 1.785) | (171.11, 147.5) / (134.5, 108.62) | (-7.076, -5.466) |
| symmetric | half | (0.5, 0.102) | (0.2, 1.08) / (0.17, 0.71) | (0.321, 0.029) | (7.97, 31.06) / (5.63, 25.25) | (0.209, -6.467) |
| symmetric | half_fixed | 0.102 | 1.05 / 0.73 | 0.035 | 30.58 / 25.66 | -6.439 |
| complementary | both | (-0.272, -0.272) | (4.06, 3.59) / (3.22, 2.93) | (0.21, -1.739) | (107.87, 110.81) / (75.29, 86.34) | (-20.581, -21.878) |
| complementary | either | (0.815, 0.815) | (8.12, 8.37) / (6.26, 7.96) | (0.259, 3.264) | (302.29, 267.32) / (197.35, 184.3) | (-18.103, -16.387) |
| complementary | half | (0.5, 0.295) | (0.27, 1.19) / (0.21, 0.7) | (0.41, -0.061) | (8.32, 25.71) / (5.84, 20.71) | (0.23, -19.147) |
| complementary | half_fixed | 0.295 | 1.22 / 0.76 | -0.022 | 25.3 / 20.94 | -19.124 |

The Monte Carlo se of sqrt(n) bias at 64k is sqrt(n Var / 100), about 0.5 for `half`
and 1.0-1.4 for the cutoff rules.

- **Oracle.** sqrt(n) bias is near zero, and n Var is stable from 16k to 64k. The
  pairwise argmax behaves like a root-n estimator, as Sherman's argument predicts.
- **Feasible, tensor.**
  - At 64k the bias is small for the weighted-average rules.
  - For the cutoff rules one coordinate is off by -2.5 to +3.3 sqrt(n) units, about 2-2.5
    Monte Carlo se.
  - n Var is 20-40 times the oracle's and still falls from 16k to 64k, so the
    two-dimensional outcome fit dominates and the asymptotic regime is not reached.
- **Feasible, additive.** sqrt(n) bias grows with n (to -20 in `complementary`) because
  alpha or b is not additive there. Use it as a check only.

## Data (`two_score_apply.py`; eps = 0.05; bootstrap with full re-estimation)

| data, Y, c | deployed | weights fixed: cutoff | weights chosen: (weight, cutoff) | both | either |
|---|---|---|---|---|---|
| Chile, university 2025, c = 0.1 (B = 40) | (CL + M1)/2 >= 458 | 417.7 (sd 6.5) | (0.57, 417.6), sd (0.09, 6.4); 47% of draws hit an edge | (384, 349), sd (6, 24) | unstable (sd ~100; 80% edge) |
| Chile, university 2025, c = 0.2 (B = 40) | same | 506.6 (13.4) | (0.51, 507.2), sd (0.06, 10.0) | (486, 443), sd (17, 35) | unstable (90% edge) |
| Florida, next-year points, c = 0 (B = 100) | (3 ach + 4 gains)/7 >= 53.5 | 53.9 (1.2) | (0.31, 53.9), sd (0.07, 1.2) | 76% edge | (62.0, 56.7), sd (4.9, 3.0) |
| Chile, any enrollment 2025, c = 0 | as above | edge (effect slightly negative) | edge | edge | edge |
| Census tracts, change in log jobs 2017-19, c = 0 | poverty >= 20 or income ratio <= 80 | — | edge | edge | edge |

- **Chile.** The costs 0.1 and 0.2 (enrollment probability per eligible student) only
  illustrate the interior case. They are not derived from data.
- **Florida.** Moving only the cutoff finds no reason to leave 53.5. The chosen weight on
  achievement, 0.31 (sd 0.07), is 1.7 sd below the deployed 0.43.
- **Value gains** over the deployed rule are small in every case (Florida +0.04 to +0.05
  points per school; Chile c = 0.1 +0.001).
- **Census tracts.** Eligibility's effect on job growth is about zero, so there is no
  interior optimum. U_hat at the deployed rule is -0.0004.

## Open

- The penalty level (pen = 0.03) is fixed. Choosing it by cross-validation, or letting it
  shrink with n, is untested.
- The feasible n Var has not settled by 64k. A larger n, or an alpha fit with lower
  variance, would show whether the root-n regime holds.
- Theory: the oracle part extends `manuscript/pairwise_theory.tex` (Sherman 1993, 1994).
  The penalized two-dimensional outcome fit, the generated index and the level-set trim
  are not covered. Nothing here is proved.
