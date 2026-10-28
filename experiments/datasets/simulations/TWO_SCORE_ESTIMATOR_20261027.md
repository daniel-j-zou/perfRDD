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

## Follow-up: cross-validated penalty and larger n (2026-10-27, Claude)

At the author's request ("Tune and check").

**Penalty by cross-validation.** `outcome_fit(pen="cv")` is now the default. It picks pen
from PEN_GRID by 5-fold cross-validation of the squared prediction error of Y on the
outcome fold. The grid was 0.001-10 for the simulations below, and 0.001-100 after data
runs reached the top of the grid. The penalty matrix (second differences in each
coordinate) leaves functions linear in each coordinate unpenalized, so a very large
penalty fits alpha and b as bilinear in (eta1, eta2).

**Simulations** (`two_score_sim.py --estimators tensor tensor_cv --tag _cv`): 200 / 100 /
100 / 50 data sets at n = 16k / 64k / 256k / 1M; `tensor` keeps pen = 0.03. The 16k and
64k draws are the same data sets as above.

| design | rule | estimator | n Var at 16k / 64k / 256k / 1M | sqrt(n) bias at 1M |
|---|---|---|---|---|
| symmetric | half_fixed | oracle | 1.05 / 0.73 / 0.79 / 0.84 | -0.23 |
| symmetric | half_fixed | tensor | 30.6 / 25.7 / 31.4 / 23.5 | 0.78 |
| symmetric | half_fixed | tensor_cv | 31.5 / 25.6 / 31.7 / 24.8 | -0.584 |
| symmetric | half | oracle | (0.2, 1.08) / (0.17, 0.71) / (0.22, 0.79) / (0.2, 0.84) | (-0.054, -0.225) |
| symmetric | half | tensor | (7.97, 31.1) / (5.63, 25.2) / (6.1, 31.5) / (6.46, 23.5) | (-0.054, 0.768) |
| symmetric | half | tensor_cv | (7.23, 31.6) / (5.48, 25.4) / (6.01, 31.6) / (6.24, 24.8) | (-0.334, -0.592) |
| symmetric | both | oracle | (5.48, 5.69) / (5.6, 4.69) / (6.13, 6.74) / (7, 5.11) | (-0.932, 0.035) |
| symmetric | both | tensor | (203, 206) / (120, 158) / (180, 117) / (126, 162) | (0.104, 1.69) |
| symmetric | both | tensor_cv | (161, 187) / (116, 148) / (178, 115) / (123, 164) | (-3.17, 1.22) |
| symmetric | either | oracle | (4.71, 3.87) / (3.07, 4.48) / (3.18, 3.19) / (3.46, 4.1) | (0.052, -0.497) |
| symmetric | either | tensor | (171, 148) / (134, 109) / (95.9, 146) / (143, 122) | (2.36, 0.887) |
| symmetric | either | tensor_cv | (170, 145) / (134, 109) / (96.8, 147) / (144, 117) | (1.92, -1.99) |
| complementary | half_fixed | oracle | 1.22 / 0.76 / 0.81 / 1.02 | -0.218 |
| complementary | half_fixed | tensor | 25.3 / 20.9 / 27.3 / 18.8 | 0.774 |
| complementary | half_fixed | tensor_cv | 26.4 / 20.6 / 27.4 / 19.5 | -0.53 |
| complementary | half | oracle | (0.27, 1.19) / (0.21, 0.7) / (0.25, 0.81) / (0.3, 1.02) | (-0.029, -0.218) |
| complementary | half | tensor | (8.32, 25.7) / (5.84, 20.7) / (6.43, 27.3) / (7.42, 18.8) | (0.017, 0.745) |
| complementary | half | tensor_cv | (7.6, 26.7) / (5.71, 20.5) / (6.36, 27.4) / (7.29, 19.5) | (-0.22, -0.548) |
| complementary | both | oracle | (4.06, 3.59) / (3.22, 2.93) / (3.71, 4.15) / (5.64, 4.42) | (-0.633, -0.091) |
| complementary | both | tensor | (108, 111) / (75.3, 86.3) / (113, 73.5) / (76.8, 105) | (0.531, 1.34) |
| complementary | both | tensor_cv | (103, 113) / (76.9, 83.3) / (115, 72.5) / (77.1, 106) | (-1.81, 0.767) |
| complementary | either | oracle | (8.12, 8.37) / (6.26, 7.96) / (6.55, 6.89) / (7.63, 9.33) | (0.114, -0.337) |
| complementary | either | tensor | (302, 267) / (197, 184) / (169, 229) / (251, 214) | (2.7, 1.54) |
| complementary | either | tensor_cv | (282, 240) / (190, 179) / (166, 229) / (248, 207) | (2.76, -1.4) |

- **Oracle.** n Var is flat from 16k to 1M.
- **Feasible.**
  - n Var stops falling after 64k and stays flat to 1M, within the Monte Carlo error of
    about 20% at 50 data sets. It is 30-40 times the oracle's for the weighted-average
    rules and 20-40 times for the cutoff rules. So the estimator reaches a root-n regime
    from about n = 64k, with a large variance constant. The earlier reading that the
    asymptotic regime was not reached rested on 16k and 64k only.
  - The penalty level does not change the variance.
- **Cross-validated penalty.**
  - The chosen penalty falls with n: mostly 1 at 16k, 0.3 at 64k-256k, 0.1 at 1M.
  - Its sqrt(n) bias in the cutoff is -3.0 at 16k, -1.9 at 256k and -0.6 at 1M for the
    weighted-average rules, a smoothing bias that fades with n.
  - At 1M, both penalties have sqrt(n) bias within about 1-2 Monte Carlo se.

**Data with the cross-validated penalty.**

| data | penalty chosen (per rotation) | weights fixed: cutoff (boot sd) | weights chosen: (weight, cutoff) (boot sd) | both (boot sd) | either |
|---|---|---|---|---|---|
| Chile, university 2025, c = 0.1 (B = 40) | 1, 1, 10, 3 | 416.0 (6.2) | (0.55, 415.9), sd (0.07, 6.5); 20% of draws hit an edge | (380, 353), sd (6, 18) | unstable (sd 89-96) |
| Chile, university 2025, c = 0.2 (B = 40) | 1, 1, 10, 3 | 503.6 (11.4) | (0.52, 503.9), sd (0.04, 11.4) | (478, 452), sd (16, 23) | unstable (sd 70-74) |
| Florida, next-year points (B = 100) | 100 in every rotation | 53.6 (1.1) | (0.31, 53.6), sd (0.06, 1.1) | 59% edge | (61.4, 56.3), sd (3.6, 1.6) |
| Census tracts; Chile any enrollment | 30-100; 1-100 | edge | edge | edge | edge |

- The estimates move by at most a few points from the pen = 0.03 runs.
- In Florida and the census tracts, cross-validation picks the top of the grid, so alpha
  is fitted as bilinear in (eta1, eta2).

## More data (2026-10-28, Claude)

The estimator (cross-validated penalty, eps 0.05, c = 0; Y = next year's score) on the
third-round datasets. Bootstrap with full re-estimation; "edge" is the share of draws
with a rotation at the edge of the policy box.

| data (B) | deployed | weights fixed: cutoff (sd) | weights chosen: (weight, cutoff) (sd); edge | both (sd); edge | either |
|---|---|---|---|---|---|
| Texas A/B (50) | 0.7 M + 0.3 CtG >= 89.5 | 87.9 (0.9) | (0.56, 87.3), sd (0.07, 0.9); 66% | (86.8, 82.0), sd (1.8, 1.3); 4% | unstable (100% edge) |
| North Carolina B/C (100) | 0.8 A + 0.2 G >= 69.5 | 69.5 (1.6) | (0.88, 68.4), sd (0.06, 1.4); 84% | (66.7, 50.3); 94% | collapses to a cutoff on achievement |
| NC low-performing (100) | designated if score < 54.5 and G < 85 | — | — | score < 59.6 and G < 90.0, sd (2.5, 3.5); 69% | unstable |
| Louisiana two-score (100) | 0.75 assessment + 0.25 progress >= 75 | 78.6 (5.2) | (0.71, 82.4); 86% | 62% edge | (80.2, 115.9), sd (5.5, 12.2) |
| CDFI investment areas, two-score | poverty >= 20 or unemployment >= 1.5 x national | — | — | edge | edge (effect on job growth about zero) |

The Chile cohorts are in `../chile_two_score/README.md`. With the weights fixed, the
estimated cutoff is 416-436 at c = 0.1 and 476-504 at c = 0.2 across four samples. The
chosen reading weight is 0.55-0.61 in every sample.

- **Texas** is the cleanest school result. With the weights fixed, the A/B cutoff is
  estimated about 1.9 sd below the deployed 89.5. The both-cutoffs rule is well
  determined (4% edge).
- **North Carolina** gives no reason to move the B/C cutoff. The low-performing rule
  would designate more schools (score cutoff 59.6 against 54.5, about 2 sd).
- **Louisiana**'s two-score sample (1,089) is too small for the four-role split.
- **Gains in value** over the deployed rules are small everywhere (Texas +0.009 to +0.022
  points; NC +0.007 to +0.015; NC low-performing +0.08).

## Fourth-round data (2026-10-28, Claude)

Same settings (cross-validated penalty, eps 0.05, c = 0 unless stated; 40 bootstrap
draws with full re-estimation; "edge" is the share of draws with a rotation at the edge
of the policy box). Datasets: `../MULTI_RULE_DATASETS_20261027.md`, "Fourth round".

New in `two_score.py`: a deployed rule `unit_halfplane`, w_i Q1 + (1 - w_i) Q2 >= c_i with
a weight and cutoff per unit (Wisconsin). pi0 and the deployed value are computed
pairwise, O(n_e n_x), with each index unit's own (w_j, c_j); `Rule.subset` restricts the
arrays to a fold or a bootstrap draw. Tests: it equals the plain half-plane when w and c
are constant, and pi0 matches brute force.

| data (n) | deployed | weights fixed: cutoff (sd); edge | both (sd); edge | reading |
|---|---|---|---|---|
| NYC A/B (5,348) | 0.4 Q1 + 0.6 Q2 >= 0 (percent units) | 4.6 (3.2); 10% | (-21.4, 0.1), sd (9.8, 5.3); 40% | the only interior estimate this round; 1.4 sd above the deployed cut |
| NYC C/D (3,673) | same | -9.5 (2.8); 95% | edge | large, noisy effect (6.1, sd 2.5) with selective attrition below the cut |
| Wisconsin 4 stars (13,710) | per-school weight and cutoff | lower edge (treat all); 100% | edge | flat: at c = 2 the four rotations give cutoffs from 45 to 90; at c = 3 all go to the top |
| Indiana D/F and B/C (2,590) | 0.5 P + 0.5 G >= 59.95 / 79.95 | sd 7.6 and 6.7; 90% and 78% | sd 13-16 | too small |
| Gainful employment, not passing (7,065) | a > 8 and d > 20 | — | (7.7, 11.8), sd (2.3, 7.2); 100% | effect on closure about 1 point (sd 0.8); flat |
| California Red, ELA / math (49,451 / 49,442) | status < 0 and change < 3 | — | sd about 10 and 5; 100% | no effect of Red on next status (-0.05 and 0.05, sd 0.04) |
| Chile top 10% (225,050) | PAES >= 458 or grade >= class cutoff | — | grade cutoff at the box edge | PAES cutoff 397 (c = 0) and 448 (c = 0.1); the grade route is pushed out, weakly identified (R^2 0.03) |

- Only NYC A/B gives an interior estimate, and it is not distinguishable from the
  deployed cut.
- Where the estimated effect is flat (Wisconsin, California, gainful employment) or
  noisy (NYC C/D), every rule class goes to the edge, as in the census-tract data.
- The datasets with clear results remain Texas, North Carolina and the Chile cohorts
  (previous section).
