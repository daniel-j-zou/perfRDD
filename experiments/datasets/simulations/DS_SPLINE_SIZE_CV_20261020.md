# Spline sizes: cross-validation against the fixed rule (2026-10-20, Claude)

Robustness check requested by the author (weekly scope review, 2026-10-20). The paper
uses a fixed rule for the spline sizes, and cross-validation over a grid inside the
(A5) range is a check only.

- Script: `experiments/scripts/ds_spline_size_cv.py`.
- Methods: `experiments/methods/spline_size_cv.py`; tests in
  `experiments/tests/test_spline_size_cv.py`.
- Results: the ignored `experiments/runs/ds_spline_size_cv/`.

## Setup

Each data set gets one random four-block split and is estimated twice with the
rotated four-role estimator. The seeds are those of `ds_four_role_coverage.py`, so the
rule arm repeats that script's point estimates (checked to 1e-8).

- **Rule:**
  - outcome knots max(3, round(m^{11/60})), with m the smaller arm count in J;
  - density basis max(8, round(2 n_T^{11/60})).
- **CV**, chosen in each rotation:
  - outcome knots minimize the 5-fold cross-validated squared error of the
    arm-wise fit on the outcome fold, over max(3, round(m^a)) for
    a in (0.15, 0.17, 0.19, 0.21, 0.23);
  - the density basis minimizes the least-squares cross-validation score of the
    density block on the index fold, over max(8, round(2 n_T^a)) for
    a in (0.13, 0.155, 0.18, 0.205, 0.23). The leave-one-out fit is in closed
    form, checked against brute force.
- All exponents lie inside the new range, n^{1/7} << K << n^{1/4} and
  n^{1/8} << L << n^{1/4}.
- 300 data sets per cell; point estimates only, no bootstrap.

## Results (average of the four rotated thresholds)

| design, n | sqrt(n) bias: rule | CV | n Var: rule | CV | paired difference, mean (sd), sqrt(n) units |
|---|---:|---:|---:|---:|---|
| displayed, 16,000 | 0.88 (0.35) | 1.54 | 37.35 | 37.65 | +0.66 (0.48) |
| displayed, 64,000 | 0.68 (0.32) | 1.08 | 31.25 | 30.83 | +0.40 (0.48) |
| nonlinear, 3,200 | 0.26 (0.13) | 0.24 | 5.10 | 5.02 | -0.02 (0.12) |
| nonlinear, 12,800 | -0.21 (0.14) | -0.13 | 5.63 | 5.82 | +0.08 (0.15) |

Bias standard errors are in brackets. The paired difference CV minus rule has
standard error sd/sqrt(300), about 0.028 in the displayed design. n MSE, rule against
CV:

| design, n | rule | CV |
|---|---:|---:|
| displayed, 16,000 | 38.0 | 39.9 |
| displayed, 64,000 | 31.6 | 31.9 |
| nonlinear, 3,200 | 5.15 | 5.06 |
| nonlinear, 12,800 | 5.65 | 5.82 |

Sizes chosen, counted over 4 rotations x 300 data sets. K is the number of outcome
basis functions per arm and L the number of density basis functions.

| design, n | rule K / L | CV outcome K | CV density L |
|---|---|---|---|
| displayed, 16,000 | 8 / 9 | 7: 909, 8: 174, 9: 73, 10: 44 | 8: 721, 9: 351, 11: 99, 13: 29 |
| displayed, 64,000 | 9 / 12 | 8: 860, 9: 217, 10: 87, 12: 36 | 8: 200, 9: 106, 11: 851, 15: 37, 19: 6 |
| nonlinear, 3,200 | 7 / 8 | 7: 1007, 8: 193 | 8: 922, 9: 278 |
| nonlinear, 12,800 | 8 / 9 | 7: 948, 8: 164, 9: 88 | 8: 591, 9: 309, 10: 262, 13: 38 |

## Reading

- **Variance.** CV does not change the variance: n Var agrees within Monte Carlo
  error in all four cells.
- **Bias.** In the displayed design, CV shifts the estimate upward by 0.66 and 0.40
  sqrt(n) units. That is significant against a paired s.e. of about 0.03, and it
  roughly doubles the bias. In the nonlinear design the two agree.
- **Choices.** CV mostly picks the smallest outcome basis on its grid and a density
  basis at or below the rule's. Least-squares CV targets the L2 error of the whole
  density block, not the bias of the criterion's derivative at phi*.
- **Conclusion.** These runs support the fixed rule, with CV as a check. None of
  this enters the theory; Appendix 1 covers any selection from a fixed finite grid.
