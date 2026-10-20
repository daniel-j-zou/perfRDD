# Four-role estimator: bootstrap coverage, nonlinear design and oracle arm (2026-10-19)

## Question

This run closes the simulation gaps in the todo box of `differing_slopes.tex`
Section 5 (Numerical study):

1. bootstrap coverage of the rotated four-role estimator in the displayed design,
   across `n`, with the analytic variance as a cross-check only;
2. the nonlinear `alpha, b` design of Section 5.3, rerun with the four-role
   estimator and the spline outcome fit, with coverage;
3. an oracle-nuisance arm in both designs;
4. adaptive quadrature on the trim window, replacing the Gauss-Hermite
   population targets in three older scripts.

The paper text is unchanged. The tables below await the author's approval.

## Estimator and bootstrap

Runner: `experiments/scripts/ds_four_role_coverage.py`. It reuses the fits of
`ds_local_efficiency_sim.py`, which implements the paper's estimator.

- **Split.** Four random blocks. Rotation `k` assigns the first-stage, outcome,
  index and evaluation roles to blocks `k, k+1, k+2, k+3` (mod 4).
- **Fits.**
  - OLS first stage.
  - Arm-wise stacked cubic-spline outcome fit on `J = [-1.75, 1.75]`, with
    `max{3, [m^{11/60}]}` interior knots.
  - Spline `g` and `rho_X` on `T = [-3, 3]`, with `max{8, [2 n_T^{11/60}]}`
    functions.
  - Trim endpoints from empirical quantiles of the fitted index.
- **Criterion and argmax.** The hard-trimmed evaluation average, normalized by
  the evaluation fold size. It is maximized over `Phi = [-1.5, 1.5]` on a
  61-point grid, refined by a bounded scalar search.
- **Estimators recorded.**
  - `rotated average`: the average of the four rotated thresholds (the
    paper's estimator).
  - `pooled maximizer`: the maximizer of the averaged criterion.
  - `fixed`: rotation 0 alone, point estimate only.
- **Oracle arm.** Same folds, with every nuisance at its true value (`gamma`,
  `alpha`, `beta2`, `g`, `rho_X`, `[l0, u0]`). Only the evaluation averages are
  estimated. Its first-order `n`-scaled variance is `Var psi_U / H^2`.
- **Bootstrap.** Full re-estimation. Each draw resamples `n` observations with
  replacement, splits the resample into four new random blocks and re-runs
  every fit. There are 199 draws per data set. The intervals are the 95%
  percentile interval and the normal interval, estimate +- 1.96 bootstrap s.d.
- **Analytic check.** The coverage of estimate +- 1.96 sqrt(V/n), with the
  first-order variance `V` from `ds_local_efficiency.Population`. It is a
  check, not an inference method.

## Designs

| | displayed (Section 5.1) | nonlinear (Section 5.3) |
|---|---|---|
| covariates | `X ~ N(0, I_3)`, `T = (X1+X2+X3)/sqrt3` | `X ~ N(0, I_2)`, `T = X1` |
| `alpha(eta)` | `2 + eta` | `0.35 + 0.9 eta + 0.35 (eta^2 - 1)` |
| `b(eta)` | `eta^2 / 2` | `0.2 + 0.6 eta + 0.5 (eta^2 - 1)` |
| `beta1`, `beta2` | `(0.3, -0.2, 0.1)`, `(0.4, -0.25, 0.15)` | `(0.3, -0.2)`, `(0.8, 0.25)` |
| `sigma`, `c`, `eps` | 0.5, 2.25, 0.1 | 0.5, 0.25, 0.1 |
| target `phi*_eps` | 0.545103 | 0.152143 |
| `H = U''(phi*)` | -0.107534 | -0.219539 |
| first-order `n Var`: rotated, fixed, oracle | 31.09, 124.4, 1.256 | 4.73, 18.9, 0.0375 |

Both designs use the paper's windows `J`, `T` and `Phi`, so (DS1)(i) holds:
`V_Phi = [-2.78, 2.78]` lies inside `T` with a buffer. The older Section 5.3
runs used `Phi = [-3, 3]` and full-sample fits. Each target is the zero of
`U'` by adaptive quadrature on the trim window, and matches the Gauss-Legendre
value of `Population` to 1e-8. In the nonlinear design, `tau_eps` has a single
zero on `Phi` (checked on a 601-point grid).

## Results: displayed design

Columns:

- `sqrt(n) bias`: bias times `sqrt(n)`, with its Monte Carlo s.e.
- `n Var`: `n` times the Monte Carlo variance.
- `mean n Var*`: `n` times the bootstrap variance, averaged over data sets.
- `percentile`, `normal`: coverage of the two bootstrap intervals.
- `analytic (check)`: coverage of the analytic-variance check interval.

| n | estimator | sqrt(n) bias (s.e.) | n Var | mean n Var* | percentile | normal | analytic (check) | percentile width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4,000 | rotated average | +1.76 (0.27) | 36.33 | 42.21 | 0.956 | 0.950 | 0.930 | 0.3889 |
| 4,000 | pooled maximizer | +0.54 (0.24) | 29.45 | 35.52 | 0.948 | 0.968 | 0.962 | 0.3545 |
| 4,000 | oracle, average | +0.00 (0.05) | 1.18 | 1.25 | 0.954 | 0.960 | 0.968 | 0.0680 |
| 16,000 | rotated average | +0.83 (0.26) | 34.29 | 33.68 | 0.926 | 0.942 | 0.934 | 0.1758 |
| 16,000 | pooled maximizer | +0.27 (0.26) | 32.98 | 32.12 | 0.924 | 0.938 | 0.936 | 0.1717 |
| 16,000 | oracle, average | -0.00 (0.05) | 1.28 | 1.25 | 0.936 | 0.946 | 0.942 | 0.0339 |
| 64,000 | rotated average | +0.87 (0.39) | 30.82 | 31.39 | 0.950 | 0.965 | 0.955 | 0.0846 |
| 64,000 | pooled maximizer | +0.61 (0.39) | 30.35 | 31.05 | 0.955 | 0.965 | 0.960 | 0.0841 |
| 64,000 | oracle, average | +0.08 (0.08) | 1.26 | 1.26 | 0.950 | 0.945 | 0.945 | 0.0170 |

There are 500 data sets per cell at `n = 4,000` and `16,000`, and 200 at
`64,000` (added 2026-10-20). Each has 199 bootstrap draws, with no failed fits.
The coverage s.e. is about 0.010 at 500 data sets and 0.015 at 200. The fixed
assignment has `n Var` 171.3, 138.7 and 113.3 at the three sizes, against 124.4
predicted.

## Results: nonlinear design

The supplementary linear-fit arm replaces the spline outcome fit by a fit
linear in `eta` on each arm. It is outside the scope's oracle-only comparison
and is reported only for the author's decision on whether Section 5.3 keeps its
contrast.

| n | estimator | sqrt(n) bias (s.e.) | n Var | mean n Var* | percentile | normal | analytic (check) | percentile width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 800 | rotated average | -0.07 (0.30) | 43.77 | 54.23 | 1.000 | 0.950 | 0.624 | 1.0130 |
| 800 | pooled maximizer | +0.17 (0.72) | 258.91 | 347.39 | 1.000 | 0.912 | 0.630 | 2.9362 |
| 800 | oracle, average | +0.00 (0.01) | 0.04 | 0.04 | 0.926 | 0.940 | 0.946 | 0.0262 |
| 800 | linear fit, average | -1.07 (0.10) | 5.14 | 4.79 | 0.904 | 0.906 | - | 0.2956 |
| 1,600 | rotated average | -0.01 (0.17) | 13.98 | 22.37 | 0.988 | 0.962 | 0.850 | 0.4949 |
| 1,600 | pooled maximizer | -0.86 (0.43) | 91.74 | 119.82 | 0.986 | 0.974 | 0.868 | 0.8357 |
| 1,600 | oracle, average | +0.00 (0.01) | 0.03 | 0.04 | 0.952 | 0.968 | 0.964 | 0.0186 |
| 1,600 | linear fit, average | -1.31 (0.09) | 4.15 | 4.35 | 0.896 | 0.906 | - | 0.1996 |
| 3,200 | rotated average | +0.07 (0.10) | 5.49 | 6.19 | 0.946 | 0.956 | 0.920 | 0.1609 |
| 3,200 | pooled maximizer | +0.06 (0.10) | 5.32 | 8.71 | 0.946 | 0.956 | 0.922 | 0.1575 |
| 3,200 | oracle, average | +0.01 (0.01) | 0.04 | 0.04 | 0.946 | 0.958 | 0.956 | 0.0131 |
| 3,200 | linear fit, average | -2.02 (0.09) | 4.38 | 4.21 | 0.788 | 0.828 | - | 0.1391 |
| 6,400 | rotated average | +0.09 (0.10) | 5.45 | 5.21 | 0.944 | 0.938 | 0.934 | 0.1096 |
| 6,400 | pooled maximizer | +0.06 (0.10) | 5.30 | 5.12 | 0.944 | 0.940 | 0.936 | 0.1086 |
| 6,400 | oracle, average | +0.01 (0.01) | 0.03 | 0.04 | 0.948 | 0.954 | 0.958 | 0.0093 |
| 6,400 | linear fit, average | -2.81 (0.09) | 4.26 | 4.10 | 0.692 | 0.712 | - | 0.0970 |
| 12,800 | rotated average | -0.21 (0.14) | 5.63 | 4.71 | 0.920 | 0.943 | 0.943 | 0.0735 |
| 12,800 | pooled maximizer | -0.23 (0.14) | 5.61 | 4.67 | 0.920 | 0.933 | 0.933 | 0.0731 |
| 12,800 | oracle, average | -0.01 (0.01) | 0.03 | 0.04 | 0.943 | 0.953 | 0.963 | 0.0066 |
| 12,800 | linear fit, average | -4.55 (0.12) | 4.49 | 3.75 | 0.337 | 0.353 | - | 0.0655 |

There are 500 data sets per cell, except 300 at `n = 12,800`, where the coverage
s.e. is about 0.013. The run shared the CPU with another session's job and was
stopped at 300. Each data set has 199 draws, with no failed fits.

## Reading

**Displayed design.**

- **Oracle arm.** The oracle has coverage within Monte Carlo noise of 95% for
  every interval type, and `n Var` 1.18 to 1.28 against 1.256.
- **Bias.** The rotated average has `sqrt(n)` bias +1.76 at 4,000, +0.83 at
  16,000 and +0.87 (s.e. 0.39) at 64,000. From 4,000 to 16,000 it halves, as a
  bias of order `1/n` should. The 16,000 value matches the paper's Table 1
  (0.89, 2,000 data sets). The 64,000 value is within 1.2 s.e. of Table 1's
  0.41. The pooled maximizer's bias is smaller (0.54, 0.27, 0.61).
- **Bootstrap variance.** It is 16% above the Monte Carlo variance at 4,000, 2%
  below at 16,000 and 2% above at 64,000. At 64,000 the Monte Carlo `n Var`
  is 30.82, against 31.09 predicted.
- **Normal intervals.** Coverage 0.950, 0.942 and 0.965 for the rotated average.
- **Percentile intervals.** Coverage 0.956, 0.926 and 0.950. Most of the shortfall at
  16,000 comes from using 199 draws. With numpy's default interpolation, the
  2.5% and 97.5% points of 199 normal draws are on average 1.915 s.d. from the
  centre. So even an exact normal bootstrap has percentile coverage 0.940, and
  normal-interval coverage 0.948 (200,000 simulated intervals). 0.926 is about
  1.4 Monte Carlo s.e. below 0.940, and the oracle shows the same pattern
  (0.936 against 0.946).
- **Analytic check.** The interval with the first-order variance covers 0.930,
  0.934 and 0.955. It ignores the bias, and at the two smaller `n` the Monte
  Carlo variance is above 31.09.

**Nonlinear design.**

- **Small samples (`n` = 800, 1,600).** The four-role spline estimator is far
  from its first-order behaviour: `n Var` 43.8 and 14.0 against 4.73. Each
  nuisance is fitted on a quarter of the sample; the outcome fold at `n = 800`
  has 200 observations. The pooled maximizer lands on the edge of `Phi` in 12%
  of data sets at 800 and in 1.8% at 1,600. The bootstrap intervals are wide
  and over-cover: percentile 1.000 and 0.988, normal 0.950 and 0.962.
- **Larger samples (`n` = 3,200 to 12,800).**
  - The bias is within 0.21 `sqrt(n)` units, at most 0.002 on the threshold
    scale.
  - Percentile coverage is 0.946, 0.944 and 0.920; normal coverage 0.956,
    0.938 and 0.943.
  - `n Var` is 5.45 to 5.63, 15-19% above the first-order 4.73 (Monte Carlo
    s.e. 6-8%). The bootstrap variance moves from above the Monte Carlo
    variance to below it, at 4.71 at 12,800.
  - It is not settled whether the gap closes at larger `n` or whether the
    analytic value is off for this design. The analytic value has not been
    checked against a long run here, unlike the displayed design.
- **Comparison with the old Section 5.3 table.** The old full-sample spline
  fit covered 0.94-0.96 from `n = 800`. The four-role estimator needs about
  `n = 3,200` in this design.
- **Linear-fit arm (supplementary).** It has a stable pseudo-target bias of
  about -0.033 to -0.040 on the threshold scale. Its coverage falls from 0.90
  at 800 to 0.69 at 6,400 (the old full-sample table had 0.68) and 0.34 at
  12,800.

## Bootstrap scheme check

A full re-estimation bootstrap copies some observations into two roles, which
the four-role estimator never does. The diagnostic keeps each data set's own
split and resamples within each block (`--scheme within`). It ran on the first
40 displayed-design data sets at `n = 4,000`. On the same data sets, the within-block bootstrap gives a larger
`mean n Var*` than the main scheme. For the rotated average it is 53.3
against 45.7 (paired difference 7.6, s.e. 1.7), and the Monte Carlo value is
36.3 (500 data sets). For the pooled maximizer it is 60.8 against 37.5 (s.e.
7.0); for the oracle, 1.23 against 1.25. So copies across roles do not
explain why the bootstrap overstates the variance at `n = 4,000`, and keeping
the split does not help. The main scheme stays.

## Population targets: Gauss-Hermite replaced

`experiments/methods/trimmed_quadrature.py` adds `trimmed_normal_mean`:
adaptive Gauss-Kronrod quadrature (`scipy.integrate.quad`) on the trim window.
It now computes the population utility in:

- `differing_slopes_simulation.py`;
- `differing_slopes_full_pipeline.py`;
- `differing_slopes_nonlinear_outcome.py`, whose target
  `differing_slopes_nonlinear_bootstrap.py` imports.

The new targets:

| scenario (script) | Gauss-Hermite, trim at nodes | quadrature on the window |
|---|---:|---:|
| four-role check design (`simulation` baseline and error-law scenarios; `full_pipeline` baseline, clustered; `nonlinear_outcome` linear) | -0.1200638 | -0.1197699 |
| check design with `beta2 = 0`; alpha-only pseudo-target (`simulation`) | -0.3392919 | -0.3192384 |
| strong interaction (`simulation`) | -0.0730099 | -0.0737808 |
| quadratic misspecification (`full_pipeline`) | 0.4954968 | 0.4844159 |
| weak curvature (`full_pipeline`) | -0.5000000 | -0.5000000 (exact) |
| upper boundary (`full_pipeline`) | 3 (edge) | 3 (edge) |
| nonlinear `alpha, b` (`nonlinear_outcome`) | 0.1593373 | 0.1521433 |

The values agree with `ds_identification_check.py` and with the corrections
already in `this_week.tex` and the paper. The weak-curvature target is exactly
-0.5, because there `tau_eps(phi) = 0.1 + 0.2 phi`. Saved runs under
`experiments/runs/` were not recomputed.

## Reproduce

```sh
cd code
PYTHONPATH=. python -m experiments.scripts.ds_four_role_coverage --design displayed \
  --n 4000 16000 --reps 500 --boot 199 --workers 7
PYTHONPATH=. python -m experiments.scripts.ds_four_role_coverage --design displayed \
  --n 64000 --reps 200 --boot 199 --workers 7
PYTHONPATH=. python -m experiments.scripts.ds_four_role_coverage --design nonlinear \
  --n 800 1600 3200 6400 --reps 500 --boot 199 --workers 7
PYTHONPATH=. python -m experiments.scripts.ds_four_role_coverage --design nonlinear \
  --n 12800 --reps 300 --boot 199 --workers 7
PYTHONPATH=. python -m experiments.scripts.ds_four_role_coverage --design displayed \
  --n 4000 --reps 40 --boot 199 --workers 1 --scheme within
PYTHONPATH=. python -m experiments.scripts.ds_four_role_coverage --design nonlinear --report
```

The per-data-set rows, with every bootstrap draw, are in the ignored
`experiments/runs/ds_four_role_coverage/` (`<design>_n<n>.jsonl`), with
`<design>_summary.json`. Tests:
`experiments/tests/test_ds_four_role_coverage.py`.

## Limits

- Simulation evidence only; there is no bootstrap validity theorem.
- Spline sizes follow the fixed exponent rule, not the constrained
  cross-validation that is still open on the task board.
- The density basis is the full clamped cubic basis, as in Section 5.1.

-- Claude, 2026-10-19
