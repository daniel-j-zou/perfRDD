# Sensitivity of the target and the four-role estimator to eps (2026-10-20, Claude)

At the author's request (talk about how the estimator changes with eps, in the
simulations and the taxi data). In the paper as `differing_slopes.tex` §5.3, Table 3.
Script: `experiments/scripts/ds_eps_sensitivity.py`; results in the ignored
`experiments/runs/ds_eps_sensitivity/` (`summary.json`, `rows_n*.json`).

## Design

The displayed design of §5 (`paper_baseline`), with two changes so that (DS1)(i)
holds at every eps in the grid 0.05-0.25:
- policy set Phi = [-1, 1.5] instead of [-1.5, 1.5];
- density interval T = [-3.5, 3.5] instead of [-3, 3].

At eps = 0.05, [l0, u0] = [-1.645, 1.645] sits inside J = [-1.75, 1.75], and
V_Phi = [-2.645, 3.145] inside T. Each data set gets one four-role split and one set
of fits (first stage, outcome fit on J, density fits); only the trim endpoints and
the evaluation trim change with eps. Estimator: the average of the four rotated
thresholds, 61-point grid plus bounded refinement. 2000 data sets at n = 16,000,
1000 at n = 64,000; seed 20261020.

## Results

Population values by adaptive quadrature (`Population.decomposition`); V is the
first-order n-scaled variance of the rotated estimators.

| eps | u0 | phi*_eps | -H | V | sqrt(n) bias, 16k | n Var, 16k | sqrt(n) bias, 64k | n Var, 64k |
|---:|---:|---:|---:|---:|---|---:|---|---:|
| 0.05 | 1.645 | 0.470 | 0.137 | 20.1 | 0.52 (0.10) | 21.6 | 0.13 (0.14) | 19.2 |
| 0.075 | 1.440 | 0.505 | 0.122 | 24.8 | 0.57 (0.12) | 26.7 | 0.16 (0.15) | 23.6 |
| 0.10 | 1.282 | 0.545 | 0.108 | 31.1 | 0.67 (0.13) | 34.2 | 0.25 (0.17) | 29.8 |
| 0.125 | 1.150 | 0.590 | 0.093 | 39.5 | 0.90 (0.15) | 44.0 | 0.49 (0.19) | 37.8 |
| 0.15 | 1.036 | 0.640 | 0.080 | 50.7 | 1.29 (0.17) | 57.7 | 0.69 (0.22) | 48.7 |
| 0.20 | 0.842 | 0.755 | 0.056 | 84.9 | 2.52 (0.22) | 98.4 | 0.99 (0.28) | 81.0 |
| 0.25 | 0.674 | 0.886 | 0.037 | 143.3 | 3.76 (0.28) | 153.1 | 1.24 (0.37) | 136.9 |

- **Target.** phi*_eps rises from 0.470 to 0.886, away from the untrimmed 0.4262.
  The central difference at 0.10, (0.5903 - 0.5050)/0.05 = 1.706, matches
  Corollary C2's 1.707.
- **Precision.** -H falls fourfold and V rises sevenfold over the grid. At the
  optimum H = -p_eps tau_eps', and a narrower window lowers p_eps. Here a smaller
  eps is better on both counts.
- **Variance.** n Var / V is 1.07-1.16 at 16k and 0.95-0.96 at 64k at every eps
  (Monte Carlo relative se about 3% and 4.5%). A fixed assignment has 1.02-1.08 times
  V_fixed = 4V at 64k.
- **Bias.** Grows with eps and falls with n; at 64k at most 1.24 (0.11 sd).
- **Edge of Phi.** At 16k and eps = 0.25, 5.4% of data sets have a rotation at the
  grid edge (0.55% at eps = 0.20); none at 64k.
- **Paired steps** (same data and fits, consecutive eps): the mean step tracks the
  target step, and its sd is about a fifth of the level's sd.

| step | target | mean, 16k | sd, 16k | mean, 64k | sd, 64k |
|---|---:|---:|---:|---:|---:|
| 0.05 -> 0.075 | 0.0347 | 0.0351 | 0.0069 | 0.0348 | 0.0033 |
| 0.075 -> 0.10 | 0.0401 | 0.0409 | 0.0080 | 0.0405 | 0.0038 |
| 0.10 -> 0.125 | 0.0451 | 0.0470 | 0.0092 | 0.0461 | 0.0042 |
| 0.125 -> 0.15 | 0.0500 | 0.0531 | 0.0106 | 0.0508 | 0.0048 |
| 0.15 -> 0.20 | 0.1143 | 0.1240 | 0.0231 | 0.1155 | 0.0106 |
| 0.20 -> 0.25 | 0.1319 | 0.1416 | 0.0271 | 0.1329 | 0.0135 |

Level sd at eps = 0.10: 0.046 (16k) and 0.022 (64k).

The taxi counterpart is `../taxi/EPS_SENSITIVITY_20261020.md`.

```bash
python -m experiments.scripts.ds_eps_sensitivity --n 16000 64000 --reps 2000 1000 --workers 6   # about 25 min
```
