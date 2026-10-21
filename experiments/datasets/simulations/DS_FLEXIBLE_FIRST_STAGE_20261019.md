# Flexible first stages under the hard trim (2026-10-19)

Script: `experiments/scripts/ds_flexible_first_stage_check.py` (test: `experiments/tests/test_ds_flexible_first_stage.py`). Raw summaries: `experiments/runs/ds_flexible_first_stage/<tag>/summary.json` (git-ignored). Context: `RESEARCH_LOG.md` 2026-10-20; parked as future work on the task board.

Design: the Gaussian common-slope baseline (alpha = 2 + eta, hard trim at eps = 0.10, phi* = 0.7313), with the index either spread over three covariates (`spread`, gamma = (1,1,1)/sqrt(3), the baseline) or carried by one (`single`, gamma = e1). T ~ N(0,1) in both, so the linear-first-stage prediction is 43.68. The true first stage is linear, so every spline family contains it and only the variance changes with K. Estimator: shared-training plug-in, 4-fold cross-fit (`_cf`) and one fold (`_one`). First stages: `linear` (OLS), `additive` (cubic B-splines in each covariate, K interior knots), `index` (cubic B-splines in the OLS pilot index), `mlp` (one tanh hidden layer of width K). N Var = N x Var; Monte Carlo SE of N Var is about sqrt(2/R) of its value.

Prediction from the trim-point terms (index-type first stage): N Var rises over the linear value by about 3.4, 8.6, 18.1 and 22.6 at K = 4, 8, 16, 32. Additive splines in the `spread` design should stay flat in K.

## Spread index, spline first stages (300 reps)

```
python -m experiments.scripts.ds_flexible_first_stage_check --design spread --n 10000 40000 --reps 300 --seed 20261019 --configs linear:0 additive:4 additive:8 additive:16 additive:32 index:4 index:8 index:16 index:32 --tag spread_splines_20261019
```

| First stage | N=10000 N Var | N=10000 sqrt(N) bias (se) | N=40000 N Var | N=40000 sqrt(N) bias (se) |
|---|---|---|---|---|
| `linear_K0` cross-fit | 34.1 | -0.32 (0.34) | 38.6 | -0.20 (0.36) |
| `additive_K4` cross-fit | 39.3 | -0.87 (0.36) | 46.2 | -0.08 (0.39) |
| `additive_K8` cross-fit | 39.6 | -1.41 (0.36) | 46.6 | -0.32 (0.39) |
| `additive_K16` cross-fit | 39.4 | -2.72 (0.36) | 46.8 | -0.90 (0.40) |
| `additive_K32` cross-fit | 39.3 | -5.11 (0.36) | 46.3 | -2.15 (0.39) |
| `index_K4` cross-fit | 84.6 | +0.51 (0.53) | 74.8 | +0.80 (0.50) |
| `index_K8` cross-fit | 96.4 | +1.30 (0.57) | 88.0 | +0.97 (0.54) |
| `index_K16` cross-fit | 100.3 | +1.26 (0.58) | 96.3 | +1.14 (0.57) |
| `index_K32` cross-fit | 105.2 | +1.08 (0.59) | 103.9 | +1.01 (0.59) |

## Single-covariate index, additive splines (300 reps)

```
python -m experiments.scripts.ds_flexible_first_stage_check --design single --n 10000 40000 --reps 300 --seed 20261019 --configs linear:0 additive:4 additive:8 additive:16 additive:32 --tag single_splines_20261019
```

| First stage | N=10000 N Var | N=10000 sqrt(N) bias (se) | N=40000 N Var | N=40000 sqrt(N) bias (se) |
|---|---|---|---|---|
| `linear_K0` cross-fit | 39.7 | +0.13 (0.36) | 44.0 | -0.13 (0.38) |
| `additive_K4` cross-fit | 79.9 | -0.50 (0.52) | 86.7 | +0.13 (0.54) |
| `additive_K8` cross-fit | 89.1 | -0.60 (0.55) | 101.2 | +0.56 (0.58) |
| `additive_K16` cross-fit | 90.7 | -2.02 (0.55) | 107.5 | +0.00 (0.60) |
| `additive_K32` cross-fit | 90.6 | -4.27 (0.55) | 105.0 | -1.32 (0.59) |

## Spread index, neural-network first stages (150 reps)

```
python -m experiments.scripts.ds_flexible_first_stage_check --design spread --n 40000 --reps 150 --seed 20261019 --configs linear:0 mlp:4 mlp:16 mlp:64 --tag spread_mlp_20261019
```

| First stage | N=40000 N Var | N=40000 sqrt(N) bias (se) |
|---|---|---|
| `linear_K0` cross-fit | 42.0 | -0.45 (0.53) |
| `mlp_K4` cross-fit | 75.5 | -2.80 (0.71) |
| `mlp_K16` cross-fit | 74.1 | -2.10 (0.70) |
| `mlp_K64` cross-fit | 63.3 | -0.57 (0.65) |

## Reading

- Spread index: additive splines are flat in K after a one-time rise over linear (the variance cost of the larger first-stage model). Index splines keep growing.
- Single-covariate index: additive splines are index splines, and N Var grows.
- Growth from K = 4 to 32 is +29 (spread, index) and +18 (single, additive) at N = 40,000, against +19 predicted from the trim-point terms.
- Additive splines at K = 32 carry a bias that shrinks like K/n (sqrt(N) bias -5.11 at N = 10,000, -2.15 at N = 40,000): own-observation bias from fitting the first stage on the shared training sample.
- Neural networks: N Var 63 to 76 against 42 for linear, not monotone in width, with some bias at small widths; inconclusive.
