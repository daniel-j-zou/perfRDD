# Debiased evaluation score against the plug-in (2026-10-13)

Script: `experiments/scripts/debiased_score_check.py`. Theory: `manuscript/this_week.tex`, section "A debiased evaluation score". Raw summaries: `experiments/runs/debiased_score_check/<tag>/summary.json` (git-ignored).

Designs share the Gaussian common-slope DGP: X ~ N(0, I), eta ~ N(0, 1) independent, T = gamma'X, Y = D alpha(eta) + 0.5 eta^2 + X'beta + noise. Baseline alpha = 2 + eta (phi* = 0.7313, sigma^2/H^2 = 43.68); nonlinear alpha = 2 + eta + 0.5 sin(2 eta) (phi* = 0.4608). `pi*` is the plug-in, `dr*` the debiased estimator; `_one` is one evaluation fold, `_cf` the cross-fitted average; `pi4_fixed`/`pi4_rot` are the paper's four-role plug-in. The debiased estimator takes Newton steps on the debiased score from the shared-training plug-in, with the plug-in curvature; its first-stage loading is a central difference in gamma. N Var = N x Var(phi_hat), N the total sample size. The Monte Carlo SE of N Var is about sqrt(2/R) of its value.

## Variances, baseline

```
python -m experiments.scripts.debiased_score_check --design baseline --n 20000 40000 --reps 500 --seed 20261013 --estimators pi4_fixed pi4_rot pi2_one dr2_one pi2_cf dr2_cf pi4f_one dr4f_one pi4f_cf dr4f_cf --rep-mult 3 --tag baseline_20261013
```

| Estimator | Prediction | N=20000 N Var | N=20000 sqrt(N) bias (se) | N=40000 N Var | N=40000 sqrt(N) bias (se) |
|---|---|---|---|---|---|
| `pi4_fixed` | 174.7 | 181.6 | +1.69 (0.60) | 187.7 | +1.00 (0.61) |
| `pi4_rot` | 43.7 | 40.1 | +0.76 (0.28) | 40.9 | -0.08 (0.29) |
| `pi2_one` | 87.3 | 82.3 | +0.89 (0.41) | 81.4 | +0.20 (0.40) |
| `dr2_one` | 87.3 | 94.4 | +0.01 (0.43) | 86.9 | -0.60 (0.42) |
| `pi2_cf` | 43.7 | 39.9 | +0.65 (0.28) | 39.7 | -0.11 (0.28) |
| `dr2_cf` | 43.7 | 41.9 | +0.18 (0.29) | 42.3 | -0.28 (0.29) |
| `pi4f_one` | 67.7 | 59.5 | +0.33 (0.34) | 63.4 | -0.11 (0.36) |
| `dr4f_one` | 174.7 | 181.5 | +0.95 (0.60) | 164.3 | -1.02 (0.57) |
| `pi4f_cf` | 43.7 | 40.0 | +0.57 (0.28) | 40.6 | -0.07 (0.29) |
| `dr4f_cf` | 43.7 | 40.7 | +0.37 (0.29) | 41.1 | -0.26 (0.29) |

## Coarse outcome fit (no interior knots), nonlinear design

```
python -m experiments.scripts.debiased_score_check --design nonlinear --n 20000 80000 --reps 500 --seed 20261013 --estimators pi4f_cf dr4f_cf --rep-mult 0 1 4 12 --outcome-knots 0 --tag nonlinear_coarse_20261013
```

| Estimator | N=20000 N Var | N=20000 sqrt(N) bias (se) | N=80000 N Var | N=80000 sqrt(N) bias (se) |
|---|---|---|---|---|
| `dr4f_cf_exact` | 18.5 | +0.21 (0.19) | 19.9 | -0.16 (0.20) |
| `dr4f_cf_m1` | 18.9 | +0.18 (0.19) | 19.9 | -0.26 (0.20) |
| `dr4f_cf_m12` | 18.6 | +0.22 (0.19) | 19.8 | -0.15 (0.20) |
| `dr4f_cf_m4` | 18.6 | +0.23 (0.19) | 19.8 | -0.16 (0.20) |
| `dr4f_one_exact` | 82.6 | +0.20 (0.41) | 86.4 | +0.02 (0.42) |
| `dr4f_one_m1` | 82.1 | +0.14 (0.41) | 85.2 | -0.14 (0.41) |
| `dr4f_one_m12` | 82.9 | +0.22 (0.41) | 86.6 | +0.03 (0.42) |
| `dr4f_one_m4` | 82.6 | +0.19 (0.41) | 87.1 | +0.00 (0.42) |
| `pi4f_cf` | 19.1 | +0.32 (0.20) | 19.9 | -0.23 (0.20) |
| `pi4f_one` | 33.1 | +0.42 (0.26) | 33.5 | -0.22 (0.26) |

## Default outcome knots, nonlinear design

```
python -m experiments.scripts.debiased_score_check --design nonlinear --n 20000 80000 --reps 500 --seed 20261013 --estimators pi4f_cf dr4f_cf --rep-mult 0 1 --tag nonlinear_default_20261013
```

| Estimator | N=20000 N Var | N=20000 sqrt(N) bias (se) | N=80000 N Var | N=80000 sqrt(N) bias (se) |
|---|---|---|---|---|
| `dr4f_cf_exact` | 18.5 | +0.22 (0.19) | 20.0 | -0.15 (0.20) |
| `dr4f_cf_m1` | 18.4 | +0.22 (0.19) | 19.9 | -0.13 (0.20) |
| `dr4f_one_exact` | 81.4 | +0.20 (0.40) | 85.2 | +0.03 (0.41) |
| `dr4f_one_m1` | 81.2 | +0.17 (0.40) | 85.3 | +0.06 (0.41) |
| `pi4f_cf` | 18.7 | +0.36 (0.19) | 20.0 | -0.11 (0.20) |
| `pi4f_one` | 32.8 | +0.46 (0.26) | 33.8 | -0.12 (0.26) |

## Coarse density basis L = 6 for g, baseline

```
python -m experiments.scripts.debiased_score_check --design baseline --n 20000 80000 --reps 300 --seed 20261013 --estimators pi4f_cf dr4f_cf --rep-mult 0 --density-basis 6 --iterations 3 --tag baseline_coarse_density_L6_20261013
```

| Estimator | N=20000 N Var | N=20000 sqrt(N) bias (se) | N=80000 N Var | N=80000 sqrt(N) bias (se) |
|---|---|---|---|---|
| `dr4f_cf` | 37.2 | +0.65 (0.35) | 45.3 | +0.00 (0.39) |
| `dr4f_one` | 158.3 | -0.07 (0.73) | 174.3 | +0.23 (0.76) |
| `pi4f_cf` | 32.2 | +18.23 (0.33) | 40.6 | +35.67 (0.37) |
| `pi4f_one` | 53.6 | +18.56 (0.42) | 60.8 | +35.56 (0.45) |

## Coarse density basis L = 8 for g, baseline

```
python -m experiments.scripts.debiased_score_check --design baseline --n 20000 80000 --reps 300 --seed 20261013 --estimators pi4f_cf dr4f_cf --rep-mult 0 --density-basis 8 --iterations 3 --tag baseline_coarse_density_L8_20261013
```

| Estimator | N=20000 N Var | N=20000 sqrt(N) bias (se) | N=80000 N Var | N=80000 sqrt(N) bias (se) |
|---|---|---|---|---|
| `dr4f_cf` | 37.1 | +0.66 (0.35) | 44.9 | -0.02 (0.39) |
| `dr4f_one` | 160.7 | -0.01 (0.73) | 176.0 | +0.21 (0.77) |
| `pi4f_cf` | 34.9 | +2.38 (0.34) | 42.8 | +3.67 (0.38) |
| `pi4f_one` | 57.4 | +2.75 (0.44) | 64.3 | +3.56 (0.46) |
