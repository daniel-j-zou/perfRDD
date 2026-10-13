# Local efficiency of the four-role DS estimator at Gaussian eta (2026-10-13, Claude)

Theory: `manuscript/this_week.tex`, section "Local efficiency at Gaussian residuals".
Scripts (all in `code/experiments/scripts/`):

- `ds_local_efficiency.py` computes the population quantities by closed form and
  one-dimensional quadrature for Gaussian designs: phi*, H, the role-score
  variances, r_out arm by arm, Abar, J_Y, I_gamma, V_OLS and V_eff.
  `--mc-check` adds an independent least-squares projection check.
  `--sensitivity` varies the paper design one feature at a time.
- `ds_local_efficiency_sim.py` runs the four-role estimator (spline outcome fit on J,
  spline g and rho_X, quantile endpoints, average of four rotated thresholds) with the
  OLS first stage and with the efficient (stacked) first stage, on the same data and folds.
- `mbr_efficiency_check.py` checks the conditions behind Mukherjee et al.'s Theorem 2.10.

Environment: `/Users/zoudj/miniconda3/envs/srsd/bin/python`, run from `code/`.

## Designs

- **paper_baseline**: the displayed design of `differing_slopes.tex` Section 5.
  - X ~ N(0, I_3), gamma = (0, 1, 1, 1)/sqrt(3), eta ~ N(0, 1), D = 1{Q > 0}.
  - a(eta) = 2 + eta, b(eta) = eta^2/2.
  - beta1 = (0.3, -0.2, 0.1), beta2 = (0.4, -0.25, 0.15).
  - sigma_0 = sigma_1 = 0.5, c = 2.25, eps = 0.1.
  - J = [-1.75, 1.75], T support [-3, 3], Phi = [-1.5, 1.5].
- **paper_lownoise**: paper_baseline with sigma_0 = sigma_1 = 0.25.
- **paper_hetero**: paper_baseline with sigma_d^2(eta) = 0.25 exp(kappa eta), kappa = 1.
- **ds_linear**: the earlier four-role DS check design.
  - X ~ N(0, I_2), T = X1, eta ~ N(0, 1).
  - b = 0.2 + 0.6 eta, a = 0.35 + 0.9 eta.
  - beta1 = (0.3, -0.2), beta2 = (0.8, 0.25).
  - sigma = 0.5, c = 0.25, J = [-2, 2], Phi = [-3, 3].

## Population results (`python -m experiments.scripts.ds_local_efficiency --mc-check`)

| design | phi* | H | Var psi_U | Var psi_o | Var psi_T | first stage OLS | first stage eff | V_OLS | V_eff |
|---|---|---|---|---|---|---|---|---|---|
| paper_baseline | 0.545103 | -0.107534 | 0.01452 | 0.21814 | 0.04339 | 0.08345 | 0.04793 | 31.089 | 28.018 |
| ds_linear | -0.119770 | -0.218030 | 0.00025 | 0.21062 | 0.00593 | 0.000028 | 0.000028 | 4.561 | 4.561 |

- **Shares (paper_baseline).** The OLS first stage is 23.2% of sigma_DS^2. The efficiency
  gain is 9.88%.
- **Information about the slopes.** diag J_Y = 1.432 against the score-equation
  information Var(X)/sigma_eta^2 = 1. The efficient slope variance is 0.432 times the
  OLS variance.
- **Intercept of Abar.** -7e-18 and -2e-17 (location invariance).
- **Cross-check at beta2 = 0.** The paper design gives phi* = 0.731292, H = -0.0720,
  b_l = -0.0141, b_u = 0.0621, Var psi_U = 0.0185 and Var psi_T = 0.0363. These match the
  independent common-slope decomposition of 2026-10-01.
- **MC projection check (4e5 draws, generic spline bases, no closed forms).**
  - paper_baseline: diag I_gamma 2.404/2.441/2.414 against 2.432, and
    Var(Pi psi_gamma) 0.0473 against 0.0479.
  - ds_linear: diag I_gamma 0.989/0.993 against 1.
- **ds_linear target.** phi* = -0.1197699. Adaptive quadrature agrees. The value
  -0.120064 from `differing_slopes_full_pipeline.population_truth` is a Gauss-Hermite
  artifact of the hard trim indicator.

## Sensitivity (paper design, one change at a time; `--sensitivity`)

| change | V_OLS | V_eff | gain | first-stage share | slope variance ratio |
|---|---|---|---|---|---|
| none | 31.09 | 28.02 | 9.9% | 23.2% | 0.43 |
| sigma = 0.25 | 16.94 | 11.55 | 31.8% | 42.6% | 0.17 |
| sigma = 1 | 87.68 | 86.55 | 1.3% | 8.2% | 0.74 |
| J = +-1.5 | 31.12 | 28.85 | 7.3% | 20.0% | 0.49 |
| J = +-2.5 | 31.78 | 26.93 | 15.3% | 28.6% | 0.35 |
| J = +-3.5 | 32.14 | 26.65 | 17.1% | 30.0% | 0.32 |
| b = 0 | 24.27 | 24.27 | 0.0% | 1.6% | 1.00 |
| b = eta^2/2 + eta^3/6 | 27.25 | 25.51 | 6.4% | 12.4% | 0.37 |
| a = 2 + eta + 0.3 eta^2 | 21.31 | 20.55 | 3.6% | 6.3% | 0.31 |
| beta2 = 0 | 78.35 | 67.15 | 14.3% | 33.9% | 0.43 |

At beta2 = 0 the common-slope estimator (no DX block) has rotated variance 43.68 and a
first-stage share of 0.8%. The differing-slopes estimator, in the same design, has 78.35
and 33.9%.

## Non-Gaussian eta (variance 1; `ds_local_efficiency.py` designs paper_t5, paper_mix)

| eta | phi* | H | I_eta Var eta | V OLS | V stacked (LS rows) | V adaptive = bound |
|---|---|---|---|---|---|---|
| Gaussian | 0.5451 | -0.1075 | 1.000 | 31.09 | 28.02 | 28.02 |
| t5 | 0.5960 | -0.1038 | 1.250 | 36.04 | 32.71 | 32.06 |
| mixture, lam = 0.8 | 0.4705 | -0.1202 | 1.462 | 23.58 | 21.45 | 20.90 |

The intercept of Abar is zero in both designs.

## Within-arm heteroskedasticity (`python -m experiments.scripts.ds_local_efficiency_hetero`)

The paper design with sigma_d^2(eta) = 0.25 exp(kappa eta). "Stacked" uses arm-constant
weights.

| kappa | rotated paper | rotated stacked | bound | fixed paper | fixed stacked | 2 Cov(psi_o, psi_gamma^w) | stacked slope var |
|---|---|---|---|---|---|---|---|
| 0 | 31.09 | 28.02 | 28.02 | 124.4 | 112.1 | 0 | 0.432 |
| 0.5 | 31.35 | 30.75 | 29.87 | 125.4 | 113.9 | 0.026 | 0.465 |
| 1 | 34.65 | 36.79 | 33.58 | 138.6 | 129.4 | 0.051 | 0.565 |
| 1.5 | 41.76 | 46.82 | 39.59 | 167.0 | 161.1 | 0.076 | 0.727 |

The stacked first stage still beats OLS for the slopes and in a fixed assignment. The
rotated average loses through the within-observation covariance between the outcome score
and the outcome rows of the first-stage score, which is zero only under (H).

Weighted estimators (weights 1/sigma_D^2(eta)), rotated:

| kappa | paper | w_ols (weighted outcome, OLS first stage) | w_stacked (both weighted) = bound |
|---|---|---|---|
| 0 | 31.09 | 31.09 | 28.02 |
| 0.5 | 31.35 | 34.90 | 29.87 |
| 1 | 34.65 | 41.18 | 33.58 |
| 1.5 | 41.76 | 50.55 | 39.59 |

Weighting only the outcome fit hurts, because the weighted outcome loading is larger:
Abar^w_X = (-0.275, -0.220, -0.254) at kappa = 1, against (-0.191, -0.135, -0.169)
unweighted. The identity Abar^w_X = Abar_X + E[S_J r_out R_xi^w] was checked numerically
(equal to 5 decimals).

## Simulation (`python -m experiments.scripts.ds_local_efficiency_sim --design <d> --n 16000 32000 --reps 1000 --workers 8`)

1000 replications per cell, so each N Var has a Monte Carlo s.e. of about 4.5%. Both
first stages run on the same data and folds. "Slopes" is m Var of the first-stage
slopes on one fold of size m = N/4, averaged over coordinates.

**paper_baseline**

| N | rotated OLS | rotated stacked | paired diff | fixed OLS | fixed stacked | slopes OLS | slopes stacked | bias OLS / stacked |
|---|---|---|---|---|---|---|---|---|
| 16000 | 33.27 | 29.92 | 3.36 | 140.38 | 129.54 | 1.017 | 0.506 | +0.0067 / +0.0059 |
| 32000 | 30.49 | 27.94 | 2.55 | 119.39 | 106.75 | 1.007 | 0.474 | +0.0029 / +0.0026 |
| prediction | 31.09 | 28.02 | 3.07 | 124.36 | 112.07 | 1.000 | 0.432 | 0 |

The rotated OLS and stacked estimators have correlation 0.94. The paired difference
has a Monte Carlo s.e. of about 0.6.

**ds_linear**

| N | rotated OLS | rotated stacked | paired diff | fixed OLS | fixed stacked | slopes OLS | slopes stacked |
|---|---|---|---|---|---|---|---|
| 16000 | 4.61 | 4.62 | -0.01 | 18.50 | 18.50 | 1.023 | 1.160 |
| 32000 | 4.84 | 4.83 | 0.01 | 18.25 | 18.22 | 1.035 | 1.115 |
| prediction | 4.56 | 4.56 | 0 | 18.24 | 18.24 | 1.000 | 1.000 |

Here J_Y = 0. The stacked first stage is asymptotically equivalent to OLS, but its
slopes are noisier in finite samples, because the outcome rows carry spurious
information through the estimated c_d'. The threshold is unaffected because its
first-stage loading is negligible in this design.

**paper_lownoise** (sigma = 0.25; `ds_local_efficiency_sim.py`, N = 32000, 1000 reps)

| | rotated OLS | rotated stacked | paired diff | fixed OLS | fixed stacked | slopes OLS | slopes stacked |
|---|---|---|---|---|---|---|---|
| simulation | 16.51 | 11.45 | 5.06 | 65.37 | 43.40 | 1.007 | 0.178 |
| prediction | 16.94 | 11.55 | 5.39 | 67.76 | 46.20 | 1.000 | 0.167 |

The OLS and stacked estimators have correlation 0.81.

**paper_hetero** (kappa = 1; `ds_local_efficiency_weighted_sim.py`, N = 32000, 1000 reps;
`--oracle` adds true-variance weights; the same seeds, so the paper and fitted rows
coincide across the two runs)

| estimator | rotated N Var | prediction | fixed | bias |
|---|---|---|---|---|
| paper (OLS first stage) | 34.96 | 34.65 | 134.6 | +0.0023 |
| stacked, unweighted | 37.60 | 36.79 | 128.7 | +0.0022 |
| weighted outcome, OLS first stage | 42.16 | 41.18 | 157.3 | +0.0031 |
| both weighted, fitted variance (6-function spline, same fold) | 36.85 | 33.58 | 139.6 | +0.0026 |
| both weighted, true variance | 34.25 | 33.58 | 126.0 | +0.0023 |

Paired (paper minus variant):
- stacked: -2.63 (prediction -2.14);
- weighted outcome with OLS: -7.20 (prediction -6.53);
- both weighted with fitted variance: -1.89;
- both weighted with true variance: +0.71 (prediction +1.07).

The theory holds with the true weights. With fitted weights, the cost of estimating a
variance that spans a 33-fold range on J (about 8%) exceeds the 3% gain at this N.

## First-stage convergence (`python -m experiments.scripts.ds_local_efficiency_fs_check --m 4000 16000 64000 --reps 1000`)

m Var of the slopes on a sample of size m, averaged over coordinates; limits in parentheses.

| eta | m | OLS | stacked (LS rows) | adaptive |
|---|---|---|---|---|
| Gaussian | 4000 | 1.022 | 0.495 | 0.514 |
| Gaussian | 16000 | 0.951 | 0.456 | 0.466 |
| Gaussian | 64000 | 1.020 | 0.428 | 0.435 |
| Gaussian | limit | (1.000) | (0.432) | (0.432) |
| t5 | 4000 | 1.007 | 0.528 | 0.479 |
| t5 | 16000 | 1.016 | 0.477 | 0.433 |
| t5 | 64000 | 0.953 | 0.443 | 0.401 |
| t5 | limit | (1.000) | (0.463) | (0.413) |
| mixture | 4000 | 0.975 | 0.457 | 0.408 |
| mixture | 16000 | 0.968 | 0.413 | 0.363 |
| mixture | 64000 | 0.996 | 0.419 | 0.360 |
| mixture | limit | (1.000) | (0.406) | (0.339) |

**paper_mix, adaptive first stage** (`ds_local_efficiency_adaptive_sim.py`, N = 32000,
1000 reps; same data and folds)

| first stage | rotated N Var | prediction | slopes m Var (m = 8000) | slope limit |
|---|---|---|---|---|
| OLS | 22.88 | 23.58 | 0.985 | 1.000 |
| stacked, LS rows | 20.55 | 21.45 | 0.429 | 0.406 |
| adaptive | 20.16 | 20.90 | 0.372 | 0.339 |

Paired differences: OLS minus adaptive 2.72 (prediction 2.68); stacked minus adaptive
0.39 (prediction 0.56).

## Mukherjee et al. Theorem 2.10 (`python -m experiments.scripts.mbr_efficiency_check`)

Design: X = Z ~ N(0, 1), b = eta^3/3, s_eps^2 = 1.

| eta | truncation | OLS | WLS | bound |
|---|---|---|---|---|
| N(0, 1) | none | 12.4289 | 12.4289 | 12.4289 |
| N(0, 1) | 1.5 sd | 13.2473 | 13.2473 | 12.4289 |
| N(0, 0.25) | none | 11.0454 | 11.0434 | 11.0434 |
| N(0, 4) | none | 16.7159 | 16.5348 | 16.5348 |
| N(0, 4) | 1.5 sd | 20.2568 | 18.8886 | 16.5348 |
| t5, var 1 | none | 11.6761 | 11.6761 | 11.6633 |
