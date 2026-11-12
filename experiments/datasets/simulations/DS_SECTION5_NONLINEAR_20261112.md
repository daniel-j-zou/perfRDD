# Section 5 displayed design with a nonlinear alpha and b: Tables 1 and 3 (2026-11-12, Claude)

Author request (2026-11-12): "I want a nonlinear alpha and beta in section 5. Use one of the
new ones. Choose a good one, and make the changes now"; then "update the other parts".
"Beta" is read as the baseline b(eta). In the paper as `differing_slopes.tex` §5.1-5.3 and
Appendix 7 (Table 1 `tab:ds-four-role-paper`, Table 3 `tab:ds-eps`). The Table 2 bootstrap
coverage rerun is deferred (author, 2026-11-12: "lets not do bootstrap for now, but leave it
as a to do"); Table 2 still shows the linear design.

## Design

Design key `paper_nonlinear` in `experiments/scripts/ds_local_efficiency.py`. Same as the
previous displayed design (`paper_baseline`) except for alpha and b:

- X ~ N(0, I_3), T = gamma' X~ with gamma = (0, 1, 1, 1)/sqrt 3, eta ~ N(0, 1), Q = T + eta,
  D = 1{Q > 0};
- Y = D{alpha(eta) + X'(0.4, -0.25, 0.15)} + b(eta) + X'(0.3, -0.2, 0.1) + eps,
  eps ~ N(0, 0.5^2);
- **alpha(eta) = 2 + 0.4 eta + 0.8 tanh(1.5 eta)**, **b(eta) = eta^2/2 + 0.3 sin(2 eta)**;
- c = 2.25, trim level 0.10, window [-1.2816, 1.2816], J = [-1.75, 1.75],
  T-interval [-3, 3], Phi = [-1.5, 1.5].

Population (adaptive quadrature; `Population`, `bias_constant`, `untrimmed_target`):

| quantity | new design | previous (linear) |
|---|---:|---:|
| phi*_eps | 0.468829 | 0.545103 |
| H | -0.128227 | -0.107534 |
| untrimmed phi* | 0.394118 | 0.42618 |
| sigma^2/H^2, rotated / fixed | 26.68 / 106.7 | 31.09 / 124.4 |
| oracle Var psi_U / H^2 | 1.524 | 1.256 |
| B = dsigma^2/(2H^2) - U''' sigma^2/(2H^3) | 15.969 = 6.486 + 9.482 | 21.568 |
| dphi*/deps (Cor ds-eps) | 1.162 | 1.707 |

## Runs

Run scripts and logs (git-ignored): `experiments/runs/ds_section5_nonlinear/`
(`run_tables13.sh`, `run_eps.sh`, `tables13.log`, `eps.log`, `eps64k.log`).

- Table 1: `ds_bias_decomposition.py`, `DS_BIAS_DESIGN=paper_nonlinear`, 2000 reps at
  n = 16,000 and 64,000 (`runs/ds_bias_decomposition_nl/`), 1000 reps at n = 32,000
  (`runs/ds_bias_decomposition_nl_32k/`); seed 20261014.
- Table 3: `ds_eps_sensitivity.py`, `DS_EPS_DESIGN=paper_nonlinear`, Phi = [-1, 1.5],
  T-interval [-3.5, 3.5]; 2000 reps at n = 16,000 (`runs/ds_eps_sensitivity_nl/`) and 1000
  at n = 64,000 (`runs/ds_eps_sensitivity_nl_64k/`, a separate job with the same seeds as
  the chain would have used); seed 20261020.

## Table 1: four-role estimator, sqrt(n) bias (MC se) and n Var

| n | reps | fixed n Var | avg n Var | pooled n Var | avg bias | pooled bias | avg bias / sd |
|---:|---:|---:|---:|---:|---|---|---:|
| 16,000 | 2000 | 112.2 | 28.28 | 27.63 | 0.632 (0.119) | 0.237 (0.118) | 0.119 |
| 32,000 | 1000 | 116.2 | 26.10 | 25.51 | 0.048 (0.162) | -0.237 (0.160) | 0.009 |
| 64,000 | 2000 | 106.7 | 26.08 | 25.99 | 0.289 (0.114) | 0.098 (0.114) | 0.057 |
| first order | | 106.7 | 26.68 | 26.68 | 0 | 0 | |

- Variances: at 64k the rotated estimators are 2-3% below 26.68 (MC se about 3%), and the
  fixed assignment matches 106.7; fixed / average = 4.09.
- The average's sqrt(n) bias halves from 16k to 64k (0.63 to 0.29), as an order-1/n term should.
- n = 32,000 (1000 reps, MC relative se of n Var about 4.5%): the fixed assignment is 9% above
  106.7, about two MC standard errors. This run drew a low score term: the oracle's score term,
  an exact mean-zero sample average, is -0.099 (0.038), and every nuisance combination is
  shifted by about -0.1. The paired second-order parts are unaffected: average - score
  0.369 (0.012) against 4B/sqrt n = 0.357, pooled - score 0.084 (0.005) against 0.089,
  combination - score -0.011 (0.006). Only the 32k variances are in the paper.

### Second-order decomposition (Remark ds-bias)

Paired against each replication's own score term -Psi/H (rotation average):

| | 16,000 | 64,000 | prediction 16k / 64k |
|---|---|---|---|
| average - score | 0.526 (0.012) | 0.254 (0.006) | 4B/sqrt n = 0.505 / 0.252 |
| pooled - score | 0.131 (0.006) | 0.063 (0.003) | B/sqrt n = 0.126 / 0.063 |
| average - pooled | 0.395 (0.010) | 0.191 (0.005) | 3B/sqrt n = 0.379 / 0.189 |
| combination - score | -0.000 (0.007) | -0.001 (0.003) | 0 |
| combination (4 pool - avg)/3, sqrt(n) bias | 0.106 (0.117), n Var 27.51 | 0.034 (0.114), n Var 25.99 | b_Psi |

Both parts of B are positive: U'''(phi*) = 0.091 > 0 (criterion skewed, +9.48) and
d sigma^2/d phi = 0.213 > 0 (noise grows to the right, +6.49).

Score-term bias -E Psi / H in sqrt(n) units, by which nuisances are estimated
(G first stage, O partially linear fit, D density fit, E trim endpoints):

| combo | 16,000 | 64,000 |
|---|---|---|
| ---- (oracle) | +0.013 (0.027) | -0.028 (0.028) |
| G--- | -0.004 (0.023) | -0.018 (0.023) |
| -O-- | +0.075 (0.085) | +0.025 (0.086) |
| --D- | +0.071 (0.046) | +0.009 (0.046) |
| ---E | +0.048 (0.031) | -0.019 (0.031) |
| GODE (all estimated) | +0.106 (0.117) | +0.035 (0.114) |

The score term is not distinguishable from zero at either n. The density-fit spline score
bias by quadrature (`density_spline_bias`) is +0.0125 (L = 9, 16k), +0.0345 (L = 10, 32k)
and -0.0055 (L = 12, 64k). The partially linear fit's splines only approximate alpha and b
now, but its score term (-O--) shows no detectable bias.

Oracle arm (every nuisance at its true value): average sqrt(n) bias 0.021 (0.027) and
-0.024 (0.028); n Var 1.46 and 1.54 against 1.524; fixed oracle n Var 5.88 and 6.11
against 6.10.

## Table 3: sensitivity to the trim level

| eps | u0 | phi*_eps | -H | V | bias 16k | n Var 16k | bias 64k | n Var 64k |
|---:|---:|---:|---:|---:|---|---:|---|---:|
| 0.05 | 1.645 | 0.420 | 0.155 | 18.4 | 0.24 (0.10) | 19.6 | 0.12 (0.13) | 17.7 |
| 0.075 | 1.440 | 0.442 | 0.142 | 21.9 | 0.24 (0.11) | 23.3 | 0.13 (0.14) | 20.8 |
| 0.10 | 1.282 | 0.469 | 0.128 | 26.7 | 0.28 (0.12) | 28.8 | 0.16 (0.16) | 25.5 |
| 0.125 | 1.150 | 0.500 | 0.114 | 33.4 | 0.39 (0.13) | 36.2 | 0.31 (0.18) | 31.6 |
| 0.15 | 1.036 | 0.536 | 0.100 | 42.5 | 0.62 (0.15) | 46.8 | 0.47 (0.20) | 40.2 |
| 0.20 | 0.842 | 0.625 | 0.073 | 72.5 | 1.50 (0.20) | 81.0 | 0.91 (0.26) | 67.4 |
| 0.25 | 0.674 | 0.740 | 0.049 | 129.0 | 3.02 (0.27) | 142.8 | 1.26 (0.34) | 117.7 |

- **Target.** Rises from 0.420 to 0.740, away from the untrimmed 0.394. Between 0.075 and
  0.125: (0.50021 - 0.44205)/0.05 = 1.163, against 1.162 from Cor ds-eps.
- **Precision.** -H falls from 0.155 to 0.049; V rises sevenfold, 18.4 to 129.0.
- **Variance.** n Var / V is 1.065-1.118 at 16k and 0.912-0.960 at 64k (MC relative se
  about 3% and 4.5%; the cells share data sets, so they move together). Fixed / V_fixed:
  1.05-1.10 at 16k, 0.985-1.055 at 64k.
- **Bias.** Grows with eps and falls with n; at 64k at most 1.26 (0.12 sd).
- **Edge of Phi.** 1.1% of data sets at 16k and eps = 0.25 (5.4% in the linear design);
  none at 64k.
- **Why the eps = 0.10 bias is below Table 1's.** The wider T-interval [-3.5, 3.5] changes
  the density basis. Its spline score bias by quadrature is -0.197 at 16k, -0.003 at 64k
  (and +0.695 at 32k with L = 10, not run), against +0.0125 and -0.0055 on [-3, 3]. With
  4B/sqrt n = 0.505 and 0.252, this accounts for 0.28 against 0.63 at 16k.

Paired steps (same data and fits, consecutive eps):

| step | target | mean 16k | sd 16k | mean 64k | sd 64k |
|---|---:|---:|---:|---:|---:|
| 0.05 -> 0.075 | 0.0222 | 0.0222 | 0.0054 | 0.0222 | 0.0026 |
| 0.075 -> 0.10 | 0.0268 | 0.0270 | 0.0064 | 0.0269 | 0.0030 |
| 0.10 -> 0.125 | 0.0314 | 0.0323 | 0.0076 | 0.0320 | 0.0035 |
| 0.125 -> 0.15 | 0.0362 | 0.0380 | 0.0090 | 0.0369 | 0.0042 |
| 0.15 -> 0.20 | 0.0889 | 0.0959 | 0.0211 | 0.0907 | 0.0096 |
| 0.20 -> 0.25 | 0.1147 | 0.1267 | 0.0288 | 0.1160 | 0.0129 |

The sd of the estimate at eps = 0.10 is 0.0424 (16k) and 0.0200 (64k).
