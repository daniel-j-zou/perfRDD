# Theorem-aligned fully decoupled hard-trim simulation (2026-09-15)

## Fixed, role-rotated, and full-sample comparison (2026-09-15)

The simulation driver now reports the three requested sample-use cases:

1. `decoupled_8block`: one fixed theorem-aligned role assignment;
2. `rotated_8block`: all eight cyclic role assignments over the same physical
   eight-way partition, with the eight held-out criteria pooled before taking
   the argmax; and
3. `full_sample`: every first-stage fit, nuisance fit, and utility evaluation
   uses the full sample.

The rotated estimator is an implementation diagnostic, not a consequence of
the single-split CLT: its eight criteria share observations across rotations,
so its variance includes cross-rotation covariance.

A short Gaussian-density check (30 replications at each of `n=1,000, 2,000,
4,000`) illustrates the finite-sample trade-off. At `n=4,000`, RMSE was
`0.322` for the fixed split, `0.443` for the rotated split, and `0.106` for
full-sample reuse; boundary rates were 0%, 7%, and 0%, respectively. At
`n=8,000` (20 replications), the corresponding RMSEs were `0.198`, `0.074`,
and `0.072`, with no boundary solutions. The small-sample reversal is driven
by unstable one-eighth nuisance fits; the rotation stabilizes once each block
contains enough observations. These are diagnostic runs, not replacements
for the 200-replication theorem-matched rate table.

## Purpose

The earlier “decoupled” benchmark used six role blocks but reused one main
first-stage projection.  This run implements the construction used by the
hard-trim CLT: separate first-stage estimates for the outcome, density, and
evaluation blocks, plus separate lower- and upper-boundary fits.

## Fold construction

Each replication partitions the observations into eight equal-sized,
seeded, disjoint blocks:

| Block | Role |
|---|---|
| `gamma_alpha` | first-stage OLS for the outcome nuisance |
| `gamma_g` | first-stage OLS for the density nuisance |
| `gamma_U` | first-stage OLS for utility evaluation |
| `boundary_l` | lower endpoint first-stage OLS and quantile |
| `boundary_u` | upper endpoint first-stage OLS and quantile |
| `outcome` | spline outcome nuisance fit |
| `density` | Gaussian running-variable density fit |
| `utility` | held-out hard-trimmed criterion and optimization |

The five first-stage fits are therefore `gamma_alpha`, `gamma_g`, `gamma_U`,
and the two boundary fits.  The theorem-facing estimator is reported as
`decoupled_8block`.  Ordinary five-fold cross-fitting and full-sample OLS are
retained only as empirical comparisons.

## DGP and commands

The known-target Gaussian DGP is unchanged: (X\sim N(0,I_3)),
(T=X^\top\gamma), (eta\sim N(0,1)) independent of (X),
(Q=T+\eta), (D=1\{Q>0\}),

\[
Y=D(2+\eta)+\tfrac12\eta^2+X^\top(0.3,-0.2,0.1)^\top+\varepsilon,
\qquad \varepsilon\sim N(0,0.5^2).
\]

The hard target is \(\phi_\epsilon^*=0.7312916803\) with 10% two-sided
trimming.  The implementation is in
`experiments/scripts/hard_trim_gaussian_baseline.py` and
`experiments/scripts/hard_trim_crossfit_regularization.py`.

Commands run from the code repository root:

The first two commands use the legacy Gaussian location-scale density as a
sensitivity check.  The final command uses the fixed-support spline density
and is the theorem-matched run.

```text
python3 -m experiments.scripts.hard_trim_crossfit_regularization \
  --n 1000 2500 5000 10000 --reps 200 --workers 4 --ridge 0 \
  --out experiments/runs/theorem_decoupled_20260915

python3 -m experiments.scripts.hard_trim_crossfit_regularization \
  --n 10000 20000 40000 80000 --reps 100 --workers 4 --ridge 0 \
  --out experiments/runs/theorem_decoupled_large_20260915

python3 -m experiments.scripts.hard_trim_crossfit_regularization \
  --n 10000 20000 40000 80000 --reps 200 --workers 4 --ridge 0 \
  --density spline --out experiments/runs/theorem_decoupled_spline_final_20260915
```

The `experiments/runs/` directory is intentionally ignored; the JSON/CSV/PNG
outputs remain available in the current workspace and the summary below is the
durable record.

## Results

The Gaussian-density sensitivity run shows the finite-sample cost of eight-way
splitting:

| (n) | RMSE | (n\times\mathrm{MSE}) | bias | boundary rate |
|---:|---:|---:|---:|---:|
| 1,000 | 0.990 | 979.2 | -0.131 | 0.420 |
| 2,500 | 0.512 | 654.5 | 0.014 | 0.095 |
| 5,000 | 0.301 | 454.1 | 0.027 | 0.020 |
| 10,000 | 0.226 | 512.2 | 0.020 | 0.005 |

In the larger Gaussian-density run, the decoupled estimator stabilizes:

| (n) | RMSE | (n\times\mathrm{MSE}) | bias | boundary rate |
|---:|---:|---:|---:|---:|
| 10,000 | 0.232 | 538.1 | 0.014 | 0.000 |
| 20,000 | 0.147 | 432.4 | -0.000 | 0.000 |
| 40,000 | 0.104 | 432.9 | 0.010 | 0.000 |
| 80,000 | 0.075 | 451.8 | 0.009 | 0.000 |

The table-matched spline-density run gives the final rate check:

| (n) | RMSE | (n\times\mathrm{MSE}) | bias | boundary rate |
|---:|---:|---:|---:|---:|
| 10,000 | 0.223 | 495.6 | 0.030 | 0.005 |
| 20,000 | 0.146 | 429.1 | 0.014 | 0.000 |
| 40,000 | 0.102 | 418.1 | 0.006 | 0.000 |
| 80,000 | 0.079 | 501.5 | 0.002 | 0.000 |

The log--log RMSE slope for this final run is (-0.499).  The pooled
(n\times\mathrm{MSE}) is 461.1.  The scaled bias decreases from 2.96 at
10,000 to 0.60 at 80,000, consistent with a finite-sample bias that vanishes.

The five-fold and full-sample comparison slopes are \(-0.521\) and \(-0.518\),
respectively.  The approximately constant (n\times\mathrm{MSE}) from
20,000 onward and the disappearance of boundary solutions support the
theorem's root-(n) prediction.  The roughly ten-fold larger asymptotic
variance is expected: each principal score block uses only one eighth of the
sample, and five independent first-stage fits contribute sampling noise.

These runs test centering and rate under the theorem's split and nuisance
density specification.  They do not by themselves validate an analytic
variance estimator or bootstrap coverage.
