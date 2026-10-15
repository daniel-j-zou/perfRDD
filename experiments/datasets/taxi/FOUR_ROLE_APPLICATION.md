# Taxi application with the four-role estimator (2026-10-15, Claude)

The paper's estimator (`differing_slopes.tex` §3: OLS first stage, stacked spline
outcome fit on a fixed region J, spline g and rho_X on the index fold, quantile trim
endpoints, average of the four rotated thresholds) applied to the January 2009 VTS
taxi data. Write-up with figure and tables: `manuscript/this_week.tex` §7. Script:
`experiments/scripts/taxi_four_role.py`; figure: `taxi_four_role_figures.py`; tests:
`experiments/tests/test_taxi_four_role.py`.

## Specification

| piece | choice | why |
|---|---|---|
| sample | Haggag–Paci restrictions **without** the $5–$25 fare range; zero-distance rides and tip entry errors dropped; n = 633,616, 11.0% treated | the range selects on the running variable (author decision, 2026-10-15); the restricted sample is a comparison |
| score | Q = sqrt-scale Box–Cox of the fare (lambda = 1/2) | fare = $2.50 + $0.40 per meter click; the time charge grows with distance, so eta's spread rises with distance in dollars and falls in logs; lambda = 1/2 makes it flat. Chosen from (fare, Z) only. A cutoff on Q is a cutoff on the fare. |
| score and baseline covariates Z | cubic B-spline in log distance (quintile knots), hour, day of week, weekday-hour x weekend, group ride (36 columns) | removes the mean dependence of eta on the covariates |
| effect covariate X | trip distance | menu arithmetic: suggestions are 20–30% of the fare vs $2–$4; a richer D*X block is not identified (at fixed eta treated rides are the long ones) |
| trim | eps = 0.05; both endpoints fall outside the observed residuals, so the window is J = central 99% of eta-hat (99.0% of rides) | the widest population the data support |
| design | J, density interval, Phi = [$6, $24] fixed from (Q, Z) before outcomes; outcome knots n_eff^(11/60) | as in the simulations |
| inference | full re-estimation bootstrap over rides, 199 draws, percentile | no driver identifiers in the public file, so rides are treated as independent |

## Result

phi-hat = **$12.80**, i.e. show the percentage menu from **$12.90** (fares lie on a
$0.40 grid). Rotations $12.38–$13.33; pooled-criterion argmax $12.82; own-score
objective selects the same cell. Bootstrap 95% interval [$12.23, $13.19] (sd $0.24; 199 draws, all interior and below $15;
181 select the $12.90 or $13.30 cell). Estimated gain over
the deployed $15: 0.55 cents per ride (band 0.37–0.80); at $15 the marginal
rides would tip 19 cents more under the percentage menu (band 16–23). Residual-only model (beta2 = 0): lower boundary
of Phi (treat every ride).

## Robustness (eps = 0.05 unless stated)

| variation | phi-hat | cell |
|---|---:|---:|
| main | 12.80 | 12.90 |
| other fold seeds | 12.78–12.89 | 12.90 |
| J excludes 0.25%, 1%, 2% tails | 12.97–13.19 | 13.30 |
| eps = 0.075 / 0.10 / 0.125 | 12.82 / 12.20 / 7.70 | 12.90 / 12.50 / 8.10 |
| lambda = 1/4 / log | 13.06 / 12.96 | 13.30 / 13.30 |
| effect quadratic in distance | 12.30 | 12.50 |
| baseline linear in distance | 11.11 | 11.30 |
| restricted $5–$25 sample | 8.21 (rotations 6.00–10.34) | 8.50 |
| restricted, quadratic effect | 6.00 | boundary |

Larger eps shrinks the population to the slowest rides (53% of rides at 0.10, 23% at
0.125), whose optimum is lower. The linear baseline fits the untreated rides worse
(RSS 543,999 vs 541,833; hour-of-day residual means up to 12.6 vs 2.2 cents).
Truncating fares at $25 removes the long treated rides that identify the distance
slope of the effect (beta2 0.46 -> 0.18; sign flips with a quadratic term).

## Reproduce

```bash
python -m experiments.scripts.taxi_four_role --boot 199 --workers 5     # main + extras
python -m experiments.scripts.taxi_four_role --sample restricted --no-extras
python -m experiments.scripts.taxi_four_role --effect distance2 --no-extras
python -m experiments.scripts.taxi_four_role --baseline linear --no-extras
python -m experiments.scripts.taxi_four_role --lam 0.25 --no-extras     # also --lam 0
python -m experiments.scripts.taxi_four_role_figures \
    --summary experiments/runs/taxi_four_role/summary_full_lam0.5_eps0.05.json
```

Outputs go to the ignored `experiments/runs/taxi_four_role/`. The CMT benchmark in the
figure reads `experiments/runs/taxi_low_fare_proxy/fare_proxy.csv`
(`python -m experiments.scripts.taxi_low_fare_proxy`).

## Open

- Theory: the specification uses different covariates in the score/baseline (Z) and
  the effect (distance), and distance enters Z through spline columns; the paper's
  theorem and its quantile lemma need those two small extensions.
- Clustered (driver) inference is not possible with the public data.
- The superseded diagnostics (`taxi_differing_slopes.py`, full-sample ridge, $12.66–$12.72
  on the restricted sample) are kept for the record.
