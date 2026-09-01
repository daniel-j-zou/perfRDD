# Taxi 30,000-trip bootstrap results

## Conclusion

On the paper-restricted January 2009 sample, the no-cost driver-tip objective is
weakly decreasing over the economically feasible threshold grid. The point curve is
numerically tied at its maximum from $2.50 through $3.80, after which it declines. The
implementation reports `phi*=2.5` because `argmax` returns the first tied grid point.
All 199 full-reestimation bootstrap replications likewise report the lower boundary,
conditional on that tie-breaking rule.

This is evidence that the numerical conclusion is stable on the locked sample. It is
not evidence that moving the historical screen below $15 has a causally identified
effect: the required counterfactual transport assumption remains much stronger than
the local RDD.

## Reproduction

From the code repository root:

```bash
/Users/zoudj/miniconda3/envs/srsd/bin/python \
  -m experiments.scripts.taxi_bootstrap_30k \
  --replications 199 --workers 6
```

Outputs are written to the ignored directory
`experiments/runs/taxi_bootstrap_30k/`:

- `summary.json` — specification, headline estimates, and limitations;
- `utility_bands.csv` — estimated relative utility and bootstrap bands;
- `bootstrap_estimates.csv` — one row per successful replication;
- `bootstrap_curves.npz` — baseline and replication-level utility curves;
- `utility_curve_bootstrap.png` — utility band and gain distribution;
- `baseline/alpha_curve.csv` — fitted conditional percentage-menu effect;
- `baseline/baseline_curve.csv` — fitted untreated baseline component;
- `baseline/beta_coefficients.csv` — fitted linear-control coefficients;
- `baseline/outcome_components.png` — combined outcome-model diagnostic.

## Locked specification

- Data: public TLC January 2009 Vendor credit-card records.
- Published main-RDD restrictions: no tolls, taxes, or surcharges; daytime windows;
  standard-meter increments; fares from $5 to $25.
- Restricted source rows: 541,318; deterministic analysis sample: 30,000 (`seed=0`).
- Outcome: tip amount in dollars; objective: expected driver tip revenue (`cost=0`).
- Treatment: percentage suggestions at fares at or above $15.
- Policy grid: $2.50--$25.00 in $0.05 increments.
- Exact hard trim: `eps=0.1`, pilot support `[-6,11]`.
- Estimator: full sample, ridge scale `0.001`.
- Bootstrap: 199 i.i.d. trip resamples; every nuisance and the argmax re-estimated.

The original article's assignment used total base amount to choose the screen. The
published main-RDD restrictions eliminate tolls, taxes, and surcharges, so fare is the
base amount in this analysis. This corrects the earlier generic taxi pilot, which used
all filtered VTS credit-card records and is not a paper-comparable application sample.

## Results

| Quantity | Estimate |
|---|---:|
| Successful bootstrap replications | 199 / 199 |
| Hard-window average effect | $0.4056 per trip |
| Estimated hard-trim interval for fare residual | [0.216, 8.629] |
| Range of fitted alpha on hard-trim grid | [$0.384, $0.953] per trip |
| Hard-trim retention | 34.09% |
| No-cost `phi*` | $2.50 (lower boundary) |
| Numerically tied point-estimate maximizer range | [$2.50, $3.80] |
| Bootstrap lower-boundary share | 100% |
| Bootstrap 2.5th--97.5th percentiles of plateau upper endpoint | [$3.70, $4.05] |
| Gain from all-fare percentage suggestions versus current $15 rule | 34.71 cents per hard-trimmed trip |
| Bootstrap standard error of gain | 3.29 cents |
| Centered-bootstrap 95% interval for gain | [27.16, 40.69] cents |
| Simultaneous-band interval at $2.50 | [27.45, 41.98] cents |

The percentile range of the first-grid-point argmax is `[$2.50,$2.50]`, but it should
not be presented as an ordinary regular confidence interval: the optimum is on a flat
boundary plateau and `np.argmax` deterministically selects its first point. The utility
band and gain relative to the current policy are the meaningful inference.

The fitted `alpha(eta)` curve is positive over the entire hard-trim interval; this
specification therefore finds no residual-defined subgroup with a negative estimated
percentage-menu effect. That is not evidence that the effect is positive at every fare.
The historical treatment is deterministic in fare, so there is no fixed-versus-percentage
comparison at the same fare. Moreover, the outcome model restricts the treatment effect
to depend on the fare residual `eta`, not on the fare level or the dollar values displayed
by the menu. Transporting the fitted effect below $15 consequently imposes rather than
tests menu-value invariance.

The fitted linear-control coefficients are in dollars per one-standard-deviation change
because the controls were standardized on the restricted source sample:

| Control | `beta` |
|---|---:|
| Trip distance | 0.3510 |
| Passenger count | -0.0018 |
| Hour of day | 0.0361 |
| Day of week | -0.0287 |

These are partial-regression components, not causal effects. The accompanying component
figure also reports the nonlinear untreated baseline `b(eta)`, which is distinct from
the finite-dimensional coefficient vector `beta`.

## What the bootstrap establishes

- Sampling variation within the locked 30,000-trip sample does not change the policy
  ranking: every full re-estimation selects the lower boundary.
- Re-estimating the first stage, trim endpoints, spline, policy probabilities, and
  utility curve produces a reasonably concentrated gain distribution.
- The bootstrap implementation can be ported to a larger stratified sample or cluster.

## What remains unresolved

- Trips are resampled independently. The public TLC file lacks the anonymized driver
  and car identifiers used by Haggag and Paci, and January has only 31 daily clusters.
- The nuisance design remains ill-conditioned (baseline condition number about
  297,000; bootstrap median about 304,000), despite covariate standardization.
- The estimated hard-window average effect ($0.406) has not yet been reconciled with
  the original paper's conventional local-RD benchmark of roughly $0.27--$0.30.
- Only 34% of the locked sample survives the hard trim; support and `eps` sensitivity
  remain necessary.
- The result extrapolates the local $15 discontinuity to much lower fares. At low fares,
  fixed dollar suggestions can mechanically exceed percentage suggestions, making
  policy invariance especially doubtful.
- The public data cannot reveal where the low-fare effect changes sign without an
  additional menu-response model or another source of assignment variation: below $15
  only the fixed menu is observed, and at or above $15 only the percentage menu is
  observed.
- The analysis observes credit-card but not cash tips and does not identify effects on
  payment method, demand, repeat riding, or other driver revenue.

The appropriate current claim is therefore numerical and sample-specific: **under the
estimated no-cost policy criterion, the lower-boundary recommendation is bootstrap
stable.** It is not yet a publication-ready causal recommendation to change the screen.
