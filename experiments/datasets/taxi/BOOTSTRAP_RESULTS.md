# Taxi 30,000-trip bootstrap results

## Conclusion

On the paper-restricted January 2009 sample, the no-cost driver-tip objective is
strictly decreasing over the economically feasible threshold grid. The estimated
policy therefore assigns percentage suggestions to all eligible fares (`phi*=2.5`).
All 199 full-reestimation bootstrap replications select that same lower boundary.

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
- `utility_curve_bootstrap.png` — utility band and gain distribution.

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
| Hard-trim retention | 34.09% |
| No-cost `phi*` | $2.50 (lower boundary) |
| Bootstrap lower-boundary share | 100% |
| Gain from all-fare percentage suggestions versus current $15 rule | 34.71 cents per hard-trimmed trip |
| Bootstrap standard error of gain | 3.29 cents |
| Centered-bootstrap 95% interval for gain | [27.16, 40.69] cents |
| Simultaneous-band interval at $2.50 | [27.45, 41.98] cents |

The percentile range of the argmax is `[$2.50,$2.50]`, but it should not be presented
as an ordinary regular confidence interval: the optimum is at the policy boundary.
The utility band and gain relative to the current policy are the meaningful inference.

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
- The analysis observes credit-card but not cash tips and does not identify effects on
  payment method, demand, repeat riding, or other driver revenue.

The appropriate current claim is therefore numerical and sample-specific: **under the
estimated no-cost policy criterion, the lower-boundary recommendation is bootstrap
stable.** It is not yet a publication-ready causal recommendation to change the screen.
