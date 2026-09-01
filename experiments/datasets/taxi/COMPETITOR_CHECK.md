# Competitor-only validation of the taxi outcome decomposition

## Design

The CMT (Competitor) payment system used percentage suggestions on both sides
of the $15 fare.  It therefore has no treatment discontinuity at $15.  We use
CMT as a placebo sample: retain CMT credit-card rides under the same published
restrictions, assign the artificial split `D = 1{Fare_Amt >= 15}`, and fit the
same exact hard-trim model used for the VTS application.  The CMT `alpha` is a
placebo threshold contrast, not a causal estimate of the VTS menu effect.

The comparison uses January 2009, a locked 30,000-ride subsample (seed 0),
`eps=0.1`, nuisance support `[-6, 11]`, ridge scale `0.001`, and the same VTS
restricted-sample standardization for all four controls.  Run it with:

```text
python -m experiments.scripts.taxi_competitor_check
```

The raw January parquet is present locally, so both source samples are
available without downloading the full-year archive.

## Verified point results

| quantity | VTS actual $15 split | CMT artificial placebo split |
|---|---:|---:|
| restricted source rows | 541,318 | 478,012 |
| analysis rows | 30,000 | 30,000 |
| hard-trim retention | 34.09% | 28.72% |
| hard-trim interval in `eta` | [0.216, 8.629] | [0.423, 8.660] |
| weighted mean `alpha` in hard window | +0.406 | +0.091 |
| minimum `alpha` on hard-window grid | +0.384 | −0.288 |
| maximum `alpha` on hard-window grid | +0.953 | +0.109 |
| mean absolute `alpha` on hard-window grid | 0.481 | 0.094 |

As a direct local check, mean tip dollars at the adjacent standard-meter fare
cells are:

| vendor | $14.90 | $15.30 | difference |
|---|---:|---:|---:|
| VTS | $2.269 | $2.626 | **+$0.357** |
| CMT | $2.411 | $2.461 | **+$0.050** |

The baseline curves have a similar shape on the common hard-trim interval
(correlation 0.968), but differ in level (VTS minus CMT mean −$0.178; RMSE
$0.290).  This level difference is expected because the vendors used different
menus and is not evidence that the two baselines should coincide.

## Interpretation and limits

1. The adjacent-fare CMT placebo is nearly smooth while VTS has the expected
   local jump.  This supports the claim that the VTS discontinuity is associated
   with the menu change rather than a generic fare-grid break.
2. The CMT fitted placebo `alpha` is not exactly zero over the broad hard-trim
   window.  Its nonzero shape is a specification warning: under an artificial
   split, residual vendor/menu composition and the restricted linear-control
   outcome model can be absorbed by `alpha(eta)`.  The CMT fit should therefore
   be used as a falsification/diagnostic, not as a second causal estimate.
3. The VTS positive `alpha` is much larger than the CMT placebo contrast, but
   CMT does not identify the VTS counterfactual low-fare fixed-dollar outcome.
   A causal claim about moving the VTS threshold below $15 still requires a
   menu-aware outcome model or an additional overlap/transport assumption.

All machine-readable values and component exports are in the ignored run
directory `experiments/runs/taxi_competitor_check/`; the reproducible source
is `experiments/scripts/taxi_competitor_check.py`.
