# Direct CMT prediction check of the VTS decomposition

The useful confirmation test is not to compare two arbitrary alpha curves. It
is to take the VTS-fitted outcome model and ask whether it predicts actual CMT
percentage-menu rides below $15:

```text
VTS fixed prediction       = b(eta) + beta'X
VTS percentage prediction  = b(eta) + beta'X + alpha(eta)
```

The VTS model is fit on the locked 30,000-ride VTS sample. Both predictions are
then evaluated on CMT rides using the VTS first-stage index and the VTS control
scale. All summaries use the VTS hard-trim interval; CMT rides below $15 are
actual percentage-menu rides.

Run:

```text
python -m experiments.scripts.taxi_competitor_prediction_check
```

## Results

The VTS fit is internally calibrated:

| check | observations | mean observed minus predicted |
|---|---:|---:|
| VTS low fare − fixed prediction | 8,513 | −$0.001 |
| VTS high fare − percentage prediction | 1,715 | −$0.005 |

That is, the model reproduces the two observed VTS regimes in the sample used
to fit it. The CMT transfer check is different:

| check | observations | mean observed minus predicted |
|---|---:|---:|
| CMT low fare − VTS fixed prediction | 144,131 | −$0.140 |
| CMT low fare − VTS percentage prediction | 144,131 | **−$0.542** |
| CMT high fare − VTS percentage prediction | 26,253 | −$0.100 |

Subtracting the high-fare CMT residual as a rough vendor/menu calibration still
leaves a low-fare residual of **−$0.443** relative to the VTS-implied percentage
prediction. Thus CMT does **not** confirm that the positive VTS `alpha(eta)` can
be transported unchanged below $15. Instead, actual CMT percentage rides are
below the VTS-model percentage prediction, which is consistent with a negative
low-fare percentage-versus-fixed contrast.

This is the strongest conclusion available from the public comparison:

1. The VTS model fits the observed local fixed/percentage regimes internally.
2. CMT supplies an out-of-sample sign check that points toward a negative
   low-fare effect.
3. CMT rejects a naive extrapolation of the local positive alpha, but cannot
   identify the VTS counterfactual causally because the vendors, percentage
   menus, and unobserved driver/route composition differ.

The comparison figure and fare-level predictions are in the ignored run
directory `experiments/runs/taxi_competitor_prediction_check/`.
