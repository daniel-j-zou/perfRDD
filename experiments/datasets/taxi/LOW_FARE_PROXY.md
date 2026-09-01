# Low-fare CMT proxy for the VTS menu effect

## Why the original `alpha(eta)` does not turn negative

The hard-trim VTS estimate is identified from the observed $15 menu change. In
the supported VTS window, the percentage regime produces higher tips, so a
positive fitted `alpha(eta)` is the result we should expect. Changing the ridge
penalty or forcing the spline to cross zero would be extrapolation, not new
identification.

The low-fare question is different: what would happen if VTS applied a
percentage menu where it currently shows fixed dollar amounts? That contrast is
not observed within VTS. CMT provides an auxiliary comparison because its
percentage suggestions apply below $15.

## Proxy construction

Using all paper-restricted January 2009 records with $5 <= fare < $15, we fit

```text
tip = fare-cell fixed effects
    + common control slopes
    + CMT-by-control slopes
    + CMT-by-fare-cell effects
```

The reported proxy is the predicted CMT-minus-VTS tip difference evaluated at
the mean VTS controls in each fare cell. CMT-minus-VTS is interpreted as
percentage-menu-minus-fixed-menu only under conditional vendor exchangeability.
The model uses the same four controls and VTS standardization as the hard-trim
application, but it is indexed by fare rather than by the PerfRDD residual
`eta`.

Run it with:

```text
python -m experiments.scripts.taxi_low_fare_proxy
```

## Verified result

The restricted low-fare sample contains 484,123 VTS and 428,062 CMT rides over
25 standard-meter fare cells. The VTS-distribution-weighted proxy is **−$0.212
per low-fare trip** (the unadjusted difference is −$0.214). The proxy is
negative on 90.5% of the VTS low-fare mass and crosses zero at approximately
**$12.8**.

Selected adjusted contrasts (CMT minus VTS) are:

| fare | proxy | HC0 95% interval |
|---:|---:|---:|
| $5.30 | −$0.370 | [−$0.384, −$0.355] |
| $8.10 | −$0.276 | [−$0.291, −$0.261] |
| $10.90 | −$0.098 | [−$0.117, −$0.079] |
| $12.50 | −$0.020 | [−$0.047, +$0.006] |
| $12.90 | +$0.009 | [−$0.021, +$0.038] |
| $13.30 | +$0.044 | [+$0.015, +$0.073] |
| $14.90 | +$0.144 | [+$0.105, +$0.183] |

The negative low-fare pattern is therefore visible in both raw and
control-adjusted comparisons and is economically aligned with the menu amounts:
percentage suggestions are below the fixed-dollar suggestions at low fares.

## What this does and does not establish

- It gives a credible **proxy** for the sign and rough size of the low-fare
  menu contrast, and it supplies the negative component the local VTS RDD cannot
  identify.
- It is not a causal VTS counterfactual. Vendor choice, unobserved driver/route
  composition, and different percentage menus can all contaminate CMT-minus-VTS.
  The public January file lacks the driver identifiers needed for the paper's
  within-driver vendor comparison.
- The HC0 intervals treat trips as independent and are exploratory; they are not
  the final application inference procedure.

The appropriate next specification is a menu-aware model that treats CMT as an
auxiliary sample with vendor effects and uses the low-fare proxy as a sensitivity
or calibration component. The original hard-trim `alpha(eta)` should remain the
local VTS treatment effect, not be mechanically altered to force a negative
tail.

The figure and cell-level export are written to the ignored run directory
`experiments/runs/taxi_low_fare_proxy/`.
