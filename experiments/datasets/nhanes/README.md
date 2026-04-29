# NHANES — HbA1c diabetic threshold RDD

| Field | Value |
|---|---|
| **Q** | HbA1c (`LBXGH`, %) |
| **Threshold** | 6.5 (ADA diabetic diagnosis cutoff) |
| **Treatment** | `1{Q >= 6.5}` |
| **X** | age, sex, race/ethnicity, BMI, family poverty index ratio |
| **Y** | systolic blood pressure (`BPXSY1`, mmHg) |
| **n** | ~5k after complete-case filter |
| **Cycle** | 2017-2018 (J) |
| **Source** | https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ |

## Building

```bash
python -m experiments.datasets.nhanes.download
```

Pulls four small `.XPT` (SAS transport) files (~6 MB total) from CDC.

## Caveats

- This is a *cross-sectional* discontinuity, not a clean policy RDD.
  The 6.5% threshold is a diagnostic cutoff — being above it correlates
  with receiving a diabetic diagnosis and antihypertensive treatment,
  but the assignment is partly endogenous.
- An alternative threshold worth trying: 7.0% (ADA "uncontrolled
  diabetes" management trigger) for a sub-sample already above 6.5.
- For longitudinal effects you'd need NHANES → NDI mortality linkage
  (publicly available with a separate file).

## Switching to other cycles

Replace the `_J` suffix throughout `download.py` and `adapter.py` to use
a different two-year cycle:
- `_I` → 2015-2016
- `_H` → 2013-2014
- `P_` prefix → pre-pandemic 2017-2020 combined
