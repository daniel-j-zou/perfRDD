# CMS Hospital Readmissions Reduction Program: a kinked dose on readmission ratios

| Field | Value |
|---|---|
| **Q** | `load_dose(cond="hf")`: ERR_HF - peer-group median ERR_HF, the hospital's heart-failure excess readmission ratio minus the median of its dual-proportion quintile. Any of the six conditions can be chosen |
| **Threshold** | 0: a kink, not a jump. The condition starts to add to the reduction at Q > 0 |
| **Treatment** | A = payment reduction on FY t base operating DRG payments, in percent, continuous in [0, 3] |
| **Rule class** | multi-dimensional score, continuous dose. A = min(3, sum over six conditions of 100 NM w_c (ERR_c - med_c)_+). Along one condition, A is flat below 0 and rises with a hospital-specific slope above it |
| **X** | dual proportion; log eligible discharges and DRG payment ratio of each condition; peer-group, state (CCN prefix) and FY indicators: 70 columns |
| **Y** | ERR of the same condition in the FY t+3 file, whose 3-year window (July t-2 to June t+1) does not overlap FY t's. `horizon=4` gives a window that starts after the FY t final rule |
| **n** | panel 24,304 hospital-years (FY 2019-2022, 2024-2027); `load_dose()` 10,608 (2,813 hospitals; FY 2019, 2021, 2022, 2024) |
| **Source** | CMS IPPS final rules, HRRP supplemental data files FY 2013-2027 (cms.gov "Archived Supplemental Data Files"); 42 CFR 412.154 |

**Fit: weak.**
- **The rule is exact, but the design is a kink on a six-dimensional score.**
  - The dose is not a step function of one score. Through the other five conditions,
    84% of hospitals already have a reduction at Q = 0 (`a_ref > 0`).
- **X predicts Q poorly.** R^2 is 0.14, and sd(eta) is 0.065 against sd(Q) of 0.071.
- **The outcome is limited.**
  - Windows overlap for two of three years between consecutive files.
  - FY 2020 has no t+3 outcome, because the FY 2023 file is missing.
- Use it as a kink example, or as a multi-score rule, not as a step-dose design.

## The rule (FY 2019+)

The variable-description tab of each file gives this rule:

- factor = 1 - min(0.03, NM * sum_c 1{n_c >= 25 and ERR_c > med_c} * w_c * (ERR_c - med_c)).
- The payment reduction is 100 * (1 - factor).
- The conditions are AMI, COPD, HF, pneumonia, CABG and THA/TKA.
- w_c is the condition's share of the hospital's base DRG payments. NM is the
  neutrality modifier, 0.953-0.965.
- med_c is the median ERR in the hospital's peer group, a quintile of the dual-eligible
  proportion.

Before FY 2019 the median was national (ERR > 1), with no peer groups or NM. Those
files (FY 2013-2018) are downloaded but not parsed.

**Rule check (`adapter.rule_check()`).**

| FY | Hospitals | Reduced | At the 3% cap | Factor matches to 4 decimals | Penalty flags match the rule |
|---|---:|---:|---:|---:|---:|
| 2019 | 3,173 | 0.819 | 0.015 | 0.974 | 0.998 |
| 2020 | 3,130 | 0.825 | 0.018 | 0.969 | 0.999 |
| 2021 | 3,080 | 0.826 | 0.013 | 0.967 | 0.996 |
| 2022 | 3,047 | 0.821 | 0.013 | 0.975 | 0.999 |
| 2024 | 3,037 | 0.776 | 0.003 | 0.979 | 0.998 |
| 2025 | 2,980 | 0.786 | 0.004 | 0.977 | 0.999 |
| 2026 | 2,945 | 0.782 | 0.005 | 0.975 | 0.999 |
| 2027 | 2,912 | 0.802 | 0.003 | 0.974 | 0.999 |

- Every remaining difference is at most 0.007 percentage points. That is the rounding
  of the published inputs (ERR, median and payment ratio to 4 decimals).
- All 40 penalty-flag mismatches have the published ERR equal to the published median,
  so they are ties created by the same rounding.

## Dose design

`load_dose(cond="hf", horizon=3)`:

| `extras` key | Content |
|---|---|
| `A` | formula reduction (%), equal to the published one within 0.011 for every row |
| `a_ref` | per-row reduction from the other five conditions (the dose at Q <= 0), not 0 |
| `dose_levels` | None (continuous) |
| `schedule` | `schedule(q, group)`, where `group` is the row index: min(3, a_other + slope * max(q, 0)) |
| `group` | row index (the schedule is hospital-specific); `fy` holds the fiscal year |
| `slope`, `cap` | 100 NM w_c (0 when n_c < 25); 3.0 |
| `Z_names` | `dual`, `log_n_<cond>`, `w_<cond>` |
| `published_dose`, `ccn`, `raw` | |

## Readiness (2026-10-28)

| Quantity | HF, t+3 | HF, t+4 | Pneumonia, t+3 |
|---|---:|---:|---:|
| n (hospitals) | 10,608 (2,813) | 7,962 (2,731) | 10,858 (2,854) |
| R^2 of Q on X (within FY) | 0.140 (0.139) | 0.164 | 0.122 |
| sd(Q), sd(T), sd(eta) | 0.071, 0.026, 0.065 | 0.073, 0.030, 0.067 | 0.069, 0.024, 0.065 |
| Share with A = 0 | 0.110 | 0.103 | 0.118 |
| Share at the 3% cap | 0.008 | 0.012 | 0.008 |
| A: 10 / 50 / 90% | 0 / 0.33 / 1.38 | 0 / 0.38 / 1.50 | 0 / 0.32 / 1.36 |
| Share with Q > 0 | 0.499 | 0.500 | 0.496 |
| eta 10/50/90%, Q <= 0 | -0.104 / -0.042 / 0.003 | -0.105 / -0.043 / 0.006 | -0.102 / -0.042 / 0.003 |
| eta 10/50/90%, Q > 0 | -0.008 / 0.039 / 0.112 | -0.011 / 0.040 / 0.114 | -0.006 / 0.038 / 0.109 |
| Within 0.25 sd(eta) of the kink | 0.199 | 0.198 | 0.205 |
| corr(Y, Q) | 0.38 | 0.35 | 0.34 |

**Bunching.** Counts are in windows of 0.01 on each side of a point.

| Point | HF: above / below (ratio) | Pneumonia |
|---|---|---|
| 0 (kink) | 659 / 653 (1.01) | 660 / 718 (0.92) |
| -0.06 | 445 / 334 (1.33) | 458 / 377 (1.21) |
| -0.03 | 655 / 582 (1.13) | 682 / 598 (1.14) |
| 0.03 | 508 / 601 (0.85) | 506 / 603 (0.84) |
| 0.06 | 372 / 390 (0.95) | 403 / 417 (0.97) |

There is no sign of bunching at the kink. The ratio follows the slope of the density,
which peaks near 0.

## Getting the data

    python -m experiments.datasets.cms_hrrp.download          # idempotent, 40.5 MB
    python -m experiments.datasets.cms_hrrp.download --check

- The download is plain `curl`, with no login, form, terms page or CAPTCHA. It saves
  `data/raw/supp/hrrp_fy{2013..2022, 2024..2027}.zip`, 40,513,473 bytes, each
  extracted to its own folder.
- The adapter reads the FY 2019-2027 tab-delimited or 508 CSV versions.
  `data/processed/hrrp_panel.csv.gz` caches the panel.
- **Broken links on cms.gov, 2026-10-28.**
  - The FY 2023 archive page links a ZIP that is byte-identical to the FY 2024 file. It
    is not used, so FY 2023 is missing.
  - The FY 2026 archive page links the FY 2025 ZIP. The FY 2026 file was taken from the
    FY 2026 IPPS final rule home page instead.
  - The HRRP page's "Payment Reduction Methodology" and "FY 2027 Fact Sheet" PDFs serve
    the HAC Reduction Program documents. Neither was kept.

## Caveats

- The six ERRs come from the same hospital and window and are correlated. Q on one
  condition treats the other five as fixed (`a_ref`).
- Penalties persist across years, so later ERRs reflect several years of penalties.
- COVID-19: CMS does not use Q1-Q2 2020 claims (national exception). The FY 2022 window
  is July 2017 to December 2019, and FY 2024's is July-December 2019 plus July 2020 to
  June 2022 (variable descriptions). Hospitals with individual exceptions also lose
  claims.
- Hospitals repeat across years; cluster by `extras["ccn"]`.
