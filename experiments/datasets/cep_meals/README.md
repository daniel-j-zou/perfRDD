# Texas CEP school meals: free claiming share on the identified student percentage

| Field | Value |
|---|---|
| **Q** | The score: the site's identified student percentage (ISP, 0-1), TDA's `SiteISP`, "reported April of the previous program year": students certified for free meals without an application (SNAP, TANF, FDPIR, Medicaid where approved, homeless, migrant, runaway, Head Start, foster) over enrolled students |
| **Threshold** | 0.40 for elections through program year 2023-24; 0.25 from 2024-25 (Texas applied the 26 Oct. 2023 rule from SY 2024-25, ISP of April 1, 2024) |
| **Treatment** | Dose `extras["A"]` = free claiming share OFFERED on the site's own ISP: 1{Q >= c_t} x min(1, 1.6 Q). a_ref = 0 (not offered). The received dose (CEP x claiming share implied by the site's lunch reimbursement) is `extras["A_received"]` |
| **Rule class** | A jump plus a kink: 0 below the threshold, a jump to 1.6 c_t (0.64 before, 0.40 after the 2023 rule), linear with slope 1.6, capped at 1 from Q = 0.625. Two schedules (c = 0.40, 0.25) that differ only for Q in [0.25, 0.40) |
| **X** | Log enrollment, grade composition (K-5, 6-8, 9-12 shares, pre-K, number of grades), charter, alternative-instruction campus (AskTED), FSMC-managed, ESC region, log CE enrollment, log number of CE sites, leave-one-out CE ISP (enrollment-weighted ISP of the CE's other sites) + missing flag, program-year indicators (39 columns) |
| **Y** | Lunch participation: lunches served / (lunch days x enrollment), program year t (the year the dose applies). Options: `outcome="breakfast"`, `"meals"` (lunches + breakfasts per enrolled student per lunch day) |
| **n** | 57,310 site-years (9,254 sites, 1,169 contracting entities), program years 2015-16, 2016-17, 2018-19, 2022-23, 2023-24 (c = 0.40) and 2024-25, 2025-26 (c = 0.25) |
| **Source** | Texas Department of Agriculture (TDA) School Nutrition Programs datasets on data.texas.gov: site participation (ISP, CEP flag) PY 2015-16..2026-27 and meal reimbursement claims PY 2015-16..2025-26; TEA AskTED; 7 CFR 245.9(f) (govinfo annual editions 2023, 2024, 2025); 88 FR 65778 |

**Fit: weak.** The data exist, link and are large, but the dose the formula sets is not the
dose schools get.
- **Adoption is a choice, and the thresholds barely move it.** At the site level, CEP
  rises smoothly with the ISP: 1-8% below 0.40 in 2016-19, 18-24% at 0.40-0.50, 64-91%
  above 0.625 (table below). At 0.40 there is no jump (1-point bins, 2016-24: 17.1% at
  0.39, 18.0% at 0.40). After the threshold fell to 0.25, sites with ISP 0.25-0.40 that
  newly elected CEP numbered 64, 27 and 16 in 2024-25, 2025-26 and 2026-27, against 53 and
  63 in the two years before, under the old threshold (through groups). The second
  schedule changes little in practice.
- **The received dose is set by the group, not by the site.** A CE claims one percentage
  for all its CEP sites (87-92% of multi-site CE-years). The claiming share implied by the
  reimbursement equals min(1, 1.6 x site ISP) within 1 point for only 21-46% of CEP
  site-years. It equals min(1, 1.6 x the CEP group's ISP, maximised over t-3..t) within
  1 point for 77-85% of CE-years in 2018-19 to 2025-26.
- **The offered dose is exact by construction, and it is an intention-to-treat dose.**
  Its effect runs through adoption: 6% of sites adopt at offered 0 (through groups),
  22% (pre) / 48% (post) at 0.64-0.70, and 79% / 89% at the cap. At offered 0.64-0.80,
  adoption roughly doubled between the pre and post years, so the offered dose's effect
  drifts over time.
- **Estimator.** In the supported cells, offered doses 0.64-0.85 against 0 raise lunch
  participation by 1 to 3.5 points (rotation sd 0.3-1.4 points). The cap (a = 1) against
  0 is -2 to -4 points in the bins where it is supported (score 0.55-1). The
  score-direction modifier test rejects strongly (-0.55, se 0.05; r2_absorbed 0.91), so
  the transport restriction fails.
- **What works.** R^2 of Q on X is 0.63, 34% of site-years sit at dose 0 and 25% at the
  cap. The outcome is continuous and in the same files, and the claims rule is visible
  in the reimbursement dollars.

## The rule (7 CFR 245.9(f))

Text from the govinfo CFR annual editions (`data/raw/cfr/`). The 2025 edition reads:

- (f)(3)(i): "A local educational agency, group of schools, or school must have an
  identified student percentage of at least 25 percent, as of April 1 of the school year
  prior to participating in the community eligibility provision ... Individual schools
  participating in a group may have less than 25 percent identified students, provided
  that the average identified student percentage for the group is at least 25 percent."
  The 2023 edition has "40 percent" in both places.
- (f)(4)(v)(A): "To determine the free claiming percentage, multiply the applicable
  identified student percentage by a factor of 1.6. The product of this calculation may
  not exceed 100 percent. The difference between the free claiming percentage and 100
  percent represents the paid claiming percentage." The applicable ISP is April 1 of the
  prior year in year 1. In years 2-4 of the 4-year cycle it is "the higher of" that and
  the ISP as of April 1 of the year before the cycle.
- (f)(4)(vi): "A 1.6 multiplier must be used for an entire 4-year cycle." (f)(1)(iii):
  the ISP may be computed for a school, a group of schools or the whole LEA.
- (f)(4)(ix): a grace year for ISP 15-25% in year 4. (f)(5)-(7): LEAs report lists by
  April 15; the state publishes eligible (>= 25%) and near-eligible (15-25%) schools and
  LEAs by May 1.
- **The 2023 change.** Final rule "Child Nutrition Programs: Community Eligibility
  Provision - Increasing Options for Schools", 88 FR 65778, 26 Sept. 2023: "This rule is
  effective October 26, 2023." The multiplier stayed at 1.6.
- **Texas timing.** TDA notice "Community Eligibility Provision (CEP) Data Report in
  TX-UNPS" (https://squaremeals.org/News/SNP-News/Article/3522/...): "This option will go
  into effect for SY 2024-2025, based on ISP data from April 1, 2024." CEs report counts
  between March 1 and 20, and TX-UNPS computes each school's ISP and group ISPs. FNS memo
  SP 22-2023 let states request waivers for mid-year 2023-24 elections. The data show no
  sign that Texas used one (CE-level adoption below 0.40 in 2023-24 is 0-7%, as before).

**Federal lunch rates recovered from the data.** The per-year OLS of non-CEP sites'
lunch reimbursement on free, reduced and paid lunch counts (`adapter.lunch_rates`) fits
99% of sites within 2 cents per lunch. The coefficients average the +2 cent rate of
high-need SFAs and include the performance-based add-on.

| PY (end year) | 2016 | 2017 | 2018 | 2019 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| free | 3.151 | 3.242 | 3.311 | 3.390 | 4.426 | 4.350 | 4.540 | 4.711 |
| paid | 0.352 | 0.356 | 0.363 | 0.366 | 0.852 | 0.481 | 0.511 | 0.533 |

The 2024-25 notice (89 FR 56720, govinfo FR-2024-07-10, 2024-15175) gives a free rate of
4.43 / 4.45 (below / at least 60% free and reduced) and a paid rate of 0.42 / 0.44, plus
9 cents performance-based. That is 4.52-4.54 and 0.51-0.53, which matches 4.540 and 0.511.
2022-23 includes the Keep Kids Fed Act supplement.

## Rule check (2026-11-12)

`adapter.rule_check()` covers CEP sites with more than 100 lunches. The implied share is
s = (reimbursement per lunch - paid rate) / (free rate - paid rate).
- **site**: min(1, 1.6 x site ISP).
- **group**: min(1, 1.6 x max over t-3..t of the enrollment-weighted ISP of the CE's CEP
  sites), one value per CE-year (median implied share).
- tol = 0.01.

| PY | CEP sites | site: within 0.01 / 0.03 | implied above / below site rule | CE-years | one share per CE | group: within 0.01 / 0.03 | implied >= current group rule |
|---|---:|---|---|---:|---:|---|---:|
| 2016 | 1,640 | 0.21 / 0.35 | 0.41 / 0.24 | 171 | 0.89 | 0.29 / 0.42 | 0.82 |
| 2017 | 1,654 | 0.23 / 0.41 | 0.32 / 0.28 | 191 | 0.89 | 0.37 / 0.47 | 0.67 |
| 2018* | 2,053 | 0.37 / 0.54 | 0.36 / 0.10 | 241 | 0.87 | 0.64 / 0.72 | 0.96 |
| 2019 | 2,678 | 0.46 / 0.57 | 0.32 / 0.11 | 328 | 0.92 | 0.83 / 0.91 | 0.98 |
| 2020 | 3,226 | 0.44 / 0.51 | 0.35 / 0.15 | 414 | 0.92 | 0.85 / 0.91 | 0.97 |
| 2021 | 2,589 | 0.41 / 0.50 | 0.38 / 0.13 | 461 | 0.91 | 0.81 / 0.90 | 0.95 |
| 2022 | 381 | 0.36 / 0.42 | 0.40 / 0.19 | 88 | 0.90 | 0.81 / 0.85 | 0.94 |
| 2023 | 4,202 | 0.41 / 0.50 | 0.30 / 0.21 | 566 | 0.91 | 0.77 / 0.85 | 0.97 |
| 2024 | 4,598 | 0.43 / 0.55 | 0.28 / 0.17 | 626 | 0.91 | 0.80 / 0.87 | 0.98 |
| 2025 | 4,804 | 0.39 / 0.50 | 0.33 / 0.18 | 685 | 0.90 | 0.80 / 0.88 | 0.98 |
| 2026 | 4,831 | 0.28 / 0.37 | 0.49 / 0.14 | 701 | 0.89 | 0.79 / 0.88 | 0.99 |

\* PY 2017-18's `SiteISP` is identical to PY 2016-17's for all 8,145 sites in both files.
TDA carried it forward, so 2018 is left out of the dose sample (`STALE_ISP_YEARS`).

- **What the table shows.** The implied share is a CE-level number. It is never far
  below 1.6 x the current group ISP, as the "higher of" rule implies. It exceeds the
  current rule mostly where ISPs fell. In 2025-26, site ISPs dropped (mean 0.46 against
  0.50) while cycles kept the earlier ISP.
- **Why 2016-17 are worse.** Cycles that began in 2014-15 or 2015-16 used ISPs before the
  first file.
- **Sub-district groups** are not identified in the files. The pooled group here is all
  CEP sites of the CE.

## Adoption by ISP

Share of public and charter sites with CEP = Y (TDA participation files), by site ISP:

| ISP | 2016-19 | 2020-22 | 2023-24 | 2025 | 2026 | 2027 |
|---|---:|---:|---:|---:|---:|---:|
| 0-0.15 | 2 | 6 | 5 | 7 | 6 | 4 |
| 0.15-0.20 | 1 | 4 | 4 | 4 | 6 | 5 |
| 0.20-0.25 | 1 | 3 | 5 | 8 | 11 | 12 |
| 0.25-0.30 | 3 | 6 | 8 | 10 | 18 | 20 |
| 0.30-0.35 | 6 | 9 | 13 | 21 | 35 | 35 |
| 0.35-0.40 | 8 | 16 | 24 | 32 | 43 | 50 |
| 0.40-0.45 | 18 | 26 | 32 | 42 | 56 | 60 |
| 0.45-0.50 | 24 | 36 | 48 | 58 | 69 | 73 |
| 0.50-0.55 | 39 | 53 | 62 | 67 | 79 | 80 |
| 0.55-0.625 | 50 | 66 | 74 | 79 | 86 | 85 |
| 0.625-0.70 | 64 | 73 | 83 | 87 | 89 | 88 |
| 0.70-0.80 | 71 | 82 | 87 | 88 | 92 | 94 |
| 0.80-1 | 73 | 82 | 91 | 89 | 89 | 88 |

- **Below-threshold sites.** The 2026-27 participation file (no claims yet) is included.
  Sites under the threshold adopt through groups and through cycles that started at a
  higher ISP.
- **The 2025-27 rise at 0.25-0.40** comes mostly from existing CEP sites whose ISP fell.
  New CEP sites (CEP = Y, previous year N) by ISP band:

| PY | new CEP sites | ISP < 0.25 | 0.25-0.40 | >= 0.40 | sites leaving CEP |
|---|---:|---:|---:|---:|---:|
| 2023 | 556 | 4 | 53 | 499 | 52 |
| 2024 | 435 | 9 | 63 | 363 | 22 |
| 2025 | 275 | 4 | 64 | 207 | 12 |
| 2026 | 123 | 2 | 27 | 94 | 36 |
| 2027 | 53 | 0 | 16 | 37 | 117 |

- **CE level.** CEs electing CEP for all their sites, by the CE's mean site ISP
  (`adoption(level="ce")`). Below 0.40 this is 0-7% in every year 2016-2024. At 0.40-0.45
  it is 9-26%, and at 0.625 and above 39-89%. In 2026 and 2027, CEs at 0.35-0.40 reach 42%
  and 46%, and CEs at 0.25-0.30 reach 9% and 18%.
- **Is the offered (intention-to-treat) dose meaningful?** It is meaningful as the price
  the program offers. It is a weak instrument for the schedule.
  - Adoption among sample sites at offered 0 is 5.7% (pre) / 6.4% (post).
  - It rises with the offered share: pre 22 / 33 / 50 / 64 / 79% for 0.64-0.70 / 0.70-0.80
    / 0.80-0.90 / 0.90-1 / the cap; post 16% for (0, 0.5], 33% for 0.5-0.6, 40% for
    0.6-0.64, then 48 / 63 / 76 / 83 / 89%.
  - corr(offered, received) = 0.62.
  - The jumps at the two thresholds are small next to the slope in Q.

## Dose design

`load_dose(outcome="lunch", years=(2016, 2017, 2019, 2023, 2024, 2025, 2026), dose="offered")`
returns an `RDDSample`.

| Key | Content |
|---|---|
| `Q`, `threshold` | site ISP (0-1); 0.40 (`treatment_rule` uses 0.40 / 0.25 by year) |
| `extras["A"]` | offered share 1{Q >= c_t} min(1, 1.6 Q); with `dose="received"`, CEP x implied share |
| `extras["a_ref"]`, `["dose_levels"]` | 0.0; None |
| `extras["schedule"]` | `adapter.schedule(q, py, unit="share")` reproduces `A` exactly |
| `extras["group"]`, `["regime"]` | program year (2025 = 2024-25); threshold c_t |
| `extras["Z_names"]` | `elem`, `log_enr`, `charter` |
| `extras["ce_id"]`, `["site_id"]`, `["campus_id"]` | TDA contracting entity (cluster on it); CE-site; TEA 9-digit campus id |
| `extras["cep"]`, `["A_offered"]`, `["A_received"]`, `["s_impl"]` | CEP flag; both doses; implied claiming share (CEP sites) |
| `extras["isp_grp_max4"]`, `["isp_ce"]` | max over t-3..t of the CEP group's ISP; CE-wide ISP |
| `extras["y_lunch"]`, `["y_breakfast"]`, `["y_meals"]`, `["enrollment"]`, `["df"]` | outcomes, average monthly enrollment, panel rows |

- **Sample.** Public and charter sites with an ISP, at least 6 claim months, enrollment
  of 20 or more, and Y in (0, 1.5].
- **Years left out.** Program years 2019-20 to 2021-22 (COVID closures and nationwide
  free-meal waivers: the 2021-22 claims file has 2,635 sites) and 2017-18 (stale ISP).
- **Doses.** Offered doses in (0, 0.64) exist only after the threshold change: 3,102
  site-years with ISP 0.25-0.40 in 2024-25 and 2025-26.

## Readiness (2026-11-12)

`adapter.readiness(load_dose())`:
- **Size and fit.** n = 57,310, p = 39. R^2 of Q on X is 0.633 (0.286 without the
  leave-one-out CE ISP; 0.59-0.64 within each year). sd(Q) = 0.216, sd(T) = 0.172,
  sd(eta) = 0.131.
- **Doses.** 34.4% of the sample is at dose 0 and 24.7% at the cap.
- **Outcome.** Y has mean 0.628, sd 0.187, and corr(Y, Q) = 0.65.

| Offered dose | n | eta 10% | 50% | 90% |
|---|---:|---:|---:|---:|
| 0 | 19,713 | -0.249 | -0.058 | 0.048 |
| 0.400-0.652 | 3,758 | -0.176 | -0.036 | 0.066 |
| 0.652-0.718 | 3,756 | -0.131 | -0.004 | 0.107 |
| 0.718-0.784 | 3,757 | -0.111 | 0.006 | 0.116 |
| 0.784-0.851 | 3,762 | -0.092 | 0.018 | 0.136 |
| 0.851-0.918 | 3,753 | -0.076 | 0.035 | 0.157 |
| 0.918-0.984 | 3,771 | -0.061 | 0.041 | 0.169 |
| 0.984-1 (cap) | 15,040 | -0.020 | 0.078 | 0.231 |

**Bunching.** Counts are site-years 1 point below / above each point.
- **Before the change, at 0.40:** 566 / 650 (ratio 1.15). Placebos: 1.00 at 0.30, 1.02 at
  0.35, 1.02 at 0.45, 1.15 at 0.50.
- **After the change, at 0.25:** 159 / 144 (0.91). Placebos: 1.06 at 0.20, 0.90 at 0.30,
  1.05 at 0.35, 1.09 at 0.40.
- TX-UNPS computes the ISP from the identified and enrolled counts that CEs report.

**Persistence.** The sd of the year-on-year change in site ISP is 0.07-0.12. 83-94% of
sites are on the same side of 0.40 as the year before.

## Estimator (2026-11-12)

Command as in `dose_config.py`. Log: `experiments/runs/dose_schedule/cep_meals.log`.
- **Setup.** Spline in the dose with l = 2 interior knots. K = 7, k = 4. Folds and SEs are
  clustered by CE. Score bins are those of `dose_config.EDGES`.
- **Reading the cells.** Each cell is the effect against offered 0 on lunch participation
  (share of enrolled students eating lunch per day), shown as estimate (rotation sd,
  coverage).

| Score bin | a = 0.55 | 0.64 | 0.75 | 0.85 | 1 (cap) |
|---|---|---|---|---|---|
| 0.15-0.25 | 0.023 (0.008, 94%) | 0.030 (0.006, 89%) | 0.018 (0.031, 82%) | 0.048 (0.014, 74%) | 0.368 (0.413, 55%) |
| 0.25-0.30 | 0.018 (0.010, 95%) | 0.022 (0.007, 93%) | 0.021 (0.019, 90%) | 0.041 (0.014, 82%) | 0.190 (0.219, 63%) |
| 0.30-0.35 | -0.057 (0.106, 95%) | 0.018 (0.008, 94%) | 0.027 (0.009, 92%) | 0.040 (0.013, 86%) | 0.047 (0.033, 67%) |
| 0.35-0.40 | -0.099 (0.168, 94%) | 0.013 (0.007, 95%) | 0.025 (0.009, 94%) | 0.035 (0.013, 90%) | 0.006 (0.011, 76%) |
| 0.40-0.45 | -0.121 (0.191, 92%) | 0.011 (0.008, 94%) | 0.021 (0.009, 94%) | 0.030 (0.013, 91%) | 0.001 (0.017, 79%) |
| 0.45-0.50 | -0.775 (1.108, 90%) | 0.003 (0.018, 94%) | 0.019 (0.009, 95%) | 0.027 (0.013, 93%) | -0.002 (0.011, 83%) |
| 0.50-0.55 | -0.826 (1.172, 86%) | 0.000 (0.018, 92%) | 0.015 (0.008, 95%) | 0.023 (0.012, 95%) | -0.009 (0.014, 88%) |
| 0.55-0.625 | -1.583 (2.222, 80%) | -0.010 (0.030, 89%) | 0.012 (0.006, 94%) | 0.019 (0.010, 95%) | -0.019 (0.019, 93%) |
| 0.625-0.70 | -2.686 (3.760, 74%) | -0.023 (0.049, 84%) | 0.010 (0.005, 90%) | 0.016 (0.008, 93%) | -0.022 (0.017, 96%) |
| 0.70-0.80 | -6.466 (9.052, 66%) | -0.066 (0.111, 78%) | 0.008 (0.003, 86%) | 0.013 (0.005, 90%) | -0.026 (0.018, 97%) |
| 0.80-1 | -34.7 (48.7, 46%) | -0.386 (0.584, 61%) | 0.007 (0.015, 72%) | 0.004 (0.021, 79%) | -0.044 (0.028, 90%) |

Note: the printed bin labels in the log round to one decimal. The rows above are the
`EDGES` 0.15, 0.25, 0.30, ...

- **Modifier test.** coef -0.545 (se 0.050), r2_absorbed 0.913, n_fit 35,341: rejects. For
  T higher by 0.1, the effect of offered 1 against 0 is 5.5 points lower.
- **a = 0.45 and 0.55** are held only by post-period sites with ISP 0.25-0.37. From score
  0.30 up the rotations disagree (sd 0.1-9), so these cells are not supported even where
  coverage reads above 90%. Coverage checks eta overlap, not the number of units at the
  dose. In bins 0.15-0.30, a = 0.55 is +0.018 to +0.023 (sd 0.008-0.010). With l = 3,
  a = 0.40-0.52 explodes from score 0.30 up.
- **Seed 1** (`cep_meals_variant_l2_seed1.log`): 0.64-0.85 at bins 0.25-0.50 is +0.009 to
  +0.056 (seed 0: +0.003 to +0.041). The cap at bins 0.55-1 is -0.009 to -0.031.
  Modifier -0.512 (se 0.053).
- **Pre-threshold years only** (2016, 2017, 2019, 2023, 2024; one schedule with a jump and
  a cap; `cep_meals_variant_l2_pre.log`): 0.75-0.85 at bins 0.30-0.55 is +0.010 to +0.036.
  0.64 is unstable below 0.40 (rotation sd 0.06-0.27). The cap against 0 is -0.03 to
  -0.05 at 0.55-1 and explodes below. Modifier -0.496 (se 0.057), r2_absorbed 0.90. The
  jump and the cap give the test power without the second schedule.
- **Breakfast** (`cep_meals_variant_l2_breakfast.log`; mean 0.385): 0.64-0.85 at bins
  0.25-0.55 is +0.027 to +0.068. The cap is +0.016 to +0.018 at 0.55-0.80 and -0.007 at
  0.80-1. Modifier -0.242 (se 0.099).
- **Received dose** (`dose="received"`, chosen, not a schedule;
  `cep_meals_variant_received.log`): CEP at a share of 0.75-1 against no CEP is +0.03 to
  +0.12 lunch participation, falling with the ISP. a = 0.64 explodes, since few CEP sites
  are near 0.64. Modifier -0.371 (se 0.038).

Reproduce a variant: `app.DATASETS['x'] = dose_config.variant(outcome="breakfast")`, or
`variant(years=(2016, 2017, 2019, 2023, 2024))`, or
`variant(dose="received", a_grid=[0, .64, .75, .85, .95, 1])`.
Then call `app.run('x', ...)`.

## Data and links

**Site participation** (TDA "Contact Information and Site-Level Program Participation",
one dataset per PY 2015-16..2026-27):
- 8,450-8,700 meal sites a year, of which 8,160-8,500 are public or charter.
- `SiteISP` is non-missing for 96-99% of sites. Columns: `CEP` Y/N, `Provision2`, the
  grade flags, ESC region, county-district code and FSMC.
- PY 2015-16 to 2017-18 store the ids as numbers, so the zero padding is restored.

**Claims** (TDA "Meal Reimbursement Information", PY 2015-16..2025-26):
- Monthly site claims: enrollment, free/reduced/paid eligibility, meals by category and
  reimbursement dollars. Fetched as site x year aggregates.
- For CEP sites every meal is reported in the "free" served column, with all enrollment
  free-eligible. The claiming split shows only in the dollars.
- The 2025-26 dataset stores the dollar columns as text; they are cast in the query.

**Linking.**
- Participation and claims join on (program year, CEID, SiteID) for 99.4-99.7% of
  participation sites, except the COVID years (2020-21: 84%; 2021-22: 30%).
- TEA campus id = county-district code (6 digits) + the last 3 digits of SiteID. It is
  found in TEA's AskTED (May 2026, active campuses only) for 90% of PY 2015-16 sites and
  98.5% of PY 2025-26 sites, and 97% of matched names agree.
- AskTED carries the NCES school id, so CCD and TAPR files can be joined by campus id.

## Caveats

- **The dose is the federal claiming share.** It sets the school's revenue. Students at a
  CEP school eat free whatever the share, so a share affects participation only through
  adoption and the school's budget.
- **Adoption is chosen, depends on history and grows over time.** Cycles are 4 years,
  with "higher of" ISPs, grace years, grouping, and adoption doubled at a given offered
  dose between 2016-19 and 2025-26. The model's f(eta, a) has no time index.
- **The received share is a group number.** Sub-district groups and cycle-start years are
  not in the files. The 4-year max is an approximation (77-85% within 1 point).
- **Timing.** The ISP is from April 1 of the prior year. Y is the same program year as
  the dose. PY 2022-23 has the Keep Kids Fed Act supplement (+40 cents per lunch), which
  changes the finances of adoption.
- **Texas policy change.** squaremeals.org links a TDA page "Reduced-Price School Meals Now
  Free" (not read) and a news report says the legislature funded it. This would change
  non-CEP participation in the latest years; the start date was not verified.
- **Falling ISPs.** The mean site ISP was 0.50 in 2024-25, 0.46 in 2025-26 and 0.44 in
  2026-27. The cause was not checked.
- **Missing covariates.** Race and ethnicity shares and urbanicity are not in X (see
  author steps). Without the leave-one-out CE ISP, R^2 falls to 0.29.

## Getting the data

    # from code/ (the repository root); never run Python inside data/
    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; runpy.run_module('experiments.datasets.cep_meals.download', run_name='__main__')"

- **Access.** Fetched on 2026-11-12 with `urllib` GETs, with no token, login, form,
  cookie or challenge, at 1.5 s between requests.
- **Size.** 51 files, 70.7 MB (`data/raw/MANIFEST.tsv` has the SHA-256 of each). The panel
  cache `data/processed/cep_meals_panel.csv.gz` is 14 MB and builds in about 7 s; delete
  it or call `panel(rebuild=True)` after changing `adapter.py`.

| data/raw/ | Source | Bytes | robots.txt |
|---|---|---:|---|
| `tda_participation/part_<py>_<id>.csv` (12) | `https://data.texas.gov/resource/<id>.csv?$select=...` (ids in `download.PARTICIPATION`); contact-person columns not requested | 41,039,877 | data.texas.gov: `Crawl-delay: 1`; `/resource/`, `/api/views/` allowed (only `/api/odata/`, `/api/collocate*`, browse/catalog facets disallowed) |
| `tda_claims/claims_<py>_<id>.csv` (11) | same host, SoQL `$group=ceid,siteid` aggregates (`download.CLAIMS`) | 24,955,093 | same |
| `tda_meta/*.json` (23) | `https://data.texas.gov/api/views/<id>.json` (column definitions) | 2,126,785 | same |
| `tea_askted/askted_hzek-udky.csv` | data.texas.gov TEA "AskTED Data - May 12, 2026", non-personal columns | 2,267,542 | same |
| `cfr/` (4) | govinfo CFR 2023/2024/2025 title 7 vol 4 sec. 245.9 XML; FR 2023-20294 HTML | 295,018 | www.govinfo.gov: only `/core/`, `/profiles/`, README files disallowed |

Not used:
- **www.ecfr.gov.** Its robots.txt disallows `/api/versioner/v1/full/` (one request was
  made there before the robots check; it returned an error and was deleted), and its pages
  answer scripts with a 302 to `unblock.federalregister.gov`.
- **educationdata.urban.org.** Its robots.txt disallows `/api/`. Two test requests were
  made before the robots check and nothing was saved.
- **TEA TAPR** (rptsvr1.tea.texas.gov). Data come from a download form (not submitted).

## Author steps

| What | Why | Where |
|---|---|---|
| TEA TAPR campus files 2016-2026 (attendance rate, STAAR, enrollment by race/ethnicity) | attendance and test-score outcomes; race shares for X | the Advanced Data Download form on each year's page, e.g. https://rptsvr1.tea.texas.gov/perfreport/tapr/2024/index.html (submitting the form was not done); save CSVs into `experiments/datasets/cep_meals/data/raw/tea_tapr/<year>/` |
| or NCES CCD school membership (race) and EDGE locale files for Texas | race shares, urbanicity for X | https://nces.ed.gov/ccd/files.asp (robots.txt allows; national membership zips are large, about 100+ MB per year; say if a script should fetch one year) |
| TDA CEP eligibility / near-eligibility lists (LEA and group ISPs, year of cycle) | the exact group ISP and cycle start that set the received share | not found on squaremeals.org by search; TDA Food and Nutrition (squaremeals.org) would need to say where the May 1 lists are posted |
