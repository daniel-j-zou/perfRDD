# ESEA Title I-A Targeted Grants: formula weight on the SAIPE child-poverty share, school districts

| Field | Value |
|---|---|
| **Q** | Census SAIPE share of children aged 5-17 in poverty in the district, `pov517 / pop517`, income year t-3 (the estimate ED uses for the allocation received in F-33 fiscal year t) |
| **Threshold** | 0.05: Targeted Grant eligibility (q >= 5% and at least 10 formula children; the default sample has pop517 >= 200, so the count rule never binds above 5%) |
| **Treatment** | Dose `A` = formula weighted children per school-age child, `g(q) 1{q >= 0.05}`; `a_ref = 0`. Kinks at 15.58, 22.11, 30.16, 38.24%; jump from 0 to 0.05 at 5%. `D = 1{q >= 0.05}` |
| **Rule class** | Convex piecewise-linear schedule with an eligibility jump (ESEA sec. 1125). A second, number scale (on the count of formula children) binds for large districts (`dose="max"`). The state expenditure factor gives one dollar schedule per state-year (`dose="dollars"`) |
| **X** | log enrolment (t-1), log current spending per pupil (t-1), race shares (black, Hispanic, Asian, other; CCD fall 2014), NCES locale (12 codes), elementary/secondary-only district, 49 state and 3 year indicators (71 columns); `lag_q=True` adds last year's SAIPE share |
| **Y** | `load_dose(outcome=...)`: `cur_pp` (default; F-33 current spending per pupil in the year the money is received), `inst_pp`, `log_cur_pp`, `d_cur_pp` (change from t-1), `title1_pp` (first stage), `fedrev_pp`; `lead=1` for the following year |
| **n** | 43,205 district-years (10,958 districts), F-33 fiscal years 2016-2019 |
| **Source** | Census SAIPE school-district estimates; Census Annual Survey of School System Finances (F-33); NCES Common Core of Data (LEA membership) and EDGE geocodes (locale); ESEA sec. 1125 as summarized in CRS R45141 |

**Fit: moderate. The rule is sharp and clean, but the dose is a small, fuzzy part of actual Title I money.**
- **The rule is clean.** `g(q)` is statutory and depends on the score alone. SAIPE
  estimates are model-based, so districts cannot sort, and there is no bunching at 5% or
  at any kink.
- **X predicts Q partly.** R^2 of Q on X is 0.59, and sd(eta) is 6.3 points. With last
  year's SAIPE share in X, R^2 is 0.92 and sd(eta) is 2.7 points. That eta is mostly
  year-to-year SAIPE estimation noise, a plausibly as-good-as-random component.
- **The reference region is small.** Only 5.0% of district-years are below 5% (2,167),
  and they are affluent, mostly suburban districts.
- **The formula dose explains little of the actual money beyond a straight line in q.**
  - On F-33 Title I revenue per school-age child, the R^2 is 0.523 for A and 0.525 for q
    linear. Together they reach 0.534.
  - Last year's Title I revenue (hold harmless) gives 0.80.
  - The convex part is in the data: given q, one unit of A adds $334 (cluster se 35) of
    Title I revenue per school-age child.
  - Title I is $262 per pupil against $12,463 of current spending. The Targeted Grant is
    about a quarter of Title I-A.
- No published Targeted Grant by district is in hand (`published_dose = None`; see AUTHOR
  TO-DO).

## Dose design

`load_dose(years=(2016, 2017, 2018, 2019), outcome="cur_pp", dose="weight", lead=0, lag_q=False)`

| `dose` | A | Depends on | Range in sample (10/50/90%) |
|---|---|---|---|
| `"weight"` (default) | `g(q) 1{q >= 0.05}`: percentage-scale weighted formula children per school-age child | q only | 0.066 / 0.177 / 0.524 (max 2.04) |
| `"max"` | `max(g(q), h(q N) / N) 1{eligible}`, N = pop517: the larger of the two scales | q and N | 0.067 / 0.186 / 0.535 |
| `"dollars"` | weight x state factor: 0.4 x state current spending per pupil, bounded to [0.8, 1.2] x national (F-33, year t-3). Authorized dollars per school-age child before the ratable reduction, which is common to all districts in a year | q, state-year | $305 / $756 / $2,092 |

`g` has slope 1 on [0, 15.58%], 1.75 to 22.11%, 2.5 to 30.16%, 3.25 to 38.24% and 4 above.
So g(0.05) = 0.05, g(0.1558) = 0.1558, g(0.2211) = 0.2701, g(0.3016) = 0.4713,
g(0.3824) = 0.7339 and g(0.5) = 1.2043.

The number scale `h` has slopes 1, 1.5, 2, 2.5 and 3 on [0, 691], [691, 2,262],
[2,262, 7,851], [7,851, 35,514] and above, in formula children. It binds for 12.3% of
district-years, almost all in the largest size quintile (61% of districts with
pop517 > 4,705).

- **`extras`.**
  - `A`, `a_ref` (0), `dose_levels` (None).
  - `schedule(q, group, pop517=None)` reproduces `A` exactly for all three `dose`
    options. `pop517` is needed only for `"max"`.
  - `group` is `"<ST>_<year>"`, and `state`, `year` are also stored.
  - `Z_names`: log_enroll_lag, log_cur_pp_lag, sh_black, sh_hisp.
  - `leaid` (NCES 7-digit), `published_dose` (None).
  - `title1_pp`, `title1_per_child`: actual F-33 Title I revenue in year t.
  - `A_weight`, `A_max`, `A_dollars`, `num_scale_binds`, `pop517`, `pov517`,
    `appe_factor`, `q_lag1`, `kinks`, and `frame` (the full analysis DataFrame).
- **Second schedules.** In dollars, the state factor makes 51 x 4 state-year schedules.
  At a fixed q the dose varies by state (factor $3,424-$5,136 per weighted child at the
  authorized level). The number scale gives large districts a different, higher schedule.
- **Other doses at nearby scores.** Basic Grants start at q > 2%. Concentration Grants
  jump on at q > 15% (or more than 6,500 formula children). Education Finance Incentive
  Grants (EFIG) use Targeted-type weights within states. All of these move actual Title I
  money near the same scores.

## The rules

- **Formula (ESEA sec. 1125; CRS R45141, Table 2).**
  - The grant is the weighted count x the state expenditure factor x a national ratable
    reduction.
  - Formula children are the SAIPE poor children aged 5-17, plus neglected/delinquent,
    foster and TANF-above-poverty children. Those extra counts are a few percent and are
    not in public files; the dose uses SAIPE alone.
  - The expenditure factor is 0.4 x the state's average per-pupil expenditure, bounded
    to 80-120% of the national average. ED's definition (ESEA sec. 8101) is current
    expenditure per pupil in average daily attendance in the third preceding year. Here
    it is F-33 current spending per fall member, Census year t-3.
- **Eligibility.** At least 10 formula children and at least 5% of the 5-17 population.
- **Hold harmless.** 85, 90 or 95% of the prior-year amount, depending on the poverty
  rate. It is not modelled and makes actual dollars sticky.
- **Timing.** ED fiscal year t-1 funds (school year t-1 to t, Census F-33 fiscal year t)
  use SAIPE income year t-3; for example, F-33 2016 uses SAIPE 2013.
  - `timing_check()` correlates F-33 Title I per child in year t with the formula weight
    from SAIPE t-1 ... t-4: 0.694, 0.708, 0.720, 0.724.
  - t-3 and t-4 are close. Title I money is drawn down over 27 months, so F-33 revenue
    mixes two allocation years. SAIPE is also highly autocorrelated.

## Rule check: formula dose vs actual Title I revenue (2026-10-28)

`adapter.rule_check()`. Y is F-33 item C14 (Title I revenue through the state, year t) per
school-age child. Same sample as `load_dose()` (n = 43,205):

| Regressors | R^2 |
|---|---:|
| q (linear) + year | 0.525 |
| A = g(q) 1{q >= 5%} + year | 0.523 |
| A + q + year | 0.534 |
| A_max (number scale) + q + year | 0.532 |
| A x state factor + year | 0.557 |
| Part A proxy: Basic q 1{q > 2%}, Concentration q 1{q > 15% or > 6,500}, Targeted A_max, each x state factor, + year | 0.564 |
| Part A proxy + state x year | 0.603 |
| lagged Title I revenue (hold harmless) + year | 0.804 |
| lagged Title I + Part A proxy + state x year | 0.828 |

- **Correlations.** corr(A, actual Title I per school-age child) = 0.72; per pupil, 0.66.
- **Per weighted child.** OLS of actual per child on A, q and year dummies: $334 per unit of
  A (cluster se 35), the convex Targeted/EFIG part.
- **Segment slopes.** With a linear spline in q (knots 2, 5, 15, 15.58, 22.11, 30.16,
  38.24%), the slopes are $1,250-1,650 per unit of q below 22%, $1,495 on 22-30%, and
  $1,934-2,327 above 30%. The money gets steeper at high poverty, as the weights do, but
  much less sharply.
- **Jump at 5%.** The spline's jump at 5% is small: $5.6 per child. If the $334 slope
  applied, the jump would be 0.05 x $334 = $17.
- **Zero revenue.** 2.6% of eligible district-years report zero Title I revenue.
- **Why actual money is fuzzy.**
  - C14 pools Basic, Concentration, Targeted and EFIG grants, plus hold-harmless amounts.
  - States re-allocate: alternative poverty data for districts under 20,000 people,
    charter LEAs, boundary changes, and the 7% school-improvement reserve.
  - The formula counts include children who are not in SAIPE.
  - Money is drawn down across fiscal years.

## Readiness (2026-10-28)

`python -m experiments.datasets.title1.readiness` (eta = Q - OLS fit of Q on X; n = 43,205):

| X | p | R^2(Q on X) | sd(Q) | sd(T) | sd(eta) | Share at dose 0 |
|---|---:|---:|---:|---:|---:|---:|
| base | 71 | 0.591 | 0.098 | 0.076 | 0.063 | 0.050 |
| base + last year's SAIPE share (`lag_q=True`) | 72 | 0.923 | 0.098 | 0.094 | 0.027 | 0.050 |

By year (base X): R^2 / sd(eta) / share at dose 0:
- 2016: 0.588 / 0.065 / 0.040
- 2017: 0.583 / 0.063 / 0.043
- 2018: 0.595 / 0.062 / 0.052
- 2019: 0.602 / 0.060 / 0.066

eta quantiles by dose decile (dose = A, weighted children per school-age child):

| Dose bin (A) | n | eta 10% | eta 50% | eta 90% | `lag_q` eta 10% | 50% | 90% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (q < 5%) | 2,167 | -0.087 | -0.040 | 0.000 | -0.016 | -0.005 | 0.004 |
| 0.050-0.080 | 4,104 | -0.089 | -0.042 | 0.000 | -0.021 | -0.005 | 0.008 |
| 0.080-0.105 | 4,104 | -0.090 | -0.039 | 0.008 | -0.027 | -0.006 | 0.011 |
| 0.105-0.128 | 4,104 | -0.081 | -0.030 | 0.017 | -0.029 | -0.004 | 0.015 |
| 0.128-0.151 | 4,103 | -0.076 | -0.016 | 0.026 | -0.029 | -0.003 | 0.019 |
| 0.151-0.188 | 4,105 | -0.074 | -0.006 | 0.041 | -0.031 | -0.002 | 0.024 |
| 0.188-0.233 | 4,105 | -0.065 | 0.008 | 0.055 | -0.032 | 0.000 | 0.029 |
| 0.233-0.292 | 4,103 | -0.059 | 0.017 | 0.073 | -0.033 | 0.001 | 0.034 |
| 0.292-0.385 | 4,104 | -0.042 | 0.030 | 0.093 | -0.032 | 0.004 | 0.040 |
| 0.385-0.537 | 4,104 | -0.028 | 0.046 | 0.119 | -0.034 | 0.008 | 0.049 |
| 0.537-2.04 | 4,104 | -0.007 | 0.081 | 0.180 | -0.031 | 0.017 | 0.069 |

- **Base X.** Units at the same dose span about 0.09-0.19 in eta (10-90%), but the median
  eta rises with the dose (-0.04 to +0.08). The top decile and the reference group
  overlap only in the tails.
- **With `lag_q`.** The eta distributions of all doses nearly coincide, but they are
  narrow (2.7 points). Transport then rests on SAIPE noise, and T carries most of q.

**Bunching.** District-years in [c - w, c) and [c, c + w), w = 0.5 / 1 percentage point:

| Point | Below / above (w = 0.5) | Ratio | Ratio (w = 1) |
|---|---|---:|---:|
| 5.00 (eligibility) | 518 / 610 | 1.18 | 1.28 |
| 15.58 | 941 / 902 | 0.96 | 0.96 |
| 22.11 | 713 / 640 | 0.90 | 0.93 |
| 30.16 | 330 / 327 | 0.99 | 0.91 |
| 38.24 | 144 / 140 | 0.97 | 0.93 |

- **Placebo points.** 7.5, 10, 12.5, 19, 26, 34, 42, 46 and 50% give ratios of 0.73-1.14
  (w = 0.5) and 0.77-1.06 (w = 1).
- **The 5% ratio is the slope of the density.** Near 5% the density rises steeply.
  Placebo points at 3.5, 4, 4.5, 5.5, 6 and 6.5% give 1.04-1.34 (w = 0.5) and 1.08-1.59
  (w = 1), and 5% is inside both ranges.
- **No sorting.** None is expected, since SAIPE estimates are model-based.

**Outcome noise.** Y = current spending per pupil has sd $4,084. Given X and a cubic in q,
the residual sd is $1,260 (R^2 0.905); for log spending it is 0.069. The formula
convexity is worth about $330 per weighted child in actual money. Dose differences
within the supported range are therefore $20-150 per child, roughly 0.02-0.12 residual
sd. The design is underpowered unless pass-through is large.

## Sample

- **Merge.** `panel()` joins every SAIPE district (income year t-3) to F-33 year t by
  NCES LEA id; 98-99% of SAIPE districts match.
- **`analysis_frame()` keeps:**
  - F-33 school level 01/02/03 (elementary, secondary, unified).
  - pop517 >= 200.
  - Enrolment >= 100 in t-1, t and t+lead.
  - Current spending per pupil of $3,000-60,000 in t-1 and t+lead.
  - CCD race shares and the previous SAIPE share present.
- **Years.** 2016-2019 by default: F-33 fiscal years, i.e. school years 2015-16 to
  2018-19, with SAIPE 2013-2016.
  - FY 2020 (spring-2020 closures) and FY 2021 (ESSER, which was allocated in proportion
    to Title I shares) are available but not in the default.
  - `lead=1` with 2019 uses FY 2020 spending.
- **Covariates.**
  - Race shares come from the CCD fall-2014 membership file (fall 2015 for 10 districts).
    They are fixed over the panel and predate every receipt year.
  - Locale is the NCES EDGE code for school year t-1. 2016-17 is used for earlier years,
    since EDGE 2015-16 has locale only in an xlsx supplement.

## Getting the data

    python -m experiments.datasets.title1.download            # everything (109 MB)
    python -m experiments.datasets.title1.download --check

Run Python from the repo root (`... -I -c "import sys; sys.path.insert(0, '.'); import runpy; runpy.run_module('experiments.datasets.title1.download', run_name='__main__')"`).

- **Access.** All files are direct links with no login, form, agreement or CAPTCHA
  (checked 2026-10-28).
- **Zips.** They are kept as zips and read in memory; nothing is extracted or executed.

| Folder | Files | Bytes | Source |
|---|---|---:|---|
| `saipe/` | `ussd11.txt`-`ussd17.txt` (income years 2011-2017) + layouts | 12.4 M | https://www2.census.gov/programs-surveys/saipe/datasets/<y>/<y>-school-districts/ |
| `f33/` | `elsec13.txt`-`elsec21.txt` (all items, $1,000), `school15doc.pdf`, `school19doc.doc` | 67.5 M | https://www2.census.gov/programs-surveys/school-finances/tables/<y>/secondary-education-finance/ |
| `ccd/` | LEA directory (029) and wide membership (052), 2014-15 and 2015-16 | 7.4 M | https://nces.ed.gov/ccd/data/zip/ (URLs from the files.asp JSON API) |
| `edge/` | `EDGE_GEOCODE_PUBLICLEA_{1617,1718,1819}.zip` | 21.3 M | https://nces.ed.gov/programs/edge/data/ |

- **Item codes (F-33 documentation).**
  - V33: fall membership.
  - C14: federal revenue through the state, Title I.
  - TCURELSC: current spending, elementary-secondary.
  - TCURINST: current spending on instruction.
  - TFEDREV: federal revenue.
- **F-33 file quirks.** Each line ends with a trailing comma, so the file is read with
  `index_col=False`. The `STATE` column is the Census Bureau's alphabetical state code,
  not FIPS; the FIPS code is the first two digits of `NCESID`.
- **CCD long files.** The CCD LEA membership files for 2016-17 onward are long (about
  50 MB each). The server stalled on them on 2026-10-28, so they are not in
  `download.py`.

## AUTHOR TO-DO

1. **SEDA achievement outcomes (edopportunity.org).**
   - The "Get the Data" page (https://edopportunity.org/opportunity/data/) asks for an
     e-mail address, and submitting it accepts the SEDA Data Use Agreement. The
     agreement bars publishing the files.
   - Not downloaded. If you want test-score outcomes, enter your e-mail there, accept the
     DUA, and download the district-level (geographic or administrative district)
     achievement files (CSV).
   - Save them under `experiments/datasets/title1/data/raw/seda/` and tell the session
     the file names. The adapter will merge on the NCES LEA id.
2. **ED Title I LEA allocations by grant type.**
   - These are the published Basic, Concentration, Targeted and EFIG amounts by district,
     FY 2015-2018. They would give `published_dose`: the actual Targeted Grant, rather
     than all of C14.
   - ED posts them on its budget pages (the Title I allocation pages on www2.ed.gov,
     under `/about/overview/budget/titlei/`). www2.ed.gov refuses scripted requests, and
     earlier sessions were told not to work around that; the pages were not opened here.
   - Download the state-by-state LEA allocation spreadsheets in a browser and save them
     under `experiments/datasets/title1/data/raw/ed_allocations/`.

## Caveats

- **Formula children.** SAIPE poor children are most, not all, of the formula count, and
  ED adjusts SAIPE for boundary changes and charter LEAs.
- **Overlapping districts.** Elementary- and secondary-only districts overlap
  geographically (for example CA, IL, AZ, MT). SAIPE gives each its own grade-relevant
  5-17 population.
- **Outcome timing.** `cur_pp` is spending in the receipt year. Title I funds can be
  carried over for a year, so `lead=1` is a reasonable alternative.
- **Small districts.** For districts with fewer than 20,000 people, states may replace
  SAIPE with other poverty data, so their actual money is less tied to Q.
- **Dollars.** All amounts are nominal.
