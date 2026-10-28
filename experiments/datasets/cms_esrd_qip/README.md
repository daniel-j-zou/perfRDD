# CMS ESRD Quality Incentive Program: payment-reduction dose on the TPS

| Field | Value |
|---|---|
| **Q** | TPS - mTPS: the facility's Total Performance Score (integer, 0-100) minus the minimum TPS of the payment year. Q puts every payment year on one scale, on which the schedule is the same in PY 2014-2026 |
| **Threshold** | -0.5: a reduction starts at Q <= -1. Dose cutoffs on Q: -0.5, -10.5, -20.5, -30.5 (`adapter.CUTOFFS_Q`) |
| **Treatment** | `load_dose()`: A = Medicare ESRD payment reduction in CY t, 0 / 0.5 / 1.0 / 1.5 / 2.0 %, 0.5 per 10 points below the minimum. `load()` is the same sample, with D = 1{A > 0} |
| **Rule class** | a step dose (five levels) on one score; the minimum moves by payment year (group) |
| **X** | lagged QIP results (TPS percentile in PY t-1 and t-2, PY t-1 reduction, eight PY t-1 measure scores, with missing flags), corporate chain (7 flags), hospital-based CCN, facility age, log eligible patients, state and payment-year indicators: 95 columns (4 continuous). `dfc=True` adds profit status, log stations, late shift, PD and home HD (PY 2020+) |
| **Y** | `load_dose()`: TPS in PY t+2, whose performance year is CY t, the year the PY t cut is paid. `outcome="tps_t1"`: TPS in PY t+1. `"shr"` / `"smr"`: standardized hospitalization rate in CY t and mortality rate in CY t-3..t (Dialysis Facility Compare, per 100 patient-years) |
| **n** | 57,328 facility-years (8,414 facilities), PY 2014-2019, 2021, 2023, 2024. `tps_t1`: 65,152; `shr`: 33,980; `smr`: 33,964 |
| **Source** | CMS ESRD QIP Performance Score Summary Reports (PSSR), PY 2012-2026 (cms.gov); Dialysis Facility Compare archived snapshots 2020-2026 (data.cms.gov Provider Data Catalog); CMS "Program Details" fact sheets; 42 CFR 413.177 |

**Fit: good for a step-dose design.** The weak points are the moderate R^2 and the small
doses.
- **The rule is exact.** The formula reproduces the published reduction for all 91,681
  scored facility-years, PY 2012-2026 (below). CMS publishes the minimum ahead of time: the
  PY 2025-2028 summary already lists 56 and 57 for PY 2027 and 2028.
- **The dose varies at fixed X.**
  - sd(eta) is 11.1 points, about one step (10 points).
  - The fitted score T = E[Q | X] overlaps across all five doses. The 10-90% range of T
    is [4.0, 28.3] at dose 0 and [-17.1, 16.8] at dose 2.0.
- **No sign of sorting.** The count ratio across each cutoff is 1.12-1.31, the same as
  at placebo points (0.84-1.31). A Poisson log-density jump at the first cutoff is
  -0.03 (se 0.07).
- **Weak points.**
  - R^2 is 0.46 pooled, but only 0.25 within payment year; it is 0.11 without the lagged
    QIP results.
  - 82% of facility-years have no reduction. Only 768 and 451 facility-years sit at
    1.5% and 2.0%.
  - The largest cut is 2% of Medicare dialysis payments for one year.
  - Measures and weights change every year, so the outcome scale changes too.

## The rule

The 42 CFR 413.177 schedule, as stated in each year's CMS fact sheet (`data/raw/docs/`):

| Payment year | Minimum TPS (mTPS) | No reduction | 0.5% | 1.0% | 1.5% | 2.0% | Source of the minimum |
|---|---:|---|---|---|---|---|---|
| 2012 | 26 (max 30) | 26-30 | 21-25 | 16-20 | 11-15 | 0-10 | fact sheet |
| 2013 | 30 (max 30) | 30 | - | 26-29 | 21-25 | 0-20 | fact sheet |
| 2014 | 53 | >= m | m-10..m-1 | m-20..m-11 | m-30..m-21 | < m-30 | fact sheet |
| 2015 | 60 | " | | | | | fact sheet |
| 2016 | 54 | | | | | | fact sheet |
| 2017 | 60 | | | | | | fact sheet |
| 2018 | 49 | | | | | | fact sheet |
| 2019 | 60 | | | | | | fact sheet |
| 2020 | 59 | | | | | | PSSR bands (no fact sheet on cms.gov) |
| 2021 | 56 | | | | | | PSSR bands (no fact sheet on cms.gov) |
| 2022 | none | no facility was reduced (COVID-19 special rule; the PSSR shows N/A and no TPS) | | | | | |
| 2023 | 83 | | | | | | PSSR bands (no fact sheet on cms.gov; six measures were suppressed that year) |
| 2024 | 57 | | | | | | fact sheet |
| 2025 | 55 | | | | | | fact sheet and the PY 2025-2028 summary |
| 2026 | 53 | | | | | | fact sheet and the PY 2025-2028 summary |

PY 2027 and 2028 minimums are 56 and 57 (fact sheets). Their scores are not yet out.

- **Timing.**
  - PY t scores performance in CY t-2. The improvement period is usually CY t-3, but
    the PSSR lists CY 2019 for PY 2023-2024.
  - The final PSSR is dated between October of t-1 and February of t, after a preview
    period. The cut applies to all Medicare ESRD PPS payments for services in CY t.
- **Formula.**
  - `adapter.schedule(q, group)` returns the reduction in percent from Q and the payment
    year. `dose_from_tps(tps, py)` does the same from the TPS.
  - For PY 2014-2026 it is 0.5 * min(4, ceil(-Q / 10)) when Q < 0, else 0.
  - Q is an integer, so the effective cutoffs are -0.5, -10.5, -20.5 and -30.5.
- **PY 2012-2013** are in the panel but not in `load_dose()`. The TPS was out of 30
  points and takes at most 17 distinct values, and PY 2013 has no 0.5% band.

## Rule check (2026-10-28)

`adapter.rule_check()` compares the formula with the published "Payment Reduction
Percentage" for every facility with a TPS.

| PY | Scored | Agreement | Share reduced | Max TPS reduced | Min TPS not reduced |
|---|---:|---:|---:|---:|---:|
| 2012 | 4,939 | 1.000 | 0.309 | 24 | 26 |
| 2013 | 5,101 | 1.000 | 0.106 | 27 | 30 |
| 2014 | 5,846 | 1.000 | 0.048 | 52 | 53 |
| 2015 | 5,650 | 1.000 | 0.061 | 59 | 60 |
| 2016 | 6,048 | 1.000 | 0.055 | 53 | 54 |
| 2017 | 6,229 | 1.000 | 0.204 | 59 | 60 |
| 2018 | 6,549 | 1.000 | 0.142 | 48 | 49 |
| 2019 | 6,822 | 1.000 | 0.274 | 59 | 60 |
| 2020 | 7,051 | 1.000 | 0.417 | 58 | 59 |
| 2021 | 7,330 | 1.000 | 0.402 | 55 | 56 |
| 2023 | 7,574 | 1.000 | 0.107 | 82 | 83 |
| 2024 | 7,561 | 1.000 | 0.265 | 56 | 57 |
| 2025 | 7,505 | 1.000 | 0.332 | 54 | 55 |
| 2026 | 7,476 | 1.000 | 0.434 | 52 | 53 |

- All 91,681 match, at every band edge, not only the first. In each year the edges of
  the published bands bracket the minimum to one point.
- For PY 2020, 2021 and 2023 the minimum is read off these edges; no fact sheet for
  those years is on cms.gov. Every edge of the five bands is consistent with that
  single value.
- Facilities without a TPS (too few eligible patients on every measure, or new) get no
  reduction. They are not in the samples (78-631 a year).

## Dose design

`load_dose(outcome="tps_t2", years=None, dfc=False) -> RDDSample`

| `extras` key | Content |
|---|---|
| `A` | formula reduction (%), `schedule(Q, group)` |
| `a_ref` | 0.0 |
| `dose_levels` | `[0, 0.5, 1.0, 1.5, 2.0]` |
| `schedule` | `schedule(q, group) -> dose`, for any PY 2012-2026 (NaN for 2022) |
| `group` | payment year |
| `Z_names` | `chain_davita`, `chain_fresenius`, `hospital_based`, `age_years`, `log_patients` (+ `dfc_forprofit`, `log_stations` with `dfc=True`) |
| `published_dose` | the PSSR reduction (equal to `A` everywhere) |
| `cutoffs`, `min_tps` | -0.5, -10.5, -20.5, -30.5; mTPS by PY |
| `ccn` | facility id, for clustering |
| `tps` | the unshifted TPS |
| `Y_alt` | `tps_t1`, `tps_t2`, `q_t1`, `q_t2` (minus that year's mTPS), `pct_t1`, `pct_t2` (percentile within PY), `reduction_t1`, `reduction_t2`, `shr`, `smr` |
| `raw` | the panel rows of the sample |

- **Choice of Q.** Q = TPS - mTPS, not the raw TPS.
  - The minimum moves between 49 and 83. In Q, the schedule is the same function in
    every PY 2014-2026, and `group` only matters through the mean of Q.
  - Payment-year indicators are in X.
- **Choice of Y.** The default outcome is the TPS two payment years later.
  - PY t+1 scores CY t-1, which is mostly over when the PY t result is previewed
    (summer of t-1) and published (late t-1). That makes `tps_t1` close to a placebo
    with partial exposure.
  - PY t+2 scores CY t, the year the cut is paid and the year the facility must post
    its PY t certificate.
  - PY 2020 has no t+2 outcome (PY 2022 has no TPS), and PY 2021 has no t+1 outcome.
- **Z.** These covariates vary among facilities with the same score: chain, hospital
  base, age, size, and with `dfc=True` profit status and stations.

## Readiness (2026-10-28)

From `adapter.readiness(load_dose())`. Q is regressed by OLS on [1, X], T is the fit
and eta = Q - T.

| Quantity | `tps_t2` (default) | `tps_t1` | `shr` | `tps_t2`, `dfc=True` |
|---|---:|---:|---:|---:|
| Payment years | 2014-19, 2021, 2023, 2024 | 2014-20, 2023-25 | 2018, 2019, 2021, 2023, 2024 | 2021, 2023, 2024 |
| n (facilities) | 57,328 (8,414) | 65,152 (8,620) | 33,980 (7,898) | 21,510 (7,758) |
| p | 95 | 96 | 91 | 94 |
| R^2 of Q on X (pooled) | 0.455 | 0.457 | 0.340 | 0.302 |
| R^2 within payment year | 0.247 | 0.261 | | |
| sd(Q), sd(T), sd(eta) | 15.0, 10.1, 11.1 | 15.0, 10.2, 11.1 | 13.2, 7.7, 10.7 | 12.9, 7.1, 10.8 |
| Share at dose 0 | 0.823 | 0.803 | 0.765 | 0.745 |
| corr(Y, Q) | 0.23 | 0.62 | -0.15 | 0.06 |

Default sample, by dose level:

| Dose (%) | n | Share | eta: 10% / 50% / 90% | T: 10% / 50% / 90% |
|---:|---:|---:|---|---|
| 0 | 47,160 | 0.823 | -8.0 / 2.8 / 13.8 | 4.0 / 12.4 / 28.3 |
| 0.5 | 6,569 | 0.115 | -18.8 / -9.6 / 0.1 | -5.2 / 5.0 / 13.6 |
| 1.0 | 2,380 | 0.042 | -25.4 / -14.8 / -3.7 | -10.8 / 0.5 / 10.3 |
| 1.5 | 768 | 0.013 | -35.9 / -22.5 / -9.7 | -15.6 / -2.0 / 10.4 |
| 2.0 | 451 | 0.008 | -58.9 / -35.7 / -18.5 | -17.1 / -1.2 / 16.8 |

**Bunching.** Each ratio is the count of facility-years at the two integer values of Q
just above an edge over the two just below. The jump is a Poisson fit of log counts on
a quadratic on each side, within 12 points, with overdispersion-scaled se.

| Edge (Q) | Above / below | Ratio | Log-density jump (se) |
|---|---|---:|---|
| -0.5 (0 / 0.5%) | 2,005 / 1,785 | 1.12 | -0.03 (0.07) |
| -10.5 (0.5 / 1.0%) | 912 / 710 | 1.28 | 0.01 (0.06) |
| -20.5 (1.0 / 1.5%) | 272 / 207 | 1.31 | -0.04 (0.11) |
| -30.5 (1.5 / 2.0%) | 114 / 89 | 1.28 | 0.19 (0.18) |
| placebo 14.5 | 3,014 / 3,599 | 0.84 | 0.02 (0.20) |
| placebo 9.5 | 3,540 / 3,458 | 1.02 | -0.05 (0.15) |
| placebo 4.5 | 3,003 / 2,495 | 1.20 | -0.01 (0.09) |
| placebo -5.5 | 1,429 / 1,205 | 1.19 | 0.01 (0.07) |
| placebo -15.5 | 516 / 425 | 1.21 | -0.02 (0.09) |
| placebo -25.5 | 168 / 128 | 1.31 | 0.17 (0.15) |

- The density of Q rises to the right below the mode, so every ratio below the mode
  exceeds 1. The cutoff ratios match their neighbours, and no jump is significant.
- Counts at Q = -3..2 are 783, 859, 926, 960, 1,045 and 1,253.

**By payment year** (default sample):

| PY | n | mTPS | sd(Q) | sd(eta) | R^2 | Share at dose 0 | corr(Y, Q) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 5,595 | 53 | 14.3 | 13.6 | 0.10 | 0.96 | 0.29 |
| 2015 | 5,346 | 60 | 12.5 | 11.0 | 0.22 | 0.95 | 0.35 |
| 2016 | 5,751 | 54 | 11.5 | 10.1 | 0.23 | 0.95 | 0.31 |
| 2017 | 6,131 | 60 | 11.5 | 9.9 | 0.25 | 0.80 | 0.42 |
| 2018 | 6,369 | 49 | 13.5 | 11.3 | 0.30 | 0.86 | 0.46 |
| 2019 | 6,626 | 60 | 13.2 | 10.8 | 0.33 | 0.73 | 0.48 |
| 2021 | 7,119 | 56 | 15.0 | 11.8 | 0.38 | 0.60 | 0.16 |
| 2023 | 7,136 | 83 | 9.3 | 9.0 | 0.06 | 0.90 | 0.25 |
| 2024 | 7,255 | 57 | 13.3 | 11.8 | 0.21 | 0.74 | 0.48 |

- PY 2014 has a weak first stage (lags come from the 30-point PY 2013 scale) and few
  reduced facilities (4%).
- PY 2023 used special scoring with six measures suppressed. It has mTPS 83, a
  compressed Q and a low R^2.
- PY 2021's outcome is the PY 2023 score (mean 91), hence its low corr(Y, Q).

## Sample and variables

- **Unit.** Facility (CCN) x payment year. No CCN repeats within a payment year once
  the PSSR footnote rows are dropped.
- **Lagged QIP results.**
  - The TPS percentile is taken within its payment year, so the 30-point PY 2013 and
    special PY 2023 scales are comparable.
  - The eight measure scores (0-10) are Kt/V, vascular access (topic or combined),
    hypercalcemia, NHSN bloodstream infection, SRR, STrR, SHR and ICH CAHPS. Each has a
    missing flag; missing entries are set to 0.
  - PY 2023 has no lag-1 score (PY 2022 had none) and uses the lag-2 percentile.
- **Chain.** From the PSSR "Corporate Ownership" field at the end of the performance
  year. Flags: DaVita (incl. Gambro), Fresenius (incl. Renal Care Group and Liberty),
  DCI, US Renal Care and American Renal; other chains vs independent; and unknown
  (blank, all of PY 2012).
- **Hospital-based.** The CCN's last four digits are in 2300-2499 or 3500-3799, the
  CMS ranges for hospital-based and hospital satellite dialysis units.
- **Facility age.** Years from the certification date to January 1 of the performance
  year.
- **Patients.** Eligible patients in the performance year: URR (PY 2012-2014), Kt/V
  adult HD (2015-2018) or Kt/V comprehensive (2019+). A measure denominator from the
  score's own year, not the score. '<11' and missing counts (0.8%) are set to 5 and
  flagged.
- **DFC (`dfc=True`).**
  - Characteristics come from the January snapshot of PY t. Snapshots are available for
    2020, 2021 and 2023-2026; the 2022 file repeats 2021.
  - The SHR / SMR outcomes for CY t come from the January t+2 snapshot.
  - There is no SHR for CY 2020: CMS excepted 2020 data for COVID-19.
- **Left out of X.** The PY t measure scores, rates and patient-level counts, which
  build Q, and the DFC star rating, which is built from the same measures.

## Getting the data

    python -m experiments.datasets.cms_esrd_qip.download            # idempotent, ~205 MB
    python -m experiments.datasets.cms_esrd_qip.download --check    # sizes on disk
    python -m experiments.datasets.cms_esrd_qip.download --only pssr docs

Run it from the repo root. Every file is fetched with plain `curl` (its default
settings), with no login, form, terms page or CAPTCHA. ZIPs are extracted, each into
its own folder. The workbooks are read with a standard-library reader (`xlsx.py`), so
`openpyxl` is not needed. `data/processed/esrd_qip_panel.csv.gz` caches the panel,
built in about 2 minutes; delete it after changing the adapter.

| Files | Source | Bytes | Use |
|---|---|---:|---|
| `pssr/pssr_py{2012..2026}.zip` | cms.gov, "ESRD QIP: Public Reporting & Certificates" (`download.PSSR` lists each URL) | 132,743,933 | TPS, published reduction, measure scores, chain, certification date, patients |
| `archive/dialysis-facilities_{2020-01-31, 2021-01-31, 2023-01-18, 2024-01-18, 2025-01-15, 2026-01-21}.zip` | `https://data.cms.gov/provider-data/sites/default/files/dataset-archives/theme/dialysis-facilities/` | 57,790,156 | DFC facility files: profit, stations, SHR, SMR |
| `docs/*.pdf` | cms.gov fact sheets "PY 20xx Program Details" (2012-2019, 2024-2028), QIP summaries PY 2012-2016 and 2025-2028, and the two QIP pages that link them | 4,840,165 | the minimum TPS and the scale |
| `meta/*.json` | Provider Data Catalog metastore and archive listing (`/provider-data/api/1/...`) | 565,865 | which snapshot holds what |

Total downloaded: 195,940,119 bytes; about 604 MB on disk with the extracted files and
the cache.

- **Access.** Every source answered scripted requests made with plain curl. The
  Provider Data Catalog API refused one early request that sent a custom User-Agent
  string (HTTP 403, Akamai "Access Denied").
- **Snapshot.** Downloaded 2026-10-28. PSSR files are final reports and do not change.
  The current DFC data (PY 2026 QIP, October 2026 DFC) is in the catalog but not used.

## Caveats

- **The measures change every year.** TPS weights, measures and the achievement or
  improvement scoring change by payment year, and PY 2023 used special scoring. Y in
  points is not comparable across years without the PY indicators in X; `Y_alt` holds
  percentile and Q-scale versions.
- **Dynamic treatment.** Doses persist from year to year.
  - Of facilities with a cut in PY t, 59% have one in PY t+1, against 18% of those
    without. corr(Q_t, Q_{t+1}) = 0.58.
  - Y at PY t+2 also reflects the PY t+1 dose, which is announced during CY t.
- **Mechanics in Y.** The TPS rewards improvement relative to the facility's own
  earlier performance. A low score in t makes a later improvement score easier, so
  mean reversion is part of any raw effect on Y.
- **Units repeat** over up to 9 payment years. Cluster by `extras["ccn"]`. Changes of
  ownership keep the CCN. 0.7-4.8% of facilities leave the files each year (closure or
  a new CCN).
- **The minimum for PY 2020, 2021 and 2023** comes from the data, not a CMS document on
  cms.gov. The Federal Register final rules (CY 2018, 2019 and 2021 ESRD PPS) state it;
  they were not downloaded (govinfo.gov is outside the allowed sources).
