# Medicare Advantage county benchmarks: the applicable percentage as a dose

| Field | Value |
|---|---|
| **Q** | Percentile rank (0-100) of the county's ranking FFS rate among the ranked counties of payment year t. The ranking rate is the county's risk-standardised FFS rate of year t-1, excluding the phased-out IME, from CMS's rate calculation file. `score="logratio"`: Q = 100 log(rate / the year's 75th-percentile cutoff) |
| **Threshold** | 75 (`logratio`: 0): the edge of the 95% region. The other edges are 25 and 50 in every year (exact; see "The rule") |
| **Treatment** | Dose A = the applicable percentage, as a fraction (0.95, 1.00, 1.075, 1.15, and means of two of them). `load()`: D = 1{Q <= 75}, i.e. a dose above the 95% region |
| **Rule class** | A 4-level step schedule in Q, recomputed every year. A is the mean of this year's quartile percentage and last year's, so the schedule depends on the group (payment year, last year's percentage): 40 groups |
| **X** | 70 columns: lagged MA penetration and log eligibles (July t-1); log population, 65+ and 85+ shares, median age (Census, July t-1); log median household income and poverty rate (SAIPE, t-1); log population density; the county's average FFS risk score; the 2004 urban-floor flag; 9 year and 50 state indicators (11 continuous or binary + 59 indicators) |
| **Y** | `load_dose()`: MA penetration (%) in July of payment year t. `outcome="dpen"`: change from July t-1. `outcome="pen_lead"`: July t+1 |
| **n** | 31,181 county-years (3,126 counties), payment years 2017-2026. `years=(2013, 2026)` adds the ACA transition years: 43,681 |
| **Source** | CMS "Ratebooks & Supporting Data" (rate calculation data and ratebooks, 2010-2027); CMS "MA State/County Penetration" (July files, 2012-2026); Census population estimates (vintages 2019 and 2025), SAIPE 2010-2024 and the 2020 Gazetteer; SSA 1853(n) |

**Fit: good for the dose design, with a moderate first stage.**
- **The rule is reproduced.** The formula gives CMS's published applicable percentage for
  50,296 of 50,298 county-years in the 50 states and DC (2012-2027). It also gives the
  published 0%-bonus benchmark to the cent in every year.
- **No manipulation is plausible, and none is seen.** CMS computes the score from five
  years of FFS claims. Within 1% of a cutoff, county-years above vs below are 0.97 to 1;
  at placebo quantiles the ratio is 1.00.
- **X predicts Q moderately.** R^2 = 0.51 and sd(eta) = 20.1 percentile points. eta
  overlaps across all dose levels (table below). Adding last year's rank to X
  (`lag_score=True`) raises R^2 to 0.97, and then overlap is thin.
- **Caveats.**
  - The cap at the pre-ACA rate binds for 23% of the sample, mostly above the reference
    (see "Caps").
  - Penetration is very persistent: R^2 of Y on X is 0.987, and the mean yearly change is
    2.2 points (sd 2.5).
  - Q itself moves little from year to year, so the same counties sit near the cutoffs
    for several years. Cluster by `extras["unit_id"]` (SSA county code).

## The rule

The source is SSA 1853(n), as CMS implements it in the rate calculation files. The
formula at the top of `risk<year>.csv` reads: rate = FFS rate excluding phase-out IME x
(quartile % + bonus %), capped at the pre-ACA rate. The 2012 Rate Announcement
(`docs/announcement2012.pdf`) gives the transition blend and says territories do not
enter the cutoffs. The rest was found in the data and is checked below.

1. **Ranking rate.** For payment year t, a county is ranked by its FFS rate of year t-1.
   This is the "FFS Rate excluding Phase-out IME" column of CMS's `risk<t-1>.csv`. For
   2012 it is the "Estimated 2010 FFS Rate" column of `risk2012.csv`.
   - The year-t rate itself does not reproduce the published quartiles: 73-92% match.
   - The ranked set is the counties of the 50 states and DC listed in year t's file:
     3,143-3,144 counties.
   - Two new Alaska codes in 2024 (02063, 02066, the Valdez-Cordova split) have no 2023
     rate. They carry the parent's rate (02261). This reproduces CMS's 2024 cutoffs.
2. **Cutoffs.** The 25th, 50th and 75th percentiles of the ranking rates, linearly
   interpolated (numpy's default, Excel's QUARTILE).
   - Ties go down: a county exactly at a cutoff gets the lower-cost quartile, and so the
     higher percentage. 22 county-years (2012-2027) have a rate equal to a cutoff to the
     cent.
   - The pure percentage is 115% at or below the 25th percentile, 107.5% up to the 50th,
     100% up to the 75th, and 95% above it.
   - On the rank scale used for Q, with i the number of ranked rates strictly below the
     county's and n the number ranked, Q = 100 i / (n - 1). Then Q <= 25 holds exactly
     when the rate is at or below the interpolated 25th percentile, and likewise at 50
     and 75. So the edges are 25/50/75 in every year.
3. **Averaging (from 2013).** The applicable percentage is the mean of the pure
   percentages of years t and t-1. A county that stays in its quartile keeps the pure
   level. A county that moves one quartile gets 0.975, 1.0375 or 1.1125, and a two-step
   move gives 1.0125 or 1.05.
   - CMS's file `2016 Quartiles and Applicable Percentages.csv` lists the 2015 and 2016
     pure quartiles and the blend.
   - The formula matches all three columns for all 3,144 counties.
   - `extras["group"]` = "<year>|<last year's pure percentage>", so `extras["schedule"]`
     is a 4-step function of Q within each group.
4. **Transition (2012-2016).** The benchmark blends the pre-ACA rate (weight w) and A x
   FFS rate (weight 1 - w).
   - w = max(0, 1 - (t - 2011)/T), where T = 2, 4 or 6 is the county's "Transition Years".
   - All counties are on the ACA formula from 2017. `extras["aca_share"]` = 1 - w.
5. **Cap.** The benchmark at every bonus level is at most the pre-ACA rate. So the 0%-bonus
   benchmark is min(w x pre-ACA + (1 - w) x A x FFS_t, pre-ACA).
   - Note that A multiplies the year-t FFS rate, while the ranking uses the year-(t-1)
     rate.
6. **Quality bonus.** Not part of A. It adds 5 points (3.5 for some years and star
   levels), doubled in "qualifying counties", and is capped too. Those counties are
   urban-floor counties with 2009 MA penetration of 25% or more and an FFS rate below the
   national average. They are `extras["qualifying"]`, 7.4% of the sample.

## Rule check (2026-10-28)

`adapter.rule_check()` covers every county of the 50 states and DC, 2012-2027.

| Year | Ranked | Cutoffs ($ PMPM: 25 / 50 / 75%) | A matches published | 0%-bonus rate matches | Share capped |
|---|---:|---|---:|---:|---:|
| 2012 | 3,144 | 639.87 / 691.12 / 743.87 | 1.0000 | 1.0000 | 0.140 |
| 2013 | 3,144 | 644.46 / 691.56 / 750.66 | 1.0000 | 1.0000 | 0.148 |
| 2014 | 3,144 | 668.78 / 716.83 / 777.05 | 1.0000 | 1.0000 | 0.197 |
| 2015 | 3,144 | 698.82 / 745.60 / 805.60 | 1.0000 | 1.0000 | 0.270 |
| 2016 | 3,144 | 678.11 / 722.35 / 782.31 | 1.0000 | 1.0000 | 0.240 |
| 2017 | 3,144 | 709.14 / 755.38 / 817.98 | 1.0000 | 1.0000 | 0.300 |
| 2018 | 3,143 | 741.92 / 790.04 / 854.87 | 1.0000 | 1.0000 | 0.317 |
| 2019 | 3,143 | 763.70 / 812.29 / 881.60 | 1.0000 | 1.0000 | 0.302 |
| 2020 | 3,143 | 809.98 / 859.11 / 932.53 | 1.0000 | 1.0000 | 0.303 |
| 2021 | 3,143 | 856.64 / 906.35 / 987.62 | 1.0000 | 1.0000 | 0.212 |
| 2022 | 3,143 | 885.15 / 938.28 / 1022.94 | 1.0000 | 1.0000 | 0.176 |
| 2023 | 3,143 | 930.29 / 988.99 / 1077.19 | 1.0000 | 1.0000 | 0.192 |
| 2024 | 3,144 | 978.28 / 1040.88 / 1133.72 | 0.9997 | 0.9997 | 0.206 |
| 2025 | 3,144 | 996.41 / 1058.74 / 1156.68 | 0.9997 | 0.9997 | 0.174 |
| 2026 | 3,144 | 1013.44 / 1080.06 / 1182.29 | 1.0000 | 1.0000 | 0.100 |
| 2027 | 3,144 | 1101.84 / 1174.81 / 1285.99 | 1.0000 | 1.0000 | 0.126 |

- **The one miss.** Jasper County, SC (SSA 42260) has a ranking rate of $1,040.89 in 2024.
  The formula's median cutoff is $1,040.875, so it lands just above the median. CMS
  placed it below, which gives the miss in 2024 and, through the averaging, in 2025.
- **Capped flag.** `capped` (blend >= pre-ACA rate) agrees with CMS's "0% Bonus Rate
  Capped" flag in 99.97-100% of rows (2015-2027). CMS counts equality as capped.
- **Territories.** Puerto Rico, the Virgin Islands, Guam and the others are assigned
  with the same cutoffs. Every territory row with a ranking rate matches. The misses are
  Guam in 2012-2014 (published as N/A) and codes with no prior-year rate (Guam, American
  Samoa and the Northern Marianas in 2015; new codes in 2020 and 2025), which CMS gives
  115%. Territories are not in the sample.
- In the sample, `extras["schedule"](Q, group)` reproduces `extras["A"]` for every row,
  with either score.

## Dose design

| Quantity | Value |
|---|---|
| `extras["A"]` | applicable percentage from the formula (fraction); `published_dose` = CMS's "Quartile Percent factor" (equal in 31,179 of 31,181 rows) |
| `a_ref` | 0.95, the dose of the region above the 75th percentile for counties that were also there last year |
| `dose_levels` | 0.95, 0.975, 1.0, 1.0125, 1.0375, 1.05, 1.075, 1.1125, 1.15 (1.05 never occurs in the sample; 1.0125 four times) |
| `schedule(q, group)` | the rule above; `group` = "<year>\|<last year's pure percentage>" |
| `group` | 40 labels: 10 payment years x 4 values of last year's pure percentage |
| `Z_names` | `pen_lag`, `log_income`, `log_density`, `urban_floor_2004` |
| unit id | `unit_id` (SSA county code); also `fips` |
| other extras | `year`, `prev_pct`, `pure_pct`, `ffs_rank_rate` ($), `q_rank`, `q_logratio`, `cutoffs` (by year), `ffs` (FFS_t), `preaca`, `benchmark0` (published 0%-bonus rate), `benchmark_ratio` (= benchmark0 / FFS_t), `capped`, `aca_share`, `qualifying`, `trans_years`, `eligibles`, `y_masked`, `pen`, `pen_lag`, `pen_lead`, `dpen` |

- **How A is defined.** A is the statutory applicable percentage. It is not the
  effective ratio of the benchmark to the FFS rate.
  - **Caps.** In capped counties the effective ratio is lower: `benchmark_ratio`
    averages 1.136 where A = 1.15, 1.094 where A = 1.1125, 1.063 where A = 1.075 and 1.0
    or less elsewhere.
  - **Transition years.** With `years=(2013, ...)`, A carries weight `aca_share` < 1 in
    2013-2016.
  - **Bonuses.** Quality bonuses are left out.
- **Timing.** "Year" is the payment year t.
  - CMS publishes the rates in April of t-1. Plans bid in June of t-1. Enrolment for
    January of t happens in October-December of t-1.
  - X is measured in calendar year t-1: penetration in July of t-1 (before the year-t
    plans exist), Census estimates of July 1 of t-1, SAIPE of t-1 (2024 for t = 2026).
  - Y (July of t) is the first penetration that the year-t benchmark can move.
- **Overlap.** Units with the same Q get different doses: across years Q differs only
  through the averaging group, and the bonus and cap differ too. The dose is a step
  function of Q within a group.

## Readiness (2026-10-28)

`adapter.readiness()` on `load_dose()` (rank score):

| Quantity | Value |
|---|---:|
| n, units, p | 31,181, 3,126, 70 |
| R^2 of Q on X | 0.510 (`logratio` score: 0.510; with last year's rank in X: 0.965) |
| sd(Q), sd(T), sd(eta) | 28.75, 20.53, 20.13 (`logratio`: sd(Q) 11.35, sd(eta) 7.95 log points) |
| Share at the reference dose (A = 0.95) | 0.228 |
| Share with D = 1 (Q <= 75) | 0.754 |
| Within 1% of a cutoff (ranking rate), above : below | 4,911 : 5,039 = 0.97 (2012-2027, all three cutoffs) |
| Same at placebo quantiles 12.5 / 37.5 / 62.5 / 87.5 | 5,497 : 5,486 = 1.00 |

eta = Q - OLS fit, by dose level:

| A | n | eta q10 | q50 | q90 | mean Y (%) |
|---:|---:|---:|---:|---:|---:|
| 0.95 | 7,114 | -0.6 | 15.1 | 36.4 | 25.6 |
| 0.975 | 1,084 | -7.2 | 11.2 | 33.3 | 29.3 |
| 1.0 | 6,393 | -12.1 | 10.0 | 28.4 | 33.6 |
| 1.0125 | 4 | -12.0 | -4.4 | 46.1 | 46.7 |
| 1.0375 | 1,755 | -18.1 | 4.5 | 22.3 | 35.8 |
| 1.075 | 6,139 | -22.7 | -2.1 | 13.2 | 39.4 |
| 1.1125 | 1,692 | -31.6 | -11.5 | 3.6 | 40.2 |
| 1.15 | 7,000 | -37.5 | -20.3 | -5.6 | 43.5 |

- With `lag_score=True` the eta quantiles shrink to about +-3 to +-13. Last year's rank
  then explains the dose almost fully.
- `years=(2013, 2026)`: n = 43,681, R^2 = 0.478, sd(eta) = 20.8.
- The bunching ratios are in the dollar rate: the rank is uniform by construction.

## Sample

- **Units.** County (SSA code) x payment year, 50 states and DC, 2017-2026.
- **Who drops.**
  - Of 31,434 county-years, 253 are dropped for a missing Y, X or crosswalk.
  - Los Angeles has a second SSA code (05210) with no penetration record.
  - Five Alaska boroughs and the two new Valdez-Cordova codes are not in the penetration
    files under the ratebook codes.
  - Connecticut's eight counties lack Census covariates from 2021 on. The Census uses
    planning regions from vintage 2022, and CMS kept the old counties.
  - Alaska and Connecticut are absent from the July 2025 and 2026 penetration files.
- **Masked enrolment.** Enrolment of 1-10 is shown as "*". These are set to 5 and
  flagged in `extras["y_masked"]` (394 rows, mostly small counties in 2017-2020).
- **Crosswalk.** SSA to FIPS comes from the penetration files (latest year a code
  appears).
- **Outcome.** Mean Y = 35.5%, sd 17.4. corr(Y, Q) = -0.40: high-spending counties have
  lower penetration in this sample, after the cap.
  - **What counts as MA.** Penetration counts all MA contract types in CMS's file
    (local/regional CCP, PFFS, MSA, cost, PACE, demonstrations) over all Medicare
    eligibles.

## Getting the data

    python -m experiments.datasets.medicare_ma_benchmarks.download          # ~108 MB
    python -m experiments.datasets.medicare_ma_benchmarks.download --check
    python -m experiments.datasets.medicare_ma_benchmarks.download --only ratebooks

- **Access.** Everything comes from www.cms.gov and www2.census.gov over plain HTTPS. No
  login, form, agreement or CAPTCHA was needed (downloaded 2026-10-28).
- **Penetration links.** The penetration zip links are scraped from each month's CMS
  page. The page is saved under `pages/`.
- **Extraction.** The adapter extracts the archives with `zipfile` into
  `data/raw/extracted/<archive>/` (101 MB), and checks member paths.
- **Cache.** The panel is cached in `data/processed/ma_panel.csv.gz` (7 MB, version 2).
  Pass `build_panel(force=True)` or delete the file after changing the adapter.

| Folder | Files | Bytes | Source |
|---|---|---:|---|
| `ratebooks/` | `calc_<y>.zip`, `ratebook_<y>.zip` (2010-2027), `2010paymentrates.zip`, `applicablepercentagescy2016.zip` | 51 MB | https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/ratebooks-supporting-data (one page per year) |
| `penetration/` | `pen_<y>_07.zip` (2012-2026) | 1.2 MB | https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data/ma-state/county-penetration |
| `census/` | `cc-est2019-agesex-<ss>.csv` (51), `cc-est2025-agesex-all.csv`, `est<yy>all.txt` (2010-2024), `2020_Gaz_counties_national.zip` | 39 MB | www2.census.gov popest, SAIPE, Gazetteer |
| `docs/` | `announcement2012.pdf`, `announcement2017.pdf`, `saipe_layout_2024.txt` | 5.2 MB | CMS Rate Announcements; Census SAIPE layout |

Used from the calculation zips: `risk<y>.csv` (one row per county: ranking and current
FFS rates, pre-ACA rate, transition years, "Quartile Percent", qualifying county,
benchmarks by bonus level). Also the Non-PACE risk-score table, and for 2014 the revised
Puerto Rico file. The `.xls/.xlsx` workbooks are not read: the environment has no Excel
reader, and the CSVs hold the same tables.

## Caveats

- **The dose is a price for plans, not for enrollees.**
  - The benchmark sets what plans can bid against. Penetration responds through plan
    entry, rebates and extra benefits.
  - The year-t benchmark also scales with FFS_t, which moves smoothly with Q. That level
    effect is part of g(eta) and X, not of A.
- **Caps and bonuses blur the dose.**
  - 23% of the sample is capped at the pre-ACA rate; then the applicable percentage does
    not move the benchmark at the margin.
  - The capped counties are concentrated at A >= 1.0375 (34-46% capped) and absent at A
    <= 0.975.
  - `benchmark_ratio` is the effective 0%-bonus dose.
- **Persistence.** Q is a five-year average and changes slowly. Counties that switch
  quartile produce the averaged levels, and most counties stay in one group for years.
  The same counties supply the near-cutoff observations in several years.
- **Score scale.** The rank is uniform within the year by construction. Use the dollar
  rate (`ffs_rank_rate`, `q_logratio`) for density checks.
- **Not covered.** Regional PPO benchmarks, employer group plans (the payment rules
  differ), ESRD and PACE rates, Part D. The quality bonus is not modelled.
