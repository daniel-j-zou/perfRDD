# US census tracts: the CDFI Investment Area test as an either-cutoff rule on three scores (ACS 2016-2020, 2020 tracts)

| Field | Value |
|---|---|
| **Q** | `(pov, mfi_ratio, unemp_ratio)`: tract poverty rate (%), tract median family income as % of the benchmark, and tract unemployment rate over the 5.4% national rate; ACS 2016-2020 5-year, as published by the CDFI Fund (blanks for zero rates and top-coded incomes filled from the ACS summary file) |
| **Threshold** | 20 on the poverty rate, 80 on the MFI ratio, 1.5 on the unemployment ratio |
| **Treatment** | `1{Q1 >= 20 or Q2 <= 80 or Q3 >= 1.5}`: CDFI Program Investment Area (12 CFR 1805.201(b)(3)(ii)), from the 2016-2020 ACS data the CDFI Fund published on January 5, 2023 |
| **Rule class** | deployed: a cutoff on each of three scores, any one suffices (union of three half-spaces). The Empowerment Zone and county population-loss routes are exceptions; `load()` drops those tracts. Counterfactual: other cutoff triples, two-of-three, weighted sums |
| **X** | 2020 decennial counts from the Census Planning Database 2023 (`*_CEN_2020` columns and the 2020 census self-response rate only): log population, log density, urban share, five age shares, Hispanic / NH Black / NH Asian / NH other shares, institutional and non-institutional group quarters, family / married-couple / female-householder / single-person / with-children household shares, household size, renter share, vacancy, self-response rate, OMB metro flag, 48 state indicators. p = 72. No ACS estimate enters X |
| **Y** | `load()`: change in log(1 + workplace jobs), LODES8 WAC all jobs, 2022 to 2023; `load_placebo()`: the same change 2021 to 2022 |
| **n** | 76,961 tracts (48 states + DC) with all three scores and the outcome, outside the exceptions; 79,106 for `load_all()`; 50,028 for `load_two_score()` |
| **Source** | CDFI Fund (cdfifund.gov), Census ACS summary file and Planning Database (www2.census.gov), LEHD LODES8 (lehd.ces.census.gov); all public, no registration |

Loaders in `adapter.py`:
- `load()`: the three-score sample above. CDFI's published flag equals the rule for
  every tract.
- `load_placebo()`: the same tracts, outcome 2021 to 2022.
- `load_all()`: also keeps the Empowerment Zone and population-loss tracts. D is the
  score rule; `extras["eligible_published"]` is CDFI's flag, which counts the exceptions.
- `load_two_score(y_col="dlog_jobs_22_23")`: the tracts that fail the income test, with
  Q = (poverty rate, unemployment ratio); see "Two-score sample" below.
- `rule_check()`: the checks in "The rule".

## Getting the data

    python -m experiments.datasets.us_tracts_ciia.download     # idempotent, about 0.54 GB

Everything lands in `data/raw/` (gitignored). The first `load()` builds
`data/processed/us_tracts_ciia.csv.gz` (one row per tract, 23 MB) in about 40 s.
Downloaded 2026-10-28; sizes are bytes on disk.

| File | URL | Size |
|---|---|---:|
| `CDFI_Investment_Areas_ACS_2016_2020.xlsb` (the scores, the benchmark MFIs, CDFI's flag for each criterion and overall, 85,395 tracts) | https://www.cdfifund.gov/system/files/2023-01/CDFI_Investment_Areas_ACS_2016_2020.xlsb | 3.8 MB |
| `ACS_2016_2020_CDFI_Investment_Areas_FAQs_BEA.pdf` (the Fund's FAQ: criteria, release and transition dates; not read by the adapter) | https://www.cdfifund.gov/system/files/2023-01/ACS_2016_2020_CDFI_Investment_Areas_FAQs_BEA.pdf | 0.26 MB |
| `acs2020/acsdt5y2020-b17001.dat` (poverty status, all geographies) | https://www2.census.gov/programs-surveys/acs/summary_file/2020/prototype/5YRData/acsdt5y2020-b17001.dat | 119.9 MB |
| `acs2020/acsdt5y2020-b19113.dat` (median family income) | same folder, `acsdt5y2020-b19113.dat` | 18.1 MB |
| `acs2020/acsdt5y2020-b23025.dat` (employment status) | same folder, `acsdt5y2020-b23025.dat` | 40.2 MB |
| `pdb2023tr.csv` (Planning Database 2023, 2020 tracts) | https://www2.census.gov/adrm/PDB/2023/pdb2023tr.csv | 197.3 MB |
| `lodes8/{st}_wac_S000_JT00_{2021,2022,2023}.csv.gz`, 148 files | https://lehd.ces.census.gov/data/lodes/LODES8/{st}/wac/ | 160.6 MB |

Total 540.3 MB. LODES8 has no Alaska file after 2016, no Michigan file for 2022 or 2023,
and no workplace files for Puerto Rico; 2023 is the latest year. The ACS tables are the
Census Bureau's prototype table-based summary files for 2016-2020 (pipe-delimited, one
file per table, all summary levels; tracts are the `1400000US` rows). The adapter reads
the CDFI workbook with the standard-library reader in `../us_tracts_lic/_excel.py`.

**Why 2016-2020 on 2020 tracts.** The CDFI Fund's 2011-2015 Investment Area file is no
longer on cdfifund.gov: `.../system/files/documents/cdfi-investment-areas-acs-2011-2015.xlsb`
returns "Page Not Located", and the geographic-reports listing (checked 2026-10-28) has
the 2006-2010 and 2016-2020 Investment Area files but not 2011-2015. The 2011-2015 BEA
file on the listing has tract poverty and unemployment but only the NMTC-style income
ratio and no Investment Area flag. So this dataset is the 2016-2020 vintage, with 2020
decennial covariates and LODES8 (2020 blocks) outcomes; nothing carries over from
`us_tracts_lic` except code.

## The rule

12 CFR 1805.201(b)(3)(ii): a tract is an Investment Area if its poverty rate is at least
20%; or its unemployment rate is at least 1.5 times the national rate; or its MFI is at
most 80% of the greater of the metro and national metro MFI (metro tracts) or of the
statewide and national non-metro MFI (non-metro tracts); or it lies wholly within an
Empowerment Zone or Enterprise Community; or its county lost at least 10% (metro) or 5%
(non-metro) of its population. CDFI's file gives every 2020 tract's scores, the five
benchmark inputs, a yes/no flag per criterion and the overall flag. National values:
unemployment 5.4%, metro MFI $83,486, non-metro MFI $63,655. CDFI measures the county
population change between the 2011-2015 and 2016-2020 ACS totals.

Checks over all 85,395 tracts in CDFI's file (`adapter.rule_check()`):
- **Rule against CDFI's flag.** The three-score rule, or either exception route,
  reproduces CDFI's overall flag for all 85,395 tracts (agreement 1.000). The rule alone
  agrees for 0.9939. Every disagreement (520) is a tract that CDFI flags only through an
  exception. No rule-eligible tract is unflagged. Outside the exception tracts the rule
  agrees for all tracts. A blank score never qualifies.
- **Per-criterion flags.** CDFI's poverty and income flags equal `pov >= 20` and
  `mfi_ratio <= 80` on the published scores for every tract. No published ratio sits at
  exactly 80.
- **The unemployment cutoff.** CDFI flags as No all 447 tracts with an unemployment rate
  of exactly 8.1%, which is 1.5 times 5.4%. Its ratio column shows 1.5 there (stored in
  single precision). In double precision 8.1 / 5.4 = 1.4999999999999998, so CDFI
  evidently compared a floating-point ratio with 1.5. The adapter computes
  `unemp_ratio = rate / 5.4` in double precision, so `Q3 >= 1.5` reproduces every flag.
  On the published rates the effective cutoff is 8.2% (ratio 1.5185); under the
  regulation's "at least 1.5 times" the 8.1% tracts would qualify. 408 sample tracts sit
  at 8.1%. Do not round Q3 before applying the rule.
- **Benchmark.** The MFI ratio recomputed from CDFI's own columns (tract MFI over the
  benchmark) matches the published ratio within 1.5e-5. Among tracts with a ratio, the
  benchmark is the metro MFI for 34,403, the national metro MFI for 35,070, the
  statewide non-metro MFI for 6,683 and the national non-metro MFI for 6,864.
- **Scores against the ACS summary file.** Poverty (100 x B17001_002 / B17001_001) and
  unemployment (100 x B23025_005 / B23025_003), rounded half up to 0.1, and tract MFI
  (B19113) equal CDFI's values for every tract where CDFI shows one.
- **CDFI's blanks** (the FAQ: "Blank, zero, or N/A ... indicates" no estimate). CDFI
  leaves a blank for every zero rate (511 poverty, 2,266 unemployment), for 540 incomes
  top-coded at $250,001, 17 bottom-coded at $2,499, 27 poverty rates of 100% and 5
  unemployment rates of 100%. The adapter fills the zeros and the top-coded incomes from
  the ACS (ratio 250,001 / benchmark, a lower bound far above 80). None of these can
  qualify, so the flags still match. The 49 blanks at 100% or the bottom code (46
  tracts) stay blank, since CDFI treats them as not qualifying; 30 of these tracts are
  eligible by another route. Flags `pov_filled`, `unemp_filled` and `mfi_topcoded` are in `extras`. Truly
  missing after the fill: poverty 1,099, MFI 2,375, unemployment 1,022, mostly empty or
  group-quarters tracts.
- **Exceptions.** 1,369 tracts are in an Empowerment Zone or Enterprise Community. 1,569
  are in population-loss counties (495 metro, 1,074 non-metro; 511 of them in Puerto
  Rico). The union is 2,858 tracts, 2,347 in the 50 states and DC.

Sample: drop Puerto Rico (981 tracts, no LODES workplace files), then 2,347 exception
tracts, then 2,173 tracts with a score still blank, then 2,933 tracts without the outcome
(Michigan 2,768, Alaska 165). No remaining tract lacks covariates. n = 76,961.

## Readiness (2026-10-28)

`python -m experiments.scripts.multi_rule_readiness us_tracts_ciia`

| Quantity | Poverty rate | MFI ratio | Unemployment ratio |
|---|---:|---:|---:|
| R^2 of Q on X | 0.632 | 0.624 | 0.257 |
| sd(T), sd(eta) | 8.35, 6.38 | 32.1, 24.9 | 0.40, 0.69 |
| sd(eta) max/min across T deciles | 3.11 | 2.24 | 2.49 |

- corr(T) = -0.84 (poverty, MFI), 0.84 (poverty, unemployment), -0.76 (MFI,
  unemployment); corr(eta) = -0.22, 0.21, -0.09.
- In natural units the weakest direction of T is the unemployment ratio, only because
  its scale is small (sd 0.22, 0.014 of the root mean sd(eta)). Scaled by sd(eta_j), the
  weakest direction is about (-0.35, 0.03, 0.94): unemployment against poverty, with sd
  0.30 sd(eta). Trade-offs involving unemployment are weakly identified, as its score
  model is weak.
- 45.6% are eligible: 14.5% on income only, 9.1% on poverty and income, 8.3% on all
  three, 5.9% on unemployment only, 4.5% on income and unemployment, 2.4% on poverty
  only, 0.9% on poverty and unemployment. 15.5% lie within 0.25 sd(eta_j) (Euclidean, in
  sd(eta_j) units) of the boundary of the either rule. The script's near-boundary share
  (0.020) assumes a weighted-sum rule; ignore it.
- State indicators add 0.014-0.022 to each R^2 (0.618, 0.602, 0.238 without them).
- On the log scale (log(1 + poverty), log ratio, log(1 + unemployment ratio)) the
  sd(eta) ratios fall to 1.56, 1.30 and 1.42, with R^2 0.61, 0.69 and 0.25.
- Outcome: mean change in log jobs 2022-2023 is 0.031 (sd 0.18): 0.034 for ineligible
  and 0.029 for eligible tracts. Placebo 2021-2022: 0.053 (sd 0.19), 0.059 and 0.046.
  0.02% of tracts have zero jobs in 2022.

## Two-score sample (`load_two_score()`)

Tracts that fail the income test, outside the exceptions: MFI ratio above 80, or no MFI
estimate (a blank never qualifies). Among them the deployed rule is
`1{poverty >= 20 or unemployment ratio >= 1.5}`, an either rule on two scores, so the
two-score estimator applies directly. n = 50,028 (48,947 with a ratio above 80, 1,081
with no estimate); 15.6% are eligible: 9.2% on unemployment only, 4.3% on poverty only,
2.1% on both.

| Quantity | Poverty rate | Unemployment ratio |
|---|---:|---:|
| R^2 of Q on X | 0.519 | 0.134 |
| sd(T), sd(eta) | 5.63, 5.41 | 0.24, 0.60 |
| sd(eta) max/min across T deciles | 3.52 | 2.65 |

corr(T1, T2) = 0.68 and corr(eta1, eta2) = 0.19. Scaled by sd(eta_j), the weakest
direction is about (0.27, -0.96), sd 0.28 sd(eta). 8.2% lie within 0.25 sd(eta_j) of
the boundary. Mean outcome 0.033 for both eligible and ineligible tracts.

What the restriction does:
- **It selects on a score.** The sample is `Q2 > 80` on the MFI ratio, which has its own
  noise eta2. Within it, E[eta_j | X] is no longer zero when eta_j is correlated with
  eta2: corr(eta2, eta1) = -0.22 and corr(eta2, eta3) = -0.09 in the full sample. A tract
  with a high predicted income is in the sample whatever its income noise; a tract with a
  predicted ratio near 80 is in it only if its noise is high, and then its poverty noise
  tends to be low. So X independent of eta holds only approximately in this sample, and
  the score model's T is the conditional mean given selection.
- **It fixes the income test.** Counterfactual rules learned here are rules for tracts
  above the income cutoff, holding the deployed income criterion as it is. They do not
  say how to trade income against poverty or unemployment.
- **The unemployment score is weak.** R^2 is 0.13: ACS tract unemployment rates are
  dominated by sampling error, more so among the higher-income tracts kept here.

## Caveats

- **Geography and timing.** CDFI's file, the Planning Database 2023, the ACS summary
  file and LODES8 blocks all use 2020 tracts. Connecticut keeps its eight legacy counties
  in all four (CDFI's FAQ, question 14), so GEOIDs match directly. Every LODES8 tract is
  in CDFI's file. The 2020 census counts date from April 2020, inside the 2016-2020 ACS
  window: X does not predate the scores, but it is measured independently of the ACS
  sampling noise. The 2020 counts carry the Census Bureau's disclosure-avoidance noise,
  which is independent of the ACS.
- **Heteroskedastic noise.** In natural units sd(eta) changes by a factor of 2.2-3.1
  across T deciles, so X independent of eta fails, as in every real dataset screened so
  far. The noise is mostly ACS sampling error.
- **Discrete scores.** Poverty and unemployment rates are rounded to 0.1, so Q3 moves in
  steps of 1/54. 153 sample tracts sit at poverty 20.0 (eligible), 408 at unemployment
  8.1% (not eligible; see "The unemployment cutoff") and 1,962 at unemployment 0. The MFI
  ratio is continuous but stored in single precision.
- **Top-coded income.** 535 sample tracts have MFI top-coded at $250,001
  (`extras["mfi_topcoded"]`); their ratio is a lower bound (about 300%), far above the
  cutoff. Drop them if the score model needs uncensored Q2.
- **Exceptions.** `load()` drops the Empowerment Zone / Enterprise Community tracts and
  the tracts in population-loss counties, so the published flag equals the rule. This
  selects on the zone list and on county population change (ACS county totals, which
  are controlled to population estimates), not on the tract scores. `load_all()` keeps
  them; there 429 sample tracts are eligible only through an exception.
- **Weak, late exposure.** CDFI published the 2016-2020 Investment Areas on January 5,
  2023. New certification applicants had to use them at once; certified CDFIs could keep
  their 2011-2015 Investment Areas until January 5, 2024 or until recertification. LODES
  counts jobs at about the start of the second quarter, so 2023 is a few months after
  publication. The 2011-2015 status (on 2010 tracts, file no longer published) was in
  force before, is correlated with D, and is not observed here. Investment Area status
  affects jobs only through CDFI lending: certified CDFIs must direct at least 60% of
  their financing to their Target Market. Expect small effects. When LODES8 publishes
  2024 or later, add the year to `LODES_YEARS` in `download.py`.
- **Outcome missingness.** 3.8% of tracts in the 50 states and DC have no outcome:
  Michigan (no 2022 or 2023 file) and Alaska (no file after 2016). Puerto Rico has no
  LODES workplace files.
- The 2020 self-response rate is missing for 713 tracts, nearly all with no household
  population (empty or group-quarters tracts). It is set to the median, with a flag. No
  `load()` tract lacks it, so the constant flag is dropped there; `load_two_score()`
  keeps it (3 tracts).
