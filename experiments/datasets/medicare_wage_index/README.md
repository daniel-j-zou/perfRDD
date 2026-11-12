# Medicare low wage index hospital policy: wage index increase on the hospital's wage index (FY 2020-2024)

| Field | Value |
|---|---|
| **Q** | The hospital's FY t wage index before the low wage index adjustment and the 5% cap (Table 2 column "FY t Wage Index Prior to Quartile and Transition/Cap"), after reclassification and the rural floor. FY 2016-2019 (no policy): the FY t wage index in Table 2. `load_dose()` gives Q in wage index points (x100): 84.57 is a wage index of 0.8457 |
| **Threshold** | The year's 25th percentile wage index value p25(t): 84.57 (FY 2020), 84.69 (2021), 84.37 (2022), 84.27 (2023), 86.67 (2024) |
| **Treatment** | Dose A = 0.5 max(p25(t) - Q, 0), in the same points; 0 in FY 2016-2019 and FY 2025-2027 |
| **Rule class** | A kink (half the gap to a percentile) on one score; five policy-year schedules with nearly equal thresholds, plus no-policy years (dose 0 at every score) |
| **X** | log own AHW from the FY t-4 cost report (the data behind Q) and its 2-year change, log beds, occupancy, resident-to-bed ratio, case-mix index, Medicare share, DPP, log Medicare cases, operating CCR, rural, large urban, reclassified, 412.103 rural, RRC, SCH, MDH, teaching, FY and census-region indicators (`x="state"`: state indicators) |
| **Y** | `log_ahw` (default): log of the hospital's average hourly wage from its FY t cost report (Table 2 of FY t+4, column "Average Hourly Wage FY t+4"). `dlog_ahw`: the same minus the FY t-4 value. `log_cases`: log Medicare cases in FY t+1 (Impact File FY t+3) |
| **n** | 14,547 hospital-years (2,213 hospitals), FY 2016-2023, Q < 100 and eta trimmed to the dosed range; 2,495 dosed. Without the two restrictions: 23,163 (3,185 hospitals) |
| **Source** | CMS IPPS final-rule Table 2 (FY 2016-2027, www.cms.gov); IPPS Impact Files via `medicare_dsh`; FY 2020 final rule 84 FR 42326-42328; IFC 89 FR 80405 |

**Fit: weak for the wage outcome.** The rule is reproduced for every hospital in every policy
year, R^2 of Q on X is 0.71 in the analysis sample (0.85 in the full sample), and 83% of
hospital-years are at the reference dose. But the estimated effect on hospital wages is
indistinguishable from zero and from a placebo run on pre-policy years. The five policy
schedules have almost the same threshold, so the score-direction modifier is absorbed
(r2_absorbed 0.9998). Doses above 4.5 points rest on few hospital-years and are unstable.

## The rule

FY 2020 IPPS final rule (CMS-1716-F, 84 FR 42044, 16 Aug 2019), section III.N.2.a, 84 FR 42326-42328:
"increase the wage index for hospitals with a wage index value below the 25th percentile wage
index by half the difference between the otherwise applicable final wage index value for a year
for that hospital and the 25th percentile wage index value for that year across all hospitals".
The rule says it applies "for at least 4 fiscal years beginning October 1, 2019". It was made
budget neutral by an adjustment to the standardized amounts of all hospitals, which the FY
indicators absorb. Footnote 6 of each year's Table 2 restates the rule and gives p25(t).

| FY | p25(t) | Cap on decreases after the policy (Table 2 footnote 6) |
|---|---|---|
| 2020 | 0.8457 | 5% below the FY 2019 wage index, all hospitals (transition) |
| 2021 | 0.8465 final rule, 0.8469 correction notice (used) | 5%, for decreases caused by the new OMB delineations |
| 2022 | 0.8437 | 5%, for hospitals capped in FY 2021 |
| 2023 | 0.8427 | 5% permanent cap, all hospitals (42 CFR 412.64(h)(7)) |
| 2024 | 0.8667 | 5% |
| 2025 final rule / correction notice | 0.9007 / 0.9009 | policy extended "for at least three more years" (89 FR 69301-69308), then removed by the IFC |
| 2025 as paid, 2026, 2027 | no policy | 5%; transitional exceptions for former low wage index hospitals |

- **End of the policy.** On 23 July 2024 the D.C. Circuit held that the Secretary lacked
  authority to adopt the policy for FY 2020. Bridgeport Hosp. v. Becerra, 108 F.4th 882.
  - CMS-1808-IFC ("Changes to the FY 2025 IPPS rates due to court decision", 89 FR 80405,
    published 3 Oct 2024, effective 30 Sep 2024) recomputed the FY 2025 wage indexes without the
    policy and removed its budget neutrality factor.
  - The IFC also pays a non-budget-neutral transitional exception to former low wage index
    hospitals whose FY 2025 wage index fell more than 5% below FY 2024. CMS estimated 113
    hospitals and $41 million.
  - The FY 2026 final rule (90 FR 36536) finalised the IFC and did not continue the policy. A
    transition covers hospitals whose FY 2026 wage index is below 0.95 x 0.95 of FY 2024.
  - So the court ruling that ended the policy was the appellate decision of July 2024, and
    every FY 2025 discharge was paid without it.
- **Which wage index.** The policy applies to the wage index after MGCRB and Lugar
  reclassification, 412.103 rural reclassification and the rural floor ("otherwise applicable
  final wage index"), and before the 5% cap. Table 2 prints it as "Prior to Quartile and
  Transition" (FY 2020-2022) or "Prior to Quartile and Cap" (FY 2023-2025).
- **p25(t)** is the 25th percentile of that column across all Table 2 hospitals, Maryland
  included. It reproduces the published value exactly for FY 2020, 2023 and 2025. For FY 2021,
  2022 and 2024 it is within 0.003 (0.8438, 0.8448, 0.8659): the hospital set differs slightly.
- **Lag.** The FY t wage index uses cost reports for periods beginning in FY t-4 (84 FR
  42301-42302). CMS chose "at least 4 years" so that wage increases made under the policy reach
  the wage index (84 FR 42326-42327).
  - The FY t dose can first change wages in the cost report for periods beginning in FY t
    (1 Oct t-1 to 30 Sep t). That report appears in Table 2 of FY t+4 as "Average Hourly Wage
    FY t+4".
  - Table 2 lists "each hospital's ... adjusted average hourly wage used to construct the wage
    index values for the past 3 years" (84 FR 42323). The values are identical across Table 2
    vintages.
  - Hourly wages are available up to the FY 2023 cost report (FY 2027 Table 2), so doses with an
    outcome cover FY 2020-2023.

## Rule check (2026-11-12)

`adapter.rule_check()` compares the published increase ("With Quartile" minus "Prior to
Quartile") with the formula, tolerance 0.00015 (both columns are printed to 4 decimals).
Maryland is excluded.

| FY | n | Match | Dose > 0 | Cap binds |
|---|---:|---:|---:|---:|
| 2020 | 3,436 | 1.0000 | 0.252 | 0.042 |
| 2021 | 3,388 | 1.0000 | 0.256 | 0.031 |
| 2022 | 3,339 | 1.0000 | 0.251 | 0.009 |
| 2023 | 3,348 | 1.0000 | 0.253 | 0.037 |
| 2024 | 3,325 | 1.0000 | 0.253 | 0.019 |
| 2025 final rule / correction notice (removed by the IFC) | 3,252 | 1.0000 | 0.251 | - |

- **Cap binds** means the final wage index differs from "With Quartile": the 5% cap raised it.
  The cap never lowers it.
  - For those hospitals the formula dose does not move their payment, so `load_dose` drops them
    (`drop_cap=False` keeps them).
- **Impact File.** Its wage index equals Table 2's final value for 94-95% of hospitals in FY
  2016-2017 and 2020-2024.
  - The FY 2018-2019 copies predate the correction notices: median gap 0.05-0.07%.
  - So Q comes from Table 2, not the Impact File.
- **Density at p25(t).** Above : below in a 0.5-point window, pooled: 1.12 in FY 2020-2024 and
  1.13 in FY 2016-2019. In a 1-point window: 1.06 and 1.15.

## Dose design

| Quantity | Value |
|---|---|
| `extras["A"]` | 0.5 max(100 p25(t) - Q, 0) in points (x100); 0 in FY 2016-2019. `published_dose` = Table 2's "With Quartile" minus "Prior", x100 |
| `a_ref` | 0 |
| `dose_levels` | None (continuous) |
| `schedule(q, group)` | the formula; `group` = "<FY>", or "<FY>\|placebo=<p25>" for a pseudo-policy year |
| `Z_names` | `log_beds`, `rural`, `log_ahw_lag` |
| cluster id | `unit_id` (CCN); `cbsa` (payment CBSA) also given |
| other extras | `year`, `p25`, `wi_final`, `outcomes` (all three), `cutoffs`, `units` |

**Sample** (`load_dose()` defaults).
- IPPS hospitals in Table 2 with a 6-digit numeric CCN; Maryland and Puerto Rico are excluded.
  Puerto Rico's wage indexes are about 0.4-0.5, which would give doses of about 20 points.
- Hospital-years where the cap binds are excluded.
- Q < 100 (`q_max`): hospitals with a higher wage index have covariates far from those of the
  dosed hospitals. Their dosed-score covariate values T span 58-110 against 58-176 overall.
- eta within the 0.5-99.5% range of the dosed units' eta (`trim_eta`). Dosed eta spans
  [-20.5, 13.2] points against [-20.3, 26.8] overall. Without this, the eta spline box covers
  regions with no dosed unit and the tensor fit explodes (estimates in the hundreds).
- Units need the FY t and FY t-4 cost-report AHW and the Impact File covariates. Hospitals
  without wage data in a year (new CCNs, closures, multi-campus "B" CCNs) drop out.

**Readiness** (`adapter.readiness()`).

| Sample | n | R^2 of Q on X | sd(eta) | Share at A = 0 (policy years) |
|---|---:|---:|---:|---:|
| analysis (default) | 14,547 | 0.713 (policy years 0.701) | 3.93 | 0.828 (0.643) |
| `x="state"` | 14,787 | 0.788 | 3.40 | 0.831 |
| all Q, no trim (`q_max=None, trim_eta=False`) | 23,163 | 0.853 (`x="state"` 0.898; `x="nowage"` 0.707) | 7.89 | 0.891 (0.773) |

- **Doses.** In the analysis sample: 646, 631, 628 and 590 dosed hospital-years in FY
  2020-2023. Among dosed units the median dose is 2.07 points and the maximum 9.76.
  - In the full sample, corr(A, eta) among dosed units is 0.03 and corr(A, T) is -0.60: the
    dose varies with the part of the score that X predicts.
- **eta quantiles (10/50/90)** by dose: reference -4.4 / 0.6 / 5.4. Dose deciles 1-9 run from
  about -7 / -2 / 3 to -9 / -3 / 2. The top decile (6.2-9.8 points) is -8.7 / -5.0 / -1.0.
- **Outcome.** log AHW has mean 3.735 (about $42.6 an hour) and sd 0.179. The 4-year change has
  sd 0.10.

## Results (2026-11-12)

Runs:
- Command: `app.run("medicare_wage_index", ...)` with `dose_config.config`.
- Logs: `experiments/runs/dose_schedule/medicare_wage_index.log`, `_state.log`, `_placebo.log`,
  `_log_cases.log`.
- Settings: seed 0, clustered by hospital, dose basis with one interior knot (l=1).
- **Why l=1.** With l=3, the a = 4.5 and 6 columns reach 0.3-4 in log points, with rotation sd
  as large.

Effect on log AHW against A = 0, by score bin: estimate (rotation sd, coverage). The dose
occurs only in bins below 84.7.

| Score bin | a=1 | a=2 | a=3 | a=4.5 | a=6 |
|---|---|---|---|---|---|
| 60-75 | -0.012 (0.004, 93%) | -0.015 (0.005, 92%) | -0.012 (0.007, 93%) | -0.005 (0.011, 92%) | 0.031 (0.047, 97%) |
| 75-80 | -0.004 (0.004, 95%) | -0.007 (0.003, 95%) | -0.007 (0.001, 93%) | 0.006 (0.012, 93%) | 0.907 (1.201, 93%) |
| 80-84.7 | 0.001 (0.002, 95%) | 0.001 (0.002, 97%) | 0.003 (0.005, 92%) | 0.018 (0.009, 94%) | 0.796 (0.988, 91%) |
| 84.7-90 | 0.007 (0.002, 95%) | 0.008 (0.004, 98%) | 0.010 (0.009, 90%) | 0.028 (0.014, 93%) | 1.173 (1.443, 86%) |
| 90-100 | 0.018 (0.006, 86%) | 0.019 (0.007, 96%) | 0.019 (0.017, 74%) | 0.050 (0.034, 80%) | 3.336 (4.145, 64%) |

- **Modifier test.** Coefficient -0.0021 (se 0.0055), r2_absorbed 0.9998 (n_fit 9,652). The
  five policy schedules have nearly equal thresholds and the no-policy years have A = 0, so the
  column is absorbed.
- **State indicators** (`wi_x="state"`): -0.010 to +0.017 at a = 1-3 in every bin. Modifier
  0.0071 (se 0.0054), r2_absorbed 0.9999.
- **Placebo.** FY 2016-2019, with the rule applied in FY 2018-2019 at that year's 25th
  percentile: -0.006 to +0.017 at a = 1-3 points (rotation sd 0.005-0.022). This is the same
  size as the main estimates.
- **Check: one linear dose term.** OLS of Y on A, X and a cubic in eta, clustered by hospital.
  Slope per point:

  | Run | Slope per point (se) |
  |---|---|
  | main | -0.0014 (0.0010) |
  | `x="state"` | -0.0000 (0.0010) |
  | placebo FY 2018-19 | -0.0019 (0.0009) |
  | FY 2016-2020 only (first policy year) | +0.0012 (0.0011) |

  A 3-point dose is about 2% of the operating payment per case for a hospital at a wage index
  of 0.8, taking a 62% labor share. The main slope gives -0.4% (se 0.3%) on the hourly wage for
  that dose.
- **Fallback outcome: log Medicare cases in FY t+1** (FY 2016-2024, n 16,465):
  - In the dosed bins: +2.9% to -2.9% at a = 1-3, with rotation sd 0.6-2.8%.
  - In the transported bins (84.7-100): -1.9% to -5.9%. Those bins have no dose; the effect is
    attributed through eta.
  - Linear slope -0.0071 (se 0.0041) per point.
  - The FY t+1 cases fall in 2021-2025 (COVID). Not credible as a dose effect.

## Caveats

- **Repeated doses.** A low wage index hospital is dosed in FY 2020, 2021, 2022 and 2023 at
  nearly the same level. So the FY t outcome reflects the doses received up to t.
  `wi_years=(2016, 2020)` keeps only the first policy year.
- **COVID.** The FY 2020-2021 cost reports (hourly wages) fall in the public health
  emergency. The FY indicators absorb the mean change, not a change that differs by area wage
  level.
- **Identification.** Within a policy year, A is a kinked function of Q = T + eta. Identification
  of f comes from the kink, from the linearity of the baseline in X, and from the no-policy
  years (comparable hospitals at dose 0). This is close to a difference in differences between
  low- and middle-wage hospitals before and after FY 2020.
- **The dose is a price.** It raises the labor-related share of the IPPS operating payment
  (62% for a wage index at or below 1; section 403 of Pub. L. 108-173, 84 FR 42325). Medicare is
  only part of a hospital's revenue.
- **The score is an area wage.** The hospital's own wage is part of its area's AHW. The lagged
  own wage in X makes R^2 0.85 (full sample); without it R^2 is 0.71.
- **Reclassification is in X.** It is chosen by the hospital (MGCRB applications are filed
  before the FY begins). After FY 2020 it may respond to the policy.

## Getting the data

    python -m experiments.datasets.medicare_wage_index.download        # ~11 MB of zips + 4.4 MB of rule text
    python -m experiments.datasets.medicare_dsh.download               # Impact Files (already local)

- **Access.** Plain HTTPS; no login, form or agreement. Downloaded 2026-11-12. Nothing was
  fetched from data.nber.org.
  - The Impact Files for FY 2016-2024 used for X are the NBER copies already in
    `medicare_dsh/data/raw` (downloaded on 2026-10-28 by that adapter). FY 2025-2027 come from
    cms.gov.
- **robots.txt.**
  - www.cms.gov disallows /core/, /profiles/, /node/, /media/, /user/, /admin/ and query strings
    other than the sitemap. /files/ and /medicare/ are allowed.
  - www.federalregister.gov disallows only search, e-mail, /my/ and /auth/ paths.
  - Copies of both are in `data/raw/robots/`.
- **Table 2 URLs** were found in the cms.gov sitemap. The FY 2019-2024 final-rule home pages
  return "Page Not Found" at the new URL pattern. `data/raw/pages/sitemap_ipps_urls.txt` keeps
  the IPPS-related sitemap entries.
- **Files.** The zips are read with `zipfile` into `data/raw/extracted/`; the FY 2025 zip holds
  two inner zips (final rule + CN, and IFC). Nothing downloaded is executed.

| File | Source |
|---|---|
| `tables/fy2016_tables_2_3_4.zip` .. `fy2018` | https://www.cms.gov/medicare/medicare-fee-for-service-payment/acuteinpatientpps/downloads/fy2016-cms-1632-fr-table-2-3.zip (fy2017-cms-1655-, fy2018-cms-1677-) |
| `tables/fy2019_tables_2_3_4.zip`, `fy2020` | .../downloads/fy2019-cms-1694-fr-tables-2-3-4.zip, .../downloads/fy2020-fr-tables-2-3-4.zip |
| `tables/fy2021_tables_2_3_4.zip` | https://www.cms.gov/files/zip/tables-2-3-4a-and-4b-fy-2021-wage-index-tables-final-rule-and-correction-notice.zip |
| `tables/fy2022..2027` | https://www.cms.gov/files/zip/fy-2022-ipps-fr-tables-2-3-4a-4b.zip, fy2023-ipps-fr-tables-2-3-4a-4b.zip, fy2024-ipps-fr-tables-2-3-4.zip, fy-2025-ipps-final-rule-tables-2-3-and-4a-and-4b.zip, fy2026-ipps-fr-tables-2-3-4a-4b.zip, fy2027-ipps-fr-tables-2-3-4a-4b.zip |
| `docs/fr_2019-16762_fy2020.txt` | https://www.federalregister.gov/documents/full_text/text/2019/08/16/2019-16762.txt |
| `docs/fr_2024-22765_ifc.txt` | https://www.federalregister.gov/documents/full_text/text/2024/10/03/2024-22765.txt |

- **Table 2 versions used.** For each FY, the last version published:
  - correction notice: FY 2016-2021 and 2024 (FY 2017: CN2);
  - FY 2022: the correcting amendment (F2);
  - FY 2023 and 2026: the final rule;
  - FY 2027: the CN;
  - FY 2025: the IFC, which is what was paid. The FY 2025 FR/CN "With Quartile" column is kept
    as `wi_quart_fy2025_fr`.
- **Cache.** `data/processed/wage_index_panel.csv.gz` (version 1). Delete it after changing the
  adapter.

## Not verified

- The hourly wage in Table 2 is the "adjusted" hospital AHW. Whether it is occupational-mix
  adjusted at the hospital level was not checked.
- The exact hospital set CMS used for p25(t) in FY 2021, 2022 and 2024 was not found
  (reproduced within 0.003).
- The D.C. district court decision that preceded the appeal was not read. The dates above come
  from the IFC text.
- Not looked at: the FY 2024 policy year (its outcome, the FY 2024 cost report, appears in the
  FY 2028 Table 2, about August 2027) and the FY 2025 removal (a reverse schedule; cost report
  FY 2025 in the FY 2029 Table 2).
