# Medicare DSH: the operating DSH adjustment on the disproportionate patient percentage

| Field | Value |
|---|---|
| **Q** | Disproportionate patient percentage (DPP), in percent, as listed in the IPPS Impact File for FY t: SSI fraction + Medicaid fraction, from the Provider Specific File and the latest SSI data |
| **Threshold** | 15 (eligibility). Kink at 20.2; for capped hospitals the 12% cap is reached at DPP = 27.62 |
| **Treatment** | Dose A = operating DSH adjustment, in percent of the DRG payment, from the formula. `load()`: D = 1{Q >= 15} |
| **Rule class** | A jump (0 to 2.5%) and a kink on one score. A cap applies for one class of hospitals (two schedules), and the whole formula is scaled by 0.25 from FY 2014 |
| **X** | 69 columns: log beds, occupancy, resident-to-bed ratio, case-mix index, Medicare share of days, log Medicare cases (FY t-2 MedPAR), wage index, operating cost-to-charge ratio; rural, large-urban, RRC, SCH, MDH and teaching flags; FY and state indicators |
| **Y** | `load_dose()`: log Medicare cases in FY t+1 (MedPAR count printed in the FY t+3 Impact File). `outcome="exit"`: hospital absent from the FY t+2 Impact File. `outcome="dpp_lead"`: DPP in the FY t+3 file |
| **n** | 19,061 hospital-years (3,369 hospitals), FY 2008-2013. `years=(2014, 2024)`: 33,407 |
| **Source** | CMS IPPS Impact Files: NBER's CSV copies for FY 2005-2024, CMS final-rule files for FY 2025-2027; 42 CFR 412.106(d) |

**Fit: usable but weaker than the MA benchmarks.**
- **The rule.** It is reproduced exactly from FY 2007 on.
- **First stage.** X predicts Q modestly: R^2 = 0.37, sd(eta) = 13.0 points.
- **Weaknesses.**
  - Q is CMS's projection basis (the DPP in the Provider Specific File, from cost reports
    about two to three years old). The DSH actually paid for FY t is settled on the FY t
    cost report, so the received dose is a noisy function of Q.
  - From FY 2014 most DSH money goes through uncompensated-care payments. Those switch on
    at the same 15% eligibility line and are not in A.
  - There is a modest excess of hospitals just above 15%: 1.23 in a 1-point window
    against 0.96-1.03 at placebo points, though 1.01 in a 0.25-point window.

## The rule

42 CFR 412.106(d), as CMS applies it in the Impact Files. The cap classes and
exceptions below were checked against the files; they are not taken from the regulation
text, which was not downloaded.

| DPP (Q) | Operating DSH adjustment (percent) |
|---|---|
| < 15 | 0 |
| 15 to < 20.2 | 2.5 + 0.65 (Q - 15) |
| >= 20.2 | 5.88 + 0.825 (Q - 20.2) |

- **Cap of 12%** (`capped_class = 1`):
  - urban hospitals (payment-area urban, `URSPA`) with fewer than 100 beds; a missing bed
    count counts as fewer than 100;
  - rural hospitals with fewer than 500 beds, unless they are a rural referral center
    (provider type 7, 15, 17, 22) or, from FY 2007, a Medicare-dependent hospital (14, 15).
  - **MDH exceptions.** In the FY 2013 file MDH status had lapsed and MDHs are capped. In
    the FY 2015 file MDHs have a blank provider type and are uncapped.
  - **Urban RRCs** with fewer than 100 beds are capped.
- **Pickle hospitals** (412.106(c)(2)) get 35% (8.75% from FY 2014). They are recognised
  by that published value: 9 a year to FY 2019, 1-2 after. They are excluded from the
  sample.
- **From FY 2014** (section 1886(r)) the operating DSH adjustment is 25% of the above. The
  other 75%, reduced by the change in the uninsured rate, is paid as an uncompensated-care
  payment (UCP).
  - The UCP is shared among DSH-eligible hospitals by their uncompensated-care costs
    (`extras["ucp_adj"]`, the share of the pool).
  - A UCP is listed almost only when Q >= 15: 29 exceptions in FY 2014-2027, 25 of
    them in FY 2019.
  - About 330-420 eligible hospitals a year have none listed (Maryland and Puerto Rico are
    included in that count).
- **Schedule.** `extras["schedule"](q, group)` applies the table. `group` =
  "<FY>|capped" or "<FY>|uncapped"; the year sets the 0.25 factor.

## Rule check (2026-10-28)

`adapter.rule_check()` compares the formula with the published operating DSH adjustment
(`dshopg`, `DSHOPP` from FY 2020). It covers all IPPS hospitals in the 50 states and DC
(Maryland excluded), with agreement within 0.0005.

| FY | n | Pickle | Match | Eligible (Q >= 15) | Capped class | At the cap |
|---|---:|---:|---:|---:|---:|---:|
| 2005 | 3,884 | 9 | 0.9959 | 0.708 | 0.453 | 0.341 |
| 2006 | 3,726 | 9 | 0.9973 | 0.726 | 0.426 | 0.362 |
| 2007 | 3,578 | 9 | 1.0000 | 0.748 | 0.367 | 0.381 |
| 2008 | 3,516 | 9 | 1.0000 | 0.757 | 0.361 | 0.381 |
| 2009 | 3,520 | 9 | 1.0000 | 0.771 | 0.358 | 0.390 |
| 2010 | 3,499 | 9 | 1.0000 | 0.762 | 0.351 | 0.383 |
| 2011 | 3,455 | 9 | 1.0000 | 0.778 | 0.346 | 0.407 |
| 2012 | 3,404 | 9 | 1.0000 | 0.787 | 0.338 | 0.424 |
| 2013 | 3,403 | 9 | 1.0000 | 0.799 | 0.393 | 0.438 |
| 2014-2027 | 3,386 → 2,979 | 1-9 | 1.0000 (one miss, FY 2016) | 0.80-0.87 | 0.34-0.39 | 0.44-0.53 |

- **Misses.**
  - FY 2005-2006: 26 misses, cause not traced (probably transitional MMA rules), so the
    default sample starts in FY 2008.
  - Three urban hospitals in FY 2010 have no bed count, and CMS caps them. Treating a
    missing count as fewer than 100 beds reproduces this.
  - FY 2016: one rural SCH with 598 beds that CMS caps.
- **Within the sample** the formula equals the published dose in every row, and
  `schedule(Q, group)` reproduces A.

## Dose design

| Quantity | Value |
|---|---|
| `extras["A"]` | operating DSH adjustment (percent) from the formula; `published_dose` = CMS's value (same in every sample row) |
| `a_ref` | 0 (DPP < 15) |
| `dose_levels` | None (continuous above 15: jump, kink, cap) |
| `schedule(q, group)` | the table above; `group` = "<FY>\|capped" or "<FY>\|uncapped" (12 groups in FY 2008-2013) |
| `Z_names` | `log_beds`, `resbed`, `cmi`, `rural` |
| unit id | `unit_id` (CMS certification number) |
| other extras | `year`, `capped_class`, `factor` (1 or 0.25), `ucp_adj`, `beds`, `log_cases`, `exit`, `dpp_lead`, `dpp_lag1`, `cutoffs` |

- **Why FY 2008-2013 is the default.** Over these years the DSH formula is the whole
  policy.
  - From FY 2014 the formula dose is a quarter as large. A UCP of similar or larger size
    switches on at the same 15% line, so A understates the jump at 15.
  - FY 2005-2006 still carry pre-MMA cap rules for a few hospitals.
  - Y needs the FY t+3 file, so t <= 2024.
- **Timing.** Impact File FY t is published with the final rule in August of t-1. Its DPP
  is the interim payment basis for FY t, and its "Bills" count is FY t-2 MedPAR.
  - Y = log of the "Bills" count in the FY t+3 file, which is FY t+1 MedPAR: the first
    full year after the dose.
  - X is from the FY t file. The covariates are cost-report or claims values from before
    FY t.
- **Shape of the dose.**
  - 33% of the sample is in the capped class, and 40% of the sample is at the 12% cap.
  - Q's 10/50/90% points are 9.1, 24.3 and 47.9.
  - Uncapped hospitals keep rising to 71% at Q = 100, so the upper tail of A is driven by
    large urban hospitals.

## Readiness (2026-10-28)

`adapter.readiness()` on `load_dose()`:

| Quantity | Value |
|---|---:|
| n, hospitals, p | 19,061, 3,369, 69 |
| R^2 of Q on X | 0.371 (FY 2014-2024: 0.410) |
| sd(Q), sd(T), sd(eta) | 16.42, 10.00, 13.02 (FY 2014-2024: sd(eta) 12.95) |
| Share at the reference dose (A = 0) | 0.208 (FY 2014-2024: 0.159) |
| Above : below, 0.5-point windows | 15: 242 : 201 = 1.20; 20.2: 347 : 344 = 1.01; 27.62 (cap point): 272 : 276 = 0.99 |
| Same at placebo points 10 / 12.5 / 17.5 / 25 / 30 / 35 | 1.00 / 0.93 / 0.94 / 0.99 / 0.84 / 0.91 |

eta = Q - OLS fit, at the reference and by decile of A among A > 0:

| Group | n | eta q10 | q50 | q90 |
|---|---:|---:|---:|---:|
| A = 0 | 3,974 | -23.3 | -9.7 | 0.7 |
| A 2.50-4.32 | 1,509 | -13.6 | -5.2 | 2.2 |
| A 4.32-5.86 | 1,509 | -12.4 | -4.2 | 3.3 |
| A 5.86-7.79 | 1,508 | -12.1 | -3.5 | 3.7 |
| A 7.79-10.02 | 1,510 | -10.6 | -2.2 | 5.4 |
| A 10.02-12.00 | 3,550 | -6.0 | 2.7 | 16.7 |
| A = 12.00 (cap) | 2,205 | -3.4 | 5.4 | 22.0 |
| A 12.00-14.26 | 3,182 | -5.1 | 3.6 | 17.9 |
| A 14.26-18.98 | 1,508 | -6.8 | 2.5 | 9.9 |
| A 18.98-28.63 | 1,509 | -1.3 | 7.6 | 16.0 |
| A 28.63-71.46 | 1,509 | 6.5 | 22.9 | 44.5 |

- **Bunching at 15, pooled over all FY 2005-2027 hospital-years.**
  - Above : below is 1.01 (0.25-point window), 1.08 (0.5) and 1.23 (1.0).
  - At 20.2 it is 1.00-1.01, and at placebo points 0.96-1.11.
  - Split by period, the 0.5-point ratio is 1.06 before FY 2014 and 1.11 after, when
    crossing 15 also brings the UCP.
  - The DPP depends on Medicaid days that hospitals report and contest, so some sorting
    near 15 is plausible.
- **Outcome.** Y has mean 7.47 (about 1,750 Medicare cases) and sd 1.25. R^2 of Y on X is
  0.93, because last cases are in X. corr(Y, Q) = 0.02.
  - `outcome="exit"`: n = 19,859, mean 2.75%.

## Sample

- **Units.** IPPS hospital x FY, FY 2008-2013, 50 states and DC.
- **Who drops.**
  - Of 21,382 hospital-years, Maryland and Puerto Rico remove 585 (Maryland is outside
    IPPS; Puerto Rico has no SSI).
  - Pickle hospitals remove 54. A DPP missing or above 100 removes 102 (the files contain
    a few DPP > 1 as a fraction).
  - A missing FY t+3 record removes 1,020 (closures, conversions to critical access, new
    CCNs). The rest drop for missing X.
- **Mergers and CCN changes** are not tracked. A hospital that changes CCN looks like an
  exit.

## Getting the data

    python -m experiments.datasets.medicare_dsh.download          # ~37 MB
    python -m experiments.datasets.medicare_dsh.download --check

- **Access.** Plain HTTPS; no login, form or agreement (downloaded 2026-10-28).
- **Not downloaded.** The eCFR text of 412.106 was not fetched: ecfr.gov is outside the
  sources authorised for this task, and its API answered 406.
- **Extraction.** The FY 2025-2027 zips are extracted with `zipfile` into
  `data/raw/extracted/`. The adapter reads the tab-delimited "Final Rule" text file, not
  the correction notice.
- **Cache.** `data/processed/dsh_panel.csv.gz` (version 2). Delete it after changing the
  adapter.

| File | Source | Bytes |
|---|---|---:|
| `impact/impact<y>.csv`, `desc_impact<y>.txt` (FY 2005-2024; descriptions to 2018) | https://data.nber.org/cms/impact/<y>/ (NBER copies of the CMS Impact File PUF; originals listed on CMS's "Historical Impact Files" page and the FY final-rule pages) | 26.5 MB |
| `impact/fy2025_impact.zip` | https://www.cms.gov/files/zip/fy-2025-ipps-fr-impact-file.zip | 4.8 MB |
| `impact/fy2026_impact.zip` | https://www.cms.gov/files/zip/fy2026-ipps-fr-impact-file.zip | 1.6 MB |
| `impact/fy2027_impact.zip` | https://www.cms.gov/files/zip/fy2027-ipps-fr-impact-file.zip | 3.0 MB |

## Caveats and possible extensions

- **The dose is a price.** The DSH adjustment raises Medicare's payment per case. Its
  effect on Medicare volume is a supply response, and on finances it is income.
- **Settled vs projected DPP.** See "Fit".
  - `dpp_lead` (the FY t+3 file's DPP, from cost reports near FY t) can stand in for the
    settled DPP. `extras["dpp_lag1"]` is the previous year's file.
  - The Impact File DPP moves within a hospital from year to year as SSI and cost-report
    data update.
- **Financial outcomes need HCRIS, which this folder does not have.** Operating margin and
  uncompensated-care cost (S-10) come from the cost reports.
  - CMS posts the raw HCRIS files at www.cms.gov (Cost Reports, Hospital-2010 form; about
    60-100 MB per year).
  - Each IPPS final-rule page also posts an "HCRIS data file" (23 MB for FY 2025) and a
    "Medicare DSH supplemental data file" with each hospital's uncompensated-care data.
  - Both fit the disk budget but need a cost-report parser.
- **Capital DSH** (a separate formula on the low-income percentage) is not modelled.
