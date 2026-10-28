# HRSA shortage areas: an integer HPSA score and the NHSC funding cutoff by year

| Field | Value |
|---|---|
| **Q** | Primary care HPSA score, an integer from 0 to 25, from the HRSA HPSA detail file (snapshot of 2026-10-26). 69% of the sample was rescored in HRSA's September 2025 national update |
| **Threshold** | c - 0.5 on Q. c is the effective NHSC loan repayment (LRP) cutoff for the fiscal year: 13 in FY 2025 (the default), 17 in FY 2024, 16 in FY 2023 and FY 2020. FY 2021 and FY 2022 have no binding cutoff |
| **Treatment** | `1{Q >= c}`: the HPSA's sites are in the score range that NHSC LRP funded that year. This is a funding priority, not an eligibility rule |
| **Rule class** | one score with a yearly cutoff. The score is a fixed-weight points sum of four component scales (2 x ratio + poverty + infant health + travel). `load_components()` gives a three-score view |
| **X** | 84 columns, 14 of them continuous. HPSA level: designation type, population type, metro status, rural share, component geography, number of components, log designation population, years since designation and US-Mexico border. County level (AHRF, 2022-2024 data): population, density, poverty, share aged 65+, primary care physicians, NPs and PAs per 10k, rural-urban code, income and uninsured share. Plus 51 state indicators |
| **Y** | **Placeholder, not an outcome.** NHSC primary care FTE per 10,000 residents in the HPSA's counties in 2024 (AHRF). It predates the 2025 scores. `load_sites_2026()` uses NHSC-approved sites per 10,000 residents in October 2026 instead |
| **n** | 8,174 HPSAs: 7,197 designated and 977 proposed for withdrawal |
| **Source** | data.hrsa.gov: HPSA detail, Health Sites, NHSC LRP Program Applicant Metrics FY 2020-2025 and AHRF 2024-2025. HRSA's notices in the Federal Register, 2003-2018 |

## The rule

**The score.** HRSA's criteria for "HPSAs of greatest shortage" (FR 68:32531, 30 May 2003,
doc 03-13478) give each factor a 0-5 scale. For primary care the four factors are:

- population per FTE primary care physician: from >= 3,000:1 (1 point) to >= 10,000:1
  (5 points). With no physicians, the scale uses the population instead (from >= 500 to >= 2,500);
- percent of the population below 100% of poverty: from >= 15% (1 point) to >= 50% (5 points);
- infant health: infant mortality or low birth weight;
- travel time or distance to the nearest source of care.

The ratio scale is double-weighted, so the maximum is 25. Two of the four inputs are in the file:
the designation population and FTE (giving the ratio), and the poverty percent.
`component_check()` rebuilds those two components for the 3,932 non-withdrawn geographic and
population HPSAs. Score - 2 x ratio scale - poverty points lies in [0, 10], the range of the two
unobserved components, for **all 3,932**. Two alternatives fail this check: doubling poverty as
well (99.6%, minimum -5) and using single weights (62%).

The weights differ by type of HPSA:

- Facility HPSAs (FQHCs, look-alikes, rural health clinics and IHS/tribal/urban Indian
  facilities) are scored on the same factors applied to the facility. The file has no component
  inputs for them.
- Correctional facilities get 21, 15 or 9 from their degree-of-shortage group, so they are
  excluded.
- Dental scores run from 0 to 26 (ratio and poverty both doubled, plus fluoridation). Mental
  health scores run from 0 to 25 on different factors.

**The cutoff.** The NHSC funds clinicians at sites in descending order of HPSA score. Two kinds
of official source give the cutoff by year.

*1. NHSC LRP Program Applicant Metrics (data.hrsa.gov, FY 2020-2025).* These report traditional
LRP applicants and awards by site HPSA score. The scores are pooled across primary care, dental
and mental health sites. `lrp_award_table(fy)` reads the table. `effective_cutoff()` returns the
lowest score from which every score with at least 20 applicants has an award rate of at least
25%. It returns no cutoff when the pooled rate below that score is not under 25%.

| FY | Cutoff c | Award rate at c | At c - 1 | At or above c | Below c | Applicants |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 16 | 0.777 | 0.148 | 0.788 | 0.059 | 6,459 |
| 2021 | none | | | 0.863 overall, flat in the score | | 5,849 |
| 2022 | none | | | 0.718 overall, flat in the score | | 5,264 |
| 2023 | 16 | 0.731 | 0.017 | 0.752 | 0.013 | 4,981 |
| 2024 | 17 | 0.487 | 0.059 | 0.750 | 0.036 | 6,647 |
| 2025 | 13 | 0.594 | 0.021 | 0.771 | 0.024 | 6,354 |

- The jump at c is sharp: from 2-15% at c - 1 to 49-78% at c. The score at c is partly funded
  (ties broken by other priorities). Some awards below c still occur (1-6%), probably through
  funding categories that the metrics do not separate.
- FY 2021 and FY 2022 carried American Rescue Plan funding, and the award rate is flat in the
  score.
- The author's example, a 2024 state report naming 16 as the lowest funded score for primary
  care and mental health LRP, matches FY 2023 (16). FY 2024 is 17.

*2. HRSA notices in the Federal Register (2003-2018).* Each notice sets the minimum score for
NHSC scholar placement (and, until FY 2012, for LRP priority). This threshold is strict:
scholars may serve only at sites at or above it. `fr_check()` re-reads every number below from
the notice text, and all 19 notices agree. The last three columns are scholar (SP) thresholds.

| Notice | FR citation | Period | LRP | SP: PC physicians | SP: psychiatrists | SP: dentists |
|---|---|---|---:|---:|---:|---:|
| 03-30820 | 68 FR 69709 | 7/2003-6/2004 | 14 | 14 | 18 | 18 |
| 04-18409 | 69 FR 49906 | 7/2004-6/2005 | 14 | 14 | 20 | 20 |
| 05-17180 | 70 FR 51356 | 7/2005-6/2006 | 14 | 14 | 20 | 20 |
| E6-9974 | 71 FR 36348 | 7/2006-6/2007 | 17 | 14 | 19 | 19 |
| E7-11423 | 72 FR 32853 | 7/2007-6/2008 | 17 | 14 | 19 | 18 |
| E8-13454 | 73 FR 34025 | 7/2008-6/2009 | 17 | 14 | 19 | 17 |
| E9-12531 | 74 FR 25568 | FY 2010 LRP | 10 | | | |
| E9-14741 | 74 FR 29708 | 7/2009-6/2010 | | 17 | 17 | 17 |
| 2010-15356 | 75 FR 36102 | 2010-11 | | 17 (PAs 15) | 17 | 17 |
| 2010-28083 | 75 FR 68368 | FY 2011 LRP | 10 | | | |
| 2011-19505 | 76 FR 46305 | 2011-12 | | 16 | 16 | 16 |
| 2012-1844 | 77 FR 4332 | FY 2012 LRP | 14 | | | |
| 2012-8928 | 77 FR 22329 | 2012-13 | | 16 | 16 | 16 |
| 2013-15112 | 78 FR 38061 | 2013-14 | | 16 | 16 | 16 |
| 2014-08264 | 79 FR 20890 | 2014-15 | | 14 | 14 | 14 |
| 2015-11049 | 80 FR 26569 | 2015-16 | | 16 | 16 | 16 |
| 2016-10527 | 81 FR 27143 | 2016-17 | | 17 | 17 | 17 |
| 2017-09024 | 82 FR 20897 | 2017-18 | | 17 | 17 | 17 |
| 2018-10699 | 83 FR 23470 | 2018-19 | | 18 | 18 | 18 |

The thresholds and their notices are stored in `FR_THRESHOLDS`, and the LRP cutoffs in
`LRP_CUTOFFS`.

- Some notices set separate thresholds for NPs, PAs and CNMs. For example, the 2004-05 and
  2005-06 notices set 13 for NPs and PAs and 8 for CNMs.
- The 2018 notice says that later thresholds would be published on the Health Workforce
  Connector rather than in the Federal Register. The FR API returns no such notice after 2018.

**Not verified (source blocks scripts).** Later scholar thresholds are on nhsc.hrsa.gov, which
returns 403 to scripted requests. Web-search summaries of those pages give:

- class year 2026: 21 for primary care physicians and NPs, 19 for PAs and CNMs, 21 for
  psychiatry and 14 for dentists;
- class year 2027: 21, except 18 for PAs.

The author's example of 19 for primary care physicians (a 2019 HRSA notice, class year 2020)
also comes from that site. None of these numbers is used in the code.

## Getting the data

    python -m experiments.datasets.hrsa_hpsa.download          # idempotent
    python -m experiments.datasets.hrsa_hpsa.download --check  # sizes on disk

The 33 files total 170.3 MB and go to `data/raw/` (gitignored).

- There is no login, form or CAPTCHA.
- The `DD_Files` are regenerated daily or weekly, so a new download is a new snapshot. These
  files were downloaded on 2026-10-28. The HPSA files carry the record date 2026-10-26.
- The adapter caches one HPSA-level table per discipline as
  `data/processed/hrsa_hpsa_{PC,DH,MH}_v1.csv.gz`. Other sessions run the registry against this
  checkout, so bump `CACHE_VERSION` after changing the build.
- `.xlsx` files are read with `experiments/datasets/fl_school_grades/_spreadsheet.py`, because
  the environment has no `openpyxl`.

| File (data.hrsa.gov/DataDownload/...) | Bytes | Use |
|---|---:|---|
| `DD_Files/BCD_HPSA_FCT_DET_PC.csv` | 49,121,341 | primary care HPSAs, one row per component; Q, X, components |
| `DD_Files/BCD_HPSA_FCT_DET_DH.csv` | 29,006,500 | dental HPSAs (`load_dental`) |
| `DD_Files/BCD_HPSA_FCT_DET_MH.csv` | 25,299,252 | mental health HPSAs (`load_mental`) |
| `DD_Files/SDMS_AUTO_HPSA_SITE_PC.csv` | 4,526,102 | sites of auto-designated facility HPSAs; not used yet |
| `DD_Files/SITE_DASHBOARD.csv` | 38,530,008 | Health Sites: 24,383 NHSC-approved sites with county; `load_sites_2026` |
| `DD_Files/HPSA_DATAMART_METADATA.XLSX` | 27,297 | data dictionary |
| `AHRF/AHRF_2024-2025_CSV.zip` | 23,287,136 | county X and the placeholder Y (`nhsc_fte_*_23/24`) |
| `StaticDocuments/FY2020%20NHSC%20LRP%20Program%20Applicant%20Metrics.xlsx` | 42,660 | FY 2020 award rates by score |
| `StaticDocuments/FY2021%20NHSC%20LRP%20Program%20Applicant%20Metrics.xlsx` | 42,724 | FY 2021 |
| `StaticDocuments/FY2022%20NHSC%20LRP%20Program%20Applicant%20Metrics.xlsx` | 43,094 | FY 2022 |
| `StaticDocuments/FY2023%20NHSC%20LRP%20Program%20Applicant%20Metrics.xlsx` | 43,002 | FY 2023 |
| `StaticDocuments/FY%202024%20NHSC%20LRP%20Program%20Applicant%20Metrics.xlsx` | 46,013 | FY 2024 |
| `StaticDocuments/FY2025%20NHSC%20LRP%20Program%20Applicant%20Metrics.xlsx` | 41,430 | FY 2025 |

The Federal Register notices are 20 plain-text files,
`https://www.federalregister.gov/documents/full_text/text/<yyyy/mm/dd>/<doc>.txt`, totalling
271,408 bytes. They are the 19 threshold notices in the table above plus 03-13478 (the scoring
criteria). Dates are in `download.FR_NOTICES`.

**Checked and not used:**

- NHSC Field Strength FY 2019-2025: state totals only.
- NHSC SP and S2S applicant metrics: these have no site score, because the applicants are
  students.
- The HPSA shapefiles.
- Older AHRF releases. The 2023-2024 release would add NHSC FTE for 2022.

**No site-level NHSC file:** the NHSC provider data mart dictionary
(`NHSC_PROV_DATAMART_METADATA.XLSX`) lists site fields with HPSA IDs and clinician FTE, but no
such file is offered for download.

**Blocked:** nhsc.hrsa.gov, bhw.hrsa.gov and www.hrsa.gov return 403 to scripts (2026-10-28).
This covers the scholar threshold pages, the scoring-criteria page and the "how we calculate
scholar placement scores" page.

## Rule check (2026-10-28)

- `lrp_cutoff_check()` derives the cutoffs in the first table from the raw applicant metrics.
  All six agree with `LRP_CUTOFFS`.
- `fr_check()`: all 19 Federal Register thresholds agree with the notice text.
- `component_check()`: the reconstructed components fit the score for all 3,932 area HPSAs.
  `load_components()` reproduces D from the sum of its three columns for all 3,873 HPSAs.
- **No unit-level check of D.** No public file says which HPSAs' sites received awards in a
  year, so D cannot be checked HPSA by HPSA. The applicant metrics check the rule only in
  aggregate, by score. The adapter does not set `extras["eligible_published"]`.

## Readiness (2026-10-28)

`python -m experiments.scripts.multi_rule_readiness hrsa_hpsa` reports, for `load()` (FY 2025,
c = 13):

| Quantity | Value |
|---|---:|
| n, p (continuous X) | 8,174, 84 (14) |
| R^2 of Q on X | 0.468 |
| sd(T), sd(eta) | 2.56, 2.73 |
| sd(eta) max/min across T deciles | 1.74 |
| Treated share | 0.805 |
| Within 0.25 sd(eta) of the cutoff (scores 12 and 13) | 11.4% |

**R^2 without parts of X.** Without the state indicators, R^2 is 0.32. Without the county
covariates it is 0.41. The HPSA-level columns alone give 0.20, and the county columns alone 0.21.

**By type** (`extras["group"]`):

| Type | n | R^2 | Treated share |
|---|---:|---:|---:|
| Geographic | 1,112 | 0.26 | 0.53 |
| High-needs geographic | 294 | 0.53 | 0.73 |
| Population | 2,467 | 0.39 | 0.84 |
| FQHC | 1,321 | 0.59 | 0.88 |
| FQHC look-alike | 175 | 0.69 | 0.81 |
| Rural health clinic | 1,891 | 0.53 | 0.90 |
| IHS / tribal / urban Indian | 914 | 0.64 | 0.75 |

**Other loaders.** The first four rows use the same Q and X as `load()`, so R^2, sd(T) and
sd(eta) are unchanged: 0.468, 2.56 and 2.73.

| Loader | n | Treated share | R^2 | sd(eta) | sd(eta) max/min | Near cutoff |
|---|---:|---:|---:|---:|---:|---:|
| `load_fy(2024)`, c = 17 | 8,174 | 0.388 | 0.468 | 2.73 | 1.74 | 24.8% |
| `load_fy(2023)` or `load_fy(2020)`, c = 16 | 8,174 | 0.524 | 0.468 | 2.73 | 1.74 | 25.2% |
| `load_area()` (geographic and population only) | 3,873 | 0.745 | 0.388 | 2.63 | 1.27 | 14.9% |
| `load_with_mcta()` (MCTA score in X) | 8,174 | 0.805 | 0.541 | 2.54 | 1.82 | 11.4% |
| `load_dental()` (0-26) | 7,461 | 0.854 | 0.571 | 2.98 | 1.62 | 7.7% |
| `load_mental()` (0-25) | 6,438 | 0.895 | 0.441 | 2.33 | 1.56 | 9.0% |

**Three-score view (`load_components()`).** This covers the 3,873 area HPSAs. Q = (ratio points
0-10, poverty points 0-5, infant health + travel points 0-10), and D = 1{Q1 + Q2 + Q3 >= c}.

| Quantity | Ratio points | Poverty points | Infant + travel |
|---|---:|---:|---:|
| R^2 | 0.20 | 0.57 | 0.44 |
| sd(T), sd(eta) | 1.08, 2.16 | 0.71, 0.63 | 1.13, 1.27 |
| sd(eta) max/min across T deciles | 2.05 | 3.15 | 1.40 |

- corr(eta) is near zero for every pair (at most 0.07). corr(T) is 0.54 between poverty and
  infant + travel.
- The weakest direction of T has sd 0.56, which is 0.37 of sd(eta).
- Each component is itself a coarse step function. The ratio points take the values
  0/2/4/6/8/10, and 76% of the poverty points are 0 or 1.

## Outcome

There is no credible post-assignment outcome yet.

**What is missing.**

- No public HRSA file reports NHSC clinicians or awards by HPSA.
- The AHRF's county NHSC FTE stops at 2024. The scores in force now were mostly set in
  September 2025, and the file keeps no score history: rescoring overwrites the old score.
- The next AHRF release would add 2025, and the one after that 2026, the first year clearly
  after these scores.

**`load()`: the placeholder.** Y is the NHSC primary care FTE per 10,000 residents in the HPSA's
counties in 2024, named `y_placeholder_pre_nhsc_fte_per10k_2024`.

- Its mean is 0.39, its sd 0.70, and 36% of units are zero. corr(Y, Q) = 0.13.
- It rises smoothly with the score, from 0.25 at scores 12-13 to 0.59 at 19. There is no jump
  at 13.
- `extras["y_pre_2023"]` holds the 2023 value; its correlation with 2024 is 0.65.

**`load_sites_2026()`.** Y is NHSC-approved sites per 10,000 residents in the HPSA's counties,
from the Health Sites file of October 2026, 13 months after the rescoring.

- n = 8,228. The mean is 2.98 and corr(Y, Q) = -0.13.
- It counts approved sites, not clinicians or awards. Most approvals predate 2025, and it
  differs only at county level.
- Credibility is low.

**`load_dental` and `load_mental`** use the dental and mental health NHSC FTE in the same way.

## Caveats

- **The score is an integer from 0 to 25.**
  - It takes 26 values. In the main sample the mean is 15.3, the sd 3.7 and the median 16.
  - The distribution is: 0-6, 153 HPSAs; 7-9, 501; 10-12, 942; 13-15, 2,296; 16-18, 2,780;
    19-21, 1,266; 22-25, 236. The mode is 16, with 13.6% of the sample.
  - Each component is itself a step function of an underlying ratio or rate. The "continuous
    score plus latent term" model therefore fits only approximately, and the boundary mass is a
    whole score value: 570 HPSAs at 13 and 365 at 12. Methods that need a continuous density
    of Q at the cutoff must treat the score as discrete. One option is to recover the
    continuous inputs (ratio and poverty percent) for area HPSAs.
- **The cutoff is a funding priority, not an eligibility rule.**
  - D says that the HPSA's score was in the funded range. It does not say that its sites
    received awards. Clinicians apply, and award rates at or above c run at 49-94% by score.
  - Some awards below c still occur (1-6%). The cutoff is set after the fact by the budget and
    the applicant pool, and it moves between 13 and 17 across years.
  - Only scholar placement has a strict threshold (the Federal Register table), and no
    verifiable threshold exists for it after 2018.
  - The LRP tables pool primary care, dental and mental health sites, so the primary care
    cutoff may differ from the pooled one.
- **Timing.**
  - FY 2025 awards were made on scores in force before the September 2025 update. The scores
    in this snapshot will drive the FY 2026 cycle, whose cutoff is not yet published.
  - `load()` therefore applies FY 2025's cutoff to the current scores. `extras["score_date"]`
    and `extras["rescored_2025_09"]` (68.8%) record when each score was set. 1,009 scores date
    from 2026.
- **X is not independent of the score's inputs in every column.**
  - The designation population is the numerator of the ratio component.
  - County poverty, physician supply and rurality are separate measurements of quantities that
    also drive the components.
  - FTE, the formal ratio, the HPSA's own poverty percent, degree of shortage and status are
    excluded from X. The first three are kept in the cache for `load_components()`.
  - The MCTA score (corr 0.53 with Q) shares inputs with the HPSA score, so it is only in
    `load_with_mcta()`.
  - sd(eta) varies 1.74-fold across T deciles.
- **Units and clustering.**
  - Many HPSAs share a county (median 2 per county set, maximum 118), so county-level X and Y
    are repeated. Inference should cluster by county.
  - 2.7% of HPSAs span several counties; their county measures are population-weighted sums
    over those counties.
- **Exclusions and gaps.**
  - Excluded: withdrawn HPSAs (their score is no longer in force), 554 correctional facilities,
    11 "Other Facility" HPSAs and 114 HPSAs without AHRF county data.
  - Of the 114, 67 are in Connecticut: 54 have planning-region codes with no NHSC data in AHRF
    2024-2025, and 13 have no county code. The other 47 are 21 in territories, 4 with state
    "XX" and 22 facility HPSAs with no county code.
  - One HPSA has two scores across its rows; the first is used (`score_conflict`).
- **The snapshot changes.** The HPSA and Health Sites files are regenerated daily or weekly.
  Re-downloading changes the sample, so keep the raw files if exact replication matters.
