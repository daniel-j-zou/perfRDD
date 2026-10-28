# CEJST v2.0 census tracts: low income AND a burden indicator

| Field | Value |
|---|---|
| **Q** | (Q1, Q2) on a 0-100 percentile scale: Q1 = low-income percentile (adjusted % of people below 200% of the federal poverty line, college students excluded); Q2 = max over the 23 percentile burden indicators paired with low income (lead paint counted only where the home-value percentile is below 90) |
| **Threshold** | (65, 90) |
| **Treatment** | `1{Q1 >= 65 and Q2 >= 90}` (`treatment_rule`), the core both-cutoffs rule. The published final flag is in `extras["eligible_published"]` |
| **Rule class** | two scores, AND rule; the published designation adds OR-branches outside Q (yes/no burdens, workforce, tribal lands, neighbours, v1.0 grandfathering) |
| **X** | % Black, American Indian / Alaska Native, Asian, Native Hawaiian / Pacific Islander, two or more races, Hispanic, other race; % under 10 and over 64; % not enrolled in higher education; log(1 + population); 51 state indicators (62 columns; 11 continuous) |
| **Y** | **Placeholder: all zeros.** The file has no outcome measured after designation |
| **n** | 73,273 tracts (2010 boundaries) in the 50 states, DC and Puerto Rico |
| **Source** | CEQ, CEJST version 2.0 (released 19-20 December 2024): `2.0-communities.csv`, the codebook and the Technical Support Document. Original CDN files retrieved from Internet Archive captures of 22-23 January 2025 |

**Fit: weak.** `load()` is built with a placeholder outcome, not skipped.
- **No outcome.** The file has no outcome after designation, so `load()` is useful for
  rule and score checks (readiness) only, not for estimation.
- **Weak trade-off between the two scores.** Q2 is a maximum over 23 percentiles, so it is
  at or above 90 for 78% of tracts and for 92% of low-income tracts. The low-income
  cutoff does most of the work, and the weakest direction of T has sd 0.16 of sd(eta).
- **The designation is no longer in use.** Federal agencies used it for Justice40 from
  v1.0 (November 2022) until public access ended on 22 January 2025 (PEDP's mirror
  page).

**Version.** The task brief said "version 2.0 (2022)". Version 1.0 was released in
November 2022. Version 2.0, which is what PEDP's mirror serves and what is downloaded
here, was released in December 2024 (`readme-version-2.0.md`; TSD cover, "December
2024"). It keeps v1.0's rule and adds the territory, water-boundary and grandfathering
changes listed below.

## The rule

Source: *Version 2.0 of the CEJST: Technical Support Document* (December 2024; TSD),
Table 1 and sections II.B-F, and CEQ's scoring code `score_narwhal.py` (in `data/raw/code/`).

**1. Categories of burden.** A tract meets a category if it is at or above the 90th
percentile on any of its indicators AND at or above the 65th percentile for low income:

| Category | Indicators (>= 90th percentile unless noted) |
|---|---|
| Climate change | expected agriculture, building and population loss rates; projected flood risk; projected wildfire risk |
| Energy | energy cost; PM2.5 |
| Health | asthma; diabetes; heart disease; low life expectancy |
| Housing | historic underinvestment = yes; housing cost; lack of green space; lack of indoor plumbing; lead paint (and median home value < 90th percentile) |
| Legacy pollution | abandoned mine land = yes; Formerly Used Defense Site = yes; hazardous-waste proximity; Superfund (NPL) proximity; RMP-facility proximity |
| Transportation | diesel particulate matter; transportation barriers; traffic proximity and volume |
| Water and wastewater | underground storage tanks and releases; wastewater discharge |

**2. Workforce development.** Linguistic isolation (not used in Puerto Rico), low median
income relative to the area, poverty (< 100% FPL) or unemployment at or above the 90th
percentile, AND at least 10% of adults 25+ without a high-school diploma. This category
uses the education threshold *instead of* low income.

**3. Low income.** The share of people in households at or below 200% of the federal
poverty line, excluding college students. The source is ACS 2018-2022, with the student
adjustment from 2015-2019 table B14006, and it is imputed from neighbours when missing.
A tract is low income at or above the 65th percentile.

**4. Additions outside the burden thresholds.**
- **Tribal lands.** Tracts with at least 99.5% of their area on the lands of Federally
  Recognized Tribes.
- **Neighbours.** Tracts completely surrounded by disadvantaged tracts and at or above
  the 50th low-income percentile. Water boundaries are excluded from "surrounded" in
  v2.0, and grandfathered tracts do not count as neighbours.
- **Territories.** Tracts in American Samoa, Guam, the Northern Mariana Islands and the
  US Virgin Islands that meet the low-income threshold alone.
- **Grandfathering.** Every tract disadvantaged in v1.0 stays disadvantaged in v2.0.

**5. Percentiles.** Percentiles are national ranks, and the CSV floors them to whole
numbers (codebook: "All percentiles are floored"). So "floored >= 90" is exactly
"rank >= 0.90". Two columns are on a 0-1 scale (low income, no kitchen/plumbing), and
the adapter puts them on 0-100.

## Rule check (2026-10-28)

**Pieces of the rule, from the published percentiles.**
- **Low income.** `Is low income?` = `1{Q1 >= 65}` for every tract outside the four
  island territories, which use a territory percentile.
- **Burden flags.** All 22 published "... >= 90th percentile ... and is low income" flags
  are reproduced exactly from the floored percentiles and the low-income flag.
  - For lead paint, this holds with home value `< 90` on the floored percentile. With
    `<= 90` there are 25 mismatches.
  - No-kitchen/plumbing has no published flag, but leaving it out breaks the next check
    for 153 tracts.
- **"Without considering neighbors".** The published flag is rebuilt from the published
  columns: core OR yes/no burdens with low income OR workforce OR tribal overlap OR
  island tract with low income. It matches for 74,133 of 74,134 tracts. The exception is
  a Guam tract with no income data.

**Both-cutoffs rule against the published final flag.** `adapter.rule_check()`, on the
`load()` sample:

| Quantity | Tracts |
|---|---:|
| Sample | 73,273 |
| Core D = 1 (`Q1 >= 65 and Q2 >= 90`) | 23,875 (32.6%) |
| Published "Identified as disadvantaged" | 28,453 (38.8%) |
| **Agreement** | **0.9375** |
| Disagreements with core D = 1 and published = 0 | 0 |
| Disagreements with D = 0 and published = 1, attributed in this order: | 4,578 |
| - yes/no burden (historic underinvestment, abandoned mine, FUDS) with low income | 92 |
| - workforce (education-paired) | 2,407 |
| - tribal overlap (>= 99.5% tribal land) | 303 |
| - neighbours (surrounded, low income >= 50th percentile) | 272 |
| - grandfathered from v1.0 | 1,504 |

Against "without considering neighbors" (no neighbour or grandfathering additions),
agreement is 0.962. The core rule is a subset of every published flag. The extra
branches only add tracts.

## Readiness (2026-10-28)

`python -m experiments.scripts.multi_rule_readiness cejst_tracts` reports:

| Quantity | Q1 (low income) | Q2 (max burden) |
|---|---:|---:|
| R^2 on X | 0.474 | 0.199 |
| sd(T), sd(eta) | 19.68, 20.75 | 2.98, 5.98 |
| sd(eta) max/min across T deciles | 1.58 | 2.62 |

| Quantity | Value |
|---|---:|
| n, p (continuous X) | 73,273, 62 (11) |
| Treated share (core rule) | 0.326 |
| Agreement with the published flag | 0.9375 |
| corr(T1, T2), corr(eta1, eta2) | 0.541, 0.279 |
| Weakest direction of T; its sd; over sd(eta) | (0.083, -0.997); 2.50; 0.164 |
| Within 0.25 sd(eta1) of the low-income cutoff | 11.1% |

Reading the readiness output:
- **`near_boundary_share`.** The adapter sets `extras["weights"] = (1, 0)` and
  `extras["cutoff"] = 65`, so this figure measures distance to the low-income cutoff
  only. The script's default (average of Q at the first threshold) has no meaning for
  an AND rule. 8.0% of tracts (5,889) are within 2 points of the L-shaped boundary.
- **Q2 is bunched near the top.** Its quartiles are 90 / 95 / 98, which explains the
  low R^2, the small sd(T2) and the strong heteroskedasticity.
  - P(Q2 >= 90) is 0.92 when Q1 >= 65 and 0.70 when Q1 < 65.
  - `extras["argmax_burden"]` records which indicator gives the maximum.

## Getting the data

    python -m experiments.datasets.cejst_tracts.download                 # ~0.42 GB
    python -m experiments.datasets.cejst_tracts.download --no-shapefile  # skip the 374 MB zip
    python -m experiments.datasets.cejst_tracts.download --check

Files are written to `data/raw/` (gitignored). There is no login, form or CAPTCHA.

| File | URL | Bytes | SHA-256 (first 16) |
|---|---|---:|---|
| `2.0-communities.csv` | https://web.archive.org/web/20250122234715id_/https://static-data-screeningtool.geoplatform.gov/data-versions/2.0/data/score/downloadable/2.0-communities.csv | 45,316,831 | ee6851b008da9704 |
| `cejst-technical-support-document.pdf` | https://web.archive.org/web/20250122234715id_/https://static-data-screeningtool.geoplatform.gov/data-versions/2.0/data/score/downloadable/cejst-technical-support-document.pdf | 793,196 | 25dd3cadefe630cf |
| `CEQ-CEJST-Instructions.pdf` | https://web.archive.org/web/20250122234717id_/https://static-data-screeningtool.geoplatform.gov/data-versions/2.0/data/score/downloadable/CEQ-CEJST-Instructions.pdf | 233,861 | 2933cd1c3b1448a2 |
| `2.0-shapefile-codebook.zip` | https://web.archive.org/web/20250123171315id_/https://static-data-screeningtool.geoplatform.gov/data-versions/2.0/data/score/downloadable/2.0-shapefile-codebook.zip | 374,345,668 | 41c0de7c721d26f2 |
| `2.0-codebook.csv`, `readme-version-2.0.md` | extracted from the zip (it also holds `usa.zip`, the tract shapefile, left inside) | 42,050; 1,400 | 4c97cfaa3276c0f4 (codebook) |
| `code/score_narwhal.py` | https://raw.githubusercontent.com/Public-Environmental-Data-Partners/cejst-2/45f46d52d2fd3797a3c67836e918d4d5762c0010/data/data-pipeline/data_pipeline/score/score_narwhal.py | 47,050 | |
| `code/field_names.py` | same commit, `.../score/field_names.py` | 33,936 | |
| `code/field_descriptions_for_codebook.yml` | same commit, `.../content/config/field_descriptions_for_codebook.yml` | 5,738 | |

Total downloaded: 420,776,280 bytes.

## Provenance

- **The official site is gone.** `screeningtool.geoplatform.gov` and its file CDN
  `static-data-screeningtool.geoplatform.gov` went offline on 22 January 2025. The CDN
  host does not respond today (checked 2026-10-28).
- **The data files are Wayback captures of the original CDN URLs.** They come from
  22-23 January 2025: the CSV, TSD and instructions from the End of Term 2024 crawl, the
  zip from "Save Page Now".
  - The `id_` form returns the archived bytes without rewriting.
  - Each response carries the original server's `Last-Modified` header, 20-21 December
    2024 UTC, the v2.0 release. The zip's members are dated 2024-12-20.
- **Cross-check of the TSD.** The TSD is byte-identical to the copy in PEDP's repository
  (git blob `90437307...`).
- **Cross-check of the CSV.** No second copy of the CSV was found. Its consistency is
  checked internally: every published per-indicator flag and the "without considering
  neighbors" flag are reproduced from its own columns (see the rule check).
- **The PEDP mirror is not the file source.** Public Environmental Data Partners re-host
  the v2.0 map at https://screening-tools.com/climate-economic-justice-screening-tool
  (via https://digitalhistory.github.io/j40-pages/en/). Its download page still links
  to the dead CDN, so it serves no files.
  - PEDP's public copy of the code (CC0), https://github.com/Public-Environmental-Data-Partners/cejst-2,
    is the source of the scoring code, pinned to commit `45f46d52` (2025-12-08).
- **Not downloaded.** Version 1.0 files (`data-versions/1.0/...`) also exist in the
  Wayback Machine. The v1.0 designation is in the v2.0 CSV as
  `Identified as disadvantaged in v1.0`.

## Caveats

- **No outcome.** `Y` is all zeros (`extras["y_placeholder"] = True`). Any estimator
  run on `load()` returns nothing meaningful. A post-designation outcome would have to
  come from outside, for example federal grant awards by tract.
  - Such an outcome would also need a crosswalk: CEJST uses 2010 tracts and most
    post-2020 data use 2020 tracts (relationship file linked in PEDP's README).
- **Covariates.**
  - The race/ethnicity and age shares are ACS figures the TSD says are "not used in the
    methodology". They are likely from the same ACS 5-year sample as the low-income
    rate, so their sampling errors may correlate with eta1.
  - `pct_not_in_college` is an input to the student adjustment of Q1.
  - 0.2% of tracts have imputed income (`extras["income_imputed"]`).
- **Excluded tracts.**
  - The four island territories (132 tracts) have no burden data and a separate rule.
  - 729 tracts in the states, DC and PR are dropped: 703 with zero population (no
    demographics) and 26 with no population or low-income data.
- **Partially tribal tracts.** Tracts partly on tribal land are "partially
  disadvantaged" in the tool. The final flag (and this adapter) counts only tracts with
  at least 99.5% tribal land.
- **Spatial dependence.** Tracts are spatially dependent, and the neighbour rule makes D
  depend on neighbours' scores. Inference should account for this, for example by
  clustering by county (the first 5 characters of `extras["geoid"]`).
