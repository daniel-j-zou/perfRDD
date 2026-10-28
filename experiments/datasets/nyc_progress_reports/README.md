# NYC School Progress Reports: an A-F grade from the sum of three category scores

| Field | Value |
|---|---|
| **Q** | `load()` and the other single-score loaders: overall score - c, where c is the published minimum overall score of the higher grade for the school's type (elementary, K-8, middle) and year. `load_two_score()`: (Q1, Q2), with Q1 = 100 (overall - Progress - 0.4 c) / 40 and Q2 = 100 (Progress - 0.6 c) / 60 |
| **Threshold** | 0 on Q. On the overall score, the cut differs by type and year (table below). `load()` uses the C cut, for example 32.6 / 38.4 / 39.0 (elementary / K-8 / middle) in 2007-08 |
| **Treatment** | `load()`: `1{Q >= 0}`, grade C or better rather than D/F. The consequences fall on D/F (D = 0). `load_ab`, `load_bc`, `load_df`: the other boundaries, D = the higher grade |
| **Rule class** | A fixed-weight sum of category scores (Environment 15 + Performance 25 + Progress 60 points, plus additional credit). Grade bands with group-specific cutoffs (school type x year). From 2009-10, floors can raise the grade to at most C |
| **X** | Free-lunch, ELL, special-education, Black, Hispanic, Asian and male shares; log enrollment; share in grades 3-8; DOE peer index (by type); previous year's overall-score percentile; K-8 / middle; charter; 31 community-school-district indicators; year indicators. 63 columns, 12 continuous |
| **Y** | `load()`: next year's overall score (same DBN, graded EMS report). `load_next_progress`: next year's Progress score. `load_no_grade_next`: no graded report next year |
| **n** | `load()`: 4,663 school-years (1,151 schools), 397 of them D/F. `load_ab` / `load_bc`: 6,338. `load_df`: 4,445. `load_two_score()`: 3,673. Years are spring 2007-2012, pooled |
| **Source** | NYC Open Data (Socrata): the School Progress Report summaries for 2006-07 to 2012-13 and the 2006-2012 Demographics and Accountability Snapshot. The DOE Educator Guides, full EMS result files and three school reports per year, via web.archive.org |

EMS means the DOE's elementary / middle / K-8 report type. High schools, transfer
schools, early-childhood, District 75 and YABC reports have different rules and are not
included. Years are named by their spring: 2010 = school year 2009-10.

## The rule

**Overall score.** The DOE Educator Guides (each year's "Considerations in Computing the
Overall Progress Report Grade" section) define it as:

    overall = Environment + Performance + Progress + additional credit

- Each category is a points score. Each metric is graded by its "percent of range"
  against a peer group of similar schools (75%) and the city (25%). The weights were
  67/33 in 2006-07.
- Category maxima are 15 / 25 / 60 from 2007-08 on. In 2006-07 they were 15 / 30 / 55;
  that year's file publishes the categories as fractions, which the adapter multiplies
  by these maxima.
- Additional credit "can only improve a school's overall Progress Report Score".
- Check: in every year the published category scores plus additional credit equal the
  published overall score to within rounding (max |difference| 0.05 / 0 / 0.1 / 0.2 in
  2007-08 / 2009-10 / 2010-11 / 2011-13).

**Grade.** The grade is A-F by cut scores on the overall score, per school type and year:

| Year | Elementary (A / B / C / D) | K-8 | Middle | Source |
|---|---|---|---|---|
| 2006-07 | 64.11 / 49.91 / 38.79 / 31.06 | 64.33 / 50.36 / 38.59 / 30.08 | 65.22 / 50.50 / 38.86 / 31.27 | inferred (below) |
| 2007-08 | 59.6 / 45.8 / 32.6 / 28.4 | 66.7 / 49.7 / 38.4 / 23.4 | 65.9 / 49.7 / 39.0 / 29.4 | 2007-08 guide, App. A |
| 2008-09 | 68 / 54 / 43 / 33 | same | same | 2008-09 guide, App. A |
| 2009-10 | 58.5 / 40.7 / 29.1 / 19.1 | 57.1 / 41.2 / 29.5 / 18.0 | 62.1 / 42.6 / 30.6 / 18.5 | 2009-10 guide, App. A |
| 2010-11 | 56.7 / 40.6 / 25.7 / 18.3 | 57.1 / 41.3 / 25.3 / 14.5 | 58.2 / 41.5 / 23.8 / 14.0 | school reports, p. 1 |
| 2011-12 | 64.7 / 49.8 / 35.8 / 25.6 | 64.1 / 51.8 / 37.6 / 30.0 | 65.5 / 51.7 / 36.3 / 23.2 | school reports, p. 1 |
| 2012-13 | 60.1 / 48.8 / 36.5 / 30.0 | 63.1 / 53.2 / 40.6 / 32.0 | 67.7 / 53.0 / 37.3 / 29.1 | school reports, p. 1 |

**How the cuts were set.**
- **2006-07.** Cuts were set at the 85th / 45th / 15th / 5th percentiles of the
  pre-additional-credit score within school type, and applied to the overall score
  (Rockoff and Turner; checked against NBER WP 14564).
  - The 2006-07 guide was never archived, so the cut values are inferred: each is the
    lowest published score that got the higher grade.
  - The true cut lies above the highest score with the lower grade. The gaps are:
    - elementary: A (63.99, 64.11], B (49.72, 49.91], C (38.64, 38.79], D (30.90, 31.06];
    - K-8: (63.10, 64.33], (49.76, 50.36], (38.04, 38.59], (29.35, 30.08];
    - middle: (65.21, 65.22], (50.24, 50.50], (38.81, 38.86], (29.27, 31.27].
  - Grades are a step function of the score, with no overlap, in every group.
- **2009-10 on.** The cuts follow a set distribution: 25/35/25/10/5% of schools by grade
  in 2009-10, and 25/35/30/7/3% in 2010-13.
- **2010-11 to 2012-13.** The guides do not print the cuts. They are read from the
  "Progress Report Grades" table on page 1 of the reports of 01M015 (elementary), 01M034
  (K-8) and 01M292 (middle).

**Floors (published guides; they raise a grade, never lower it).**

| Year | Floor |
|---|---|
| 2009-10 | Graded A in 2008-09: at least C. Graded B: at least D. Average math and ELA in the top 25% of the type: at least C |
| 2010-11 | Average ELA and math proficiency in the top 33%: at least C |
| 2011-12, 2012-13 | Top third citywide: at least C. Graded A the year before: at least D |

How the adapter reproduces the floors (`state_rule`):
- The prior grade comes from the Open Data file's own previous-year column.
- "Top" is the mean of ELA and math average proficiency (median proficiency in 2009-10,
  the only proficiency in that file), taken from the DOE full EMS result files.
- The mean is ranked within school type among graded schools. Ranking across all EMS
  schools reproduces the floors worse.
- No floor reaches B, so the A/B and B/C boundaries are sharp every year.

**Consequences.** The DOE Consequences page (captured 2007-10-13, in `docs/`) says:
- A schools receive financial rewards.
- D or F schools "will be subject to school improvement measures and target setting".
  Without progress they face "possible leadership change ..., restructuring, or
  closure"; the same holds for a C three years running.

Rockoff and Turner (AEJ: Economic Policy 2(4), 2010; text checked against NBER WP 14564)
add three facts:
- 7 F and 2 D schools were told in December 2007 that they would close or phase out.
- Closure and principal removal were case by case, "not based on an explicit formula".
- Grades went to principals on 2007-09-24 and to the public on 2007-11-05, before the
  January-March 2008 tests.

Winters and Cowen (2012, EEPA 34(3): 313-327) also study this design. Only their abstract
was checked: an F grade raised ELA in the first year, and other grades had no detectable
effect.

## Rule check (2026-10-28, `adapter.rule_check()`)

Graded EMS school-years (elementary, K-8, middle):

| Year | Graded | Grade = score rule | Grade = rule with floors | Raised by a floor | C/D: all | C/D: floor-eligible dropped | C/D dropped | D/F: dropped -> agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2006-07 | 1,001 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 | 0 -> 1.000 |
| 2007-08 | 1,042 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 | 0 -> 1.000 |
| 2008-09 | 1,058 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 | 0 -> 1.000 |
| 2009-10 | 1,085 | 0.887 | 0.997 | 122 | 0.896 | 0.987 | 928 | 1,039 -> 1.000 |
| 2010-11 | 1,108 | 0.995 | 1.000 | 5 | 0.995 | 1.000 | 374 | 374 -> 1.000 |
| 2011-12 | 1,132 | 0.988 | 1.000 | 14 | 0.989 | 1.000 | 378 | 496 -> 1.000 |
| 2012-13 | 1,130 | 0.983 | 0.998 | 17 | 0.985 | 0.997 | 378 | 515 -> 1.000 |

- **A/B and B/C.** `1{overall >= c}` equals the published grade for every school-year,
  2006-07 to 2012-13.
- **The remaining misses.**
  - 2009-10: 03M333 and 31R080 (K-8, prior grade B, scores 27.9 and 18.8) were
    published C. They are just outside the top-25% proxy (72nd-74th percentile). The DOE
    measure was "average Math and ELA score", which the file does not hold. A third
    school, 03M166, is inside the proxy and was published D.
  - 2012-13: 11X180 and 84Q705 sit at a tie in rounded proficiency, at the
    top-third line.
- **The loaders' D against the published grade** (`extras["eligible_published"]`):
  - `load()`: 4,661 of 4,663 (the two 2009-10 schools above).
  - `load_two_score()`: 3,671 of 3,673.
  - `load_ab`, `load_bc`, `load_df`: 1.000.
- Open Data and the DOE result files agree on every overall score and grade in
  2009-10 to 2012-13.

## Sample

| Step (spring 2007-2012) | School-years |
|---|---:|
| EMS reports of elementary, K-8 and middle schools | 6,641 |
| graded A-F with an overall score | 6,426 |
| Y observed (`load_ab`, `load_bc`) | 6,338 |
| no floor that guarantees C (`load`, `load_cd`) | 4,663 |
| no floor that guarantees C or D (`load_df`) | 4,445 |

- **The floor exclusion** depends only on the prior grade and on same-year proficiency,
  not on the overall score, so both sides of the cut are dropped alike.
- **2009-10 shrinks most.** 84% of schools were graded A in 2008-09. `load()` keeps
  146 school-years that year.
- **`load()` by year:** 989 / 1,027 / 1,053 / 146 / 721 / 727 (2007-2012).
- **D/F by year:** 120 / 55 / 7 / 41 / 91 / 83.
  - 2008-09 has almost no D/F: 84% of schools got an A.
- **`load_two_score()`** uses 2008-2012 only, because the category maxima changed after
  2006-07. It drops one school-year without a Progress score.
- **Grade shares (A / B / C / D / F):**

| Year | Shares |
|---|---|
| 2006-07 | .23 / .38 / .26 / .09 / .04 |
| 2007-08 | .38 / .41 / .15 / .05 / .02 |
| 2008-09 | .84 / .13 / .02 / .005 / .002 |
| 2009-10 to 2011-12 | .25 / .35 / .30-.35 / .04-.07 / .005-.03 |

## Fit and identification

These numbers come from an OLS of Q on X in a scratch script. `multi_rule_readiness`
was not run.

| Loader | n | Treated | R^2 | sd(T), sd(eta) | sd(eta) max/min over T deciles | Within 0.25 sd(eta) of the cut |
|---|---:|---:|---:|---:|---:|---:|
| `load` (C/D) | 4,663 | 0.915 | 0.392 | 10.4, 12.9 | 1.38 | 6.9% |
| `load_ab` | 6,338 | 0.370 | 0.390 | 10.8, 13.5 | 1.23 | 13.4% |
| `load_bc` | 6,338 | 0.701 | 0.368 | 10.3, 13.4 | 1.24 | 14.3% |
| `load_df` | 4,445 | 0.979 | 0.410 | 10.7, 12.9 | 1.34 | 3.1% |

- **R^2 for `load`:** 0.365 without the lagged percentile; 0.373 without the district
  indicators.
- **Two-score view (`load_two_score()`, C/D):**
  - R^2 is 0.512 for Q1 and 0.332 for Q2 (0.428 and 0.326 without the lag).
  - corr(T1, T2) = 0.75; corr(eta1, eta2) = 0.52.
  - The weakest direction of T is (0.59, -0.81), with sd 5.9 = 0.41 sd(eta). The
    trade-off is far better identified than in Texas (0.16).

## Suggested policy boxes

- **Single score (Q units = overall points minus the deployed cut).**
  - `load`: cuts in [-10, 15], where a D/F share runs from about 3% to 30%. The Q
    quantiles 5 / 10 / 25 / 50% are -4.5 / 1.3 / 10.1 / 21.3.
  - `load_ab`: [-15, 15].
  - `load_bc`: [-15, 15].
  - `load_df`: [-5, 15].
- **`load_two_score()` (C/D).** Units are percent of the category maximum, net of the
  category's share of the cut. The deployed rule is the half-plane λ = 0.4, φ = 0.
  - fixed weights (0.4, 0.6): φ in [-10, 15];
  - free weights: λ in [0.1, 0.9], φ in [-15, 20];
  - both cuts: Q1 in [-30, 30], Q2 in [-40, 20];
  - either cut: Q1 in [-10, 70], Q2 in [-20, 60].
  - Quantile ranges (1st-99th): Q1 runs from -9.8 to 72.0 (median 29.5); Q2 from -23.5
    to 57.0 (median 19.8).
  - D/F schools: the 5th-95th percentiles are -15.8 to 19.8 for Q1 and -30.0 to 1.0
    for Q2.
- **`load_two_score("AB")`** (n 5,348, treated 0.395):
  - φ in [-20, 20];
  - both cuts: Q1 in [-40, 40], Q2 in [-50, 30];
  - either cut: Q1 in [-20, 60], Q2 in [-30, 50].

## Getting the data

    python -m experiments.datasets.nyc_progress_reports.download          # idempotent
    python -m experiments.datasets.nyc_progress_reports.download --check  # sizes, hashes

**About the downloads.**
- All 28 files were downloaded on 2026-10-28: 25,414,714 bytes in `data/raw/`, which is
  gitignored.
- `download.py` records each file's SHA-256 and checks each file's type before keeping it.
  The Internet Archive sometimes returns a status-200 HTML "Temporarily Offline" page, so
  the script retries.
- The script was re-run for three files (one CSV, one PDF, the HTML page), and all three
  SHA-256 hashes matched.

**NYC Open Data:** `https://data.cityofnewyork.us/api/views/<id>/rows.csv?accessType=DOWNLOAD`

| File | Id | Bytes |
|---|---|---:|
| `opendata/pr_2006-07.csv` | weg5-33pj | 140,120 |
| `opendata/pr_2007-08.csv` | yayv-apxh | 219,593 |
| `opendata/pr_2008-09.csv` | vrn4-2abs | 241,001 |
| `opendata/pr_2009-10.csv` | 4n2j-ut8i | 176,168 |
| `opendata/pr_2010-11.csv` | upwt-zvh3 | 180,076 |
| `opendata/pr_2011-12.csv` | mjux-q9d4 | 200,872 |
| `opendata/pr_2012-13.csv` | cvh6-nmyi | 206,305 |
| `opendata/demographics_2006-2012.csv` | ihfw-zy9j | 1,578,546 |

**The second Open Data id for each year.** These files repeat the ones above and were not
kept:
- fzv4-jan3, dj4e-3xrn, cgz5-877h and ffnc-f3aa have the same scores, up to rounding in
  a few rows.
- yig9-9zum lacks high schools.
- xne4-4v8f is byte-identical to mjux-q9d4.
- 6je4-4x7e repeats cvh6-nmyi.
- The multi-year grade files are 5fsg-d8c9, rwa3-b3wr and 4y63-yw9e.

**Internet Archive.** The URL pattern is
`http://web.archive.org/web/<timestamp>id_/http://schools.nyc.gov/<path>`.

| File | Timestamp | Path | Bytes |
|---|---|---|---:|
| `doe/2009_2010_EMS_PR_Results_2010_12_28.xls` | 20170220055833 | NR/rdonlyres/A9535748-ABF4-4D3D-96E7-FE34FB229297/0/... | 3,912,192 |
| `doe/2010_2011_EMS_PR_Results_2013_04_30.xlsx` | 20150422025931 | NR/rdonlyres/E595859D-5AF8-4100-AB4B-6058527FA427/0/... | 2,176,500 |
| `doe/2011_2012_EMS_PR_Results_2013_02_25.xlsx` | 20150422025159 | NR/rdonlyres/A8A10B87-7BB2-4BBA-97BE-12753D7DE3E6/0/... | 4,728,832 |
| `doe/2012_2013_EMS_PR_Results_2013_12_19.xlsx` | 20150203230231 | NR/rdonlyres/1550033E-3F15-4746-BD1A-DF3364721785/0/... | 5,184,947 |
| `docs/ProgressReportEducatorGuide_EMS_091608.pdf` (2007-08) | 20090514194644 | NR/rdonlyres/DEFA8A3D-7BB8-4502-BEFC-F977FB206542/43571/... | 104,004 |
| `docs/EducatorGuide_EMS_0804099.pdf` (2008-09) | 20130311155243 | NR/rdonlyres/DF48B29F-4672-4D16-BEEA-0C7E8FC5CBD5/65798/... | 102,708 |
| `docs/EducatorGuide_EMS_2011_03_10.pdf` (2009-10) | 20120102111302 | NR/rdonlyres/4015AD0E-85EE-4FDE-B244-129284A7C36C/0/... | 266,008 |
| `docs/EducatorGuide_EMS_2011_10_03.pdf` (2010-11) | 20120102111302 | NR/rdonlyres/A82481C5-A351-47BA-BF8C-9F353E9CFB22/0/... | 285,036 |
| `docs/EducatorGuide_EMS_2013_01_04.pdf` (2011-12) | 20130311152933 | NR/rdonlyres/99B4C4A8-D8EE-482D-BDEA-72B2BB001518/0/... | 283,297 |
| `docs/EducatorGuide_EMS_20131118.pdf` (2012-13) | 20150906101531 | NR/rdonlyres/7B6EEB8B-D0E8-432B-9BF6-3E374958EA70/0/... | 566,831 |
| `docs/consequences_2007-10-13.html` | 20071013200848 | Accountability/ProgressReports/Consequences/default.htm | 19,099 |
| `docs/school_reports/Progress_Report_<year>_EMS_<M015, M034, M292>.pdf` (9 files) | see `download.py` | OA/SchoolReports/<2010-11, 2011-12, 2012-13>/... | 340,468-678,032 each |

**What is missing.**
- **The 2006-07 EMS Educator Guide** (`.../DF48B29F-4672-4D16-BEEA-0C7E8FC5CBD5/27499/EducatorGuide_EMS.pdf`)
  has no archived copy; the only capture is a 2017 404.
- **No 2006-07 single-school reports** were found in the archive.
- **No source was blocked.** No 403s, challenges or forms appeared. The live DOE pages
  now redirect (301), so the archive was used.

**Cache.** The adapter caches `data/processed/nyc_progress_reports_panel.csv.gz` (7,842
EMS school-years, 2006-07 to 2012-13).
- The file is written to a temporary file and renamed, so concurrent loads are safe.
- Delete it after editing `adapter.py`.

## Outcome

- **`load()`:** Y has mean 58.0 and sd 19.3.
  - corr(Y, Q) is 0.18 pooled and 0.27-0.42 within a year. Overall scores are not very
    persistent from year to year.
  - The score level moves a lot by year (mean overall score 54, 58, 80, 47, 46, 55 in
    2007-2012). The year indicators absorb only the level.
- **`load_next_progress`:** Y has mean 33.0 and sd 12.1.
- **`load_no_grade_next`** (n 4,746): the share with no graded report next year is 0.5%
  for C-or-better schools and 13.1% for D/F. Within 5 points of the cut it is 1.8% for
  C-or-better and 7.8% for D/F.

## Caveats

- **Selective attrition at the consequential boundary.** Of 456 D/F school-years, 60
  (13%) have no next-year grade, against 0.5% of the others. These are mostly schools
  with a report but no grade (phase-out, too few students). So `load()`'s Y is missing
  selectively below the cut. Use `load_no_grade_next` or bound the effect.
- **Floors.** The C/D and D/F loaders drop floor-eligible school-years. That costs 26%
  of the sample (most of 2009-10).
  - The top-third proxy uses rounded proficiency, so a few ties remain (2 in 2012-13,
    which is not in the loaders).
  - 2009-10 uses median, not average, proficiency, which leaves the 2 misses above.
- **Rule changes across years.**
  - Category maxima: 15 / 30 / 55 in 2006-07.
  - Horizon weights: 67/33 in 2006-07.
  - Metrics: growth percentiles from 2009-10, and new state cut scores in 2010.
  - Cut-setting: percentiles in 2006-07, a set distribution from 2009-10.
  - 2008-09 is nearly all A.
  - Q is centered per type and year, but the scale of Q and Y still changes.
- **Cuts placed on the data, no bunching.**
  - In the percentile years a school sits exactly at the cut: Q = 0 for 11 school-years
    in `load` and 27 in `load_ab`.
  - Cuts were set after the scores were computed, and grades were unknown until release,
    so schools could not sort around them.
  - Counts within 1 point below and above the cut: 43 / 41 (C/D), 116 / 126 (A/B),
    123 / 119 (B/C), 17 / 24 (D/F). The few local asymmetries look like chance; for
    example, A/B in 2009-10 has 11 vs 34.
- **2006-07 cuts are inferred.** Any value inside the gaps above gives the same D. Q
  uses the upper end of each gap.
- **Covariates.**
  - Demographics are the October register of the rating year, not test-based. They are
    missing for 2-8% of school-years (median-filled with an indicator).
  - Free lunch through 2008-09 becomes free or reduced-price lunch from 2009-10; the
    year indicators absorb the switch.
  - The middle-school peer index is the students' earlier 4th-grade proficiency.
  - The lagged percentile is missing in 2006-07 (indicator).
- **Units.**
  - Schools repeat across up to six years, so cluster by DBN. District effects are
    modest (R^2 0.373 -> 0.392).
  - 12 school-years change type the next year (`extras["y_level_changed"]`).
  - Charter schools (district 84) are included, with an indicator. In 2006-07 their
    Environment score was attendance only.
