# Wisconsin school report cards: a weighted sum whose weights depend on a covariate, then star bands

| Field | Value |
|---|---|
| **Q** | `load()`: x - (c - 0.05), where x is the unrounded overall score and c the year's cut score. `load_two_score()`: (Achievement A, Growth G), shape (n, 2), in score points (0-100, published to a tenth) |
| **Threshold** | 0, on Q. On x: c - 0.05, with c = 70 for 4 stars in 2020-21 to 2023-24 and 73 in 2015-16 to 2018-19 (table below). `load_two_score()`: per-unit cutoff c_i on the normalized index (below) |
| **Treatment** | `load()`: `1{Q >= 0}`, a rating of Exceeds Expectations (4 stars) or better rather than Meets Expectations or below. `load_stars5`, `load_stars3`, `load_stars2`: the other boundaries, D = the higher rating. `load_two_score()`: `1{w_i Q1 + (1 - w_i) Q2 >= c_i}`. D is the published star rating; no formal consequence attaches to it in these years (see "What D is") |
| **Rule class** | a weighted sum of four scores whose achievement/growth split is a piecewise-linear function of a covariate (% economically disadvantaged), then year-specific cut scores. In (A, G): a half-plane with a school-specific slope and a school-specific intercept |
| **X** | % economically disadvantaged (the weighting covariate); % Black, Hispanic, Asian, American Indian, two or more races; % with disabilities; % English learners; % open enrollment; log enrollment and its change from the prior year; log district enrollment; lowest and highest grade; charter, virtual, independent charter; school type; locale; CESA and year indicators; 3 missing-data flags (45 columns; 12 continuous) |
| **Y** | the next report card year's published overall score (2018-19 -> 2020-21; no report cards for 2019-20) |
| **n** | `load()` and the boundary loaders: 15,161 school-years (2,068 schools, 463 districts), rating years 2015-16 to 2018-19 and 2020-21 to 2023-24. `load_two_score()`: 13,710 (the school-years with a Growth score) |
| **Source** | Wisconsin DPI School and District Report Card Data Download Files 2015-16 to 2024-25, WISEdash certified enrollment files 2014-15 to 2024-25, and DPI's Report Card Technical Guides 2015-16 to 2025-26, all static files on dpi.wi.gov (downloaded 2026-10-28) |

## The rule

Wis. Stat. 115.385 requires an annual report card with an overall score and five
performance categories, and requires Achievement and Growth to be weighted by the share of
economically disadvantaged (ED) students. The Technical Guide ("Building the Overall Score")
gives the computation, unchanged in form from 2015-16 to 2025-26.

The four priority-area scores are each on 0-100 and published to a tenth:
- Achievement (A);
- Growth (G), from value-added;
- C: Closing Gaps (2015-16 to 2018-19), replaced by Target Group Outcomes (2020-21 on);
- On-Track to Graduation (O). Until 2018-19 it was named On-Track and Postsecondary Readiness.

The steps:

1. **Achievement multiplier.** With e the school's ED share (the fraction published in the
   file, three decimals),

       a(e) = 0.9                      if e <= 0.05
            = 1 - (4/3 e + 1/30)       if 0.05 < e < 0.65    (= 0.9 - 0.8 (e - 0.05) / 0.6)
            = 0.1                      if e >= 0.65

   - When G is present, m_A = a(e) and m_G = 1 - a(e).
   - When G is absent (most high schools before 2018-19), A gets m_A = 0.5 (1 from 2020-21;
     the two are equivalent after normalization) and m_G = 0.
   - m_C = 0.5 when C is present, else 0.
2. **Weights.** Let f = 3 if the school has a 3rd-grade ELA or an 8th-grade math On-Track
   component, else f = 3.2. Then w_K = m_K / (m_A + m_G + m_C) * f / 4 for K = A, G, C, and
   w_O = 1/4 (f = 3) or 1/5 (f = 3.2). The weights sum to 1.
   - Example: a school with all four areas and a 3rd/8th component gets w_A + w_G = 1/2,
     w_C = w_O = 1/4. Its A/G split is (a(e), 1 - a(e)): 0.45/0.05 at e <= 5% and
     0.05/0.45 at e >= 65%.
3. **Overall score.** x = w_A A + w_G G + w_C C + w_O O + ded.
   - ded <= 0 is the Student Engagement Indicators deduction: -5 each for missing the
     absenteeism and dropout goals, and in 2015-16 also -5 or -10 for test participation.
     Only 2015-16 to 2018-19 have it.
   - The published overall score is x rounded half up to a tenth.
4. **Rating.** A cut on the published score:

| Report card year | 5 stars (Significantly Exceeds) | 4 (Exceeds) | 3 (Meets) | 2 (Meets Few) | 1 (Fails to Meet) | Source |
|---|---:|---:|---:|---:|---|---|
| 2015-16 to 2018-19 | 83 | 73 | 63 | 53 | below 53 | Technical Guides 2015-16 to 2018-19 |
| 2019-20 | no report cards (COVID-19) | | | | | report card portal |
| 2020-21 to 2023-24 | 83 | 70 | 58 | 48 | below 48 | `whats_new_2020-21.pdf`; Technical Guides 2020-21 to 2023-24 |
| 2024-25 (and 2025-26) | 84 | 71 | 60 | 49 | below 49 | `standard_setting_report_2025.pdf`, Table 4 |

- **Cut scores.** The 2020-21 cuts were set by equipercentile linking when Target Group
  Outcomes replaced Closing Gaps and the deductions were dropped.
- **The 2024-25 cuts** come from the 2025 standard setting (Center for Assessment panel; the
  panel's medians 84/72/61/50 were amended to 84/71/60/49). The 2024-25 Technical Guide and
  "What's New" show the table only as an image. The data confirm it: published scores and
  ratings never overlap these cuts.
- **The ED share.** The guides say it is "rounded to the nearest whole percentage point".
  The data say otherwise. With the three-decimal fraction, the formula reproduces every
  published weight and score from 2017-18 on. With whole percents it reproduces only
  81-88% of the scores.
- **The weights are published** in the file from 2017-18 on ("Score weighting ..."). They
  are not in the 2015-16 and 2016-17 files, but the 2015-16 and 2016-17 guides give the
  same formula, and "What's New for 2017-18" says no calculations changed.

**The two-score form.** Let s_i = w_A + w_G (0.5, 0.533, 0.75 or 0.8, depending on which
areas are present) and offset_i = w_C C + w_O O + ded. The rule "published overall >= c" is
then

    a(e_i) A + (1 - a(e_i)) G  >=  c_i := (c - 0.05 - offset_i) / s_i.

So in (A, G) the deployed rule is a half-plane. Its slope is set by the school's ED share,
and its intercept by the school's other two priority areas and deductions.

## What D is

The rating is public and must be published:
- by January 31 each school board must publish a Class 1 notice with every school's rating;
- every public school must give each parent the report card and a list of educational
  options;
- schools link the report card from their home page.

(Source: DPI, https://dpi.wi.gov/accountability/report-cards/notification-requirements.)

The one formal consequence of a school's rating is the **Opportunity Schools and Partnership
Program** (OSPP), created by 2015 Wisconsin Act 55 (Wis. Stat. 119.9002, 119.9004).
- **Milwaukee.** An MPS school is eligible for transfer to a county-appointed commissioner
  if MPS as a district is rated Fails to Meet in the most recent report card, and the
  school is rated Fails to Meet (or is in a vacant or underutilized building).
- **Other districts.** Schools are eligible only in a district with more than 15,000
  students that was rated Fails to Meet in the two most recent report cards and received
  intradistrict transfer aid.
- **It never applied in these years.** The Legislative Audit Bureau (reports 21-8 and
  23-7) found that no school was ever transferred. `adapter.district_ratings()` shows that
  MPS was rated Meets Few or Meets in every year from 2015-16 to 2024-25 (55.3 to 59.5),
  never Fails to Meet. Racine Unified was Fails to Meet in 2015-16 only.
- So the 1-star rating of the 253 MPS school-years rated Fails to Meet in the sample
  triggered nothing.

**Federal ESSA designations** (CSI, TSI, ATSI) come from a separate federal index, not from
the report card score (DPI, https://dpi.wi.gov/accountability/federal).

**D is therefore a publicly reported rating with no formal consequence attached** in the
years used. `load()` uses the 4-star boundary (Exceeds vs Meets), which has by far the most
school-years near it. `load_stars2()` is the boundary the OSPP statute refers to.

| Boundary (loader) | Cut on x | Treated share | School-years with \|Q\| < 1 | \|Q\| < 2 |
|---|---|---:|---:|---:|
| 5 stars (`load_stars5`) | 82.95 | 0.142 | 647 | 1,259 |
| 4 stars (`load`, `load_stars4`) | 72.95 / 69.95 | 0.504 | 1,172 | 2,312 |
| 3 stars (`load_stars3`) | 62.95 / 57.95 | 0.848 | 598 | 1,223 |
| 2 stars (`load_stars2`) | 52.95 / 47.95 | 0.963 | 150 | 332 |

The cut on x is given as 2015-16 to 2018-19 / 2020-21 to 2023-24. Q counts pool all years.

## Getting the data

    python -m experiments.datasets.wi_report_cards.download          # idempotent
    python -m experiments.datasets.wi_report_cards.download --check  # sizes, hashes

- The 44 files total 67.7 MB and go to `data/raw/` (gitignored).
- `download.py` records each file's SHA-256 from 2026-10-28 and reports any difference;
  DPI re-uploaded the report card files in March 2026.
- Every request was a plain script request (curl, or `urllib` with its standard user agent).
  None was refused.

| Files | Bytes | Use |
|---|---:|---|
| `https://dpi.wi.gov/sites/default/files/imce/accountability/xlsx/YYYY-YY_school_reportcard_data.xlsx`, 2015-16 to 2024-25 (9 files, no 2019-20) | 14,440,085 | scores, weights, ratings, demographics, school type, locale, CESA |
| `https://dpi.wi.gov/sites/default/files/imce/accountability/xlsx/YYYY-YY_district_reportcard_data.xlsx`, same years (9 files) | 2,644,203 | district ratings (`district_ratings()`, MPS) |
| `https://dpi.wi.gov/sites/default/files/wise/downloads/enrollment_certified_YYYY-YY.zip`, 2014-15 to 2024-25 (11 files) | 8,309,616 | prior-year enrollment, county |
| `https://dpi.wi.gov/media/{4773, 7165, 38758, 38744, 38801, 38824, 18694, 22118, 26129, 55851}/download?inline` | 17,324,857 | Technical Guides 2015-16, 2016-17, 2017-18, 2018-19, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26 |
| `https://dpi.wi.gov/media/{38755, 38791, 24748}/download?inline` | 1,503,651 | "What's New" 2017-18, 2020-21, 2024-25 |
| `https://dpi.wi.gov/media/24749/download?inline` | 637,387 | 2024-25 Scoring Updates |
| `https://dpi.wi.gov/media/24721/download?inline` | 22,845,653 | 2025 standard-setting report (2024-25 cut scores) |

**Where the links are.**
- The report card data files are linked from the report card portal's "Report Card Data
  Download Files" page,
  https://apps6.dpi.wi.gov/reportcards/HistoricalReports?SelectedSchoolYear=2025&IsPublic=True.
- The guides are linked from https://dpi.wi.gov/accountability/resources (one page per
  year: `/2024-45` (sic), `/2023-24`, ..., `/2018-19`, `/archived`) and from
  https://dpi.wi.gov/accountability/report-cards/standard-setting.
- The enrollment files are listed at https://dpi.wi.gov/wisedash/public/download-files.

**Not found where the brief pointed.** WISEdash's download listing no longer has a "Report
Cards" type:
- https://dpi.wi.gov/wisedash/download-files now redirects (301) to `/wisedash/public/download-files`;
- `.../wisedash/download-files/type?field_wisedash_upload_type_value=Report%20Cards`
  returns 404;
- the same filter on `/wisedash/public/download-files` lists no files (2026-10-28).

The portal files above are the report card data.

**Reading.**
- The adapter reads the `.xlsx` files with `fl_school_grades/_spreadsheet.py`, a
  standard-library reader (the environment has no `openpyxl`).
- It reads the enrollment CSVs from inside the zip files.

**Cache.**
- `data/processed/wi_report_cards_panel.csv.gz` (1.9 MB) holds every report card row with a
  district code, 2015-16 to 2024-25. Private Choice-school rows have no district code and
  are dropped.
- It is written to a temporary file in the same folder and moved into place with
  `os.replace`.
- Delete it after changing `_read_rc`, `state_rule` or `_build`.

## Rule check (2026-10-28)

`adapter.rule_check()` takes public schools with a published score and rating. It compares
the overall score computed from the published A, G, C, O, deductions and ED share (rounded
half up) with the published score:

| Report card year | Scored public schools | With Growth | Computed = published score | Max \|diff\| | Exact .x5 ties (rounded up) | Weights = published (3 dp) | Stars from x = published |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015-16 | 1,930 | 1,505 | 0.9922 (15 off) | 0.1 | 25 | not published | 1.000 |
| 2016-17 | 1,920 | 1,502 | 0.9938 (12 off) | 0.1 | 22 | not published | 1.000 |
| 2017-18 | 1,910 | 1,499 | 1.000 | 0 | 29 | 1.000 | 1.000 |
| 2018-19 | 1,933 | 1,887 | 1.000 | 0 | 23 | 1.000 | 1.000 |
| 2020-21 | 1,902 | 1,847 | 1.000 | 0 | 22 | 1.000 | 1.000 |
| 2021-22 | 1,920 | 1,862 | 1.000 | 0 | 36 | 1.000 | 1.000 |
| 2022-23 | 1,940 | 1,893 | 1.000 | 0 | 38 | 1.000 | 1.000 |
| 2023-24 | 1,939 | 1,896 | 1.000 | 0 | 26 | 1.000 | 1.000 |
| 2024-25 | 1,920 | 1,875 | 1.000 | 0 | 25 | 1.000 | 1.000 |

- **From 2017-18 on**, the computed score equals the published one for all 13,464 scored
  public schools.
  - One 2017-18 school has no ED share; its weight comes from the published weights
    (`weights_source` = "published").
  - The rounding is half up: every exact tie is published rounded up.
- **2015-16 and 2016-17.** 27 of 3,850 computed scores differ from the published ones, all
  by 0.1. None of the 27 crosses a cutoff, so the stars implied by x agree for every school.
- **Published rating against the cut scores** (applied to the published score): 1.000 in
  every year.
- **The loaders' D against the published rating** (`extras["eligible_published"]`): 1.000
  for all four boundaries, for both `load()` and `load_two_score()`. On the two-score
  sample, `load_two_score()`'s D equals `load()`'s D for every school-year.

**Sample.**

| Step (rating years 2015-16 to 2018-19, 2020-21 to 2023-24) | School-years left |
|---|---:|
| public school report cards | 16,849 |
| published overall score and one of the five ratings (drops 1,454 alternate-accountability ratings and 1 blank) | 15,394 |
| Y observed | 15,161 |
| `load_two_score()`: with a Growth score | 13,710 |

Of the 233 without Y:
- 154 have no row the next report card year (closure, merger or new school code);
- 79 got an alternate rating the next year.

## Readiness-style statistics (2026-10-28)

These were computed with the helpers of `experiments/scripts/multi_rule_readiness.py`
(`_ols`, `_sd_ratio`) in a scratch script. The script's `main()` was not run, so the shared
summary file was not written.

| Quantity | `load()` (4 stars) |
|---|---:|
| n, p (continuous X) | 15,161, 45 (12) |
| R^2 of Q on X | 0.452 |
| sd(T), sd(eta) | 7.56, 8.32 |
| sd(eta) max/min across T deciles | 1.78 |
| Treated share | 0.504 |
| Within 0.25 sd(eta) of the cutoff | 15.8% |
| Agreement with the published rating | 1.000 |

- Adding 463 district indicators raises R^2 to 0.53 (sd(eta) 7.68).
- Adding the previous report card year's score (`extras["q_lag"]`; n = 13,070) raises R^2
  to 0.75 (sd(eta) 5.50). It is not in X, because it depends on the previous rating.
- Other boundaries: R^2 0.45-0.47, near-cutoff shares 8.6% (5 stars), 8.4% (3 stars) and
  2.3% (2 stars).

**Two-score view: `load_two_score()`**, Q = (A, G):

| Quantity | Achievement | Growth |
|---|---:|---:|
| R^2 | 0.809 | 0.101 |
| sd(T), sd(eta) | 16.38, 7.96 | 4.76, 14.18 |
| sd(eta) max/min across T deciles | 1.24 | 1.16 |

- corr(T1, T2) = 0.783 and corr(eta1, eta2) = 0.328.
- The weakest direction of T is (0.23, -0.97). Its sd is 0.25 of sd(eta).
- 10.7% of school-years lie within 0.25 sd(eta) of their own deployed boundary
  (row-wise index (w_i, 1 - w_i) . Q - c_i).
- Achievement is well predicted by demographics; growth is not.

**What `extras["weights"]` and `extras["cutoff"]` hold.** Here they are a fixed *reference*
rule: the median w_i (0.42, so (0.42, 0.58)) and the median c_i (66.25). That keeps scripts
that expect a length-2 weight vector and a scalar cutoff (`multi_rule_readiness`,
`two_score_apply`) working. The deployed rule is in:
- `extras["unit_weights"]`: (n, 2), rows (w_i, 1 - w_i);
- `extras["unit_cutoff"]`: (n,), c_i;
- `extras["scale"]` (s_i) and `extras["offset"]` (offset_i).

`multi_rule_readiness`'s near-boundary share uses the reference rule: 14.8%, against 10.7%
for the deployed per-unit rule.

**The deployed parameters.**
- w_i (weight on Achievement): quantiles 0 / 5 / 25 / 50 / 75 / 95 / 100% =
  0.10 / 0.10 / 0.22 / 0.42 / 0.61 / 0.81 / 0.90. 16.2% of school-years sit at 0.1 (ED >= 65%)
  and 1.1% at 0.9 (ED <= 5%).
- c_i: quantiles 1 / 5 / 25 / 50 / 75 / 95 / 99% = 47.2 / 52.4 / 61.6 / 66.2 / 71.8 / 83.1 /
  94.2 (range 40.4-122.2). A c_i above 100 means the school cannot reach 4 stars, given its
  other areas.
- s_i: 0.5 for 9,876 school-years, 0.533 for 2,007, 0.75 for 1,601, 0.8 for 226.
- Score ranges, 1st-99th percentile: A 10.9-97.0, G 29.0-98.3. The deployed index
  w_i A + (1 - w_i) G has a 5th-95th percentile range of 45.6-88.6.

**Suggested policy box** for rules λ A + (1 - λ) G >= c on this sample:
- λ in [0.1, 0.9], the statutory range of the deployed weights;
- c in [50, 85], which covers the 5th-95th percentiles of the deployed c_i (52.4-83.1);
- for a wider search, c in [45, 90].

A fixed (λ, c) rule ignores the other two priority areas, which the deployed rule nets out
through c_i.

## Outcome

Y is the next report card year's published overall score, matched by district and school
code. Its mean is 71.0, its sd 11.4, and corr(Y, Q) = 0.86.

| Rating year | Outcome year | n |
|---|---|---:|
| 2015-16 | 2016-17 | 1,898 |
| 2016-17 | 2017-18 | 1,895 |
| 2017-18 | 2018-19 | 1,888 |
| 2018-19 | 2020-21 (no 2019-20 report cards; new C area and cut scores) | 1,875 |
| 2020-21 | 2021-22 | 1,883 |
| 2021-22 | 2022-23 | 1,886 |
| 2022-23 | 2023-24 (new state test cut scores; one-year achievement scale adjustment) | 1,922 |
| 2023-24 | 2024-25 (achievement on two years; A and TGO normed to Growth) | 1,914 |

**Timing.** DPI releases a year's report cards in November of the following school year;
the 2017-18 cards came out in November 2018. The rating is therefore public before the
spring tests that enter the outcome year's card.

## Caveats

- **The outcome overlaps the running variable by construction.**
  - Achievement averages up to three years of tests, so the year t+1 score reuses years t
    and t-1. The 2020-21 cards used 2017-18, 2018-19 and 2020-21.
  - On-Track uses the prior year's attendance, graduation and chronic absenteeism, so part
    of Y is fixed before the rating is released.
  - corr(Y, Q) = 0.86. A next-year score is a weak test of a response to the rating.
- **The rule changes across years.**
  - Closing Gaps (2015-16 to 2018-19) is replaced by Target Group Outcomes (2020-21 on).
  - The deductions end after 2018-19.
  - The cut scores change in 2020-21 and 2024-25.
  - Achievement scaling changes in 2023-24 and 2024-25.
  - The weight formula a(e) and its 5% / 65% kinks are the same in every year.
  - Q is centred at each year's own cutoff. The year indicators absorb level shifts only.
- **Missing priority areas.**
  - Most high schools had no Growth score before 2018-19 (about 22% of scored schools in
    2015-16 to 2017-18). Their weights do not depend on e, and they are left out of
    `load_two_score()`.
  - Target Group Outcomes is missing for about 29% in 2020-21 and 2021-22.
- **Bunching.** None is visible. The counts of x in 0.1-point bins on either side of each
  cutoff are flat. For example, 600 below and 572 above within 1 point of the 4-star cut.
  DPI computes the scores centrally from multi-year data.
- **Covariates.**
  - X is from the report card year's third-Friday-of-September enrollment, before the
    spring tests.
  - % economically disadvantaged both enters X and sets the weights, as the design intends.
  - The report card's enrollment equals the WISEdash count for 99.8% of school-years from
    2017-18 on, but only 48-69% in 2015-16 and 2016-17, presumably a different count date.
  - X has 2 school-years with no demographics, 32 independent-charter school-years with no
    district fields, and 31 with no prior-year enrollment. Each block is median-imputed
    with one flag.
  - Locale is filled within school across years.
- **Units.**
  - Schools repeat for up to 8 rating years, so cluster by school, or by district (district
    indicators raise R^2 from 0.45 to 0.53).
  - eta is heteroskedastic (sd ratio 1.78).
- **Excluded.**
  - Private Choice-school report cards: different population, enrollment only of Choice
    students, no district code.
  - Alternate-accountability schools, which have no score.
  - 2024-25 serves only as an outcome; its rating has no outcome year yet.
- **The two-score rule needs the loaded rows.** The treatment rule closes over per-unit
  arrays, so it raises an error if Q does not have the loaded shape. Subsample through
  `extras` (for example `unit_weights[idx]`, `unit_cutoff[idx]`).
