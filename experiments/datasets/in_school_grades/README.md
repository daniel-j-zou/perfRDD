# Indiana state A-F school grades: fixed-weight sums of domain points with grade bands

| Field | Value |
|---|---|
| **Q** | `load()`: overall points - (c - 0.05), with c the cutoff of the boundary. `load_two_score()`: (Performance 3-8, Growth 4-8) for schools graded on those two domains alone |
| **Threshold** | 0, on Q. On the overall points: c - 0.05 = 59.95 for the D/F boundary (89.95, 79.95, 69.95 for the others). The same for every school and both years |
| **Treatment** | `load()`: `1{Q >= 0}`, a D or better rather than an F. So the consequential grade (F) is D = 0. `load_ab`, `load_bc`, `load_cd`: the other boundaries, D = the higher grade |
| **Rule class** | a weighted sum of domain points with weights fixed by grade span (3-8: 0.5 / 0.5; 9-12: 0.2 / 0.2 / 0.6; combined schools by enrollment share), then grade bands (90 / 80 / 70 / 60) |
| **X** | % free/reduced-price meals, % Black, Hispanic, Asian, multiracial, % English learners, % special education, log enrollment, lowest and highest grade enrolled, corporation log enrollment and % free/reduced meals, charter, locale (city, suburb, town, missing), domain set (high school, combined, other). 20 columns, 10 continuous |
| **Y** | `load()`: the school's 2017-18 overall points. Pooled loaders: the next spring's mean E/LA and math pass rate, grades 3-8 |
| **n** | `load()`: 1,716 public schools (2016-17 grades). `load_two_score()`: 1,323. `load_pooled()` and `load_two_score_pooled()`: 2,590 school-years (1,334 schools; 2016-17 and 2017-18 grades) |
| **Source** | IDOE state A-F school grade results 2016-17 and 2017-18 ("School Domain Scores"), IDOE enrollment files, 511 IAC 6.2-10; all static files on in.gov (rule text from Cornell LII) |

Rating year y means school year (y-1)-y. "2017" is 2016-17.

## The rule

511 IAC 6.2-10 was filed 2015-06-15, took effect 2016-03-01 and was readopted unchanged in
2021. It governed the 2015-16 to 2018-19 grades. Section 3 sets the overall points. IDOE's
presentation of the model (`docs/f-accountability-presentation.pdf`) gives the same
formulas.

| Grade span (enrolled share) | Performance | Growth | Multiple Measures |
|---|---|---|---|
| grades 3-8 (EW3-8) | 50% (Performance 3-8) | 50% (Growth 4-8) | - |
| grades 9-12 (EW9-12) | 20% (Performance 10) | 20% (Growth 10-12) | 60% (graduation 30%, college and career readiness 30%) |

- **Combined schools.** A school with both spans weights the rows by EW3-8 and EW9-12, its
  share of students enrolled in grades 3-8 and 9-12. The same school year's enrollment file
  reproduces the published 3-8 weight within 0.005 for 85% of combined schools (median
  difference 0.0006).
- **Missing domains.** A domain without enough students (30 for performance, 40 for growth,
  10 for graduation) drops out, and the rule moves its weight:
  - 3-8 without growth: Performance 100%.
  - 9-12 without growth: Performance 40%, Multiple Measures 60%.
  - 9-12 without Multiple Measures: Performance 50%, Growth 50%.
- **New schools.** A school open three years or less may elect to be graded on growth
  alone (section 10). It then has a single domain with weight 1.
- **Feeder schools** (K-2) take the performance of their receiving schools (section 7).
  They have no row in the domain sheets.
- **Domain points.**
  - Performance: the E/LA and math pass rates, averaged, each multiplied by the
    participation rate when it is below 95%.
  - Growth: the average of a values-table score over the top 75% and bottom 25% of
    students. It runs from 0 to 150, so overall points can exceed 100. 47% of grades 3-8
    schools have growth above 100.
- **Grade bands.** 90.0-100.0 = A, 80.0-89.9 = B, 70.0-79.9 = C, 60.0-69.9 = D, below = F
  (section 3(b)), with overall points published to one decimal. The data confirm that the
  published overall is round-half-up of the weighted sum.
- **The A rule.** Section 3(e)-(f): a school that misses the subgroup gap rule gets a B
  instead of an A. It never shows in the data: every published grade equals the band of
  the published overall points.

**Consequences.**
- An F counts toward state intervention under IC 20-31-9 (consecutive years in the lowest
  category). Chalkbeat reported in 2016-2017 that a first F in 2015-16 or later starts a
  four-year track. 50 of the 91 F schools in `load()` also had an F in 2015-16.
- An F also opens the Choice Scholarship (voucher) "F" public school track for the
  school's attendance area in the next school year (IDOE, `8-F-Schools-for-Choice-Eligibility.pdf`).
- The other boundaries carry no formal sanction, so `load()` uses D/F.

**Timing.**
- The State Board placed the 2016-17 grades on 2017-10-04 and the 2017-18 grades on
  2018-11-14 (SBOE memos). Both dates fall before the next spring's tests.
- The 2016-17 domain sheet counts (620 A, 688 B, 454 C, 190 D, 124 F, 28 No Grade) equal
  the October 2017 release.

| Rating year | Published | Points? | Notes |
|---|---|---|---|
| 2015-16 | letter grade | no | first year of 511 IAC 6.2-10; used only as `grade_prev` |
| 2016-17 | letter grade + domain points + weights | yes | `load()` |
| 2017-18 | letter grade + domain points + weights | yes | Y of `load()`; Q of the pooled loaders. 49 grades were "Appeal Pending" (no points) |
| 2018-19 | hold-harmless letter grade | no | Public Law 2-2020 (signed 2020-02-12): the better of the 2017-18 grade and the computed 2018-19 grade (SBOE memo of 2020-03-04) |
| 2019-20 | hold-harmless letter grade | no | no spring 2020 tests |
| 2020-21 on | none | - | grading stopped; a new model (511 IAC 6.2-12, cut points 85 / 75 / 65 / 55, by grade span) was approved 2026-03-04. No results are published yet, and the model is not comparable |

## Getting the data

    python -m experiments.datasets.in_school_grades.download          # idempotent
    python -m experiments.datasets.in_school_grades.download --check  # sizes, hashes

The 24 files total 16.6 MB and go to `data/raw/` (gitignored). All were downloaded on
2026-10-28 with Python's default `urllib` headers. `download.py` records each file's
SHA-256 and reports a difference.

| File | Bytes | Use |
|---|---:|---|
| https://www.in.gov/doe/files/af-2017-school-grade-placement-03192018.xlsx | 240,281 | 2016-17 grades (after appeals) and domain points |
| https://www.in.gov/doe/files/2018-state-school-grade-results.xlsx | 243,865 | 2017-18 grades and domain points |
| https://www.in.gov/doe/files/final-2016-af-school-grades-20170519.xlsx | 102,916 | 2015-16 grades |
| https://www.in.gov/doe/files/sboefinal-18-19-gradeshh22620.xlsx | 101,436 | 2018-19 hold-harmless grades |
| https://www.in.gov/doe/files/2020-af-placements.xlsx | 107,495 | 2019-20 hold-harmless grades |
| https://www.in.gov/doe/files/af-2017-school-grade-summary.xlsx | 20,088 | statewide counts (check) |
| https://www.in.gov/doe/files/2018-state-school-grade-summary.xlsx | 23,249 | statewide counts (check) |
| https://www.in.gov/doe/it/data-center-and-reports/Data-Files/school-enrollment-ethnicity-and-free-reduced-price-meal-status-2006-27.xlsx | 3,563,050 | enrollment, race/ethnicity, free/reduced meals |
| https://www.in.gov/doe/it/data-center-and-reports/Data-Files/school-enrollment-ell-special-education-2006-27.xlsx | 3,279,684 | English learners, special education |
| https://www.in.gov/doe/it/data-center-and-reports/Data-Files/school-enrollment-grade-2006-27.xlsx | 4,239,547 | grade span, 3-8 vs 9-12 shares |
| https://www.in.gov/doe/it/data-center-and-reports/Data-Files/2026-2027-school-directory-2026-10-22.xlsx | 465,823 | locale, county, corporation type (only these columns are read) |
| https://www.in.gov/doe/files/ISTEP-2018-Grade3-8-Final-School.xlsx | 604,053 | pooled Y for 2016-17 |
| https://www.in.gov/doe/files/ilearn-2019-grade3-8-final-school.xlsx | 1,820,169 | pooled Y for 2017-18 |
| https://www.in.gov/doe/files/f-accountability-presentation.pdf | 1,091,745 | IDOE's description of the model |
| https://www.in.gov/sboe/files/Memos_AF-Placement.pdf | 231,542 | 2017-18 placement, 45 audits, 49 appeals |
| https://www.in.gov/sboe/files/SBOE_17-18-AF-timeline-2.pdf | 41,022 | 2017-18 timeline |
| https://www.in.gov/sboe/files/2019-AF-placement_SBOE-2.pdf | 130,192 | 2018-19 hold harmless (PL 2-2020) |
| https://www.in.gov/doe/files/8-F-Schools-for-Choice-Eligibility.pdf | 90,432 | voucher "F" track |
| https://www.law.cornell.edu/regulations/indiana/511-IAC-6.2-10-3 (also -4, -5, -6, -7, -10) | 28,043-35,699 each | rule text |

- **Where the links are.** The grade files are linked from IDOE's "Data Reports Archive"
  page (https://www.in.gov/doe/it/data-center-and-reports/data-reports-archive/). The
  enrollment files and the directory are linked from
  https://www.in.gov/doe/it/data-center-and-reports/.
- **Renamed every year.** IDOE renames the data-center files each year (`-2006-27` becomes
  `-2006-28`). If a URL returns 404, take the current link from that page. The 2017 and
  2018 sheets are what the adapter reads.
- **Enrollment sheets.** The enrollment sheet named `2017` is school year 2016-17: its 3-8
  shares match the 2016-17 combined-school weights (85% within 0.005), while the `2016` and
  `2018` sheets match 17-18%.
- **Reading.** The adapter reads the `.xlsx` files with `fl_school_grades/_spreadsheet.py`
  (standard library; no `openpyxl`).
- **Cache.** The adapter caches `data/processed/in_school_grades_panel.csv.gz` (4,259
  school-years, 2016-17 and 2017-18) with an atomic write. Delete it after changing
  `adapter.py`.

**Not available.**
- No statewide file with the points behind the 2015-16 or 2018-19 grades exists on in.gov.
  The Wayback Machine's copies of the old doe.in.gov data-reports page (November 2016 to
  February 2019) list only letter-grade files for 2015-16. The 2018-19 grades (March 2020)
  appear only as the hold-harmless file.
- For 2018-19 the SBOE memo says schools received their computed data, and Chalkbeat
  described the public spreadsheet as bare-bones.
- So only one year (2016-17) has next-year overall points. The pooled loaders add 2017-18
  with a test-based outcome.

Nothing was blocked. Every in.gov and law.cornell.edu request returned 200. One Wayback
CDX prefix query returned an error page; it was not needed.

## Rule check (2026-10-28)

`adapter.rule_check()` compares, for each graded school (letter A-F), the published
overall points with the weighted sum of the published domain points and weights, rounded
half up. It also compares the published grade with the band of the published overall.

| Year | Domain set | Graded | Public | Overall reproduced | Grade = band(published overall) | Grade = band(computed) |
|---|---|---:|---:|---:|---:|---:|
| 2016-17 | grades 3-8 (0.5 / 0.5) | 1,575 | 1,361 | 1.000 | 1.000 | 1.000 |
| 2016-17 | high school (0.2 / 0.2 / 0.6) | 263 | 244 | 0.996 | 1.000 | 1.000 |
| 2016-17 | combined | 161 | 116 | 0.627 | 1.000 | 0.845 |
| 2016-17 | one domain or a domain missing | 77 | 48 | 1.000 | 1.000 | 1.000 |
| 2017-18 | grades 3-8 | 1,563 | 1,352 | 1.000 | 1.000 | 1.000 |
| 2017-18 | high school | 273 | 251 | 1.000 | 1.000 | 1.000 |
| 2017-18 | combined | 163 | 114 | 0.908 (0.994 within 0.1) | 1.000 | 1.000 |
| 2017-18 | one domain or a domain missing | 62 | 43 | 1.000 | 1.000 | 1.000 |

- **Published grade against the published overall.** Bands at 90 / 80 / 70 / 60 give the
  published grade for 2,076 of 2,076 schools (2016-17) and 2,061 of 2,061 (2017-18).
- **Overall from the domain points.**
  - Exact for every school graded on grades 3-8 alone (3,138 school-years) and for every
    single-domain school.
  - All years: 2,015 of 2,076 (2016-17) and 2,046 of 2,061 (2017-18).
- **The misses are combined schools**, plus one high school (Arlington, computed 58.9,
  published 58.4, F either way).
  - In 2017-18 the published weights have three decimals, so the sum misses by at most
    0.1 for all but one.
  - In 2016-17 the published weights of some combined schools are garbled: 51 of 161 sum to
    more than 1.01 (up to 2.73, for example a 3-8 weight of 0.94). For the 105 combined
    schools with all five domains, recomputing the weights from the enrollment file brings
    89% within 0.1.
  - `load()` uses the published overall, so these misses do not affect Q or D.
- **The loaders' D against the published grade** (`extras["eligible_published"]`, the grade
  after appeals): 1,716 of 1,716 at every boundary, and 2,590 of 2,590 in the pooled
  loaders.
- **Grades 3-8 schools.** The two-score rule `0.5 Q1 + 0.5 Q2 >= c - 0.05` reproduces the
  grade for all of them. The index is on a 0.05 grid, and 16 graded school-years (both
  years, public and non-public) sit exactly at a cutoff minus 0.05; they round up to the
  higher grade. The rule allows 1e-9 of float slack.
- **Appeals.**
  - 2016-17: three grades changed between the October 2017 placement and the March 2018
    file, all at non-public schools.
  - 2017-18: 49 grades were embargoed as "Appeal Pending", with no points, 36 of them
    public. The 45 with 2016-17 points had a median of 82.9 that year.
- **Hold harmless.** `hold_harmless_check()`: of 2,012 schools with a 2017-18 and a
  2018-19 letter grade, the 2018-19 grade is never lower. It is the same for 94.0% and
  higher for 6.0%, the schools whose computed 2018-19 grade beat 2017-18. No 2018-19
  points exist, so these years are not in any rule check.

## Sample

| Step (2016-17 grades) | Schools |
|---|---:|
| rows in the domain sheet | 2,104 |
| letter grade A-F (drops 28 "No Grade") | 2,076 |
| public schools (drops 307 non-public) | 1,769 |
| 2017-18 overall points observed (`load()`) | 1,716 |
| of which graded on grades 3-8 alone (`load_two_score()`) | 1,323 |

- **Missing Y.** Of the 1,769 public schools, 53 have no Y:
  - 36 are "Appeal Pending" in 2017-18;
  - 16 have no 2017-18 row (closed or renumbered);
  - 1 is "No Grade".
- **`load()` by domain set:** grades 3-8 1,323, high school 238, combined 113, growth only
  30, other 12.
- **`load()` by grade:** A 462, B 592, C 402, D 169, F 91.
- **Units.** 353 corporations. Four schools run by state turnaround operators are in the
  sample (`extras["turnaround_operator"]`).
- **Pooled loaders.** Grades 3-8 public schools graded in 2016-17 (1,304 with Y) and
  2017-18 (1,286), with `year_2018` in X.

Mass near each cutoff (2016-17 public graded schools within 2 points):

| A/B | B/C | C/D | D/F |
|---:|---:|---:|---:|
| 235 | 208 | 108 | 43 |

**Bunching** (`bunching()`): graded public schools within 1 point below and above each
cutoff, both years.

| Cutoff | Below | Above | Binomial p |
|---:|---:|---:|---:|
| 90 | 115 | 128 | 0.44 |
| 80 | 93 | 114 | 0.16 |
| 70 | 56 | 50 | 0.63 |
| 60 | 13 | 18 | 0.47 |

No significant excess above any cutoff. The 2017-18 appeals (49 schools without points)
could hide some near-cutoff schools.

## Readiness (2026-10-28)

The same statistics as `experiments.scripts.multi_rule_readiness`, computed in memory
(that script was not run, so `runs/multi_rule_readiness/summary.json` is unchanged).

| Loader | n | p | R^2 of Q on X | sd(T), sd(eta) | sd(eta) max/min across T deciles | Treated share | Near cutoff | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `load()` (D/F) | 1,716 | 20 | 0.423 | 8.13, 9.50 | 1.57 | 0.947 | 3.0% | 1.000 |
| `load_cd()` | 1,716 | 20 | 0.423 | same | same | 0.848 | 7.2% | 1.000 |
| `load_bc()` | 1,716 | 20 | 0.423 | same | same | 0.614 | 14.3% | 1.000 |
| `load_ab()` | 1,716 | 20 | 0.423 | same | same | 0.269 | 16.4% | 1.000 |
| `load_pooled()` (D/F) | 2,590 | 18 | 0.454 | 8.30, 9.10 | 1.47 | 0.945 | 3.2% | 1.000 |

- **Y in `load()`.** Mean 82.5, sd 12.3, corr(Y, Q) = 0.78.
- **Y in the pooled loaders.** The mean pass rate is 61.0 (ISTEP+ 2018) and 48.2 (ILEARN
  2019); corr(Y, Q) = 0.75. For 2016-17, the ISTEP+ 2018 pass rate correlates 0.996 with
  the 2017-18 Performance (3-8) points.

**Two-score view: `load_two_score()`.** Q = (Performance 3-8, Growth 4-8), weights
(0.5, 0.5), cutoff 59.95. The sample is the grades 3-8 part of `load()`, and D equals the
published grade for every school.

| Quantity | Performance | Growth |
|---|---:|---:|
| R^2 | 0.679 | 0.121 |
| sd(T), sd(eta) | 12.37, 8.50 | 4.65, 12.51 |
| sd(eta) max/min across T deciles | 1.80 | 1.27 |
| 1% / 50% / 99% quantiles | 21.8 / 64.7 / 91.6 | 66.0 / 98.9 / 128.6 |

- **Correlations.** corr(T1, T2) = 0.868 and corr(eta1, eta2) = 0.503.
- **Weakest direction.** The weakest direction of T is (0.32, -0.95). Its sd is 2.20,
  0.205 of sd(eta). That is better than `tx_school_ratings` (0.16) and `chile_two_score`
  (0.18), below `fl_school_grades` (0.48).
- **Treated share.**
  - D/F: 0.941 treated, 3.3% near the cutoff.
  - B/C (`load_two_score("BC")`): 0.554, 14.8%.
  - A/B: 0.240, 13.1%.
- **Pooled version.** `load_two_score_pooled()` (n = 2,590) has R^2 0.693 and 0.114,
  corr(T) 0.814, and a weakest-direction ratio of 0.234.
- **Growth is barely predicted by X** (R^2 0.12), so most of the trade-off information
  comes from Performance.

**Suggested policy box for the two-score rule** (the index is λ Q1 + (1 - λ) Q2 >= c, with
λ the weight on Performance):

- **Weight.** λ in [0.3, 0.7] (deployed 0.5). Indiana weighted performance and growth
  equally in both grade spans.
- **Cutoff.** c in [55, 70] for the lowest category (deployed 59.95).
- **Treated share across the box.** The share below the cutoff (the F share) runs from
  1.0% (λ = 0.3, c = 55) to 35.6% (λ = 0.7, c = 70); it is 5.9% at the deployed rule.
- **The two choices are not separable.** Growth is centred near 100 and Performance near
  65, so raising λ lowers the index level and must be paired with a lower c to keep the
  same share.
- **If the weight is fixed at 0.5,** c in [50, 70] gives an F share of 1.4%-17.9%.
- **For the B/C boundary,** c in [70, 90] with the same λ range.

| λ \ c | 55 | 60 | 65 | 70 |
|---:|---:|---:|---:|---:|
| 0.3 | 0.010 | 0.016 | 0.036 | 0.078 |
| 0.4 | 0.015 | 0.026 | 0.065 | 0.113 |
| 0.5 | 0.024 | 0.059 | 0.103 | 0.179 |
| 0.6 | 0.051 | 0.090 | 0.157 | 0.253 |
| 0.7 | 0.088 | 0.145 | 0.229 | 0.356 |

## Caveats

- **One cohort.** Next-year overall points exist only for the 2016-17 grades. Hold harmless
  removed the 2018-19 points, and the 2015-16 points were never published. So `load()`
  has no year indicators and n = 1,716.
  - The pooled loaders double n for grades 3-8 schools. Their Y is a test pass rate, and
    the test changed between the two outcome years: ISTEP+ in 2018, ILEARN in 2019, with
    pass rates 13 points lower. `year_2018` absorbs the level shift only.
- **D/F is thin.**
  - 91 F schools, 43 schools within 2 points of 60, 3% within 0.25 sd(eta).
  - B/C and A/B have about five times the mass but no formal consequence.
- **Hold harmless.** The 2018-19 and 2019-20 grades are not computed grades. They are kept
  only as `extras["grade_2019_hold_harmless"]` and never enter a rule check or an outcome.
  - PL 2-2020 was signed after the spring 2019 tests, so it did not change incentives for
    the pooled outcome year.
- **Appeals.** 36 public schools with 2017-18 "Appeal Pending" have no Y. If appeals
  cluster just below cutoffs, their loss is not random.
- **Combined schools.** Their weights vary by school (enrollment shares), so `load()` mixes
  rules. `load_two_score()` uses only the fixed-weight grades 3-8 schools.
  - 2016-17 published weights are wrong for some combined schools. The published overall,
    used for Q, is not affected.
- **Covariates.**
  - The enrollment files give counts for the same school year. They are not computed from
    test scores.
  - Free/reduced meal shares of 100% appear under community eligibility.
  - Locale and charter status come from the 2026-27 directory, matched by school or
    corporation number. 149 `load()` schools (9%) have no locale (`locale_missing`): 138
    are not in the current directory (closed since) and 11 are "Not Applicable" there. Charter status for closed charters comes from the corporation
    number range (9100+ outside the state, ESC and university blocks).
- **Non-public schools** are graded under the same rule (307 in 2016-17). They are excluded
  because the enrollment files have no demographics for them. They are in the cache with
  `nonpublic = 1`.
- **Units and errors.** One row per school in `load()`; the pooled loaders repeat schools
  (1,334 schools in 2,590 rows), so cluster by school. eta is mildly heteroskedastic (sd
  ratio 1.6). Growth domain points over 100 (up to 150) let overall points exceed 100.
- **Shared assumptions.** As in every real dataset screened so far, X independent of eta
  holds only approximately.
