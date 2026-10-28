# Texas A-F campus ratings: 70% of the better of two domains plus 30% of a third

| Field | Value |
|---|---|
| **Q** | `load()`: x - (c - 0.5), where x = 0.7 max(SA, SP) + 0.3 CtG is the unrounded overall scaled score and c the published cutoff. `load_two_score()`: (M, CtG) with M = max(SA, SP). `load_three_score()`: (SA, SP, CtG) |
| **Threshold** | 0, on Q. On x: c - 0.5 = 89.5 for the A/B boundary (79.5, 69.5, 59.5 for the others). The same in every year and campus type |
| **Treatment** | `load()`: `1{Q >= 0}`, an overall A rather than a B. `load_bc`, `load_cd`, `load_df`: the other boundaries, D = the higher grade |
| **Rule class** | the better of two scores, then a fixed-weight sum with a third (0.7, 0.3), then grade bands (90 / 80 / 70 / 60) |
| **X** | % economically disadvantaged, % emergent bilingual, log number of students, district log enrollment and % economically disadvantaged, charter, lowest and highest grade served, school type, year and ESC region indicators (34 columns; 5 continuous) |
| **Y** | `load()`: the next rated year's overall scaled score. Other loaders: next Student Achievement score, next Closing the Gaps score, no row next rated year |
| **n** | 38,075 campus-years (8,140 campuses), non-AEA, all three domains, rated 2019, 2022, 2023, 2024 or 2025, with Y observed |
| **Source** | TEA 2026 Statewide Multi-Year Ratings Spreadsheet (2018-2026 sheets) and the Accountability Manuals, Chapter 5, 2018-2026, all static files on tea.texas.gov |

SA is Student Achievement, SP School Progress (the better of Part A: Academic Growth and
Part B: Relative Performance) and CtG Closing the Gaps. Each is a scaled score from 0 to
100, published as a whole number.

## The rule

Chapter 5 ("Calculating Ratings") of each year's Accountability Manual gives the overall
campus score in the same steps every year:

1. M = the better of SA and SP. If either is below 60, M is at most 89.
2. overall = 0.7 M + 0.3 CtG. A campus without a CtG score gets overall = M.
3. Rounding: the manual rounds the weighted domain terms to one decimal and the overall
   score to a whole number. Since the domain scores are whole numbers, 0.7 M and 0.3 CtG
   are exact to one decimal, and the overall score is round-half-up(x). The data confirm
   half-up: x ends in .5 for many campuses, and all of them round up.
4. The "3 Fs" cap: if three of SA, Academic Growth, Relative Performance and CtG are
   below 60, the overall score is at most 59.
5. The "3 Ds" cap (2023 onward): if three of the four are below 70, at most 69.

| Rating year | Overall cut points (A / B / C / D) | Exceptions in the rule | Source |
|---|---|---|---|
| 2018 | 90 / 80 / 70 / 60, districts only | Campuses got Met Standard (>= 60) or Improvement Required. 3 Fs cap with no SA condition | `ch5_2018.pdf` |
| 2019 | 90 / 80 / 70 / 60 | 3 Fs cap only if SA < 60 | `ch5_2019.pdf` |
| 2020, 2021 | none | Not rated: "Declared State of Disaster" (COVID-19) | multi-year spreadsheet |
| 2022 | 90 / 80 / 70; below 70 "Not Rated: Senate Bill 1365" | 3 Fs cap if SA < 60 | `ch5_2022.pdf` |
| 2023 | 90 / 80 / 70 / 60 | 3 Fs if SA < 60; 3 Ds if SA < 70 | `ch5_2023.pdf` |
| 2024, 2025, 2026 | 90 / 80 / 70 / 60 | as 2023 | `ch5_2024.pdf`, `ch5_2025.pdf`, `ch5_2026.pdf` |

- The weights (0.7, 0.3), the max and the 89 cap are the same in every manual,
  2018-2026. The rule for a missing CtG (overall = M) is stated from 2019 on.
- **The 2023 refresh** left the overall cut points alone. It changed the raw-score cut
  points behind the domain scores. Campus CCMR (non-AEA) went from 60 / 48 / 39 / 26 to
  88 / 78 / 64 / 51. School Progress Part A for elementary schools went from
  82 / 75 / 69 / 64 to 80 / 71 / 63 / 56. CtG for elementary schools went from
  95 / 85 / 48 / 23 to 74 / 60 / 33 / 12. The graduation-rate conversion table and the
  Relative Performance lookup tables changed too, and the 3 Ds cap was added.
  - The STAAR, CCMR, Part A and CtG cut tables in the 2024-2026 manuals equal the 2023
    ones. So domain scores are comparable from 2023 on, but not across 2022/2023.
  - The year indicators absorb level shifts only.
- **AEA campuses** (alternative education accountability) use AEA cut points. In
  2018-2022 they could add up to 10 bonus points to the overall score. They are excluded.
- **Paired campuses** have no tested grades of their own (mostly PK-2). They are paired
  with another campus or with the district and publish its overall score, with no domain
  scores. They are named in the 2023+ sheets ("Paired Type"); before 2023 they show as
  an overall score with no domain scores. Both are excluded.
- **Local accountability systems (LAS).** Under Chapter 11, a district with an approved
  LAS plan blends a local score, up to 50%, into the overall score of campuses rated C
  or better. The domain scores stay the state ones. TEA posts no static list of LAS
  districts. The adapter therefore flags a district-year as LAS when a non-AEA campus's
  published overall differs from the Chapter 5 score. That gives:
  - Dallas ISD in 2019 (182 of 232 campuses differ) and in 2022 (176 of 237);
  - Snyder ISD in 2019 (3 of 4);
  - Bellville ISD in 2023-2026 (3-5 of 5).

  Every published overall that differs from the rule falls in one of these
  district-years. Their 463 campus-years are excluded.
- **Litigation.** TEA's letters ("TAA correspondence") give the public release dates.
  The 2023 ratings were posted on TXSchools.gov on 2025-04-24 (final ratings on
  2025-08-21), after a 15th Court of Appeals ruling. The 2024 ratings were posted on
  2025-08-15, the same day as the 2025 ratings. Both were blocked in court until then.
  Districts saw the underlying data earlier: the 2023 subsets in TEAL from 2023-11-16.

## Getting the data

    python -m experiments.datasets.tx_school_ratings.download          # idempotent
    python -m experiments.datasets.tx_school_ratings.download --check  # sizes, hashes

The 11 files total 25.0 MB and go to `data/raw/` (gitignored). `download.py` records each
file's SHA-256 from 2026-10-28 and reports a difference, since TEA replaces files in place.
One run on 2026-10-28 got a transient `503 Backend fetch failed` from tea.texas.gov. The
unchanged script succeeded on retry.

| File | Bytes | Use |
|---|---:|---|
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2026-statewide-multi-year-ratings-spreadsheet.xlsx | 18,621,904 | all data: one "State Summary" sheet per year, 2018-2026 |
| https://tea.texas.gov/texas-schools/accountability/academic-accountability/performance-reporting/chapter-5-calculating-2018-ratings.pdf | 555,454 | rule 2018 |
| https://tea.texas.gov/sites/default/files/Chapter%205%20Calculating%202019%20Ratings_adopted.pdf | 393,543 | rule 2019 |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/chapter-5-2022-calculating-ratings.pdf | 1,414,427 | rule 2022 |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/chapter-5-2023-calculating-ratings-2.pdf | 967,695 | rule 2023 |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2024-accountability-manual-chapter-05.pdf | 1,149,589 | rule 2024 |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/chapter-5-2025-calculating-ratings-0.pdf | 627,870 | rule 2025 |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/chapter-5-2026-calculating-ratings-1.pdf | 649,768 | rule 2026 |
| https://tea.texas.gov/sites/default/files/Chapter%2011%202019%20Local%20Accountability%20Systems_adopted.pdf | 104,491 | LAS 2019 |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/chapter-11-2025-local-accountability-systems-0.pdf | 154,282 | LAS 2025 |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/chapter-3-2025-school-progress-domain-0.pdf | 391,602 | % econ. disadvantaged from the PEIMS October snapshot |

- **The spreadsheet** is linked from TEA's 2026 rating-system page. It holds every
  district and campus in every year, with these fields: overall and domain ratings and
  scaled scores; school type and grades served; the AEA flag and charter status (2019+);
  number of students; % economically disadvantaged and % emergent bilingual (LEP before
  2022); ESC region and county; and pairing (2023+).
- **Agreement with TEA's single-year files** (checked 2026-10-28):
  - 2019: matches `2019-state-overall-ratings.xlsx` for every campus in the fields
    compared: overall, SA, SP and CtG scores, number of students and % economically
    disadvantaged. Only 4 rating labels are spelled differently.
  - 2023 and 2024: identical to the sheets in
    `2025-enhanced-statewide-summary-after-appeal-updated-0.xlsx`.
  - 2025: differs only in 10 domain-score cells, filled in after appeals.
- **Reading.** The adapter reads the `.xlsx` with `fl_school_grades/_spreadsheet.py`, a
  standard-library reader, because the environment has no `openpyxl`.
- **Cache.** The adapter caches `data/processed/tx_school_ratings_panel.csv.gz`, which
  holds all 62,878 campus-years of 2018, 2019 and 2022-2026. Delete it after changing
  `adapter.py`.

**Not used: the form-driven downloads.** TEA's "Accountability Data Downloads"
(`https://rptsvr1.tea.texas.gov/perfreport/account/acct_download?year=YYYY`, 2023-2026)
and the per-year `.../perfreport/account/YYYY/download.html` pages (2018, 2019, 2022) work
through a selection form. The form submits to a SAS broker (`/cgi/sas/broker`), and there
are no static files. The same holds for the TAPR downloads
(`https://rptsvr1.tea.texas.gov/perfreport/tapr/YYYY/download/DownloadData.html`). This
dataset needs neither, because the static spreadsheet has the scores and demographics.

Variables found only there have to be downloaded by hand:
- **Mobility.** From TAPR: campus level, category "Profile: Staff, Student, and Annual
  Graduates", one file per year.
- **Domain component scores** (STAAR, CCMR, graduation; Part A and B raw scores). From
  the accountability download: "Accountability Component Scores" (2023+), or the domain
  categories.

The `DL_index.html` path in the brief returns 404 for every year.

## Rule check (2026-10-28)

`adapter.rule_check()` compares, for non-AEA campuses with all three domain scores, the
published overall score with the Chapter 5 score computed from the published SA, SP and
CtG (and, for the caps, Growth and Relative Performance):

| Rating year | Campuses | AEA | Paired | Three domains, non-AEA | Agreement, all | LAS campuses | Agreement, excl. LAS | 0.7 M + 0.3 CtG alone | 3 Fs caps | 3 Ds caps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 8,759 | 377 | 346 | 7,570 | 1.000 | 0 | 1.000 | 0.991 | 65 | - |
| 2019 | 8,838 | 379 | 357 | 7,624 | 0.976 | 223 | 1.000 | 0.995 | 40 | - |
| 2022 | 8,966 | 375 | 363 | 7,771 | 0.977 | 225 | 1.000 | 0.992 | 60 | - |
| 2023 | 9,044 | 381 | 382 | 7,882 | 0.9996 | 5 | 1.000 | 0.958 | 96 | 234 |
| 2024 | 9,082 | 371 | 394 | 7,896 | 0.9995 | 5 | 1.000 | 0.951 | 163 | 223 |
| 2025 | 9,084 | 406 | 381 | 7,903 | 0.9994 | 5 | 1.000 | 0.961 | 129 | 177 |
| 2026 | 9,105 | 418 | 387 | 7,888 | 0.9996 | 5 | 1.000 | 0.972 | 66 | 156 |

- **Overall score.** Outside the LAS district-years, the computed overall score equals
  the published one for every campus, 54,066 of 54,066. Without the 3 Fs / 3 Ds caps,
  the 70/30 formula alone agrees for 95-99%. The 89 cap on M never binds.
- **Published rating against the cut points**, applied to the published overall:
  - 2022-2026: 1.000.
  - 2019: 8,030 of 8,031. Pecos H S is rated D with an overall score of 80.
  - 2018: 7,896 of 7,900. Four campuses are Met Standard at 58-59.
- **AEA campuses** (information only): the rule without bonus points matches 31% in 2019
  and 13% in 2022 (bonus points), and 99% in 2023-2026.
- **The loaders' D against the published rating** (`extras["eligible_published"]`):
  - A/B boundary, both loaders: agrees for 38,075 of 38,075.
  - B/C: 3 misses (0.99992). One is Pecos H S. The other two are 3 Ds caps on campuses
    with x of 81.1 and 79.6, published at 69.
  - C/D: 621 misses (0.984). These are 3 Ds caps to 69 and 3 Fs caps to 59 on campuses
    with x >= 69.5.
  - D/F: 467 misses (0.988), from the 3 Fs caps. In 2022 D and F were both published as
    SB 1365, so this boundary uses the published score that year.

**Sample.**

| Step (rating years 2019, 2022-2025) | Campus-years left |
|---|---:|
| all campus rows | 45,014 |
| non-AEA | 43,102 |
| published overall score (drops Not Rated) | 41,129 |
| own domain scores (drops paired) | 39,252 |
| all three domains (drops 176 without CtG) | 39,076 |
| not an LAS district-year | 38,613 |
| an ordinary rating label (drops 4 "NR-Data Integrity", 2019) | 38,609 |
| Y observed | 38,075 |

The boundary with the most mass is A/B. The number of campus-years within 2 points of
each cutoff on x is 5,955 (A/B), 5,281 (B/C), 2,486 (C/D) and 1,062 (D/F). x takes 538
distinct values.

## Readiness (2026-10-28)

`python -m experiments.scripts.multi_rule_readiness tx_school_ratings` reports, for the
A/B boundary:

| Quantity | Value |
|---|---:|
| n, p (continuous X) | 38,075, 34 (5) |
| R^2 of Q on X | 0.371 |
| sd(T), sd(eta) | 6.31, 8.22 |
| sd(eta) max/min across T deciles | 2.13 |
| Treated share (A) | 0.225 |
| Within 0.25 sd(eta) of the cutoff | 16.2% |
| Agreement with the published rating | 1.000 |

- sd(eta) falls steadily from 10.6 in the lowest T decile to 5.0 in the highest. The
  ceiling does not explain it: 0.01% of campuses have x >= 99.5.
- The 20 (year, school type) groups have n of 359-4,413, R^2 of 0.26-0.46 and treated
  shares of 0.15-0.34. R^2 is lowest for combined (K-12) campuses.

The other boundaries use the same X, n and outcome (R^2 0.371, sd(eta) 8.22):

| Boundary | Treated share | Near cutoff | Agreement with published |
|---|---:|---:|---:|
| B/C (`load_bc`) | 0.601 | 14.6% | 0.99992 |
| C/D (`load_cd`) | 0.858 | 6.8% | 0.984 |
| D/F (`load_df`) | 0.958 | 2.9% | 0.988 |

Other facts about the fit:
- Without the region indicators, R^2 is 0.30.
- With 1,181 district indicators, R^2 is 0.52 and sd(eta) 7.1.
- Adding the previous rated year's overall score (`extras["q_lag"]`) raises R^2 to 0.55
  and lowers sd(eta) to 6.9. It is not in X, because it depends on the previous rating.

**Two-score view: `load_two_score()`**, Q = (M, CtG), weights (0.7, 0.3), cutoff 89.5.
The sample is the same, and D equals `load()`'s D for every unit:

| Quantity | M = max(SA, SP) | CtG |
|---|---:|---:|
| R^2 | 0.373 | 0.298 |
| sd(T), sd(eta) | 6.10, 7.91 | 7.03, 10.79 |
| sd(eta) max/min across T deciles | 2.42 | 1.71 |

- corr(T1, T2) = 0.949 and corr(eta1, eta2) = 0.735.
- The weakest direction of T is (-0.76, 0.65). Its sd is 1.47, 0.16 of sd(eta). The
  trade-off between the two scores is therefore weakly identified from X, about as
  weakly as in `chile_two_score` (0.18) and well below `fl_school_grades` (0.48).

**Three-score view: `load_three_score()`**, Q = (SA, SP, CtG):

| Quantity | SA | SP | CtG |
|---|---:|---:|---:|
| R^2 | 0.588 | 0.263 | 0.298 |
| sd(eta) max/min across T deciles | 1.92 | 1.70 | 1.71 |

- corr(T) is 0.79-0.90. The weakest direction of T has sd 0.18 of sd(eta).
- The readiness script's "near boundary" share uses equal weights here, so it does not
  describe the max rule.

## Outcome

`load()` uses Y = the next rated year's overall scaled score, matched by campus number.
- Its mean is 80.8, its sd 10.4, and corr(Y, Q) = 0.66.
- In LAS district-years, the outcome is the state score recomputed from the published
  domains, not the blended score.
- A campus that is paired in the outcome year has no Y, because its published score is
  its partner's.

The outcome year depends on the rating year. `extras["y_year"]` records it:

| Rating year | Outcome year | n |
|---|---|---:|
| 2019 | 2022 (no ratings in 2020-2021) | 7,222 |
| 2022 | 2023 (refreshed domain cut points) | 7,487 |
| 2023 | 2024 | 7,782 |
| 2024 | 2025 | 7,797 |
| 2025 | 2026 | 7,787 |

Of the 38,609 eligible campus-years, 534 (1.4%) have no Y:
- 465 have no row in the next rated year's file (closure, merger or renumbering; 147 of
  them for 2019 -> 2022);
- 52 are paired in the outcome year;
- 17 are Not Rated in the outcome year.

Other loaders:
- `load_next_achievement` (next SA, n = 38,078).
- `load_next_ctg` (next CtG, n = 38,048).
- `load_closed` (no row next rated year: mean 1.2%, n = 38,609).

## Caveats

- **The treatment was not always public before the outcome.**
  - The 2023 and 2024 ratings were blocked in court and posted on 2025-04-24 and
    2025-08-15. So the 2024 and 2025 scores that serve as their outcomes come from
    tests taken before the ratings were public.
  - Districts did have the underlying data earlier (TEAL). `extras["released_after_outcome"]`
    flags these two years (15,579 campus-years).
  - The years with a normal release are 2019 (outcome three years later, across
    COVID-19), 2022 and 2025.
- **The rating labels differ by year.**
  - In 2022, campuses below 70 got "Not Rated: SB 1365" instead of D or F
    (`extras["sb1365_year"]`).
  - 2018 campuses got no letter grade, so 2018 serves only as the lagged score for 2019.
    TEA's 2018 page also links a January 2019 "2017-18 A-F Ratings for Campuses" report,
    which was not used.
  - 2026 serves only as the outcome for 2025.
- **The score changes over the years.**
  - The 2023 refresh changed the domain scaling (see "The rule"), and 2023 introduced
    the 3 Ds cap.
  - The year indicators absorb level shifts only.
  - The domain scores are whole numbers, so x lies on a 0.1 grid. Many campuses sit
    exactly at x = 89.5, where they round up to an A. The two-score rule therefore uses
    a 1e-9 tolerance.
- **The two-score rule is not the full rule.** The 3 Fs / 3 Ds caps also depend on
  Growth and Relative Performance.
  - They never move a campus across 89.5, so the main loaders are exact.
  - They move 2 campuses across 79.5, 620 across 69.5 and 467 across 59.5.
  - For the lower boundaries, D (from x) and the published grade can differ.
- **Covariate timing.**
  - % economically disadvantaged, % emergent bilingual and the number of students
    describe the same school year (PEIMS October snapshot), before the spring tests.
  - % economically disadvantaged also sets the Relative Performance cut points, so it
    enters SP mechanically.
  - Mobility is not included (TAPR is form-driven; see above).
- **Units.**
  - Campuses repeat across up to five rating years (8,140 campuses), so inference should
    cluster by campus. District effects are large (R^2 0.37 -> 0.52 with district
    indicators), so clustering by district is worth considering.
  - eta is heteroskedastic (sd ratio 2.1).
- **Exclusions.** AEA, paired, missing-CtG and LAS campuses are excluded; see the sample
  table.
- **Shared assumptions.** As in every real dataset screened so far, X independent of eta
  holds only approximately.
