# Louisiana school performance scores: A-F letter grades from cutoffs on the SPS

| Field | Value |
|---|---|
| **Q** | Published School Performance Score (SPS, 0-150, one decimal) minus (own cutoff - 0.05) |
| **Threshold** | 0, on Q. On the SPS: the published minimum for the higher grade, minus 0.05. The scale is A 90, B 75, C 60, D 50 in every graded year 2018-2025 (`CUTOFFS`) |
| **Treatment** | `load()`: `1{Q >= 0}`, a grade of B or better rather than C. `load_ab`, `load_cd`, `load_df`: the other boundaries, with D = the higher grade |
| **Rule class** | several cutoffs on one score (A/B/C/D/F bands); the score is a fixed-weight sum of component indices, with weights set by the school's grade span; `load_two_score()`: deployed weighted sum 0.75 assessment + 0.25 progress |
| **X** | From the October 1 enrollment count of the same school year: % economically disadvantaged, % Black, % Hispanic, % Asian, % female, % limited English proficient, log enrollment, shares of enrollment in PreK-2, 6-8 and 9-12, has grade 8, charter. Plus selective admissions (magnet), school type, year and parish indicators (89 columns; 10 continuous) |
| **Y** | `load()`: next SPS year's published SPS. Other loaders: next assessment index, next grade (A=4 ... F=0), no SPS next year |
| **n** | 5,839 school-years (1,302 schools) graded A-F in spring 2018, 2019 and 2022-2024, with Y observed. `load_two_score()`: 1,089 |
| **Source** | LDOE School Performance Score files, Multiple Statistics by Site (October 1 enrollment), and LDOE's SPS overview, FAQs and SPS calculators, all from doe.louisiana.gov |

## Why Louisiana

The candidates were checked in the order given (2026-10-28):
1. **Arizona A-F: not usable by script.** Every path on www.azed.gov, including
   `robots.txt` and the `/sites/default/files/` file paths, returns a Cloudflare "Just a
   moment..." challenge (HTTP 403). azreportcards.azed.gov loads, but it is an interactive
   app with no bulk component files. The A-F data files are listed on
   https://www.azed.gov/accountability-research/data for a manual download.
2. **Louisiana: used.** The SPS files, the enrollment files and the guides download from
   doe.louisiana.gov with no login, form or challenge.

Mississippi, Oklahoma and Ohio were not checked.

## The rule

Each school's SPS is a fixed-weight sum of component indices. Each index is on a 0-150
scale and is published to one decimal. The weights depend on the school's grades.
They come from LDOE's 2021-22 and 2022-23 SPS calculators (`data/raw/guides/`) and are
confirmed by the files:

| School | 2017-18, 2018-19 | 2020-21 to 2024-25 (adds interests & opportunities, I&O) |
|---|---|---|
| K-8, no grade 8 | 75% assessment + 25% progress | 70% assessment + 25% progress + 5% I&O |
| K-8 with grade 8 | 70% assessment + 25% progress + 5% dropout/credit accumulation (DCAI) | 65 / 25 / 5 DCAI / 5 I&O |
| High school | 12.5% assessment, 12.5% progress, 25% ACT/WorkKeys, 25% graduation rate index, 25% strength of diploma | the same with 20% graduation rate index and 5% I&O |
| Combination | K-8 and high-school SPS weighted by their numbers of testers | same |

The calculators round each index to 0.1, then round the weighted sum to 0.1, half up.
The grade compares that rounded SPS with the scale:

| Spring year | A | B | C | D | Source |
|---|---|---|---|---|---|
| 2018, 2019 | 90.0 | 75.0 | 60.0 | 50.0 | `2019-2020-sps-updates.pdf` ("2017-18 through 2020-21") |
| 2022, 2023 | 90.0 | 75.0 | 60.0 | 50.0 | grade formulas in the 2021-22 and 2022-23 calculators; `2023-sps-release-faq.pdf` (2022-23) |
| 2024, 2025 | 90.0 | 75.0 | 60.0 | 50.0 | no scale document found on the LDOE pages checked; confirmed by the files (rule check below) |

F is everything below 50.

- The 2019-20 overview planned a rise to 95/80/65/50 for 2021-22 and to 100/85/70/50 for
  2024-25. The files show it never took effect.
- 2019-20 has no SPS (COVID-19). 2020-21 has *simulated* SPS (values prefixed `C19-`)
  without letter grades, "for informational purposes". They serve only as the outcome
  for 2018-19.
- 2024-25 is the last year of this formula. The "Grow. Achieve. Thrive." model starts in
  2025-26 (`2024-2025-sps-release-for-systems.docx`).

"Published SPS >= c" is exactly "unrounded SPS >= c - 0.05", so `load()` centres the
published SPS at c - 0.05. Q lies on a 0.1 grid with 837 distinct values in the sample,
and it is never exactly 0. Because the scale is the same every year,
`extras["group"]` (the year) and `extras["group_cutoffs"]` carry one cutoff, 74.95 on
the SPS for B/C. `extras["cutoffs"]` holds each unit's four cutoffs on the SPS scale.

The boundary with the most mass is B/C. The number of school-years within 2 points of
each cutoff is 352 (A/B), 518 (B/C), 426 (C/D) and 255 (D/F).

The stakes:
- Comprehensive Intervention Required (CIR) for schools graded D or F over several
  years (2023 FAQ). This is the policy treatment at C/D and D/F, so there it is 1 - D.
- The "Academically Unacceptable School" (AUS) label at D/F. Only F schools carry it:
  565 of the 612 F school-years in 2018-2025 (column `aus` in the panel). AUS schools
  must offer school choice (2020-21 simulated-SPS FAQ).
- The public letter grade itself.

## Getting the data

    python -m experiments.datasets.la_school_grades.download          # idempotent
    python -m experiments.datasets.la_school_grades.download --check  # sizes on disk

All 19 files total 8.1 MB. They are written to `data/raw/` (gitignored). The source is
LDOE's public document library, `doe.louisiana.gov/docs/default-source/`, linked from
the performance-scores, enrollment-data and accountability pages. There is no login,
form or CAPTCHA.

- **Reading.** The environment has neither `xlrd` nor `openpyxl`, so the adapter reuses
  the standard-library reader `experiments/datasets/fl_school_grades/_spreadsheet.py`.
- **Cache.** The adapter caches `data/processed/la_school_grades_panel.csv.gz`. It holds
  all 8,818 school-years on the regular and closed-school sheets, 2018-2025. Delete it
  after changing `adapter.py`.

| File (under `https://doe.louisiana.gov/docs/default-source/`) | Bytes | Use |
|---|---:|---|
| `data-management/2018-school-performance-scores.xlsx` | 474,915 | 2017-18 (new formula block) |
| `data-management/2019-school-performance-scores.xlsx` | 396,382 | 2018-19 |
| `data-management/2021-school-performance-scores.xlsx` | 464,703 | 2020-21 simulated SPS; outcome for 2018-19 |
| `data-management/2022-school-performance-scores.xlsx` | 562,499 | 2021-22 |
| `data-management/2023-school-performance-scores.xlsx` | 508,555 | 2022-23 |
| `data-management/2024-school-performance-scores.xlsx` | 349,463 | 2023-24 |
| `data-management/2025-school-performance-scores.xlsx` | 349,784 | 2024-25; outcome for 2023-24 |
| `data-management/oct-2017-multi-stats-(total-by-site-and-school-system).xlsx` | 750,426 | covariates 2017-18 |
| `data-management/oct-2018-multi-stats-(total-by-site-and-school-system).xlsx` | 730,428 | covariates 2018-19 |
| `data-management/oct-2021-multi-stats-(total-by-site-and-school-system).xlsx` | 492,894 | covariates 2021-22 |
| `data-management/oct-2022-multi-stats-(total-by-site-and-school-system).xlsx` | 500,242 | covariates 2022-23 |
| `data-management/oct-2023-multi-stats-(total-by-site-and-school-system)_web.xlsx` | 500,568 | covariates 2023-24 |
| `data-management/oct-2024-multi-stats-(total-by-site-and-school-system).xlsx` | 369,330 | covariates 2024-25 |
| `accountability/2019-2020-sps-updates.pdf` | 1,025,943 | letter-grade scale, index definitions |
| `data-management/2023-sps-release-faq.pdf` | 308,996 | 2022-23 scale; A cap for schools in UIR |
| `data-management/simulated-sps-release-faq-document.pdf` | 85,002 | 2020-21 simulated SPS |
| `teacher-toolbox-resources/2021-2022-sps-calculator.xlsx` | 62,939 | weights, rounding |
| `teacher-toolbox-resources/2022-2023-sps-calculator.xlsx` | 57,072 | weights, rounding, grade formula |
| `accountability/2024-2025-sps-release-for-systems.docx` | 137,068 | 2024-25 is the last year of the formula |

## Rule check (2026-10-28)

`adapter.rule_check()` does two comparisons:
- the published grade against the grade computed from the published SPS and `CUTOFFS`,
  for every school-year with a published A-F grade on the regular and closed-school
  sheets;
- for K-8 and high schools with every index unmasked, the published SPS against the
  SPS recomputed from the published indices with the weights and rounding above.

| Spring year | Rows | Graded A-F | T (turnaround) | Grade agreement | Published B, computed A | K-8 SPS reproduced | High-school SPS reproduced |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 1,338 | 1,336 | 2 | 0.9985 | 2 | 930 of 930 | 166 of 166 |
| 2019 | 1,293 | 1,291 | 2 | 0.9992 | 1 | 921 of 921 | 170 of 170 |
| 2021 (simulated) | 1,216 | 0 | 0 | - | - | 766 of 766 | none unmasked |
| 2022 | 1,180 | 1,177 | 3 | 1.0000 | 0 | 646 of 646 | none unmasked |
| 2023 | 1,276 | 1,274 | 2 | 1.0000 | 0 | 474 of 474 | 140 of 140 |
| 2024 | 1,262 | 1,261 | 1 | 0.9976 | 3 | 283 of 283 | 80 of 80 |
| 2025 | 1,253 | 1,252 | 1 | 0.9984 | 2 | 234 of 234 | 74 of 74 |

- **Grade agreement.** 7,583 of 7,591 published grades (99.89%) match the cutoffs. All
  8 mismatches are at A/B: a computed A published as B. This fits the rule in the 2023
  FAQ that a school with a subgroup in Urgent Intervention Required cannot earn an A.
  The files do not carry that flag, so this explanation is not verified.
- **Formula.** Every unmasked K-8 SPS (4,254) and high-school SPS (630) is reproduced
  exactly.
  - From 2020-21 on, the I&O index is always included. The 2019-20 overview's
    "hold harmless" rule (drop I&O if it lowers the SPS) does not appear in the files.
  - Indices are masked (`>145`, `<5`, `~`) for small or extreme groups. The K-8 I&O
    index is masked for 205-735 schools a year from 2022, and the high-school I&O index
    for every high school in 2021 and 2022, so fewer schools can be checked then.
  - Combination-school SPS are not recomputed: they need tester counts that are not
    published.
- **Agreement in the samples.** `extras["eligible_published"]` agrees with D for all
  5,839 school-years at B/C, C/D and D/F, and for 5,833 at A/B.

## Readiness (2026-10-28)

`python -m experiments.scripts.multi_rule_readiness la_school_grades` reports, for the
B/C boundary:

| Quantity | Value |
|---|---:|
| n, p (continuous X) | 5,839, 89 (10) |
| R^2 of Q on X | 0.765 |
| sd(T), sd(eta) | 15.05, 8.34 |
| sd(eta) max/min across T deciles | 1.19 |
| Treated share (B or better) | 0.491 |
| Within 0.25 sd(eta) of the cutoff | 9.3% |
| Agreement with the published grade | 1.000 |

By year (the `group`), n is 1,118-1,225, R^2 is 0.77-0.80 and the treated share is
0.47-0.52. By school type:

| Type | n | R^2 | Treated share |
|---|---:|---:|---:|
| Elementary/middle | 4,213 | 0.77 | 0.41 |
| High | 894 | 0.80 | 0.70 |
| Combination | 732 | 0.78 | 0.70 |

The other boundaries use the same X, n and outcome:

| Boundary | Treated share | Near cutoff | Agreement |
|---|---:|---:|---:|
| A/B (`load_ab`) | 0.173 | 6.4% | 0.9990 |
| C/D (`load_cd`) | 0.785 | 7.5% | 1.000 |
| D/F (`load_df`) | 0.923 | 4.5% | 1.000 |

Other facts about the fit:
- Without the parish indicators, R^2 is 0.71.
- Adding the previous SPS year's score (`extras["q_lag"]`, 4,428 rows) raises R^2 from
  0.78 to 0.92 and lowers sd(eta) to 4.8. It is not in X, because it depends on the
  previous grade.

**Two-score view: assessment against progress.** The weights are fixed only within a
component set, and the progress index enters every K-8 formula. `load_two_score()`
therefore takes K-8 schools with no grade-8 (DCAI) component in 2018 and 2019, before
the I&O index existed.
- For them, SPS = round_0.1(0.75 Q1 + 0.25 Q2), where Q1 is the K-8 assessment index and
  Q2 the K-8 progress index, both as published.
- The grade is B or better exactly when 0.75 Q1 + 0.25 Q2 >= 74.95. The rule reproduces
  the published SPS and grade for all 1,089 school-years (569 in 2018, 520 in 2019).
- Schools with grade 8 (DCAI, 5%) or with I&O (from 2020-21) would need a third score or
  a unit-specific offset. The I&O index is masked as `>145` for most schools, so the
  offset would not be exact. Neither is implemented.

| Quantity | Assessment | Progress |
|---|---:|---:|
| Mean, sd of Q | 62.8, 18.7 | 92.2, 11.4 |
| R^2 | 0.841 | 0.343 |
| sd(T), sd(eta) | 17.2, 7.5 | 6.7, 9.3 |
| sd(eta) max/min across T deciles | 1.74 | 1.34 |

- corr(T1, T2) = 0.79 and corr(eta1, eta2) = 0.50.
- The weakest direction of T is (0.31, -0.95). Its sd is 3.9, 0.46 of sd(eta), about
  the same as `fl_school_grades` (0.48).
- 41% are treated, 8.0% lie near the boundary, and agreement is 1.000.

## Outcome

`load()` uses Y = the next SPS year's published SPS, matched by site code. Its mean is
74.4, its sd 17.5, and corr(Y, Q) = 0.95.

Of the 6,020 eligible school-years (graded A-F, not alternative, not K-2), 181 (3.0%)
have no Y. None of them has a row in the next file:
- 86 are on that year's "Closed School" sheet;
- 82 more are 2018-19 rows. The 2020-21 simulated file has no school in Calcasieu or
  Cameron parish (53 and 4 of these rows). Both parishes were hit by Hurricanes Laura
  and Delta in 2020; the files do not give a reason.
- 13 are in other years.

The outcome year depends on the grading year:
- 2018-19 is followed by the 2020-21 simulated SPS (`extras["outcome_simulated"]`).
- `extras["y_year"]` records the outcome year: 2019 (1,225), 2021 (1,121), 2023
  (1,118), 2024 (1,190), 2025 (1,185).

Other loaders:
- `load_next_assessment`: n = 5,838.
- `load_next_grade`: n = 4,718. The 2018-19 rows drop out, because 2020-21 has no grades.
- `load_closed`: Y = no SPS next year, n = 6,020, mean 3.0%.

## Caveats

- **Scores and units.**
  - Q is a school aggregate on a 0.1 grid. Its noise includes cohort sampling noise, so
    eta is heteroskedastic: in the two-score view, sd(eta) for assessment changes
    1.74-fold.
  - Schools repeat across up to five years (1,302 schools), so inference should cluster
    by school.
- **The outcome shares data with Q.** LDOE averages the K-8 progress index over two
  years. Next year's SPS therefore reuses one year of this year's growth data, which
  inflates corr(Y, Q) beyond any effect of the grade.
- **High R^2.**
  - School type and grade span explain much of Q: high schools score higher, with 70%
    at B or better against 41% of K-8 schools.
  - Within each type, R^2 is still 0.77-0.80.
- **The score changes over the years.**
  - I&O was added in 2020-21, and ELPT progress points are in the assessment index.
  - Year indicators absorb level shifts only.
- **Covariate timing.** All enrollment covariates are October 1 counts for the same
  school year, fixed before the spring tests. The 2024-25 file reports percentages as
  text (`"87.9%"`); they are parsed.
- **Exclusions.**
  - `T` (turnaround) grades: 10 school-years in 2018-2024.
  - Alternative schools: they have a separate sheet and formula from 2018-19; in
    2017-18 they are flagged on the regular sheet (28 school-years).
  - K-2 schools with no enrollment in grades 3-12 (291 school-years in the sample
    years): their scores come from the schools they feed, and for 38% of them the
    indices equal another school's exactly.
  - 2024-25: no next-year SPS under this formula.
- **The treatment and the method's assumptions.**
  - D is the published grade, which is sharp in Q except for the 8 A-caps at A/B.
  - Because a school's grade feeds its next score through the stakes, `q_lag` is
    excluded from X.
  - Like every real dataset screened so far, X independent of eta holds only
    approximately.
