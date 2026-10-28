# Florida school grades: A-F bands on the percent of points, cutoffs by school type

| Field | Value |
|---|---|
| **Q** | Unrounded percent of points x = total points / number of components (each component 0-100; up to 11 before 2023-24 and 12 since), minus (own cutoff - 0.5) |
| **Threshold** | 0, on Q. On x: the published minimum for the higher grade, minus 0.5. Group cutoffs by spring year and school type are in `CUTOFFS` and in the table below |
| **Treatment** | `load()`: `1{Q >= 0}`, a grade of B or better rather than C. `load_ab`, `load_cd`, `load_df`: the other boundaries, with D = the higher grade |
| **Rule class** | several cutoffs on one score (A/B/C/D/F bands); group-specific cutoffs (school type x year); the score is a fixed-weight sum of its components |
| **X** | % economically disadvantaged, % minority, log October membership, charter, Title I, alternative/ESE center, collocated, virtual, number of components, school type, year and district indicators (88 columns; 3 continuous) |
| **Y** | `load()`: next grading year's unrounded percent of points. Other loaders: next ELA or math achievement, next grade (A=4 ... F=0), not graded next year |
| **n** | 23,373 school-years (3,678 schools) graded A-F in spring 2017-2019 and 2022-2025, with Y observed |
| **Source** | FLDOE School Grades spreadsheets, Guides to Calculating School Grades, and Membership in Florida Public Schools (Survey 2), all from FLDOE's public file CDN |

## The rule

FLDOE's guides describe the rule as follows. Each component is a whole-number percent
of eligible students meeting a criterion: achievement in ELA, math, science and social
studies; learning gains and lowest-25% gains in ELA and math; middle-school acceleration;
graduation rate; and college and career acceleration. Since 2023-24 there is also
grade-3 ELA achievement. A component counts when at least 10 students are eligible.

The percent of points is the sum of the components divided by 100 times their number,
expressed as a whole number. The grade comes from that whole number and the year's
scale for the school's type:

| Spring year | Elementary | Middle | High | Combination | Source |
|---|---|---|---|---|---|
| 2015-2019, 2021, 2022 | 62 / 54 / 41 / 32 | same | same | same | `SchoolGradesCalcGuide{15-19,21,22}.pdf` |
| 2023 (informational baseline, new FAST tests) | 62 / 50 / 33 / 22 | 68 / 58 / 40 / 31 | 70 / 60 / 40 / 23 | 67 / 56 / 35 / 23 | `SchoolGradesCalcGuide23.pdf`; scale printed in each sheet of `SchoolGrades23.xlsx` |
| 2024, 2025 | 62 / 54 / 41 / 32 | 64 / 57 / 44 / 34 | 64 / 57 / 44 / 34 | 64 / 57 / 44 / 34 | Guides 24, 25; scale printed in the spreadsheet |
| 2026 | 62 / 54 / 41 / 32 | 64 / 57 / 44 / 34 | 65 / 60 / 45 / 35 | 65 / 60 / 45 / 35 | Guide 26; scale printed in the spreadsheet |

Each entry is the minimum percent for A / B / C / D; F is everything below D.

- The 2026 high and combination scales rose automatically under s. 1008.34(3)(c)1.,
  F.S., because more than 75% of those schools earned an A or B in 2024-25.
- 2019-20 has no grades (COVID-19).
- 2020-21 grades were optional (Emergency Order 2021-EO-02): only the 634 schools that
  opted in have a grade, and none of them was a D or an F.

The published percent equals round-half-up(points / components) in every row. So
"published percent >= c" is exactly "x >= c - 0.5". The adapter therefore uses the
unrounded x, which takes 2,216 distinct values in the sample, and centres it at
c - 0.5. `extras["group"]` is the (year, type) cell, for example `2024-middle`.
`extras["group_cutoffs"]` maps each cell to its cutoff on x for the loader's boundary.
`extras["cutoffs"]` holds each unit's four cutoffs on x, and `extras["cutoff_table"]`
the published scale. Within the sample, the cutoffs differ by type only in 2023-2025.

The boundary with the most mass is B/C. The number of school-years within 2 points of
each cutoff is 2,713 (A/B), 3,066 (B/C), 1,051 (C/D) and 137 (D/F).

The stakes are the School Recognition Program for an A or a one-grade improvement
(s. 1008.36, F.S.) and improvement or turnaround for D and F schools (s. 1008.33, F.S.).
For the C/D and D/F loaders, the policy treatment is therefore 1 - D.

## Getting the data

    python -m experiments.datasets.fl_school_grades.download          # idempotent
    python -m experiments.datasets.fl_school_grades.download --check  # sizes on disk

All 33 files total 31.8 MB. They are written to `data/raw/` (gitignored).

- **Source.** FLDOE's file CDN. The main www.fldoe.org site returns 403 to scripts, so
  the files were located from web-search results and by their file names. There is no
  login, form or CAPTCHA.
- **Reading.** The environment has neither `xlrd` nor `openpyxl`, so `_spreadsheet.py`
  reads the `.xls` (BIFF8) and `.xlsx` files with the standard library.
- **Cache.** The adapter caches `data/processed/fl_school_grades_panel.csv.gz`, which
  holds all 37,339 school-years from 2015 to 2026. Delete it after changing `adapter.py`.

| File | Bytes | Use |
|---|---:|---|
| https://cdn.fldoe.org/file/18534/SchoolGrades14.xls | 1,497,600 | 2013-14, old model; not used |
| https://cdn.fldoe.org/file/18534/SchoolGrades15.xls | 1,129,984 | 2014-15 baseline (no learning gains); cache only |
| https://cdn.fldoe.org/file/18534/SchoolGrades16.xls | 1,305,600 | 2015-16; cache, lagged Q |
| https://cdn.fldoe.org/file/18534/SchoolGrades17.xls | 2,112,000 | 2016-17 |
| https://cdn.fldoe.org/file/18534/SchoolGrades18.xls | 2,115,072 | 2017-18 |
| https://cdn.fldoe.org/file/18534/SchoolGrades19.xls | 2,145,792 | 2018-19 |
| https://cdn.fldoe.org/file/18534/SchoolGrades21.xls | 2,341,376 | 2020-21 (optional grades); outcome for 2018-19 |
| https://cdn.fldoe.org/file/18534/SchoolGrades22.xlsx | 880,107 | 2021-22 |
| https://cdn.fldoe.org/file/18534/SchoolGrades23.xlsx | 828,469 | 2022-23 baseline, one sheet per school type |
| https://cdn.fldoe.org/file/18534/SchoolGrades24.xlsx | 959,836 | 2023-24 |
| https://cdn.fldoe.org/file/18534/SchoolGrades25.xlsx | 940,291 | 2024-25 |
| https://cdn.fldoe.org/file/18534/SchoolGrades26.xlsx | 961,745 | 2025-26; outcome for 2024-25 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide15.pdf | 852,836 | scale 2014-15 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide16.pdf | 407,364 | scale 2015-16 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide17.pdf | 704,552 | scale 2016-17 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide18.pdf | 641,067 | scale 2017-18 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide19.pdf | 831,744 | scale 2018-19 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide21.pdf | 1,008,778 | scale and opt-in rules 2020-21 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide22.pdf | 1,187,629 | scale 2021-22 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide23.pdf | 677,593 | scales 2022-23 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide24.pdf | 747,024 | scales 2023-24 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide25.pdf | 803,109 | scales 2024-25 |
| https://cdn.fldoe.org/file/18534/SchoolGradesCalcGuide26.pdf | 775,936 | scales 2025-26 |
| https://cdn.fldoe.org/file/7584/1617MembBySchoolByGrade.xlsx | 582,176 | membership 2016-17 |
| https://cdn.fldoe.org/file/7584/1718MembBySchoolByGrade.xlsx | 674,999 | membership 2017-18 |
| https://cdn.fldoe.org/file/7584/1819MembBySchoolByGrade.xlsx | 583,209 | membership 2018-19 |
| https://cdn.fldoe.org/file/7584/1920MembBySchoolByGrade.xlsx | 577,739 | membership 2019-20; not used |
| https://cdn.fldoe.org/file/7584/2021MembBySchoolByGrade.xlsx | 579,512 | membership 2020-21; not used |
| https://cdn.fldoe.org/file/7584/2122MembBySchoolByGrade.xlsx | 581,895 | membership 2021-22 |
| https://cdn.fldoe.org/file/7584/2223MembBySchoolByGrade.xlsx | 591,647 | membership 2022-23 |
| https://cdn.fldoe.org/file/7584/2324MembBySchoolByGrade.xlsx | 583,866 | membership 2023-24 |
| https://cdn.fldoe.org/file/7584/2425MembBySchoolByGrade.xlsx | 595,543 | membership 2024-25 |
| https://cdn.fldoe.org/file/7584/2526MembBySchoolByGrade.xlsx | 578,281 | membership 2025-26 |

Files that do not exist on the CDN (404, checked 2026-10-27):
- `SchoolGrades20` and `SchoolGradesCalcGuide20`, since no grades were issued;
- the 2014-15 and 2015-16 membership files under this name pattern.

## Rule check (2026-10-27)

`adapter.rule_check()` compares the published grade with the grade computed from the
published percent and `CUTOFFS`, for every school with a published A-F grade:

| Spring year | Rows | Graded A-F | I (incomplete) | No grade | Agreement |
|---|---:|---:|---:|---:|---:|
| 2015 | 3,338 | 3,296 | 42 | 0 | 1.000 |
| 2016 | 3,358 | 3,320 | 38 | 0 | 1.000 |
| 2017 | 3,324 | 3,306 | 18 | 0 | 1.000 |
| 2018 | 3,331 | 3,319 | 12 | 0 | 1.000 |
| 2019 | 3,337 | 3,328 | 9 | 0 | 1.000 |
| 2021 | 3,399 | 634 | 0 | 2,765 (no opt-in) | 1.000 |
| 2022 | 3,409 | 3,368 | 41 | 0 | 1.000 |
| 2023 | 3,444 | 3,419 | 25 | 0 | 1.000 |
| 2024 | 3,471 | 3,451 | 20 | 0 | 1.000 |
| 2025 | 3,465 | 3,453 | 12 | 0 | 1.000 |
| 2026 | 3,463 | 3,449 | 14 | 0 | 1.000 |

- Every published A-F grade matches the computed grade (34,343 of 34,343).
- The published percent equals round-half-up(points / components) in every row with a
  percent.
- The scale printed in the 2023-2026 spreadsheets matches `CUTOFFS` for all four types.
- The files mark no school as "under review". Grade `I` rows have no percent and are
  excluded. In the loaders' sample, `extras["eligible_published"]` (the published grade
  is at least the higher grade) agrees with D for all 23,373 school-years.

## Readiness (2026-10-27)

`python -m experiments.scripts.multi_rule_readiness fl_school_grades` reports, for the
B/C boundary:

| Quantity | Value |
|---|---:|
| n, p (continuous X) | 23,373, 88 (3) |
| R^2 of Q on X | 0.522 |
| sd(T), sd(eta) | 8.57, 8.20 |
| sd(eta) max/min across T deciles | 1.33 |
| Treated share (B or better) | 0.614 |
| Within 0.25 sd(eta) of the cutoff | 13.4% |
| Agreement with the published grade | 1.000 |

The 28 (year, type) groups give the following ranges:

| Type | n | R^2 range | Treated-share range |
|---|---:|---|---|
| Elementary | 12,577 | 0.57-0.68 | 0.52-0.67 |
| Middle | 3,916 | 0.71-0.74 | 0.49-0.70 |
| High | 3,388 | 0.55-0.66 | 0.55-0.83 |
| Combination | 3,492 | 0.45-0.54 | 0.65-0.77 |

These per-group fits use all 88 columns on 400-1,800 schools, so they are optimistic.
Pooled by type, R^2 is 0.55, 0.68, 0.56 and 0.45.

The other boundaries use the same X, n and outcome:

| Boundary | Treated share | R^2 | sd(eta) | sd(eta) max/min | Near cutoff |
|---|---:|---:|---:|---:|---:|
| A/B (`load_ab`) | 0.350 | 0.531 | 8.18 | 1.35 | 11.9% |
| C/D (`load_cd`) | 0.949 | 0.535 | 8.20 | 1.40 | 4.6% |
| D/F (`load_df`) | 0.995 | 0.548 | 8.26 | 1.45 | 0.6% |

Other facts about the fit:
- Without the district indicators, R^2 is 0.43.
- Adding the previous year's x (`extras["q_lag"]`) raises R^2 to 0.72 and lowers sd(eta)
  to 6.2. It is not in X, because it depends on the previous year's grade.

**Two-score view: achievement against learning gains.** The file supports it wherever
learning gains exist: 2016-2019, 2021, 2022 and 2024-2026. It does not exist for 2015 or
for the 2023 baseline.

The weights are fixed only within a component set. `load_two_score()` takes elementary
schools graded on exactly seven components (ELA, math and science achievement plus the
four gains components) in 2017-2019 and 2022, n = 6,940. For them
x = (3 Q1 + 4 Q2) / 7, where Q1 is the achievement mean and Q2 the gains mean. The rule
reproduces the published grade for all of them.

| Quantity | Achievement | Gains |
|---|---:|---:|
| R^2 | 0.715 | 0.349 |
| sd(T), sd(eta) | 13.3, 8.4 | 5.9, 8.1 |
| sd(eta) max/min across T deciles | 1.41 | 1.08 |

- corr(T1, T2) = 0.71 and corr(eta1, eta2) = 0.53.
- The weakest direction of T is (0.33, -0.94). Its sd is 3.9, 0.48 of sd(eta), so the
  trade-off between the two scores is better identified than in `chile_two_score`
  (0.18).
- 2024-2026 elementary schools have eight components (grade-3 ELA added), with weights
  (1/2, 1/2). Middle, high and combination schools add acceleration and graduation
  components, so a third score would be needed. Neither is implemented.

## Outcome

`load()` uses Y = the next grading year's unrounded percent of points. Its mean is 57.9,
its sd 12.7, and corr(Y, Q) = 0.78. The outcome is matched by district, school and
virtual-provider number.

Of the 23,644 school-years graded A-F in the sample years, 271 (1.1%) have no Y:
- 252 have no row in the next year's file (closure, merger or renumbering);
- 19 have an `I` grade next year.

The outcome year depends on the grading year:
- 2018-19 is followed by 2020-21 (no 2019-20 grades). Its percent of points is published
  for every school, including those that did not opt in.
- 2021-22 is followed by 2022-23, which used achievement components only on the new
  FAST tests.
- `extras["y_year"]` records the outcome year.

Other loaders:
- `load_next_ela` (n = 23,309) and `load_next_math` (n = 23,301).
- `load_next_grade` (n = 20,696). The 2018-19 rows mostly drop out, because 2020-21
  grades exist only for opt-in schools.
- `load_closed`: Y = no row next year. The mean is 1.1%, n = 23,644.

## Caveats

- **Scores and units.**
  - Q is a school aggregate: a fixed-weight sum of rounded percentages of students. Its
    noise includes cohort sampling noise, which is larger in small schools, so eta is
    heteroskedastic (sd(eta) max/min 1.33).
  - x lies on a grid of fractions k / (number of components).
  - Schools repeat across up to seven years (3,678 schools), so inference should cluster
    by school.
- **The score changes over the years.**
  - 2023 is an informational baseline on new tests, scored on achievement only.
  - Grade-3 ELA was added in 2024.
  - The cutoffs for each type changed in 2023, 2024 and 2026.
  - Year indicators absorb level shifts only. `extras["informational"]` flags 2023.
- **Covariate timing.**
  - % economically disadvantaged and % minority describe the same school year's
    students. They are fixed before the spring tests and the summer grade release.
  - The 2024-25 file has no % minority, so the school's latest earlier value (2023-24)
    is used. 37 rows remain missing and are flagged.
  - Membership is the October (Survey 2) count of the same school year. Counts below 10
    are masked in the file and set to 5 (15 rows).
- **Exclusions.**
  - 2020-21 is excluded as a treatment year: opting in was a choice made with the score
    known, and no D or F school opted in.
  - 2014-15 and 2015-16 are excluded for lack of membership files; 2014-15 also has no
    learning gains. Both are in the cache.
  - 2025-26 has no outcome yet.
  - Few alternative/ESE center schools take a grade: 6 school-years in the sample, so
    `alt_ese` is nearly constant.
- **The treatment and the method's assumptions.**
  - D is the published grade, which is sharp in Q. The consequences differ by boundary
    and over time.
  - Because a school's grade feeds its next score through the stakes, `q_lag` is
    excluded from X.
  - Like every real dataset screened so far, X independent of eta holds only
    approximately.
