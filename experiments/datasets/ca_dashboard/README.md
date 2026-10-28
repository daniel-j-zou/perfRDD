# California School Dashboard: a 5 x 5 decision matrix of status and change (ELA and math)

| Field | Value |
|---|---|
| **Q** | `load_two_score()`, `load_math_two_score()`: (status relative to the Very Low / Low boundary, change), shape (n, 2), in DFS points. `load()`, `load_math()`: the same centred status, shape (n,) |
| **Threshold** | Two-score: (-0.05, 2.95). Status is centred at the lowest Low status of the school's cut-score set (ELA: -70.0 for grades 3-8 cut scores, -45.0 for grade 11; math: -95.0, -115.0). Both scores are on a 0.1 grid. Single score: the unit's cutoff is min(-0.05, prior + 2.95) (`extras["cutoff"]`) |
| **Treatment** | D = 1 if the school's all-students indicator color is **Red**: status level Very Low and change level Declined Significantly, Declined or Maintained. Two-score: `1{Q1 < -0.05} * 1{Q2 < 2.95}`. Single score: `1{Q < min(-0.05, prior + 2.95)}`. `level="red_orange"`: Red or Orange |
| **Rule class** | A 5 x 5 table of status level x change level. Red is a quadrant (a "both" rule, both scores below a cutoff). Red or Orange is a staircase: Very Low with any change, or Low with change below +3.0. The single-score view is a cutoff on status that moves with the prior status |
| **X** | From the Census Day enrollment file of the same school year: % socioeconomically disadvantaged, % English learners, % students with disabilities, % foster youth, % homeless, Black / American Indian / Asian / Filipino / Hispanic / Pacific Islander / multiracial shares, log enrollment; the district's log enrollment and % socioeconomically disadvantaged; charter, high-school cut scores, census-missing flags; year and county indicators (82 columns, 15 continuous). `load()` and `load_math()` add the prior status (83) |
| **Y** | The next Dashboard year's status on the same indicator (DFS points), matched by the 14-digit CDS code. For 2019 the next year is 2022 |
| **n** | ELA 49,451 school-years (8,989 schools), 3,359 Red. Math 49,442 (8,987), 3,657 Red. Dashboards 2017 (Fall), 2018, 2019, 2023, 2024 and 2025. The single-score and two-score views keep the same units |
| **Source** | CDE California School Dashboard academic-indicator and census-enrollment data files 2017-2026 (www3.cde.ca.gov/researchfiles/cadashboard/) and the 2023, 2024 and 2026 Dashboard Technical Guide mini-guides for the academic indicators |

**Status** is the average Distance from Standard (DFS). It is the mean distance of the
continuously enrolled test takers' Smarter Balanced scale scores from the lowest Level 3
("Standard Met") score, for grades 3-8 and 11. Students who took the CAA get a
substituted score. When participation is below 95%, Lowest Obtainable Scale Scores
(LOSS) are added. From 2026 the measure is called "Distance from Proficiency"; the scale
is unchanged. **Change** is the status minus the prior year's status, both as CDE
computes them in the current year. Both are published to a tenth of a point.

## The rule

The academic-indicator mini-guides of the Dashboard Technical Guide (2023, 2024, 2026:
"Status Cut Scores, Change Cut Scores, and Five-by-Five Colored Tables") give the table,
with color codes 1 = Red, 2 = Orange, 3 = Yellow, 4 = Green and 5 = Blue:

| Status level | Declined Significantly (<= -15.1) | Declined (-15.0 to -3.0) | Maintained (-2.9 to +2.9) | Increased (+3.0 to +14.9) | Increased Significantly (>= +15.0) |
|---|---|---|---|---|---|
| Very High | Green | Green | Blue | Blue | Blue |
| High | Green | Green | Green | Green | Blue |
| Medium | Yellow | Yellow | Yellow | Green | Green |
| Low | Orange | Orange | Orange | Yellow | Yellow |
| Very Low | **Red** | **Red** | **Red** | Orange | Orange |

The status cut scores give the lowest status of the Low, Medium, High and Very High
levels:

| Cut-score set | Low | Medium | High | Very High | Source |
|---|---:|---:|---:|---:|---|
| ELA, grades 3-8 | -70.0 | -5.0 | +10.0 | +45.0 | guides 2023, 2024, 2026; all files |
| ELA, grade 11 | -45.0 | 0.0 | +30.0 | +75.0 | files 2018-2026 |
| Math, grades 3-8 | -95.0 | -25.0 | 0.0 | +35.0 | guides 2023, 2024, 2026; all files |
| Math, grade 11 | -115.0 | -60.0 | 0.0 | +25.0 | files 2018-2026 |

- **Which cut scores apply.** High schools (grades 7-12) and high-school districts use the
  grade 11 cut scores. Elementary, middle and K-12 schools and elementary and unified
  districts use the grades 3-8 ones (guide Table 7; `hscutpoints` in the files).
- **Source of the grade 11 cut scores.** The guides refer to CDE's "Five-by-Five Colored
  Tables" web page for them, which is blocked (see below). They are read off the files
  instead: each status level's minimum and maximum published status sit on either side
  of one 0.1 step at these values, in every year.
- **Red cells**, as (status level, change level) counting from Very Low and from
  Declined Significantly: (1, 1), (1, 2), (1, 3). That is, Very Low status (at or below
  -70.1 for ELA grades 3-8) and a change of +2.9 or less.
- **The same every year.** The table, the change cut points and the status cut points
  are identical in 2017, 2018, 2019, 2023, 2024, 2025 and 2026. The guides give them for
  2023, 2024 and 2026. The published levels and colors confirm every cell and cut point
  in the 2017-2026 files (rule check below). There is one exception:
  - **2019 DASS schools** (Dashboard Alternative School Status) had their own Very Low
    cut. It falls between -125.8 and -125.0 for ELA grades 3-8, -110.3 and -110.0 for
    ELA grade 11, about -175 for math grades 3-8 and about -185 for math grade 11. DASS
    schools are excluded from the samples.
- **Color requires n.** A school gets a color only with 30 or more students in both the
  current and the prior year's DFS denominator (`currnsizemet` and `priornsizemet` from
  2023). With 11-29 students the status is published without a color.
- **Pair and share.** Schools with no tested grades (TK-2) are assigned the grade 3 DFS
  of a feeder school, the district or the county (`pairshare_method`). Their status is
  not their own, so they are excluded.
- **No-test Orange.** In 2018 and 2019, 31 and 37 schools (ELA) that did not test were
  published Orange with no status or change (`notestflag` in 2019). They are excluded.

**Centring.** Q1 = status - (lowest Low status of the school's cut-score set). The Very
Low / Low boundary is then 0 for every school (Very Low is Q1 <= -0.1). So the Red rule
`1{Q1 < -0.05} * 1{Q2 < 2.95}` is the same for grades 3-8 and grade 11 schools.

The other boundaries differ by cut-score set. `extras` holds the full table:

| Key | Content |
|---|---|
| `matrix_colors`, `matrix_codes` | the 5 x 5 table, rows Very Low to Very High, columns Declined Significantly to Increased Significantly |
| `red_cells`, `orange_cells` | the cells as (status level, change level), 1-based |
| `status_level_min` | lowest centred status of levels 2-5: grades 3-8 `(0, 65, 80, 115)` (ELA) or `(0, 70, 95, 130)` (math); grade 11 `(0, 45, 75, 120)` (ELA) or `(0, 55, 115, 140)` (math) |
| `status_level_min_raw` | the same in DFS points |
| `status_boundaries` | `status_level_min` - 0.05 |
| `change_level_min`, `change_boundaries` | `(-15.0, -2.9, 3.0, 15.0)` and `(-15.05, -2.95, 2.95, 14.95)` |
| `cut_set`, `unit_status_level_min` | the unit's set (`k8` or `hs`) and its row of cuts |
| `status_offset` | the centring constant |

`level="red_orange"` uses the unit's Low / Medium cut, so its rule holds the per-unit
cuts and applies only to the sample's n units.

**What Red does.**
- It is the color published on the Dashboard for the indicator.
- For districts, a student group that is Red in two or more LCFF priority areas makes
  the district eligible for Differentiated Assistance.
- Red feeds school identification: CSI low-performing (based on all of a school's
  indicator colors) and ATSI (student groups).
- D here is the all-students color on one indicator. The consequences of a single Red
  depend on the school's other indicators.

**Red rather than Red or Orange.** Red is the color named by the DA and CSI / ATSI
rules, and it is a single region. Red or Orange (`level="red_orange"`; 35% treated for
ELA) is also provided. It is the staircase variant: status Very Low, or status Low and
change below +3.0.

## 2022 (status only)

The 2022 Dashboard published status and status levels with no change and no colors,
because 2021 had no comparable tests. Handling:
- 2022 is not a treatment year in the main loaders.
- It is the outcome year for 2019.
- Its status (as recomputed by CDE) is the prior status inside the 2023 file.

`load_status_2022()` is a separate single-cutoff sample:
- D = Very Low status level in 2022 (`1{Q < -0.05}`), Y = the 2023 status.
- ELA n = 8,779 with 1,191 Very Low; math n = 8,778 with 1,688.
- 2022 status levels are published only for schools with 30 or more students. In 2022,
  the lowest status level played the role of Red for DA and CSI.

## Single-score view (`load()`, `load_math()`)

Change is status minus prior status, and the prior status is predetermined. So with
Q = status and the prior status P in X:

    Red  <=>  Q < -0.05 and Q - P < 2.95  <=>  Q < min(-0.05, P + 2.95).

The cutoff on status is the Very Low line, except for schools whose prior was below
-3.0. For those, the cutoff is P + 2.95, which is lower. That holds for 9.1% of ELA
units. `treatment_rule(Q)` evaluates the table at (Q, Q - P) with the unit's P, so it
applies to the sample's n units.

**P is the prior status implied by the published change**, P = status - change
(`extras["prior_status"]`, centred like Q).
- CDE rounds status and change separately: the change is evidently rounded after
  differencing the unrounded statuses.
- So the published change differs by 0.1 from the difference of the two published
  statuses for 24.7% of ELA units (12,196 of 49,451).
- With the implied P, the rule reproduces every published color.
- With the published prior status (`extras["prior_status_published"]`), it misses 2 ELA
  and 6 math units, all at a change of 2.9 / 3.0.
- The implied P differs from the published prior by at most 0.1.

**The two-score view leaves P out of X** (`load_two_score()`). With P in X, the change
would be status minus a covariate, and the two noises would be identical.

## Getting the data

    python -m experiments.datasets.ca_dashboard.download          # idempotent
    python -m experiments.datasets.ca_dashboard.download --check  # sizes, hashes

All files were downloaded on 2026-10-28 with Python's default `urllib` user agent, with no
login, form, challenge or browser headers. The 29 files total 439.0 MB and go to
`data/raw/` (gitignored). `download.py` records each file's SHA-256 and reports a
difference.

| File | Bytes | Use |
|---|---:|---|
| https://www3.cde.ca.gov/researchfiles/cadashboard/ela38download2017f.txt | 13,897,202 | ELA 2017 (Fall), grades 3-8 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/ela11download2017f.txt | 2,715,050 | ELA 2017 grade 11: status and change, no levels or colors (not used) |
| https://www3.cde.ca.gov/researchfiles/cadashboard/math38download2017f.txt | 13,915,383 | math 2017, grades 3-8 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/math11download2017f.txt | 2,733,239 | math 2017 grade 11 (not used) |
| https://www3.cde.ca.gov/researchfiles/cadashboard/eladownload2018.txt | 17,950,706 | ELA 2018 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/mathdownload2018.txt | 17,994,474 | math 2018 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/eladownload2019.txt | 18,581,730 | ELA 2019 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/mathdownload2019.txt | 18,633,505 | math 2019 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/eladownload2022.txt | 19,315,757 | ELA 2022 (status only) |
| https://www3.cde.ca.gov/researchfiles/cadashboard/mathdownload2022.txt | 19,400,865 | math 2022 (status only) |
| https://www3.cde.ca.gov/researchfiles/cadashboard/eladownload2023.txt | 25,622,839 | ELA 2023 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/mathdownload2023.txt | 25,947,723 | math 2023 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/eladownload2024.txt | 26,721,809 | ELA 2024 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/mathdownload2024.txt | 27,060,824 | math 2024 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/eladownload2025.txt | 26,632,258 | ELA 2025 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/mathdownload2025.txt | 26,972,402 | math 2025 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/eladownload2026.txt | 26,506,645 | ELA 2026 (outcome for 2025) |
| https://www3.cde.ca.gov/researchfiles/cadashboard/mathdownload2026.txt | 26,825,661 | math 2026 (outcome for 2025) |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2017.txt | 4,706,566 | X, 2017 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2018.txt | 10,882,919 | X, 2018 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2019.txt | 10,806,513 | X, 2019 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2022.txt | 10,751,338 | X, 2022 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2023.txt | 10,720,623 | X, 2023 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2024.txt | 10,768,321 | X, 2024 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2025.txt | 10,723,144 | X, 2025 |
| https://www3.cde.ca.gov/researchfiles/cadashboard/censusenrollratesdownload2026.txt | 10,680,531 | X, 2026 |
| https://www.cde.ca.gov/ta/ac/cm/documents/dbguideacad23.docx | 579,638 | rule, 2023 guide |
| https://www.cde.ca.gov/ta/ac/cm/documents/dbguideacad24.docx | 565,835 | rule, 2024 guide |
| https://www.cde.ca.gov/ta/ac/cm/documents/dbguideacad26.docx | 343,993 | rule, 2026 guide |

- **The indicator files.** Each file is tab-delimited, with one row per state, county,
  district and school and per student group. The columns are status, prior status,
  change, the two denominators, status level, change level, color, `box` (the cell),
  and the flags (charter, COE, DASS, high-school cut scores, pair and share,
  participation).
- **The census files.** Each gives total enrollment and the share of each student group
  per school and district, on Census Day (October of the school year). That is before the
  spring tests that set that year's status.
- **The 2026 files** have Last-Modified 2026-09-23 and carry colors. They serve only as
  the outcome for 2025.
- **Cache.** The adapter caches `data/processed/ca_dashboard_panel.csv.gz`, which holds
  all 153,548 school-year-subject rows of 2017-2026 for all students. It is written
  atomically (a temporary file, then `os.replace`). Delete it after changing
  `adapter.py`.

**Blocked: the CDE web pages.**
- Every `.asp` page under https://www.cde.ca.gov/ta/ac/cm/ answers a script with `302`
  to a Radware bot-manager challenge (`https://validate.perfdrive.com/?...`). This
  includes https://www.cde.ca.gov/ta/ac/cm/ itself, the data-file pages
  https://www.cde.ca.gov/ta/ac/cm/acaddatafiles.asp and
  https://www.cde.ca.gov/ta/ac/cm/datafilesfall17.asp, the record layouts, and the
  five-by-five color-table page https://www.cde.ca.gov/ta/ac/cm/fivebyfivecolortables.asp.
  The scripts here do not use them.
- The brief's guess https://www.cde.ca.gov/ta/ac/cm/acadindcdata.asp also redirects to
  the challenge.
- The static files on www3.cde.ca.gov and under `/ta/ac/cm/documents/` are not behind
  the challenge.
- The file names come from a web search and from the Internet Archive's public URL index
  of `www3.cde.ca.gov/researchfiles/cadashboard/`.
- The bare directory listing https://www3.cde.ca.gov/researchfiles/cadashboard/
  redirects to a WAF alert page (`https://www6.cde.ca.gov/wafalert.html`). So do `HEAD`
  requests for the files; `GET` works.

**Missing: the older guides.** These guides return 404 at CDE, as archived documents:
- https://www.cde.ca.gov/ta/ac/cm/documents/dashboardguide19.pdf
- https://www.cde.ca.gov/ta/ac/cm/documents/dashboardguide18.pdf
- the 2017 guide
- https://www.cde.ca.gov/ta/ac/cm/documents/dbguideacad22.docx
- https://www.cde.ca.gov/ta/ac/cm/documents/dbguideacad25.docx

For those years the rule rests on the published levels and colors in the data files.
These agree with the 2023+ table everywhere (rule check below). An attempt to fetch
archived copies from the Internet Archive failed (`503 Temporarily Offline`).

## Rule check (2026-10-28)

`adapter.rule_check()` uses all-students school rows. It compares the published status
level, change level and color with the levels and color computed from the published
status, change and denominators. The comparison uses the cut points above, the table, and
the rule "30 or more students in both denominators".

| Year | Schools | Published color | Red | DASS | Pair and share | No-test Orange | Status level | Change level | Color from levels | Full rule (incl. no color) | Red |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 7,597 | 7,236 | 498 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2018 | 9,807 | 8,723 | 466 | 868 | 91 | 31 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2019 | 9,845 | 8,751 | 308 | 876 | 91 | 37 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2022 | 9,925 | - | - | 898 | 83 | 0 | 1.000 | - | - | - | - |
| 2023 | 9,933 | 8,915 | 1,043 | 873 | 83 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2024 | 9,907 | 8,914 | 951 | 862 | 78 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2025 | 9,887 | 8,904 | 628 | 862 | 78 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2026 | 9,873 | 8,873 | 690 | 852 | 95 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The table is for ELA. Math gives the same agreement of 1.000 in every column and year,
with 391, 606, 471, 1,089, 956, 758 and 771 Red in 2017-2026.

- **Status level.** Agreement covers every row with a published level except the 2019
  DASS schools. Those agree only at 0.61 (ELA) and 0.69 (math), because of their own
  Very Low cut.
- **"Color from levels"** applies the table to the published levels, for colored rows
  outside the exceptions.
- **"Full rule"** covers every row outside the exceptions, including the 0 = no color
  rows from the n-size rule.
- **The loaders' D** equals the published Red for every unit: 49,451 of 49,451 (ELA) and
  49,442 of 49,442 (math). The same holds for Red or Orange.

**Sample.**

| Step (Dashboards 2017-2019, 2023-2025) | ELA | Math |
|---|---:|---:|
| all-students school rows | 56,976 | 56,975 |
| published color 1-5 (drops n < 30 in either year, no prior, 2017 grade 11) | 51,443 | 51,435 |
| not DASS | 50,238 | 50,228 |
| not pair and share | 49,928 | 49,918 |
| not no-test Orange | 49,869 | 49,859 |
| Y observed | 49,451 | 49,442 |

| Year | Outcome year | ELA n | ELA Red | Math n | Math Red | High-school cut scores |
|---|---|---:|---:|---:|---:|---:|
| 2017 | 2018 | 7,171 | 488 | 7,170 | 382 | 0 (grade 11 had no colors) |
| 2018 | 2019 | 8,342 | 348 | 8,334 | 470 | 1,176 |
| 2019 | 2022 | 8,260 | 241 | 8,257 | 370 | 1,171 |
| 2023 | 2024 | 8,566 | 921 | 8,565 | 953 | 1,208 |
| 2024 | 2025 | 8,574 | 833 | 8,576 | 828 | 1,215 |
| 2025 | 2026 | 8,538 | 528 | 8,540 | 654 | 1,212 |

`cut_set="k8"` keeps the grades 3-8 schools (ELA 43,469, 2,879 Red). `cut_set="hs"`
keeps the high schools (5,982, 480 Red).

Of the 49,869 eligible ELA school-years, 418 have no Y:
- 386 have no row in the next year's file;
- 7 are pair and share in the outcome year;
- 25 have a row but no status.

## Fit (ELA, two-score X)

| Quantity | Status (centred) | Change |
|---|---:|---:|
| R^2 of Q on X | 0.710 | 0.030 |
| sd(T), sd(eta) | 40.1, 25.4 | 2.6, 14.5 |
| corr(T1, T2), corr(eta1, eta2) | -0.07 | 0.25 |

- **Math** gives R^2 0.695 and 0.019, sd(eta) 29.8 and 13.4, and corr(eta) 0.25.
- **Within a year**, X explains about 1% of the change (R^2 -0.004 to 0.035). Change is
  nearly unpredictable from school characteristics.
- **`load()`** (with the prior status in X) gives an R^2 of status of 0.916 and
  sd(eta) of 13.8.
- **Outcome.** Y has mean -9.7 and sd 48.4 for ELA (math: -38.4 and 55.6).
  corr(Y, status) is 0.94 (math 0.95).

**Mass near the Red corner (ELA).**
- 4,629 school-years have centred status within 10 points of the Very Low line.
- 1,221 are Very Low with a change within 5 points of +3.0.
- 1,393 are within 10 points of the status line and 5 points of the change line.

## Student groups (extension)

The files also give status, change and color for each student group in each school.
Examples are socioeconomically disadvantaged, English learners, students with
disabilities, race groups, homeless, foster youth and, from 2024, long-term English
learners. `adapter.student_group_rows()` counts the school-level student-group rows with
a published color:

| Year | 2017 | 2018 | 2019 | 2023 | 2024 | 2025 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| ELA rows | 31,316 | 36,388 | 37,250 | 37,271 | 47,910 | 39,323 | 229,458 |
| ELA Red | 4,618 | 4,258 | 3,214 | 7,880 | 9,053 | 5,570 | 34,593 |
| Math rows | 31,272 | 36,312 | 37,179 | 37,254 | 47,925 | 39,472 | 229,414 |

That is about 4.6 times the all-students n, before the DASS, pair-and-share and outcome
filters. The groups use the school's cut scores and the same table. Red for a student
group feeds ATSI. Group-level X would come from the same census files.

## Suggested policy box

The policy is a pair of cutoffs (s, c): D = 1{Q1 < s} * 1{Q2 < c}, in centred status
and change points. The deployed rule is (s, c) = (0, 3.0), with the 0.05 margins of the
0.1 grid. Suggested box:

- **Status s from -20 to +30.**
  - For ELA grades 3-8 that is a status of -90 to -40. For grade 11 it is -65 to -15.
  - The box stays inside the Very Low and Low levels: the Low / Medium line is at
    +45 to +70 centred, depending on the cut-score set.
  - At s = -20 the treated share is still 2.4% at the deployed c (1,188 units). Below
    that the corner thins out: 1.3% at (-30, 3).
- **Change c from -15 to +15.** These are the Declined Significantly and Increased
  Significantly cut points. The box covers 78% of the change distribution.

Treated share for ELA at the corners of a wider grid:

| s \ c | -15 | -3 | +3 | +15 |
|---|---:|---:|---:|---:|
| -30 | 0.006 | 0.011 | 0.013 | 0.016 |
| 0 | 0.021 | 0.052 | 0.068 | 0.088 |
| +30 | 0.042 | 0.129 | 0.187 | 0.269 |

## Caveats

- **The change score is almost unpredictable from X** (R^2 0.03; sd(T2) 2.6 against
  sd(eta2) 14.5). The change direction of any better rule rests on the noise, not on
  differences between schools in X. The prior status would predict it, but it cannot
  enter X in the two-score model (see above).
- **Recomputed prior status.** The prior status in year t's file is CDE's recomputation
  under year t's rules (enrollment, participation and LOSS). It equals the previous
  year's published status for:
  - 93.6% of matched schools in 2018 against 2017;
  - 60.2% in 2019 against 2018;
  - 87.0% in 2023 against 2022, where every difference is negative;
  - more than 99.9% in 2024 and 2025;
  - 97.9% in 2026.

  The rule uses the recomputed prior, so the loaders do too. Y is the next year's
  published status, not the next file's recomputed prior.
- **Rounding.** Status and change are rounded separately. See the single-score section.
- **Years.**
  - 2020 and 2021 have no Dashboard. The 2019 outcome is the 2022 status, three years
    and the COVID-19 closures later. The 2023 change is measured against 2022.
  - 2017 has grades 3-8 schools only.
  - 2022 has no colors.
  - The year indicators absorb level shifts only. The ELA Red share runs from 2.9% (2019)
    to 10.8% (2023).
- **Small n.**
  - Schools with fewer than 30 students in either year get no color and are dropped.
  - Status with 11-29 students is published but is not in the sample.
  - The 2019 Red count is small: 241 ELA, 370 math.
- **Floors.** Ten ELA and 15 math units sit at the lowest obtainable DFS (-333.0 /
  -348.0), all LOSS.
- **Exclusions.**
  - DASS (alternative) schools: their own cut in 2019, and a different population.
  - Pair-and-share schools.
  - No-test Orange schools.
  - No colored all-students school row carries the county-office (`coe_flag`) flag, so
    that indicator is constant in the sample and dropped from X.
- **Covariate timing.** The census shares are Census Day (October) counts of the same
  school year, before the spring tests.
  - The socioeconomically disadvantaged share is not used in the rule.
  - Same-year test participation, CAASPP results in other subjects and the prior status
    (two-score view) are left out of X.
- **Units.** Schools repeat across up to six years (8,989 schools in 959 districts and
  58 counties), so inference should cluster by school. Districts share a DA status,
  which is worth considering for clustering.
- **Treatment content.** A single Red is a public signal. Funding and intervention
  follow from Red in combination with other indicators: DA for districts, CSI / ATSI for
  schools. So D is a bundle of public reporting and a step toward identification.
