# North Carolina School Performance Grades: a fixed-weight sum of two scores, and the low-performing designation

| Field | Value |
|---|---|
| **Q** | (achievement score A, growth score G), shape (n, 2), in score units. Both are on 0-100, rounded to a tenth; G is the EVAAS growth index converted to 100 points (range 50-100) |
| **Threshold** | `load()`: 0.8 A + 0.2 G >= 69.5, the published B cutoff of 70 minus 0.5. Other boundaries: 84.5 (A/B), 54.5 (C/D), 39.5 (D/F). `load_low_performing()`: 54.5 on the index and 85 on G |
| **Treatment** | `load()`: `1{0.8 Q1 + 0.2 Q2 >= 69.5}`, a grade of B or better rather than C. `load_ab`, `load_cd`, `load_df`: the other boundaries, with D = the higher grade. `load_low_performing()`: `1{0.8 Q1 + 0.2 Q2 < 54.5} * 1{Q2 < 85}`, the designation itself |
| **Rule class** | a fixed-weight sum (0.8, 0.2) of two scores with grade bands; and a two-cutoff rule (a half-plane in the weighted index intersected with a half-plane in G) for the low-performing designation. The cutoffs are the same for every school and year |
| **X** | % economically disadvantaged; Black, Hispanic, Asian, American Indian and multiracial shares; English-learner and disability shares; log final ADM and its change from the prior year; % inexperienced teachers; charter, Title I, calendar, school type and level; county and year indicators (129 columns; 11 continuous) |
| **Y** | `load()`: next grading year's published SPG score. Other loaders: next achievement, next growth, next low-performing designation, closure |
| **n** | 12,425 school-years (2,617 schools) graded A-F with both scores in spring 2018, 2019 and 2022-2024, with Y observed |
| **Source** | NC DPI School Report Card (SRC) data set, 2024-25 release (panel of all years), and DPI's Technical Guide for School Accountability, all from www.dpi.nc.gov |

## The rule

G.S. 115C-83.15(d) and the DPI technical guide (section 2.1) define the grade. The
achievement score A is a composite: the numerators of all achievement indicators
(reading and math, science, English-learner progress and, in high schools, graduation
rate, ACT/WorkKeys, Math 3 and Biology) are summed and divided by the summed
denominators, then rounded to a tenth. The growth score G is the EVAAS composite growth
index, clipped to [-10, 10], converted to 100 points and rounded to a tenth. The SPG
score is 0.8 A + 0.2 G, or A alone when the school has no growth score, rounded to a
whole number. The grade is

| Grade | SPG score | Cutoff on 0.8 A + 0.2 G |
|---|---|---|
| A | 85-100 | >= 84.5 |
| B | 70-84 | >= 69.5 |
| C | 55-69 | >= 54.5 |
| D | 40-54 | >= 39.5 |
| F | 39 or less | below 39.5 |

This 15-point scale is in the statute and applies to every year in the data, 2013-14 to
2024-25. The statute also bars plus or minus designations. 2014-15 to 2016-17 still
published "A+NG" (A with no significant achievement gaps), which the adapter reads as
A.

Growth status comes from the index: Exceeded at 2.0 or above, Met from -2.0 to 1.99, and
Not Met below -2.0. In every row of the data (2014-2025), Exceeded is exactly G >= 85.0,
Met is 70.0 <= G <= 84.9 and Not Met is G <= 69.9.

**Half-point rounding.** The published SPG score is round-half-up(0.8 A + 0.2 G), so
"published score >= c" is exactly "0.8 A + 0.2 G >= c - 0.5". A and G lie on a 0.1 grid,
so the index lies on a 0.02 grid and can end in exactly .50. In 2018-2025 it does so
for 296 two-score school-years, and DPI rounds every one of them up. In the loaders'
sample, 14 school-years sit exactly on a grade cutoff (2 at A/B, 7 at B/C, 2 at C/D and
3 at D/F). The adapter therefore uses the unrounded A and G and
treats an index exactly at c - 0.5 as reaching the grade. The treatment rules allow
1e-6 of floating-point slack, and the cache computes the index in integer hundredths
(8 A10 + 2 G10).

**Low-performing schools.** G.S. 115C-105.37(a) defines them as schools that "earn an
overall school performance grade of D or F and a school growth score of 'met expected
growth' or 'not met expected growth'". The technical guide (section 9.8) codes this as
grade D or F and growth status Met or Not Met. In the scores this is

    LP = 1{0.8 A + 0.2 G < 54.5} * 1{G < 85}.

A school escapes the designation if its index reaches 54.5 or its growth score reaches
85. The two boundaries meet at (A, G) = (46.875, 85). The growth arm binds only for
schools with A below 46.875, which can have a D or F and still exceed growth. In the
sample, 517 of the 3,814 D or F school-years exceeded growth and were not designated.
Schools without a growth status are never designated. Designations are recorded for
2018 on (`rcd_acc_lowperf`). The 2019-20 and 2020-21 rows repeat the 2018-19 list,
because no grades were issued in those years.

Other features of the data:
- **K-2 schools** have no tested grades. They inherit the SPG of the school that
  receives most of their students (`k2_feeder = Y`, about 40 a year, with no A or G of
  their own) and are excluded.
- **Alternative schools** in the modified accountability model (`asm_option = Y`, about
  95 a year) have no published grade and are excluded.
- **Masked grades**: 20-23 schools a year in 2023-2025 have the grade masked for
  insufficient data and are excluded.
- **Schools without a growth score** (13-31 a year) are graded on A alone. They are off
  the two-score rule and are excluded, which drops 119 school-years in the sample years.
- **Grades by year**: grades exist for 2013-14 to 2018-19 and 2021-22 to 2024-25.
  2019-20 and 2020-21 have none (COVID-19); the SRC tables have no rows for them.

The stakes are low-performing status and its improvement plans (G.S. 115C-105.37,
-105.39). The A-F grade itself is published on each school's report card. The B/C
boundary carries no statutory consequence beyond the published letter. The C/D boundary
is the low-performing boundary for schools that did not exceed growth.

The boundary with the most mass is B/C. The number of school-years in the loaders'
sample within 2 points of each cutoff (`adapter.boundary_mass()`) is 408 (A/B), 1,203
(B/C), 1,162 (C/D) and 435 (D/F). B/C leads C/D at every width from 0.5 to 5 points, by
1-5%.

## Getting the data

    python -m experiments.datasets.nc_school_grades.download          # idempotent
    python -m experiments.datasets.nc_school_grades.download --check  # sizes on disk

All four files total 272.7 MB. They are written to `data/raw/` (gitignored).

- **Source.** www.dpi.nc.gov, with no login, form or CAPTCHA. Scripts were not blocked.
- **The SRC release is a panel.** One release holds every year of each table:
  `rcd_acc_spg1` covers 2014-2017 and `rcd_acc_spg2` covers 2018 on. DPI replaces the
  release each July, so the URLs below point at the 2024-25 release (zip dates
  2026-07-15) and may change.
- **Reading.** The environment has no `openpyxl`, so the adapter reads the `.xlsx`
  members straight from the zips with the standard-library reader in
  `experiments/datasets/fl_school_grades/_spreadsheet.py` (imported, not copied). It
  streams `rcd_acc_pc.txt` (1.06 GB uncompressed) in chunks and never extracts it.
- **Cache.** The adapter caches `data/processed/nc_school_grades_panel.csv.gz` (3.3 MB).
  The cache holds all 26,277 school-level SPG rows from 2014 to 2025, and building it
  takes about 30 s. Delete it after changing `adapter.py`.

| File | Bytes | Use |
|---|---:|---|
| https://www.dpi.nc.gov/src-data-set-2024-251-2/open (`src_datasets_2425_1_of_2.zip`) | 202,420,397 | SPG, growth, low-performing, EDS, ADM, test counts |
| https://www.dpi.nc.gov/src-data-set-2024-252-2/open (`src_datasets_2425_2_of_2.zip`) | 69,310,425 | school directory, teacher tables |
| https://www.dpi.nc.gov/srcdatadictionary202425xlsx/open (`SRC_Data_Dictionary_202425.xlsx`) | 73,347 | column definitions, masking codes |
| https://www.dpi.nc.gov/dpischoolgradetechnicalguidedraft2025-26/download?attachment (`docs/DPISchoolGradeTechnicalGuideDraft2025-26.docx`) | 917,459 | the rule: sections 2.1 (SPG) and 9.8 (low-performing) |

The adapter uses these members of the zips:

| Member | Uncompressed bytes | Years | Content |
|---|---:|---|---|
| `rcd_acc_spg1.xlsx` | 849,504 | 2014-2017 | A, G, SPG score, grade, growth status |
| `rcd_acc_spg2.xlsx` | 5,307,720 | 2018-2025 | the same, plus K-2 feeder and masking flags |
| `rcd_acc_eg.xlsx` | 3,922,213 | 2018-2025 | EVAAS growth index |
| `rcd_acc_lowperf.xlsx` | 486,014 | 2018-2025 | published low-performing flags |
| `rcd_acc_eds.xlsx` | 416,778 | 2017-2025 | % economically disadvantaged |
| `rcd_acc_pc.txt` | 1,057,010,547 | 2014-2025 | test counts by subgroup (counts only) |
| `rcd_adm.xlsx` | 1,228,429 | 2006-2025 | final average daily membership |
| `rcd_location.xlsx` | 4,393,160 | 2006-2025 | county, charter, level, type, calendar, Title I |
| `rcd_experience.xlsx` | 2,086,524 | 2006-2018 | % teachers with 0-3 years of experience |
| `rcd_eq.xlsx` | 1,091,560 | 2019-2025 | % beginning teachers |

Not available as a file:
- **2025-26 SPG scores.** DPI's accountability page (checked 2026-10-28) publishes them
  only in the NCDPI Testing Dashboard (go.ncdpi.gov/AccountabilityDashboards). The
  2025-26 "School Assessment and Other Indicator Data" workbook (101 MB) has growth
  and indicator data but no SPG. So 2024-25 has no outcome until DPI's 2025-26 SRC
  release (expected July 2027) or a manual dashboard export.
- **School-level race/ethnicity of enrolled students.** It is only in DPI's SBS DART
  and Statistical Profile applications, not in a file on www.dpi.nc.gov. The adapter
  uses the subgroup shares of the year's test takers instead.

## Rule check (2026-10-28)

`adapter.rule_check()` compares, by year:
- the published grade with the 15-point scale applied to the published SPG score, for
  every graded school;
- the published score and grade with the rule applied to the published A and G, for
  schools with both scores;
- the published low-performing flag with the statute, and with the two-cutoff rule in
  (A, G).

K-2 feeders are left out.

| Spring year | Graded | No growth | Both scores | Grade vs published score | Score rule | Grade rule | Status vs G | LP published | LP: statute | LP: (A, G) rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 2,424 | 17 | 2,407 | 1.000 | 0.798 | 0.988 | 1.000 | - | - | - |
| 2015 | 2,446 | 15 | 2,431 | 1.000 | 0.807 | 0.984 | 1.000 | - | - | - |
| 2016 | 2,459 | 13 | 2,446 | 1.000 | 0.784 | 0.985 | 1.000 | - | - | - |
| 2017 | 2,478 | 14 | 2,464 | 1.000 | 0.796 | 0.982 | 1.000 | - | - | - |
| 2018 | 2,499 | 25 | 2,474 | 1.000 | 1.000 | 1.000 | 1.000 | 479 | 1.000 | 1.000 |
| 2019 | 2,504 | 16 | 2,488 | 1.000 | 1.000 | 1.000 | 1.000 | 488 | 1.000 | 1.000 |
| 2022 | 2,556 | 24 | 2,532 | 1.000 | 0.9996 | 1.000 | 1.000 | 864 | 1.000 | 1.000 |
| 2023 | 2,560 | 29 | 2,531 | 1.000 | 1.000 | 1.000 | 1.000 | 804 | 1.000 | 1.000 |
| 2024 | 2,557 | 25 | 2,532 | 1.000 | 1.000 | 1.000 | 1.000 | 735 | 1.000 | 1.000 |
| 2025 | 2,546 | 31 | 2,515 | 1.000 | 1.000 | 1.000 | 1.000 | 685 | 1.000 | 1.000 |

**2018-2025.**
- The grade rule reproduces every published grade: 15,072 of 15,072 two-score
  school-years.
- The published low-performing flag matches the statute and the (A, G) rule in every
  row (4,055 designations).
- The score rule misses one published score in 15,072: school 200350 in 2022, where
  A = 73.6 and G = 81.9 give 75.26, but the published score is 76. Both are a B, so the
  grade is unaffected.
- In the loaders' sample, `extras["eligible_published"]` agrees with D for all 12,425
  school-years, both for `load()` (published A or B) and for `load_low_performing()`
  (published LP flag).

**2014-2017** are excluded from the loaders for three reasons:
- The file publishes A as a whole number, so the score rule fits only about 80% of
  rows. Allowing A anywhere in [A - 0.5, A + 0.5] raises the fit to 98.7-99.2%.
- Each of the remaining 94 rows (1.0%; 54 of them high schools) falls in one of two
  groups. 61 have A = 95, which looks like the file's masking of achievement above 95%.
  45 have a published score equal to A alone even though they have a growth score.
  12 are in both groups.
- 147 of the 9,748 computed grades (1.5%) differ from the published grade.

The published grade still matches the published score in every row, and the 2014-2017
rows stay in the cache.

## Readiness (2026-10-28)

`python -m experiments.scripts.multi_rule_readiness nc_school_grades` reports, for the
B/C boundary:

| Quantity | Achievement A | Growth G |
|---|---:|---:|
| R^2 of Q on X | 0.789 | 0.112 |
| sd(T), sd(eta) | 14.91, 7.71 | 3.94, 11.08 |
| sd(eta) max/min across T deciles | 1.27 | 1.33 |

| Quantity | Value |
|---|---:|
| n, p (continuous X) | 12,425, 129 (11) |
| corr(T1, T2), corr(eta1, eta2) | 0.431, 0.409 |
| Weakest direction of T; its sd | (0.12, -0.99); 3.53 |
| Weakest-direction sd / sd(eta) | 0.37 |
| Treated share (B or better) | 0.299 |
| Within 0.25 sd(eta_index) of the cutoff | 8.9% |
| Agreement with the published grade | 1.000 |

The other loaders use the same Q and X, so R^2, sd(T), sd(eta), corr(T), corr(eta) and
the weakest direction are identical:

| Loader | Treated share | Near cutoff | Agreement with published |
|---|---:|---:|---:|
| `load_ab` (A) | 0.067 | 3.0% | 1.000 |
| `load_bc` = `load()` (A or B) | 0.299 | 8.9% | 1.000 |
| `load_cd` (C or better) | 0.693 | 8.7% | 1.000 |
| `load_df` (D or better) | 0.940 | 3.2% | 1.000 |
| `load_low_performing` (designated) | 0.265 | 8.7% | 1.000 |

- **Growth is weakly predicted.** EVAAS growth is a value-added measure that conditions
  on each student's prior scores, so school characteristics explain little of it
  (R^2 = 0.11). The weakest direction of T is almost the growth axis, with sd 0.37 of
  sd(eta): better identified than `chile_two_score` (0.18), worse than Florida's
  achievement/gains pair (0.48).
- **R^2 by level.** Pooled within level, R^2 is (0.77, 0.10) for elementary
  (n = 6,272), (0.87, 0.21) for middle (2,339) and (0.86, 0.39) for high (2,412).
- **Other X sets.**
  - Without the county indicators, R^2 is (0.75, 0.07).
  - Adding the share of academically gifted test takers (`extras["share_aig"]`, left
    out because AIG identification draws on earlier test scores) gives (0.81, 0.11).
  - Adding the previous grading year's A and G (`extras["q_lag"]`) gives (0.89, 0.24)
    and lowers sd(eta) to (5.5, 10.3). The lag is not in X, because the previous score
    set the previous grade and designation.
- **Treated share by year.** The share with B or better is 0.36, 0.37, 0.23, 0.27 and
  0.28 in 2018, 2019, 2022, 2023 and 2024. The low-performing share is 0.19, 0.19, 0.34,
  0.32 and 0.29.

## Outcome

`load()` uses Y = the next grading year's published SPG score (a whole number). Its mean
is 61.4, its sd 14.4, and corr(Y, 0.8 Q1 + 0.2 Q2) = 0.89. It is matched by the
six-character school code.

Of the 12,557 school-years with both scores in the sample years, 132 (1.1%) have no Y:
- 125 have no SPG row in the next grading year. 106 of them are closed: the school is
  not in the next school year's directory.
- 4 become alternative-model schools, with no grade.
- 3 have a masked grade next year.

The outcome year depends on the grading year:
- 2018-19 is followed by 2021-22, since 2019-20 and 2020-21 have no grades.
- Every other year is followed by the next one.
- `extras["y_year"]` records the outcome year.

Other loaders:
- `load_next_ach` (n = 12,414) and `load_next_growth` (n = 12,384).
- `load_low_performing_next_lp` (n = 12,432): the low-performing rule with Y = 1 if
  designated next grading year. Its mean is 0.28.
- `load_closed` (n = 12,557): Y = not in the next school year's directory. The mean is
  0.9%.
- The cache also has `y_log_adm_change` (next year's enrollment change) and
  `y_grade_points`.

## Caveats

- **Scores and units.**
  - A is a school aggregate of student pass rates; its noise includes cohort sampling
    noise, larger in small schools. G is a value-added index whose noise depends on the
    number of students with prior scores.
  - sd(eta) max/min is 1.27 for A and 1.33 for G, so eta is heteroskedastic.
  - G is bounded at 50 and 100 (index clipped at +-10): 0.8% of the sample is at 50 and
    1.0% at 100.
  - Schools repeat across up to five years (2,617 schools), so inference should cluster
    by school.
- **Covariate timing and construction.**
  - % economically disadvantaged is DPI's accountability EDS rate for the same school
    year. Values masked as above 95% or below 5% are set to 97.5 and 2.5.
  - Race, English-learner and disability shares are shares of the same year's test
    scores (counts only, from `rcd_acc_pc`), not of enrollment. A subgroup with fewer
    than 10 scores has no row and is set to 0.
  - Final ADM is the same school year's. `adm_change` uses the prior year's ADM; it is
    missing for 125 rows.
  - DPI notes that 2024-25 ADM may be inflated by the change of student information
    system. This affects only `y_log_adm_change` for 2023-24.
  - % inexperienced teachers is 0-3 years of experience through 2018 and beginning
    teachers from 2019. The year indicators absorb the level shift. It is missing for
    179 rows.
  - None of these variables is computed from test scores. They are fixed before the
    spring tests and the September release of grades, but they describe the same school
    year.
- **The rule over time.**
  - The weights, scale and low-performing definition are the same in every sample year.
  - The achievement indicators changed with ESSA in 2017-18, and the tests themselves
    changed over time. Year indicators absorb level shifts only.
  - 2021-22 follows two years without grades, and its low-performing share (0.34) is the
    highest.
- **Exclusions.** The loaders exclude:
  - K-2 feeders, alternative-model schools, masked grades and schools without growth
    (see the rule section);
  - 2014-2017, whose published A is a whole number;
  - 2024-25, which has no outcome yet.
- **The treatment and the method's assumptions.**
  - D is the published grade or designation, sharp in Q.
  - The consequences differ by boundary: only the low-performing designation (C/D with
    growth not exceeded) triggers state intervention.
  - Because a school's grade and designation feed its next score through the stakes,
    lagged scores are excluded from X.
  - Like every real dataset screened so far, X independent of eta holds only
    approximately.
