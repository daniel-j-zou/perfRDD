# North Carolina NCLB Adequate Yearly Progress: an "every target met" rule and the Title I sanction ladder, 2003-04 to 2009-10

| Field | Value |
|---|---|
| **Q** | `load()`: Q = min over all accountable cells of (p - min(CI, SH)), in percentage points: Ahn & Vigdor's (2014) assignment variable. `load_two_score()`: (Q_R, Q_M), the same minimum over reading cells and over math cells. A cell is a school x grade span (3-8, 10) x subject x student group |
| **Threshold** | 0 on Q. Per cell, the effective threshold is min(CI, SH): CI = AMO - 1.645 sqrt(AMO(1 - AMO)/n), SH = 1 - 0.9 (1 - p last year) |
| **Treatment** | `1{Q < 0}`: the school missed AYP; the treated side is **below** the cutoff. In `load()`'s sample (Title I schools already in School Improvement) a miss moves the school up one sanction stage the next year: choice -> SES -> corrective action -> restructuring plan -> restructuring |
| **Rule class** | many cutoffs with "all targets met" logic: AYP = every cell's p >= its own effective threshold, i.e. Q_R >= 0 and Q_M >= 0 (the "both" rule); missing AYP is the "either" rule Q_R < 0 or Q_M < 0. The thresholds differ by cell (AMO by year, span and subject; CI by group size; SH by last year's p) |
| **X** | % free/reduced-price lunch; log enrollment; test-taker shares of Black, Hispanic, Asian, American Indian, multiracial, LEP and disabled students (counts only); % fully licensed teachers, % with 0-3 years, % advanced degree, teacher turnover, % National Board; charter, magnet, alternative, special school, year-round, Title I (known from 2006-07), grade-span and category indicators, NCES locale, district (88 in `load()`) and year indicators. 132 columns (14 continuous) in `load()` |
| **Y** | next year's % proficient, all students, both spans pooled by valid scores: math (`load()`), reading (`load_reading()`) |
| **n** | `load()`: 607 school-years (500 schools), 349 treated, in MAIN_YEARS 2004, 2005, 2010. `load(years=ALL_YEARS)`: 1,723 (621 schools), 1,127 treated, 2004-2010 |
| **Source** | NC DPI School Report Card research database 2010-11 (Access file holding 2000-01 to 2010-11), DPI ABCs lists, NC Accountability Workbook (amended 2010-01-07), all via the Internet Archive; Ahn & Vigdor (2014), NBER WP 20511 |

Spring years: 2004 is school year 2003-04.

## The rule

NC's Consolidated State Application Accountability Workbook (sec. 3.2, 3.2c, 5.5) and
DPI's "Adequate Yearly Progress in North Carolina" page (archived 2011-12-12) define AYP:

- **Groups.** School as a whole, White, Black, Hispanic, American Indian, Asian,
  multiracial, economically disadvantaged (free/reduced lunch), LEP and students with
  disabilities; a group counts with 40 or more full-academic-year (140 days) scores across
  the tested grades. The school as a whole always counts.
- **Targets.** For each counted group and subject (reading, math) and each span (EOG
  grades 3-8; grade 10 from English I + writing and Algebra I): 95% participation and
  proficiency at or above the AMO. Plus one other academic indicator for the school
  (attendance >= 90% or +0.1, or the cohort graduation rate >= 80% or +0.1).
- **Exceptions.** A 95% confidence interval around the group's proficiency (from 2003-04);
  safe harbor (10% fewer non-proficient students than last year, OAI progress, 95%
  participation); participation averaged over 2-3 years when below 95%; from 2005-06 a
  growth model counts students on track to proficiency. AYP = every target met.
- **AMOs (%)**, from the workbook and identical to the database's target column:

| Spring years | 3-8 reading | 3-8 math | 10 reading | 10 math |
|---|---:|---:|---:|---:|
| 2003-2004 | 68.9 | 74.6 | 52.0 | 54.9 |
| 2005 | 76.7 | 81.0 | 35.4 | 70.8 |
| 2006-2007 (new math test) | 76.7 | 65.8 | 35.4 | 70.8 |
| 2008-2010 (new reading test) | 43.2 | 77.2 | 38.5 | 68.4 |
| 2011 | 71.6 | 88.6 | 69.3 | 84.2 |

The workbook gives no CI formula. The adapter uses a one-sided 95% interval on the AMO's
binomial standard error, CI = AMO - 1.645 sqrt(AMO(1 - AMO)/n), with n the group's valid
scores. Among z in {1.28, 1.645, 1.96, 2.33} and an observed-p standard error, this fits
the published status best, or within 0.5 points of the best, in 2004, 2005, 2010 and 2011.
SH needs the same cell to have been accountable the year before.

**Ahn & Vigdor's running variable.** "Effective threshold = min(boundary for CI exemption,
boundary for SH exemption)"; "subgroup gap = actual proficiency - effective threshold";
the assignment variable is the minimum subgroup gap over groups with at least 40 tested
students; positive means AYP made (their sec. 3.1). The adapter computes exactly this,
over both subjects and both spans; `load_two_score()` keeps the two subjects apart. The
growth model, participation and the OAI are outside Q; the loaders drop school-years with
any participation rate below 95% so that AYP is decided by proficiency alone.

### The sanction and the treatment

A Title I school that misses AYP in the same subject two years running enters Title I
School Improvement; each further miss moves it up a stage (Table 1 of Ahn & Vigdor; DPI
page): year 1 public school choice, year 2 + supplemental educational services (tutoring),
year 3 corrective action, year 4 restructuring plan, year 5 restructuring (charter
conversion, staff replacement, outside management...). A school making AYP stays at its
stage and leaves after two consecutive makes. Seven districts reversed choice and SES
from 2006-07, the whole state from 2008-09. Non-Title I schools only amend their
improvement plan.

The database's `SSM_IMPROVMNT_YRS_NUM` in year t is the improvement year *for t+1*, set by
AYP through t: entering year 1 follows misses in t-1 and t (125 of the 307 schools that
made AYP in 2004 and missed in 2005 and 2006 carry year 1 in the 2005-06 record; the others
are presumably not Title I), and a school that makes AYP keeps its year. So `si_years` (the improvement year during t) is last year's
value, and a miss in t moves an improving school to `si_years + 1` in t+1
(`extras["stage_at_risk"]`).

**Ahn & Vigdor (2014).** Thomas Ahn and Jacob Vigdor, "The Impact of No Child Left
Behind's Accountability Sanctions on School Performance: Regression Discontinuity Evidence
from North Carolina", NBER Working Paper 20511, September 2014, https://www.nber.org/papers/w20511
(no journal version found).
NCERDC student data, grades 3-8, AYP years 2002-03 to 2008-09, about 8,000 school-years;
outcome = mean year-over-year change in students' standardized scores. Findings (their
Tables 3-5):

- All schools: missing AYP raises math gains by 0.018 SD (p = 0.057), reading 0.000.
- First threat (no prior misses): math +0.047 SD (transfer threat) and +0.028 SD (SES
  threat); reading about +0.02, not significant.
- Intermediate sanctions (transfers, SES, corrective action): no significant effects.
- Restructuring plan: reading -0.056 SD. Restructuring itself: math +0.055 SD, reading
  +0.026 SD, "the strongest evidence of positive test score impacts".

So the effects are modest, largest at the top of the sanction ladder. The brief's
"sizable positive effects" overstates the paper.

**Choice of treatment.** The local jumps below put the effect in the same place: positive
for schools already in School Improvement (a miss advances them a stage) and largest at
corrective action or later, not positive for first-threat, entering or all schools. So
`load()` uses Title I schools with improvement year >= 1 during t. Its stage at risk:
SES 278, corrective action 152, restructuring plan 101, restructuring 76 (MAIN_YEARS).

## Getting the data

    python -m experiments.datasets.nc_ayp.download          # idempotent, ~172 MB
    python -m experiments.datasets.nc_ayp.download --check  # sizes and SHA-256

Both DPI hosts are gone (www.ncreportcards.org, abcs.ncpublicschools.org do not resolve),
so every file comes from the Internet Archive with a plain request (no login, form, CAPTCHA
or user-agent change). Fetched 2026-10-29.

| File (`data/raw/`) | Bytes | SHA-256 (first 12) | URL |
|---|---:|---|---|
| `School_Report_Card_2010-2011.zip` | 166,143,401 | f1dab82126d3 | https://web.archive.org/web/20120309005721id_/http://www.ncreportcards.org/src/School%20Report%20Card%202010-2011.zip |
| `docs/2011-Data_Dictionary.doc` | 1,740,800 | fa98e5d65e12 | https://web.archive.org/web/20120308203519id_/http://www.ncreportcards.org/src/2011-Data%20Dictionary.doc |
| `docs/nccsa_workbook_2010.pdf` | 352,685 | e14f1f9c7f45 | https://web.archive.org/web/20221006022015id_/https://www2.ed.gov/admins/lead/account/stateplans03/nccsa.pdf |
| `docs/dpi_ayp_overview_2011.html` | 38,756 | c98fe86ac761 | https://web.archive.org/web/20111212050205id_/http://www.ncpublicschools.org:80/nclb/abcayp/overview/ayp |
| `abcs/abcs_2003-04.xls` | 567,296 | 09d9a768db3c | https://web.archive.org/web/20040901201555id_/http://abcs.ncpublicschools.org:80/abcs/servlet/abcIGetExcelRpt?pRptType=0&null |
| `abcs/abcs_2004-05.xls` | 608,768 | 245a789c464d | ...20050907074307id_/...abcIGetExcelRpt?pRptType=0&pPage=1 |
| `abcs/abcs_2005-06.xls` | 710,656 | 4bfdf538659c | ...20061102191421id_/...abcIGetExcelRpt?pRptType=0&pPage=93 |
| `abcs/abcs_2006-07.xls` | 776,192 | 689a49f4d1c0 | ...20070917044413id_/...abcIGetExcelRpt?pRptType=0&pPage=96 |
| `abcs/abcs_2008-09.xls` | 665,088 | 84b7b67d4ddb | ...20090918210853id_/...abcIGetExcelRpt?pRptType=0&pPage=100 |
| `abcs/abcs_2009-10.xls` | 667,648 | 99bbd553efe4 | ...20100820204312id_/...abcIGetExcelRpt?pRptType=0&pPage=102 |

Full URLs and hashes are in `download.py`.

- **The database.** The zip holds one Access 2000 (Jet 4) file, 675 MB, with 71 tables
  and every year from 2000-01 to 2010-11. The adapter uses:
  - `SCH_ADEQUATE_PROGRESS` (43,097 rows, 2002-03 to 2010-11): per school, year, span and
    subject, each group's valid scores, eligible students, % tested, target and %
    proficient;
  - `SCHOOL_SUMMARY`: published AYP, number of targets and targets met, improvement
    indicator, year, phase and reason, % free/reduced lunch, enrollment, teacher
    measures, category, locale;
  - `SITE`: name, charter, school type, calendar, Title I (2006-07 on).
- **Reading.** The environment has no mdbtools, pyodbc or access_parser, so `_mdb.py`
  is a small standard-library Jet 4 reader: catalog, table definitions, data pages, text
  compression, numeric types and memo fields. For all 71 tables the rows read equal the
  row count in the table definition (43,097 and 24,687 for the two main tables). The `.xls` lists are
  read with `fl_school_grades/_spreadsheet.py` (imported).
- **Cache.** `build()` extracts the database to a temporary file in `data/processed/`,
  reads the five tables, deletes the file and writes `nc_ayp_cells.csv.gz` (5.6 MB, one
  row per cell) and `nc_ayp_panel.csv.gz` (2.0 MB, 24,693 school-years 2002-2011), both
  atomically. About 16 s. Delete them after changing `adapter.py`.
- **Not used.**
  - DPI's disaggregated performance zips at https://accrpt.tops.ncsu.edu/docs/disag_datasets/
    (2001-02 to 2025-26; linked from DPI's Green Book archive). The host's robots.txt
    disallows all agents and the Archive has no copy. The database makes them unnecessary.
  - DPI's AYP Subgroup Results application (accrpt.ncpublicschools.org/app/YYYY/ayp/) is
    a POST form; only the empty form is archived.
  - ED's EDFacts files and AIR's National AYP and Identification Database answer 403 to
    scripts.
  - No later SRC database (2011-12) or 2010-11 ABCs list is archived, so 2010-11 has no
    outcome and 2011-12 (the last AYP year) is missing.

## Rule check (`rule_check()`, 2026-10-29)

Schools with cells in both subjects. "Targets" = share whose published number of AYP
targets equals 1 + proficiency cells + participation cells. "Rule, all" = proficiency rule
plus participation >= 95% against published AYP. "Rule, part. OK" = proficiency rule
against published AYP among schools whose every participation rate is >= 95% (the loaders'
condition). GR = share with a growth-model code in the ABCs list.

| Year | Schools | Targets | Made, published | Made, rule | Rule, all | Part. OK | Rule, part. OK | GR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2003 | 2,170 | 0.000 | 0.477 | 0.211 | 0.677 | 1,710 | 0.656 | - |
| 2004 | 2,181 | 0.990 | 0.708 | 0.700 | 0.956 | 2,037 | **0.973** | 0 |
| 2005 | 2,199 | 0.985 | 0.576 | 0.547 | 0.940 | 1,905 | **0.964** | 0 |
| 2006 | 2,250 | 0.933 | 0.457 | 0.284 | 0.804 | 1,773 | 0.805 | 0 |
| 2007 | 2,297 | 0.911 | 0.449 | 0.492 | 0.874 | 1,968 | 0.882 | 0 |
| 2008 | 2,342 | 0.282 | 0.314 | 0.361 | 0.899 | 1,991 | 0.895 | - |
| 2009 | 2,401 | 0.969 | 0.708 | 0.541 | 0.811 | 1,908 | 0.857 | 0.023 |
| 2010 | 2,427 | 0.972 | 0.577 | 0.522 | 0.936 | 1,997 | **0.976** | 0.002 |
| 2011 | 2,450 | 0.974 | 0.276 | 0.260 | 0.972 | 2,145 | **0.980** | - |

- **Published status.** The database's AYP flag equals "all targets met" in 99.6-100% of
  schools each year, and the archived ABCs lists in 99.5-100% (2004-2010 lists).
- **Clean years: 2004, 2005, 2010, 2011.** 96-98% agreement. In `load()` (MAIN_YEARS),
  `extras["eligible_published"]` (published miss) equals D for 592 of 607 school-years
  (0.975). The 15 exceptions sit at the cutoff (median |Q| 0.35 points). 13 of them are
  rule-misses that DPI counted as made with codes CI, SH or TAS (Title I targeted-
  assistance schools may be judged on served students only): the CI is computed on
  full-academic-year scores, of which the database's valid scores are an upper bound.
- **Why 2006-2009 fit worse.**
  - 2006: the new math test. Safe harbor in math compared 2005-06 with 2004-05 rescored on
    the new standard through a concordance table (workbook sec. 3.2). The database has only
    the old-standard 2005 rate, so the rule misses most math safe harbors: 320 of 346
    disagreements are published makes. A probit shift calibrated on the statewide change
    lifts agreement only to 0.83, so it is not used.
  - 2008: the new reading test (same issue for reading safe harbor), and 3,280 cells have
    no reported participation rate, so participation misses are invisible (published
    target count is 2 higher for 1,515 schools).
  - 2007-2008: in 400-470 elementary and middle schools a year the rule counts one or two
    fewer missed targets than DPI published. The extra published misses sit near the
    cutoff (median margin of the school's closest met cell +1.6 points). The cause is not
    identified; the 1% and 2% caps on alternate-assessment proficiency (from 2006-07) are
    a candidate.
  - 2009: retests (first used in 2008-09) and growth (2.3% GR). 247 of 273 disagreements
    are published makes and 40% of schools used safe harbor; adding 3 points to every 2009
    cell lifts agreement only to 0.90.
  - 2003: the published 2002-03 AYP does not follow these cells (the CI started in 2003-04
    and the target count never matches); 2003 serves only as the safe-harbor base.

## Local effect check (`local_jump`, 2026-10-29)

Jump in Y at Q = 0, treated (missed) minus untreated, local linear with a triangular
kernel of half-width h (points of Q), separate slopes, school-clustered se; "+X" adds X
linearly. sd(Q) = 11.0 and sd(Y) = 16.3 in `load()`.

| Loader (years) | n | treated | agree | h=5 | h=5 +X | h=10 | h=10 +X |
|---|---:|---:|---:|---:|---:|---:|---:|
| `load()` math (MAIN) | 607 | 349 | 0.975 | 3.82 (3.20) | 4.00 (2.17) | 2.15 (2.44) | 1.98 (1.55) |
| `load()` math (ALL) | 1,723 | 1,127 | 0.880 | 3.45 (1.86) | **3.42 (1.26)** | 1.52 (1.36) | **2.11 (0.84)** |
| `load_reading()` (MAIN) | 607 | 349 | 0.975 | 5.29 (3.45) | 3.82 (2.34) | 3.87 (2.68) | 2.64 (1.31) |
| `load_reading()` (ALL) | 1,723 | 1,127 | 0.880 | 6.04 (2.27) | **3.64 (1.14)** | 3.10 (1.70) | **2.15 (0.75)** |
| `load_restructuring()` math (ALL) | 444 | 297 | 0.874 | 7.38 (3.10) | 5.84 (2.94) | 3.64 (2.44) | 3.06 (1.80) |
| `load_entering()` math (ALL) | 866 | 420 | 0.842 | -2.25 (1.95) | -1.77 (1.87) | -2.96 (1.49) | -1.89 (1.25) |
| `load_first_threat()` math (MAIN) | 3,621 | 782 | 0.974 | -2.60 (2.26) | 0.87 (1.10) | -2.96 (1.62) | 0.83 (0.78) |
| `load_all()` math (MAIN) | 5,867 | 1,909 | 0.972 | 0.18 (1.47) | 1.34 (0.76) | -0.80 (1.07) | 0.47 (0.55) |
| `load_all()` math (ALL) | 13,403 | 5,572 | 0.909 | -0.14 (0.88) | 0.72 (0.48) | -0.66 (0.63) | 0.49 (0.35) |

- In schools already in improvement, missing AYP raises next year's proficiency by about
  2-4 points (0.13-0.25 sd of Y), significant with covariates in 2004-2010 and positive
  but imprecise in MAIN_YEARS alone. The jump is larger at corrective action or later.
- Elsewhere it is near zero or negative. This matches Ahn & Vigdor: no detectable effect
  of intermediate sanctions in their data, the largest effect at restructuring. The
  first-threat jump they report (+0.03-0.05 SD of student gains) is too small to see in
  school proficiency levels.
- Years 2004, 2005, 2007, 2008, 2010 (n = 1,205, agreement 0.934): math 2.97 (1.78) at
  h = 5 and 1.67 (1.11) at h = 10, with X.

## Readiness numbers

| Loader | R^2 of Q on X | R^2 of Q_R | R^2 of Q_M | Treated share |
|---|---:|---:|---:|---:|
| `load()` (MAIN) | 0.415 | 0.580 | 0.386 | 0.575 |
| `load()` (ALL) | 0.436 | 0.532 | 0.380 | 0.654 |
| `load_all()` (MAIN) | 0.488 | 0.572 | 0.490 | 0.325 |

- The binding cell in `load()` is math in 448 of 607 school-years; the binding group is
  Black (162), free/reduced lunch (150), disabled (148), all students (83), LEP (39).
- Each loader takes `years=` (MAIN_YEARS, ALL_YEARS or any subset of 2004-2010) and the
  improvement-sample loaders take `y_col="y_read"`.
- **Extras.** `school_id` (cluster), `lea`, `year`, `y_year`, `q_read`, `q_math`,
  `eligible_published` (published miss), `ayp_published`, `binding_group`,
  `binding_subject`, `stage_at_risk`, `si_years`, `si_years_next`, `ayp_lag1`, `title1`,
  `abc_code`, this year's `prof_read` and `prof_math` (not in X), `cutoff` or
  `cutoffs`/`rule`, `side = "below"`.

## Caveats

- **Density at the cutoff.** On 1-point bins, local-linear density fits are 20-28%
  higher just above Q = 0 than just below:
  - all schools 2004-2010: 479 vs 394 per point;
  - `load()`: 31.7 vs 24.7 (MAIN) and 81.1 vs 68.1 (ALL).

  Ahn & Vigdor report the same pattern (+20%, McCrary test not significant in their
  sample). Schools may manage which students test or count, so this needs a formal test
  before inference.
- **D is the rule, not the published status.** Exceptions are 2.5% in MAIN_YEARS and 12%
  in 2004-2010, concentrated near the cutoff; with ALL_YEARS the true treatment is fuzzy
  in Q.
- **Same subject.** Entering improvement requires misses in the same subject. The
  improvement sample needs no such check, because any miss advances an improving school.
  The `entering` sample treats any miss in t-1 as qualifying.
- **Title I.** The Title I flag starts in 2006-07. `load()` needs no flag (schools in
  improvement are Title I by definition), but `load_entering()` is limited to 2007-2010
  and `load_first_threat()` / `load_all()` mix Title I and other schools.
- **Exclusions.** The loaders drop school-years with any participation rate below 95%
  (7-21% a year; participation averaging recovers only 2-5% and fits worse) and those
  without next-year proficiency (1-2%: closures, reconfigurations).
- **Y across test changes.** Y crosses the new math test (outcome year 2006) and the new
  reading test (2008), so year indicators absorb level shifts only. Y is the next year's
  proficiency level, not a gain. Q and Y share the same students' tests, so mean reversion
  is smooth in Q but strong.
- **Timing.** The sanction for a miss in t applies in t+1, and Y is measured in spring
  t+1. The effect is a mix of the sanction and the incentive to escape the next stage.
- **Covariates.** They describe the same school year (fixed before spring testing, not
  computed from scores). Test-taker shares come from the AYP valid-score counts. Lagged
  proficiency is excluded from X because last year's p sets the safe-harbor threshold
  (`extras["prof_*"]` hold this year's).

## Suggested policy boxes

- **`load()`.** Cutoff c on Q in [-10, +10] points (about +-1 sd(Q)); the deployed rule
  is c = 0. The mass is thin beyond +-15.
- **`load_two_score()`.** Make-AYP region {Q_R >= c_R, Q_M >= c_M} with (c_R, c_M) in
  [-10, +10]^2. Q_M binds more often (Q_M < Q_R in 74% of school-years), so a box
  asymmetric toward lower c_M is natural.
- **Alternatives.** A weighted index w Q_R + (1 - w) Q_M >= c, w in [0, 1], c in [-10, 10],
  is an alternative family to compare with the deployed "both" rule.
