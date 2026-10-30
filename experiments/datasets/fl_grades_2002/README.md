# Florida A+ school grades, 2002-2011: the D/F boundary on grade points

| Field | Value |
|---|---|
| **Q** | `load()`: grade points minus the year's D minimum (280 on the 600-point scale, spring 2002-2006; 395 on the 800-point scale, 2007-2011). Integer valued. `load_two_score()`: (status points + retake bonus, growth points), each minus half the D minimum |
| **Threshold** | 0. **The treated side is below it**: D = 1{Q < 0} = graded F (`treatment_rule`); `extras["treated_side"] = "below"` |
| **Treatment** | An F grade: Opportunity Scholarship vouchers (to 2005), state assistance and intervention for D and F schools, public pressure |
| **Rule class** | one cutoff on a fixed-weight sum of components; in the two-score view a weighted sum of two scores, weights (1, 1); cutoff by era (year groups) |
| **X** | NCES CCD for the same school year: % free or reduced-price lunch, % non-white, % Black, % Hispanic, log membership, pupil/teacher ratio, Title I, magnet, charter, locale; FLDOE school type; year and district indicators (96 columns, 6 continuous) |
| **Y** | `load()`: next year's percent of possible points (mean of all status and growth components, 0-100). Other loaders: next reading/math proficiency, next status or growth mean, next grade, not graded next year |
| **n** | 26,713 school-years (3,231 schools), 437 graded F; 338 within 10 points of the cutoff (129 F) |
| **Source** | FLDOE school-grades spreadsheets 2001-2014 and technical assistance papers (cdn.fldoe.org); NCES Common Core of Data public school universe 2001-02 to 2013-14 (nces.ed.gov) |

## The rule

From 2001-02 a Florida school earns one point for each percent of its eligible students
meeting a criterion, and the grade is read from the total. Sources: the FLDOE "Guide to
Calculating School Grades" technical assistance papers `SchoolGradesCalcTAP03.pdf` to
`SchoolGradesCalcTAP13.pdf` and `SchoolGradesCalcGuide14.pdf`, and the one-page guide
sheets. The 2001-02 guide sheet (`0102GuideSheet.pdf`) is the only 2002 source; no TAP02
exists.

| Spring years | Components | Max | A | B | C | D | Source |
|---|---|---:|---:|---:|---:|---:|---|
| 2002-2006 | reading, math, writing proficiency; reading and math gains; lowest-25% reading gains | 600 | 410 | 380 | 320 | 280 | guide sheet 0102; TAP03-TAP06 Table 1 |
| 2007-2011 | the six above plus science proficiency and lowest-25% math gains; +10 retake bonus for some high schools | 800 (810) | 525 | 495 | 435 | 395 | TAP07-TAP11 Table 1 |
| 2010-2014, schools with grade 12 | 800 FCAT points + 800 graduation, acceleration and readiness points | 1,600 | 1,050 | 990 | 870 | 790 | TAP10 Table 7 (2014 type 03: 1,120 / 1,040 / 880 / 800, Guide14 Table 8) |
| 2012-2014, middle schools with acceleration | 800 + 100 acceleration | 900 | 590 | 560 | 490 | 445 | TAP12 Table 7 |
| 2012-2014, 6-12 and K-12 schools | weighted | 1,700 | 1,115 | 1,050 | 925 | 840 | TAP12 Table 9 |

The 2001-02 guide sheet warns that the 2002 scale "may vary by as much as 5%". The
published 2002 grades match 410/380/320/280 exactly, so no variation was used.

The two scores are:
- **status**: percent at FCAT level 3 or above in reading and math, plus the writing
  measure (the mean of the percent scoring 3.0 or above and the percent scoring 3.5 or
  above); science from 2007;
- **growth**: percent making learning gains in reading and math, plus the lowest 25%
  making reading gains, and math gains from 2007.

The published points equal the sum of the published components plus the bonus in every
FCAT-scale row, 2002-2011.

**Extra conditions**, from the same papers:
- **Grade A.** It needs 95% tested and adequate progress of the lowest 25% in the
  current year: at least 50% make gains. From 2008 a school can also qualify with 40% and
  a 1-point improvement, or a 5-point improvement.
  - 2002-2003: an A also needs lowest-25% reading gains within 10 points of all students'
    gains.
  - 2010-2014, high-school scales: an A also needs the at-risk graduation-rate target.
- **Grades B and C.** These lose a letter without adequate progress in the current or
  the previous year. 2002 has no previous year, so B and C are not reduced. 2012 has no
  such requirement.
- **Under 90% tested.** The school gets an "I", or after review a grade one letter
  lower. This is the only condition that can turn a D on points into an F before 2013.
- **2012-2014.** No grade may fall more than one letter below the previous year's.
  FLDOE adds "gap points" to reach the lower grade's minimum.
- **2013-2014.** A school with fewer than 25% of students reading at grade level drops a
  letter. This can turn a D into an F.

So for the loaders' years the deployed D/F rule is:

    F  <=>  points < D minimum,  or  (a D on points and under 90% tested).

The loaders use D = 1{points < D minimum}. The second clause binds for 2 school-years
(see the rule check).

**Stakes.** The TAPs cite State Board Rule 6A-1.09981, which provides "assistance and
intervention provisions for low performing schools (D and F)".
- **Vouchers.** F schools' students were eligible for Opportunity Scholarships (s.
  1002.38, F.S.). The Florida Supreme Court struck the program down in *Bush v. Holmes*,
  919 So. 2d 392 (Fla. Jan. 5, 2006). `extras["voucher_era"]` flags 2002-2005.
- **Not checked against primary sources here.** The two-F-grades-in-four-years
  eligibility, and the later Differentiated Accountability interventions.

**Literature (checked).** All three use RDD or related designs around the 2002 F cutoff
and find F schools improved:
- Chiang (2009), *J. Public Econ.* 93(9-10): 1045-1057.
- Figlio and Rouse (2006), *J. Public Econ.* 90(1-2): 239-255.
- Rouse, Hannaway, Goldhaber and Figlio (2013), "Feeling the Florida Heat?", *American
  Economic Journal: Economic Policy* 5(2): 251-281.

## Rule check (2026-10-29)

`adapter.rule_check()` computes each school's grade from its final points, the year's
scale and the extra conditions above, using the flags printed in the files. It compares
that grade with the published one for every school-year graded A-F:

| Spring year | Graded A-F | Published F | Computed F | D/F agreement | Full-grade agreement |
|---|---:|---:|---:|---:|---:|
| 2002 | 2,424 | 64 | 64 | 1.000 | 1.000 |
| 2003 | 2,518 | 35 | 35 | 1.000 | 0.999 |
| 2004 | 2,652 | 47 | 47 | 1.000 | 0.994 |
| 2005 | 2,771 | 78 | 78 | 1.000 | 0.997 |
| 2006 | 2,790 | 21 | 21 | 1.000 | 0.997 |
| 2007 | 2,838 | 83 | 83 | 1.000 | 0.996 |
| 2008 | 2,892 | 45 | 45 | 1.000 | 0.996 |
| 2009 | 2,952 | 43 | 43 | 1.000 | 0.998 |
| 2010 | 3,000 | 55 | 55 | 1.000 | 0.994 |
| 2011 | 3,038 | 38 | 38 | 1.000 | 0.996 |
| 2012 | 3,101 | 53 | 53 | 1.000 | 1.000 |
| 2013 | 3,140 | 116 | 116 | 1.000 | 0.996 |
| 2014 | 3,236 | 192 | 192 | 1.000 | 0.994 |

- **D/F line.** Every published F status is reproduced: 37,350 of 37,350 school-years
  with points.
  - 2004 has two F schools with no points in the file; they are excluded.
  - On points alone, ignoring the extra conditions, D/F agreement is 1.000 in every year
    except three:
    - 2005 (0.9993): 2 schools were D on points with 89% and 85% tested;
    - 2013 and 2014: the reading-25% rule.
- **Full grade.** The 121 remaining disagreements (0.3%) are all on the A/B, B/C and C/D
  lines. Most are C schools that the adequate-progress rule as transcribed would lower to
  D, but FLDOE did not. The likely causes are the "previous year" adequate-progress
  definition or appeals.
- **Grades without points.** The files hold 106 N (2002), 80 N (2003), and 47 (2004),
  59 (2006) and 62 (2007) P "pending" grades. They are not graded A-F and are excluded.

Within the loaders' sample, `extras["eligible_published"]` (published F) equals D for
26,711 of 26,713 school-years. The 2 exceptions are the 2005 participation cases above.

## Sample and boundary mass

`load()` covers spring 2002-2011: school-years graded A-F on the FCAT-only scale (600 or
800 points), with points and a next-year outcome.

**Excluded:**
- **Schools with grade 12 in 2010-2011.** They were graded on 1,600 points (5 F), with a
  third score.
- **2012-2014 (361 F).** FLDOE topped up points to enforce the one-letter-drop limit,
  which puts mass exactly at cutoffs. The reading-25% rule makes D/F a two-condition
  rule, and middle and 6-12 schools have their own scales. These years are in `build()`
  and `rule_check()`.
- **F schools without a next-year outcome (67).** 62 have no next-year row; 5 have a row
  without components.

Of the 511 published F grades in 2002-2011, the sample keeps 437. The rest are the 5
1,600-point schools, the 2 without points and the 67 without an outcome.

| Spring year | n | F | Within 10 points, \|Q\| <= 10 | F within 10 |
|---|---:|---:|---:|---:|
| 2002 | 2,411 | 62 | 68 | 25 |
| 2003 | 2,495 | 26 | 25 | 9 |
| 2004 | 2,623 | 41 | 31 | 12 |
| 2005 | 2,748 | 64 | 31 | 10 |
| 2006 | 2,761 | 16 | 22 | 7 |
| 2007 | 2,804 | 81 | 52 | 18 |
| 2008 | 2,866 | 35 | 29 | 14 |
| 2009 | 2,914 | 36 | 35 | 17 |
| 2010 | 2,535 | 45 | 28 | 12 |
| 2011 | 2,556 | 31 | 17 | 5 |
| **Total** | **26,713** | **437** | **338** | **129** |

- **Within 20 points:** 722 school-years, 220 of them F.
- **Treated share:** 1.6%.
- **Where the cutoff sits:** 2.2 sd(Q) below the mean. F schools were rare by 2006
  (16 F).

## X

All covariates come from the NCES CCD school universe file for the same school year. They
are matched by district number (`STID`) and the last four digits of the school number
(`SEASCH`), which match FLDOE's numbers. CCD matches all but 12 school-years in the sample;
pupil/teacher ratio is missing for 508. Missing values get the median and a flag.
Membership is the October count. FRL and race shares are counts divided by membership or
by the race total.

- **Data checks.** CCD's shares correlate with the FRL and minority rates printed in the
  FLDOE files: 0.94-0.99 for FRL and 0.998-1.000 for minority (2003, 2005, 2008, 2011).
  CCD is used because the FLDOE files lack these rates in 2002 and 2004 and never report
  enrollment.
- **Excluded.** Lagged points (`extras["q_lag"]`) and the previous grade
  (`extras["grade_prev"]`) are not in X, because they depend on the previous year's
  grade.

| Score | R^2 on X | sd(Q) | sd(T) | sd(eta) |
|---|---:|---:|---:|---:|
| Points (`load`) | 0.616 | 62.4 | 48.9 | 38.7 |
| Status (`load_two_score`) | 0.703 | 43.6 | 36.5 | 23.7 |
| Growth | 0.328 | 26.2 | 15.0 | 21.5 |

- **Fit of points.** Without district indicators R^2 is 0.576. Adding `q_lag` raises it
  to 0.725.
- **Heteroskedasticity.** The max/min ratio of sd(eta) across deciles of T is 1.29.
- **Mass near the cutoff.** 1.2% of school-years are within 0.25 sd(eta) of it.

## Outcome and the size of the effect

Y is matched by district and school number to the next spring's file. The local linear
jump at the cutoff below has a uniform kernel, separate slopes on each side and standard
errors clustered by school. It is the F side minus the D side.

| Y (next year) | Mean (sd) | h = 10 | h = 20 | h = 30 | h = 20 with X |
|---|---|---:|---:|---:|---:|
| % of possible points (`load`) | 67.3 (8.8) | +2.67 (1.57) | **+3.07 (1.09)** | +3.05 (0.86) | +2.48 (0.97) |
| Reading/math proficiency (`load_next_proficiency`) | 67.4 (15.8) | +5.57 (3.01) | +4.15 (2.02) | +4.22 (1.57) | +2.49 (1.26) |
| Status mean (`load_next_status`) | 69.0 (12.5) | +2.59 (2.03) | +3.33 (1.37) | +2.82 (1.07) | +2.57 (1.08) |
| Growth mean (`load_next_growth`) | 65.6 (7.4) | +2.74 (1.71) | +2.81 (1.25) | +3.28 (0.99) | +2.40 (1.17) |
| Grade A=4..F=0 (`load_next_grade`) | 3.17 (1.03) | +0.56 (0.24) | +0.51 (0.17) | +0.50 (0.13) | +0.47 (0.16) |
| No row next year (`load_closed`) | 0.010 | +0.03 (0.05) | +0.03 (0.03) | +0.03 (0.03) | +0.04 (0.03) |

- **Window sizes.** h = 20 uses 699 school-years (220 F); h = 10 uses 320 (129 F).
- **Effect on next year's points.** F schools score about 3 percentage points higher
  next year, 0.35 sd of Y. This matches the literature's direction.
- **By era (h = 20).** The jump is +1.91 (1.74) in 2002-2005 and +4.11 (1.35) in
  2006-2011.
- **Placebos (h = 20).** The jump is +3.2 (3.7) in % FRL and +9.9 (7.2) in lagged
  points, neither significant.
- **Closure.** Closure is rare near the cutoff (3.7%) and does not jump significantly.

## Two-score view: status against growth

`load_two_score(era)` reproduces the published F for the same 26,711 of 26,713
school-years as `load()`, with an identical D. The eras:
- **`"all"` (default).** Pools the years, centring each score at half the year's D
  minimum, so the rule is Q1 + Q2 < 0.
- **`"2002-2006"`.** Raw points, cutoff 280: n = 13,038, 209 F.
- **`"2007-2011"`.** Raw points, cutoff 395: n = 13,675, 228 F.

| Era | R^2 status / growth | corr(Q1, Q2) | corr(T1, T2) | corr(eta1, eta2) | Weakest direction of T: sd / mean sd(eta) |
|---|---|---:|---:|---:|---:|
| all | 0.703 / 0.328 | 0.57 | 0.76 | 0.46 | 9.3 / 0.41 |
| 2002-2006 | 0.705 / 0.399 | 0.62 | 0.77 | 0.50 | 8.6 / 0.45 |
| 2007-2011 | 0.729 / 0.265 | 0.53 | 0.75 | 0.45 | 9.4 / 0.38 |

The weakest direction of T is about (0.3, -0.95), i.e. mostly growth. This view has
three properties:
- the deployed weights (1, 1) are a real policy choice: status and growth points count
  equally;
- growth is much less predictable from demographics than status;
- the high-school retake bonus (0 or 10 points, 2007-2011) is counted with status.

## Getting the data

    python -m experiments.datasets.fl_grades_2002.download          # idempotent
    python -m experiments.datasets.fl_grades_2002.download --check  # sizes, SHA-256

All 73 files, 180.8 MB, were downloaded on 2026-10-29 with a plain `urllib` request
(Python's default User-Agent). There was no login, form or CAPTCHA. `download.py` holds
the size and SHA-256 of every file (`MANIFEST`).

- **FLDOE.** Files come from `https://cdn.fldoe.org/file/18534/` (robots.txt allows
  all). The archive page `www.fldoe.org/.../school-grades/archives.stml` returns 403 to
  scripts and was not used. The file names were found through web search and the
  Internet Archive's CDX index of `fldoe.org/core/fileparse.php/18534/urlt/*`. Every file
  then came straight from the CDN; no archived copy was needed.
- **NCES.** Files come from `https://nces.ed.gov/ccd/`, listed on
  https://nces.ed.gov/ccd/pubschuniv.asp.
- **Reading.** The adapter reads the `.xls` files with
  `experiments/datasets/fl_school_grades/_spreadsheet.py` (imported, not copied) and the
  CCD zips in place.
- **Cache.** `data/processed/fl_grades_2002_panel.csv.gz` (3 MB, all 40,435
  school-years 2001-2014). Delete it after changing the adapter.

| File(s) on cdn.fldoe.org/file/18534/ | Bytes | Use |
|---|---:|---|
| SchoolGrades01.xls | 626,176 | 2001 grades (old model): grade history only |
| SchoolGrades02.xls | 531,456 | 2002 |
| SchoolGrades03.xls | 871,936 | 2003 (SAS-style columns; FRL and minority as fractions) |
| SchoolGrades04.xls | 888,832 | 2004 (no FRL or minority) |
| SchoolGrades05.xls ... SchoolGrades09.xls | 1,089,024; 1,024,000; 1,492,992; 1,395,200; 1,479,680 | 2005-2009 |
| NonHighSchoolGrades10.xls ... 14.xls | 1,448,960; 1,398,784; 1,668,608; 2,123,776; 2,143,232 | 2010-2014, schools without grade 12 |
| HighSchoolGrades10.xls ... 14.xls | 380,416; 770,560; 544,256; 552,960; 647,168 | 2010-2014, schools with grade 12 |
| SchoolGrades12.xls, 13.xls, 14.xls | 1,271,296; 1,404,928; 1,497,600 | combined 2012-2014 files, not used |
| SchoolGradesCalcTAP03.pdf ... TAP13.pdf, SchoolGradesCalcGuide14.pdf | 180,750-1,247,453 each | the rule, 2003-2014 |
| 0102GuideSheet.pdf ... 0809GuideSheet.pdf, 0910SchoolGradesGuideSheet.pdf ... 1314SchoolGradesGuideSheet.pdf | 81,690-1,491,389 each | one-page rule summaries; 0102 is the 2002 source |

| File(s) on nces.ed.gov/ccd/ | Bytes | Use |
|---|---:|---|
| data/zip/sc011aai_dat.zip, sc021aai, sc031aai, sc041bai, sc051aai, sc061cai (`_dat.zip`) | 5.8-7.0 MB each | 2001-02 to 2006-07, fixed width, states A-I |
| data/txt/psu01lay.txt, psu021alay, psu031alay, psu041blay, psu051alay, psu061clay | about 0.1 MB each | record layouts (column positions) |
| data/zip/sc071b_txt.zip, sc081b, sc092a; Data/zip/sc102a, sc111a_supp, sc122a, sc132a | 14.3-15.8 MB each | 2007-08 to 2013-14, tab-delimited |
| matching layouts psu071blay ... sc132alay | 27-42 KB each | variable definitions |

Files that do not exist under these names (404, checked 2026-10-29): `SchoolGrades10.xls`,
`SchoolGrades11.xls`, `SchoolGradesCalcTAP02.pdf`, `SchoolGradesCalcTAP14.pdf`.

## Caveats

- **Few treated units.** F schools are 1.6% of the sample, with 129 within 10 points.
  By year, the mass near the cutoff is 17-68 school-years.
- **The cutoff and the scale change in 2007.** Science and lowest-25% math gains were
  added, the scale went from 600 to 800 points, and the D minimum went from 280 to 395.
  - Q is in points on both scales, so a point is one-sixth vs one-eighth of a percent of
    the maximum.
  - Year indicators absorb level shifts only. `load_two_score(era)` separates the eras.
- **Tests change over time.** The 2012 grades use FCAT 2.0 reading and math, and a new
  writing measure: the percent at 3.0 or above, instead of the mean of the 3.0 and 3.5
  rates (TAP12). These affect the 2011 -> 2012 outcome; year indicators absorb the mean
  shift.
- **The treatment changes over time.** It is vouchers plus stigma in 2002-2005 and
  stigma plus state intervention after. The jump is larger in 2006-2011 (`voucher_era`).
- **Points are grains of a sum of rounded percents.** Q takes integer values, so the
  cutoff falls between -1 and 0.
  - Small schools have noisier components, so eta is heteroskedastic.
  - Schools repeat over up to 10 years, so inference should cluster by
    `extras["school_id"]`.
- **Other selection and timing issues.**
  - 62 F schools (and others) have no next-year row: closure, renumbering or conversion.
    The closure jump is small and not significant.
  - When few students take writing or science, FLDOE substitutes another value: reading
    proficiency in 2003, the district average from 2004.
  - CCD membership and FRL are October counts of the same school year, fixed before the
    spring tests.
- **The extra conditions.** Under 90% tested can turn a D into an F (2 cases in the
  sample). The loaders' rule ignores it, so D is sharp in Q except for these 2.

## Suggested policy boxes for the estimator

- **`load()`.** The deployed cutoff is Q = 0. Natural boxes are shifts of the D minimum,
  for example c in [-30, +30] points around the deployed cutoff (about +/- 0.75 sd(eta)),
  applied as a common shift to both eras. A wider box quickly leaves the region with F
  mass, since only 437 school-years are F at all.
- **`load_two_score("all")`.** Weights w = (w1, 1 - w1) with w1 in [0.3, 0.7] around the
  deployed (0.5, 0.5), and a cutoff shift in [-30, +30] on the sum. Within an era, a
  rule could also require a minimum on growth, an "either" variant of the
  lowest-25% condition.
