# Texas campus accountability 2013-2017: Improvement Required from targets on four indexes

| Field | Value |
|---|---|
| **Q** | Margins over the index targets, in index points: m_k = (unrounded index k) - (target_k - 0.5), k = 1..4, so m_k >= 0 exactly when the target is met. M12 = min(m1, m2) in 2013-2014 and max(m1, m2) from 2015. `load()`: min(M12, m3, m4), shape (n,). `load_two_score()`: (M12, min(m3, m4)), shape (n, 2). `load_three_score()`: (M12, m3, m4) |
| **Threshold** | 0 on every margin (cutoffs (0, 0) for the pair) |
| **Treatment** | D = 1 = Improvement Required (IR): `1{Q < 0}` for `load()`; `1{Q1 < 0 or Q2 < 0}` for the pair (`extras["rule"] = "either"`). The treated side is below the cutoff |
| **Rule class** | cutoffs on several scores with "and"/"or" logic: Met Standard needs every required index in 2013-2014, and "Index 1 OR Index 2" AND Index 3 AND Index 4 from 2015. Targets differ by year and campus type |
| **X** | % economically disadvantaged, % English language learners, mobility rate, log enrollment, lowest and highest grade served, middle school, charter, district log enrollment and % economically disadvantaged, ESC region and year indicators (31 columns; 8 continuous). From the summaries' "Campus Demographics" (PEIMS enrollment data, not test scores) |
| **Y** | `load()`: the next rated year's Index 1 (% of STAAR tests passed, all subjects), unrounded. `load_gain()`: its change from this year. `load_2017()`: the 2018 A-F Student Achievement domain score |
| **n** | `load()`: 11,543 campus-years (5,866 campuses, 997 districts), non-AEA elementary and middle schools rated in 2013 or 2014, 1,061 IR (9.2%). With 2015 (Y two years on): 17,315. `load_2017()`: 5,921 (269 IR) |
| **Source** | TEA static campus Accountability Summaries 2013, 2014, 2015 and 2017 (34,532 PDFs on rptsvr1.tea.texas.gov), TEA's final campus rating lists 2013-2017, and the Accountability Manuals 2013-2017 |

## The rule

Every year from 2013 to 2017 TEA's Accountability Manual defines the campus rating (2013:
Chapters 3-5 and Appendix J; 2014-2017: Chapter 2 on ratings criteria and targets, and
Chapter 4 "Performance Index Indicators"). Each campus gets four index scores, each
0 to 100: points earned as a percent of maximum points.

| Index | Elementary and middle schools |
|---|---|
| 1 Student Achievement | % of STAAR tests at the satisfactory standard (phase-in Level II through 2015; "Approaches Grade Level" in 2017), all subjects |
| 2 Student Progress | weighted STAAR growth (met or exceeded progress), all subjects, all students and student groups |
| 3 Closing Performance Gaps | weighted performance of the economically disadvantaged group and the lowest-performing racial/ethnic groups of the prior year |
| 4 Postsecondary Readiness | the STAAR component only: % of students at the final Level II / postsecondary standard in two or more subjects. Not evaluated for elementary and middle schools in 2013 |

**Rounding.** The manuals say the Index 2 score "is the rounded result of total points
divided by the maximum points". Their example: "59.87% is rounded to 60%; 79.49% is
rounded to 79%; and 89.5% is rounded to 90%" (2015 manual, Ch. 4; the same in 2013). That
is round half up to a whole number. In the data:
- the published Index 1, 2 and 3 scores equal round-half-up(100 x points / maximum) for
  every campus-year;
- for every non-AEA campus, the published Index 4 equals round-half-up of the one-decimal
  STAAR component.

So a campus meets index k exactly when its unrounded index is at least target_k - 0.5.
The adapter works with the margins

    m_k = (unrounded index k) - (target_k - 0.5),     k = 1, 2, 3, 4.

The met/missed test is done in integers (200 x points >= (2 target - 1) x maximum), so a
ratio of exactly x.5 rounds up, as TEA's do. William B Travis EL (2014) has m2 = 0.000
exactly and is listed as meeting Index 2.

**Targets** for non-AEA elementary / middle schools. Every summary prints its own four
targets, which is where the adapter reads them.

| Year | Index 1 | Index 2 | Index 3 | Index 4 (STAAR only) | Source |
|---|---|---|---|---|---|
| 2013 | 50 | 30 / 29 | 55 | not evaluated | 2013 manual, Ch. 3 and App. J |
| 2014 | 55 | 33 / 28 | 28 / 27 | 12 / 13 | 2014 manual, Ch. 2; Index 2 and 3 targets are "about the fifth percentile" of 2014 performance, fixed at release |
| 2015 | 60 | 30 / 28 | 28 / 27 | 12 / 13 | 2015 manual, Ch. 2; Index 2 fifth percentile |
| 2016 | 60 | fifth pct. | fifth pct. | 12 / 13 | 2016 manual, Ch. 2. No campus summaries, not in the data |
| 2017 | 60 | 32 / 30 | 28 / 26 | 12 / 13 | 2017 manual, Ch. 2 |

High schools and K-12 campuses have other targets. In 2017 these were Index 2 17 and
Index 3 30; Index 4 was 60 with all four components or 21 with STAAR only. In 2014 their
Index 2 was not evaluated. AEA campuses have their own targets (2017: 35, 8, 13, and
33/45).

**Which indexes must be met.** The "Index 1 OR Index 2" provision does not cover the
whole period: it starts in 2015.

- **2013 and 2014: every index.** Met Standard is "assigned to districts and campuses
  that meet the target on all indexes for which it has performance data in 2014" (2014
  manual, Ch. 2). The 2013 manual (Ch. 3) has the same wording. IR = missing any one.
- **2015, 2016 and 2017: "Index 1 OR Index 2 AND Index 3 AND Index 4."** The 2015 manual
  says: "Unlike in previous years, districts and campuses will not be required to meet the
  target on all four indexes for 2015 accountability" (Ch. 2). An index counts only when
  the campus has data for it. For example: "A campus with performance data for only
  Index 1 and Index 3 must meet the target on both indexes."

So, with M12 = min(m1, m2) in 2013-2014 and max(m1, m2) from 2015 (M12 = m1 when the
campus has no Index 2, e.g. a campus whose lowest tested grade is new):

    Met Standard  <=>  M12 >= 0 and m3 >= 0 and m4 >= 0     (m4 only from 2014)
    IR (D = 1)    <=>  min(M12, m3, m4) < 0.

The "or" matters. Among elementary and middle schools, 607 missed Index 1 in 2015, and
487 of them were saved by Index 2. In 2017 the counts are 616 and 501.

**Small campuses.** Small numbers analysis applies when the all-students group has fewer
than 10 tests. The index is then computed on a uniform three-year aggregate (Index 1-3)
or two-year aggregate (Index 4) (2015 manual, Ch. 4). The summaries print the aggregate,
so the rule above applies to the published points. The smallest Index 1 denominator is 10
tests in all summaries and 32 in the loaders' sample. Campuses with too little data even
so are Not Rated.

**Excluded campuses.**
- **AEA campuses** (alternative education campuses under alternative education
  accountability) have lower targets and the label Met Alternative Standard. They are
  flagged by that label or by a non-AEA Index 1 target.
- **Paired campuses** have no tested grades (e.g. PK-2) and carry a partner's rating.
  Their summary names the partner ("This campus is paired with ...") and shows N/A for
  every index.
- **Not Rated campuses**: DAEPs, JJAEPs, residential facilities, and campuses with no data.

**Appeals and data integrity.** TEA's final campus lists ("FINAL 20xx ACCOUNTABILITY
RATINGS") mark campuses whose rating changed on appeal ("Appeal") and the one rating
changed for data integrity in 2013 ("Data"). The 2013-2015 static summaries carry the
final rating. The 2017 summaries are dated August 15, 2017, before appeals.
`extras["eligible_published"]` uses the final list.

## Why the treatment matters

IR is the "unacceptable" rating under Texas Education Code (TEC) Chapter 39, Subchapter E
(sections 39.102-39.116, listed in Appendix C of the 2013 manual). Only what was verified
is listed here.

- **Interventions from the first IR year.** The 2013 manual (Ch. 9) says the Texas
  Accountability Intervention System "determines the level of intervention and support
  the campus or district receives based on performance history as well as current year
  state accountability rating". The 2015 manual (Ch. 9) says districts with an IR rating
  "will be required to follow directives from the commissioner". Its appeals section
  lists "Professional Service Provider (PSP) requirements" among the intervention
  requirements.
- **Campus intervention team.** Under TEC 39.106 as amended by HB 1842 (84th Legislature,
  2015, enrolled text), the commissioner assigns a campus intervention team to a campus
  below standard. The team:
  - conducts a targeted on-site needs assessment;
  - recommends actions and helps develop a targeted improvement plan;
  - holds a public meeting at the campus and presents the plan at a public hearing of the
    board;
  - helps the commissioner monitor progress.
- **Repeated IR** (TEC 39.107 as amended by HB 1842):
  - After two consecutive unacceptable ratings, the commissioner "shall order the campus
    to prepare and submit a campus turnaround plan". The plan is implemented after a third.
  - If the campus is unacceptable for three more consecutive years after the order, the
    commissioner must appoint a board of managers or close the campus.
  - HB 1842 applies from the 2015-16 ratings. A transitional section, TEC 39.1071, maps
    the 2013-14 and 2014-15 ratings into the count.
  - Before HB 1842, a campus rated IR two years running had to submit a reconstitution
    plan. The campus intervention team decided whether the principal and teachers could
    stay (Legislative Budget Board, *Evaluation of House Bill 1842 ... and Campus
    Turnaround Plans*, November 2018).
- **Evidence on effects.**
  - The LBB report compares campuses rated IR for a second year after HB 1842 with those
    before it:
    - 2015-16 cohort: 55.2% met standard the next year, and 72.4% at least once by
      2017-18;
    - 2013-14 cohort: 34.4% and 58.9%.

    It reads this as suggesting a positive effect but adds that it "could be attributed
    in part to an overall improvement in accountability ratings". This is a before/after
    comparison, not a causal design.
  - For 1990s Texas, Deming, Cohodes, Jennings and Jencks (*Review of Economics and
    Statistics* 2016; NBER w19444) find that high schools facing the risk of a
    Low-Performing rating raised scores on the high-stakes exams. Their students later had
    more college completion and higher earnings.
  - I found no regression-discontinuity study of the 2013-2017 IR rating.
  - The local check below is the direct evidence. It shows a jump of about 1 to 1.6
    Index 1 points.
- **Attenuation.** Met Standard campuses that fail system safeguards also get TAIS
  attention (2013 manual, Ch. 9). The contrast at the cutoff is between IR and Met
  Standard, not between intervention and none.

## Getting the data

    python -m experiments.datasets.tx_index_2013.download            # about 70 minutes
    python -m experiments.datasets.tx_index_2013.download --check    # sizes and hashes
    python -m experiments.datasets.tx_index_2013.download --no-summaries

**Download rules.** Every file is a static file on a TEA server, fetched with a plain
request using the default Python user agent.
- rptsvr1.tea.texas.gov has no robots.txt; tea.texas.gov's disallows none of these paths.
- `download.py` records the SHA-256 of each file from 2026-10-29 and reports a different
  hash, since TEA can replace files in place.

**Manuals, lists and directory** (15 files, 20,448,094 bytes, fetched 2026-10-29):

| File | Bytes | Use |
|---|---:|---|
| https://rptsvr1.tea.texas.gov/perfreport/account/2013/manual/manual.pdf | 5,484,370 | 2013 rule (full manual) |
| https://rptsvr1.tea.texas.gov/perfreport/account/2014/manual/ch02-09.pdf | 1,562,578 | 2014 rule (Ch. 2-9) |
| https://rptsvr1.tea.texas.gov/perfreport/account/2015/manual/manual.pdf | 2,767,360 | 2015 rule (full manual) |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2016manualchapters2-9.pdf | 2,056,521 | 2016 rule (Ch. 2-9) |
| https://tea.texas.gov/school-and-district-leaders/accountability/academic-accountability/performance-reporting/2017-manual-chapter-2-9.pdf | 620,297 | 2017 rule (Ch. 2-9) |
| https://rptsvr1.tea.texas.gov/perfreport/account/2013/campuslist.pdf | 639,476 | final campus ratings 2013 |
| https://rptsvr1.tea.texas.gov/perfreport/account/2014/campuslist.pdf | 842,978 | final campus ratings 2014 |
| https://rptsvr1.tea.texas.gov/perfreport/account/2015/campuslist.pdf | 945,237 | final campus ratings 2015 |
| https://rptsvr1.tea.texas.gov/perfreport/account/2016/campuslist.pdf | 800,479 | final campus ratings 2016 (IR history only) |
| https://rptsvr1.tea.texas.gov/perfreport/account/2017/List_Campus_By_Rating.pdf | 974,078 | final campus ratings 2017 |
| https://rptsvr1.tea.texas.gov/perfreport/account/{2013,...,2017}/campdist.js | 744,237 / 746,433 / 751,979 / 752,547 / 759,524 | TEA's campus directory behind its search page: campus, district, ESC region, county. Also the list of campuses to fetch |

**Campus Accountability Summaries.** One static PDF per campus and year:

    https://rptsvr1.tea.texas.gov/perfreport/account/{year}/static/summary/campus/c{campus}.pdf

- **What each holds:**
  - the rating and the "Met Standards on" / "Did Not Meet Standards on" lists;
  - the four index scores with points earned and maximum points, and the four targets;
  - the Index 4 components;
  - the campus demographics: type, size, grade span, % economically disadvantaged,
    % English language learners, mobility rate, and from 2017 % special education.
- **Coverage.** Every campus in each year's `campdist.js` had one:

  | Year | PDFs | Bytes |
  |---|---:|---:|
  | 2013 | 8,555 | 1,347,824,175 |
  | 2014 | 8,574 | 1,374,551,623 |
  | 2015 | 8,646 | 1,386,991,696 |
  | 2017 | 8,757 | 1,217,813,122 |
  | Total | 34,532 | 5,327,180,616 |

- **The run.** Fetched 2026-10-30 00:42-01:55 UTC with 4 parallel requests.
- **The PDFs are not kept.** The script runs `pdftotext -layout` (poppler 26.02) on each
  PDF in memory and writes the text to `data/raw/summaries/{year}/c{campus}.txt` (about
  68 MB a year). `data/raw/summaries/{year}/manifest.csv` records, for each campus:
  - the HTTP status;
  - the PDF's size and SHA-256;
  - the fetch time.

  `--keep-pdf` keeps the PDFs too.
- The campus-list PDFs get the same treatment (`lists/campuslist_{year}.txt`). The
  adapter reads only text.

**2016 is missing.** Every 2016 summary URL returns 404 (22 of 22 sampled campuses).
- TEA's 2016 search page builds the summary on request through its SAS broker
  (`prgopt=2016/acct/summary_dyn.sas`, a POST form). No static copy was found.
- The Internet Archive's CDX API returned "Temporarily Offline" every time it was tried
  (2026-10-29/30), so archived copies could not be checked.
- The static 2016 final campus list is used for the IR history only.
- The effect: 2015 ratings have no next-year outcome, and the default sample is the 2013
  and 2014 ratings.

**Not used: the form-driven downloads.**
- TEA's "Accountability Data Download"
  (`https://rptsvr1.tea.texas.gov/perfreport/account/{year}/download.html`; 2013:
  `camp_download.html`) has the same index scores as flat files (CAMPRATE, CAMPIDX1-4).
  It works only through a POST form to `/cgi/sas/broker`.
- The TAPR downloads and the 2014-2018 TAPR campus reports are the same.
- Filling 2016 needs the author: download the 2016 campus file "Accountability Rating,
  Index Scores, and Reference" (CAMPRATE) and the Index 1-4 files by hand. A reader for
  them would then have to be added.

## Rule check (2026-10-30)

`adapter.rule_check()` takes non-AEA campuses with their own index data and a final
rating of Met Standard or IR. These are all campus types, not only the loaders' sample.

| Year | Summaries | AEA | Paired | Checked | Index 1-3 rounding | IR, computed | IR, summary | IR, final list | Agree with summary | Agree with final list | Changed on appeal / data integrity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2013 | 8,555 | 254 | 258 | 7,377 | 1.000 | 694 | 693 | 693 | 0.99959 | 0.99959 | 2 / 1 |
| 2014 | 8,574 | 268 | 263 | 7,428 | 1.000 | 679 | 674 | 674 | 0.99933 | 0.99933 | 5 / 0 |
| 2015 | 8,646 | 279 | 289 | 7,472 | 1.000 | 567 | 564 | 564 | 0.99960 | 0.99960 | 3 / 0 |
| 2017 | 8,757 | 269 | 308 | 7,591 | 1.000 | 334 | 334 | 325 | 1.00000 | 0.99881 | 9 / 0 |

- **Every disagreement is explained.** Each is a campus whose rating was changed on appeal
  (to Met Standard) or the 2013 data-integrity change (Girls & Boys Prep Academy EL, IR
  with every index met). There are 19 appeals and 1 data-integrity change, and every
  flagged campus is one of the disagreements.
- **The computed met/missed result** for each of the four indexes equals the summary's
  "Did Not Meet Standards on" list for every checked campus-year. Under the OR rule TEA
  still lists a missed Index 1 or 2 there.
- **The bar chart** of each summary shows the same four index scores as its table.
- **Index 4 only counts when printed.** Some small high schools show a STAAR component but
  no Index 4 score. They are not evaluated on Index 4, and the rule reproduces their
  ratings only if Index 4 is ignored where no score is printed.
- **In the loaders' sample** (`load()`), D equals TEA's final rating for 11,536 of 11,543
  campus-years (0.99939). The 7 exceptions are 6 granted appeals and the data-integrity
  case. `extras["rating_note"]` flags them.

## Sample

| Step (rating years 2013 and 2014) | Campus-years left |
|---|---:|
| all campus summaries | 17,129 |
| elementary or middle school | 12,648 |
| non-AEA | 12,625 |
| not paired | 12,104 |
| final rating Met Standard or IR | 11,834 |
| Index 1 and 3 present (and Index 4 in 2014) | 11,660 |
| Y observed | 11,543 |

- **What remains.** 8,318 elementary and 3,225 middle schools, 483 of them charters; 5,751
  in 2013 and 5,792 in 2014.
- **No Index 2.** 219 campus-years have no Index 2 score (e.g. no prior-year scores for
  their tested grades). For them M12 = m1, as in the rule.
- **No Y: 117 campus-years.** 101 have no summary the next year (closed or renumbered),
  8 are paired next year, 6 are Not Rated next year, and 2 have no Index 1 next year.
- **The other years:**
  - 2015 (`years=(2013, 2014, 2015)`): 5,772 campus-years with Y = 2017 Index 1, 458 IR.
  - 2017 (`load_2017()`): 5,921 campus-years, 269 IR.

## Which index binds, and the two-score view

The smallest required margin (`extras["binding"]`) and the margins missed by IR campuses,
elementary and middle schools:

| Rating year | n | IR | Smallest margin: M12 / m3 / m4 | IR campuses missing m1 / m2 / m3 / m4 | IR with M12 < 0 |
|---|---:|---:|---|---|---:|
| 2013 | 5,751 | 537 | 4,272 / 1,478 / 1 | 94 / 235 / 370 / n.a. | 300 |
| 2014 | 5,792 | 524 | 3,553 / 1,651 / 588 | 236 / 239 / 259 / 200 | 417 |
| 2015 | 5,772 | 458 | 688 / 4,104 / 980 | 388 / 120 / 345 / 233 | 110 |
| 2017 | 5,921 | 269 | 1,792 / 4,044 / 85 | 257 / 108 / 217 / 47 | 109 |

(The 2013 "m4" is a single campus with an Index 4 score.)

**Choice of pair.** The brief suggested (m1, m3), or (max(m1, m2), m3).

| Pair (`pair=`) | D equals the full rule (2013-2014) | (2013-2015) | (2017) |
|---|---:|---:|---:|
| (m1, m3) (`"1_3"`) | 0.9664 | 0.9648 | 0.9422 |
| (M12, m3) (`"12_3"`) | 0.9955 | 0.9925 | 0.9997 |
| (M12, min(m3, m4)) (`"12_34"`, default) | 1.0000 | 1.0000 | 1.0000 |

- **(m1, m3) misses 388 IR campuses** in 2013-2014: those that missed only Index 2 or
  Index 4.
- **(M12, m3)** misses the 52 campuses in 2014 that missed only Index 4. Index 4 binds
  often in 2014 and 2015 and rarely in 2017.
- **The default is therefore Q = (M12, min(m3, m4)).** It reproduces the rule exactly in
  every year:
  - Q1 is the achievement-or-progress margin, with "or" logic from 2015 and "and" before.
  - Q2 is the gaps-and-readiness margin.
  - The treatment is IR iff Q1 < 0 or Q2 < 0.
- **Split of the 1,061 IR in 2013-2014:** 356 miss on Q1 only, 344 on Q2 only and 361 on
  both.
- `load_three_score()` keeps m3 and m4 apart. It covers 2014 on, since Index 4 was not
  evaluated for these campuses in 2013.

## Covariates and fit

X (31 columns):
- continuous (8): `pct_econ_dis`, `pct_ell`, `mobility`, `log_students`,
  `district_log_students`, `district_pct_econ_dis`, `low_grade`, `high_grade`;
- indicators: `mobility_missing` (132 rows, filled with the median), `middle`, `charter`,
  `year_2014`, and 19 ESC region indicators (Region 4, Houston, is the base).

Where the variables come from:
- Campus demographics are the summary's (PEIMS fall enrollment of the rated school year,
  before the spring tests).
- District values aggregate all of the district's campus summaries that year: enrollment
  sums, and % economically disadvantaged weighted by enrollment.
- Charter is a district number with 8 as its fourth digit.
- The ESC region comes from `campdist.js`.

**R^2 of each score on X** (`load()` sample, 2013-2014; OLS with intercept):

| Score | R^2 | sd |
|---|---:|---:|
| Q = min margin | 0.331 | 8.16 |
| M12 | 0.266 | 8.15 |
| m1 (Index 1) | 0.590 | |
| m2 (Index 2) | 0.233 | |
| m3 (Index 3) | 0.457 | 10.56 |
| min(m3, m4) | 0.470 | 10.73 |
| m4 (Index 4, 2014) | 0.642 | 15.28 |
| Y | 0.573 | 12.04 |

**The pair (M12, min(m3, m4)):**
- corr(Q1, Q2) = 0.62.
- Fitted values: sd(T) = (4.20, 7.35), corr(T1, T2) = 0.87.
- Residuals: sd(eta) = (6.99, 7.81), corr(eta1, eta2) = 0.50.
- The weakest direction of T is (-0.88, 0.47). Its sd is 1.81, 0.24 of the mean sd(eta).

With 2015 included, R^2 is 0.382 for Q and 0.412 for M12. In 2017 it is 0.488 for Q.

## Outcome and the local check

**Y.** `load()` uses Y = the next rated year's Index 1, unrounded (100 x points /
maximum), matched by campus number.
- Mean 75.96, sd 12.04, corr(Y, Q) = 0.66.
- `extras["y_year"]` records the outcome year.
- The ratings come out in August, and the outcome is the next spring's STAAR, so Y is
  measured after the designation and its interventions start.

What each rating year uses as Y:

| Rating year | Y | Note |
|---|---|---|
| 2013 | 2014 Index 1 | |
| 2014 | 2015 Index 1 | 2015 excludes grades 3-8 mathematics from accountability (commissioner's decision of April 8, 2015; 2015 manual, Ch. 7), so this Index 1 has no 3-8 math |
| 2015 | 2017 Index 1 (two years on) | `years=(2013, 2014, 2015)`; no 2016 data |
| 2017 | 2018 Student Achievement domain scaled score | `load_2017()`, read through `tx_school_ratings`'s reader. 953 of the 8,757 campuses rated in 2017 have no 2018 domain score (closed, renumbered, paired or without a score); 59 of them are in the sample's frame |

**Local check.** A local-linear fit of Y on Q with a triangular kernel and separate slopes.
The jump is the coefficient on D (IR minus Met Standard at Q = 0). The standard error is
clustered by campus. "With X" adds the 31 covariates.

| Sample, Y | h | n (IR) | Jump, no X | Jump, with X |
|---|---:|---:|---|---|
| 2013-2014, next Index 1 | 3 | 1,384 (465) | 1.10 (1.28) | 1.35 (0.93) |
| | 5 | 2,456 (665) | 1.02 (0.98) | 1.17 (0.75) |
| | 8 | 4,351 (837) | 1.20 (0.79) | 1.09 (0.61) |
| 2013-2014, change in Index 1 (`load_gain`) | 3 | 1,384 | | 1.59 (0.72) |
| | 5 | 2,456 | | 1.12 (0.58) |
| | 8 | 4,351 | | 1.02 (0.47) |
| 2013-2015, next Index 1 | 5 | 3,629 (1,010) | 0.77 (0.76) | 0.92 (0.60) |
| 2013-2015, change | 8 | 6,276 | | 0.94 (0.43) |
| 2017, 2018 SA domain score | 5 | 862 (205) | 1.52 (0.93) | 1.42 (0.86) |
| | 8 | 1,574 (246) | 1.28 (0.78) | 1.09 (0.73) |

- **The estimates.** IR raises the next year's Index 1 by about 1 to 1.6 points. That is
  about 0.1 sd of Y and about 0.2 sd of the year-to-year change (sd 5.2).
  - The levels are not significant at 5%; clustering by district gives se 0.72 at h = 5.
  - The change in Index 1 is significant at 5% at h = 3 and 8 (t = 2.2); at h = 5,
    t = 1.95.
  - The sign is the same in every sample, including 2017 with a different outcome.
  - The effect is modest, not large.
- **Balance and density.**
  - The current Index 1 does not jump at the cutoff: 0.05 (0.54).
  - Neither does % economically disadvantaged: -0.67 (1.72).
  - Counts of Q in 1-point bins from -3 to 3 are 129, 145, 203, 253, 296, 370, with no
    heaping just above 0.
- **Persistence.** Near the cutoff (|Q| < 3), 30.3% of IR campuses and 19.9% of Met
  Standard campuses are IR again the next year (`extras["ir_next"]`).
- `python -m experiments.datasets.tx_index_2013.analysis.checks` reproduces every number
  in this README.

## Caveats

- **Only two rating years have a next-year outcome.** Static summaries exist for 2013,
  2014, 2015 and 2017.
  - The 2015 ratings get a two-year-ahead Y. They span the 2016 rating, whose IR also
    counts.
  - The 2017 ratings get a different outcome scale (2018 A-F).
  - Downloading the 2016 index files by hand (form) would make 2015 a next-year year and
    add 2016 ratings with Y = 2017 Index 1.
- **The rule and the scores change across years.**
  - "And" logic in 2013-2014, "or" from 2015.
  - Index 4 starts in 2014.
  - Index 1-3 targets change. The 2014 Index 2 and 3 targets and the 2015 Index 2
    target were set at about the fifth percentile of that year's results.
  - Index 3 was rebuilt for 2014 (2013 manual, Ch. 13); its target fell from 55 to 28/27.
  - The Index 1 standard and the 2015 math exclusion also change.
  - Q is in index points relative to each year's targets, and the year indicators absorb
    level shifts only.
- **Rule exceptions.** 19 granted appeals (all types, 2013-2015 and 2017) and 1
  data-integrity change. The 2017 summaries are the August versions. 7 sample campus-years have D
  different from the published rating.
- **Repeated IR.** About a fifth of IR campuses in the sample are in their second
  consecutive IR year in the 2013+ data (`extras["ir_streak"]`: 839 first, 216 second;
  2011 ratings are not in the data). Consequences escalate with the streak, so the jump
  averages first-year and repeat IR.
- **Units.** Campuses repeat across years (5,866 campuses in 11,543 rows). Cluster by
  campus, or by district (997).
- **Mechanics of Q.**
  - Q is a minimum of margins, so it has kinks.
  - % economically disadvantaged enters Index 3 through group membership. It is a
    campus characteristic, not a score.
- **Shared assumption.** As in every real dataset screened so far, X independent of eta
  holds only approximately.

## Suggested policy boxes

- **`load()`**: a cutoff c on Q in [-5, 5] index points, where IR = 1{Q < c}. The treated
  share goes from 3.4% (c = -5) through 9.2% (c = 0) to 24.7% (c = 5). About 7.6% of
  campuses lie within 2 points of 0.
- **`load_two_score()`**: cutoffs (c1, c2) in [-5, 5]^2 on (M12, min(m3, m4)), with the
  either rule (IR if Q1 < c1 or Q2 < c2). This moves the achievement/progress target
  against the gaps/readiness target.
  - 5th and 10th percentiles of Q1: -1.1 and 2.0.
  - 5th and 10th percentiles of Q2: -1.0 and 2.5.
- **A weighted-sum alternative**: IR if w Q1 + (1 - w) Q2 < c, with w in [0.2, 0.8] and c
  in [-5, 5]. It asks whether a compensatory rule would pick campuses that gain more from
  the interventions.

## Extras

The loaders' `extras` hold:
- the unit ids: `campus` (cluster id), `district`, `region`, `year`, `campus_type`;
- the scores and targets: `margins` (m1-m4), `m12`, `index_unrounded`,
  `index_published`, `targets`, `n_tests`, and `index1_now` (this year's Index 1);
- the rule: `binding`, `d_full` (the exact rule), `cutoffs`/`cutoff`, `rule` ("either"),
  `side` ("below"), `rule_text`, `or_rule_year`;
- the published rating: `eligible_published` (TEA's final IR), `rating_published`,
  `appeal`, `rating_note`;
- the history and outcome: `ir_streak`, `ir_next`, `y_year`, `y_definition`.

The cache `data/processed/tx_index_2013_panel.csv.gz` (34,532 campus-years) holds every
parsed summary field. Delete it after changing `adapter.py`.
