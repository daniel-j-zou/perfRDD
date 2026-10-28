# US gainful-employment programs: the 2015 debt-to-earnings rates, an either rule for passing and a both rule for failing

| Field | Value |
|---|---|
| **Q** | `load_two_score()`: (a, d) = (annual earnings D/E rate %, discretionary income D/E rate %, capped at 100), the final rates at the 2017-01-09 release. `load()`: d among programs with a > 8. `load_annual()`: a among programs with d > 20 |
| **Threshold** | pass: a <= 8 or d <= 20; fail: a > 12 and d > 30 (34 CFR 668.403(c)). The same for every program |
| **Treatment** | `load_two_score()`: `1{a > 8 and d > 20}`, not passing (zone or fail). `load_two_score_fail()`: `1{a > 12 and d > 30}`, fail. `load()`: `1{d > 20}`. `load_annual()`: `1{a > 8}`. The treated side is always "above the cut point(s)" |
| **Rule class** | pass is an either rule (a cutoff on each score, "or"); its complement, not passing, and fail are both rules ("and") |
| **X** | log D/E cohort size, log(1 + IPEDS program completions) in 2014-15 and 2015-16, log(1 + institution completions 2015-16), log number of rated GE programs and of IPEDS campuses at the institution; credential level, institution type (control x length), 2-digit CIP family, state and IPEDS locale indicators (82 columns in the two-score loaders, 69 in `load_annual()`, 58 in `load()`, where some indicators are constant; 6 continuous). No debt or earnings |
| **Y** | `load*()`: 1{zero IPEDS completions in 2019-20} ("closed"). `*_dlog()`: log(1 + completions 2019-20) - log(1 + completions 2015-16) |
| **n** | `load_two_score()` and `load_two_score_fail()`: 7,065 programs (2,327 institutions). `load_annual()`: 3,329. `load()`: 1,798 |
| **Source** | ED's final 2015 D/E rates file (studentaid.gov); 34 CFR 668.403-668.404 as of 2017-01-15 (ecfr.gov); NCES IPEDS completions C2015_A-C2021_A, directory HD2015-HD2021 and the CIP 2010-2020 crosswalk (nces.ed.gov) |

A GE program is a 6-digit OPEID x credential level x 6-digit CIP. GE programs are every
certificate program at any Title IV institution and every degree program at a for-profit
one.

## The rule

The 2014 GE rule (79 FR 64890, 2014-10-31, effective 2015-07-01) gives, in 34 CFR
668.404(a), two rates. Both use the annual loan payment P (the median debt of the
completer cohort amortized over 10, 15 or 20 years by credential) and annual earnings E
(the higher of the cohort's mean and median SSA earnings):

    a = P / E,                     d = P / (E - 1.5 x poverty guideline).

For the 2015 rates the poverty term is 1.5 x 11,770 = 17,655 (earnings of calendar 2014).
The file's discretionary denominator equals max(E - 17,655, 0) for every program.

668.403(c), as in force on 2017-01-15 (`data/raw/rule/`), sets the outcome:
- **Pass**: d <= 20 or a <= 8.
- **Fail**: (d > 30 or discretionary earnings <= 0) and (a > 12 or E = 0).
- **Zone**: not passing and not failing.

So pass is an either rule and fail a both rule, on (a, d). Ineligibility would have
followed two fails in three years, or four years of zone or fail. The rule was rescinded
(84 FR 31392, 2019-07-01, effective 2020-07-01) before any program lost aid. The 2015
rates are the only official ones.

How ED publishes the rates, from the data and the file's glossary (`GE-Glossary.doc`):
- **Precision.** Each rate is truncated, not rounded, to 0.01 and capped at 999.99. A
  discretionary rate with positive debt and non-positive discretionary earnings is
  published as 100.00 (with zero debt, 0.00).
- **Transitional rates** (668.404(g)). They were computed when a program did not pass and
  had 10 or more completers in the transitional cohort (2014-15). They use that cohort's
  debt and the same earnings. The final rates are the lower of draft and transitional
  (668.404(g)(3)). The glossary says ED "takes a program's best rate" across all four.
  Because each status is monotone in its rate, that is the rule above applied to
  a = min(standard, transitional) and d = min(standard, transitional). These are the
  adapter's Q.
- **Alternate-earnings appeals** (668.406). Zone and fail programs could appeal with
  survey or state earnings. ED later replaced the rates of 109 programs with approved
  appeals. Their published result is marked `*` (94 programs) or `**` (15), and so are
  the replaced rates. The glossary does not explain `**`. The numerator and denominator columns keep the original SSA values. So the
  adapter recomputes every rate from them, which gives the rates as of the release.
- **Timing.** ED issued the final rates on 2017-01-09 (82 FR 13227). The appeal and
  disclosure deadlines were then extended (82 FR 13227; 82 FR 30975; 82 FR 39362).

## Rule check (2026-10-28)

`adapter.rule_check()`, over all 8,637 programs in the file:

| Check | Result |
|---|---|
| Rates recomputed from numerator / denominator = published rates (programs without an approved appeal; standard and transitional, both dimensions) | 1.000 for all four rates |
| Pass / zone / fail from the rule on (a, d) = published result, programs without an approved appeal | 8,528 of 8,528 |
| The same, with all programs | 8,565 of 8,637 for not passing (0.9917); 8,577 of 8,637 for fail (0.9931) |
| Rule on the standard rates only (no transitional) = published, not passing, no approved appeal | 0.979 |

- **Counts at release** (the rule on the recomputed rates): 6,595 pass, 1,239 zone, 803
  fail. The file as published (after appeals): 6,667 pass, 1,227 zone, 743 fail.
- **Exception 1: approved appeals.** All 72 not-passing mismatches and all 60 fail
  mismatches are programs with an approved appeal. Their published rates and result are
  post-appeal. At release, the 109 approved appeals were 64 fail, 44 zone and 1 pass. In
  the file they are 73 pass, 32 zone and 4 fail. The one pass at release (`PASS**`, a =
  11.36, d = 17.47) passes on its discretionary rate, so why it appealed is not clear.
  `extras["appeal_approved"]` flags them, and `extras["result_published"]` holds the
  post-appeal result.
- **Exception 2: transitional rates.** They are part of the deployed rule, not an
  exception. Without them the rule misses 251 programs: 178 that passed only on
  transitional rates and 73 that moved from fail to zone on them.
  `extras["a_std"]` and `extras["d_std"]` hold the standard rates.
- In the loaders' sample, D equals the rule at release for every program. Agreement with
  the published (post-appeal) flag, `extras["eligible_published"]`, is 0.9901 for
  `load_two_score()` (70 approved appeals) and 0.9921 for `load_two_score_fail()` (56).
  Without approved appeals it is 1.000.

## Getting the data

    python -m experiments.datasets.gainful_employment.download          # idempotent
    python -m experiments.datasets.gainful_employment.download --check  # sizes, hashes

All 22 files total 79.2 MB and go to `data/raw/` (gitignored). They were downloaded on
2026-10-28 with a plain urllib request and an honest user agent; no source blocked it.
`download.py` records each file's SHA-256 and reports a difference.

| File | Bytes | Use |
|---|---:|---|
| https://studentaid.gov/sites/default/files/GE-DMYR-2015-Final-Rates.xls | 3,639,808 | the rates, results, appeals; one row per program |
| https://studentaid.gov/sites/default/files/GE_SSA_Earnings_2014.xls | 2,355,200 | "Verified Student Count", the D/E cohort size (X); its earnings are not used |
| https://studentaid.gov/sites/default/files/GE-Glossary.doc | 71,168 | glossary of the 2015 rates file |
| https://studentaid.gov/sites/default/files/GE2011InformationalRates102512_0.XLS | 4,508,160 | 2011 informational rates (2011 rule); inspected, not used |
| https://studentaid.gov/sites/default/files/GEGlossary102512.doc | 70,656 | its glossary |
| https://nces.ed.gov/ipeds/datacenter/data/C2015_A.zip ... C2021_A.zip (7 files) | 8.4-8.9 MB each, 60,571,836 | completions by UNITID x CIP x award level, 2014-15 to 2020-21 |
| https://nces.ed.gov/ipeds/datacenter/data/HD2015.zip ... HD2021.zip (7 files) | 1.0-1.2 MB each, 7,617,648 | UNITID -> OPEID, locale, closing dates |
| https://nces.ed.gov/ipeds/cipcode/Files/Crosswalk2010to2020.csv | 311,326 | CIP 2010 -> CIP 2020 |
| https://www.ecfr.gov/api/versioner/v1/full/2017-01-15/title-34.xml?part=668&section=668.403 | 2,871 | the rule as of 2017-01-15 |
| https://www.ecfr.gov/api/versioner/v1/full/2017-01-15/title-34.xml?part=668&section=668.404 | 9,935 | the rate formulas, transition period |

- **Reading.** The environment has no `xlrd` and studentaid.gov has no CSV of the file
  (the `.csv`, `.xlsx` and `.zip` names return 404). So the adapter reads the `.xls` with
  the standard-library BIFF8 reader in `experiments/datasets/fl_school_grades/_spreadsheet.py`
  (imported, not copied). The IPEDS zips are read in place, using the revised `_rv`
  member.
- **Discovery.** The studentaid.gov GE data page is a JavaScript application. The file
  names of the other GE files came from the Internet Archive's CDX index of
  `studentaid.gov/sites/default/files/GE*`. The files themselves were downloaded from
  studentaid.gov.
- **eCFR.** The API answers 406 unless the request accepts a compressed response, so
  `download.py` sends `Accept-Encoding: gzip`.
- **Cache.** The adapter caches `data/processed/gainful_employment_programs.csv.gz`
  (0.8 MB, all 8,637 programs; about 15 s to build). It writes a temporary file in the same
  folder and renames it. Delete the cache after changing `adapter.py`.

**Blocked: a second year.** The 2014 "informational" D/E rates released with the 2014
proposed rule (the 2012 GE informational rates) are only on www2.ed.gov. It answered 403
to a script on 2026-10-28, so they were not downloaded:
- https://www2.ed.gov/policy/highered/reg/hearulemaking/2012/2012-informational-rates033114-508.xls
- its companions in the same folder: `2012-informational-rates-variables-033114-508.xls`,
  `2012-ge-rates-variations.xls` and `2013-debt-earnings-data.xls`.

The 2011 informational rates on studentaid.gov are not a substitute. They follow the
2011 rule (repayment rate and D/E rates, informational only), and only 5,505 of 13,765
rows have D/E rates. No later official rates were issued under the 2014 rule.

## Sample and IPEDS match

| Step | Programs |
|---|---:|
| 2015 final-rates file | 8,637 |
| US institutions (drops 5 foreign) | 8,632 |
| 6-digit OPEID in IPEDS HD2016 (drops 151 programs at 49 institutions absent from the 2016-17 directory, e.g. Marinello School of Beauty, Westwood College, Everest) | 8,481 |
| program completions > 0 in 2015-16 | 7,158 |
| not all of its 2015-16 campuses closed before 2017-01-09 by IPEDS CLOSEDAT (drops 93, e.g. Sanford-Brown, Daniel Webster College, Missouri College) | 7,065 |

**Linking.** A UNITID belongs to a 6-digit OPEID if HD2016 or that year's HD file says so,
so a change of OPEID after 2016 does not end a program. A program's completions in a year
are the first-major completions (CTOTALT, MAJORNUM 1) summed over:
- its UNITIDs;
- its 6-digit CIP;
- the IPEDS award levels of its credential: undergraduate certificate -> 1, 2, 4 (20, 21
  from 2019-20); associate -> 3; bachelor's -> 5; post-baccalaureate certificate -> 6;
  master's -> 7; doctoral -> 17, 18, 19; first professional -> 18; graduate certificate
  -> 6, 8.

IPEDS uses CIP 2010 through 2018-19 (C2019_A) and CIP 2020 from 2019-20 (C2020_A). NCES's
crosswalk maps the CIP 2010 code. Only 40 programs in the sample have a CIP that moved or
was deleted (`extras["cip_changed_2020"]`). The share of programs with zero completions
rises smoothly from 2018-19 (0.226) to 2019-20 (0.275), with no jump at the CIP change.

**Match rates** (US programs):
- The institution (6-digit OPEID) is in HD2016: 0.983.
- The program has completions in 2014-15 or 2015-16: 0.887 of all US programs (0.892 of
  those in HD2016).
- The program has completions in 2015-16: 0.829 (0.844 of those in HD2016).

Of the 1,323 programs in HD2016 without 2015-16 completions:
- 237 are at institutions with no completions at all that year (closed, or stopped
  reporting).
- Of the other 1,086, 223 had completions in 2014-15 (they ended before 2015-16).
- 438 have completions in the same 4-digit CIP and level, likely a different 6-digit CIP
  in IPEDS than in the GE reporting.
- 183 have completions in the same CIP at another level.

No fallback match is used. The D/E cohort size matches the SSA file for every program.

## Readiness (2026-10-28, computed as in `multi_rule_readiness`, not written to its summary)

OLS of each score on [1, X]:

| Loader | n | Treated | R^2 | sd(T), sd(eta) | sd(eta) max/min over T deciles | Near cutoff (0.25 sd eta) |
|---|---:|---:|---:|---:|---:|---:|
| `load()`: d, among a > 8 | 1,798 | 1,591 (0.885) | 0.517 | 24.6, 23.8 | 5.06 | 0.169 |
| `load_annual()`: a, among d > 20 | 3,329 | 1,591 (0.478) | 0.539 | 3.36, 3.11 | 1.98 | 0.164 |
| `load_two_score()`: a | 7,065 | 1,591 (0.225) | 0.602 | 3.69, 3.00 | 4.21 | - |
| `load_two_score()`: d (capped) | 7,065 | | 0.458 | 27.8, 30.3 | 3.06 | - |
| `load_two_score_fail()` | 7,065 | 607 (0.086) | as above | | | |

- **The two-score view.** corr(T1, T2) = 0.569 and corr(eta1, eta2) = 0.526.
  - The readiness script's "weakest direction" (sd 3.03, 0.14 of sd(eta)) is just the
    a axis, because d has 8-10 times the spread of a. With each score scaled by its
    sd(eta), the weakest direction of T has sd 0.67, so the trade-off between the two
    rates is identified from X.
  - The script's "near boundary" share uses an equal-weight index, which does not
    describe an either or both rule.
- **Mass at the boundary.** The pass boundary has most of its mass on the annual arm:
  - 693 programs have a within 1 point of 8 with d > 20;
  - 141 have d within 2.5 points of 20 with a > 8.

  For the fail boundary, the counts are 283 (a near 12 with d > 30) and 45 (d near 30
  with a > 12). Three programs sit exactly at a = 8.00 and one at d = 30.00.
- **Does a single score make sense?** Yes as a sharp design: given a > 8, passing is
  exactly d <= 20, and given d > 20, exactly a <= 8. But the requested view, `load()` on
  d, is thin: only 207 of its 1,798 programs pass (on d), and sd(eta) varies 5-fold over
  T. `load_annual()` has twice the n, a balanced treated share, and the annual arm
  carries most of the boundary mass. Of the single-score views it is the one to use.
- **Without the cap at 100**, d has R^2 0.155 and an sd ratio of 9.7. 392 programs (5.5%)
  have d between 100 and 999.99 (small positive discretionary earnings). Another 21.3%
  sit at the coded 100 (debt with non-positive discretionary earnings). Together they
  are 52% of the not-passing programs (39% coded, 13% above 100).
  `extras["d_uncapped"]` keeps the published scale.

## Outcome

Y is measured after the release (2017-01-09): the program's IPEDS completions in 2019-20
(C2020_A, award year 2019-07-01 to 2020-06-30), against 2015-16 (C2016_A, award year
ending 2016-06-30, before the release).

| Result at release | n | Closed (Y) | dlog | Closed 2020-21 | Placebo dlog 2014-15 -> 2015-16 | All 2015-16 campuses closed by mid-2020 |
|---|---:|---:|---:|---:|---:|---:|
| pass | 5,474 | 0.216 | -0.87 | 0.239 | -0.022 | 0.088 |
| zone | 984 | 0.412 | -1.57 | 0.435 | -0.109 | 0.228 |
| fail | 607 | 0.590 | -2.20 | 0.618 | -0.136 | 0.311 |
| all | 7,065 | 0.275 | -1.08 (sd 1.58) | 0.299 | | 0.127 |

- Closure is 0.111 at public, 0.349 at for-profit and 0.370 at nonprofit institutions.
- 45% of closed programs had all their 2015-16 campuses close by mid-2020
  (`extras["campuses_closed_by_2020"]`). These are campus or institution closures; the
  largest are Brightwood College, Argosy University, Vatterott College, Everest College,
  Virginia College and several Art Institutes.
- `extras` also holds the 2020-21 closure (`y_closed_2021`), the placebo change
  2014-15 -> 2015-16 and the completions of every year 2014-15 to 2020-21.

## Literature (checked 2026-10-28)

- Kelchen, R., and Z. Liu (2022). "Did Gainful Employment Regulations Result in College
  and Program Closures?" *Education Finance and Policy* 17(3): 454-478,
  doi:10.1162/edfp_a_00340.
  - It uses a regression discontinuity design on for-profit programs' GE results.
  - Its abstract reports that passing was associated with a lower likelihood of program
    and college closure.
- Fountain, J. H. (2019). "The Effect of the Gainful Employment Regulatory Uncertainty on
  Student Enrollment at For-Profit Institutions of Higher Education." *Research in Higher
  Education* 60(8): 1065-1089, doi:10.1007/s11162-018-9533-z.
- Cellini, S. R., and N. Turner (2019). "Gainfully Employed? Assessing the Employment and
  Earnings of For-Profit College Students Using Administrative Data." *Journal of Human
  Resources* 54(2): 342-370, doi:10.3368/jhr.54.2.1016.8302r1. This is background on the
  earnings of for-profit students.
- **The 2023 rule.** "Financial Value Transparency and Gainful Employment," 88 FR 70004
  (2023-10-10, effective 2024-07-01).
  - It keeps the D/E cut points but drops the zone: pass if annual <= 8 or discretionary
    <= 20 (on median earnings), fail otherwise.
  - It adds an earnings-premium test against the state's median high-school-graduate
    earnings.
  - The adapter does not use it.

The bibliographic details come from Crossref and the Federal Register API.

## Caveats

- **Appeals.** D is the result at release. 105 programs in the sample later won an
  alternate-earnings appeal. For 70 of them (not passing) or 56 (fail), the published
  status changed between the release and the outcome. Their treatment is fuzzy.
- **The policy is censored for stricter cut points.** Transitional rates exist only for
  programs that did not pass on standard rates.
  - For a rule stricter than the deployed one (a cut below 8 or a d cut below 20), the
    final rates of programs that passed on standard rates are unknown and may be lower
    than Q.
  - In the sample, 799 programs have a_std in (6, 8] with d > 15, and 229 have d_std in
    (15, 20] with a > 6. None has transitional rates.
  - Looser rules have no such problem.
- **Mass point in d.** 21% of programs sit at d = 100 (coded). The cap at 100 is exact for
  any discretionary cutoff below 100 but leaves the mass point.
- **Weak, short-lived stakes.** No program ever lost aid. Appeal and disclosure
  deadlines were pushed back in 2017 (82 FR 13227, 82 FR 30975, 82 FR 39362). The rule
  was proposed for rescission in 2018 (83 FR 40167) and rescinded in 2019. What changed at the cut points is a public label (zone or fail)
  and an expected path to ineligibility, so effects work through institutions'
  anticipation.
- **Closure is measured by IPEDS reporting.**
  - Zero completions can reflect a teach-out, a pause, a CIP or award-level recode in
    IPEDS, or an institution that changed OPEID and UNITID.
  - 2019-20 includes the start of COVID-19.
  - Programs that fail to match (17% of US programs have no 2015-16 completions in the
    6-digit CIP) are out of the sample. Failing programs are more often unmatched (0.78
    matched vs 0.84 for passing).
- **Covariates.** Completions in 2014-15 and 2015-16 are predetermined (before the draft
  rates), but they are also the baseline of the dlog outcome. The D/E cohort size is a
  count, not a function of debt or earnings. Earnings and debt are excluded on purpose,
  since they form the rates.
- **Pre-trends.** Failing and zone programs were already shrinking (placebo dlog -0.14
  and -0.11 vs -0.02 for passing). Models of Y should use the baseline covariates.
- **Clustering.** 7,065 programs come from 2,327 institutions, and chains run many
  programs, so inference should cluster by OPEID (`extras["opeid6"]`).

## Suggested policy box (`load_two_score()`)

- **Pass / not-pass boundary:** annual cut a in **[6, 10]** and discretionary cut d in
  **[15, 25]**, around the deployed (8, 20).
  - D changes somewhere in the box for 1,626 programs (23%): 1,530 on the annual strip
    (a in (6, 10], d > 15) and 423 on the discretionary strip (d in (15, 25], a > 6).
  - Narrowing a to [7, 9] leaves 974.
  - The lower halves of both ranges are subject to the transitional-rate censoring
    above. If that matters, use [8, 10] x [20, 25].
- **Fail boundary** (`load_two_score_fail()`): a in **[10, 14]**, d in **[25, 35]**,
  around (12, 30). D changes for 631 programs (8.9%), mostly on the annual strip (607
  vs 130).
