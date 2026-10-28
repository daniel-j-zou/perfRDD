# CMS nursing-home star ratings: state-specific percentile cut points, and a staffing decision matrix

| Field | Value |
|---|---|
| **Q** | `load()`: log(1 + c2) - log(1 + x), where x is the weighted health-inspection score (deficiency points of the last three standard surveys and recent complaints; lower is better) and c2 the state's published 80th-percentile cut point for the month in which the rating was set. `load_points()`: c2 - x in points. `load_two_score()`: (RN - c_RN, total - c_total), case-mix adjusted nurse hours per resident day minus the 5-star cut points |
| **Threshold** | 0 on Q. On x: one cut point per state and month (CMS's monthly tables, 53-54 states and territories). Staffing: (0.710, 4.418) to data month 2018-03, (0.884, 4.238) to 2019-03, (1.049, 4.038) from 2019-04 |
| **Treatment** | `load()`: `1{Q >= 0}`, a health-inspection rating of 2+ stars rather than 1 (treated = the better rating). `load_23`, `load_34`, `load_45`: 3+, 4+, 5 stars. `load_two_score()`: `1{Q1 >= 0 and Q2 >= 0}`, a 5-star staffing rating |
| **Rule class** | group-specific cut points (state x month percentiles of the score, held at the month of the facility's last score change); several cut points on one score; a 5 x 5 decision matrix on two scores whose 5-star cell is a both rule |
| **X** | log certified beds, occupancy, ownership (for-profit, government, church; for-profit corporation, partnership, individual, LLC), chain and log chain size (from that year's ownership file), in a hospital, CCRC, Medicare-only, Medicaid-only, resident and family councils, change of ownership in the last 12 months, log years certified, urban (county in a CBSA), case-mix nurse hours (log total, RN share), state and year indicators (85 columns, 6 continuous). `*_lagged`: plus the score of the month before |
| **Y** | log(1 + total health-inspection score of the facility's next standard survey), as first published. `*_residents`: log residents at that next survey - log residents now |
| **n** | `load()` and the other inspection loaders: 68,431 inspection events (15,765 facilities), 2013-09 to 2019-10. `load_two_score()`: 66,322 |
| **Source** | CMS Nursing Home Compare monthly downloads 2013-08 to 2019-10 (NBER mirror, data.nber.org); CMS state cut-point tables (cms.gov zip 2008-2016, monthly PDFs 2016-2019 via the Internet Archive); Five-Star Technical Users' Guides 2012-2020 (Internet Archive) |

The unit is an **inspection event**: a facility-month in which a new standard survey becomes
cycle 1 of the score. CMS re-rates the facility that month, so the rating, its cut point and
the next survey (Y) are all defined per event.

## The rule

Every version of the Technical Users' Guide in `data/raw/guides/` (July 2012, February
2015, July 2016, January 2017, February, April and July 2018, April and October 2019, April
2020) describes the same health-inspection method (the 2018 freeze aside):

1. **Score.** Each health deficiency gets points by scope and severity (Table 1: 0 for A-C
   up to 150 for L, more for substandard quality of care). Repeat revisits add 50, 70 or 85
   percent of the cycle's points. Complaint deficiencies are assigned to 12-month periods.
   The three most recent standard-survey cycles are weighted 1/2, 1/3 and 1/6 (60/40 with
   only two). The sum is the weighted score x, published to 0.001
   (`WEIGHTED_ALL_CYCLES_SCORE`); lower is better.
2. **Stars within the state.** "The top 10 percent (with the lowest health inspection
   weighted scores) in each state receive a health inspection rating of five stars. The
   middle 70 percent of facilities receive a rating of two, three, or four stars, with an
   equal number (approximately 23.33 percent) in each rating category. The bottom 20
   percent receive a one-star rating." The cut-point tables' notes make this exact: 5 stars
   x <= P10, 4 stars <= P33.33, 3 stars <= P56.67, 2 stars <= P80, 1 star x > P80. Guam,
   the Virgin Islands (and Puerto Rico in some months) use the national distribution.
3. **Monthly cut points, held ratings.** "Rating thresholds are re-calibrated each month
   ... However, the rating for a given facility is held constant until there is a change in
   the weighted health inspection score for that facility."

So the rating is stars(x, cut points of month tau), where tau is the data month of the
facility's last change. The data show what counts as a change (see the rule check): a change
in x, or a new (or re-dated) standard survey in cycle 1 even when x happens to stay the
same. Changes inside the cycles that leave x unchanged do not re-rate.

| Data months | Health-inspection rule | Staffing table (RN and total cut points for 2-5 stars) | Guide |
|---|---|---|---|
| to 2015-01 | three cycles, monthly state cut points | RN .283/.379/.513/.710, total 3.262/3.661/4.173/4.418; April 2012 combination | July 2012 |
| 2015-02 to 2018-01 | as above | same cut points; February 2015 combination | Feb 2015, July 2016, Jan 2017 |
| 2018-02 to 2019-03 | **frozen**: re-scored in 2018-02 from the two standard surveys before 2017-11-28 (60/40) and complaints of the 24 months before; later surveys not used | from 2018-04 (PBJ data): RN .246/.383/.587/.884, total 3.176/3.552/4.010/4.238; same combination | Feb, Apr, Jul 2018 |
| 2019-04 on | three cycles again; every facility re-scored in 2019-04. From 2019-10 the abuse icon caps the rating at 2 stars | RN .317/.508/.731/1.049, total 3.108/3.580/4.038/4.408; average rounded toward the RN rating | Apr, Oct 2019 |

**Staffing.** Adjusted hours = reported / case-mix x national average, rounded to 0.001.
Each of RN and total nurse hours gets 1-5 stars; a table combines them. The **5-star cell
is a both rule in every period**: RN 5 and total 5 until 2019-03 (the guides: "To receive
an overall staffing rating of five stars, facilities must achieve a rating of five stars for
both RN and total staffing", April 2018), RN 5 and total 4+ from 2019-04. Exceptions give
1 star (PBJ era): no staffing data submitted, 7+ days without an RN (from July 2018; 4+ days
from April 2019), a failed audit. The files mark them with footnote 12, "No Staffing Data
Submitted" or "Downgraded to 1 Star".

**Overall rating** (every guide): start from the inspection stars; +1 if staffing is 4-5
stars and above the inspection rating, -1 if staffing is 1 star; +1 if QM is 5 stars, -1 if
1 star; stay within 1-5; a 1-star inspection rating can rise by at most one star; a
non-graduated Special Focus Facility gets at most 3 stars (to 2019-03), and from 2019-04 SFFs
get no ratings. The QM rating uses cut points on QM points, but QM points are not in the
downloads (only the QM stars), so the QM rule is not reproduced here.

The full staffing tables are in `adapter.STAFF_TABLES` and `extras["staffing_tables"]`.

## Rule check (2026-10-28)

`adapter.rule_check()` over all 922,449 facility-months of the 59 data months:

| Check | Facility-months | Agreement |
|---|---:|---:|
| Inspection stars (5 levels), cut points of the current month | 743,956 rated, outside the freeze | 0.9597 |
| Inspection stars, cut points of the month of the last change (tau known exactly) | 442,873 | 0.99990 (0.99997 without the abuse cap) |
| ... the same, 1-star vs 2+ boundary | 442,873 | 0.99999 (4 mismatches) |
| ... rows whose rating was set this month | 142,942 | 1.000 |
| Freeze months (2018-02 to 2019-03), tau known | 167,996 | 1.000 |
| Staffing stars from the table, excluding the 1-star exceptions | 875,898 (7,991 exceptions) | 0.99997 (30 mismatches); by table 0.99994 / 0.99997 / 0.99998 / 0.99999 |
| Staffing 5 stars = the both rule | as above | 1.000 |
| RN staffing stars | as above | 0.99996 |
| Overall rating from the published domain ratings (with the SFF cap) | 910,141 | 1.000 |

- **How the state cut points are reproduced.** Recomputing each state-month's percentiles
  from the scores in that month's file (all facilities with a score, rated or not) with SAS's
  default definition (PCTLDEF=5: the next order statistic, or the average of two when n x q
  is a whole number), q = 0.10, 0.3333, 0.56667, 0.80, gives CMS's published P10, P33.33 and
  P80 for 100% of the 3,048 state-months with 5+ facilities (within 0.002, since an average
  of two scores published to 0.001 rounds either way) and P56.67 for 99.5%. The number of
  facilities matches CMS's table in every state-month. The adapter nevertheless uses the
  published cut points.
- **The two cut-point sources agree**: the SAS files and the PDFs overlap in 2016-09 and
  2016-10 and differ by at most 0.0003 (the PDF's rounding).
- **Exceptions.** The 4 mismatches at the 1/2 boundary all follow a month missing from the
  mirror (2016-02, 2016-11, 2017-02, 2017-04), where a change and its reversal can go unseen. The
  other 5-level mismatches are 33 abuse-icon caps (2019-10) and 13 rows, all with a missing
  month between the rating month and the row. Two other readings of "a change" fit worse at
  the 1/2 boundary: re-rating only when the score changes (and no re-scoring in 2018-02)
  leaves 73 mismatches; re-rating on any change in a cycle's date or points leaves 46.
- **The current-month rule is wrong for 4% of facilities.** Held ratings are why: about a
  quarter of facilities (24.6%) have a score change in a given month (complaints aging,
  revisits, new surveys), and the others keep an older month's cut point.

**Events in the loaders** (`load()` sample, 68,431): D equals the published rating's
`1{stars >= 2}` for every event (`extras["eligible_published"]`).
- 42,479 events have the rating month known exactly (the snapshot before is the previous
  month).
- 25,952 follow a month missing from the mirror, so tau is one of two months. For 25,361 the
  two months' cut points give the same stars (`extras["tau_ambiguous"]`); the later month is
  used and Q is off by the cut point's one-month drift.
- For 591 the two months give different stars, and the published rating picks the month
  (`extras["tau_from_rating"]`); 166 of them differ at the 1/2 boundary. Using the later
  month throughout would disagree with the published rating for 63 events.

## Getting the data

    python -m experiments.datasets.nursing_homes.download          # idempotent
    python -m experiments.datasets.nursing_homes.download --check  # sizes, hashes

data.cms.gov (the Provider Data Catalog and its monthly archive) answers "Access Denied" to
scripts (checked 2026-10-27), so nothing comes from it. Everything below answered a plain
urllib request with an honest user agent on 2026-10-28. `download.py` records each file's
SHA-256 and reports a difference.

| Files | Count | Bytes | Use |
|---|---:|---:|---|
| https://data.nber.org/compare/nursing-home/YYYY/MM/DMG_CSV_DOWNLOAD<YYYYMM>01.zip, 2013/08-2015/06 | 19 | 399,281,710 | monthly Nursing Home Compare database |
| https://data.nber.org/compare/nursing-home/YYYY/M/application%252Fzip%253B%2520charset%253Dbinary, 2015/07-2019/11 | 53 | 1,450,868,136 | the same, gzip-wrapped zip |
| https://www.cms.gov/Medicare/Provider-Enrollment-and-Certification/CertificationandComplianc/Downloads/Health-Inspection-Cut-Points-2008-2016.zip | 1 | 423,672 | state cut points, SAS files, 2008-01 to 2016-10 |
| https://web.archive.org/web/<ts>id_/https://www.cms.gov/.../Downloads/cutpointstable.pdf | 39 | 6,349,908 | monthly state cut-point tables, 2016-09 to 2019-11 |
| https://web.archive.org/web/<ts>id_/https://www.cms.gov/.../Downloads/usersguide.pdf | 12 | 7,589,667 | Technical Users' Guides, July 2012 to April 2020 |
| https://data.nber.org/ssa-fips-state-county-crosswalk/2016/ssa_fips_state_county2016.csv | 1 | 179,363 | SSA county -> CBSA (urban) |

- **Snapshots.** Each archive (18.9-34.3 MB, 1.85 GB in all) is downloaded, hashed, and
  reduced to `ProviderInfo_Download.csv` and the state-averages file (every month) and
  `Ownership_Download.csv` (first snapshot of each year), stored gzip-compressed in
  `data/raw/snapshots/<YYYY_MM>/` (179 MB). The archive is then deleted (`--keep-archives`
  keeps it). `data/raw/snapshots/manifest.csv` lists each archive's URL, size, SHA-256 and
  members.
- **Which months have the underlying scores.** Every snapshot from data month 2013-08 to
  2019-10 has the weighted health-inspection score, each cycle's date and points, and the
  reported, case-mix and adjusted staffing hours, next to all the star ratings. QM points
  are never published (QM stars only). The mirror has no snapshots before 2013-08 except
  quarterly QM files (2005-2012) and stale flat files, and none after 2019-11.
- **Months.** The NBER folder month is not the data month: `FILEDATE` is. 13 folders repeat
  the previous month's data and are dropped. Data months missing from the mirror: 2013-10,
  2013-12, 2014-05, 2014-10, 2016-02, 2016-05, 2016-11, 2017-02, 2017-04, 2017-07, 2017-11,
  2018-07, 2018-11, 2019-03, 2019-06, 2019-09.
- **Not used: the `download/` folders.** `YYYY/MM/download/` (e.g.
  https://data.nber.org/compare/nursing-home/2019/12/download/NHCRatings_flatfiles.zip) hold
  the same June-2013 "flat files" in every month. They have the star ratings only, with no
  weighted score or adjusted staffing.
- **Column names.** The 2017-08 snapshot spells the columns out ("Total Weighted Health
  Survey Score"); the adapter maps them.
- **Cut points.** CMS publishes the monthly cut points on cms.gov but overwrites the PDF
  every month, so the 2016-2019 tables come from the Internet Archive's captures (one per
  month, each titled with its month). `cutpoints.py` parses them with poppler's
  `pdftotext -layout` into `state_cutpoints_2016_2019.csv` (tracked, 2,072 rows), so the
  adapter needs no PDF reader. Rebuild with
  `python -m experiments.datasets.nursing_homes.cutpoints`.
- **Caches.** `data/processed/nursing_homes_panel.parquet` (all facility-months, about 60 s
  to build) and `nursing_homes_events.parquet` (about 20 s). Both are written to a temporary
  file in the same folder and renamed. Delete them after changing `adapter.py`.

## Sample

| Step | Events |
|---|---:|
| a new standard survey becomes cycle 1 (any month) | 87,106 |
| outside the freeze (data months 2018-02 to 2019-03) | 86,286 |
| rated (score and rating published; drops 169 SFFs from 2019-04) | 86,117 |
| state cut point published | 86,117 |
| rating month known within two months | 86,106 |
| next standard survey observed by data month 2019-10 | 70,153 |
| next survey held on or after the first day of the event's data month | 68,431 |

- By year: 2013 3,535; 2014 15,527; 2015 14,539; 2016 14,256; 2017 14,362; 2018 981
  (2018-01 only); 2019 5,231, of which 5,229 are the 2019-04 re-scoring
  (`extras["freeze_return"]`).
- The last step drops 1,722 events, mostly 2019-04: CMS's cycle 1 then was a 2018 survey
  although a 2019 survey had already been held.
- `load_two_score()` uses the same events with adjusted hours and a staffing rating,
  excluding the 1-star exceptions (1,433): 66,322. It needs no rating month, so events
  after missing months are exact there. Its events all fall in two staffing periods: data
  months to 2018-01 (61,628, cut points 0.710 / 4.418) and from 2019-04 (4,694, 1.049 /
  4.038); the 2018-04 table has no events, because no new survey entered during the freeze.

## Readiness (2026-10-28, computed as in `multi_rule_readiness`, not written to its summary)

OLS of each score on [1, X]:

| Loader | n | Treated | R^2 | sd(T), sd(eta) | sd(eta) max/min over T deciles | Near cutoff (0.25 sd eta) |
|---|---:|---:|---:|---:|---:|---:|
| `load()`: log ratio, 1/2 | 68,431 | 54,652 (0.799) | 0.115 | 0.311, 0.863 | 1.31 | 0.159 |
| `load_points()`: points, 1/2 | 68,431 | 54,652 | 0.072 | 18.6, 66.5 | 1.63 | 0.191 |
| `load_lagged()`: log ratio + log(1 + score before) | 68,431 | 54,652 | 0.693 | 0.764, 0.508 | 2.40 | 0.095 |
| `load_23()` | 68,431 | 38,819 (0.567) | 0.106 | 0.296, 0.862 | 1.14 | 0.225 |
| `load_34()` | 68,431 | 23,084 (0.337) | 0.108 | 0.299, 0.862 | 1.13 | 0.199 |
| `load_45()` | 68,431 | 7,170 (0.105) | 0.136 | 0.343, 0.865 | 1.29 | 0.076 |
| `load_two_score()`: RN | 66,322 | 7,860 (0.119) | 0.417 | 0.205, 0.243 | 2.79 | - |
| `load_two_score()`: total | 66,322 | | 0.362 | 0.541, 0.718 | 2.26 | - |
| `load_two_score_lagged()`: RN, total | 66,322 | 7,860 | 0.665, 0.545 | 0.259, 0.184; 0.663, 0.607 | 3.60, 2.45 | - |

- **The inspection score is hard to predict.** Deficiency counts are noisy and the score is
  skewed (10th and 99th percentiles 10 and 357 points; 1st and 99th percentiles of c2 27 and
  175). The log ratio puts states with very different score scales on one footing and raises R^2
  from 0.07 to 0.12. State indicators add 0.02.
- **`load_lagged()`** adds the previous month's score. It shares two of the three cycles
  with the current score (the old cycles 1 and 2 become cycles 2 and 3), so most of its
  R^2 of 0.69 is mechanical: eta is then mainly the new survey. It is offered because the
  brief allows lagged values; `load()` keeps X free of any score.
- **Two-score view.** corr(T1, T2) = 0.658 and corr(eta1, eta2) = 0.494. The 5-star cell
  binds on both arms: 9.9% of events meet the RN cut and miss the total, and 13.5% meet
  the total and miss the RN.

## Outcome and consequences

- **Y** = log(1 + total score of the next standard survey), from the first month that survey
  appears. The next survey is held a median 312 days after the first day of the event's
  data month (in the sample, never before it). Mean 3.35, sd 1.37; in points, quartiles 16 /
  32 / 68 and 99th percentile 482 (`extras["y_next_score"]`). It becomes cycle 1 of the
  facility's next event, so consecutive events of one facility share data.
- **`*_residents`**: log residents at the next survey minus log residents at the event
  (mean -0.008, sd 0.14). Before PBJ, residents change with the survey: 98.7% of changes
  in the count fall in a month with a new survey.
- **What changes at the 1/2 cut point.** The inspection rating is the first step of the
  overall rating. Within 0.1 of the cut point in Q, the published overall rating averages
  1.54 stars just below and 2.52 just above: 1-star events are 1 or 2 stars overall, 2-star
  events 1 to 4 stars (the 1-star rule caps the upgrade at one star). The overall star is
  what Nursing Home Compare shows first. Its effect on demand is studied by Werner,
  Konetzka and Polsky (2016, Health Services Research 51(S2): 1291-1309,
  doi:10.1111/1475-6773.12459) and Perraillon, Konetzka, He and Werner (2019, American
  Journal of Health Economics 5(2): 165-190, doi:10.1162/ajhe_a_00115).
- **Special Focus Facilities.** The guides tie no SFF rule to the star cut points; they only
  cap (to 2019-03) or suppress (from 2019-04) SFF ratings. SFFs are far more common among
  1-star events (4.2% vs 0.16%); `extras["sff"]` flags them.
- **Staffing 5 stars** changes the overall rating only for facilities with a 4-star
  inspection rating: staffing must exceed the inspection rating, and a 4-star staffing
  rating already adds a star when the inspection rating is 3 or lower. Its consequence is
  mostly the published staffing star itself.
- A naive local-linear look at the 1/2 cut point (triangular kernel, bandwidth 0.25 in Q)
  gives -0.04 in log next score and +0.002 in log residents. The effects are small, as with
  the earlier school datasets.

## Caveats

- **Held ratings.** The rule is exact only with the month of the last change, so the
  loaders use inspection events, not all facility-months. 38% of events follow a month
  missing from the mirror (see the rule check).
- **The freeze.** No events from data months 2018-02 to 2019-03. Events in 2017-12 and
  2018-01 kept their rating only until the 2018-02 re-scoring. The 2019-04 events were
  re-scored all at once, from surveys held during the freeze.
- **Cut points move.** They are percentiles, so a state's cut point drifts every month (and
  the rule check shows ratings follow the drift). Q measures distance to the cut point that
  was deployed, not to a fixed number.
- **Ties.** Scores and cut points sit on a coarse grid; 351 events sit exactly at Q = 0
  (2+ stars, since the rule is x <= P80).
- **Covariate timing.** Ownership, beds, certification and councils are as of the event
  month; chain comes from the first snapshot of the year; urban from the FY2016 CBSA
  crosswalk. Case-mix hours are an input of the adjusted staffing hours (Q in
  `load_two_score()`), so they predict Q partly mechanically.
- **Measurement changes.** Staffing moved from CMS-671 survey data (RUG-III case mix) to PBJ
  payroll data (RUG-IV) in 2018-04; the 2019-04 events use PBJ hours and new cut points.
  Year indicators absorb level shifts only.
- **Units.** 68,431 events from 15,765 facilities (about 4.3 each), with chains and
  states; cluster by facility (`extras["provnum"]`), or by chain.
- **Outcome.** The next survey's points depend on the state's survey practice. They are
  noisy (sd of the log is 1.37) and right-skewed.

## Suggested policy boxes

- **`load()`**: a common shift t of the log-ratio cut, Q >= t, in **[-0.2, 0.2]**: the cut
  point moves to (1 + c2) e^(-t) - 1, about +-20% around each state's P80. D changes
  somewhere in the box for 14.8% of events (7.6% for [-0.1, 0.1]). In points
  (`load_points()`), **[-10, 10]** changes D for 11.2%. A percentile box (P75-P85) would
  match CMS's design better but needs each state-month's distribution (the panel has it).
- **`load_two_score()`**: RN cut in **[c_RN - 0.1, c_RN + 0.1]** and total cut in
  **[c_total - 0.3, c_total + 0.3]** (hours per resident day, around the deployed cut of the
  period). D changes somewhere for 12.7% of events: 8.6% on the RN strip, 9.0% on the total
  strip. ([-0.05, 0.05] x [-0.2, 0.2]: 7.3%.)

## Other loaders and extras

`load_residents()`, `load_two_score_residents()` (Y = log residents change),
`load_points()`, `load_lagged()`, `load_two_score_lagged()`, `load_23()`, `load_34()`,
`load_45()` (the higher inspection boundaries; abuse-capped events from 2019-10 dropped).
`extras` holds the facility (`provnum`), state, data month, the rating month
(`cut_month`), the score, the four deployed cut points, every published rating, the next
survey's date and month, the raw next score, the score 12 months later, the score before,
SFF and abuse flags, chain size, residents and beds; `load_two_score()` adds the adjusted
hours, the cut points and the staffing tables.
