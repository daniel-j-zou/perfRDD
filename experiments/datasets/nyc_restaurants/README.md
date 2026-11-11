# NYC restaurant inspections: the A-grade line at 13 points (NYC Open Data 43nn-pn8j)

| Field | Value |
|---|---|
| **Q** | `SCORE` (violation points) at a `Cycle Inspection / Initial Inspection`; integer, 0-157 in the file, median 13 |
| **Threshold** | 13.5: the letter-grade rule since July 2010 is A for 0-13 points, B for 14-27, C for 28 or more |
| **Treatment** | `1{Q <= 13.5}`: graded A on the spot (treat below; `adapter.mirrored()` gives `Q = -score`, `phi0 = -13.5`) |
| **X** | 16 cuisine indicators (the 15 most common cuisines plus other), 5 borough indicators, `n_prior` (earlier inspections of the restaurant in the file), `prior_score` (score of its latest earlier scored inspection), `year`; optionally `n_critical` (critical violations cited at the inspection, `with_critical=True`, off by default: it is a component of the score, R-squared 0.78) |
| **Y** (`load`) | the restaurant's next cycle inspection (initial or re-inspection) at least 60 days later scores 14 or more (loses or fails to get an A); `load(next_kind="initial")` uses the next cycle's initial inspection only |
| **Y** (`load_exit`) | no inspection of any kind within 18 months (548 days). Survivors only: the file holds restaurants active at the record date, so closures are not in it (see below) |
| **n** | `load()`: 15,000 inspections of 13,045 restaurants, 2023-2026 (133 rows dated 2010-2022; see "Rolling window"); `load(next_kind="initial")`: 11,201; `load_exit()`: 10,145 |
| **Treated share** | 0.477 (`load`), 0.623 (`next_kind="initial"`), 0.611 (`load_exit`) |
| **Base rate of Y** | 0.349 (`load`; 0.258 treated, 0.432 control); 0.354 (`next_kind="initial"`; 0.260 / 0.511); 0.195 (`load_exit`; 0.287 / 0.052) |
| **R-squared of Q on X** | 0.136 (`load`), 0.119 (`next_kind="initial"`), 0.100 (`load_exit`); 0.781 with `n_critical` |
| **Bunching** | all 46,793 scored cycle initial inspections: 12: **5,744**, 13: **4,806**, 14: **214**, 15: **251** (10: 3,152, 11: 2,230, 16: 405, 17: 641). Re-inspections (17,144): 2,618 / 2,419 / 106 / 95. In the `load()` sample: 1,526 / 1,290 / 98 / 113 |
| **Source** | NYC DOHMH, "DOHMH New York City Restaurant Inspection Results", NYC Open Data `43nn-pn8j`, pull (RECORD DATE) of 2026-11-10; public, no login |
| **Extras** | `camis`, `inspection_date`, `grade` (as posted at the initial inspection), `closed_at_inspection`, `n_critical`, `n_viol`, `next_days`, `next_inspection_type`, `next_score` |

The bunching is the known manipulation at the line (Ho 2012, "Fudging the nudge"):
inspectors or adjudication move 14-15 point inspections to 12-13. There are 22 times more
inspections at 13 than at 14. A density test at 13.5 fails by construction, so this dataset
serves as a manipulated-score example, not as a clean design; the score at 14+ is also
revised by adjudication after the fact ("scores are updated based on adjudication
results"), so Q here is the adjudicated score, not the score written on the day.

## The grading cycle, and which inspection is the graded one

Every restaurant gets an unannounced **cycle initial inspection**. If it scores 0-13 it is
graded A and the card goes up that day; the next cycle starts about 11-13 months later. If
it scores 14 or more it is **not graded** at the initial inspection (GRADE is blank or N in
the file); it is re-inspected, nominally a few weeks later, and the letter on the card (A,
B or C) is the **re-inspection** score. A restaurant that scores 14+ at the re-inspection
may post "Grade Pending" (Z) while it contests the score at a tribunal. Restaurants that
scored B or C are inspected more often (every 5-7 and 3-5 months).

The adapter treats the **cycle initial inspection** as the graded one and its score as Q:
it is the inspection whose score alone decides, by the 13-point rule, whether the
restaurant gets an A immediately or enters the re-inspection and grade-pending track. In
the file this rule is sharp: among cycle initial inspections with a score, 27,217 of 28,171
with 0-13 points carry grade A and none with 14+ does (the rest are blank, mostly recent
inspections; in the `load()` sample 6,724 of 7,147 treated rows show A, 423 blank). The
alternative, taking the re-inspection score as Q for the 14+ group, would mix two
inspections with different selection (only the 14+ group is re-inspected) and the
re-inspection score itself is the posted grade, so the "treatment" would be the grade and
Q the same number; it is not used.

Ambiguity to keep in mind: the re-inspection is often late. Among initial inspections with
14+ points that have a later re-inspection in the file, the gap is 56 days at the first
quartile, 126 at the median and 217 at the third; only 27 percent fall within 60 days. So
with the 60-day rule of `load()`, the "next graded inspection" of a control is its own
re-inspection in 5,955 of 7,853 control rows, and a next cycle's initial inspection in
1,898; for treated rows it is the next cycle's initial inspection in 6,829 of 7,147 (318
have a re-inspection row, mostly after a later initial). The two groups are therefore
compared on different kinds of next inspection. `load(next_kind="initial")` compares the
next cycle initial inspection on both sides (n = 11,201) at the cost of dropping controls
whose next cycle is not yet in the file. The exit loader is not affected.

## Rolling window and the file's coverage

The dataset is a rolling window: for each restaurant **active on the record date**, every
inspection up to three years before its most recent inspection. Consequences:

- Closed and out-of-business restaurants are not in the file at all. `load_exit()`'s Y (no
  inspection within 18 months) is a gap in the inspection record of a restaurant that is
  still active in November 2026. With A restaurants on a 11-13 month cycle (in this file
  the median gap to the next inspection is 492 days for treated, 126 for controls, who get
  the re-inspection), Y is 0.287 for treated against 0.052 for controls: it measures the
  inspection schedule, not exits. A proper exit outcome needs archived pulls of the
  dataset (the NYC Open Data archive or the Wayback Machine) to see who disappears.
- The first inspection of every restaurant in the window has no prior score, so requiring
  `prior_score` drops it (46,793 scored cycle initial inspections, 32,884 with a prior
  score, 28,475 with a next graded inspection at least 60 days later, 15,012 with both,
  15,000 with a borough and a cuisine). The sample is therefore restaurants with at least
  two inspections in the window, almost all 2023-2026. 133 rows carry inspection dates
  from 2010 to 2022, outside a three-year window; the dataset description warns of such
  "illogical values" and they are kept (the grade rule has applied since 2010).
- Each pull is a different sample. `download.py` saves the pull under its RECORD DATE
  (`rows_20261110.csv`) and copies it to `current.csv`, which the adapter reads.

## Getting the data

```
python -m experiments.datasets.nyc_restaurants.download          # ~145 MB, one request
python -m experiments.datasets.nyc_restaurants.download --force  # refetch a newer pull
```

The script makes two requests: the view metadata
(`https://data.cityofnewyork.us/api/views/43nn-pn8j.json`) and the CSV export
(`https://data.cityofnewyork.us/api/views/43nn-pn8j/rows.csv?accessType=DOWNLOAD`), with
a one-second pause between them. Files: `data/raw/rows_20261110.csv` (144,637,702 bytes,
294,700 rows, one row per cited violation; 27 columns), `data/raw/current.csv` (a copy),
`data/raw/metadata.json` (row-update time, columns, SHA-256). The adapter collapses the
violation rows to 92,741 inspections (`camis` x date x inspection type; the score is one
number per inspection, 10,325 inspections have none) and caches
`data/processed/inspections.parquet`. Manual route: the "Export" button on the dataset
page downloads the same CSV; name it `current.csv` under `data/raw/`.

`https://data.cityofnewyork.us/robots.txt` on 2026-11-11 (the lines that apply):

```
User-agent: *
Crawl-delay: 1
Disallow: /browse?*&category=      (and the other browse / catalog / facet / page query filters)
Disallow: /OData.svc/
Disallow: /api/odata/
Disallow: /tiles/
Disallow: /views/INLINE/rows.json?*method=clustered2*
Disallow: /api/collocate*
```

`/api/views/` and `/resource/` (the SODA API) are not disallowed; the one-second crawl
delay is respected. The dataset is published under the NYC Open Data terms of use.

## Columns used (from the view metadata)

`CAMIS` (restaurant id), `BORO`, `CUISINE DESCRIPTION` (owner-provided), `INSPECTION DATE`
(1/1/1900 = not yet inspected; dropped), `ACTION` (violations cited / none / closed /
re-closed / re-opened by DOHMH), `CRITICAL FLAG` (Critical / Not Critical / Not
Applicable, per violation row), `SCORE` (total points, "updated based on adjudication
results"), `GRADE` (A, B, C, N not yet graded, Z grade pending, P pending after a
reopening), `GRADE DATE`, `RECORD DATE`, `INSPECTION TYPE` (program / type; the cycle
program is the graded one, pre-permit operational inspections are also graded but are a
new restaurant's first cycle and are left out of Q, though they count in `n_prior` and can
supply `prior_score`).

## Design notes

- The rule treats below the threshold; `adapter.mirrored()` flips the sign.
- Points per violation are not in the file, only the critical flag; `n_critical` is the
  count of critical violation rows and is part of the score, hence off by default.
- The cuisine and borough indicator sets each sum to one, so X with an intercept is rank
  deficient; the R-squared above uses a least-squares fit that handles it. Drop
  `cuisine_other` and one borough for a full-rank design.
- Meltzer, Rothbart et al. (Syracuse CPR WP 202, 2017) find no lasting revenue effect of
  grades; Rothbart, Silver and Bae study adjudication at the lines. The policy question
  here is where the A line should sit, with the next inspection's score as the outcome.
