# Michigan foundation allowance (Proposal A): a yearly dose schedule on the prior-year allowance

| Field | Value |
|---|---|
| **Q** | The score: the district's prior-year foundation allowance F_{t-1} + eq_{t-1} ($ per pupil; eq = the section 22c equity payment per pupil in the four years that had one), minus the prior-year minimum allowance (`score="gap"`, default). `score="dollars"` keeps the dollar level. Read from MDE's final State Aid Financial Status Report of each fiscal year |
| **Threshold** | 0 = the prior-year minimum allowance. 77% of the sample district-years sit exactly at it |
| **Treatment** | Dose `extras["A"]` = F_t + eq_t - (F_{t-1} + eq_{t-1}) - x_t: the year's increase in the allowance beyond the basic increase x_t ($ per pupil). Reference a_ref = 0: districts at or above the basic, and every district in years with a uniform increase. `load()`'s D = 1{Q > 0} |
| **Rule class** | One schedule per fiscal year (24 in the sample). Eight "2x formula" years (a linear taper from 2x at the minimum to x near the basic, capped at the basic), three floor / catch-up years (2002, 2013, 2022), one small equity year (2007), twelve uniform years (A = 0 for all) |
| **X** | Log pupil membership (t-1), log local revenue per pupil (18 mills on non-homestead value, t-1), SAIPE child poverty rate (income year t-2), F-33 federal revenue share and Title I revenue per pupil (t-1), CCD 2015-16 shares Black / Hispanic / Asian / other, latitude and longitude, NCES locale group (EDGE 2016-17), fiscal-year indicators (+ missing flags): 44 columns. Option `county=True` adds county indicators |
| **Y** | `load_dose()`: change in current spending per pupil, Census F-33 FY t minus FY t-1 ($). Options (`outcome=`): `dinst` (instructional spending), `dcur_lead` (t+1 minus t-1), `dstrev` (state revenue per pupil, the money itself), `dtotrev` |
| **n** | 11,349 district-years, 491 local districts, FY 2001-2024 (default: hold-harmless districts and the district-years the formula does not reproduce are left out; outcome trimmed at 1/99%) |
| **Source** | MDE State Aid Financial Status Reports FY 2000-2026 (mdoe.state.mi.us/samspublic); State School Aid Act sec. 20 and 22c in each year's public act (legislature.mi.gov); HFA Fiscal Brief "Basics of the Foundation Allowance" (Feb. 25, 2026); Census F-33 FY 2000-2024; Census SAIPE 1999-2024; NCES CCD 2015-16 and EDGE 2016-17 |

**Fit: weak. The rule sticks; the spending outcomes do not.**
- **The rule is reproduced from the published allowances.** For the 12,736 non-hold-harmless
  district-years 2001-2026, the formula gives the published allowance (and equity payment)
  for 99.4%. Every year is at least 99.2% except FY 2007 (98.0%), 2008 (94.7%) and 2012
  (93.7%). The misses come from 36 districts, 26 of them missing in both 2008 and 2012. A
  likely cause, not verified district by district, is the act's district-specific add-ons
  (e.g. sec. 20(19), the former section 32e grants, built into the published allowance).
- **Many schedules, a large reference share.** 24 yearly schedules. 48% of the sample is at
  the reference dose (all districts in uniform years), so the modifier test is usable:
  r2_absorbed = 0.96.
- **The score is mostly one value.** 77% of district-years are exactly at the prior-year
  minimum. Among them the dose is the same for everybody in a year.
- **Covariates predict the score only weakly.** R^2 of Q on X is 0.35 (0.31 within years);
  sd(eta) = $201 against sd(Q) = $249.
- **The doses are small.** They are $23-125 per pupil in most years, $200 in 2002 and $418
  in 2022. The F-33 spending change has sd $579.
- **The estimates are not credible.** The estimator gives negative and unstable spending
  responses at small doses (about -$55 to -$100 at $25-60 per pupil, rotation sd $30-60). It
  extrapolates for scores above $300 at doses of $60 and up. The year-by-year gap between
  minimum and far-above districts moves by $100-200 with no relation to their dose gap. The
  design rests on these year x district-type contrasts, and spending does not separate the
  dose from them.
- **What could make it stick.** Test-score outcomes (MEAP / M-STEP, SEDA) need an author
  download (see "Author steps").

## The rule

Section 20 of the State School Aid Act (MCL 388.1620) sets each district's foundation
allowance per pupil every fiscal year (t = 2008 means 2007-08) from its prior-year
allowance. The prior-year allowance includes any section 22c equity payment of that year:
subsections 20(3)(d)-(f) "consider" the prior-year allowance to include it. In "2x" years
(sec. 20(3)(a)), for a district with M <= F' < Bp:

    F_t = min( B , F' + 2x - (x - c)(F' - M) / (B - M) ),   rounded up to the whole dollar,

and F_t = B at F' = Bp, F_t = F' + x above it. The symbols:
- F' is the prior-year allowance.
- M is the prior-year minimum (the act's "considered to be" amount).
- B and Bp are this year's and last year's basic (from 2019-20 "target") allowance.
- x is the basic increase, and c is the act's constant.

Districts whose 1994-95 allowance exceeded $6,500 (hold-harmless, sec. 20(3)(c)) get
F_{t-1} + min(x, CPI x F_{t-1}). Equity payments (sec. 22c) are
eq_t = min(cap, max(0, level - F_t)). The text of each section is
`adapter.act_section(fy, sec)`; `adapter.RULE` holds the parameters.

| FY t | Act (sec. 20) | Rule for non-hold-harmless districts | x | Dose A for a district at the minimum | Match |
|---|---|---|---:|---:|---:|
| 2001 | 2000 PA 297 | F_{t-1} + 300, rounded up | 300 | 0 (0-$1 rounding) | 0.998 |
| 2002 | 2001 PA 121 | F_{t-1} + 300; sec. 22c equity payment 6,500 - F_t (no cap) | 300 | 200 | 1.000 |
| 2003 | 2002 PA 521 | F' + 200 ("for 2002-2003 ... plus $200.00"; the basic itself rose to $6,700) | 200 | 0 | 0.996 |
| 2004, 2005 | 2003 PA 158, 2004 PA 351 | unchanged | 0 | 0 | 1.000, 0.998 |
| 2006 | 2005 PA 155 | + 175 | 175 | 0 | 0.998 |
| 2007 | 2006 PA 342 | + 210; sec. 22c: min(23, 7,360 - F_t) | 210 | 23 | 0.980 |
| 2008 | 2007 PA 137 | 7,108 <= F' < 8,385: F' + 96 - 28 (F' - 7,108)/1,325, cap 8,433; **rounded to the nearest dollar** (the act says "rounded up": 81.4% match, nearest 94.7%) | 48 | 48 | 0.947 |
| 2009 | 2008 PA 268 | 2x: x = 56, c = 20, M = 7,204, B = 8,489, Bp = 8,433 | 56 | 56 | 1.000 |
| 2010, 2011 | 2009 PA 121, 2010 PA 110 | unchanged (the $154 and $170 sec. 11d cuts were not in sec. 20) | 0 | 0 | 0.998, 1.000 |
| 2012 | 2011 PA 62 | F_{t-1} - 470 | -470 | 0 | 0.937 |
| 2013 | 2012 PA 201 | floor: max(F_{t-1}, 6,966) (sec. 20(3)(h)) | 0 | 120 | 0.992 |
| 2014 | 2013 PA 60 | 2x: x = 30, c = 10, M = 6,966, B = 8,049, Bp = 8,019; sec. 22c: min(50, 7,076 - F_t) | 30 | 80 | 0.998 |
| 2015 | 2014 PA 196 | F' + 50 for F' < 8,049, cap 8,099 ("For 2014-2015 ... plus $50.00"); sec. 22c: min(125, 7,251 - F_t) | 50 | 125 | 1.000 |
| 2016 | 2015 PA 85 | 2x: x = 70, c = 23, M = 7,251, B = 8,169, Bp = 8,099 | 70 | 70 | 0.998 |
| 2017 | 2016 PA 249 | 2x: x = 60, c = 20, M = 7,391, B = 8,229, Bp = 8,169 | 60 | 60 | 1.000 |
| 2018 | 2017 PA 108 | 2x: x = 60, c = 20, M = 7,511, B = 8,289, Bp = 8,229 | 60 | 60 | 1.000 |
| 2019 | 2018 PA 265 | 2x: x = 120, c = 40, M = 7,631, B = 8,409, Bp = 8,289 | 120 | 120 | 1.000 |
| 2020 | 2019 PA 58 | 2x: x = 120, c = 40, M = 7,871, B = 8,529, Bp = 8,409 | 120 | 120 | 1.000 |
| 2021 | 2020 PA 165 | unchanged | 0 | 0 | 1.000 |
| 2022 | 2021 PA 48 | catch-up: 8,700 for F_{t-1} <= 8,529, else + 171 | 171 | 418 | 1.000 |
| 2023, 2024 | 2022 PA 144, 2023 PA 103 | + 450, + 458 (all districts at the target) | 450, 458 | 0 | 1.000 |
| 2025, 2026 | 2024 PA 120, 2025 PA 15 | unchanged, + 442 | 0, 442 | 0 | 1.000 |

"Match" is the share of non-hold-harmless district-years whose published allowance (and
equity payment, within $1) equals the formula (`adapter.rule_check()`).

**Checks against the HFA table.**
- The lowest Q among formula districts equals the HFA "Minimum" of the prior year in every
  year 2001-2026. The minimum is 5,700 (FY 2000), 6,000, 6,500, 6,700, ..., 7,108, 7,204,
  7,316, 6,846, 6,966, 7,076, 7,251, 7,391, 7,511, 7,631, 7,871, 8,111, then 8,700 and the
  target (`adapter.HFA_MINIMUM`).
- The effective minimum includes the equity payment. The HFA "Change in Minimum" column is
  2x + the equity payment: for example $110 in FY 2014 (60 + 50) and $175 in FY 2015
  (statutory + 125 equity).

**Notes on the rule.**
- **FY 2007 equity payment.** The level is $7,360 in 2006 PA 342 sec. 22c, and it matches
  98.4% of districts. The HFA brief's footnote says $7,460, which matches 95.4%.
- **FY 2014 sec. 20 text.** The text has a stray "for 2013-2014 ... equal to the district's
  foundation allowance for the immediately preceding state fiscal year" clause. The data
  follow the 2x formula.
- **FY 2015 text.** The text also states a 2x sentence for districts "equal to the
  minimum". The data follow "+ $50" for all districts below the basic.
- **2022-2025 section 22c.** From FY 2022, sec. 22c pays $171 per pupil to "out-of-formula"
  districts (local revenue above the allowance). That is not an equity payment, so it is
  not added to the score.

**Hold-harmless districts** (52 districts with a 1994-95 allowance > $6,500) are left out by
default; `include_hh=True` adds them.
- **The CPI limit.** Their increase is min(x, CPI x F_{t-1}). The CPI rates implied by the
  published allowances are 2.2% (FY 2001), 3.4% (2002), 0.1% (2017), 1.23% (2022), 4.70%
  (2023) and 2.95% (2026); x binds in the other years (`adapter.CPI`).
- **Feasible intervals.** The rates are pinned down to a few hundredths of a basis point,
  e.g. 2002: (3.39968%, 3.39998%]. They agree with the U.S. CPI-U calendar-year changes of
  2000, 2015, 2020, 2021 and 2024 as remembered (not fetched).
- **Match rates.** With the CPI the match is 1.00 in most years. It is 0.12-0.68 in FY
  2018, 2019, 2023 and 2024, where sec. 20m "supplemental payments" of the prior year are
  added back. The 20m amounts are only partly parsed from the status reports (29 of 45
  districts in 2017).

## Readiness (2026-11-12)

`adapter.readiness(load_dose())` and the estimator log
(`experiments/runs/dose_schedule/mi_proposal_a.log`).

| Quantity | Value |
|---|---:|
| n, p, districts | 11,349, 44, 491 |
| R^2 of Q on X (all / within fiscal year) | 0.346 / 0.312 |
| sd(Q), sd(T), sd(eta) | $249, $146, $201 |
| Share at the reference dose A = 0 | 0.483 |
| Share at the minimum (Q = 0) | 0.765 |
| R^2 with county indicators | 0.442 |
| R^2 of the dollar score (`score="dollars"`, year indicators in X) | about 0.92 (0.89 from the year indicators alone) |

eta quantiles by dose (from the log):

| Dose ($/pupil) | n | eta 10% | 50% | 90% |
|---|---:|---:|---:|---:|
| 0 (reference) | 5,478 | -185 | -22 | 303 |
| 0-23 | 658 | -189 | -25 | 533 |
| 23-48 | 1,111 | -182 | -28 | 370 |
| 48-56 | 819 | -211 | -63 | 42 |
| 56-60 | 1,184 | -181 | -34 | 71 |
| 60-70 | 1,256 | -167 | -13 | 85 |
| 70-94 | 813 | -177 | -28 | 74 |
| 94-120 | 1,217 | -181 | -28 | 78 |
| 120 | 1,127 | -176 | -28 | 78 |
| 120-200 | 1,835 | -196 | -48 | 67 |
| 200-418 | 720 | -205 | -55 | 80 |

- **The reference group spans the score.** The reference dose is held at all scores
  (uniform years), and its eta range is wide.
- **Positive doses sit near the minimum.** They go mostly to districts at or near the
  minimum, whose eta 90th percentile is about $80. So doses of $60 and up are supported only
  for scores below about $300.

## Estimator results (seed 0)

**Main run.** `dose_config.config`, Y = change in current spending per pupil, FY 2001-2024.
Model: K = 5, k = 3, dose spline on [0, 418]. Effect vs A = 0 by score bin, as estimate
(rotation sd, coverage):

| Score bin (Q, $ above minimum) | a = 25 | a = 60 | a = 120 | a = 200 | a = 418 |
|---|---|---|---|---|---|
| 0 (at the minimum) | -67.7 (41.9, 97%) | -95.3 (43.1, 97%) | -6.1 (47.6, 94%) | 31.8 (4.5, 77%) | -698.8 (506.8, 95%) |
| 0.5-100 | -55.0 (58.1, 90%) | -93.5 (29.0, 90%) | -17.4 (31.8, 87%) | -32.9 (137.7, 87%) | -139.3 (451.4, 86%) |
| 100-300 | -52.8 (50.3, 97%) | -99.8 (35.9, 89%) | -100.6 (91.0, 79%) | -2,943 (4,139, 58%) | -1,100 (1,161, 82%) |
| 300-600 | -62.0 (30.7, 100%) | -165.7 (128.7, 45%) | -1,805 (2,094, 33%) | -80,113 (58,959, 15%) | -218,743 (310,583, 36%) |
| 600-1000 | -87.2 (71.6, 100%) | -2,065 (1,852, 4%) | -29,995 (27,222, 1%) | explodes (0%) | explodes (2%) |
| 1000-1500 | -180.3 (346.7, 50%) | explodes (0%) | explodes (0%) | explodes (0%) | explodes (0%) |

The mean spending change is $373 per pupil per year; sd $579.

**Score-direction modifier test.** coef 0.0010 (se 0.0029), r2_absorbed 0.964, n_fit 7,541:
not rejected.

**Variants** (logs in `experiments/runs/dose_schedule/`):

| Variant | Effects at the minimum, a = 25 / 60 / 120 | Modifier test |
|---|---|---|
| Y = state revenue per pupil change (`dstrev`, the money itself) | -10.2 (8.5) / -9.4 (7.3) / +45.0 (15.6) | -0.0031 (0.0012), r2_absorbed 0.96 |
| Y = instructional spending change (`dinst`) | -3.2 (5.3) / -10.2 (10.8) / +13.1 (30.2) | -0.0016 (0.0015) |
| Y = spending t+1 minus t-1 (`dcur_lead`, n 10,870) | -25.1 (90.7) / -81.2 (117.7) / -56.2 (31.8) | 0.0027 (0.0035) |
| FY 2001-2020 only (no COVID / ESSER years, n 9,472) | -40.8 (28.9) / -63.7 (104.8) / -0.6 (42.4) | 0.0032 (0.0026), r2_absorbed 0.94 |
| County indicators in X (R^2 0.44) | -103.4 (28.8) / -100.5 (50.4) / -25.2 (45.4) | 0.0000 (0.0024), r2_absorbed 0.97 |

**State revenue in F-33.** The state-revenue outcome does not recover the dose cleanly. A
plain within-year regression of the state-revenue change on A gives 0.62 (se 0.09,
clustered by district); the spending change gives -0.34 (0.18). F-33 state revenue
includes MPSERS pension pass-throughs and categoricals that change by year.

**Raw contrasts by year.** The difference in spending change between minimum districts and
districts more than $300 above the minimum ranges from -$226 to +$50 across years. It is
not related to the dose difference between the two groups. Examples: 2002 (dose gap 200):
+49; 2013 (gap 120): -31; 2016 (gap 31): -226; 2020 (gap 82): +2; 2022 (gap 406): -201.

## Sample and data

**Unit.** A local school district x fiscal year: codes CC000-CC599. Public school academies
(CC600-CC999) are left out, since their allowance is set from the host district's.
Intermediate districts have no allowance.

**Score and dose.**
- **Allowances.** "FY t Foundation" and "FY 1995 Foundation" come from each district's page
  of the final State Aid Financial Status Report (the August payment). The text is read with
  `pdftotext -layout`.
- **Equity payments.** The per-pupil equity payment is the "22c (FOUNDATION) EQUITY
  PAYMENT" amount divided by "State Aid Membership", rounded to the dollar (values 200, 23,
  50, 124.99).
- **Prior-year allowance.** Q_dollars = F_{t-1} + eq_{t-1} in 2003, 2008, 2015 and 2016
  (the years after an equity payment), and F_{t-1} otherwise.

**Covariates.**
- **Status reports.** Membership and local revenue per pupil: "Local Revenue Per
  Membership" from FY 2009, and 0.018 x "NHS T.V. Per Membership" before.
- **Census and NCES.** SAIPE (LEAID = 26 + district id); F-33 at t-1. The state code is
  linked to LEAID through the CCD 2015-16 directory: 542 of 550 district codes link. The 8
  that do not closed before 2015-16.
- **Left out of X.** Lagged spending (it nearly equals the allowance, so R^2 would approach
  1) and the 1994-95 allowance (the origin of the score).

**Outcome.**
- **Definition.** F-33 current spending for elementary-secondary education (TCURELSC, in
  $1,000) / fall enrollment (V33). F-33 fiscal year t = school year (t-1)-t, the same year
  as the allowance.
- **FY 2022.** elsec22.txt on the server is truncated (the first states only) and
  elsec22f.txt holds only flags, so 2022 comes from elsec22.xlsx.
- **Trimming.** The outcome is trimmed at its 1% and 99% quantiles (reorganizations,
  enrollment jumps).

**Cache.** `data/processed/status_panel.csv.gz` (status reports) and
`data/processed/mi_panel.csv.gz` (panel). Rebuild with `status_panel(rebuild=True)` and
`panel(rebuild=True)`. The pdftotext copies (`data/processed/status_txt/`, 141 MB) were
deleted to save disk; they are regenerated on demand, which needs `pdftotext` (poppler) on
the PATH.

## Getting the data

    # from code/ (never run Python inside data/)
    python -I -c "import sys; sys.path.insert(0, '.'); import runpy; runpy.run_module('experiments.datasets.mi_proposal_a.download', run_name='__main__')"
    # add sys.argv = ['x', '--check'] for sizes only, or ['x', '--only', 'status'|'acts'|'f33'|'saipe'|'ccd'|'hfa'|'mde_acc']

All files were fetched on 2026-11-12 with a generic browser user agent: no login, form,
cookie or challenge. Total 166 MB in `data/raw/`; F-33 keeps only the Michigan rows of each
national file, and the script prints the full file's SHA-256.

| Files (data/raw/) | Host and robots.txt (read 2026-11-12) | Bytes | Use |
|---|---|---:|---|
| `sams_status/ALL0013.pdf` ... `ALL2611.pdf` (27) | https://mdoe.state.mi.us/samspublic/Home/DownloadPDF?fileName=ALL0813.pdf&report=statusarchive (list: https://mdoe.state.mi.us/samspublic/Home/StatusArchiveReport). robots.txt disallows only /IdSrv/, /BaaSecure/ and a moecs path | 123,358,487 | allowances, equity payments, membership, local revenue |
| `acts/<year>-PA-<n>.htm` (27) | https://www.legislature.mi.gov/documents/2015-2016/publicact/htm/2015-PA-0085.htm etc. No robots.txt (404) | 16,211,321 | sec. 20, 22c text |
| `hfa_docs/*.pdf` (4) | www.house.mi.gov (robots: Allow /), sfa.senate.michigan.gov (robots: Allow *.pdf) | 1,078,176 | minimum / basic history |
| `f33/elsec<yy>_mi.csv` (25) | https://www2.census.gov/programs-surveys/school-finances/tables/<y>/secondary-education-finance/elsec<yy>.txt (2022: elsec22.xlsx). robots.txt restricts named crawlers only | 8,635,024 kept | spending, revenue |
| `saipe/sd<yy>-mi.{dat,txt}` (26) | https://www2.census.gov/programs-surveys/saipe/datasets/<y>/<y>-school-districts/ | 1,837,719 | child poverty |
| `ccd/` (3) | https://nces.ed.gov/ccd/data/zip/..., https://nces.ed.gov/programs/edge/data/... robots.txt disallows only search pages | 10,823,958 | LEAID link, race shares, locale |
| `mde_acc/*.xlsx` (9) | https://www.michigan.gov/mde/-/media/... (accountability / school index results 2015-16 to 2025-26). robots.txt disallows only /rss/, /podcast/, /emichigan/, /minewswire/ | 4,542,599 | not used yet (school-level index values) |

**Not fetched.**
- **www.mischooldata.org** (MEAP / M-STEP data files). Its robots.txt disallows ClaudeBot,
  Claude-Web and anthropic-ai, and its content signal is ai-train=no.
- **ed.gov** (EDFacts assessment files). It answers scripted requests with 403.

## Author steps

1. **Test-score outcome (the author's preferred continuous outcome).**
   - **Files.** Download district-level assessment files from MI School Data by hand: M-STEP
     grades 3-8 from https://www.mischooldata.org/grades-3-8-state-testing-includes-psat-data-performance,
     and the historical MEAP / MME Excel files linked from https://mischooldata.org/dashboard-home/.
   - **Location.** Save them into `experiments/datasets/mi_proposal_a/data/raw/mischooldata/`.
     The site says it revised all 2014-15 to 2024-25 data in June 2026.
2. **SEDA district achievement.** As an alternative, enter an e-mail at
   https://edopportunity.org/opportunity/data/, which accepts a data use agreement, and save
   the district CSVs into `data/raw/seda/`.

## Caveats

- **The score barely varies.** 77% of district-years are exactly at the minimum, so most
  of the score's variation comes from the 23% above it. Covariates explain 31% of the score
  within a year.
- **The doses are permanent level shifts.** A one-year dose raises all later allowances.
  Y is the same-year change in spending, so later years' outcomes mix several doses.
- **The doses are small** relative to the year-to-year noise in district spending. The 2022
  catch-up ($418 at the minimum) coincides with ESSER federal relief money.
- **The dose is revenue, not spending.** For in-formula districts the state pays the
  increase; for out-of-formula districts the allowance does not bind (their local revenue
  is higher).
- **Hold-harmless districts** (above the old state maximum) are a different group. Their
  rule needs the CPI and section 20m.
- **F-33 is nominal dollars.** Year indicators absorb the common inflation.
