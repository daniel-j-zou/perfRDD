# Medicare ESRD Quality Incentive Program: dialysis facilities' Total Performance Score, a weighted sum of measure scores with four payment-reduction cut points

| Field | Value |
|---|---|
| **Q** | `load()`, `load_bands()`: TPS - m, the published Total Performance Score minus the payment year's minimum TPS m. `load_two_score()`: (clinical domain score, reporting domain score), each 0-100 |
| **Threshold** | m = 60 in PY2015 and PY2017 (53 in PY2014, 49 in PY2018, 60 in PY2019). Cut points on the TPS at m, m - 10, m - 20, m - 30. `load_two_score()`: 0.75 Q1 + 0.25 Q2 >= 59.5 (the TPS is that index rounded half up) |
| **Treatment** | `load()`, `load_two_score()`: `1{Q >= 0}`, no payment reduction. `load_bands()`: D = number of the cut points 0, -10, -20, -30 at or below Q (0-4); the reduction is 0.5 x (4 - D) % of the facility's Medicare ESRD payments for the whole calendar year PY. The treated side is always "at or above" |
| **Rule class** | a weighted sum of measure scores (fixed domain weights, measure weights redistributed for missing measures) with several cut points on it; two domains with weights (0.75, 0.25) in the pooled years |
| **X** | for-profit; chain (DaVita, Fresenius, other chain; independent base); log dialysis stations; late shift (after 5 pm); offers in-center PD, home HD training; years since CMS certification; not in the DFC release; lagged (calendar year PY - 3) % fistula, % long-term catheter, % adult HD Kt/V >= 1.2, log adult HD patient-months, PD share of patient-months; TPS of PY - 2 (z-scored within PY); ESRD Network (18) and PY indicators. 39 columns in `load()` (35 in `load_next_py()`) |
| **Y** | `load()`, `load_bands()`, `load_two_score()`: the TPS of PY + 2, whose performance period is calendar year PY, the year the reduction applies. `load_clinical()`: the clinical domain score of PY + 2. `load_next_py()`: the TPS of PY + 1 (mostly pre-treatment) |
| **n** | `load()`, `load_bands()`, `load_two_score()`: 11,477 facility-years (PY2015 5,346; PY2017 6,131), 9,963 without a reduction. `load_clinical()`: 11,489. `load_next_py()`: 16,520 (PY2014 3,858; PY2017 6,184; PY2018 6,478), 14,202 without a reduction |
| **Source** | CMS "Complete QIP Data" PY2013-PY2019 and Dialysis Facility Compare facility files, from NBER's mirror (data.nber.org/compare/dialysis); 42 CFR 413.177-178; ESRD PPS final rules 76 FR 628, 76 FR 70228, 77 FR 67450, 78 FR 72156, 79 FR 66120, 80 FR 68968, 81 FR 77834 (federalregister.gov) |

## The rule

Section 1881(h) of the Social Security Act and 42 CFR 413.177-413.178 set up the program.
- **Scoring.** Each measure gets 0-10 points: the higher of an achievement score (rate
  against the national 15th percentile and the benchmark, the 90th) and an improvement
  score (against the facility's own rate in the improvement period). Reporting measures
  get points for complete reporting.
- **The TPS.** It is 10 x sum_i w_i s_i, rounded half up. The domain weights are set in
  each year's ESRD PPS final rule. A missing measure's weight goes to the other measures
  (rules below).
- **The reduction.** A facility with TPS below the minimum m loses 0.5% of its Medicare
  ESRD payments for every 10 points, or part of 10, below m, up to 2%:
  `reduction = min(2, 0.5 x ceil((m - TPS)/10))`. It applies to every Medicare payment in
  calendar year PY, and the facility must post its score certificate in PY.
- **Timing.** The performance period is calendar year PY - 2. Final scores come out in
  December of PY - 1.
- **The minimum TPS m.** It is the TPS a facility would get at the national median of
  each clinical measure plus a set number of reporting points. So m is fixed before the
  performance period, from earlier data, and does not depend on the scores being cut.

| PY | Performance period | Minimum TPS m (source) | Bands below m | Domains and weights | Redistribution of a missing measure | In the mirror |
|---|---|---|---|---|---|---|
| 2012 | CY2010 | 26 of 30 (76 FR 634-635) | 21-25: 0.5%; 16-20: 1%; 11-15: 1.5%; 0-10: 2% | Hgb<10 50%, Hgb>12 25%, URR 25%, x 3 | none (all three needed) | no |
| 2013 | CY2011 | 30 of 30 (76 FR 70261-62) | 26-29: 1%; 21-25: 1.5%; 0-20: 2% | Hgb>12, URR, 1.5 x each | none (both needed) | yes |
| 2014 | CY2012 | **53, from the data**; the rule only estimated 56 (76 FR 70282) | 10-point bands, 0.5-2% (76 FR 70282) | clinical 90% (Hgb>12, URR, vascular access), reporting 10% (ICH CAHPS, NHSN, mineral metabolism) (76 FR 70277) | proportional (equal weights) | yes, truncated |
| 2015 | CY2013 | 60 (77 FR 67514) | 50-59: 0.5% ... 0-29: 2% | clinical 75% (Hgb>12, Kt/V topic, vascular access topic), reporting 25% (four measures) (77 FR 67508) | proportional | yes |
| 2016 | CY2014 | 54 (78 FR 72223) | 10-point bands | clinical 75%, reporting 25%; hypercalcemia at 2/3 weight (78 FR 72217) | proportional | **no** |
| 2017 | CY2015 | 60 (79 FR 66187) | 10-point bands | clinical 75% (VAT, Kt/V, hypercalcemia at 2/3, NHSN BSI, SRR), reporting 25% (ICH CAHPS, mineral metabolism, anemia management) (79 FR 66184) | proportional | yes |
| 2018 | CY2016 | 49 (80 FR 69046) | 39-48: 0.5% ... 0-18: 2% | Clinical domain 90% (NHSN BSI 20, ICH CAHPS 20, SRR 10, Kt/V 18, VAT 18, STrR 7, hypercalcemia 7, as % of the domain), reporting 10% (five measures) (79 FR 66214-19) | even within the domain | yes |
| 2019 | CY2017 | 60 (81 FR 77894) | 10-point bands | Clinical 75% (ICH CAHPS 26, SRR 16, STrR 12, Kt/V 19, VAT 19, hypercalcemia 8), Safety 15% (NHSN BSI and dialysis-event reporting), Reporting 10% (five) (81 FR 77885-89) | even within the domain; a missing domain's weight evenly over all scored measures (fits 95%, below) | yes |
| 2020-2028 | PY - 2 | 59, 56, none (COVID), 83 (recomputed), 57, 55, 53, 56 (revised from 51), 57 | 10-point bands | four or five domains | even (83 FR 56999 on) | **no** (mirror ends 2019-12) |

The PY2020-2028 values come from the later rules: 82 FR 50766; 83 FR 57001; 86 FR 61918
(PY2022: no TPS and no reductions); 87 FR 67228 (PY2023: seven measures suppressed); 85 FR
71471; 86 FR 61926; 87 FR 67258; 88 FR 76464; 89 FR 89185, revised at 90 FR 53108-09; 90 FR
53114. They are not used by the adapter.

- **The PY2014 minimum.** The CY2012 rule only estimated 56 and said CMS would post the
  final value on dialysisreports.org. That page did not answer
  (`https://www.dialysisreports.org/pdf/esrd/public-measures/UpdatedBaseline-2014-FR.pdf`,
  TLS error on 2026-10-28). No later rule restates it.
  - The published PY2014 reductions fit m = 53 exactly: no reduction at 53-100, 0.5% at
    43-52, 1% at 33-42, 1.5% at 23-32, 2% at 0-22, with no exception.
  - The adapter uses 53. This is the only minimum not confirmed in the Federal Register.
- **Within-domain weights.** These were confirmed in the data:
  - The PY2017 hypercalcemia weight of 2/3 recomputes the published TPS for 0.999 of
    facilities. A weight of 1 gives 0.24.
  - The PY2018-2019 measure weights recompute it for every facility with all measures.

The adapter checks the rule (`adapter.rule_check()`). The texts of the rules for PY2012-2019
are in `data/raw/rule/`.

## Rule check (2026-10-28)

| PY | Facilities | Scored | Reduction from the published TPS = published reduction | TPS recomputed from the measure scores = published TPS |
|---|---:|---:|---|---|
| 2013 | 5,724 | 5,101 | 0.961 (no-reduction boundary 1.000) | 1.000 |
| 2014 | 4,146 | 4,095 | 1.000 | 1.000 |
| 2015 | 6,139 | 5,650 | 1.000 | 1.000 |
| 2017 | 6,550 | 6,229 | 1.000 | 0.999 (7 off by 1-2) |
| 2018 | 6,825 | 6,549 | 1.000 | 0.995 (34 off by +1) |
| 2019 | 7,086 | 6,822 | 1.000 | 0.948 |

- **The deployed rule is exact** on the published TPS in every PY from 2014 on. In the
  loaders' samples, D equals the published flag (`extras["eligible_published"]`) for
  every facility. That includes `load_two_score()`, where D comes from the recomputed
  domain scores. In `load_bands()` the published band agrees for every facility too.
- **PY2013 exception: the file shows 1.5% as "2%".** All 199 mismatches (TPS 21 or 24,
  which the rule puts at 1.5%) are published as "2%". The column is printed as a whole
  percent. TPS 27 is shown as "1%" and 0-18 as "2%", as in the rule. The no-reduction
  boundary is exact.
- **Recomputing the TPS.** Per-measure weights are rounded to five decimals and the sum
  is rounded half up. This matches CMS's handling of exact halves: when a domain has
  three measures (weights in thirds), an index of x.5 is published as x. With exact
  weights it would round up, which happens for 53 facilities in PY2015.
  - PY2018: 34 facilities without ICH CAHPS sit at x.50-x.51 and were rounded down.
  - PY2019: the 5% that miss are facilities with neither ICH CAHPS nor an NHSN score,
    mostly home programs. The rules do not say how the weight of a missing domain plus
    a missing clinical measure is shared. Equal sharing over all scored measures fits
    best (0.948); sharing by domain fits 0.939.
  - The loaders use the published TPS. Only the domain scores, `load_two_score()` (PY2015,
    PY2017, exact) and the PY2019 clinical score in `load_clinical()` rely on the
    reconstruction.
- **No offset in `load_two_score()`.** In PY2015 and PY2017 every scored facility has
  both domains (a TPS requires one clinical and one reporting measure), and the TPS is
  exactly 0.75 C + 0.25 R. So `extras["offset"]` is zero. Adding PY2018 (0.90 / 0.10) or
  PY2019 (a third domain, Safety 15%) would need per-year weights, or the extra domain as
  an offset: 0.15 x Safety, with the cutoff on 0.75 C + 0.10 R lowered by it.
- **Unscored facilities.** "No Score" or "Insufficient data" means too few patients,
  opened after the cutoff date, or no eligible measure. They get no reduction and are
  dropped.

## Getting the data

    python -m experiments.datasets.esrd_qip.download          # idempotent
    python -m experiments.datasets.esrd_qip.download --check  # sizes, hashes

All 13 files total 17.8 MB and go to `data/raw/` (gitignored). They were downloaded on
2026-10-28 with a plain urllib request and an honest user agent; no source blocked it.
`download.py` records each file's SHA-256 and reports a difference.

| File | Bytes | Holds |
|---|---:|---|
| https://data.nber.org/compare/dialysis/2013/07/download/DFCompare_flatfiles.zip | 1,093,759 | PY2013 Complete QIP Data (DFC 2013-03-29) |
| https://data.nber.org/compare/dialysis/2014/01/DFCompare_Revised_FlatFiles.zip | 1,245,338 | PY2014 QIP (truncated); DFC facility file 2014-01-03 |
| https://data.nber.org/compare/dialysis/2015/01/DFCompare_Revised_FlatFiles.zip | 1,487,229 | PY2015 QIP; DFC 2014-12-30 |
| https://data.nber.org/compare/dialysis/2017/2/application%252Fzip%253B%2520charset%253Dbinary | 2,795,751 | PY2017 QIP; DFC 2017-01-12 |
| https://data.nber.org/compare/dialysis/2018/1/application%252Fzip%253B%2520charset%253Dbinary | 3,139,048 | PY2018 QIP; DFC 2017-12-27 |
| https://data.nber.org/compare/dialysis/2019/2/application%252Fzip%253B%2520charset%253Dbinary | 3,189,617 | PY2019 QIP; DFC 2019-01-14 (a gzip-wrapped zip) |
| https://www.federalregister.gov/documents/full_text/text/... (7 rules, 2010-33143 to 2016-26152) | 140,669-1,058,831 | rule text, PY2012-PY2020 |

**What the mirror has.** I crawled the full tree (884 files; the `.sas7bdat`, `.dta` and
`.rdata` copies skipped). Its findings:
- The mirror holds monthly DFC snapshots for 2007-2010 and 2013-07 to 2019-12. There is no
  2011 or 2012 folder.
- From 2013 on, each snapshot's flat-file zip carries the QIP file of the current PY.
- 31 versions of the zip are distinct, by exact size. Their QIP files are:

| QIP file | Snapshots |
|---|---|
| PY2013 | from 2013-07 (stale copies in `download/` to 2014-09; also a loose `Total_Performance_Scores.csv`) |
| PY2014 | 2014-01 to 2014-12 |
| PY2015 | 2015-01 to 2017-01 |
| PY2017 | 2017-02 to 2017-12 |
| PY2018 | 2018-01 to 2019-01 |
| PY2019 | 2019-02 to 2019-06, 2019-11 |
| none | 2019-07 to 2019-10 |

- **No PY2016.** The DFC download kept the PY2015 file through 2017-01.
- **No PY2012**, since 2011-2012 were not mirrored, **and nothing after PY2019**: the
  mirror stops at 2019-12, before the PY2020 release.
- The other folders under https://data.nber.org/compare/ (2016-2020, medicare.gov,
  www.medicare.gov) are empty or hold Hospital Compare.
- The `download/DFCompare*.zip` files after mid-2015 are stale copies of the 2015 and 2013
  zips.

**Odd names and wrapping.**
- From 2015-07 the current download is stored under the literal name
  `application%2Fzip%3B charset%3Dbinary`.
- From 2018-04 (and in the 2019-02 file used here) NBER stored it gzip-compressed. The
  adapter unwraps it.
- The server sends no Content-Length for these files. I took their sizes from Apache's
  ETag and checked them against the downloads.

**PY2014 is truncated.** Every PY2014 file in the mirror (1,520,402 bytes, dated
2013-12-16) stops at CCN 392516, in Pennsylvania. It has 4,146 of about 5,900 facilities:
- states sorted by the CCN's state code up to PA, so no facility from RI, SC, SD, TN, TX,
  UT, VA, VT, WA, WI, WV, WY, PR or the territories;
- the cut is geographic and does not depend on the score.

PY2014 enters only `load_next_py()`.

**Facility characteristics.** They come from the DFC facility file of the first release
after each PY's final scores. For PY2015 and PY2017, 98.3% of the sample is in it; the
rest get column means and `not_in_dfc` = 1.

**Cache.** The adapter caches `data/processed/esrd_qip_facility_years.csv.gz` (1.4 MB,
all 36,470 facility-PY rows; about 12 s to build). It writes a temporary file in the same
folder and renames it. Delete the cache after changing `_build()` or the reading code.

**Not used.**
- data.cms.gov / data.medicare.gov: per the brief, they return 403 to scripts. I did not
  request them.
- The DFC files also carry the DFC star rating (from the January 2015 release), another
  threshold rule (percentile cut points of a composite score). It is not used.

## Sample

| Loader | PYs | Scored facilities | With Y | No reduction |
|---|---|---:|---:|---:|
| `load()`, `load_bands()`, `load_two_score()` | 2015, 2017 | 11,879 | 11,477 | 9,963 (0.868) |
| `load_clinical()` | 2015, 2017 | 11,879 | 11,489 | 9,971 |
| `load_next_py()` | 2014, 2017, 2018 | 16,873 | 16,520 | 14,202 (0.860) |

- **Facilities are matched across PYs by CCN.** A facility that changed its CCN counts as
  missing.
- **Y is missing more often after a reduction.** Share of PY2015 facilities with a PY2017
  TPS:

  | PY2015 result | Has a PY2017 TPS | Still in the PY2017 file |
  |---|---:|---:|
  | No reduction | 0.955 | 0.957 |
  | Reduction | 0.813 | 0.825 |

  For PY2017 -> PY2019 the shares are 0.987 and 0.974. Exits (closures, CCN changes) may
  be an effect of the treatment, so Y is selected.

## X

- **Characteristics at the designation.** These come from the DFC release after the final
  scores: profit status, chain organization, stations, late shift, PD and home HD
  training. Years since certification uses the QIP file's CMS certification date.
- **Chains.** Gambro is coded as DaVita and Renal Care Group as Fresenius, their
  acquirers. Other chains include ARA, US Renal Care, DCI, DSI and Satellite.
- **Lagged values.** They come from the QIP file's improvement (baseline) period,
  calendar year PY - 3, before the performance period:
  - fistula, long-term catheter and adult HD Kt/V rates in %;
  - adult HD patient-months;
  - PD share of patient-months;
  - plus the TPS of PY - 2, z-scored within PY (PY2013 on its 0-30 scale for PY2015;
    missing for 12%).
- **Lagged values in the score.** The improvement-period rates are the facility's own
  reference in the improvement score of the same PY. They are predetermined, but each one
  enters the score formula together with the performance rate.
- **Missing values** are set to the column mean, with an indicator for the lagged ones.
- **Some lags drop out.** In `load_next_py()` the Kt/V, PD and lagged-TPS columns are
  dropped, because PY2014 has none of them.
- **Not available:** patient demographics (age, race, diabetes). DFC does not publish them.

## Readiness (2026-10-28, computed as in `multi_rule_readiness`, not written to its summary)

OLS of each score on [1, X]:

| Loader | n | Treated | R^2 (within PY) | sd(T), sd(eta) | sd(eta) max/min over T deciles | Near cutoff (0.25 sd eta) |
|---|---:|---:|---|---|---:|---:|
| `load()` | 11,477 | 9,963 (0.868) | 0.433 (PY2015 0.384, PY2017 0.236) | 8.93, 10.22 | 1.39 | 0.090 |
| `load_next_py()` | 16,520 | 14,202 (0.860) | 0.411 (0.248, 0.199, 0.225) | 9.69, 11.59 | 1.20 | 0.089 |
| `load_two_score()`: clinical | 11,477 | 9,963 | 0.425 | 11.18, 13.01 | 1.30 | 0.085 (index) |
| `load_two_score()`: reporting | | | 0.238 | 4.39, 7.86 | 5.03 | |

- **Two-score correlations.** corr(T1, T2) = 0.372 and corr(eta1, eta2) = 0.132.
  - The raw weakest direction of T is the reporting axis (sd 4.0).
  - With each score scaled by its sd(eta), the weakest direction has sd 0.50.
- **The reporting score has a ceiling.** 61.8% of facility-years score 100, and the 10th
  percentile is 86.7. The clinical score spreads out (5th-95th percentile 40-93). So the
  trade-off is identified mostly from the clinical axis and from the 38% below the
  ceiling.
- **Discreteness.** The TPS is an integer, and in PY2015 its distribution is lumpy because
  measure scores are integers. Counts at TPS - 60 = -5, ..., 4:
  - PY2015: 35, 20, 19, 52, 35, 65, 30, 32, 87, 53;
  - PY2017: 75, 116, 123, 149, 145, 147, 184, 222, 141, 192.

  PY2017 shows no jump at the cutoff. PY2015's jump from 35 to 65 at the cutoff is no
  larger than its other lumps (32 to 87 between Q = 2 and 3). The measure standards are
  published before the performance period, so facilities know them, but scores are
  coarse integers.
- **Y by band** (`load_bands()`, mean TPS of PY + 2):

  | Band (D) | 0 (2%) | 1 | 2 | 3 | 4 (none) |
  |---|---:|---:|---:|---:|---:|
  | Mean TPS of PY + 2 | 48.3 | 50.9 | 56.9 | 61.0 | 68.9 |
  | n | 46 | 65 | 246 | 1,157 | 9,963 |

- **Y next to the cutoff.** Mean TPS of PY + 2 for Q in [-5, 0) against [0, 5):
  - PY2015: 61.5 (n 144) against 62.8 (n 243);
  - PY2017: 61.9 (598) against 64.5 (873).

  These are raw differences, not estimates.

## Caveats

- **Weak, partly anticipated stakes.**
  - A reduction is 0.5% of a year's Medicare ESRD payments per band: small next to the
    pre-designation differences between facilities.
  - The facility also posts a certificate with its score.
  - Facilities know the standards in advance and see preliminary scores in mid PY - 1.
- **Timing of Y.**
  - The TPS of PY + 1 (`load_next_py()`) measures calendar year PY - 1. It ends before
    the final PY scores are published and before the reduction starts, so treat it as
    mostly a placebo.
  - The TPS of PY + 2 measures calendar year PY, the year of the reduction. It is the
    first outcome fully after the designation. This is why `load()` uses it rather than
    the next PY.
- **Y changes definition.** TPS formulas and measure sets differ across PYs:
  - PY2017 adds SRR, NHSN BSI and hypercalcemia;
  - PY2019 adds ICH CAHPS, STrR and a Safety domain.

  The PY indicators absorb level shifts only. `load_clinical()` uses the clinical
  domain, which is closer to clinical care but also changes its measures.
- **Selection in Y.** Reduced facilities are less often scored two years later (above).
- **Rule changes between the pooled years.** m is 60 in both PY2015 and PY2017, and
  clinical/reporting are 0.75/0.25 in both. The measures differ, and in PY2017
  hypercalcemia has 2/3 weight. The deployed cutoff is the same, but the score is not the
  same function of care.
- **Only two PYs have Y at PY + 2.** PY2016 and PY2020 on are not in the mirror. With a
  source for PY2016 and PY2020-2021 files, the main sample could add PY2014 (Y = PY2016),
  PY2018 (Y = PY2020) and PY2019 (Y = PY2021).
- **Facility-years are not independent.** 5,279 of the 6,198 facilities appear in both
  PY2015 and PY2017, and chains run most of them, so cluster by CCN (`extras["ccn"]`) or by
  chain (`extras["chain"]`).
- **PY2013 is not in the loaders.** Its scale is 0-30 and only TPS 30 avoids a reduction.
  Its TPS enters X as a lag.

## Suggested policy boxes

- **`load()` (cutoff on Q = TPS - m).** Use **[-10, 10]**, i.e. m from 50 to 70. D
  changes somewhere in the box for 3,815 facility-years (33%): 1,157 in [-10, 0) and
  2,658 in [0, 10). A narrower **[-5, 5]** leaves 1,858 (16%).
- **`load_two_score()` (index w Q1 + (1 - w) Q2 >= c).**
  - The clinical weight was 0.75 in PY2015-2017 and PY2019 and 0.90 in PY2014 and
    PY2018. Suggested: w in **[0.6, 0.9]** and c in **[54.5, 64.5]**. D changes somewhere
    in the box for 4,008 facility-years (35%).
  - w in [0.65, 0.85] with c in [57.5, 61.5] leaves 2,356 (21%).
  - Because of the reporting ceiling, a lower w mostly shifts the index up for the 62% at
    100.
- **`load_bands()`.** The four cut points move together with m (band width 10, fixed
  from PY2014 on). A box on m as above covers them. The lower bands are thin: 357
  facility-years below -10 in total.
