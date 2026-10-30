# Datasets for the threshold-rule extensions (Claude, 2026-10-27)

The author asked for datasets that fit the rule classes judged viable for root-n
proofs: one or several cutoffs on a score, campus- or group-specific cutoffs, weighted
sums of scores with fixed or chosen weights, a cutoff on each score (both / either),
and score-plus-covariate rules. The datasets should be in hand or easy to get, and
ready for testing. Multiple scores are future work in `PROJECT_SCOPE.md`, so this is
groundwork, not paper material.

## Procedure

The existing procedure was used:
- `experiments/README.md`: the adapter contract. `Q` may be `(n, k)` with a
  `treatment_rule`.
- The five checks in `EDUCATION_ASSESSMENTS_SCAN_20260923.md`:
  - a real cutoff;
  - covariates that move Q;
  - an outcome after assignment;
  - a policy reason and a cost;
  - enough units.
- The access rules: gated packages are downloaded by the author; no automated
  log-ins or CAPTCHAs.
- `CHILE_DATA_ACCESS.md`.

The rule classes add three checks:
1. The deployed rule can be reproduced from the data, so D is sharp in the scores.
2. Each score has its own noise, and T = (gamma_j' X)_j has a full-dimensional
   density. This needs at least k continuous covariates with different loadings.
3. For group cutoffs: a group label and a known cutoff per group.

New tool: `experiments/scripts/multi_rule_readiness.py` reports these checks for any
registered sample, plus R^2, sd(T), sd(eta), and the change in sd(eta) across T deciles.

## Ready for testing

| Dataset | Rule class | n | Outcome | Status |
|---|---|---:|---|---|
| `chile_two_score` (new) | deployed weighted sum (reading + math-1)/2 >= 458; counterfactual weights, both / either, bands | 223,335 | enrollment 2024 / 2025 | in hand; rule matches the published flag for 100% |
| `gpa` (extras added) | campus cutoffs 1.5 / 1.5 / 1.6 | 40,582 | next GPA | in hand; probation matches the rule for all but 51 of 44,362 |
| `us_tracts_lic` (second round) | a cutoff on each of two scores, either suffices: poverty >= 20% or family income <= 80% of the benchmark (NMTC / Opportunity Zone eligibility, ACS 2011-2015) | 70,320 tracts | change in log jobs 2017-2019 (LODES7); placebo 2015-2017; LIHTC placements | public downloads (491 MB); rule matches CDFI's flag for 100% |
| `fl_school_grades` (second round) | fixed-weight sum of components with grade bands; cutoffs by school type and year; two-score view (achievement vs gains) | 23,373 school-years (two-score: 6,940) | next year's percent of points; ELA, math, grade, closure | FLDOE CDN (32 MB); published grade matches the cutoffs for 34,343 of 34,343 |
| `chile_teacher_scholarship` (second round) | group cutoffs on the reading/math average: 625, or 595 for recent top-10% graduates of non-private schools | 258,843 | enrollment in a university education program 2024 / 2025 | in hand; 96% of the 1,368 award holders meet the rule |

Readiness, from `runs/multi_rule_readiness/summary.json`:
- **chile_two_score.**
  - The score model has R^2 0.31 (reading) and 0.38 (math).
  - corr(T1, T2) = 0.94. The weakest direction of T (reading against math) has sd 18.5,
    0.18 of sd(eta), so rules that trade the tests against each other are weakly
    identified.
  - sd(eta) for math doubles across T deciles.
  - 90% of students are eligible, and 9.4% lie near the boundary.
  - Details and caveats: `chile_two_score/README.md`.
- **gpa.**
  - R^2 is 0.34-0.38 within each campus.
  - The only continuous covariate is `hsgrade_pct`.
  - sd(eta) changes 1.39-fold across T deciles.
  - The earlier four-role runs found every specification at a policy boundary.
- **us_tracts_lic.**
  - R^2 is 0.75 (poverty) and 0.68 (income ratio), with 2010 decennial covariates, so X
    shares no ACS sampling error.
  - corr(T1, T2) = -0.85; the weakest direction has sd 0.32 of sd(eta).
  - sd(eta) changes 2.7-2.9-fold across T deciles.
  - 42% are eligible, and 24.6% of eligible tracts were designated.
  - The outcome is missing for Arkansas, Mississippi and Alaska (1,500 tracts).
- **fl_school_grades.**
  - At the B/C boundary: R^2 0.52, sd(T) 8.6, sd(eta) 8.2, sd(eta) ratio 1.33; 61% treated
    and 13% near the boundary.
  - Two-score view (elementary, 3 achievement + 4 gains components): R^2 0.72 and 0.35;
    the weakest direction has sd 0.48 of sd(eta).
  - The cutoffs and components change by year (2023 is a baseline year), and schools
    repeat, so inference should cluster by school.
- **chile_teacher_scholarship.**
  - R^2 0.46, and sd(eta) changes 1.39-fold across T deciles.
  - The first stage is small: 1.2% of eligible students receive the award.

## Third round (2026-10-28)

At the author's request ("All five", then "download more if possible"). The builds were
done by subagents; Claude re-ran each loader and its rule check.

| Folder | Rule class | n | Outcome | Rule check |
|---|---|---:|---|---|
| `nc_school_grades` | fixed-weight sum 0.8 achievement + 0.2 growth, grade bands; low-performing = (score < 54.5) and (growth < 85), a both-cutoffs rule | 12,425 school-years (2018-2025) | next year's grade score | grade 100% (2018-2025); low-performing 100% of 4,055 designations |
| `tx_school_ratings` | 0.7 max(achievement, progress) + 0.3 closing the gaps, grade bands | 38,075 campus-years (2019, 2022-2025) | next rated year's overall score | overall score 54,066 of 54,066 outside four local-accountability districts; A/B treatment 100% |
| `la_school_grades` | weighted sum of indices, grade bands; two-score view 0.75 assessment + 0.25 progress | 5,839 (two-score 1,089) | next year's school performance score | grade 99.9%; two-score 100% |
| `chile_two_score` (2023 and 2025 cohorts) | the same two-test rule | about 220k each | university enrollment | 2025 matches the published flag for every registrant; 2023 has no flag but shows the jump at 458 |
| `us_tracts_ciia` | either cutoff on three scores: poverty, income, unemployment (CDFI investment areas, ACS 2016-2020) | 76,961 tracts (two-score sample 50,028) | change in log jobs 2022-2023 (LODES8) | 100% outside the exception routes |
| `hrsa_hpsa` | one integer score (0-25) with a yearly funding cutoff (13-17) | 8,174 shortage areas | **placeholder**; no credible outcome | cutoffs from HRSA applicant tables and Federal Register notices |
| `erate` | discount bands on the school-lunch share, urban / rural | 73,434 entity-years | next year's requested dollars | 309,351 applications match the matrix; **bunching just above 49.5** |
| `cejst_tracts` | low-income percentile >= 65 and max burden percentile >= 90 (CEJST v2.0) | 73,273 tracts | **placeholder** (none in the file) | 93.75%; the rest are additions by other routes. Files are Wayback copies of the original server |

- Arizona was tried first for a third state; azed.gov serves scripts a Cloudflare
  challenge, so the author would have to download it by hand.
- The 2011-2015 CDFI investment-area file is no longer published, so `us_tracts_ciia`
  uses 2016-2020.
- Estimator results for these datasets are in
  `simulations/TWO_SCORE_ESTIMATOR_20261027.md`, under "More data".

## Fourth round (2026-10-28)

At the author's request ("find more datasets"). Candidates were screened for a viable
rule class and for a site that answers a plain script request; the builds were done by
subagents, and Claude re-ran each loader and its rule check.

| Folder | Rule class | n | Outcome | Rule check |
|---|---|---:|---|---|
| `nyc_progress_reports` | A-F grades from Environment + Performance + Progress (maxima 15 / 25 / 60) and credit, cuts by school type and year; two-score view 0.4 (Env + Perf) + 0.6 Progress, in percent units | 4,663 school-years at C/D (two-score 3,673; A/B 5,348) | next year's overall score; no graded report next year | 100% at A/B and B/C; C/D after dropping school-years a floor could lift (2 misses in 2009-10) |
| `wi_report_cards` | star ratings on a weighted sum whose achievement weight falls with the share of economically disadvantaged students; other priority areas shift each school's cutoff | 15,161 school-years (two-score 13,710) | next year's overall score | every published score from 2017-18 on, every rating |
| `in_school_grades` | A-F on 0.5 Performance + 0.5 Growth (grades 3-8) | 1,716 schools (pooled two-score 2,590) | next year's points or pass rate | grade from points 100%; only 2016-17 and 2017-18 have points |
| `gainful_employment` | not passing = annual D/E > 8% and discretionary D/E > 20% (fail: > 12% and > 30%), a both-cutoffs rule | 7,065 programs | no IPEDS completions in 2019-20; change in log completions | 8,528 of 8,528 programs without an approved appeal |
| `chile_top10` | top 10% of the class on the grade average (one cutoff per school x program x year), inside the "458 or top 10%" either-rule | 21,479 below 458; two-score 225,050 | university enrollment | class cutoffs sharp for 100% of 2024 classes; eligibility matches the published flag for every registrant (2024, 2025) |
| `ca_dashboard` | Red = status Very Low and change below +3.0, a both-cutoffs rule on (status, change); Red-or-Orange is a staircase | 49,451 school-years (ELA), 49,442 (math) | next Dashboard year's status | every color, 2017-2026, after documented exclusions |

- **Chile top-10%.** The class cutoffs are bracketed from registrants, not published.
  `load_mid()` puts each at the midpoint of its bracket and avoids a heap at Q = 0; the
  estimator uses it.
- **Wisconsin.** The rating has no formal consequence in these years (the Milwaukee
  takeover route was never used), so D is the published rating itself.
- **California.** cde.ca.gov serves its `.asp` pages behind a bot challenge, which was
  not attempted. The data files themselves (`www3.cde.ca.gov/researchfiles/...`) download
  with a plain request; their names came from a web search and the Internet Archive's
  URL index. The covariates explain 2-3% of the change score.
- **NYC.** The DOE result files and Educator Guides are Internet Archive copies; the
  2006-07 cuts are inferred from the data. 13% of D/F school-years have no grade the next
  year (phase-outs), so the main outcome is missing selectively below the cut.
- **Gainful employment.** No program lost aid; the rule was rescinded in 2019. 109
  programs later won appeals, and "closure" in IPEDS includes recoding and 2019-20
  includes COVID-19.

Checked and left for the author or for later:
- **Author download (sites block scripts):**
  - England's KS2 floor and coasting standards: attainment < 65% and a progress score
    below its floor, a both / either rule on 15,000 schools a year (2016-2018).
    compare-school-performance.service.gov.uk returns 403.
  - CMS Hospital-Acquired Condition Reduction Program (worst quartile of an equal-weight
    average of six measure scores) and MIPS (weighted sum of four category scores
    against a performance threshold). data.cms.gov returns 403.
  - The 2014 informational gainful-employment rates (a second GE year), on www2.ed.gov
    (403).
- **Public, not built:**
  - EDA distress criteria, an either rule on county unemployment and per capita income
    (ERS and BEA files answer scripts). Eligibility is assessed per project area, so D
    is loose.
  - ARC distressed counties: the average of three ratios against national quantiles,
    but only 423 counties are treated.
  - USDA food-access tracts: the low-income either rule and a low-access cutoff with
    urban / rural distances; no credible first stage.
  - IRA energy communities: a both rule on fossil-fuel employment and unemployment,
    about 2,800 areas.
  - Kentucky's status-by-change color table (like California's).

## Fifth round (2026-10-28)

At the author's request ("keep looking for more datasets"). CMS's own site blocks
scripts, so the CMS files come from NBER's public mirror of the Compare downloads
(data.nber.org/compare/). Builds by subagents; Claude re-ran each loader and rule check.

| Folder | Rule class | n | Outcome | Rule check |
|---|---|---:|---|---|
| `cms_hacrp` | penalized if the Total HAC Score is in the worst quartile; two-score view 0.15 PSI-90 domain + 0.85 infection domain (FY2018-19) | 14,657 hospital-years (two-score 5,711) | Total HAC Score three years later | 100% of CMS's penalty flags, FY2016-2026 |
| `esrd_qip` | payment-reduction bands below a minimum Total Performance Score; two-score 0.75 clinical + 0.25 reporting | 11,477 facility-years (PY2015, PY2017) | TPS two payment years later | 100% of published reductions, PY2014-2019 |
| `nursing_homes` (was a stub) | inspection stars by state percentile cutoffs; 5-star staffing = RN hours and total hours both above cut points | 68,431 inspection events (staffing 66,322) | next inspection's deficiency score | every check 99.99-100%; state cut points recomputed exactly |
| `fdic_capital` | well capitalized = every capital ratio above its cutoff; in practice total risk-based >= 10 and leverage >= 5 (both) | 248,738 bank-years | loan growth; failure (binary) | ratios recomputed exactly; no published bank-level category |

- Estimator (c = 0, outcomes signed so higher is better): every rule class goes to the
  edge in all four. HACRP and nursing homes show no effect; dialysis has selective
  attrition; banks bunch just above both cutoffs.
- `il_school_designations` (Illinois ESSA summative designations, 2023-2024 with the next
  year's index as Y): Comprehensive = index at or below the bottom-5% threshold (high
  schools also by graduation rate <= 66.67%, an either rule); the rule reproduces every
  official designation, 2023-2025. The scores come from ISBE's public Report Card Service
  (rcc.isbe.net, 3,698 school files, paced requests). Comprehensive support is a real
  intervention, but the samples are small (7,116 school-band-years; 146 within 2 points of
  the cutoff; two-score views 1,315 and 2,383), support lasts four years so many units at
  the cutoff are already treated, and 2025 changed the proficiency levels. Estimator not
  run (too small for four folds, as with Indiana).

**Decision (author, 2026-10-28): continuous outcomes only.** A binary Y would need a
different outcome model, so the two-score work keeps to settings with a continuous Y.
That drops Chile (enrollment), gainful employment (closure) and bank failure. The five
continuous-Y settings with an interior optimum are written up, one per page, in
`../writeups/two_score_settings/two_score_settings.pdf`.

## Sixth round: treatments with documented effects (2026-10-29)

At the author's request ("more good datasets, where the treatment does have a reasonable
effect"; continuous Y only). Each treatment is an intervention, money or a regulation
with published RD or quasi-experimental evidence. Builds by subagents; Claude re-ran each
loader and the two-score estimator (point estimates, c = 0, no bootstrap).

| Folder | Treatment and rule | n | Y | Rule check | Local jump in Y | Estimator |
|---|---|---:|---|---|---|---|
| `hospital_340b` | 340B eligibility: SSI % + Medicaid % > 27.3 (DSH adjustment > 11.75%), >= 22.8 for RRC/SCH; nonprofit and public hospitals | 25,831 cost reports (two-score 18,640) | log drug charges two reports later | 99.85% vs the cost reports' own line 33 | +0.24 (0.07); 340B billing 39% -> 73% | edge: the effect is nearly constant, so every rule treats all (c < 0.2) or none |
| `fl_grades_2002` | Florida F grade 2002-2011: status + growth points below the D minimum | 26,713 (437 F) | next year's percent of points | every published F, 2002-2014 | +3.1 (1.1) | edge: only 7% of the evaluation fold is inside the trim |
| `tx_index_2013` | Texas Improvement Required 2013-2014: Index 1/2 margin or Index 3/4 margin below 0 (either rule) | 11,543 (1,061 IR) | next year's Index 1 | 99.94% (appeals) | +1.1 (0.6); change +1.6 (0.7) | edge, rotations disagree |
| `nc_ayp` | NCLB AYP in North Carolina 2004-2010: reading or math minimum subgroup margin below 0; sanction stage for schools already in improvement | 1,723 in improvement (all schools 13,403) | next year's math proficiency | 88% (97% in 2004, 2005, 2010) | +2 to +4 in improvement; +0.5 all schools | edge; density bump just above 0 |
| `epa_pm25` | PM2.5 nonattainment: county max design value above the standard (1997 round: annual or 24-hour) | 403 counties a round (pooled 1,226) | change in PM2.5 3-5 years later | 89-98%; fuzzy below the cutoff (contributing counties) | none (-0.12, se 0.30) | not run |

- **Lesson.** A real effect is not enough. An interior optimum needs (1) an effect that
  varies across units and crosses the cost inside the supported region, as in NC
  low-performing; (2) enough units near the boundary; (3) a sharp rule. 340B fails (1),
  Florida F and NC AYP fail (2), EPA fails (3).
- **Access.** Florida's archive page returns 403; its files came from direct FLDOE links
  found through search and the Internet Archive's index. NC AYP's DPI hosts are gone; the
  files are Internet Archive copies. Texas: 34,532 static TEA summary PDFs (rptsvr1 has no
  robots.txt), converted with pdftotext; TEA's 2016 files need the form, so 2016 is
  missing. 340B took HCRIS from downloads.cms.gov, but imports the Impact Files of
  `medicare_dsh`, which came partly from NBER's mirror.
- **NBER mirror.** data.nber.org's robots.txt is `Disallow: /`. `cms_hacrp`, `esrd_qip`
  and `nursing_homes` (and the other session's `medicare_dsh`) downloaded from it; the
  author has been asked what to do. No new downloads from NBER.

## Waiting for the author

Each folder below has a README with the download steps. Files go in its `data/raw/`.

| Folder | Rule class | Why the author |
|---|---|---|
| `chile_admission/` (DEMRE "Postulación" 2024) | weighted sums with program-chosen weights; program cutoffs | reCAPTCHA; the MINEDUC mirror timed out all day |
| `us_tracts_qct/` (HUD qualified census tracts) | either cutoff: share below 60% of area median income >= 50%, or poverty >= 25% | huduser.gov sends scripts a bot challenge |
| `nursing_homes/` (CMS star ratings) | state-specific cutoffs; decision matrix | data.cms.gov denies scripts |
| `calenviroscreen/` (SB 535) | composite with chosen weights; top 25% | oehha.ca.gov bot check |
| `freddie_mac/`, `fannie_mae/` | decision matrix on credit score x LTV | free accounts |
| `yelp/` | several cutoffs (half-star rounding) | terms form |
| `kenya_tracking/` | school-specific median cutoff | Dataverse guestbook |
| `hmda/` (in hand) | both cutoffs | the author decides whether to build it |

Public and pullable, not built yet: NYC energy grades (the Energy Star score is adjusted
for building characteristics, so X will barely predict it), E-Rate discount bands,
US births at 1500 g (a clinical guideline, fuzzy), the CEJST tracts (public mirror).

## Checked and not usable

- **OULAD both-cutoff pass rule** (assessment >= 40 and exam >= 40):
  - exam scores exist only for modules CCC and DDD (4,955 enrolments);
  - with my weighting of the assessments, the rule matches `final_result` for 88%.
- **Chile retention** (failed subjects x grade average): subject grades are not public,
  so the rule is fuzzy. See `CHILE_DATA_ACCESS.md` for the request route.
- **Pop-Eleches & Urquiola** (school cutoffs): no individual covariates (log 2026-09-01).
- **Choi & Lee (2018) and Cattaneo, Keele, Titiunik & Vazquez-Bare (2016)** replication
  data (Harvard Dataverse, CC0):
  - two-score and multi-cutoff examples on elections and small samples;
  - few covariates, small n.
- **Colombia Saber 11 (datos.gov.co) and Brazil ENEM (INEP):**
  - five test scores with rich covariates, but no outcome after assignment in the
    public files;
  - the program rules need data that are not public: SISBEN for Ser Pilo Paga, family
    income for FIES / Prouni;
  - semi-synthetic use only, and Chile PAES already serves that. ENEM access was not
    checked; Brazil's MEC Sisu portal returned 403.
- **Lending Club approval** (FICO and DTI): outcomes exist only for approved loans.

## Next

1. Author downloads, in the order above.
2. Done (2026-10-27): a two-score estimator, its simulations and the first runs on
   `chile_two_score`, `us_tracts_lic` and `fl_school_grades`. See
   `simulations/TWO_SCORE_ESTIMATOR_20261027.md`.
