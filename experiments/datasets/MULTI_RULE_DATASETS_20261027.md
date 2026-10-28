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
