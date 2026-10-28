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

## One step away

| Candidate | Rule class | What is needed |
|---|---|---|
| Chile program-level applications, process 2024 (DEMRE file "Postulación") | weighted sums with program-chosen weights; program-specific cutoffs | **Author download.** The DEMRE portal (https://portal-transparencia.demre.cl/portal-base-datos) is public, but its file search runs a reCAPTCHA, so it was not automated. Choose "Proceso de Admisión 2024", type "Postulación", then BUSCAR, and save the files into `chile_admission/data/raw/`. Files link by DEMRE's `ID_aux`, not `MRUN`. The MINEDUC mirror (with `MRUN`) timed out from here on 2026-10-27; it worked on 2026-09-29. |
| HMDA 2024 (`hmda/`, 2.2 GB of raw state files in hand) | both cutoffs: CLTV > 80 and loan amount > conforming limit | Build `dftest_export.csv` with `hmda/data_process.ipynb`. The limit varies by county (group cutoffs). Loan amount is close to CLTV x property value, so given X it has little noise of its own. From memory, not checked: public loan amounts are $10k-bin midpoints, which heap at the limit. |
| Kenya tracking (Duflo, Dupas & Kremer 2011) | school-specific median cutoff (group cutoffs) | Harvard Dataverse guestbook with personal details (author). Small, few covariates. |

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

1. Author: download the DEMRE 2024 "Postulación" files. Claude then builds a
   program-level adapter with the program weights and cutoffs.
2. Decide whether to build the HMDA processed file.
3. Testing needs a two-score estimator. The current pipeline handles one score, and
   `two_score_population.py` computes population quantities only.
