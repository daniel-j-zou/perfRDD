# Chile university-application eligibility as a two-score rule (PAES 2024)

| Field | Value |
|---|---|
| **Q** | `(CLEC_MAX, MATE1_MAX)`: best reading and best math-1 PAES scores over the valid sittings (100-1000) |
| **Threshold** | 458 on the average (Q1 + Q2) / 2; the published `PROMEDIO_CM_MAX` equals this average exactly |
| **Treatment** | `1{(Q1 + Q2)/2 >= 458}`: eligible to apply to universities in the centralized system |
| **Rule class** | deployed: fixed-weight weighted sum (1/2, 1/2); counterfactual: other weights, a cutoff on each test, a band |
| **X** | `chile_admission`'s covariates without the top-10% flag: NEM and ranking scores, school type, technical track, non-regular program, female, years since graduation, family-income group (+ missing flag), Santiago region |
| **Y** | `load()`: any undergraduate enrollment 2025; `load_univ_2024()`, `load_univ_2025()` |
| **n** | 223,335 registrants who took an elective test and are not in the top 10% of their class |
| **Source** | the `chile_admission` raw files (MINEDUC open data, no registration), linked by `MRUN` |

## The rule

DEMRE's 2024 eligibility rule is

    eligible = (an elective test taken) and ((reading + math-1)/2 >= 458 or top 10% of class).

Computed from the scores and the two flags, it agrees with the published
`HABILITACION_POST` (codes 1 and 2 = eligible, 3 = not) for all 267,341 registrants with
a valid average and NEM. The top-10% group is eligible at any score and the non-takers
of an elective at none, so the adapter drops both. In the remaining sample D is a sharp
function of Q (agreement with the published flag 1.000).

No new download is needed. The adapter reads the raw files under
`../chile_admission/data/raw/` and caches `data/processed/two_score_2024.csv.gz`.

## Readiness (2026-10-27)

`python -m experiments.scripts.multi_rule_readiness chile_two_score`

| Quantity | Reading | Math 1 |
|---|---:|---:|
| R^2 of Q on X | 0.312 | 0.382 |
| sd(T), sd(eta) | 71.6, 106.3 | 81.8, 104.0 |
| sd(eta) max/min across T deciles | 1.08 | 2.01 |

- corr(T1, T2) = 0.941 and corr(eta1, eta2) = 0.512.
- The weakest direction of T is about (-0.76, 0.66), reading against math. Its sd is
  18.5, 0.18 of sd(eta). A rule that trades one test against the other is therefore
  weakly identified, as in the population check
  (`../simulations/TWO_SCORE_POPULATION_20261027.md`).
- 89.9% are eligible, and 9.4% lie within 0.25 sd(eta) of the boundary.
- Continuous covariates are NEM and ranking (strongly collinear) and years since
  graduation (integer). The rest are binary.

## Caveats

- Math-1 residuals are heteroskedastic in T, so X independent of eta fails for that
  score, as in every real dataset screened so far.
- Each score is a maximum over sittings, not a single draw.
- Dropping students who took no elective test selects on behaviour after registration.
- D is eligibility (ITT). The single-score screens of `chile_admission` found boundary
  optima for every outcome.
