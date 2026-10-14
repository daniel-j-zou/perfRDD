# Score redesign on taxi and GPA (exploratory, 2026-10-14, Claude)

Context: an advisor's note ("Two-Level Treatments under Endogenous Score
Assignment", 2026-10-13) proposes a follow-up paper that chooses a whole linear
score S = kappa*eta + lambda'X and its cutoffs, not only a cutoff on the observed
score Q. The author asked how much redesigning the score would matter in the two
applications we already have. Nothing in `differing_slopes.tex` or
`PROJECT_SCOPE.md` changes.

Script: `experiments/scripts/score_redesign_explore.py` (tests:
`experiments/tests/test_score_redesign_explore.py`, 3 pass). Raw output:
`experiments/runs/score_redesign_explore/` (not tracked).

## Setup

- Effect model: the paper's differing-slopes fit, effect = alpha(eta) + X'beta2,
  fitted as in `taxi_differing_slopes.py` (spline in eta, n_treated^{1/5} knots,
  CV ridge on beta2 and the splines). An unpenalized refit is the sensitivity
  check. Binary treatment in both datasets.
- Population: the hard-trim window (eps = 0.1) with the historical rule kept
  outside it. Values are per unit in the window, relative to treating no one
  there, in outcome units.
- Rules: deployed cutoff; best cutoff on Q; best linear score 1{w'z >= phi}
  with z = (eta/sd(eta), X) and ||w|| = 1; the model optimum (treat iff the
  predicted net effect is positive), the best rule of any shape under the model.
- Honest split: rules are selected with one half's outcome model and evaluated
  with the other half's model on the other half's units; the two directions are
  averaged. The first stage uses the full sample.
- Extrapolation: at fixed eta the historical arm depends on Q only, so a unit
  switched to the other arm is |Q - cutoff| / ||gamma_X|| standardized-X units
  from the nearest covariate value at which that arm was observed for its eta.
- All values are model predictions. Point estimates only, no uncertainty.

## Taxi (restricted VTS, n = 541,021; window 177,789 rides, 33%)

Tip dollars per ride in the window, c = 0, CV ridge.

| rule | value | gain over deployed | treated | switched | extrapolation, sd (median / p90) |
|---|---:|---:|---:|---:|---:|
| deployed ($15) | 0.0822 | 0 | 16% | 0% | 0 |
| best cutoff ($12.90 in both halves) | 0.0985 | +0.0162 | 30% | 15% | 0.35 / 0.57 |
| best linear score | 0.0990 | +0.0167 | 31% | 15% | 0.35 / 0.62 |
| model optimum | 0.0990 | +0.0167 | 31% | 15% | 0.35 / 0.67 |

- The best cutoff captures 97% of the gain over the deployed rule that the model
  allows under any rule (96% unpenalized). The best cutoff reproduces the
  earlier $12.66-12.90 result.
- The best score direction is essentially Q's: (0.33, 0.94, 0.01, 0.09, 0.02)
  against (0.34, 0.94, -0.00, 0.03, -0.05) for (eta, distance, passengers, hour,
  day). Written as a covariate-dependent fare cutoff, the distance coefficient
  changes sign between halves; the only stable tilt lowers the cutoff by about
  $0.08-0.10 per later hour of the day.
- With c = $0.20 per ride nothing is gained by either rule (+0.0003 cutoff,
  +0.0007 score; best cutoff $14.90).
- Reading: the menu effect is a function of the fare level, so the fare is
  already the right score.

## GPA (Lindo et al., n = 40,582 with nextGPA; window 12,927, 32%)

Next-term GPA points among students with a recorded GPA (a post-treatment
selected outcome; probation affects leaving). The effect is predicted positive
for 96% of the window (98% unpenalized).

| cost c | rule | value | gain over deployed | treated | switched | extrap. sd (median / p90) |
|---|---|---:|---:|---:|---:|---:|
| 0 | deployed | 0.0784 | 0 | 39% | 0% | 0 |
| 0 | best cutoff (treats whole window) | 0.2286 | +0.1503 | 100% | 61% | 1.13 / 2.28 |
| 0 | best linear score | 0.2317 | +0.1533 | 96% | 64% | 1.14 / 2.30 |
| 0 | model optimum | 0.2312 | +0.1529 | 96% | 64% | 1.14 / 2.30 |
| 0.239 (median effect) | deployed | -0.0144 | 0 | 39% | 0% | 0 |
| 0.239 | best cutoff (treats none) | 0.0000 | +0.0144 | 0% | 39% | 0.71 / 1.87 |
| 0.239 | best linear score | 0.0179 | +0.0323 | 47% | 56% | 1.05 / 2.19 |
| 0.239 | model optimum | 0.0229 | +0.0373 | 51% | 55% | 1.01 / 2.19 |
| 0.295 (75th pct) | best cutoff (treats none) | 0.0000 | +0.0365 | 0% | 39% | 0.71 / 1.87 |
| 0.295 | best linear score | 0.0034 | +0.0399 | 21% | 46% | 0.85 / 2.08 |
| 0.295 | model optimum | 0.0051 | +0.0417 | 25% | 46% | 0.83 / 2.06 |

- With c = 0 the model says to treat the whole window, which a cutoff already
  does. Redesign matters only with a cost (the costs here are set at quantiles
  of the predicted effect and are illustrative).
- At the median cost the best cutoff gets 39% of the model's attainable gain and
  the best score 87%. Unpenalized: the best cutoff loses 0.0016, the best score
  gains 0.0245, the model optimum 0.0295.
- Why: the predicted effect runs against the score. In the window its
  correlation with the distance below the cutoff is -0.28; alpha falls from
  about 0.27 in mid-window to -0.16 for students furthest below their predicted
  GPA.
- Caveats, all serious: the selected rules are not monotone in GPA (they put
  some higher-GPA students on probation); their covariate weights differ
  between the two halves (campus and language in one, credits and age in the
  other), and beta2 is unstable across halves; switched students are a median
  1.0 and a 90th percentile 2.2 standardized-X units from any historically
  treated student with the same eta; and the outcome is selected by dropout.

## Reading for a follow-up paper

In these two applications, score redesign is either unnecessary (taxi: the
effect is aligned with the score) or rests on weak, selected and heavily
extrapolated evidence (GPA). A follow-up paper on score design needs an
application where the score is plausibly misaligned with the benefit and the
agency can change the formula. The share of the model's attainable gain that the
best cutoff captures (97% in taxi) is a cheap alignment diagnostic.
