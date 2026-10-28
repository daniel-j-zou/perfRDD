# Chile university-application eligibility: the top-10% route (PAES 2024)

| Field | Value |
|---|---|
| **Q** | `load()`: the high-school grade average `PROMEDIO_NOTAS` (1-7 scale, 0.025 grid) minus the class's top-10% cutoff, in grade points. `load_two_score()`: (`PROMEDIO_CM_MAX`, that difference) |
| **Threshold** | 0 on the relative grade. `load_two_score()`: (458, 0) |
| **Treatment** | `load()`: `1{Q >= 0}`, top 10% of the class, so eligible through the top-10% route. `load_two_score()`: `1{Q1 >= 458 or Q2 >= 0}`, eligible to apply to universities in the centralized system |
| **Rule class** | deployed: group-specific cutoffs, one per class, inside an either-rule. Counterfactual: a common shift of the class cutoffs, another PAES cutoff, or both |
| **X** | `chile_two_score`'s covariates without NEM and the ranking score: public / private school, technical track, non-regular program, female, years since graduation, family-income group (+ missing flag), Santiago region, north and south zones. p = 11 |
| **Y** | university enrollment in the admission year (2024) |
| **n** | `load()`: 21,479 elective takers with a PAES average below 458 (1,104 treated). `load_two_score()`: 225,050 elective takers (204,675 treated) |
| **Source** | the `chile_admission` raw files (MINEDUC open data, no registration), linked by `MRUN` |

## The rule

DEMRE's rule for admission 2024 and 2025 is

    eligible = (an elective test taken) and ((reading + math-1)/2 >= 458 or top 10% of class).

`chile_two_score` keeps the score route and drops the top-10% students. This folder
builds the top-10% route. No new download is needed. The adapter reads the registrant
files under `../chile_admission/data/raw/`. It takes the covariates and outcomes from
`chile_two_score.build()`. It caches `data/processed/classes_{cohort}.csv.gz` (one row
per class) and `top10_{cohort}.csv.gz`.

## Which score defines the flag (2026-10-28)

The flag is `PORC_SUP_NOTAS == 10`. `PORC_SUP_NOTAS` is MINEDUC's decile of the student
within the graduating class (10, 20, ..., 100). The DEMRE codebook (`ER_Alumnos_*.pdf`)
describes it as "Porcentaje Superior Nota", with MINEDUC as the source. For each
candidate score and grouping, `adapter.flag_sharpness(cohort)` reports the share of
groups where max(score | not flagged) < min(score | flagged). The shares are over the
groups that have both flagged and non-flagged registrants.

| Grouping | Score | 2024 groups | Sharp: groups | Sharp: students | Ties split by the flag |
|---|---|---:|---:|---:|---:|
| school x program x year | grade average | 9,915 | 1.000 | 1.000 | 0 |
| school x program x year | NEM | 9,915 | 1.000 | 1.000 | 0 |
| school x program x year | ranking | 9,915 | 0.896 | 0.826 | 342 |
| school x year | grade average | 8,826 | 0.977 | 0.945 | 12 |
| school x year | NEM | 8,826 | 0.978 | 0.946 | 2 |
| school x year | ranking | 8,826 | 0.849 | 0.706 | 288 |

- The class is the school (`RBD`) x teaching program (`CODIGO_ENS`, e.g. day
  humanities-science vs a technical track) x graduation year. Pooling the programs of a
  school breaks sharpness in about 2% of school-years.
- Within a class, the flag is a step function of the grade average in every 2024 class.
  For 2023 the shares are 0.9995 of classes and 0.9992 of students (5 classes overlap);
  for 2025 they are 0.9998 and 0.9999 (2 classes).
- NEM is a strictly increasing table transform of the grade average within a program, so
  it is equally sharp. The ranking score is not: it is capped at 1000 and reflects the
  schools attended. The deciles below the top are also monotone in the grade average.
  Only 0.02-0.06% of classes have a higher grade in a worse decile.
- Q uses the grade average rather than NEM: grade points are comparable across programs
  and graduation years, while NEM tables are not. The grade average is a mean of four
  yearly averages with one decimal, so it lies on a 0.025 grid, shown rounded to two
  decimals (x.x0, x.x3, x.x5, x.x8).
- Registrants with no school code (1,814 in 2024, mostly validated studies) have deciles
  but no identifiable class. They are dropped.

## The class cutoffs

The cutoffs are not published. `classes(cohort)` brackets each class's cutoff from its
registrants in the same cohort file:

    lo = max grade among the non-flagged  <  cut  <=  hi = min grade among the flagged.

`bracket_report(cohort)` summarizes the brackets. Classes in 2024, among 302,172
registrants with a grade average, decile and school:

| Status | Classes | Registrants | Kept |
|---|---:|---:|---|
| sharp (both sides, lo < hi) | 9,915 | 261,191 (86.4%) | yes |
| no flagged registrant | 17,323 | 38,989 (12.9%) | no: the cutoff has no upper bound |
| no non-flagged registrant | 1,890 | 1,992 (0.7%) | no: the cutoff has no lower bound |
| overlap | 0 | 0 | no |

- **Ties.** Two registrants of a class with the same grade average always share the flag:
  0 tie splits in 2024, 3 classes in 2023 and 1 in 2025, which are dropped as overlap.
  D = 1{grade >= cut} keeps all students tied at the lowest flagged grade treated. The
  2025 tie split disappears in NEM, so the decile is computed from an average more
  precise than the published two decimals.
- **Width.** The width is measured in steps of the 0.025 grid. Over current-graduate
  classes (4,350), the median width is 2 steps (0.05 grade points), the quartiles are 1
  and 6, and 34% of brackets are one step wide. A one-step bracket pins the cutoff down
  exactly. Registrant-weighted, the median is 2 steps and 46% sit in one-step classes.
  Classes of earlier graduation years have only a few re-takers, so their brackets are
  wide: over all sharp classes the median is 6 steps and the 90th percentile is 30.
- **Coverage.** 97.9% of current graduates are in sharp classes, against 64% of earlier
  graduates. The 2023 and 2025 files look the same.
- **Bracket, not cutoff.** MINEDUC computes the decile over the whole graduating class.
  Registrants are a subset of it, so the true cutoff lies anywhere in (lo, hi].

### Lowest flagged grade or midpoint

`load()` uses cut = hi, the lowest flagged grade, as specified. This choice is an upper
bound on the cutoff. It shifts Q down for every student in the class by hi - cut, which
lies between 0 and the width minus one step. So by construction the flagged students sit
closer to 0 than they really are. Each class's lowest flagged registrant (with any ties)
sits at exactly Q = 0. In `load()` that is 404 of the 1,104 treated students (37%),
because flagged students with a low PAES average are often the lowest flagged in their
class. Within 0.05 of the cutoff there are 163 students below and 555 above. In
`load_two_score()`, 11,782 students sit at Q2 = 0. The heap is an artefact: it would fail
a density test and pull a local fit on the treated side.

`load_mid()` and `load_two_score_mid()` use the midpoint (lo + hi) / 2. D is unchanged.
No student sits at 0, and the gap around 0 is symmetric (±width/2). The error is at most
half the width and has mean zero if the cutoff is uniform in the bracket. Within 0.05
there are 242 students below and 244 above. For one-step brackets the two choices differ
by half a step. **The midpoint is the better default for estimation.** The lowest
flagged grade matches the usual "first treated score" convention only where the bracket
is one step wide.

University enrollment in 2024 near the cutoff, `load()`: 0.189 for Q in [-0.1, 0)
(n = 391) and 0.293 for [0, 0.1] (n = 656). With the midpoint: 0.202 (550) and 0.291
(426).

## Rule check

`rule_check(cohort)`. `HABILITACION_POST` codes 1 and 2 mean eligible and 3 means not.
Code 2 turns out to be exactly "eligible only through the top-10% route".

| Check | 2024 | 2025 |
|---|---|---|
| DEMRE rule with the flag vs the published flag, all registrants with a valid average and NEM | 1.000 (267,341) | 1.000 (268,914) |
| code 2 vs (elective and top 10% and average < 458), same registrants | 1.000 (1,156 = 1,156) | 1.000 (849 = 849) |
| `load()`: D vs the published flag, and D vs code 2 | 1.000, 1.000 | 1.000, 1.000 |
| `load_two_score()`: either-rule on Q vs the published flag | 1.000, 0 mismatches | 1.000, 0 mismatches |

The 2023 file has the flag (`PORC_SUP_NOTAS`) but no published eligibility flag.
University enrollment in 2024 rises from 0.262 to 0.285 across the cutoff (within 0.1).

## Loaders

| Loader | Cohort | Y | n | Treated |
|---|---|---|---:|---:|
| `load()` | admission 2024 | university enrollment 2024 | 21,479 | 1,104 |
| `load_mid()` | same, cutoff at the bracket midpoint | same | 21,479 | 1,104 |
| `load_2025()` | admission 2025 | university enrollment 2025 | 16,824 | 822 |
| `load_2023_univ_2024()` | admission 2023 | university enrollment 2024 (next year) | 17,536 | 980 |
| `load_two_score()` | admission 2024 | university enrollment 2024 | 225,050 | 204,675 |
| `load_two_score_mid()` | same, midpoint | same | 225,050 | 204,675 |
| `load_two_score_2025()` | admission 2025 | university enrollment 2025 | 226,217 | 210,215 |
| `load_two_score_2023_univ_2024()` | admission 2023 | university enrollment 2024 | 218,448 | 201,892 |

How the 2024 sample is reached: there are 23,678 elective takers with an average below
458. The adapter drops 2,044 in classes with no flagged registrant, 109 with no school
code and 46 in classes with no non-flagged registrant. Of the 1,156 flagged, 1,104
remain. In the two-score sample, 258,843 elective takers lose 30,912 + 1,451 + 1,430,
which is 13%.

`extras` holds the following:
- `class_id`, `class_cut` and `class_bracket` (lo, hi), so another cut can be
  recomputed;
- `grade`, `paes_avg` and `top10`;
- `eligible_published` and `top10_route_published` (code 2), for 2024 and 2025;
- `cutoff`, or `cutoffs` and `rule` = "either" for the two-score loaders.

`extras` has no `weights`. The either-rule has none, so the near-boundary share in
`multi_rule_readiness` is not meaningful for `load_two_score()`.

## Score model and policy boxes (2026-10-28)

OLS of Q on [1, X]:

| Sample | R^2 | sd(T) | sd(eta) | sd(eta) max/min over T deciles |
|---|---|---|---|---|
| `load()` | 0.11 | 0.149 | 0.425 | 1.32 |
| `load_two_score()`, PAES / grade | 0.26 / 0.03 | 64.4 / 0.081 | 108.3 / 0.453 | 1.52 / 1.16 |

- corr(T1, T2) = 0.06 and corr(eta1, eta2) = 0.50 (two-score).
- How a shift relates to percentiles. In the median class, the top-20% boundary is 0.20
  grade points below the top-10% cutoff (IQR 0.12-0.28), and the top-30% boundary is
  0.37 below. Grades are compressed at the top: the median flagged current graduate is
  only 0.05 above the lowest flagged one. A shift of +0.1 keeps 48% of them, about the
  top 5%, and +0.2 keeps 24%.
- Suggested boxes:
  - `load()`: c in [-0.3, 0.1] grade points (-12 to +4 grid steps), about the top 25%
    down to the top 5% of a class. There are 1,868 students in [-0.3, 0), 656 in
    [0, 0.1] and 510 at 0.1 or above (midpoint: 2,267, 426 and 700).
  - `load_two_score()`: c1 in [420, 500] on the PAES average and c2 in [-0.3, 0.1] on
    the relative grade. c1 covers `chile_two_score`'s estimated cutoffs (416-504).

## Caveats

- **The cutoff is bracketed, not observed.** Q carries a class-level error of up to the
  bracket width. The width is larger for small classes and earlier graduation years.
- **Dropping classes selects on graduation year.** Classes without a flagged registrant
  hold 2% of current graduates but 34% of earlier ones. This is 9% of the `load()`
  candidates and 12% of the two-score sample.
- **The shift is in grade points, not percentiles.** A common shift of the class
  cutoffs is not "top x%": classes with compressed grades move more in percentile terms.
- **X has no continuous covariate.** T takes about 1,200 discrete values in `load()`, and
  R^2 is low. The class cutoff level (`extras["class_cut"]`) raises R^2 to 0.15. It is
  left out of X because it equals the own grade of each class's boundary student.
- **The top-10% route is small in the two-score sample.** It moves 1,104 of 225,050
  students (0.5%). The Q2 margin is identified only from the 9.5% with an average below
  458.
- **Brackets use one cohort file each.** Pooling the 2023-2025 files per class (the flag
  agrees for 99.95% of repeat registrants) would add 3% to `load()` and narrow the
  brackets a little. It would also leave 9 classes overlapping.
- **Inherited from `chile_two_score`.** The 2023 and 2025 income columns are constant.
  The elective-test restriction selects on behaviour after registration. D is
  eligibility (ITT).
