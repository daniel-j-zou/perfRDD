# Chile teacher scholarship (Beca Vocación de Profesor), admission 2024

| Field | Value |
|---|---|
| **Q** | `PROMEDIO_CM_MAX`, the best reading/math-1 PAES average, centred at the student's own cutoff |
| **Threshold** | 625, or 595 for students in the top 10% of their class who graduated in 2023 from a non-private school (group 1) |
| **Treatment** | `1{Q >= own cutoff}`: academic eligibility for the scholarship, which covers tuition in an accredited teaching degree (ITT) |
| **Rule class** | group-specific cutoffs on one score |
| **X** | `chile_admission`'s covariates with the top-10% flag, plus the group indicator |
| **Y** | `load()`: first-year enrollment in a university education program, 2024; `load_ped_2025()`: enrolled in one, 2025 |
| **First stage** | `extras["bvp_2024"]`, the 2024 award (any BVP code) |
| **n** | 258,843 registrants with a valid average and NEM who took an elective test; group 1 has 21,574 |
| **Source** | `chile_admission`'s raw files plus `Asignacion 2024_PA_PUBL.csv` (already in hand) |

## The rule

The cutoffs come from current MINEDUC/ChileAtiende pages and were checked against the
2024 awards (2026-10-27):
- 97.1% of award holders outside the top-10% route have an average >= 625.
- 90.7% of top-10% holders have an average >= 595.
- The rest came in through other routes (PACE, earlier scores).
- In the sample, 96.3% of the 1,368 award holders meet the score rule.
- The award rate is 1.21% among eligible students and 0.03% among the rest, so the
  first stage is small.

"Teaching degree" is approximated by university programs in the area "Educación",
excluding psychopedagogy. The award also requires enrolling in an accredited program and
applying for benefits.

## Readiness (2026-10-27)

`python -m experiments.scripts.multi_rule_readiness chile_teacher_scholarship`
- R^2 = 0.46 (0.45 in group 0, 0.41 in group 1).
- sd(T) = 85.7 and sd(eta) = 92.9; sd(eta) changes 1.39-fold across T deciles.
- 42% are eligible (40% in group 0, 68% in group 1), and 13.7% lie near the boundary.

## Caveats

- The outcome is rare: 4-6% enroll in an education program.
- Raw enrollment within 25 points of the cutoff is 7.0% below and 6.4% above, with no
  adjustment. A small first stage means a small ITT effect.
- Another score cutoff (800, a monthly stipend) is not coded.
