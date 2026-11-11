# Binary outcomes: the four routes on the GPA and Chile data (2026-11-10)

Exploratory run at the author's request (Claude). The routes are those of
`manuscript/binary_outcome_sketch.tex`; the paper and the scope file are unchanged
(binary outcomes are outside the paper). Script:
`experiments/scripts/binary_outcome_routes.py`; outputs in the git-ignored
`experiments/runs/binary_outcome_routes/` (`summary.json`, with the U and tau curves).

Command (from `code/`):

```bash
python -m experiments.scripts.binary_outcome_routes --settings gpa chile
```

## Settings

Two settings have a binary outcome, enough data and a clean threshold on this machine.

| setting | score and threshold | outcomes (binary) | n | treated | effect covariates |
|---|---|---|---|---|---|
| GPA (Lindo, Sanders and Oreopoulos 2010) | first-year GPA minus the campus cutoff, mirrored so that probation is "treated"; `gpa_four_role.py` design (flex score, eps 0.1, policy c in [-0.6, 0.6], probation iff GPA - cutoff <= c) | graduated within 6 years (cohorts 1996-99), within 5 years, registered in fall of year 2, did not leave | 19,757 / 24,581 / 44,362 / 44,362 | 16-17% | high-school grade percentile |
| Chile PAES 2024 (university-application eligibility, Q >= 458) | best reading/math-1 average; score covariates the adapter's 12 (NEM, ranking, flags), eps 0.1, policy [420, 520] | university enrollment 2024, any enrollment 2025, university enrollment 2025 | 267,341 | 90% | NEM and ranking |

Other binary-outcome datasets in the tree were not used: `chile_top10` (n = 21k, 5%
treated), `chile_teacher_scholarship` (group cutoffs, 5% outcome), `fdic_capital` failures
(0.5% outcome), `gainful_employment` (n = 1.8k).

Routes, all four-role with the same folds and the average of the four rotations:

- `lpm`: the paper's estimator on the binary Y (linear probability model);
- `additive`: logit baseline, additive effect on the probability scale (note, Section 3),
  fitted by nonlinear least squares;
- `logit_sp`: logit on both arms (arm-wise splines in eta, shared baseline slopes,
  effect slope on the effect covariates, ridge 1e-3), the separable Chebyshev expansion in
  the link's intercept with M = 12 terms, and the paper's spline tails with the
  2M multipliers psi_k(v_1), psi_k(v_0) (note, Section 5);
- `logit_emp`: the same logit fit with empirical tails on the density fold, which is the
  pairwise double average of the note's Section 4 computed through the expansion.

The expansion's uniform error on each setting's own ranges of c and v was at most 5e-8.

## Results

Every route selects a boundary of the policy set in every setting, and the pooled
criterion has no interior local maximum anywhere. "eff@cut" is the model's mean effect
for units within 0.05 sd(Q) of the deployed threshold; tau is the marginal net effect at
the lower end of the policy set, at the deployed threshold and at the upper end.

| setting | outcome | route | optimum | tau(lo) | tau(phi0) | tau(hi) | eff@cut | gain over deployed |
|---|---|---|---|---|---|---|---|---|
| GPA | grad6 | lpm | c = -0.6 (fewest on probation) | +0.089 | -0.065 | -0.229 | -0.064 | +1.15 pp |
| GPA | grad6 | additive | c = -0.6 | +0.079 | -0.028 | -0.154 | -0.022 | +0.70 pp |
| GPA | grad6 | logit_sp | c = -0.6 | +0.053 | -0.041 | -0.200 | -0.040 | +0.92 pp |
| GPA | grad6 | logit_emp | c = -0.6 | +0.053 | -0.039 | -0.198 | -0.040 | +0.93 pp |
| GPA | grad5 | lpm | c = -0.6 | +0.039 | -0.087 | -0.216 | -0.081 | +1.16 pp |
| GPA | grad5 | additive | c = -0.6 | +0.046 | -0.063 | -0.184 | -0.058 | +0.93 pp |
| GPA | grad5 | logit_sp | c = -0.6 | +0.089 | -0.048 | -0.201 | -0.043 | +0.96 pp |
| GPA | grad5 | logit_emp | c = -0.6 | +0.088 | -0.049 | -0.195 | -0.043 | +0.98 pp |
| GPA | return2 | lpm | c = -0.6 | -0.012 | -0.080 | -0.144 | -0.080 | +0.81 pp |
| GPA | return2 | additive | c = -0.6 | -0.016 | -0.082 | -0.142 | -0.079 | +0.81 pp |
| GPA | return2 | logit_sp | c = -0.6 | -0.034 | -0.083 | -0.148 | -0.088 | +0.82 pp |
| GPA | return2 | logit_emp | c = -0.6 | -0.033 | -0.083 | -0.149 | -0.088 | +0.82 pp |
| GPA | stay | lpm | c = -0.6 | +0.035 | -0.011 | -0.053 | -0.001 | +0.23 pp |
| GPA | stay | additive | c = -0.6 | +0.018 | -0.013 | -0.041 | +0.000 | +0.19 pp |
| GPA | stay | logit_sp | c = -0.6 | -0.003 | -0.021 | -0.049 | -0.019 | +0.24 pp |
| GPA | stay | logit_emp | c = -0.6 | -0.003 | -0.021 | -0.050 | -0.019 | +0.24 pp |
| Chile | univ_2024 | lpm | 420 (treat most) | +0.125 | +0.138 | +0.163 | +0.127 | +0.51 pp |
| Chile | univ_2024 | additive | 420 | +0.086 | +0.039 | -0.039 | +0.033 | +0.24 pp |
| Chile | univ_2024 | logit_sp | 420 | +0.139 | +0.161 | +0.173 | +0.147 | +0.58 pp |
| Chile | univ_2024 | logit_emp | 420 | +0.139 | +0.161 | +0.173 | +0.147 | +0.58 pp |
| Chile | any_2025 | lpm | 420 | +0.080 | +0.065 | +0.036 | +0.064 | +0.28 pp |
| Chile | any_2025 | additive | 420 | +0.026 | +0.023 | +0.016 | +0.022 | +0.09 pp |
| Chile | any_2025 | logit_sp | 420 | +0.047 | +0.044 | +0.036 | +0.044 | +0.17 pp |
| Chile | any_2025 | logit_emp | 420 | +0.047 | +0.044 | +0.036 | +0.044 | +0.17 pp |
| Chile | univ_2025 | lpm | 420 | +0.143 | +0.155 | +0.176 | +0.139 | +0.57 pp |
| Chile | univ_2025 | additive | 420 | +0.090 | +0.066 | +0.026 | +0.062 | +0.30 pp |
| Chile | univ_2025 | logit_sp | 420 | +0.114 | +0.141 | +0.153 | +0.129 | +0.50 pp |
| Chile | univ_2025 | logit_emp | 420 | +0.115 | +0.141 | +0.153 | +0.129 | +0.50 pp |

Local RD benchmarks (from the dataset notes; local linear, h = 25 for Chile, 0.6 for GPA):
Chile univ_2024 +0.077 (0.009), any_2025 -0.017 (0.012), univ_2025 +0.052 (0.011);
GPA grad6 -0.143 (0.058) above the high-school median and +0.003 (0.031) below.

The model's effect near the deployed threshold by terciles of the first effect covariate
(NEM for Chile, high-school grade percentile for GPA; local RD by NEM tercile for Chile from
the dataset notes: univ_2024 +0.050 / +0.073 / +0.118, any_2025 -0.040 / -0.017 / +0.017,
univ_2025 +0.022 / +0.053 / +0.098):

| setting | outcome | lpm | additive | logit (both routes) |
|---|---|---|---|---|
| Chile | univ_2024 | +0.103 / +0.151 / +0.175 | +0.075 / +0.013 / -0.024 | +0.149 / +0.163 / +0.121 |
| Chile | any_2025 | +0.080 / +0.053 / +0.050 | +0.022 / +0.021 / +0.029 | +0.051 / +0.045 / +0.051 |
| Chile | univ_2025 | +0.100 / +0.160 / +0.200 | +0.075 / +0.041 / +0.016 | +0.117 / +0.135 / +0.110 |
| GPA | grad6 | -0.023 / -0.064 / +0.103 | +0.018 / -0.045 / +0.050 | -0.006 / -0.045 / +0.073 |
| GPA | grad5 | -0.015 / -0.084 / -0.188 | +0.038 / -0.047 / -0.157 | +0.036 / -0.046 / -0.133 |
| GPA | return2 | -0.124 / -0.096 / -0.108 | -0.106 / -0.090 / -0.139 | -0.125 / -0.097 / -0.083 |
| GPA | stay | -0.002 / -0.005 / -0.011 | +0.007 / -0.016 / -0.047 | -0.026 / -0.022 / -0.011 |

In the Chile data the linear probability model reproduces the rising NEM gradient of the
local estimates at about twice the level; the logit flattens it (high-NEM students
enroll at high rates, so the probability effect shrinks for them), which is the ceiling
effect and moves the pattern away from the local one. The GPA tercile means near the
threshold are sensitive to the spline at the edges of J, where the high-HS-grade students
at the threshold sit (large eta-hat), and should not be read closely.

## Reading

- **The link does not change the answer.** In all seven outcomes the four routes agree on
  the boundary, and the sign of tau over the policy set is the same under the linear
  probability model and under the logit (one exception: the additive route's tau for
  Chile univ_2024 crosses zero at the top of the policy set, the wrong way, an interior
  minimum). In the GPA data tau falls with the threshold for every route, so the crossing
  of the graduation outcomes near c = -0.3 is a minimum of U, as in the log entry of
  2026-10-19. In the Chile data tau is positive everywhere for every route: eligibility
  raises enrollment for the marginal students at every threshold in [420, 520], so the
  criterion wants the lowest threshold.
- **The two logit routes agree to three decimals** on every number, as they should: the
  separable expansion with spline tails and the pairwise double average with empirical
  tails estimate the same functional, and the expansion error is below 1e-7.
- **The logit moves the level of the effect, not its pattern.** The effect at the threshold
  changes by up to a third (GPA grad6 -0.064 to -0.040; Chile any_2025 +0.064 to +0.044)
  but keeps its sign, and it does not move toward the local RD benchmark: for Chile
  univ_2024 the logit gives +0.147 against +0.077 locally, and for any_2025 +0.044
  against -0.017. So the gap between the global fits and the local estimates, which
  the Chile notes of 2026-09-29 called "consistent with the additive outcome model
  failing for a bounded outcome", is not closed by a link. The remaining suspects are
  the ones the notes also list: X independent of eta fails (sd(eta) rises from 75 to 107
  across deciles of T, with a U-shaped mean) and the outcome model in eta.
- **The additive route is the odd one out.** Its effects at the threshold are smaller and
  its tau flatter in every setting, and for Chile univ_2024 it alone produces a crossing.
  With a logit baseline and an additive effect, the effect block absorbs curvature of
  the baseline that the link does not fit; the route is a remark, not a method.
- **Ceiling effects are visible but small.** The Chile outcomes have means 0.52-0.82 and
  the logit effects are flatter in the index than the linear ones (tau(hi) - tau(lo) is
  0.034 against 0.038 for univ_2024, and the slope reverses sign for any_2025), which is
  what the note's uniqueness remark predicts, but nowhere near enough to turn tau around.

Not done: no bootstrap (each route is a point estimate on one split, seed 1), no
sensitivity to eps or to the policy set, and no check of the logit fit's calibration.

## Addendum (2026-11-11): very low birth weight, where the link matters

A third setting, built at the author's request: the NCHS 2002 birth-cohort linked
birth/infant-death file (`experiments/datasets/vlbw/`), Q = birth weight in grams, the
1500 g very-low-birth-weight line (treat below; mirrored score), Y = survival to one year,
score covariates gestation, plurality, sex and the mother's characteristics, effect
covariate gestation. Sample: reported weights 500-2500 g with a known gestation
(n = 285,610; 16.8% treated; survival 96.0%); eps = 0.1; policy set 1200-1800 g.
Runs: `--settings vlbw --cost c`, outputs in `experiments/runs/binary_outcome_routes_vlbw/`.

Local check at 1500 g (local linear, triangular kernel): death -1.5 pp (se 1.2) at
h = 50 g, -0.7 pp (0.7) at h = 100 g below the line, the sign of Almond et al. (2010).

| route | tau at 1800 g | at 1500 g | at 1200 g | optimum, c = 0 | c = 0.01 | c = 0.02 | c = 0.03 |
|---|---|---|---|---|---|---|---|
| lpm | +0.229 | +0.118 | -0.006 | 1800 (edge) | 1800 | 1800 | 1800 |
| additive | -0.012 | -0.029 | -0.047 | 1200 (edge) | 1200 | 1200 | 1200 |
| logit_sp | +0.004 | +0.021 | +0.074 | 1800 (edge) | 1653 | 1508 | 1425 |
| logit_emp | +0.004 | +0.022 | +0.074 | 1800 (edge) | 1657 | 1520 | 1425 |

tau is the survival gain of the marginal infant (c = 0); the optimum is the pooled
argmax in grams; the four rotations at c = 0.02 span 1450-1560 g. Trim level (logit_emp,
c = 0.02): 1452 g at eps = 0.05 (window 93% of the sample), 1550 g at eps = 0.15 (21%).
Bootstrap (20 full re-estimation draws over births, logit_emp, average of the rotations):
1512 g, sd 23, percentile 95% [1476, 1543] at c = 0.02; 1659 g, sd 28, [1608, 1700] at
c = 0.01; every draw interior (`bootstrap.json`).

- **This is the case the note anticipated.** Under the logit the benefit of the extra care
  grows as weight falls, +0.4 pp of survival at the 1800 g margin to +7.4 pp at 1200 g, an
  aligned pattern. With c = 0 everyone under 1800 g should be treated; a per-infant cost
  in survival units makes the crossing interior, and the deployed 1500 g line is the
  optimum for a cost of about 2 pp. The two logit routes agree within 12 g.
- **The linear probability model fails here, not just differs.** Survival is 96% on
  average, so an effect additive in gestation and the other covariates extrapolates to
  +23 pp at 1800 g and falls the wrong way; the criterion is monotone and the optimum sits
  at the edge for every cost. The additive-effect route is negative throughout. The link
  is what makes the setting work, the opposite of the GPA and Chile cases.
- **Effect by gestation tercile at the threshold (logit):** +2.8, +2.0 and -1.1 pp
  (least to most mature), as one would expect; the LPM gives +8.9, +15.4 and +28.3.
- **Caveats.** The 1500 g rule is a guideline, not a statute, and ventilation shows no
  clear jump in this one-year file. $X\perp\eta$ fails in spread (sd ratio 1.74 across
  deciles of the linear predictor; larger spread at low gestation) and in mean (-63 to
  +112 g). The 500-5000 g sample leaves no supported window under the design rules.
  Nothing drives c in the data; it is a free parameter reported as a sweep.

### The 5000 g macrosomia line in the same file (2026-11-11): not usable

Births of 3500-5500 g (n = 1,405,361), treat above 5000 g (0.3% of them), Y = survival.
The design rules give an empty supported window: J collapses (the quantile of the residual
among the treated falls below the quantile among the untreated), because with 0.3% treated
there is no range of the residual where both arms have probability 0.1. A local check at
5000 g: death -0.5 pp (mean 0.4%) at h = 100 g, ventilation -1.6 pp (mean 4.3%); the outcome is
rare and the effect small, as the candidate list expected. Dropped.
