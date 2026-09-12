# Assumption-swap simulation: differing slopes versus the original reduction

## Question

The hypothesis is that the differing-slopes estimator is centered when the
data-generating process satisfies

\[
W=a(\eta)+X^\top\beta_2+R_W,
\]

while the original estimator, which reduces the effect to
\(\alpha(\eta)=E(W\mid\eta)\), targets the wrong policy objective when
\(\beta_2\neq0\).  The null-interaction design provides the control case in
which both specifications are correct.

## Design

The known-target runner is
`experiments/scripts/differing_slopes_simulation.py`.  It generates

\[
X\sim N(0,I_2),\quad T=X_1,\quad \eta\sim N(0,1),\quad Q=T+\eta,
\quad D=1\{Q>0\},
\]

and

\[
Y=b_0+b_1\eta+X^\top\beta_1
  +D\{a_0+a_1\eta+X^\top\beta_2\}+\varepsilon .
\]

The baseline uses `beta2=(0.80,0.25)`, `a0=0.35`, `a1=0.90`,
`beta1=(0.30,-0.20)`, `b0=0.20`, `b1=0.60`, `sigma_eps=0.50`, cost
`c=0.25`, and hard trim probability `eps=0.10`.  The true target is computed
using the known Gaussian law of `T`; the fitted threshold maximizes the smooth
plug-in utility.  Each cell uses 300
independent replications at `n={500,1000,2000,4000}`.  A baseline follow-up
uses 200 replications at `n={8000,16000}`.

The null-interaction control sets `beta2=(0,0)`.  The strong-interaction
stress design sets `beta2=(1.60,0.50)` while leaving the rest of the DGP
unchanged.  In each replication we fit both the original alpha-only model and
the augmented model with `D*X`, and compute the plug-in delta-method variance
and normal 95% interval.  The variance calculation is conditional on the true
`eta` and fixed trim interval; it is not a test of the full generated-index
and moving-boundary theorem.

## Results

### Nonzero interaction (baseline)

The full differing-slopes target is `phi*=−0.120064`; the alpha-only formula
would imply `phi*=−0.339292` even before estimating the misspecified outcome
regression.

| n | differing-slopes bias | differing-slopes RMSE | variance ratio | coverage | original bias to full target | original coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.0014 | 0.0922 | 0.953 | 0.933 | −0.4717 | 0.947 |
| 1,000 | −0.0008 | 0.0656 | 0.939 | 0.953 | −0.4586 | 0.810 |
| 2,000 | −0.0026 | 0.0448 | 1.005 | 0.947 | −0.4621 | 0.593 |
| 4,000 | −0.0004 | 0.0327 | 0.930 | 0.947 | −0.4280 | 0.383 |
| 8,000 | 0.0016 | 0.0224 | 0.992 | 0.940 | −0.4383 | 0.080 |
| 16,000 | 0.0009 | 0.0168 | 0.875 | 0.950 | −0.4276 | 0.000 |

The differing-slopes RMSE has a log--log slope of approximately `−0.50` over
the first four sample sizes.  Its bias remains below 0.003 in absolute value.
The original estimator's variance is not the problem: its estimated variance
tracks its own Monte Carlo dispersion, but the interval is centered on the
wrong target, so coverage collapses with sample size.

### Null interaction

When `beta2=0`, both estimators target `phi*=−0.339292`.  Across
`n={500,1000,2000,4000}`, alpha-only biases are `0.0166, 0.0200, 0.0120,
0.0162`, while differing-slopes biases are `0.0074, 0.0125, 0.0073, 0.0163`.
RMSE slopes are approximately `−0.49` and `−0.52`, respectively; variance
ratios remain near one and coverage is approximately 0.93--0.97.  The extra
interaction block therefore costs some precision but does not create a
systematic bias when it is unnecessary.

### Strong interaction

With `beta2=(1.60,0.50)`, the true target is `phi*=−0.073010`.  The
differing-slopes biases are `0.0008, −0.0021, −0.0024, −0.0009` for
`n={500,1000,2000,4000}`, with coverage 0.927--0.950.  The original estimator
is increasingly displaced from the target: its biases are `−1.243, −1.626,
−1.945, −1.939`, and its coverage is only 0.573--0.723.  This magnifies the
same identification failure without changing its direction.

## Conclusion

The simulation supports the hypothesis. Under the differing-slopes assumptions
the augmented estimator is approximately unbiased and root-n, while the
original estimator is inconsistent for the full policy target when
`beta2 != 0`. The null control shows that the original estimator behaves well
when its assumptions actually hold. The variance check is also positive for
the augmented estimator in this conditional experiment, but generated-index,
estimated-endpoint, and moving-hard-boundary contributions still need to be
validated separately.

## Reproduction

```sh
cd /Users/zoudj/Documents/Codex/PerfRDD/code
PYTHONPATH=. python3 -m experiments.scripts.differing_slopes_simulation \
  --scenario baseline --n 500 1000 2000 4000 --reps 300 \
  --seed 20260912 \
  --out experiments/runs/differing_slopes_assumption_baseline_20260912.json

PYTHONPATH=. python3 -m experiments.scripts.differing_slopes_simulation \
  --scenario null_interaction --n 500 1000 2000 4000 --reps 300 \
  --seed 20260912 \
  --out experiments/runs/differing_slopes_assumption_null_20260912.json

PYTHONPATH=. python3 -m experiments.scripts.differing_slopes_simulation \
  --scenario strong_interaction --n 500 1000 2000 4000 --reps 300 \
  --seed 20260912 \
  --out experiments/runs/differing_slopes_assumption_strong_20260912.json

PYTHONPATH=. python3 -m experiments.scripts.differing_slopes_simulation \
  --scenario baseline --n 8000 16000 --reps 200 \
  --seed 20260913 \
  --out experiments/runs/differing_slopes_assumption_baseline_large_20260912.json
```
