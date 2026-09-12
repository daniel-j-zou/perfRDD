# Nonlinear treatment-effect simulation

## Question

Does the linear differing-slopes correction remain valid when the treatment
effect is nonlinear in the threshold score? The answer from this diagnostic is
no: adding `D*X` removes the original reduction's bias only for a linear
conditional treatment-effect relationship. Adding the relevant quadratic
term restores centering.

## Design

The known-target runner is
`experiments/scripts/nonlinear_slopes_simulation.py`. It generates

\[
X\sim N(0,I_2),\qquad \eta\sim N(0,1),\qquad
T=X_1,\qquad Q=T+\eta,\qquad D=1\{Q>0\},
\]

and

\[
Y=b_0+b_1\eta+X^\top\beta_1
 +D\{a_0+a_1\eta+\delta(T^2-1)\}+\varepsilon.
\]

The default (moderate) design has `delta=0.40`; the stress design has
`delta=0.80`; and the null design sets `delta=0`. The remaining parameters
are `a0=0.35`, `a1=0.90`, `beta1=(0.30,-0.20)`, `sigma_eps=0.50`, cost
`c=0.25`, and a hard 10% trim of `eta`. Because `T` is standard normal,

\[
E[(T^2-1)1\{T>z\}]=z\varphi(z),\qquad z=\phi-\eta.
\]

The population target is evaluated by Gauss--Legendre integration directly on
the hard-trim interval, avoiding a quadrature error at the trimming boundary.

We compare three outcome regressions:

1. `alpha_only`: `1 + eta + X + D + D*eta`;
2. `linear_slopes`: the first model plus `D*T` and `D*X2`;
3. `quadratic_slopes`: the second model plus `D*(T^2-1)`.

The threshold is the maximizer of the plug-in utility over `[-3,3]`. The
variance is a conditional delta-method sandwich calculation treating `eta`,
the trim interval, and the normal law of `T` as known. Thus this experiment
isolates nonlinear outcome-model misspecification rather than generated-index
or moving-boundary effects.

Each cell uses 300 replications at `n={500,1000,2000,4000}`. Results below
compare estimates with the full nonlinear population target.

## Results: moderate quadratic effect (`delta=0.40`)

The full target is `phi*=0.424625`; the restricted no-quadratic formula would
place it at `-0.319238`.

| n | alpha-only bias | linear-slopes bias | quadratic-slopes bias | quadratic variance ratio | quadratic coverage |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.676 | −0.056 | −0.025 | 0.324 | 0.937 |
| 1,000 | 0.613 | −0.059 | 0.001 | 1.012 | 0.963 |
| 2,000 | 0.598 | −0.064 | −0.003 | 1.077 | 0.957 |
| 4,000 | 0.597 | −0.065 | −0.004 | 0.926 | 0.940 |

The linear-slopes bias is stable rather than shrinking. Its variance estimates
are close to the Monte Carlo variance, but coverage declines because intervals
are centered about the wrong limit. The correctly augmented quadratic model
has negligible bias from `n=1000` onward, a tail RMSE log--log slope of about
`−0.48`, and approximately nominal variance coverage. The `n=500`
quadratic RMSE/variance cell is noisy because a few small-sample threshold
optimizations are poorly conditioned; this disappears at larger samples.

The alpha-only objective is pushed toward the upper policy bound in this
design, which is itself evidence of severe misspecification rather than a
meaningful policy conclusion.

## Results: null and stronger curvature controls

When `delta=0`, all three models target `phi*=-0.319238`. Biases at
`n={500,1000,2000,4000}` are respectively:

- alpha-only: `−0.006, −0.002, −0.009, −0.004`;
- linear-slopes: `−0.015, −0.010, −0.013, −0.004`;
- quadratic-slopes: `−0.036, −0.028, −0.018, −0.007`.

Coverage is approximately 0.95--1.00, with the richer models paying a modest
finite-sample precision cost. This is the control case in which no nonlinear
term is needed.

With stronger curvature (`delta=0.80`), the full target is `phi*=0.767420`.
The linear-slopes bias is `−0.201, −0.204, −0.208, −0.208`, while the
quadratic-slopes bias is `0.001, 0.001, −0.000, −0.002`. Quadratic variance
ratios range from 0.90 to 1.00 and coverage is 0.923--0.947. The alpha-only
estimate again moves to the upper policy bound.

## Conclusion

The simulation distinguishes two claims:

1. The original alpha-only reduction fails whenever treatment effects vary with
   the threshold score, whether that variation is linear or nonlinear.
2. The current `D*X` differing-slopes estimator is not a general solution to
   nonlinear heterogeneity. It is valid when the conditional treatment effect
   is linear in the score. For the quadratic DGP, adding `D*(T^2-1)` restores
   root-`n` behavior and correct conditional variance coverage.

For a general unknown nonlinear `W`, the natural extension is a sieve or spline
basis in treated interactions, with the policy utility integrating each basis
function over the threshold-selected region. That extension would require
separate theory for basis growth, regularization, and generated-index terms.

## Reproduction

```sh
cd /Users/zoudj/Documents/Codex/PerfRDD/code
PYTHONPATH=. python3 -m experiments.scripts.nonlinear_slopes_simulation \
  --scenario quadratic --n 500 1000 2000 4000 --reps 300 \
  --seed 20260914 \
  --out experiments/runs/nonlinear_quadratic_20260914.json

PYTHONPATH=. python3 -m experiments.scripts.nonlinear_slopes_simulation \
  --scenario null_quadratic --n 500 1000 2000 4000 --reps 300 \
  --seed 20260915 \
  --out experiments/runs/nonlinear_null_20260914.json

PYTHONPATH=. python3 -m experiments.scripts.nonlinear_slopes_simulation \
  --scenario strong_quadratic --n 500 1000 2000 4000 --reps 300 \
  --seed 20260916 \
  --out experiments/runs/nonlinear_strong_20260914.json
```
