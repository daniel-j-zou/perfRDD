# Differing slopes: identifying a level-dependent treatment effect

## The problem the taxi application exposed

The performative-RDD estimator targets the policy value
`U(φ) = E[(W − c)·1{Q ≥ φ}]`, with running variable `Q = T + η`, `T = γᵀX` the covariate
index, `η` the residual, and it models the individual effect as `α(η) = E[W | η]`. The clean
estimand collapse used in the draft,

```
U(φ) = E[(W − c) D(φ)] = E[(α(η) − c) Ḡ(φ − η)],
```

requires the maintained assumption `T ⊥ (W, η)` — the treatment effect must **not** depend on
the covariate index `T`, only on the residual `η`.

The taxi tip-menu treatment violates this. The percentage menu (20/25/30% of fare) versus the
fixed menu ($2/$3/$4) has an effect that is a function of the **fare level** `Q = T + η`, not of
`η` alone. Since dependence on `Q` at fixed `η` is dependence on `T`, `T ⊥ W` fails, and:

- The collapse above drops a covariance term `E_η[Cov_T(W, 1{T ≥ φ−η} | η)] ≠ 0`.
- A controlled simulation confirms that **even the exact `α(η) = E[W|η]`** mislocates the
  optimum — this is an estimand/identification issue, not estimation error.
- Consequently the `α`-only estimator is biased and unstable for the taxi optimum: across
  outcome (tip $, tip share, logit share), cleaning, ridge, and sample it wanders from a
  boundary (`$0`, "treat everyone") to `~$7`, and never reaches the region external evidence
  supports.

## The fix: differing treated/control slopes

Let the treated and control covariate slopes differ:

```
Y = b(η) + D·α(η) + Xᵀβ₁ + (D·X)ᵀβ₂ + ε,     effect  W = α(η) + Xᵀβ₂.
```

Now the effect can depend on `T = γᵀX` through `β₂`, so it can represent the fare-level menu
effect. Estimate `α, β₁, β₂, b` linear-in-X and maximize the differing-slopes utility
(`manuscript/this_week.tex`, updated 2026-09-23)

```
Û_J(φ) = mean_i Î_i [ (α̂(η̂_i) − c) Ḡ̂(φ − η̂_i) + β̂₂ᵀ Ĥ_X(φ − η̂_i) ],   c = 0,
```

where `Ḡ̂` and `Ĥ_X` integrate the Lebesgue-Gram spline projections `ĝ` (density of `T̂`) and
`p̂_X(t) = E(X | T=t) f_T(t)` (`experiments/methods/weighted_tails.py`). The earlier
own-fare objective `mean_i Î_i (α̂(η̂_i) + β̂₂ᵀX_i)·1{Q_i ≥ φ}` is still printed as a robustness
check. On the restricted VTS sample both select the same fare step: U_J gives φ* = $12.66,
the own-fare objective $12.51, and both mean "treat fares ≥ $12.90" (fares lie on a $0.40
lattice).

Implementation choices that matter (all chosen for honest stability):

- **Knots `≈ n_treated^{1/5}`** (~10–15), the MSE-optimal rate for a smooth `α`. The previous
  `n^{1/3}` (~39) over-parameterizes and produces a wiggly `α̂` that is noise around the same
  smooth curve; the wiggle does not move `φ*`.
- **Ridge on the `β₂` and spline blocks**, strength selected by **cross-validation** on the
  outcome (never tuned to a target). Penalizing `β₂` (it is high-dimensional) is what
  stabilizes the interaction; leaving it unpenalized reintroduces instability.

## Results (validated on the CMT-matched population)

CMT (the competitor vendor) runs percentage menus on both sides of $15, so it is an external
check only — never part of the estimator. It exists only in the paper-restricted population
(credit-card, January, daytime, no tolls/surcharge, standard meter, fares $5–25), so the
**apples-to-apples** comparison is on that population.

| model | restricted VTS ($5–25, CMT-matched) | full VTS (different population) |
|---|---:|---:|
| α only | $0.00 (boundary) | $0.00 (boundary) |
| **differing slopes** | **$12.6** | $9.2 |

External validation, all on the restricted (CMT-comparable) population:

- CMT vs VTS raw $1-bin crossover (fixed vs %): **~$12.7**
- Vendor-adjusted (−0.76pp): **~$11.2**
- Menu arithmetic ($3 = 0.25·Q): **$12**
- CMT-in-treated-group pooled fit: **$11–12**

So on the matched population the differing-slopes estimate (**$12.6**) agrees with every
external benchmark, while `α`-only degenerates to the boundary. The full-sample number ($9.2)
is a **different estimand on a different population** — only 35% of the full sample lies inside
the CMT-matched set (52% of it carries a surcharge, 45% is daytime, fares span $2.5–200) — so
CMT does not validate it; report it separately as the broad-population answer.

## Honest limits

- Not point-identified from a single sharp threshold: `β₂`'s fare direction rests on a
  linearity/extrapolation assumption, so there is a finite-sample band (~$9–15). The fix is
  robust in **sign and shape** (α̂ crosses zero, negative at low fares) and in being a clear
  improvement over `α`-only, not to the dollar.
- Interior-vs-boundary and the exact `φ*` depend on the welfare formulation (Ḡ-projected vs
  fare-direct empirical) and on the population; state which.

## Empirical rank diagnostic

The theorem needs the augmented `[D·Φ(η), Φ(η), X, D·X]` design to remain
identified. This is plausible in the restricted VTS sample, although no finite-sample check can
prove a uniform population condition. With ten equal-mass `η` bins on the hard-trim interval,
the treated share ranges from 7.2% to 40.5%, every menu-by-bin cell has at least 1,274 rides, and
the smallest eigenvalue of the conditional second-moment matrices for `(1,X)` is 0.071. The
column-normalized augmented design has condition number 9.33. After residualizing `D·X` on the
spline and common-slope nuisance columns, its four Gram eigenvalues are 0.043, 0.058, 0.094,
and 0.132 (condition number 3.04). Five- and twenty-bin checks give the same qualitative result.

One implementation detail must be corrected in a theorem-aligned estimator. The current ridge
diagnostic includes both a separate intercept and a baseline B-spline basis whose columns sum to
one. The corresponding unregularized design is exactly singular; ridge selects a numerical
solution, but the proof does not use ridge. The unregularized implementation must drop the
separate intercept (or equivalently remove the constant direction from the spline basis). This
is a parameterization repair, not a failure of the differing-slopes identification argument.

Reproduce these diagnostics with
`python -m experiments.scripts.taxi_differing_slopes_rank`; the machine-readable output is
written to `experiments/runs/taxi_differing_slopes_rank/summary.json`.

## What adding β₂ does to the theory

More than an extra term, less than a rewrite:

1. **Drop `T ⊥ W`.** The estimand no longer collapses to Ḡ alone: `U(φ) = E[(α(η)−c)Ḡ(φ−η)I₀]`
   **plus** `E[β₂ᵀX·1{T ≥ φ−η}I₀] = E[β₂ᵀH_X(φ−η)I₀]` (using `X ⊥ η`), with
   `H_X(s) = E[X·1{T > s}]` estimated through the weighted density `p_X`.
2. **Influence function / variance** gain `β̂₂`'s √n term and the empirical-average term for
   `E[X·1{T≥φ−η}I₀]`; the boundary terms acquire an additive `E[Xᵀβ₂|η=v]` piece.
3. **New rank condition**: the augmented design `[Φ(η), D·X]` must be nonsingular (α-spline and
   the interaction share the γ direction). This is what the finite-sample instability reflects.
4. **Reused unchanged**: the boundary quantile (Bahadur) expansion, moving-set linearization,
   discontinuous density representer — the technical core — because the `α(η)` part still
   collapses cleanly with Ḡ.

Reproduce the headline with `experiments/scripts/taxi_differing_slopes.py`.

## Treatment-direction gotcha (read before applying to other datasets)

Do **not** hardcode `D = 1{Q ≥ threshold}`. Take the direction from the dataset's
`treatment_rule` / `sample.D`, or from `_detect_direction(D, Q)`. In the registry:

- **above-cutoff** (`D = 1{Q ≥ thr}`): taxi, oulad, lending_default, nhanes
- **below-cutoff** (`D = 1{Q < thr}`): **gpa** — `sample.D` matches `Q≥thr` 0.00 of the time

Hardcoding `Q ≥ thr` silently flips below-cutoff designs and produces spurious optima (gpa
looked "interior $1.78" flipped; done correctly it is a boundary). Clean handling: mirror
below-cutoff designs (`Q → −Q`, `thr → −thr`), run the standard above-cutoff pipeline, and map
`φ* → −φ*`. Both the treatment indicator and the utility's `1{Q ≥ φ}` must use the right side.

## Cross-dataset β₂ screen

Reproduce with `python -m experiments.scripts.differing_slopes_screen --out
../outputs/differing_slopes_screen_20260924` (committed driver, 2026-09-24). Both models
maximize `Û_J` with spline `ĝ`, `p̂_X` (α-only: β₂ = 0); c = 0; standardized X; n_treated^{1/5}
knots and CV ridge as in the taxi script; below-cutoff designs are mirrored. Candidate cutoffs span
the 0.5–99.5% range of Q. Entries give the optimum and [share of the trim window it treats];
(bnd) marks a boundary solution: a range endpoint, or ≥99% / ≤1% of the window treated.

| dataset | dir. | cutoff | α-only `Û_J` | diff. slopes `Û_J` | diff. slopes own-score | previous own-score screen (α / DS) |
|---|---|---:|---:|---:|---:|---:|
| taxi (full VTS) | above | 15 | 3.61 [99%] | **9.30 [45%]** | 9.14 [44%] | 4.2 / 9.2 |
| taxi (restricted) | above | 15 | 5.30 [100%] (bnd) | **12.72 [24%]** (fares ≥ $12.90) | 12.51 [30%] | 0 (bnd) / 12.5 |
| oulad (dates fixed 2026-09-24) | above | 40 | 52.84 [0%] (bnd) | **42.14 [44%]** | 40.19 [59%] | 51.2 / 36.0 (buggy dates) |
| oulad, rich covariates | above | 40 | 74.84 [0%] (bnd) | 48.13 [33%] | 48.02 [27%] | — |
| lending_default | above | 30 | 38.60 [1%] (bnd) | 1.36 [100%] (bnd) | 38.60 [0%] (bnd) | 45.45 / 45.45 |
| gpa | below | 0 | 2.48 [100%] (bnd) | 2.48 [100%] (bnd) | 2.50 [100%] (bnd) | 2.6 (bnd) / 2.6 (bnd) |
| nhanes | above | 6.5 | 5.11 [100%] (bnd) | 5.10 [100%] (bnd) | 4.50 [100%] (bnd) | 3.4 (bnd) / 3.4 (bnd) |

What changes with `Û_J`:
- **Taxi:** unchanged. Full sample $9.30 (own-score $9.14, previous $9.2); restricted
  $12.72, the same fare step as before.
- **oulad:** the OULAD adapter ranked assignment dates as text until 2026-09-24, so
  earlier oulad rows used the wrong "first" TMA. With dates fixed, differing slopes gives
  42.1 (above the deployed 40; own-score 40.2); α-only treats none of the window. The
  estimate is regularization-dependent (unpenalized 37.3; ridge λ=3 reverts to α-only),
  and the window is 1.8% of the sample. Richer covariates (`load_rich`, R² 0.18) make
  it worse: 42–74 depending on the ridge, bootstrap 95% [44.5, 74.2].
- **lending_default:** no usable signal either way. The utility spans ~0.001 on a repayment
  scale. `Û_J` (treat all) and own-score (treat none) pick opposite boundaries because X ⊥ η
  fails: E[X'β̂₂ | window] = −0.016 against 0 overall (income z-scores reach 146).
- **gpa, nhanes:** boundary solutions for both models, as before; β₂ is inert. nhanes's
  `Û_J` maximum (5.1) lies on a flat plateau that treats the whole window.
The `Û_J` versus own-score gap is itself a diagnostic: the two target the same utility only
under X ⊥ η.
