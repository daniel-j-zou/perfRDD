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
effect. Estimate `α, β₁, β₂, b` linear-in-X and maximize the **empirical** utility
`Û(φ) = mean_{i: η_i∈[l₀,u₀]} (α̂(η_i) + β̂₂ᵀX_i)·1{Q_i ≥ φ}`.

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

## What adding β₂ does to the theory

More than an extra term, less than a rewrite:

1. **Drop `T ⊥ W`.** The estimand no longer collapses; `U(φ) = E[(α(η)−c)Ḡ(φ−η)I₀]` (unchanged)
   **plus** `E[β₂ᵀX·1{T ≥ φ−η}I₀]`, which needs the joint `(X, T, η)` and does not reduce to Ḡ.
2. **Influence function / variance** gain `β̂₂`'s √n term and the empirical-average term for
   `E[X·1{T≥φ−η}I₀]`; the boundary terms acquire an additive `E[Xᵀβ₂|η=v]` piece.
3. **New rank condition**: the augmented design `[Φ(η), D·X]` must be nonsingular (α-spline and
   the interaction share the γ direction). This is what the finite-sample instability reflects.
4. **Reused unchanged**: the boundary quantile (Bahadur) expansion, moving-set linearization,
   discontinuous density representer — the technical core — because the `α(η)` part still
   collapses cleanly with Ḡ.

Reproduce the headline with `experiments/scripts/taxi_differing_slopes.py`.
