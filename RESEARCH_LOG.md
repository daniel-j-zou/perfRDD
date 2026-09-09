# PerfRDD research log — shared between Claude and Codex

This tracked file is the single source of truth for **verified findings and decisions**.
The operating rules are in `COLLABORATION.md`; forward-looking work is tracked in
`../manuscript/TODO.md`.

Add new entries immediately below the divider, newest first. Date and sign every entry.
Never edit or delete another agent's entry; add a follow-up when a conclusion changes.

---

## 2026-09-08 — Restricted-VTS augmented-rank diagnostic (Codex)

Added `experiments/scripts/taxi_differing_slopes_rank.py` and ran it on all
541,021 cleaned observations in the paper-restricted VTS sample.  On the hard-
trim interval, ten equal-mass eta bins have treated shares from 7.2 to 40.5
percent and at least 1,274 observations in every menu-by-bin cell.  The minimum
eigenvalue among the binwise conditional second-moment matrices for `(1,X)` is
0.071.  The column-normalized theorem design
`[D*Phi(eta), Phi(eta), X, D*X]` has condition number 9.33, and the four
eigenvalues of the `D*X` Gram matrix after residualizing on the nuisance columns
are 0.043, 0.058, 0.094, and 0.132.  Five- and twenty-bin checks give the same
qualitative conclusion.  These diagnostics support, but cannot prove, the
maintained uniform population rank condition.

The check also exposed a concrete theorem/application parameterization mismatch.
The current ridge taxi diagnostic includes a separate intercept together with a
baseline B-spline basis whose columns sum to one.  Its unregularized design is
therefore exactly singular (condition number about 3.6e15).  Ridge makes the
diagnostic numerically well defined, but the theorem-facing unregularized
estimator must drop the separate intercept or remove the constant spline
direction.  This is a straightforward implementation repair rather than a
failure of the differing-slopes theory.

— Codex

---

## 2026-09-08 — Simulation evidence written into the prelim (Codex)

Added a concise main-text subsection in `manuscript/prelim/prelim.tex` that states
the differing-slopes simulation hypotheses, testing strategy, headline results,
and alignment with the maintained theory.  The main text reports the positive
correct-specification result (RMSE slopes about -0.57, alpha-only bias about
0.43--0.50, and larger-run variance ratios 0.98--1.11 with coverage 0.94--0.96),
the nonlinear/weak-curvature negative controls, and the boundary and clustering
warnings.  A new Appendix 5 records the DGP, estimator variants, replication
matrix, objective search, variance scope, and reproducibility paths.  The text
explicitly does not claim bootstrap validity or a completed generated-index
analytic variance theorem.  The completed full-pipeline battery remains the
source of all reported values (`differing_slopes_full_pipeline.py` and the
2026-09-08 entries below).

— Codex

---

## 2026-09-08 — Skeptical feasibility audit of differing slopes (Codex)

The differing-slopes strategy is theoretically feasible without changing the
hard-trim boundary architecture.  Under fixed-dimensional
$W=a(\eta)+X^{\circ\top}\beta_2+R_W$, retained $X\perp\eta$, conditional
mean-zero residuals, and an augmented rank condition, the outcome block remains
a partially linear sieve regression.  The policy derivative adds the
one-dimensional vector weighted density
$\rho_X(t)=E(X^\circ\mid T=t)f_T(t)$; it does not introduce a higher-dimensional
nonparametric nuisance.  The same fixed-buffer, zero-outer-trace weak
integration-by-parts repair applies componentwise.  A primitive sufficient rank
condition is uniform overlap plus uniformly nonsingular conditional second
moments of $(1,X^\circ)$ within each $(\eta,D)$ cell.  This explicitly identifies
the linear $D X^\circ$ effect through within-menu covariate variation; it remains
model-based extrapolation, not nonparametric identification from the single VTS
cutoff.

The audit corrected two proof-level issues in `manuscript/prelim/prelim.tex`.
First, the observed differing-slopes outcome residual is
$R_Y^{DS}=\epsilon+D R_W$; the earlier display wrote only $\epsilon$ while the
influence function used $R_Y^{DS}$.  Second, the vector-density linear expansion
previously omitted the first-order effect of its fold-specific generated index;
it now contains
$A_{\rho,\eta,L}^\top\sqrt{n_\rho}(\hat\gamma^\rho-\gamma)$ and derives the
identity from the deterministic Lebesgue Gram matrix.  The manuscript now states
the nonempty rate window $\sqrt nL_n^{-3}\to0$ and
$L_n^{5/2}/\sqrt n\to0$, adds the multiplier fourth-moment condition, and no
longer calls the extension closed.  Remaining formal work is the
multiplier-spline triangular-array/Lindeberg bound, empirical augmented-Gram
convergence, exact sign and fold-fraction bookkeeping, the inherited local
Bahadur lemma, and theorem/application estimator alignment.  An analytic
foldwise variance estimator exists in principle, but the author prefers full
re-estimation bootstrap inference in practice.

The taxi motivation was checked against `DIFFERING_SLOPES.md`,
`DIRECT_COMPETITOR_CHECK.md`, `LOW_FARE_PROXY.md`, and `COMPETITOR_CHECK.md`.
The original scalar reduction omits
$E[I_0\operatorname{Cov}\{W,1(T>\phi-\eta)\mid\eta\}]$.  This is exactly the
taxi problem: at fixed $\eta$, larger $T$ means a larger fare and a larger dollar
percentage suggestion, while the candidate policy also selects on $T$.  The VTS
scalar fit is internally calibrated but overpredicts low-fare CMT percentage-menu
tips by USD 0.542 (USD 0.443 after high-fare calibration); the auxiliary
fare-cell proxy is negative on 90.5 percent of low-fare VTS mass and crosses near
$12.8.  These are external sign/transport diagnostics only, because vendor
selection prevents a causal CMT-minus-VTS interpretation.  The manuscript now
states this logic in the application and Appendix 4 and flags the legacy
scalar-$\alpha$ figures/results for replacement by the theorem-aligned
$(\bar G,H_X)$ differing-slopes estimator.

— Codex

---

## 2026-09-08 — Maintained augmented rank and completed differing-slopes proof blocks (Codex)

The author approved treating the uniform augmented-rank condition as a maintained
primitive assumption for the differing-slopes extension.  Appendix 4 now uses it
to bound the inverse augmented Gram matrix and writes the corresponding spline
projection/Riesz approximation argument; the old A4 condition is not invoked for
the new $D X^\circ$ columns.  The vector weighted-density block now has an
explicit cubic-spline approximation/loading lemma, including the matrix-to-vector
dimension check, fold-specific influence representation, and weak integration by
parts limit under the already approved componentwise $C^3$ and $W^{1,2}$
conditions.  The main differing-slopes score is displayed with evaluation,
outcome, scalar-density, vector-density, and four deliberately decoupled
generated-index contributions.  Curvature consistency adds only the
$\beta_2^\top\rho_X'$ term.  Remaining open work is inherited from the original
hard-trim theorem: a primitive local Bahadur proof and a feasible variance
estimator for the deliberately decoupled split; no ordinary cross-fitting or
full-sample-reuse theorem is claimed.

— Codex

---

## 2026-09-08 — Clarified differing-slopes rank and smoothness conditions (Codex)

The manuscript now states the differing-slopes replacement for the old A7:
retain independence of $X$ and $\eta$ and the conditional mean-zero residual
conditions, while dropping $W\perp X$ (which is incompatible with nonzero
$\beta_2$ under nondegenerate covariates).  Appendix 4 defines the uniform
augmented-rank condition as a lower eigenvalue bound for
$(D N_K(\eta),N_K(\eta),X^\circ,D X^\circ)$ with a constant independent of
$K$, and explains why fixed-$K$ nonsingularity is insufficient.  At the
author's request, the vector weighted-density block now has explicit
componentwise $C^3$ smoothness for $\rho_X$ and entrywise $W^{1,2}$ regularity
for $E[XX^{\circ\top}\mid T=t]f_T(t)$.  The rank condition is defined but not
treated as author-approved; no proof claim is changed.

— Codex

## 2026-09-08 — Differing-slopes proof map added to the prelim (Codex)

Added Appendix 4 to `manuscript/prelim/prelim.tex` with the current conditional
proof status for the unrestricted finite-dimensional $D X$ extension.  The
appendix derives the altered supported target and its derivative, states the
retained versus new assumptions, defines the augmented outcome design and
Riesz loading, introduces the vector weighted-density nuisance and its
hard-boundary integration-by-parts term, and maps the unchanged moving-set and
generated-index Bahadur blocks into the conditional decoupled CLT.  The
augmented Gram/Riesz lemma, vector-density weak-pairing lemma, explicit main
influence function, and curvature verification remain marked as TODOs; no
closed differing-slopes theorem is claimed.  The task board records this
conditional appendix as complete, with those proof obligations still open.

— Codex

## 2026-09-08 — Codex review of the differing-slopes proposal

Reviewed Claude's commits `932164d` and `6ff3337`, the application note
`experiments/datasets/taxi/DIFFERING_SLOPES.md`, and the reproducible script
`experiments/scripts/taxi_differing_slopes.py`. The diagnosis is substantively
plausible for taxi: the percentage-menu effect is naturally fare-level dependent,
so the pooled $\alpha(\eta)$-only model is not sufficient. Adding
$D X^\top\beta_2$ gives conditional effect $\alpha(\eta)+X^\top\beta_2$,
but this is a linear/extrapolative approximation, not a nonparametric solution.

The reported $\$12.5$ restricted-sample and $\$9.2$ full-sample optima are
application diagnostics, not outputs of the current theorem: the script uses a
direct empirical indicator $1\{Q\ge\phi\}$, $c=0$, same-sample outcome-CV
ridge, and no $\bar G$ density plug-in or inference. The restricted CMT comparisons
are descriptive external benchmarks and do not validate the full-sample estimand.
The proposed theory changes are directionally right (drop $T\perp W$, add a rank
condition and $\beta_2$ influence terms), but the claim that the boundary CLT is
otherwise unchanged still requires a new score/variance derivation because the
utility no longer collapses to a single $\bar G$ term.

The cross-dataset screen is documented but its direction-aware driver is not a
committed reproducible script. The taxi script itself hardcodes the above-cutoff
rule, so it should not be generalized without the direction handling Claude noted.
An independent rerun was blocked by the missing `pyarrow`/`fastparquet` parquet
engine in the current Python environment, despite the local taxi parquet files.

— Codex

## 2026-09-08 — Treatment-direction gotcha + cross-dataset β₂ screen (Claude)
Applying the differing-slopes methodology across the registry surfaced a direction bug worth
flagging for all future runs: do NOT hardcode `D=1{Q≥thr}`. Take the treatment direction from
`sample.D`/`treatment_rule` or `_detect_direction(D,Q)`. **gpa is a BELOW-cutoff design**
(`sample.D` matches `Q≥thr` 0.00 of the time); taxi/oulad/lending_default/nhanes are above.
Hardcoding `≥` flips below-cutoff designs and gives spurious optima (gpa looked "interior
$1.78"; corrected it is a boundary). Handle 'below' by mirroring `Q→−Q, thr→−thr`, run the
standard above pipeline, map `φ*→−φ*`; both the treatment indicator and the utility indicator
`1{Q≥φ}` must use the right side. Cross-dataset β₂ screen (n^{1/5} knots, CV ridge, correct
direction): β₂ moves φ* only where the effect is level-dependent — taxi (α-only→boundary/low,
diff-slopes→$9–12) and oulad ($51→$36, crossing cutoff 40) — and is inert on φ* for gpa
(below, boundary), lending_default ($45.45), nhanes (boundary). So β₂ is a targeted
correction/diagnostic, not a free knob. Table + details appended to
`experiments/datasets/taxi/DIFFERING_SLOPES.md`.

## 2026-09-08 — Differing-slopes fix for the level-dependent taxi effect (Claude)
The performative-RDD estimand collapse `U(φ)=E[(α(η)−c)Ḡ(φ−η)]` requires `T⊥(W,η)` — the
effect must depend only on the residual η, not the covariate index T=γ'X. The taxi
percentage-menu effect depends on the fare LEVEL Q=T+η, violating this, so α(η)-only is
biased/degenerate for the policy optimum (φ*≈$0 boundary on the RD sample, ~$5–7 unstable on
the full sample; a controlled simulation shows even the exact α(η) mislocates φ*). Fix: let
treated/control covariate slopes differ, `Y=b(η)+Dα(η)+X'β₁+(DX)'β₂+ε`, effect `α(η)+β₂'X`,
so the effect can depend on T. With knots ≈ n_treated^{1/5} (~10; n^{1/3} over-fits → wiggle)
and CV-selected ridge on the β₂+spline blocks, the empirical-utility max on the CMT-matched
restricted VTS sample is **φ*≈$12.5**, vs α-only **$0** (boundary). This matches external
validation *on the matched population*: CMT raw $1-bin crossover $12.7, vendor-adjusted $11.2,
menu arithmetic $12, CMT-in-treated pooled $11–12. The full VTS sample gives $9.2 but is a
DIFFERENT population (only 35% overlaps the CMT-matched set: 52% surcharge, 45% daytime, fares
$2.5–200) so CMT cannot validate it. CMT is validation only; the estimator uses VTS.
Not point-identified (β₂'s fare direction rests on linearity → ~$9–15 finite-sample band); the
fix is robust in sign/shape and clearly beats α-only. Theory impact: drop `T⊥W`, redefine the
estimand on the joint (X,T,η) (β₂'X term does not collapse to Ḡ), add β̂₂'s √n influence + a
rank condition on [Φ(η),D·X]; the boundary-CLT machinery is reused unchanged. Write-up:
`experiments/datasets/taxi/DIFFERING_SLOPES.md`; reproduce with
`experiments/scripts/taxi_differing_slopes.py`.

## 2026-09-08 — Overnight simulation update: spline density and non-Gaussian rates (Codex)

The completed overnight outputs are in the ignored local artifacts
`experiments/runs/overnight_auxiliary_20260908.json` and
`experiments/runs/overnight_spline_density_20260908/summary.json`. The 100-replication
non-Gaussian run gives RMSE log--log slopes of $-0.482$ (full sample) and
$-0.501$ (three-fold) for standardized $t_5$, and $-0.489$ and $-0.481$ for
the standardized two-component mixture. There are no grid-boundary selections;
$n\,\mathrm{Var}(\hat\phi)$ is about 28--31 for $t_5$ and 133--222 for the
mixture, with the mixture constant visibly noisier.

The 500-replication spline-density comparison has target
$\phi^\star=0.731292$. Full-sample reuse without ridge has RMSE slope $-0.480$
and $n\,\mathrm{Var}$ about 42--46; five-fold cross-fitting has slope $-0.523$
and $n\,\mathrm{Var}$ about 42--52. Boundary selections are essentially absent
for these variants. The disjoint honest split has RMSE 0.620 and a 21\% boundary
rate at $n=1{,}000$, and ridge scale 0.1 drives 94\% of those samples to the
policy boundary. The finite-sample recommendation is therefore full-sample reuse
or moderate cross-fitting with no/very mild ridge, while aggressive regularization
and small honest splits are failure modes.

The full re-estimation bootstrap was interrupted after 76 of 480 outer tasks,
covering only $t_5,n=1{,}200$; all completed rows had zero bootstrap failures,
but the summary was not written and no coverage claim is made. The corresponding
verified findings and this limitation were added to the prelim (manuscript commits
`0f5f255` and `db625d6`).

— Codex

## 2026-09-08 — Structure the publication-oriented empirical roadmap (Codex)

Added a new ``Robustness and inference program'' subsection to
`manuscript/prelim/prelim.tex` (manuscript source commit `195cd79`, changelog
commit `3236dbe`). The section treats the existing Gaussian experiment as a narrow
baseline and lays out falsifiable tests for full re-estimation bootstrap coverage,
non-Gaussian root-​$n$ behavior, support/trimming sensitivity, density projection,
structural misspecification, policy-optimization stability, and the deferred
discrete extension. It records only the already verified 50-replication pilot
numbers (the $t_5$ and skewed-mixture targets and RMSE slopes) and labels the
larger overnight batch as pending. The stated bootstrap checks are computational
success criteria, not a bootstrap validity theorem; density constraints,
first-stage violations, curvature stress, and clustered dependence remain future
experiments.

The updated prelim compiled with `pdflatex → bibtex → pdflatex ×2` (28 pages,
no fatal or undefined-reference markers), and pages 10--13 containing the new
roadmap table were rendered and visually inspected.

— Codex

## 2026-09-08 — Identified the Kevin Wibisono score-explained heterogeneity preprint (Codex)

The relevant paper is Kevin Christian Wibisono, Debarghya Mukherjee, Moulinath
Banerjee, and Ya'acov Ritov, *Estimation and Inference for the Average Treatment
Effect in a Score-Explained Heterogeneous Treatment Effect Model*, arXiv:2504.17126
(submitted 23 April 2025, 44 pages). Wibisono's publication page lists the same
line of work under the working title *Estimation of Non-Randomized Heterogeneous
Treatment Effects in the Presence of Unobserved Confounding Variables*. The paper
extends Mukherjee et al.'s fixed-effect score-explained model to heterogeneous
treatment effects and estimates the ATT using first-order differencing and residual
matching on estimated latent residuals; it also discusses CATE/ITE estimates,
sample splitting, asymptotic normality, bootstrap variance estimation, simulations,
and the same Turkey/GPA applications. It is a close methodological neighbor but
still targets treatment-effect averages at a fixed cutoff, not PerfRDD's utility-
maximizing threshold.

— Codex

## 2026-09-08 — Lessons from the Mukherjee--Banerjee--Ritov score-explained treatment-effect papers (Codex)

Read the local main paper and 77-page supplement, `Mukherjee et al. - Estimation of a
score-explained non-randomized treatment effect in fixed and high dimensions`. The
paper's core model is the same latent-index partial-linear decomposition used here:
\(Q=Z^\top\gamma+\eta\) and an outcome with a discontinuous treatment term plus a
smooth function of \(\eta\). It uses deliberate three-way sample splitting, cubic
B-splines on a fixed compact interval, and a proof organized around a projected
linear representation. It explicitly treats the fixed interval as an efficiency
loss, defers a growing-support analysis to future work, and uses bootstrap intervals
because the analytic variance is difficult to estimate. These are useful precedents
for our split construction, fixed buffered spline regions, spline appendix, and
computational-bootstrap decision.

The distinction that must remain explicit is substantive: their target is a constant
treatment effect \(\alpha_0\), with a fixed treatment cutoff and a fixed nuisance
truncation \(|\hat\eta|\leq\tau\). Their truncation therefore changes the information
set and variance but not the target parameter. In PerfRDD, \(\ind\{l_0\leq\eta\leq
u_0\}\) is inside the supported utility functional, so estimated endpoints change
the target population and generate first-order moving-set, quantile-boundary, and
density-boundary terms. Their fixed-\(\tau\) argument cannot replace Blocks C--E of
the hard-supported threshold proof, and their spline growth range should not be
copied because our density and threshold derivatives impose different rate
restrictions.

The paper also leaves bootstrap consistency as an open theoretical problem, which
supports labeling our full re-estimation bootstrap as computational inference rather
than claiming a bootstrap validity theorem. The local copy is marked “Submitted to
Bernoulli,” so its layout is research-group precedent, not evidence of current
Biometrika formatting requirements.

— Codex

## 2026-09-08 — Reframe prelim TODOs around utility-maximizing threshold (Codex)

Revised `manuscript/prelim/prelim.tex` so the planning TODOs and adjacent framing
center on estimating the threshold that maximizes the supported utility function.
Removed literal performativity/feedback language, renamed the setup subsection to
threshold assignment and outcome model, and deferred the discrete-policy extension
because the main theory is continuous M-estimation. The deliberately decoupled
spline-\(\bar G\) theory and maintained identification assumptions are unchanged.

— Codex

## 2026-09-08 — Author decisions fixing the theory scope and inference (Codex)

The author fixes the theorem-level construction as follows: the asymptotic theory is
for the deliberately decoupled split only; \(\bar G\) is estimated by the spline
density block; the splines are unregularized; and the policy threshold is treated as
a continuous M-estimator rather than a fixed numerical grid. The maintained
identification assumptions include the independence structure used to obtain
\(P\{D(\phi)=1\mid\eta\}=\bar G(\phi-\eta)\) and the counterfactual stability of
\((X,\eta,W,\epsilon)\). The paper will use a full re-estimation bootstrap as
computational inference; it will not claim a feasible analytic variance estimator or
a bootstrap validity theorem at this stage. Application shortcuts (ridge, empirical
CDFs, ordinary cross-fitting, and finite grids) are diagnostics/simplifications and
are not the objects covered by the main theorem.

— Codex

## 2026-09-07 — Longer interior non-Gaussian Monte Carlo (Codex)

Ran the long-run mode for the two interior designs: 50 replications at
$n\in\{1{,}200,2{,}400,4{,}800,9{,}600\}$ for both full-sample reuse and three-fold
cross-fitting. The lognormal boundary stress design was excluded from this run. The
known-target output is in the ignored local artifact
`experiments/runs/hard_trim_robustness_long/summary.json`.

The $t_5$ RMSEs for full-sample reuse are $0.153,0.111,0.081,0.057$ across the four
sample sizes, with an estimated log--log slope $-0.473$ (cross-fit: $-0.481$). There
were no grid-boundary selections. The skewed-mixture RMSEs are $0.366,0.221,0.216,0.113$,
with slopes $-0.513$ (full sample) and $-0.479$ (cross-fit), again with no boundary
selections. The mixture has a much larger and noisier asymptotic variance constant than
$t_5$ (empirical $n\,\mathrm{Var}$ roughly 100--230 versus roughly 27--33), so it needs
more replications before reporting a precise variance number. These slopes are close to
the expected $-1/2$ rate, but this run still provides empirical variance diagnostics,
not feasible confidence intervals.

The long-run code now accepts `--laws` and `--skip-auxiliary`, records $n$-scaled
variance, and reports the log--log RMSE slope. The run completed without numerical
failures; the point-estimation API continues to mark inference as unavailable.

— Codex

## 2026-09-07 — Moderate non-Gaussian Monte Carlo and target correction (Codex)

Extended the smoke harness with law-specific population trim bounds and known targets,
and added `experiments/scripts/hard_trim_robustness_monte_carlo.py`. The moderate run
used $n\in\{600,1{,}200,2{,}400\}$ and 20 replications for each of a standardized
$t_5$, a skewed two-component Gaussian mixture, and a centered/scaled lognormal stress
design. It also repeated the support, omitted-interaction, and bootstrap diagnostics.

The target correction matters: hard trimming uses the $T$-quantile endpoints, so the
population target is not the Gaussian-$\eta$-quantile target. The corrected targets are
$\phi^\star=0.588$ for $t_5$, $0.793$ for the skewed mixture, and the upper policy
boundary $\phi^\star=3$ for the lognormal stress design. The $t_5$ RMSE declines from
0.203 at $n=600$ to 0.131 at $n=2{,}400$, with no grid-boundary selections. The mixture
has a noisier but interior target (RMSE 0.958, 0.306, and 0.320 across those sample
sizes; boundary rate 5% at $n=600$ and zero thereafter). The lognormal design is a
deliberate boundary stress test rather than an interior-optimum convergence design;
its boundary rates are 50%, 70%, and 70% for the full-sample estimator at the three
sample sizes.

The support perturbation changed the point estimate by about 0.008--0.010 on average in
12 $t_5$ replications at $n=1{,}200$. The omitted-interaction outcome remains finite,
but its reference to the correctly specified target is descriptive only. A 10-rep
iid bootstrap was finite (mean $1.17$, SD $0.42$); inference is still unavailable from
the point-estimation API. These results justify a longer run for the $t_5$ and mixture
designs after the current tests are rerun; the lognormal case should remain a separate
boundary/sieve stress table.

— Codex

## 2026-09-07 — Short robustness smoke tests for the next simulation phase (Codex)

Added `experiments/scripts/hard_trim_robustness_smoke.py` and
`experiments/tests/test_hard_trim_robustness_smoke.py`. The smoke suite uses $n=600$
and three replications for two continuous non-Gaussian running variables (standardized
$t_5$ and centered/scaled lognormal), two fixed nuisance supports, an omitted
treatment--covariate interaction, and a five-replication iid bootstrap diagnostic. The
four unit tests pass, and the full experiment test suite passes (55 tests in 94.6s;
optional HMDA/MIMIC/NLSY data tests remain skipped because their local data are absent).

The short run is numerically stable but identifies two issues to resolve before a long
Monte Carlo grid: (i) with the skewed running variable, the current policy grid
$[-1.5,1.5]$ places the estimated optimum at its upper boundary in two of three seeds;
the long study needs wider, DGP-appropriate policy bounds and a known-truth target; and
(ii) the unconstrained spline projection gives survival estimates as high as 1.076 in
this small sample. The estimator intentionally does not enforce nonnegativity or unit
mass, so this is a density-sieve stress diagnostic, not a code failure, but it must be
tracked in the longer support/basis sensitivity study. The support perturbation
$[-2.5,2.5]$ versus $[-3.5,3.5]$ changed the point estimate by about $0.025$ in this
sample. The bootstrap was finite (mean $0.88$, SD $0.46$) but remains explicitly
diagnostic because the point-estimation API reports `inference_available=False`.

No long simulation is launched yet; the smoke results justify first widening the policy
grid and adding DGP-known targets for the non-Gaussian designs.

— Codex

## 2026-09-07 — Introduction and appendix planning TODOs expanded (Codex)

Expanded the TODOs in `manuscript/prelim/prelim.tex` without changing the opening
advisor-only ``To discuss.'' section. The Introduction TODOs now ask for the
performative-feedback motivation, four-literature positioning, and a bounded list of
contributions that distinguishes proved claims from conditional proof obligations and
application diagnostics. Added an explicit spline-construction/approximation-theory
appendix TODO and labeled ridge stabilization as numerical only. The discrete-policy
placeholder now explains that it concerns a finite grid of policy tiers. Added an
application identification/reproducibility TODO and an inference TODO to discuss a
full estimator bootstrap; the analytic variance is not being claimed as currently
feasible. The proof bookkeeping remains intentionally informal, and the fixed-versus-
expanding support choice remains with the author.

— Codex

## 2026-09-07 — Taxi empirical work frozen; simulations are the active workstream (Codex)

Per the author's instruction, Codex is closing the taxi empirical work at its current
reproducible state and handing further taxi/application changes to Claude. The durable
taxi record is the VTS/CMT validation already logged above and in
`experiments/datasets/taxi/{DIRECT_COMPETITOR_CHECK,LOW_FARE_PROXY,COMPETITOR_CHECK}.md`:
the VTS model fits its own low/high regimes, while the CMT comparison supports a
negative low-fare direction but does not identify a transportable VTS causal effect.
The untracked `experiments/scripts/taxi_perfrdd_share_tip.py` is preserved as Claude's
working file and is intentionally not staged or modified here.

The simulation evidence currently available is the favorable Gaussian hard-trim design
and its spline-density replication. With 200 replications at each of $n=20{,}000$,
$40{,}000$, and $80{,}000$, the exact hard-trim estimator is centered at the known
target and its empirical variance tracks the DGP benchmark: pooled $n$-MSE divided by
the population variance is 1.003 for the decoupled honest split, 0.964 for five-fold
cross-fitting, and 0.967 for full-sample reuse. DGP-known 95% coverage is 0.945--0.975
in the Gaussian run. Replacing the Gaussian density with the manuscript spline
projection gives ratios 0.991 (honest), 0.963 (five-fold), and 0.965 (full reuse), with
coverage 0.945--0.975 after finite-sieve centering. These are favorable-design checks,
not yet a misspecification or feasible-standard-error study.

Codex therefore returns to simulations. The next run should keep the same known-truth
convergence/variance/coverage diagnostics while replacing the Gaussian running variable
with skewed/heavy-tailed continuous laws and stressing the density sieve and support
choice. A separate inference task remains: a feasible variance/CI estimator for the
decoupled split; ordinary cross-fitting and full-sample reuse are not covered by the
current theorem.

— Codex

## 2026-09-01 — Taxi robustness: logit tip-share outcome preserves the interior optimum (Claude)
Re-ran the taxi application with the outcome changed from tip dollars to the logit tip
share, $Y=\operatorname{logit}(\text{Tip}/\text{Fare})$, via new script
`experiments/scripts/taxi_perfrdd_logit_tip.py` (same sample, \$15 assignment, covariates,
fixed nuisance support, and estimator as `taxi_perfrdd_application.py`). The share is
degenerate for $3.6\%$ zero-tip rides ($p=0$) and $0.28\%$ with $p\ge1$ (entry outliers up
to $40\times$); $p$ is clipped to $[0.01,0.99]$ before the logit. Single run (no bootstrap),
full $n=1{,}528{,}292$. Results: first-stage $R^2=0.66$; $\hat\alpha(\eta)$ still changes
sign on the overlap window $[-3.65,8.62]$ (range $\approx[-0.62,0.38]$, $\approx50\%$ of
in-window mass negative); pooled-PLM screen interior $\hat\phi\approx\$9.24$ (beats the
boundary by $+20.6\%$ of welfare scale, confirmed at full $n$); inference-grade hard-trim
$\hat\phi_\epsilon\approx\$5.96$ (interior, design cond\# $5.6\times10^3$). Both interior and
below \$15, as in the dollar-outcome analysis, so the sign-change and sub-\$15 interior
optimum are not an artifact of measuring tips in dollars. Figures copied to
`../manuscript/figures/taxi_logit_{alpha,b,utility}.png`; written up as a robustness
subsection in `../manuscript/prelim/prelim.tex` (compiles, 23 pp). Only new caveat: the
boundary clip compresses the zero-tip mass to one low value rather than modeling it.

## 2026-09-01 — Taxi empirical section: interior-optimum version is the version of record (Claude)
Author (owner) decision: the interior-optimum taxi application supersedes the restricted
boundary-optimum framing that had been integrated into `../manuscript/prelim/prelim.tex`.
The empirical section now reports the full-VTS-sample result — sign-changing
$\hat\alpha(\eta)$, welfare with a strictly interior maximizer, inference-grade hard-trim
$\hat\phi_\epsilon\approx\$5.3$ (design cond\# $5.6\times10^3$, robust $\$5.0$–$7.6$), and
the iid trip bootstrap ($B=120$, $m=120$k): 100\% interior, mean $\$3.61$, median $\$3.51$,
95\% PI $[\$1.67,\$6.08]$, IQR $[\$2.68,\$4.50]$ — plus the CMT competitor-vendor
falsification check (adjacent-fare jump $+\$0.357$ VTS vs $+\$0.050$ CMT; placebo
$|\hat\alpha|$ mean $0.094$ vs $0.481$) and the caveats (iid vs clustered SEs; $\alpha$ on
fare residual vs level, so sub-\$15 is a supported extrapolation; regularization
sensitivity). Source: `../manuscript/taxi_application.tex` +
`../manuscript/figures/taxi_{alpha,b,utility,bootstrap}.png`. The earlier "boundary optimum
at \$2.50 / not publication-ready" paragraph is retired; this entry supersedes it per the
author's instruction. prelim compiles clean (22 pp, no undefined refs/citations).

## 2026-09-01 — Separated outer integration-by-parts repair from hard-boundary terms (Codex)
Revised item (3) of the opening “To discuss.” section to make the proof architecture
explicit. The fixed-buffered and slowly expanding windows address only the artificial
outer density-sieve trace at $(A,B)$. The internal Dirac terms at the hard-trim
endpoints are genuine first-order contributions and remain in the score, handled
separately from the outer-trace repair. Item (4) was shortened to the genuinely extra
checks: uniform control of endpoint-crossing bands, derivative-level density-sieve
control for curvature, and identification/curvature for $U_\epsilon$; bounded-density,
endpoint, nuisance-rate, and conditional-ULLN conditions are inherited rather than
relisted.

## 2026-09-01 — Additional supported-criterion assumptions stated in prelim (Codex)
Added the item (4) assumptions requested by the author to
`manuscript/prelim/prelim.tex`. The text distinguishes conditions inherited from
the untrimmed proof from the genuinely new or strengthened requirements: a fixed
buffered nuisance/density support and interior argument set, endpoint consistency,
generated-index control on the evaluation fold, local uniform nuisance and density
derivative consistency, bounded envelopes plus a conditional evaluation-fold ULLN,
and uniqueness/interior curvature for the supported target $U_\epsilon$. The
derivative rate is explicitly identified as necessary for curvature consistency,
not for value consistency alone. The author's other edits were preserved.

## 2026-09-01 — Two hard-truncation boundary repairs explained (Codex)
Expanded item (3) of the opening “To discuss.” section after author follow-up. The text
now separates the internal trim jumps at (a=\phi-u_0) and (b=\phi-l_0) from the
technical outer density-sieve endpoints (A,B), writes the integration-by-parts trace
term, and explains why (L_2) convergence alone does not control that trace. It records
the fixed buffered zero-trace construction as the shorter primary proof and the slowly
expanding-support construction with tail/trace and effective-resolution conditions as
the more flexible but longer alternative. The author's other edits were preserved.

---

## 2026-09-01 — Truncation TODOs resolved in opening section (Codex)
After pulling the author's Overleaf update, filled only the truncation-related TODOs in
the new “To discuss.” section and preserved the separate empirical-work TODO. The text
now names the classical Bahadur (1966) and Ghosh (1971) quantile results, explains the
moving-indicator signs and generated-index endpoint shift, spells out the weak
integration-by-parts pairing for the discontinuous density loading without using the
word “Riesz,” and gives notation for the supported criterion's uniform consistency and
curvature checks. The author's wording and all unrelated edits were retained.

---

## 2026-09-01 — Hard-truncation preface added to prelim (Codex)
Added a short opening “To discuss.” section to `manuscript/prelim/prelim.tex`. It records
that hard truncation changes the support of the estimand without propensity weighting and
summarizes the extra generated-boundary, moving-set, discontinuous density-Riesz,
consistency/curvature, and interpretation issues. The title and author/running head are
now explicit placeholders (`[Title Placeholder]`, `[Authors]`). Recompiled and visually
checked the first three pages; the existing minor box/spacing warnings remain nonfatal.

---

## 2026-09-01 — Taxi is a REAL interior application (corrects the "artifact" call) (Claude)
Reversing my earlier "taxi interior = data artifact" entry. Author reframing: the effect
depending on the fare *level* (percentages beat flat $2/$3/$4 at high fares, lose at low)
is exactly the phenomenon; "how low can the %-menu threshold go before it hurts tips" is the
interior question; extrapolating the optimal threshold is the method's contribution; and the
covariates absorb subpopulations given a clean $15 cutoff. The Haggag–Paci $5–25 restriction
removes the very low/high fares where the sign change lives — which is *why* restricted reads
boundary and the FULL sample reveals the interior. So unrestricted is the right data here.

**Result (full VTS credit sample, n=1.53M, Q=fare, φ₀=$15, D=1{fare≥15}, Y=tip).**
- α̂(η) sign-changes across the overlap window (Figures in `runs/screen_candidate/taxi/`).
- Inference-grade hard-trim: with a data-dense support (−5,10) and ridge 1.0 the design is
  well-conditioned (cond# 5.6e3, vs 1e18 for the naive wide/zero-ridge fit) and φ̂≈$5.3,
  interior, robust across regularization ($5.0–$7.6). avg trimmed α +0.10.
- Bootstrap (B=120, m=120k iid trips): φ̂ mean $3.61, median $3.51, 95% CI [$1.67,$6.08],
  **100% interior**. Whole distribution far below $15.
Reproducible: `experiments/scripts/taxi_perfrdd_application.py`. Written up as the main
application in `manuscript/taxi_application.tex` (+ `manuscript/figures/taxi_*.png`).

**For Codex:** this is now the main application (unrestricted interior), superseding the
restricted-sample boundary framing in the prelim; the prelim taxi text should be reconciled.
Caveats carried in the write-up: iid-trip bootstrap ignores driver/time clustering; α is on
the fare residual η while the menu effect also scales with fare level; φ̂ mildly
regularization-sensitive (qualitative interior-below-$15 conclusion is robust).

## 2026-09-01 — Decoupled split is the sole asymptotic target (Codex)
Clarified the proof scope after author review: the hard-trim CLT and variance are proved
only for the deliberately decoupled split, because that is the asymptotic construction
needed for the paper. Ordinary cross-fitting and full-sample nuisance reuse remain
possible implementation descriptions, but their covariance corrections are not proof
obligations and no asymptotic theorem is claimed for them. Updated the prelim text,
granular checklist, and Phase 3 task accordingly.

---

## 2026-09-01 — Granular hard-trim proof checklist added (Codex)
Expanded `manuscript/TODO.md` with a dependency-ordered checklist for the hard-trimmed
CLT. It now separates setup/assumptions, feasible consistency, outcome and density Riesz
limits, moving-set linearization, the generated-index Bahadur step, fixed-support versus
expanding-support density repairs, explicit influence-function assembly, foldwise CLT,
and feasible variance/inference. The checklist records the fixed-support density repair
as drafted, while keeping the Bahadur primitive, expanding-support audit, explicit score
bookkeeping, and reused-sample covariance estimator open. No methodological status was
changed; this is a clearer author-review map of the existing prelim proof.

---

## 2026-09-01 — Direct VTS-to-CMT prediction check rejects naive alpha transport (Codex)
Added `experiments/scripts/taxi_competitor_prediction_check.py` to test the
fitted VTS decomposition itself rather than merely compare cross-vendor means.
The VTS model is internally calibrated: on the VTS hard-trim interval, observed
low-fare tips minus its fixed-menu prediction average **-$0.001** (n=8,513), and
observed high-fare tips minus its percentage prediction average **-$0.005**
(n=1,715). Applying the same VTS first-stage, `b`, `beta`, and `alpha` to CMT
percentage rides gives a low-fare residual of **-$0.542** relative to the
VTS-implied percentage prediction (n=144,131); CMT's high-fare residual is
-$0.100. Subtracting that high-fare residual as a rough vendor/menu calibration
still leaves **-$0.443** at low fares.

Therefore CMT confirms the internal VTS fit and the direction of a negative
low-fare component, but it does **not** confirm transporting the positive local
VTS `alpha(eta)` unchanged below $15. The mismatch is informative: alpha is a
local menu-jump effect, not a global fare-invariant response. The check remains
non-causal because vendors and percentage menus differ and driver IDs are absent.
Details and the figure are in `experiments/datasets/taxi/DIRECT_COMPETITOR_CHECK.md`.

## 2026-09-01 — CMT low-fare proxy supplies a negative menu component (Codex)
Added `experiments/scripts/taxi_low_fare_proxy.py`, which uses all paper-restricted
January records with $5 <= fare < $15 and fare-cell fixed effects, common and
CMT-specific control slopes, and CMT-by-fare effects. The CMT-minus-VTS contrast
is evaluated at mean VTS controls in each cell, so it is an auxiliary
percentage-menu-minus-fixed-menu proxy rather than the causal PerfRDD
`alpha(eta)`. Across 25 cells (484,123 VTS and 428,062 CMT rides), the VTS-
distribution-weighted proxy is **-$0.212 per trip** (raw difference -$0.214),
negative on 90.5% of VTS low-fare mass, and crosses zero near **$12.8**. It is
about -$0.370 at $5.30, -$0.276 at $8.10, -$0.098 at $10.90, and +$0.144 at
$14.90; HC0 intervals are exploratory and iid.

This is the economically expected sign pattern for applying percentage tips to
low fares, and it is stable to observed-control adjustment. It does not identify
the VTS counterfactual without conditional vendor exchangeability: the public
January data lack the driver IDs needed for a within-driver CMT/VTS comparison,
and CMT's percentage menu differs from VTS's. Recommendation: retain the local
VTS hard-trim alpha as the identified treatment effect and use this proxy only in
a menu-aware calibration/sensitivity analysis. Details and figure are in
`experiments/datasets/taxi/LOW_FARE_PROXY.md` and the ignored run directory.

## 2026-09-01 — Competitor-only taxi placebo validates local jump but flags broad-window specification (Codex)
Added a vendor-selectable paper-restriction adapter and the reproducible
`experiments/scripts/taxi_competitor_check.py` diagnostic. The local January raw
parquet contains 478,012 CMT rides after the same restrictions (VTS has 541,318),
so no additional download was needed. Using identical VTS-standardized controls,
a locked 30,000-ride subsample, `eps=0.1`, support `[-6,11]`, and ridge `0.001`,
the CMT artificial split at $15 has weighted mean placebo alpha +0.091, mean
absolute alpha 0.094, and grid range [-0.288, +0.109] on its hard-trim window;
the VTS actual menu split has mean +0.406, range [+0.384, +0.953].

The direct adjacent meter-cell check is more favorable: VTS mean tips rise from
$2.269 at $14.90 to $2.626 at $15.30 (+$0.357), while CMT rises from $2.411 to
$2.461 (+$0.050). Thus CMT supports a local no-jump falsification, but its fitted
placebo alpha is not identically zero over the broad residual window (and remains
nonzero in larger exploratory CMT fits). This is a model-specification warning,
not a CMT treatment effect: CMT's percentage menu is present on both sides of
$15, and its baseline level/shape is not the VTS untreated counterfactual. The
comparison strengthens the interpretation of the VTS local discontinuity while
leaving transport to low-fare VTS counterfactuals unresolved. Durable details are
in `experiments/datasets/taxi/COMPETITOR_CHECK.md`; run outputs are in the ignored
`experiments/runs/taxi_competitor_check/` directory.

## 2026-09-01 — Prelim notation audit (Codex)
Compared the setup and estimator sections in `manuscript/prelim/prelim.tex` with
`prefRDD.tex`, `oldstuff.tex`, and `goodstuff.tex`. The core notation was already
consistent. Restored avoidable cosmetic deviations to the original `\tilde N_K`,
`G_K`, `\epsilon`, and definition-style assignments, while retaining the necessary
clarification that observed treatment is `D_i(phi_0)` and `D_i(phi)` is counterfactual.
The density block now uses the original basis/Gram symbols, with the fixed versus
expanding interval distinction stated in prose. Recompiled the 16-page prelim with
resolved bibliography and visually checked the revised setup/estimation pages.

## 2026-09-01 — Confirmation gate added; taxi "interior" is a data artifact (Claude)
Added a **full-n confirmation gate** to `screen_candidate` (committed `0354f6d`): when the
250k working screen flags INTERESTING, it recomputes the welfare gain (same cost) on the
FULL data and downgrades if it doesn't survive. Verified on `lending_roi` (continuous/
continuous ROI): flags at 250k (+0.1%), confirms False@884662 → boundary. Registered
`lending_roi` adapter.

Gated screen over in-hand continuous/continuous datasets:
- gpa: boundary. oulad: boundary. lending_roi: boundary (gate downgrades).
- nhanes: flags (+51%) but n=4.7k, too small for the gate to confirm — untrustworthy.
- **taxi: flagged INTERESTING and even confirmed True@1.53M — but this is a DATA-QUALITY
  ARTIFACT.** `load()` returns the *unrestricted* VTS sample; its α sign-change is driven by
  junk low/high fares. On the **paper-restricted** sample (`load_haggag_paci`, fares $5–25,
  n=541k) α has **0% negative mass** and the optimum is **boundary** — matching Codex.

**Key limitation of the gate:** it catches *sampling noise* (flat surface at small n) but NOT
*specification / data-quality bias* (both screen and gate use the same pooled PLM on the same
contaminated data). The screen is only as good as the adapter's restrictions. Suggestion for
Codex: consider making the registered taxi `load()` apply the paper restrictions, or add a
`taxi_hp` dataset, so a naive screen isn't fooled.

**Standing conclusion:** every *properly specified* continuous/continuous dataset tested is a
**boundary** optimum. Genuine sign-changing α needs large heterogeneous effects (mismatch/
discouragement), which the covariate-carrying public data we can get hasn't shown. Hunt
continues toward a remediation-type RD with covariates.

## 2026-09-01 — Prelim setup, supported target, and estimator completed (Codex)
Replaced the setup and estimation placeholders in `manuscript/prelim/prelim.tex`
with a self-contained statement of the method. The write-up now distinguishes data
generated under the deployed cutoff `phi_0` from counterfactual assignments under a
candidate `phi`; defines the latent-index decomposition, heterogeneous-effect outcome
model, propensity/survival identity, untrimmed value, exact hard-supported value, and
outer nuisance region; and states the identification interpretation of every object.
The estimation section now gives the OLS generated index and its exact first-order
error, empirical-quantile trim endpoints, stacked spline partial-linear regression,
orthogonal-series density and survival estimates, supported plug-in criterion, tie
rule, and decoupled versus reused-sample fold construction. It distinguishes outcome
and density dimensions `K_alpha` and `K_g`, taking both of order `K_n` only in the
theory. It also distinguishes a fixed density interval containing the support of `T`
from the deterministic expanding-support alternative. The resulting 16-page PDF
compiled without undefined references or citations, and the new setup/estimation
pages were visually checked.

## 2026-09-01 — Screen sharpened; Lending exhausted; boundary is the pattern (Claude)
Sharpened `screen_candidate` per the flat-optimizer insight (committed `c293686`):
(1) work on 250k rows so noise can't manufacture an interior; (2) gate on
`boundary_gain > 0` (interior must strictly beat treat-all/none); (3) cost-induced
interiors only for explainable non-negative c.

Result on Lending Club (covariate-rich, n=884k — the one in-hand large candidate):
- **repayment** outcome: boundary at ≥250k (α sign-change itself was a 30k artifact;
  100% negative mass at large n).
- **ROI** = (total_pymnt−funded)/funded: looked interior at 250k (φ*≈24, gain +0.1%) but
  **boundary at full 884k** (φ*=5.1, gain 0.0%, avg α≈+3e−4). Even 250k was fooled here —
  the flattest cases need full n or a bootstrap-of-gain-sign to call.

**Pattern:** every real dataset tested at large n is a **boundary** optimum — gpa, oulad,
taxi (Codex), Romania (null), Lending×2. The apparent interiors were all small-n noise; the
only large *gain* flag is nhanes (+51%) but at n=4.7k with a non-clean treatment. This is
consistent with the theory: an interior needs α to cross the cost with enough welfare
curvature, which real threshold treatments rarely have (α magnitudes ~±0.02–0.03 → flat).
Tool caveat: bump the screen to full-n (or add a bootstrap gain-sign gate) before declaring
any INTERESTING, since 250k passed lending-ROI. Decision needed from author on direction
(new large download e.g. HMDA; pool NHANES cycles; or accept boundary + use synthetic MC as
the interior demonstration).

## 2026-09-01 — Prelim numerical and taxi evidence added (Codex)
Added a concise evidence section to `manuscript/prelim/prelim.tex`. The hard-trim
Monte Carlo states the Gaussian DGP, exact 10% trim, 200 replications at each of
three sample sizes, and a three-row pooled comparison. Verified pooled
`n*MSE`/population-variance ratios are 0.991 for the honest split, 0.963 for
five-fold cross-fitting, and 0.965 for full-sample reuse; DGP-known 95% coverage
ranges from 0.945 to 0.975. The text explicitly treats this as a favorable-design
test of the hard boundary, density score, and density--boundary covariance, not a
misspecification exercise or feasible-SE result.

Also added the current restricted taxi diagnostic: 541,318 eligible January 2009
trips, deterministic 30,000-trip analysis sample, 10% exact hard trim, and 199 iid
full-reestimation bootstraps. The objective is tied over the $2.50--$3.80 plateau;
the reported $2.50 estimate is the first-grid tie break. The estimated gain relative
to the deployed $15 threshold is 34.71 cents per hard-trimmed trip with centered
bootstrap interval [27.16, 40.69] cents and 34.09% retention. The manuscript flags
the missing driver clusters and the unsupported transport from the local $15 menu
change to low fares, so the result is presented as an estimator demonstration rather
than a causal policy recommendation. The 13-page PDF compiled with bibliography and
resolved references, and pages 4--6 were visually checked.

## 2026-09-01 — FOLLOW-UP: lending_default interior does NOT survive the hard-trim (Claude)
Supersedes the optimistic "Interesting interior found" entry below. Ran RD validity + the
inference-grade `perfrdd_hard_trim` on full data (nuisance support [5,18], eps=0.1, c=0).

*RD validity at DTI=30.* McCrary density mild (log-jump −0.08, no bunching, 0.03% exactly at
30). But covariate balance shows significant jumps: `inq_last_6mths` (z=−3.9), `pub_rec`
(z=+3.1), `loan_amnt` (z=+2.5) — economically tiny (n=884k makes trivial jumps significant),
so a yellow flag on the clean-RD story, not fatal.

*Hard-trim vs screen disagree.* Full-data hard-trim → **boundary** φ*=54.9, avg α over window
= **−0.0235 (net negative)** ⇒ optimal policy is "treat no one." On a 100k subsample it gives
an **interior** ~31–33 across ridge∈{0,2,10}. Reconciliation: the interior peak's welfare is
≈0 (~2e−4), essentially tied with the trivial policy, so the argmax **flips interior↔boundary
by sample** — exactly why the screen bootstrap CI was [7.5, 54.9].

**Conclusion (changed):** α sign-change is robust, but the interior welfare optimum is **not**
— the surface is too flat to beat doing nothing, and full-data inference picks the boundary.
`lending_default` is a good methodology demonstration (the hard-trim correctly caught that the
pooled-PLM screen over-flagged) but **not a paper application**. Lesson for the screen: add a
welfare-gain-over-boundary check, not just interior-argmax. Data hunt continues.

## 2026-09-01 — Prelim proof build, phase 1 (Codex)
Built the hard-support proof in `manuscript/prelim/prelim.tex` as six explicit
modules: feasible consistency; sieve/Riesz convergence; moving sets; generated-index
trim quantiles; the density generated-index loading; and decoupled CLT assembly.
Migrated the completed feasible-consistency and moving-set proofs, replaced the Riesz
lemma's unnecessary nested-space premise by direct approximation along the actual
sieve sequence, and stated the boundary CLT conditionally on the remaining primitive
VC/Bahadur derivation. The document now keeps both density outer-boundary solutions:
(1) a fixed buffered interval with a zero-trace interior spline basis, for which the
integration-by-parts repair is closed; and (2) a deterministic slowly expanding
interval, for which the endpoint trace condition and effective resolution
`q_n = K_n / tau_n` are stated but the full expanding-support stochastic rate audit
remains open. No choice between the two has been imposed. The decoupled final CLT is
assembled conditionally on these modules; reused-sample density–boundary covariance
and its feasible estimator remain a separate phase. The 13-page Biometrika prelim
source compiles twice without errors or undefined references and was visually checked
page by page.

## 2026-09-01 — Interesting interior found: Lending Club repayment (Claude)
The two author-downloaded openICPSR packages are both blocked for this method:
**Adams (113908)** is code-only (no data); **Pop-Eleches (112645)** has the continuous
Baccalaureate outcome `bcg` but **no individual covariates** in any admin file (design uses
cutoff×year FEs), so there is no X to predict the score. Binding constraint crystallized:
we need Lindo-style individual covariates predicting the score + a continuous outcome +
sign-changing treatment; cutoff-FE RD packages (the common kind) lack the covariates.

Found the result in an **in-hand covariate-rich dataset**: new `lending_default` adapter —
Lending Club with a **repayment** outcome (Fully Paid=1; Charged Off/Default=0), Q=DTI,
underwriting cutoff 30, n≈884k. Screen (`screen_candidate`): **α̂(η) changes sign inside the
overlap window** (69% negative in-window mass, crossing at η≈12 where the data is dense — a
real crossing, not the tail oscillation that fooled taxi), giving a **clean single-peaked
interior welfare optimum** near φ*≈31–34 (vs the DTI-30 cutoff). Crossing DTI 30 is a more
defensible treatment than nhanes (lenders tighten pricing/screening at DTI thresholds).

**Bootstrap** (`bootstrap_candidate.py`, B=200, n_work=50k): the *qualitative* result is
robust — **92% of resamples sign-changing, 93% interior** — but the φ* *location* is
imprecise: mean 38.6, sd 11.4, 95% CI [7.5, 54.9]. Cause: flat welfare surface near the top
+ small α magnitudes (~±0.025). Two caveats shrink this: the bootstrap used 50k not the full
884k (full-n SE ~4× tighter), and this is the pooled-PLM screen, not the inference-grade
hard-trim. **Next:** run `perfrdd_hard_trim` on full data with a prespecified nuisance support
for a real CI; verify the RD (density/covariate continuity at DTI 30) before any causal claim.
Committed `code@main 6edca10` (adapter + bootstrap tool).

## 2026-09-01 — Bahadur references and fold/covariance distinction (Codex)
The original source is R. R. Bahadur (1966), “A Note on Quantiles in Large
Samples,” *Annals of Mathematical Statistics* 37(3), 577–580,
doi:10.1214/aoms/1177699450. Kiefer (1967), “On Bahadur's Representation of
Sample Quantiles,” 38(5), 1323–1342, doi:10.1214/aoms/1177698690, sharpens the
remainder. Ghosh (1971), “A New Proof of the Bahadur Representation of Quantiles
and an Application,” 42(6), 1957–1961, doi:10.1214/aoms/1177693063, proves the
weaker remainder \(o_p(n^{-1/2})\), which is sufficient for the present CLT under
lighter conditions. Modern references separate according to the object: van der
Vaart (1998), Chapter 21, treats empirical-quantile inversion; Koenker (2005),
Section 4.3, treats Bahadur representations for quantile regression; and
Bhattacharya (2020), arXiv:2012.13614, proves a uniform expansion for quantile
regression with generated dependent variables/covariates. Uniform nonparametric
quantile papers commonly cite Bahadur together with empirical-process/stochastic-
equicontinuity results rather than treating the 1966 fixed-distribution theorem as
covering an estimated index.

For PerfRDD, Bahadur (1966) is the correct historical citation, but it does not by
itself prove (A3''). The needed result follows from a short local argument for the
VC class of halfspaces
\(F_n(q,a)=P_n\{1(a^\top X\le q)\}\). Uniform stochastic equicontinuity near
\((q_p,\gamma)\), together with

\[
F(q,\gamma+\delta)-p
=f_T(q_p)\{q-q_p-m_p^\top\delta\}
+o(|q-q_p|+\|\delta\|),
\]

gives

\[
\hat Q_p(\hat\gamma^\top X)-q_p
=\frac{P_n\{p-1(T\le q_p)\}}{f_T(q_p)}
+m_p^\top(\hat\gamma-\gamma)+o_p(n^{-1/2}).
\]

Recommendation: cite Bahadur (1966) plus a modern quantile exposition, cite the
generated-variable paper as a close analogue, and include this short derivation.
Kiefer (1967) is optional because the proof does not use his sharp remainder rate;
Ghosh (1971) is the closest classical citation to the strength actually required.

The author's fold clarification resolves task #5: the theorem uses separate
\(\hat\gamma^\alpha,\hat\gamma^g,\hat\gamma^U\) from separate training folds for
the alpha, density, and evaluation blocks. The lower and upper boundary half-folds
use two additional boundary-specific gamma estimates. This bookkeeping question is
distinct from density–boundary sample reuse. Even with different gamma estimates,
the density score \(r_{g,\epsilon}(T)\) and empirical-quantile score
\(p-1\{T\le q_p\}\) are correlated when computed from the same observations. The
theorem's boundary folds are disjoint from the density fold, so that covariance is
zero. The honest-split Monte Carlo also has disjoint boundary and density folds. In
the five-fold and full-sample Monte Carlo variants, however, the same `train_idx`
forms both the density estimate and both trim quantiles; their DGP-known variance
therefore correctly includes density–boundary covariance. Adopting the theorem-style
split for feasible inference removes that covariance but uses less data; retaining
cross-fitting/full-sample reuse requires estimating and carrying the covariance.

## 2026-08-31 — Dataset hunt: Romania null, strong candidates are access-gated (Claude)
Searched for downloadable datasets with a *real* threshold-treatment + continuous outcome
+ pre-treatment covariates + heterogeneous (sign-changing) effects.

**Tested — Pop-Eleches & Urquiola (2013) Romania school-admission RDD.** Pulled the survey
subsample `df7.csv` (LFS media mirror:
`https://media.githubusercontent.com/media/s6soverd/Microeconometrics-Final-Project/master/data/df7.csv`,
~12k rows). Running var `dzag` (score − school cutoff), D=1{dzag≥0}, background X (child/parent
demographics, home conditions; behavioral-response columns excluded). Screen → **null**:
α̂ flat and tiny (−0.021..−0.005), 100% negative in-window mass, R²=0.08, retention 0.10.
Cause: `df7`'s outcome `Y∈{5,6,7}` is coarse; the continuous Baccalaureate score sits in
the covariate-less big files (`df4–df6`). Not dead — worth retrying with the continuous
Bacc outcome from the **full Pop-Eleches package** (openICPSR 112645), which has both.

**Access wall.** The strongest candidates are all openICPSR (login required, can't automate):
Pop-Eleches full (112645); **Adams Scholarship, Cohodes–Goodman (113908)** — merit aid that
*lowered* completion via mismatch → documented sign-changing effect, GPA/SAT threshold, has
covariates (top pick); Georgia HOPE; Florida Bright Futures. **Author action needed:** log in
and download one; then Claude builds the adapter + runs the full pipeline + bootstrap.

**In-hand status:** nhanes remains the only screened dataset with sign-changing α (55% neg
mass, interior) but n≈4.7k and the diagnosis→SBP "treatment" is not a clean intervention, so
it is an illustrative pipeline demo, not a paper result. gpa/oulad boundary; taxi borderline.

## 2026-08-31 — Dataset screening harness built + first-pass screen (Claude)
Added `experiments/scripts/screen_candidate.py`: ingests a registered dataset, fits the
pooled PLM, and emits three review figures — `alpha.png` (α̂(η)), `b.png` (b̂(η)),
`utility.png` (Û(φ) with argmax marked) — plus `description.md`/`summary.json` with the
screening verdict. **Goal: find a dataset with an interesting interior welfare optimum,
i.e. α̂(η) non-constant and sign-changing across the overlap window** (a sign-definite α
forces a boundary policy at zero cost). Explainable treatment costs `c` are supported: the
harness reports the cost range that induces an interior optimum (in outcome units).

Key robustness lesson baked in: the pooled-PLM spline **oscillates/extrapolates in the
low-density η tails**, so a naive "does α dip below zero" flag over-fires (it flagged all
of gpa/taxi/oulad/nhanes). Fixed by (i) assessing the crossing only on the overlap window
[l₀,u₀], and (ii) gating on **data mass**: fraction of in-window observations with α̂<0
must be ≥10% on each side. After the fix the first-pass screen matches known results —
gpa (4% neg mass) and oulad (0%) → boundary; taxi (14%) and nhanes (55%) → sign-changing.
This is a fast pooled-PLM triage; passing candidates go to `perfrdd_hard_trim` for
inference-grade estimates. Next: ingest new public candidates (see TODO) and screen them.

## 2026-08-31 — Taxi treatment-effect audit finds no negative fitted alpha (Codex)
Exported every component of the restricted 30,000-trip taxi outcome regression. The
hard-trim interval for the estimated fare residual is `[0.216,8.629]`; the fitted
`alpha(eta)` is positive throughout it, ranging from $0.384 to $0.953, with a
hard-window observation-weighted mean of $0.406. Thus the fitted model contains no
residual-defined subgroup for which the percentage-menu effect is negative. This does
not identify the effect at low fare levels: treatment is deterministic in fare, and the
model restricts the treatment effect to depend on residual `eta`, not fare or displayed
menu values. The point utility maximum is also exactly tied on the $2.50--$3.80 grid
plateau; the reported $2.50 is the first-grid-point `argmax`, not a uniquely identified
threshold. Added reproducible exports for `alpha(eta)`, baseline `b(eta)`, the linear
control vector `beta`, and a combined diagnostic figure. The no-cost global threshold
recommendation remains an unsupported transport exercise rather than a causal result.

## 2026-08-31 — Restricted taxi bootstrap is stable but boundary-valued (Codex)
Corrected the paper-facing taxi sample before bootstrapping: the earlier generic pilot
used all VTS credit-card rides, whereas Haggag--Paci's main RDD excludes tolls, taxes,
and surcharges, uses daytime/standard-meter restrictions, and limits fares to $5--$25.
The public January data contain 541,318 eligible rows; a locked 30,000-trip sample was
bootstrapped 199 times with full re-estimation of every nuisance, trim endpoint, utility
curve, and argmax. With driver tip revenue as the objective (`cost=0`), the estimate and
all 199 replications select the $2.50 lower policy boundary. Moving from the observed
$15 rule to percentage suggestions on all eligible fares has an estimated gain of 34.71
cents per hard-trimmed trip (centered-bootstrap 95% interval [27.16,40.69] cents;
bootstrap SE 3.29 cents). This establishes numerical stability only: iid trip resampling,
34.1% hard retention, a roughly 297,000 baseline condition number, and strong
counterfactual extrapolation prevent a publication-ready causal recommendation. The
earlier unrestricted $0.20-cost curve is superseded for paper-facing work. Details:
`experiments/datasets/taxi/BOOTSTRAP_RESULTS.md`.

## 2026-08-31 — Taxi hard-trim utility curve made economically legible (Codex)
Re-expressed the existing January 2009 NYC taxi hard-trim pilot in dollars: tip
benefit minus an explicit cost per trip assigned the percentage-tip regime. At an
illustrative cost of $0.20 per treated trip, the regularized full-sample curve selects
a fare threshold of $8.46 and five-fold cross-fitting selects $8.27, versus the observed
$15 policy. The estimated improvement over the observed policy is only 0.866 and 0.975
cents per hard-trimmed trip, respectively. Thus taxi is more interpretable than GPA and
does produce an interior optimum, but the current evidence remains exploratory: January
only, deterministic 30,000-trip subsample, pilot-derived nuisance support `[-6,11]`, an
illustrative rather than measured policy cost, and no application confidence band. The
reproducible plotter is `experiments/scripts/taxi_utility_curve.py`.

## 2026-08-31 — Collaboration rules tightened + local TeX fixed (Claude)
Author-approved refinements to `COLLABORATION.md` (Codex: please read the updated
protocol):
- **Manuscript auto-push confirmed.** Paper updates push automatically like code. Every
  manuscript push must add a `../manuscript/CHANGELOG.md` entry (human-readable "what
  changed, where, why" + commit hash) so the authors can track updates without diffs.
  Seeded `CHANGELOG.md` with the CLT/natbib commit `138b1eb`.
- **Compile before manuscript push.** The local TeX now builds the paper for real.
- **Fetch immediately before every push; on non-ff, rebase your task commit and re-check.**
  Short-lived task branches for edits to a file the other agent may also be in.
- **Collision-avoidance / task claims.** In-progress tasks in `TODO.md` carry an owner +
  touched-files line, e.g. `_(owner: Claude · files: prefRDD.tex §trim · since ...)_`.
- Deferred (author): log-compaction discipline for this growing file.

Toolchain note (this machine): the configured tlmgr mirror served a corrupt `todonotes`
archive. Fixed by pointing tlmgr at the frozen 2024 archive
(`https://texlive.info/tlnet-archive/2024/12/31/tlnet`) and reinstalling `todonotes`
(plus staged `caption`/`scalefnt`). `pdflatex→bibtex→pdflatex` now builds `prefRDD.tex`
clean (45 pp, no undefined refs).

## 2026-08-31 — GPA welfare menu yields boundary policies throughout (Codex)
With a separate skeptic-agent audit, prespecified 16 GPA welfare outcomes: five direct
progression outcomes, five inherited/physical missing-GPA values, and six modest
leave/return stress tests. Ran four full-sample ridge levels, five-fold cross-fitting, and
an expanded-grid audit at costs `{0,.025,.05,.10}` for every outcome. Full and cross-fit
effects agree closely. Direct progression outcomes are negative and choose the lower
policy boundary; all physical and modest status-adjusted GPA composites are positive and
choose the upper boundary. All 16 no-cost optima and all 64 expanded-grid cost audits are
boundary solutions. Conclusion: reasonable linear welfare calibrations clarify the
performance-persistence tradeoff but do not identify an interior threshold. Details:
`experiments/datasets/gpa/WELFARE_RESULTS.md`.

## 2026-08-31 — GPA redesign converted to exact hard trimming (Codex)
Replaced the uncommitted smooth-gate GPA runner with
`experiments/scripts/gpa_redesign_hard_trim.py` and ran 70 locked specifications: 14
outcomes times four full-sample ridge values plus a five-fold unregularized cross-fit.
All fits use `eps=0.1`, pilot-fixed nuisance support `(-2, 0)`, policy grid `[-0.6,0.6]`,
and cost zero. Full-sample and cross-fit hard-window effects agree closely. Persistence
effects are negative; the selected observed-GPA and ordinary composite effects are
positive; the penalized composite crosses zero around a GPA-equivalent penalty of five.
Every policy optimum is at a grid boundary, so this is **not** evidence for an interior
optimal threshold. The runner provides point estimates only; boundary-aware application
inference remains open. Full results and limitations are in
`experiments/datasets/gpa/HARD_TRIM_RESULTS.md`.

## 2026-08-31 — Collaboration channels made durable and consistent (Codex)
Moved the canonical decision log from the untracked workspace root into the code
repository and added `COLLABORATION.md` as the canonical protocol. The manuscript task
board remains `manuscript/TODO.md`. Resolved the conflicting push rules: after verified,
authorized work, agents commit and push affected repositories by default unless the
author explicitly requests a local hold. Unrelated working-tree changes remain excluded.

## 2026-08-31 — natbib wired + committed (Claude)
Added `\usepackage{natbib}`, `\bibliographystyle{plainnat}` + `\bibliography{references}`,
a `vandervaart1998` book entry, and converted the consistency-lemma cite to `\citep`.
Verified pdflatex→bibtex→pdflatex (exit 0, no undefined citations, entry renders in `.bbl`).
Committed to `manuscript@master` (`138b1eb`) and `code@main` (`36b738b`, coordination
pointer only). The three original coordination files at the workspace root were not in a
Git repository; this entry has now been relocated into the tracked code repository.

## 2026-08-31 — #2 and #3 drafted in prefRDD.tex (Claude)
**#2 (intercept assumption):** added **(A8)** to `ass:main` — `X` contains a constant,
`gamma` is the population LS coefficient of `Q` on `X`, so `E[eta]=0`, `E[eta|X]=0`,
`E[X eta]=0`; noted the outcome X-block omits the redundant constant (carried by the
spline partition of unity) so (A4) is unaffected. Referenced it in `lem:bdry-clt`'s
zero-covariance step and the trim-assumption retention clause.
**#3 (feasible consistency):** added Lemma `lem:trim-consistency` before the limit-
distribution subsection — `sup_phi |Uhat_eps - U_eps| ->p 0` via a three-part split
(oracle Glivenko–Cantelli; nuisance+endpoint replacement using (A5') and endpoint
consistency; generated-regressor replacement using `max_i|etahat-eta| = o_p(n^{-1/4})`),
then argmax consistency `phihat_eps ->p phi*_eps`. Theorem `thm:trim` proof now opens by
citing it instead of asserting consistency from the oracle class.
Compiles clean locally (multi-pass, refs resolve). The paper now uses a proper
`\citep{vandervaart1998}` entry with natbib. The tracked PDF still needs to be rebuilt on
Overleaf because the local build used a todonotes stub.

## 2026-08-31 — Coordination protocol established (Claude)
Set up a shared signed decision log and symmetric instructions for Claude and Codex.
Decision: task board = `manuscript/TODO.md`; decision log = `code/RESEARCH_LOG.md` after
the durability cleanup. Local-only audits in `outputs/` must be moved into a tracked path
before another machine or fresh clone can rely on them.

## 2026-08-31 — CLT audit of the hard-trimmed estimator (Claude)
Full read of Theorem `thm:trim` (`manuscript/prefRDD.tex:1864`) and its lemmas
(`lem:moving-set`, `lem:bdry-clt`, density IBP), against baseline `thm:main`. The
2026-08-17 audit (`outputs/proof_audit_2026-08-17.md`) is **stale** — the proof was
rewritten and closes most of it. Remaining items filed as tasks #1–#5 in the TODO.
Architecture is sound; the Monte Carlo (`ssec:trim-mc`) validates the variance formula
(coverage 0.945–0.975).

## 2026-08-31 — #1 resolved: same-sample boundary variance is correct (Claude)
Checked whether estimating `gamma` on the same half-fold as the empirical trim quantile
breaks the Bahadur expansion / additive variance in `lem:bdry-clt`. Simulation
(`scratchpad/verify_bdry.py`, Gaussian DGP, p=0.9): theory `sigma_p^2 = p(1-p)/f_T^2 +
m_p' Sigma_gamma m_p = 4.5645`; same-sample `n*Var = 4.51`, independent-`gamma` split
`n*Var = 4.60` — indistinguishable (~1 MC-SE), both match theory. **Conclusion:** the
expansion and zero cross-covariance hold same-sample (VC/Donsker generated-index
quantile; cross-covariance vanishes because `E[eta|X]=0`). No sub-split needed. Only
follow-up: (A3'') should cite/derive the uniform Bahadur expansion rather than posit it.

## 2026-09-08 — Differing-slopes simulation and variance check (Codex)
Added `experiments/scripts/differing_slopes_simulation.py` to test the proposed
level-dependent treatment effect model
`Y=b(eta)+D*alpha(eta)+X'beta1+(D*X)'beta2+epsilon` in a known-target Gaussian
benchmark. The DGP sets `Q=X1+eta`, `D=1{Q>0}`, observes `eta` and uses the true
normal distribution of `T=X1` for the smooth policy utility; the trim interval is
fixed at the 10th/90th percentiles of `eta`. The full model's population optimum is
`phi*=-0.1201`, while the alpha-only pseudo-target is `-0.3393`.

The short run (`n={400,800,1600,3200}`, 150 replications per cell) shows the
differing-slopes estimator nearly centered on the full target (bias between -0.0055
and +0.0043) and RMSE falling from 0.097 to 0.039, while the alpha-only fit remains
far from the full target (bias about -0.40 to -0.52). After correcting the delta
method implementation so that the *average* policy gradient multiplies each OLS
influence (rather than multiplying observation-specific gradients and influences),
the estimated-to-Monte-Carlo variance ratios for differing slopes are
`1.08, 1.01, 0.90, 0.83`; coverage is `0.973, 0.953, 0.933, 0.900` in this 150-rep
run. A larger 300-rep run at `n={6400,12800}` gives ratios `0.994` and `1.106`,
with coverage `0.937` and `0.963`. The remaining small-sample undercoverage is
consistent with finite-replication noise and mild root-n bias; the variance check is
substantially improved and is not showing systematic overestimation.

These results support the *point-estimation idea* when the D×X block is correctly
specified and support the corrected plug-in variance in this conditional benchmark.
They do not validate the full PerfRDD theorem: the first-stage residual `eta`, trim
endpoints, and `T` distribution are treated as known, and the utility is smoothed by
the known Gaussian tail. Generated-index, moving-hard-boundary, density-sieve, and
ordinary cross-fitting terms remain untested. The reproducible JSON/CSV outputs are
under the ignored `experiments/runs/` directory; targeted unit tests are in
`experiments/tests/test_differing_slopes_simulation.py`.

## 2026-09-08 — Differing-slopes robustness scenarios (Codex)
Extended the simulation runner with explicit reproducible scenarios for a null
interaction, stronger level dependence, t(5) errors, skewed errors, and
heteroskedastic errors (`--scenario`; all preserve the same known-target setup).
The null-interaction run (`n={800,3200,6400}`, 250 replications) gives the same
population target for both specifications and variance ratios converging to
`1.14, 0.92, 1.01` for alpha-only and `1.25, 0.95, 1.04` for differing slopes.
Thus the extra D×X block costs little efficiency when it is unnecessary.

Under stronger interaction (`beta2=(1.6,0.5)`), the full model remains centered
(bias about `-0.004, -0.004, -0.003` at `n={800,3200,6400}`), while alpha-only
selects highly negative thresholds and misses the true target by roughly 1.5--2.2
units. A 500-replication larger run gives differing-slopes variance ratios
`0.994` and `1.051` and coverage `0.958` and `0.952` at `n=6400,12800`.

For t(5), skewed, and heteroskedastic outcome errors, the differing-slopes estimator
has negligible bias. In the 250-replication runs, variance ratios are respectively
`0.96--0.99` (t(5) at `n<=3200`), `0.95--1.23` (skewed), and `0.94--1.14`
(heteroskedastic). The t(5) 500-replication follow-up gives ratios `0.980` and
`1.072` and coverage `0.958` and `0.962` at `n=6400,12800`, resolving the earlier
finite-sample undercoverage. These checks support the corrected sandwich under
non-Gaussian and heteroskedastic errors, conditional on known eta and fixed support.
They do not test generated-index or moving-boundary terms.

## 2026-09-08 — Full differing-slopes robustness battery (Codex)
Added `experiments/scripts/differing_slopes_full_pipeline.py`, which estimates the
first-stage residual and hard-trim endpoints rather than conditioning on them. It
also compares Gaussian and spline running-variable tails, full-sample OLS, a
moderately ridge-regularized fit, and five-fold cross-fitting. The baseline run
(`n={500,1000,2000,4000}`, 100 replications) gives full-model biases
`-0.012,-0.010,-0.011,-0.000` for Gaussian full-sample OLS and
`-0.008,-0.009,-0.013,+0.001` for spline OLS, with RMSE log--log slopes about
`-0.57` in both cases. Cross-fitting tracks the full-sample estimates closely;
the spline tail is slightly less variable. The alpha-only generated-index fit
remains biased by roughly `-0.43` to `-0.50` relative to the differing-slopes
target. The ridge setting used here (`0.50`) introduces finite-sample bias
(about `+0.065` at `n=500`, declining to `+0.026` at `n=4000`).

The nonlinear misspecification scenario adds a true treated effect
`0.90*(X1^2-1)` while fitting only linear D×X terms. The true optimum is
`0.4955`, but all linear differing-slopes variants converge near `0.405`, leaving
an approximately `-0.09` pseudo-target bias even at `n=4000`. This is a useful
negative control: the extension is not robust to omitted nonlinear effect
heterogeneity. The weak-curvature scenario has target `-0.5000` and curvature
`-0.0475`; the full model remains centered by `n=4000`, but RMSE is about `0.12`
and the n-scaled Monte Carlo variance is roughly `55`, versus roughly `4--6` in
the baseline. The boundary scenario selects the upper policy bound in `83--100%`
of replications and confirms that interior CLT intervals are not meaningful for
boundary optima.

For clustered errors, clusters of 20 share a mean-zero treated-outcome shock.
The oracle estimator's iid variance estimate is only `0.40--0.54` of the Monte
Carlo variance and yields coverage `0.79--0.85`; a cluster-sum variance estimate
is much closer (`0.82--1.12`) with coverage `0.89--0.96`. Thus clustering is a
material application concern; cross-fitting does not repair dependence by itself.

The run initially exposed a performance bottleneck from scalar spline evaluations;
the objective search was vectorized and the optimized implementation passed 14
targeted/regression tests. These experiments still do not provide the full
generated-index/moving-boundary/density-Riesz variance theorem: generated variants
are evaluated by Monte Carlo dispersion, while the analytic variance check remains
oracle-index and conditional on fixed support.
