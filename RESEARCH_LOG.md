# PerfRDD research log — shared between Claude and Codex

This tracked file is the single source of truth for **verified findings and decisions**.
The operating rules are in `COLLABORATION.md`; forward-looking work is tracked in
`../manuscript/TODO.md`.

Add new entries immediately below the divider, newest first. Date and sign every entry.
Never edit or delete another agent's entry; add a follow-up when a conclusion changes.

---

## 2026-09-23 - Nonlinear bootstrap rerun with spline g and p_X; weekly note updated (Claude)

Follow-up to the U_J switch entry below. Reran
`differing_slopes_nonlinear_bootstrap.py` with the spline tails, same seeds as
Codex's run (20260928 for linear/nonlinear at n=800,1600,3200; 20260929 for
nonlinear n=6400), 50 outer x 199 draws, 8 local workers. Outputs:
`outputs/differing_slopes_nonlinear_bootstrap_spline_20260923{,_n6400}`.
Nonlinear DGP: linear fit bias -0.035/-0.024/-0.035/-0.041 with coverage
0.92/0.90/0.82/0.56 (Gaussian tail: 0.94/0.88/0.94/0.64); quadratic fit bias
-0.016/0.000/-0.011/-0.008, coverage 0.94/0.96/0.98/0.94; spline fit bias
0.004/-0.004/-0.003/-0.011, coverage 0.96/0.94/0.96/0.92. The linear DGP stays
covered (0.90-0.96). The conclusion is unchanged, and the linear fit's
undercoverage now appears by n=3200. `this_week.tex` Simulations
table and setup sentence updated (manuscript fca05b3 and follow-up); the
coordination hold on that subsection is released.

-- Claude

## 2026-09-23 - Differing-slopes code now maximizes U_J with spline g and p_X (Claude)

At the author's request every differing-slopes estimator now maximizes
U_J(phi) = mean_i I_i[(alpha_hat_i - c) Gbar_hat(phi - eta_i) + beta2_hat' H_X_hat(phi - eta_i)]
with g_hat and p_X_hat from the Lebesgue-Gram projection in the new
`experiments/methods/weighted_tails.py` (tests: `test_weighted_tails.py`).
Changed: `taxi_differing_slopes.py`, `differing_slopes_full_pipeline.py` and
`_feasible_bootstrap.py` (Gaussian-tail feasible variants replaced by
full_spline_ols/ridge/alpha_only; oracle keeps population tails),
`differing_slopes_distributional_battery.py`, `differing_slopes_nonlinear_outcome.py`
(hence `_nonlinear_bootstrap.py`), `differing_slopes_simulation.py`,
`nonlinear_slopes_simulation.py` (sandwich variance now adds the density-sample
influence). Checks: spline-tail full-pipeline estimates are bit-identical to the
old SplineTail; DS known-target variance ratio 0.85/0.95, coverage 0.94/0.955
(n=1600/6400); quadratic-slopes coverage 0.935/0.95.

Results: taxi (restricted VTS) U_J phi* = $12.66 (treat fares >= $12.90; own-fare
check $12.51, same cell); alpha-only $5.41. Battery rerun
(`outputs/differing_slopes_distribution_spline_20260923`): unchanged except the
t5 index law, bias -0.016 (s.e. 0.002) at n=4800, which is spline smoothing bias
of the default 9-function basis (14+ functions: -0.003, RMSE 0.026). Full pipeline
and bootstrap reruns (`outputs/*_spline_20260923`): spline OLS unchanged; spline
ridge bias 0.019, RMSE 0.030 at n=6400, coverage 0.86/0.92/0.90.

**Coordination (Codex):** the nonlinear-outcome bootstrap table in
`this_week.tex` (Simulations) was computed with the Gaussian score tail. I am
rerunning it with the spline tails (same seeds) and will update that table and
its text; please leave that subsection to me until the rerun is pushed.

-- Claude

## 2026-09-23 — Defer the differing-slopes generated-index loading (Codex)

At the author's request, `manuscript/this_week.tex` Section 4.3 is now an
explicit proof TODO rather than a claimed loading calculation. The retained
rate argument is conditional: fixed-dimensional OLS gives
`gamma_hat - gamma = O_p(n^{-1/2})`, and a locally bounded derivative of the
integrated `B`-functional would transfer that rate to the generated-index
term. The derivative bound, stochastic Taylor remainder, joint first-order
loading, and hard-trim moving-set effects are unproved. The open obligation is
also recorded in `manuscript/TODO.md` (manuscript commits `6fe1469` and
`f5dfe4e`). No theorem claim or estimator changed.

-- Codex

## 2026-09-23 - this_week.tex versus the taxi treatment-effect fit (Claude)

Compared the current weekly note (manuscript 97afb42, Sections 2-3.1) with
`taxi_differing_slopes.py`. They agree on the model (W = alpha(eta) + X'beta2),
the first stage (OLS of Q on (1, X)), and the effect blocks: alpha-hat from the
D*spline(eta-hat) block and beta2-hat from the D*X block of one stacked
regression. They differ in how that regression is fit:
- The taxi design adds an explicit intercept to a clamped B-spline basis that
  already sums to one; the note's Z_K has no intercept. Under ridge this changes
  the fitted effect: with the same lambda, dropping the intercept moves W-hat by
  up to 0.23 in the window.
- Ridge (lambda*sqrt(n)) on the D*X, spline and D*spline blocks, CV-chosen at the
  grid minimum 0.1; the note's fit is unpenalized. The note's fit
  (unpenalized, no intercept) differs from the script's W-hat by up to 0.046 in
  the window (mean -0.023); beta2 for distance 0.372 vs 0.360.
- Full sample for gamma, outcome and evaluation versus separate folds.
- Spline support is the 0.5-99.5% range of eta-hat with clamped evaluation (1%
  of rows clipped but kept), not a fixed outer region with eligibility.
  Basis dimension 14 (10 interior knots, n_treated^(1/5) rule), at n^(1/5).
- Taxi X are standardized, so alpha-hat is the effect at average covariates and
  beta2 is per SD. The note uses raw X, where alpha is the effect at X = 0; it
  equals E(W|eta) only for centered X with X independent of eta.
- The note assumes continuous scores (Section 2); taxi fares are on a $0.40 lattice.
The utility aggregation difference (own-Q indicator versus integrated tails)
is already stated in the note (lines 251-253).

-- Claude

## 2026-09-23 — Weekly note uses vector weighted density directly (Codex)

At the author's request, removed the optional scalar symbol and scalar
projection from `manuscript/this_week.tex`. The note's proposed estimate is
the vector spline projection \(\hat p_X\); the differing-slopes term is
written directly as \(\hat\beta_2^\top\hat p_X(t)\). With \(\beta_2\)
held fixed in the distribution-fold calculation, linearity gives the
same identity
\(L_{\rho,\phi}(\beta_2^\top\hat p_X)
=\mathbb P_{n_\rho}[r_L(T)\beta_2^\top X]\).
The exact decomposition, approximation-error product, and
generated-index evaluation loading were restated without a separate
scalar density. No estimator, target, or rate condition changed.
Also replaced stale numeric section references after the author's new
opening section. The nine-page LaTeX note compiled without errors or
undefined references and the affected pages were visually checked.

## 2026-09-23 — Add Gongjun discussion notes to weekly note (Codex)

Added Section 1, ``Discussion with Gongjun,'' to `manuscript/this_week.tex`.
The section records possible education datasets, the Texas accelerated-instruction
cutoff proposal, its sharp versus fuzzy RD interpretation, and the distinction
between Sales--Hansen's fixed-cutoff RD setting and global threshold optimization.
Verification: `latexmk -pdf -interaction=nonstopmode -halt-on-error this_week.tex`
from `manuscript/` completed successfully and produced a nine-page PDF.

-- Codex

## 2026-09-23 — Weekly note terminology: utility derivative (Codex)

At the author's request, removed the unexplained “slope score” terminology
from `manuscript/this_week.tex`. The precise object is the contribution
\(-\theta_\phi\) of \(\beta_2^\top X\) to the derivative of the
untrimmed utility, where
\(\theta_\phi=\beta_2^\top E[p_X(\phi-\eta)]\). The note now defines
\(\hat\theta_\phi\) and expands \(\hat\theta_\phi-\theta_\phi\)
directly; negating that equivalent expansion gives the contribution to
the utility derivative. The hard-trimmed derivative still includes
\(I(\eta)\). No estimand or estimator changed. The source compiled to
eight pages and the revised pages were visually checked.

## 2026-09-23 — Weekly note §4 marked as differing-slopes next steps (Codex)

At the author's request, `manuscript/this_week.tex` §4 now presents the
remaining work toward the hard-trimmed differing-slopes threshold CLT as
next steps. Section 3 remains a conditional untrimmed slope-score
expansion; the joint baseline/slope assembly, endpoint and moving-set
terms, local uniformity, and argmax step have not been claimed proved.
This is an exposition/status clarification only: no estimator, assumption,
or formula changed. The eight-page note compiled and its revised page
was visually checked.

## 2026-09-23 — Weekly note §3 derivation clarified (Codex)

Reworked the exposition in `manuscript/this_week.tex` §3 without changing
the estimator or the displayed influence expansion. The two representations
of \(\theta_\phi=\beta_2^\top E[p_X(\phi-\eta)]
=\beta_2^\top E[Xf_\eta(\phi-T)]\) explain the separate evaluation and
distribution fold averages. The \(L_2(dt)\) representer
\(r_{\rho,\phi}(t)=f_\eta(\phi-t)\) converts the *integrated* spline
weighted-density error into a centered distribution-fold average plus
projection errors; no pointwise root-\(n\) density rate is claimed.
The \(\beta_2\) contribution requires its outcome-fold expansion and
the generated-index outcome loading. The same OLS \(\hat\gamma\) also
shifts evaluation residuals and distribution indices, so the three
loadings must be combined before variance calculation. The note still
labels the spline, generated-index, and local-uniform remainder bounds
as proof obligations. Verified with `latexmk -pdf -interaction=nonstopmode
-halt-on-error this_week.tex` and visual inspection of the affected
eight-page PDF; no LaTeX errors, undefined references, or horizontal
overflow.

## 2026-09-23 — Weekly note §2.2 weighted-density projection clarified (Codex)

Verified the moment identities used by the distribution fit:
\(E[N_L(T)]=\int N_L g\) and
\(E[N_L(T)X^\top]=\int N_L p_X^\top\), with
\(p_X(t)=E[X\mid T=t]f_T(t)\). Thus the Lebesgue-Gram spline
coefficients are obtained from averages of \(N_L(\hat T)\) and
\(N_L(\hat T)X\); this is a one-dimensional density and weighted-density
fit, not regression of \(X\) on \(T\). The scalar response
\(\hat\beta_2^\top X\) yields exactly \(\hat\beta_2^\top\hat p_X\)
by linearity, with no change to the estimator or its first-stage
uncertainty. Rewrote `manuscript/this_week.tex` §2.2 accordingly and
confirmed an eight-page LaTeX build without errors, undefined references,
or horizontal overflow. The displayed identities use the note's
compact-support simplification; an outer-support correction is needed
otherwise.

## 2026-09-23 — NAEP / TIMSS / PISA dataset scan (Claude)

Desk scan for the open "ingest and screen new public candidates" task. Nothing was
downloaded or screened. Main finding: NAEP, TIMSS, and PISA are low-stakes and matrix-sampled,
so no treatment is assigned by their scores, and their reporting cutoffs are labels. None
gives `Q -> D -> Y` with the assessment score as `Q`. They are useful as `Y` (school-level
policy scores; NAEP or NAEP-linked SEDA outcomes), as `X` (PISA at 15 explaining a later
high-stakes score in the LSAY cohorts), and as a realistic `(X, Q)` law for semi-synthetic
simulations. Top lead: the school-meal CEP cutoff on the Identified Student Percentage
(40% until 2023, 25% since; a 2025 House proposal for 60% was dropped) with SEDA or NAEP
outcomes. The reimbursement formula `min(1, 1.6 x ISP)` gives a derivable, Q-dependent
cost, a structural reason for an interior optimum. Main risk: published ISP lists are
truncated at the near-eligible floor; CCD direct-certification counts (2016-17 onward, some
states) are an untruncated proxy. Rejected: birthdate entry cutoffs (no spread in `T`) and
Maimonides class-size caps (multi-cutoff, few schools). Details, access, and sources:
`experiments/datasets/EDUCATION_ASSESSMENTS_SCAN_20260923.md`.

— Claude

## 2026-09-23 — Weekly-note scope and presentation structure (Codex)

Author direction: this_week.tex should present the differing-slopes theory
route first and simulations afterward, for a quick weekly presentation.
Keep it as a working note in this file; migration to another document
requires author approval. No material was moved into the final paper,
prelim, or slides.

Restructured the note into model/target, estimation, the iid slope-score
route, the hard-trimmed CLT roadmap, and simulations. Retained the stacked
alpha/b/beta2 fit, scalar/vector weighted-density equivalence, Riesz
derivation, three shared-OLS loadings, intercept cancellation, and explicit
remaining proof obligations. Removed the lengthy taxi audit, repeated
estimation descriptions, high-dimensional detour and closing repetition.

Source check: the existing distributional battery's _design and
estimate_threshold functions fit the correctly specified linear alpha/b
model by OLS and use known distribution functions. This is not a spline
outcome simulation. The note now says this explicitly, corrects the error
SD to 0.5, and preserves every existing table entry. Full split-specific
nonparametric validation remains a follow-up, not a claim of this batch.

## 2026-09-23 — Author clarification: nonparametric alpha, g and p_X (Codex)

The author elects to assume continuous fares and does not want discreteness
to drive this theory discussion. The intended estimator uses a separate OLS
fold for gamma, a stacked outcome regression with spline alpha and b plus
linear X and DX coefficients, and nonparametric distribution estimation.
Keep g=f_T for the baseline and add p_X(s)=E[X|T=s]f_T(s) for differing
slopes. Both are functions of a scalar index; an augmented weighted-density
projection with responses (1,X) estimates both. Scalar p_beta=beta2'p_X is
an exactly equivalent computational reduction for the slope direction.

Updated this_week.tex with the complete integrated-tail utility and its
derivative, the explicit stacked outcome design, and the comparison with
taxi_differing_slopes.py. The taxi script does use cubic splines for alpha
and b (per _basis_params/_eval_basis); its current direct observed-Q
objective still differs from the proposed spline-tail optimizer. No taxi
code or simulation results were changed. Regularization is outside the
requested comparison. This clarification does not establish independence
of X and eta in the data; the marginal-tail theory continues to maintain it.

Verified algebraic simplification: with one shared OLS estimate, the
untrimmed slope-block loadings satisfy
C_U+C_rho=-E[(Xtilde-E Xtilde)(beta2'X) f_eta'(phi-T)].
Their intercept coordinate is zero, as is the intercept column of A_beta.
Thus the total slope-block OLS intercept correction cancels. The slope
coordinates and generated-outcome correction remain. This statement is
not a claim that hard-trim boundary corrections disappear.

The fixed-threshold score expansion is explicitly conditional on outcome
linearization, spline derivative/moment rates and remainder control.
The full hard-trimmed threshold CLT still requires joint outcome,
distribution and endpoint expansions plus local uniformity and argmax
assembly. These are marked as unfinished rather than certified by a
successful compilation. Verification: latexmk build passes, no undefined
references or horizontal overflow; rendered changed pages inspected.

## 2026-09-23 - Feasible differing-slopes and bootstrap diagnostics completed (Codex)

Ran the Slurm arrays `61761710` and `61761711` from commit `ea2ee50`.  The
full-pipeline array re-estimated the first-stage index, hard-trim endpoints,
and running-variable tail in every sample, with the full `D X` block in every
reported fit (oracle, Gaussian-tail OLS, spline-tail OLS, and ridge).  It used
100 replications at `n={800,1600,3200,6400}` for the baseline, quadratic
misspecification, weak-curvature, boundary-optimum, and clustered-error DGPs.

For the regular baseline, generated-index and moving-trim OLS remain centered
as `n` grows: at `n=6400`, bias is `0.0008` for Gaussian-tail OLS and
`-0.0009` for spline-tail OLS, with RMSE `0.0267` and `0.0229`, respectively.
The spline and Gaussian tails give similar behavior.  Ridge stabilization has
the expected finite-sample bias (`0.065` at `n=800`, `0.021` at `n=6400`), so
it should be treated as an application regularizer rather than a theorem-level
estimator without a bias correction.

The negative controls behave as intended.  Under a quadratic treated effect
omitted from the linear `D X` block, the estimator converges to a pseudo-target
about `0.09` below the true optimum.  Weak curvature produces much larger
uncertainty (RMSE about `0.10` at `n=6400`, versus about `0.02--0.03` in the
regular baseline).  Boundary optima select the policy bound essentially always,
so interior CLT intervals are not appropriate.  Clustered treated shocks raise
the variance substantially; the point estimates remain centered but iid
variance calculations are not valid.

The feasible full-re-estimation bootstrap (`50` outer samples and `199`
bootstrap draws per sample at `n={800,1600,3200}`) had no failed resamples and
no boundary selections in the regular baseline.  Percentile coverage was
`0.96, 0.98, 0.98` for Gaussian-tail OLS and `0.96, 0.98, 0.98` for spline-tail
OLS.  Bootstrap-to-Monte-Carlo SD ratios were `1.14, 1.19, 1.04` (Gaussian)
and `1.24, 1.25, 1.09` (spline).  The ridge version covered only
`0.88, 0.94, 0.94`, reflecting its finite-sample bias rather than bootstrap
failure.

The new harness is `experiments/scripts/differing_slopes_feasible_bootstrap.py`;
the corresponding Slurm wrappers are `differing_slopes_full_pipeline_array.sbatch`
and `differing_slopes_feasible_bootstrap_array.sbatch`.  These are full-sample
re-estimation diagnostics, not a proof of the fully decoupled bootstrap theorem.

-- Codex

## 2026-09-23 - Does the theory cover the taxi differing-slopes fit? No (Claude)

Checked `experiments/scripts/taxi_differing_slopes.py` end to end (adapter,
basis helpers, fit) against `manuscript/this_week.tex` and the DS draft.
Reproduced: n=541,021, window [0.24, 5.46], alpha-only phi-hat $0.00,
differing-slopes phi-hat $12.51 (CV lambda 0.1, the smallest grid value).

1. **The score is discrete.** The adapter keeps standard-meter fares
   2.5+0.4k in [$5, $25]: Q takes 50 values on a $0.40 lattice. The empirical
   utility is constant between lattice points, so phi-hat identifies a cell,
   not a point: $12.51 means "treat fares >= $12.90" (cell (12.5, 12.9]).
   A continuous-Q theory (the root-n CLT, or the cube-root concern for the own-Q
   step objective) does not describe this estimator; the relevant framework is
   selection among finitely many candidate cutoffs.
2. **X independent of eta fails in the analysis sample.** The sample is selected
   on Q in [5, 25], and fare is nonlinear in distance. Mean of eta-hat by distance
   decile ranges from -0.32 to +0.35, its sd from 0.96 to 1.70, and
   P(eta-hat in window) from 0.21 to 0.46. Lattice Q with continuous T also rules
   out exact independence. The theorem-facing plug-in (G-bar(phi-eta), p_beta)
   relies on this factorization. The own-Q objective does not need it for its
   target, only the conditional-mean model E(Y|D,X,eta).
3. **Implementation versus theory.** Full sample, no fold split. Ridge on the spline and
   DX blocks. The design [1, X, DX, Phi, D Phi] is rank deficient (rank 36 of 37)
   because the clamped B-spline basis sums to one, so only the ridge identifies
   the intercept. The theorem-facing design drops it. Unpenalized (intercept
   dropped) beta2_distance is 0.372 versus 0.360, and the cutoff moves one cell,
   to "treat >= $13.30". The marginal fare $12.90 has a near-zero fitted effect
   (mean contribution -2.9e-5), so the cell choice depends on regularization.
   Also c=0 is unstated, and eta is clipped at the 0.5/99.5 percentiles.
4. The DS draft sentence that the taxi fit reuses "the boundary central limit
   machinery of the main theorem" is not supported for this estimator.

Also this session: second theory pass on `this_week.tex` (manuscript
462d169). The A_beta formula was verified numerically: on the battery design
with linear a, b, the beta2 shift per unit gamma perturbation is (0, 0.9 I),
matching (0, a1 I).

-- Claude

## 2026-09-23 - Notation X / X-tilde; this_week.tex theory audit and fixes (Claude)

**Decision (author).** \(X\) denotes the covariates and \(\tilde X=(1,X^\top)^\top\)
the intercept-augmented regressor for the regression of \(Q\):
\(T=\gamma^\top\tilde X\), \(\hat\eta=Q-\hat\gamma^\top\tilde X\),
\(\varphi_\gamma=\Sigma_{\tilde X}^{-1}\tilde X\eta\) with
\(\Sigma_{\tilde X}=\E(\tilde X\tilde X^\top)\). The former \(X^\circ\) is now \(X\).
Applied to every manuscript file except the frozen `prelim/prelim.tex` and
`prelim/slides.tex` (author's choice) and `confirmation.tex` (no-intercept
experiments). Python code already follows this convention (`X` covariates, `Xd`
with ones); historical log entries that use `X^circ` are left as written.

**Verified findings in `this_week.tex` (now fixed, manuscript commit 06df776).**
1. The density fit described as a least-squares regression of \(X\) on
   \(N_L(\hat T)\) estimates \(\E(X\mid T)\), not \(p_X=\E(X\mid T)f_T\). The Riesz
   term \(\zeta_\rho=(\beta_2^\top X)f_\eta(\phi-T)-\theta_\phi\) is the influence
   function of the Lebesgue-Gram orthogonal-series projection
   \(\hat\omega=G_L^{-1}n^{-1}\sum N_L(\hat T_j)V_j\) (as in `differing_slopes.tex`).
   Its bias is \(\int(r_L-r)(p_\beta-\Pi_Lp_\beta)\), a product of approximation
   errors, and a fixed-interval basis needs \(\mathrm{supp}(T)\subset\mathcal T\).
2. \(\hat\beta_2\) is estimated with the generated regressor \(\hat\eta^o\), so it carries a
   first-stage loading \(A_\beta\) (DX-rows of \(\E(\tilde V\tilde V^\top)^{-1}
   \E[\tilde V\{b'(\eta)+Da'(\eta)\}\tilde X^\top]\)), generally nonzero because \(D\)
   depends on \(X\). With one OLS fold the first-stage term is
   \((C_U+C_\rho+C_o)^\top\varphi_\gamma\), \(C_o=A_\beta^\top B_\phi\). This matches the
   \(A_{\rm out,DS}\) loading in `differing_slopes.tex`.
3. By \(X\perp\eta\), \(C_U=\E(\tilde X)\,\partial_\phi\theta_\phi\), and since the first
   row of \(\Sigma_{\tilde X}\) is \(\E(\tilde X)^\top\),
   \(C_U^\top\varphi_\gamma=\partial_\phi\theta_\phi\,\eta\).
4. \(C_\rho=\int r'h=-\int h'r=-\E[\tilde X(\beta_2^\top X)f_\eta'(\phi-T)]\) for the
   positive target (sign verified).
5. The slope variance is not additive with the baseline variance (shared
   evaluation, outcome and first-stage folds).

**Deferred by the author (task board): taxi and simulation sections.**
- The taxi objective \(n^{-1}\sum\hat W_i1\{Q_i\ge\phi\}\) uses \(c=0\) (unstated) and is a
  step process, plausibly cube-root (see the Codex entry below), so no root-\(n\)
  CI or naive bootstrap applies.
- Battery (`outputs/differing_slopes_distribution_20260923`): all 36 table cells
  match the JSON outputs. The theoretical \(n\,\mathrm{AVar}\) (delta method on
  the known-law criterion, OLS sandwich) is 3.98 for the baseline, not 4.43--4.47.
  Values for the other scenarios: \(X\)-\(t_5\) 3.64, \(X\)-skewed 3.45,
  mixture 5.17, \(\eta\)-\(t_5\) 3.77, \(\eta\)-skewed 3.76, error \(t_5\) 4.01,
  error skewed 3.98, heteroskedastic 5.54. A 2,000-replication rerun of the
  baseline gives 4.04 (MC s.e. 0.13) at \(n=4800\) and 3.91 (0.20) at \(n=19200\).
  With 250 replications \(n\,\mathrm{Var}\) has about 9% MC error, and all
  scenarios share one seed stream, so their fluctuations co-move. The note also
  misstates \(\varepsilon\) as standard normal; it is \(N(0,0.5^2)\).

Verification: all edited manuscript files compile as before (see CHANGELOG).

-- Claude

## 2026-09-23 - THEORY CLAIM: pairwise empirical tails (Codex)

Working out the author's requested pairwise empirical-tail extension on branch
theory/pairwise-tail in the manuscript worktree. Deliverable: a new root-level
`pairwise_theory.tex`, with an oracle/local-process proof and a precisely
conditional fitted-nuisance CLT. This avoids `this_week.tex`, currently claimed
by the distributional simulation task. No prelim or application-code changes.

-- Codex

## 2026-09-23 - Pairwise empirical-tail theory worked out (Codex)

The new note manuscript/pairwise_theory.tex derives the pairwise criterion
\(m_A^{-1}m_B^{-1}\sum_{i,j}I(\eta_i)\{a(\eta_i)-c+X_j^{\circ\top}\beta\}
1\{\eta_i+\gamma^\top X_j>\phi\}\). Under \(X\perp\eta\), its population
criterion is the original utility. The two-sample Hoeffding decomposition has
two iid projection terms and a canonical product-empirical remainder. A
localized Euclidean-class condition with \(\|k_{\phi+h}-k_\phi\|_2^2=O(|h|)\)
gives \(o_p(N^{-1})\) for that remainder near the optimizer; Sherman (1994,
Corollary 8) and Sherman (1993, Theorem 3) are the verified primary pointers.
The oracle argmax is therefore root-\(N\) normal under scalar curvature, with
variance equal to the two projection variances weighted by fold fractions.

The feasible extension keeps the existing augmented outcome-functional/Riesz
expansion and adds weak loadings for the residual-side generated index, the
index-fold generated index, and the two hard-trim endpoints. The pairwise
implementation directly estimates the scalar and weighted empirical tails, so
it does not fit a vector-valued \(p_X\), but it does not eliminate endpoint or
generated-index obligations. This theorem is not a result for the direct
own-observation taxi criterion, which remains a separate step-process problem.

Verification: pairwise_theory.tex compiles to a seven-page PDF with no
LaTeX errors or undefined references using
latexmk -g -pdf -interaction=nonstopmode -halt-on-error. Rendered pages 1,
2, 3, 4, 5, 6, and 7 were inspected. Remaining warnings are
the Biometrika class's existing vertical-box warnings and one long projection
display; no clipping or overlap was observed.

-- Codex

## 2026-09-23 - Differing-slopes distributional simulation battery completed (Codex)

Ran the Slurm array `61760936` with the full differing-slopes outcome block in
every fit (`D X_1`, `D X_2` included), 250 independent replications at each of
`n={1200,2400,4800}`, for nine scenarios: Gaussian baseline; variance-one
`t_5`, centered exponential, and two-component-mixture laws for `X_1`; the
analogous `\eta` laws; and `t_5`, centered exponential, and heteroskedastic
outcome errors. The battery is implemented in
`experiments/scripts/differing_slopes_distributional_battery.py` and submitted
by `experiments/cluster/differing_slopes_distribution_array.sbatch`.

At `n=4800`, all nine scenarios had absolute bias below `0.005`, zero boundary
selections, and RMSE between `0.027` and `0.037`. The Gaussian baseline had
`n Var(\hat\phi)` about `4.47` across the sample-size grid. The X/eta-law
changes alter the population target (as they should), while the error-law
changes preserve the target; mixtures and heteroskedastic errors have larger
variance constants (about `6.18` and `6.46`, respectively). This supports the
algebra and distribution-specific population calculations for the differing-
slopes estimator.

Scope: this is a conditional diagnostic with the index, trim interval, and
survival/weighted-tail functions supplied at their population values. It does
not validate the estimated-density, generated-index, moving-boundary, or full
bootstrap theorem. The note and task board record this limitation.

-- Codex

## 2026-09-23 - Clarify empirical threshold objectives and rate claims (Codex)

Correction to the preceding conversation: differing slopes do not by themselves
cause cube-root asymptotics. The frozen prelim estimator (prelim/prelim.tex,
lines 422--463) uses a spline density integrated to a survival function, and
then averages alpha(eta) times that survival function. The older empirical-CDF
implementation in experiments/methods/perfrdd.py, function _utility_curve,
averages the survival of all fitted T values at each residual: algebraically
this is an average over all index--residual pairs. In contrast,
taxi_differing_slopes.py, lines 77--82, uses each observation's own Q directly,
both with and without the interaction block.

For the oracle direct objective P_n[m(X,eta) 1{Q>phi}], positive conditional
second moment of m at the optimum gives local fluctuation of order
sqrt(|h|/n) against quadratic drift h^2. This suggests cube-root threshold
asymptotics for either model, subject to the usual process conditions and
negligible local nuisance error. A step-function objective alone does not
imply cube-root behavior; degeneracy at the threshold can change the rate.

A possible alternative for differing slopes is the pairwise criterion
average_{i,j} [a(eta_i)+beta_2' X_j^circ] 1{eta_i+T_j>phi} on separate folds.
Under X independent of eta, it has the desired population target. Its two
conditional projections are smooth under density regularity; a root-n argmax
proof may therefore use a Hoeffding decomposition and uniform local bounds on
the degenerate remainder without explicitly fitting p_X. This is a proposed
route, not an established extension of our theorem. Sherman (1993), The
Limiting Distribution of the Maximum Rank Correlation Estimator, provides a
verified example of root-n normality for a discontinuous pairwise criterion:
https://www.its.caltech.edu/~sherman/han.pdf (Sections 2--4). It does not by
itself establish the result with our fitted outcome, estimated index, and trim.

-- Codex

## 2026-09-23 - Scalar slope CLT note and taxi density audit (Codex)

Added manuscript note commit 11bfe6f (with changelog commit a90ab9e) in
manuscript/this_week.tex.  The note writes the differing-slopes score
-E[beta_2' p_X(phi-eta)] as a centered evaluation-fold term and a
weighted-density-fold Riesz term, then adds the linear beta_2 contribution
and the generated-residual/generated-index loadings.  It also records the
target-specific scalar alternative p_beta(t)=beta_2' p_X(t) for medium or
high-dimensional covariates.

The taxi audit is source-grounded in
experiments/scripts/taxi_differing_slopes.py and
experiments/scripts/taxi_differing_slopes_rank.py: the current diagnostic
fits the differing-slopes outcome surface with full-sample OLS for gamma
and CV-selected ridge for the spline/interaction blocks, then maximizes a
direct empirical utility.  It does not estimate p_X, H_X, or the
theorem-facing density-fold Riesz term.  It also uses raw X with an
intercept rather than the centered X-circ convention; this is a
reparameterization but should be made explicit.  The note therefore treats
the current taxi result as a differing-slopes diagnostic, not a
theorem-aligned weighted-density implementation.

Verification: the new note compiles with
latexmk -g -pdf -interaction=nonstopmode -halt-on-error this_week.tex from
manuscript/, producing a seven-page PDF without LaTeX errors, undefined
references, or overfull horizontal boxes.  The class emits its existing
vertical-box warnings.

-- Codex

## 2026-09-23 - High-replication bootstrap validation completed (Codex)

The follow-up array `61757691` ran 250 outer replications and 499 iid
bootstrap draws for each of `n={1200,2400,4800,9600}` under the same Gaussian
DGP, with all four cells completing and zero failed resamples. Full-sample
percentile coverage is `0.948, 0.960, 0.948, 0.952`; the corresponding
bootstrap-to-Monte-Carlo SD ratios are `1.108, 1.124, 1.013, 1.074`. The
coverage Monte Carlo standard errors are `0.014, 0.012, 0.014, 0.014`, so the
full-sample results are consistent with 95% coverage across the grid. For the
fixed decoupled estimator, bias is `-0.189, 0.058, 0.045, -0.008` and the
boundary rate falls from `0.36` to `0.00`; its bootstrap/Monte-Carlo SD ratios
are `0.964, 1.065, 1.070, 1.098`. The rotated estimator has bias `-0.354,
-0.043, 0.008, -0.006`, boundary rates `0.452, 0.136, 0.012, 0.00`, and SD
ratios `1.009, 1.203, 1.190, 1.035`; coverage reaches `0.968` and `0.964` at
`n=4800` and `9600`. These higher-replication results materially strengthen
the finite-sample validation: the full-sample bootstrap is well calibrated,
and both split variants approach interior, root-n behavior as the boundary
rate vanishes. They still do not validate the full generated-index/moving-
boundary bootstrap theorem or non-Gaussian/spline-density robustness.

-- Codex

## 2026-09-22 - Differing-slopes appendix prose cleanup (Codex)

Removed internal drafting boxes, proof-status language, and implementation handoff
phrasing from Appendices 2--3 of `manuscript/differing_slopes.tex`. The mathematical
content was preserved. The appendix now states the differing-slopes substitutions and
remaining vector-density remainder condition as technical prose. The active draft still
contains unresolved TODO boxes in the main text, and the archived file
`manuscript/storage/differing_slopes_theory.tex` was not changed.

Verification: `latexmk -pdf -interaction=nonstopmode -halt-on-error
differing_slopes.tex` completed successfully and produced the 37-page PDF.

-- Codex

## 2026-09-22 - Larger active-estimator bootstrap grid completed (Codex)

The restartable Slurm array `61751637` completed all four cells under
`experiments/runs/active_bootstrap_array_20260922_gaussian/`, with 50 outer
samples and 199 iid bootstrap draws per outer sample and no bootstrap failures.
For the full-sample estimator, percentile coverage is `0.96, 0.96, 0.94,
0.90` at `n={1200,2400,4800,9600}`; bootstrap-to-Monte-Carlo SD ratios are
`1.11, 1.16, 0.87, 0.98`. The fixed eight-block estimator's point bias moves
from `-0.309` to `0.018`, its point RMSE from `1.046` to `0.219`, and its
bootstrap/Monte-Carlo SD ratio reaches `1.00` at `n=9600`; its lower-sample
boundary rate is `0.40, 0.10, 0.00, 0.00`. The rotated estimator's bias moves
from `-0.541` to `-0.026`, with SD ratios `0.98, 0.95, 1.10, 0.90` and
boundary rates `0.40, 0.12, 0.02, 0.00`. Coverage uncertainty is material with
50 outer samples: the binomial Monte Carlo standard error is about `0.028` at
coverage `0.96`, `0.034` at `0.94`, and `0.042` at `0.90`. Thus the larger
grid supports sensible bootstrap scale calibration and declining bias, while
the eight-block variants need larger samples before their threshold distribution
is stable; the `n=9600` coverage values are exploratory rather than a theorem
check. A future validation run should increase outer replications for the
largest cells before making a sharp coverage claim.

-- Codex

## 2026-09-22 - Slurm execution hardening for bootstrap grid (Codex)

The first monolithic bootstrap launch (job `61751112`) was stopped after 14 of
200 outer samples when an audit showed numerical-library thread
oversubscription relative to its eight-CPU allocation and no independent
restart boundary by sample size. Its partial CSV and Slurm logs remain under
`experiments/runs/active_bootstrap_20260922_gaussian/`. The replacement array
job `61751637` uses one cell for each `n` in `{1200,2400,4800,9600}`, caps
concurrency at two cells, binds one BLAS/OpenMP thread to each Python worker,
records per-cell provenance, and resumes from completed CSV rows after a
retry. This is an execution/reproducibility change only; no estimator or DGP
was changed.

-- Codex

## 2026-09-22 - Active-estimator bootstrap harness (Codex)

Added `experiments/scripts/hard_trim_bootstrap_active.py` and the reproducible
Great Lakes batch wrapper `experiments/cluster/active_bootstrap.sbatch`. The
bootstrap diagnostic now uses the three active implementations—fixed
`decoupled_8block`, role-rotated `rotated_8block`, and `full_sample`—rather than
the retired legacy fold labels. Bootstrap draws preserve the eight-block role
assignment within each outer sample; re-randomizing roles would add split
randomization to the conditional bootstrap and is reserved for a sensitivity
run. The output reports point bias/RMSE, empirical and bootstrap standard
deviations, `sqrt(n)`-scaled bias, `n`-scaled empirical variance, percentile
coverage, boundary rates, and failed-resample counts. Smoke runs at `n=600`,
`2400`, and `9600` completed without bootstrap failures; small samples showed
boundary instability for the decoupled variants, while the `n=9600` smoke run
was centered much more closely. The script explicitly remains a finite-sample
diagnostic, not a bootstrap validity theorem.

-- Codex

## 2026-09-22 - Keep differing-slopes draft outside prelim (Codex)

Per author direction, the active differing-slopes manuscript was moved from
`manuscript/prelim/differing_slopes.tex` to
`manuscript/differing_slopes.tex` in manuscript commit `9df8014`. The frozen
prelim package remains in `manuscript/prelim/` and was not edited. The moved
draft now uses `prelim/` as the TeX support-file input path, `figures/` for
figures, and the canonical root `references.bib`; a clean root compilation
produces the same 37-page PDF with no LaTeX errors, undefined citations, or
undefined references.

-- Codex

## 2026-09-21 - Terminology audit of the frozen prelim (Codex)

Audited `manuscript/prelim/prelim.tex` for the terminology changes around the
estimated score index. The document now uses **generated-residual error** for
the identity `hat eta - eta = -X'(hat gamma-gamma)`, and uses **generated-index
loading** for the derivative coefficients `A_U`, `A_alpha`, and `A_g`; the
corresponding products `A' varphi_gamma` are the first-stage influence
contributions. This distinction is technically coherent: endpoint quantiles are
generated-index objects (`hat T`), while outcome/evaluation moving-set terms are
generated-residual objects (`hat eta`). The only stale occurrence in the frozen
prelim is the commented keyword `Generated regressor` (line 65); no rendered
prose uses that term. The phrase **generated-index correction** does not yet
appear; if adopted later, it should name the full first-stage contribution, not
the loading alone. The differing-slopes draft inherits the stale keyword and
additional scaffold terminology; that is a post-prelim cleanup item.

-- Codex

## 2026-09-21 - Terminology audit of the prelim presentation (Codex)

Audited `manuscript/prelim/slides.tex` against the frozen prelim terminology.
The deck uses `threshold` for the selected policy and `cutoff` for the deployed
rule in most places, but the M-estimation slide still says “candidate cutoff set
`Phi`”; this is a minor wording/notation carry-over, not a mathematical error.
The theory slides use “Errors in variables” (without the standard hyphens) in
the taxonomy block, while another slide uses “error-in-variables components”.
The displayed products `A' varphi_gamma` are full first-stage
influence/contribution terms; only the coefficients `A` are technically the
generated-index loadings. The deck also contains the unhyphenated backup label
`Generated index`. No manuscript or deck source was changed in this audit.

-- Codex

## 2026-09-21 - Presentation terminology is canonical for proof revisions (Codex)

Author decision: as the author revises the proof, the terminology used in
`manuscript/prelim/slides.tex` is the preferred vocabulary for the manuscript
and proof. After each author change, Codex should reread the affected passage
and propagate the presentation terms through nearby definitions, labels, and
explanations. A different technical term should be retained only when it names
a mathematically distinct object (for example, a loading coefficient versus
the full first-stage influence contribution); such exceptions should be called
out explicitly rather than silently changing the author's terminology.

-- Codex

## 2026-09-21 - Full prelim copied before differing-slopes adaptation (Codex)

Per author correction, `manuscript/prelim/differing_slopes.tex` was first replaced
with an exact copy of the full `prelim/prelim.tex` and then adapted in place. The
37-page draft preserves the prelim's complete empirical and appendix structure while
changing the primary model to
`W=a(eta)+X^circ beta_2+R_W`, adding the augmented `D X^circ` outcome block, a
vector weighted-tail density nuisance, ten deliberately decoupled roles, and
DS-specific TODOs for the remaining Riesz, moving-set, identification, and
theorem-aligned empirical checks. `prelim/prelim.tex` was not modified. The copied
and adapted manuscript compiled successfully from `manuscript/prelim` with
`latexmk -g -pdf -interaction=nonstopmode -halt-on-error differing_slopes.tex`;
the build produced no LaTeX errors or undefined-reference warnings.

-- Codex

## 2026-09-21 - Final-paper direction: differing slopes (Codex)

The author has changed the final-paper direction: differing slopes are now the
primary model, rather than the original same-slope model used for the frozen
prelim. A new standalone working draft, `manuscript/prelim/differing_slopes.tex`,
mirrors the prelim structure and replaces the outcome model, utility target,
estimator, and proof scaffold with the finite-dimensional `D X^circ` extension.
The draft compiles to a 12-page PDF and marks unresolved assumptions,
augmented-Riesz and vector-density calculations, foldwise signs/constants,
bootstrap choices, and theorem-aligned taxi/simulation re-estimation with
visible TODO boxes. The archived scaffold in
`manuscript/storage/differing_slopes_theory.tex` remains supporting material;
the new draft is the forward working document for the final paper.

-- Codex

## 2026-09-21 - Prelim package complete (Codex)

The author has finished the prelims. The written manuscript and the 50-minute
presentation are now frozen as the completed prelim deliverables: the
same-slope hard-trimmed theorem under (X\perp(W,\eta)), its deliberately
decoupled-split proof, the simulation evidence, and the taxi application with
the differing-slopes motivation. The formal differing-slopes theory remains
archived in `manuscript/storage/differing_slopes_theory.tex` and is not part of
the prelim.

`manuscript/TODO.md` now labels the remaining spline-rate, outer-support,
bootstrap, taxi, dataset, and personal theory-review items as post-prelim
follow-ups. Other sessions should not reopen or materially revise the prelim
proof or deck unless the author explicitly requests it.

-- Codex

## 2026-09-17 - DECK CLAIM released: percentile-only bootstrap slide (Codex)

The t5-only bootstrap validation frame now reports only the percentile
confidence interval: $n=1,200$ coverage 0.935 and $n=2,400$ coverage 0.950,
with the normal-bootstrap column and discussion removed. The frame retains the
mean bootstrap spread, 79,600 successful fits, and the finite-sample caveat.
Source commit `bce2f93` and changelog commit `0fb3e9e` are on Overleaf `master`;
the deck rebuilds to 42 pages and the simplified frame was visually inspected.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - percentile-only bootstrap slide; branch slides/bootstrap-percentile-only-20260917 (open)

Simplify the t5-only bootstrap validation frame by removing the normal-bootstrap
column and discussion. Retain the percentile coverage, bootstrap spread, fit
count, and finite-sample caveat. Scope excludes slide 23, all `% FINAL` frames,
the finalized Future Work section, and concurrent author edits.

-- Codex

## 2026-09-17 - DECK CLAIM released: Codex - presentation TODOs 51-56

Completed the empirical-application TODOs through 56. The deck now explains
CMT versus VTS, shows only the left taxi menu-crossover panel, fits the
residual-only utility text, introduces and moves the differing-slopes material
earlier, and adds the verified differing-slopes simulation comparison. The
simulation slide uses the documented baseline results: full target
`phi*=-0.120`, alpha-only pseudo-target `-0.339`, near-zero augmented bias,
and declining alpha-only coverage. The marker's "CTS" was resolved as CMT
versus VTS because those are the vendor labels used throughout the project;
CMT remains validation only. Manuscript source commit `9bd7cb4` and changelog
commit `5f4a0c2` are on Overleaf `master`; the 43-page deck compiled with no
overfull or underfull-box warnings, and slides 31--37 were visually inspected.
TODOs after 56 remain untouched (none were present in the synced source).

-- Codex

## 2026-09-17 - DECK CLAIM released: early $e_\phi(\eta)$ definition and language cleanup (Codex)

The first overlap slide now defines
$e_\phi(\eta)=\mathbb P\{D(\phi)=1\mid\eta\}$ before its first use. Active
simulation and empirical-application slide copy was revised to use direct
technical language in place of metaphor-heavy or vague phrasing; the
mathematical content and author-finalized Future Work section were unchanged.
The deck rebuilds to 43 pages and the page-10 and revised application frames
were visually inspected. Manuscript source commit `45b9607` and changelog
commit `0f16f42` are on Overleaf `master`; the existing author TODO and
taxi/application overflow warnings remain.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - complete presentation TODOs 51-56; empirical application and differing-slopes simulation; branch slides/todos51-56-20260917 (open)

Scope is limited to presentation TODOs 51--56: clarify the CMT/VTS roles, crop
the taxi crossover figure, repair the utility-slide fit, add and reorder the
differing-slopes explanation, and add the verified differing-slopes simulation
comparison. TODOs after 56 remain out of scope. The simulation values will be
drawn from `experiments/datasets/simulations/DIFFERING_SLOPES_ASSUMPTION_SWAP_20260912.md`.

-- Codex

## 2026-09-17 - DECK CLAIM released: t5-only bootstrap validation slide (Codex)

The bootstrap validation frame now shows only the two Student-t(5) running-
variable cells (`n=1,200` and `2,400`) and omits the running-law column, as
requested. It reports 79,600 successful bootstrap fits, percentile coverage
0.935/0.950, and normal coverage 0.980/0.975. Source commit `58c2c0d` and
changelog commit `a206540` are on Overleaf `master`; the deck rebuilds to 43
pages and the updated frame was visually inspected. The complete four-cell
validation remains in the tracked simulation note.

-- Codex

## 2026-09-17 - Slide TODO 50 completed (Codex)

Removed the rendered TODO 50 marker and the latent-type $\eta$ row from the
taxi data-to-model mapping slide. The frame now shows the observed outcome,
covariates, score, treatment, candidate cutoff, and deployed cutoff only.
The 43-page deck builds and slide 31 was visually inspected; manuscript commit
`f121d69` is on Overleaf master. TODO 51 remains untouched.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - show t5-only bootstrap validation; branch slides/bootstrap-coverage-t5-only-20260917 (open)

Narrow the existing simulation-section bootstrap frame to the requested t5
running-variable validation cells, remove the running-law column, and update the
fit count and wording. The underlying four-cell validation note remains the
complete provenance record. Scope excludes slide 23, all `% FINAL` frames, the
finalized Future Work section, and the concurrent notation-language cleanup.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - define early treatment probabilities and remove LLM-style slide wording; branch slides/notation-language-cleanup (open)

Add the definition of $e_\phi(\eta)=\mathbb P\{D(\phi)=1\mid\eta\}$
before its first use on the overlap slide, and revise audience-facing slide
copy that uses vague or presentation-generator language such as “anchor.”
Preserve the mathematical content, finalized sections, and concurrent edits.

-- Codex

## 2026-09-17 - DECK CLAIM released: correct bootstrap running-variable law label (Codex)

The validation frame now labels the t5 and mixture rows as **running-variable
laws**, matching the DGP: $X_1$ is standardized t5 or a two-component normal
mixture, while the outcome disturbance is Gaussian $N(0,0.5^2)$ in every cell.
Numerical coverage results are unchanged. Source commit `8ae1d1f` and
changelog commit `38c7565` are on Overleaf `master`; the 43-page deck compiles
and the updated frame was visually inspected. The latest synced author TODO
notation was kept buildable with an equivalent `\perp` token in commit
`b44bbb7`; the author TODO and taxi-slide overflow warnings remain otherwise
unchanged.

-- Codex

## 2026-09-17 - Terminology correction: t5/mixture are running-variable laws (Codex)

The `t5` and `mixture` labels in the bootstrap study refer to the distribution
of the first running-variable covariate $X_1$, after standardization to mean zero
and variance one. The outcome disturbance is Gaussian $N(0,0.5^2)$ in every
cell. The bootstrap notes and presentation table are being corrected from
“error law” to “running-variable law”; numerical results are unchanged.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - correct bootstrap law label; branch slides/bootstrap-coverage-law-label-20260917 (open)

Correct the validation frame's table heading and wording from “error law” to
“running-variable law,” without changing the reported coverage values or any
protected frames.

-- Codex

## 2026-09-17 - DECK CLAIM released: bootstrap coverage validation update (Codex)

The simulation-section bootstrap frame now reports the 200-outer-replication,
199-draw-per-cell validation. Percentile coverage is 0.935--0.950 across the
four t5/mixture and `n={1,200,2,400}` cells; normal coverage is 0.925--0.980.
The frame records 159,200 successful fits, zero failures, the approximately
1.5 percentage-point coverage Monte Carlo error, and the fact that this is
implementation validation rather than a bootstrap validity theorem. Source
commit `477809f` and changelog commit `faf83d3` are on Overleaf `master`; the
43-page deck rebuilds and the updated frame was visually inspected.

-- Codex

## 2026-09-17 - Bootstrap coverage validation completed (Codex)

The follow-up full-sample hard-trim re-estimation bootstrap used 200 outer
replications and 199 bootstrap draws in each of the four pilot cells
(`n={1,200,2,400}`; t5 and mixture errors), yielding 159,200 successful fits
and zero failures. Percentile coverage was 0.935, 0.950, 0.945, and 0.945,
respectively; normal coverage was 0.980, 0.975, 0.925, and 0.925. The
percentile estimates' binomial 95% Wilson intervals all include 0.95. Normal
intervals are conservative under t5 and mildly under-cover for the mixture,
so percentile intervals remain the preferred diagnostic. This is finite-sample
validation, not a bootstrap validity theorem; clustering and the fully
decoupled influence-function construction remain untested here. Results are
documented in `experiments/datasets/simulations/BOOTSTRAP_COVERAGE_VALIDATION_20260917.md`.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - update bootstrap coverage slide with validation; branch slides/bootstrap-coverage-validation-20260917 (open)

Update the existing simulation-section bootstrap frame to replace the 20-replication
pilot numbers with the completed 200-replication validation. Scope excludes slide
23, all `% FINAL` frames, and the finalized Future Work section. The frame will
show percentile versus normal coverage and retain the finite-sample caveat.

-- Codex

## 2026-09-17 - DECK CLAIM released: bootstrap coverage diagnostic slide (Codex)

The simulation section of the presentation now includes the full-sample,
hard-trim re-estimation bootstrap pilot: 20 outer samples and 99 bootstrap
draws for each of `n={1,200,2,400}` under t5 and mixture errors. Percentile
coverage is 0.95 in all four cells; normal coverage is 0.95 for the mixture and
1.00 for t5, with zero bootstrap fit failures. The frame labels this as a
finite-sample diagnostic rather than a bootstrap validity theorem and notes the
roughly ±5 percentage-point Monte Carlo error from 20 outer samples. Source
commit `99e4784` and changelog commit `979a85f` are on Overleaf `master`; the
deck compiles to 43 pages and the new frame was visually inspected.

-- Codex

## 2026-09-17 - DECK CLAIM released: Codex - slide 23 reference layout

Slide 23 now follows the supplied figure's 2x2 source-box layout, with a taller
optimization fold and clean right-angle connectors routed through the gaps. The
manuscript update is on Overleaf master in commit `0a30a6f` (source change
`1d1c442`); the 42-page deck builds with only the known taxi-slide warning.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - bootstrap coverage diagnostic slide; branch slides/bootstrap-coverage-20260917 (open)

Adding one simulation-results frame after the existing verification frame in
`manuscript/prelim/slides.tex`. The scope excludes slide 23, all `% FINAL`
frames, and the finalized Future Work section. The frame reports the completed
full-sample re-estimation bootstrap pilot as a finite-sample diagnostic, with
coverage and failure counts traced to `experiments/runs/bootstrap_coverage_small_20260917`.

-- Codex

## 2026-09-17 - DECK CLAIM released: TODOs 40/41 done (Codex)

Slides 20--21 now have non-overlapping roles: slide 20 summarizes the
single-nuisance changes and three errors-in-variables loadings, while slide 21
defines the fixed least-squares projection weight for $\psi_\alpha$ and works
through one direct-score chain-rule loading. The rendered TODO markers and the
unescaped build-breaking text were removed. Manuscript commits 848279b and
5f63acd are on master; the 42-page deck builds and slides 20--21 were
rendered together. The existing taxi-slide overfull-vbox warning remains.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - restyle slide 23 decoupled-split figure to match reference layout; branch slides/slide23-reference-layout (open)

## 2026-09-17 - DECK CLAIM released: Codex - slide 23 decoupled-split arrows cleaned

The four slide-23 connectors were rerouted around the source boxes with separate
entry heights on the optimization fold. The verified manuscript update is on
Overleaf master in commit `e627d1a`; the build note documents the pre-existing
slide-21 TODO 40 syntax error.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - TODOs 40/41; make slides 20--21 non-overlapping; branch slides/todo40-41 (open)

Follow-up to the earlier TODO 31 slide-20 restructure. The current shared deck
keeps slide 20 as the overview of single-nuisance changes and the three
error-in-variables loadings, while slide 21 contains the worked examples.
This pass will define the fixed least-squares projection weight for
$\psi_\alpha$ and add one simple chain-rule example for an index loading on
slide 21, without duplicating the slide-20 overview.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - clean slide 23 decoupled-split arrows; branch slides/slide23-arrow-cleanup (open)

## 2026-09-17 - Slide 23 diagram labels updated (Codex)

Renamed the two upper blocks in the decoupled-split diagram: “Index fits” is now
“Estimates for $\eta$” and “Response surfaces” is now “Utility function terms”.
No statistical content or computation changed; the updated 42-page deck builds and
slide 23 was visually inspected.

-- Codex

## 2026-09-17 - Slide 21 reworded (Claude)

Per author: dropped the term 'Riesz representer' (r_alpha is now described as a fixed projection weight from the linearized least-squares fit), and rewrote the generated-index example as intuition/strategy (sensitivity via differentiation x OLS index error = one extra mean-zero score, root-n preserved) instead of the A_U equation. Trimmed to fit; deck builds.

-- Claude

## 2026-09-17 - DECK CLAIM released: TODO 31 done (Claude)

Slide 20 reframed into two slides per author: (1) three kinds of first-order terms -- single-nuisance changes g*(alpha_hat-alpha) [->psi_alpha] and (alpha-c)*(g_hat-g) [->psi_g], generated-index loadings from gamma_hat-gamma, plus the mean-zero direct score; (2) one worked psi_alpha (Riesz) and one worked gamma-loading. Landed via cherry-pick onto the author's rapidly-advancing master; deck builds.

-- Claude

## 2026-09-17 - DECK CLAIM: Claude - TODO 31 restructure slide 20 (single-nuisance-change terms + generated-index loadings, with worked psi_alpha and gamma-loading examples); branch slides/todo31 (open)

-- Claude

## 2026-09-17 - DECK CLAIM released: slide 20 influence function done (Claude)

Replaced the trimmed 'Leading CLT pieces' with two slides: a derivation of the direct score psi_0=-F*, the outcome score psi_alpha via the Riesz representer, and a generated-index loading from eta-hat; then the full Psi_{main,eps,i} (three scores + three index loadings), density score, and linearization with boundary folds. Rebased onto the author's concurrent Overleaf edits (their Mukherjee/Wibisono/setup rewrites preserved). Deck builds (41pp).

-- Claude

## 2026-09-17 - DECK CLAIM: Claude - slide 20 influence function (show full Psi_i and derive a few scores); also comment a build-breaking bracketed [TODO 11]; branch slides/slide20-psi (open)

-- Claude

## 2026-09-17 - Follow-up: concise Wibisono comparison slide (Codex)

At the author's request, shortened Introduction slide 6 to focus on the
substantive distinction from Mukherjee: a constant effect $\tau$ versus a
heterogeneous effect $\alpha(\eta)$. The repeated score/error decomposition was
removed, and one line now summarizes the method: estimate the score index,
recover $\hat\eta$, and estimate $\alpha(\eta)$ nonparametrically in the
outcome model. Manuscript source commit `d1aa902` and changelog commit
`ac7570b` are on `master`; the 40-page deck compiles and slide 6 was rendered
and checked.

-- Codex

## 2026-09-17 - DECK CLAIM released: Future Work section finalization markers (Codex)

Added `% SECTION FINAL: Future Work` and matching `% END SECTION FINAL: Future Work`
markers to the complete Future Work section. `COLLABORATION.md` now defines marked
section ranges as author-only edit zones, and `tools/slide_status.py` reports them.
The marker-only manuscript update was pushed to `master` in commits `308c33c` and
`d853854`; the deck still compiles to 40 pages.

-- Codex

## 2026-09-17 - DECK CLAIM: Codex - Future Work section finalization markers; branch slides/final-section-marker (open)

The author requested section-level finalization so agents treat the entire
Future Work section of `manuscript/prelim/slides.tex` as read-only after the
author marks it final. This change adds explicit section markers, documents
their read-only semantics in `COLLABORATION.md`, and makes the advisory slide
status helper report protected sections.

-- Codex

## 2026-09-17 - DECK CLAIM released: S1 refinement done (Claude)

Confirmed from Mukherjee, Banerjee & Ritov (Bernoulli 2026, eq 1.3, Assumption 1.1, contribution 3): their outcome is a partial linear model with a nonparametric baseline b(eta)=E(nu|eta) that they estimate; the constant effect tau (their alpha_0) is root-n and independent of b's tuning. Slides now show b(eta) on the Mukherjee frame and state the distinction is constant tau vs nonparametric alpha(eta). Deck builds (40pp).

-- Claude

## 2026-09-17 - DECK CLAIM: Claude - S1 refinement, make Mukherjee's nonparametric b(eta) explicit (branch slides/s1-mukherjee-b) (open)

-- Claude

## 2026-09-17 - Follow-up: S6-S9 title QA on current manuscript master (Codex)

After the newer S6--S9 release landed on manuscript `master`, visual review
found that the long leading-CLT and threshold-CLT titles clipped in the
AnnArbor title bar. Manuscript commit `4da4be2` shortens those titles and names
Future Work IV by its substantive idea, ``Performative stability.'' The current
deck builds to 40 pages; slides 20--22 and 39--40 were rendered and checked.
The existing taxi-slide overfull-vbox warning remains outside this QA change.

-- Codex

## 2026-09-17 - DECK CLAIM released: S6-S9 slide TODOs (Codex)

Reviewed the S6--S9 presentation changes now on manuscript `master`: the
decoupled-split schematic uses independent nuisance/index/endpoint folds plus
one optimization fold; the theory opener targets the first-stage Taylor and
approximately-iid score expansion; Future Work IV explains performative
stability without notation; and Future Work V introduces jointly choosing a
multiple-score index and threshold. The CLT source display was reflowed after
render review so it fits the slide, and the long decoupled-split title was
shortened. The task board and changelog were released in manuscript commits
`0037d32` and `823607f`. The deck builds to 40 pages; rendered slides 20, 21,
39, and 40 were checked. An existing taxi-slide overfull-vbox warning remains
outside this TODO scope.

-- Codex

## 2026-09-17 - DECK CLAIM released: slide TODOs S1-S5 done (Claude)

S1 Mukherjee/Wibisono distinction block; S2/S3 relabeled setup equations plus a maintained-assumptions list (X indep (W,eta); E(eps|eta,X,D)=0; alpha,b stable across cutoffs); S4 threshold optimized over the real line (candidate set dropped); S5 new alpha/b-identification-needs-overlap slide with an e_{phi0}(eta) schematic. Deck builds (39pp), no % FINAL frame touched. S6-S9 remain with Codex.

-- Claude

## 2026-09-17 - DECK CLAIM: Claude - slide TODOs S1-S5 (Mukherjee/Wibisono distinction, setup assumptions + six-equation cleanup, drop candidate set, alpha-overlap justification slide); branch slides/todos-s1-s5 (open)

-- Claude

## 2026-09-17 - DECK CLAIM: Codex - S6-S9 slide TODOs; branch slides/todos-6-9-2026-09-17 (open)

The author assigned Codex the round-2 slide TODOs S6--S9: rework the
decoupled-split figure, restructure the theory opener, make Future Work IV
idea-focused, and add the multiple-scores Future Work slide. Claude retains
S1--S5. This claim narrows the earlier broad TODO claim by scope; Codex will
avoid editing S1--S5 and any `% FINAL` frames.

-- Codex

## 2026-09-17 - DECK CLAIM released: taxi figures enlarged + bracketed TODOs commented (build fixed); other author TODOs remain as `% [TODO]` comments (Claude)

-- Claude

## 2026-09-17 - DECK CLAIM: Claude - author slide TODOs (figures, Mukherjee/Wibisono, assumptions, decoupled-split figure, theory opener, Future Work IV, multiple-scores); branch slides/todos-2026-09-17 (open)

-- Claude

## 2026-09-17 - Reinstate slide-deck collaboration as markers + branch + log-claim (Claude)

At the author's request, replaced the removed tool-based slide lock with a
lightweight, tooling-free protocol in `COLLABORATION.md` (section "Slide-deck
collaboration"). The removed lock failed because of three-commit ceremony, a
frame parser that crashed on inline `[TODO:]` text, a pre-commit hook that
became non-executable and silently stopped enforcing, and a `SLIDE_LOCKS.json`
registry that itself collided with the Overleaf auto-sync on `master`.

New rule: author markers in the deck (`% FINAL` = read-only, `% TODO:` =
edit-request), agent deck claims announced here in `RESEARCH_LOG.md` before
non-trivial edits, work on a short-lived `slides/<topic>` branch, fast-forward
`master` after compiling, then log a one-line release. The author's live
Overleaf edits always win; fetch before merging and never force-push. Added an
advisory-only helper `code/tools/slide_status.py` (plain text scan; lists
`% FINAL` frames and recent `DECK CLAIM` lines; never blocks a commit). Verified
the helper runs against the current deck (0 final frames, 0 open claims).

-- Claude

## 2026-09-17 - Add a simple author-final marker for slides (Codex)

The collaboration protocol now defines `% FINAL` as the author-facing marker
for an individual Beamer frame. The author places it on the same line as the
frame's `\begin{frame}{...}` declaration. Agents must treat that frame as
read-only and may edit it only after the author explicitly names the slide and
requests a revision. The marker can be located with `rg -n '% FINAL'
manuscript/prelim/slides.tex`.

-- Codex

## 2026-09-17 - Persist author presentation guidance in slide source (Codex)

Added a comment block above the section-opener macro in
`manuscript/prelim/slides.tex` so future agents can see the standing
presentation guidance without relying on chat history. It documents the
prelim-aligned section map, title-only centered dividers, standard Beamer
Computer Modern fonts, disabled backup slides, preservation of intervening
author edits, and the currently suspended locking workflow. The comments do
not change the rendered deck.

Verification: the deck still builds to 40 pages and `prelim.tex` still builds
to 34 pages. The manuscript source commit is `9680ff4`.

-- Codex

## 2026-09-17 - Author slide TODO follow-up implemented (Codex)

The preliminary talk was updated in `manuscript/prelim/slides.tex` while
preserving the author's latest synced edits. The Mukherjee and Wibisono setups
are now separate Introduction slides; Setup and target states the maintained
independence and hard-trim convention; Estimation is presented as four steps;
and Large-sample theory gives a high-level assumption/proof map plus the
six-block influence-function assembly. The Simulation studies opener now states
the centering, root-$n$ rate, and variance-tracking checks before the DGP. The
Empirical application opener maps the taxi variables to $(Y,X,Q,phi,D,eta)$
and flags residual-only misspecification when effects depend on fare-level
covariates. Discussion is renamed Future Work, the old summary is disabled, and
the author's four Future Work I--IV slides are restored. Obsolete detailed theory
and backup frames remain in source but are disabled in the default build.

Verification: `latexmk -g -pdf -interaction=nonstopmode -halt-on-error slides.tex`
produced a 40-page deck without overfull boxes in the edited frames; key edited
slides were rendered and visually inspected. The full `prelim.tex` build exited
successfully and produced a 34-page PDF with the repository's existing layout
warnings. Manuscript source and changelog commits are `ae8a29b` and `d0bde5a`.

-- Codex

## 2026-09-17 - Section dividers reduced to centered titles (Codex)

The seven prelim-matched section opener frames now contain only the section
title, centered vertically and horizontally on a plain slide. The explanatory
purpose text has been removed at the author's request. The manuscript source
commit is `2d37561`; a stale inline TODO reintroduced by an automatic remote
sync was removed in follow-up commit `1e28c56`.

-- Codex

## 2026-09-17 - OLS first-stage subsection shortened (Codex)

The `Estimating gamma` subsection in `manuscript/prelim/prelim.tex` now states the
ordinary least-squares fit once, keeps the source and held-out block convention,
the exact generated-residual identity, its first-order role, and the same-fold
boundary fits. The repeated OLS explanation and the triangular-equation literature
detour were removed. The revised manuscript still compiles to 34 pages with no
undefined references. The source change is in manuscript commit `ee4ed62`, with
the final pushed changelog update in `af95efb`.

-- Codex

## 2026-09-17 - Align slide sections with prelim and suspend slide locks (Codex)

The deck now follows the seven main sections of the prelim: Introduction,
Setup and target, Estimation, Large-sample theory, Numerical study, Empirical
application, and Discussion. The temporary slide-lock registry, workflow
document, lock script, and pre-commit hook were removed at the author's request;
slide edits use ordinary Git coordination for now. The manuscript source change
is commit `c01969f`.

-- Codex

## 2026-09-17 - Section 4 compressed for the prelim presentation (Codex)

Section 4 of `manuscript/prelim/prelim.tex` now keeps the theorem-facing material:
the R1--R7 regularity conditions, identification and feasible-consistency lemmas,
the spline dimension rate, and the hard-trimmed threshold CLT. The detailed Riesz
representer construction, moving-set expansion, generated-index loadings, weighted
score, and fold-weighted variance remain in Appendices 2--3. The score and loading
labels were moved with their definitions. Two stale references to the deleted
`tab:taxi-2x2` table were removed.

The previous source compiled to 35 pages, with Section 4 spanning pages 8--10.
The revised source compiles to 34 pages, with Section 4 spanning pages 8--9 and
Section 5 beginning on page 10. The revised build reports no undefined references.
The source change is in manuscript commit `5275154`, with the changelog hash
recorded in follow-up commit `4097bd0`.

-- Codex

## 2026-09-17 - Section roadmaps replaced by concise openers (Codex)

The slide deck no longer inserts table-of-contents roadmap frames at section
boundaries.  Each of the five main sections now begins with one simple opener
showing the section title and its one-sentence purpose.  The default deck
remains 41 pages because the backup appendix is still disabled.  The slide
source and lock-release commits are `b05ef90` and `52d4bd5` in the manuscript
repository.

-- Codex

## 2026-09-17 - Backup slides hidden from default talk build (Codex)

The eight backup frames remain in `manuscript/prelim/slides.tex` for committee
use, but the appendix is now inside a default-off `\\iffalse` block.  The
default slide PDF therefore contains 41 pages rather than 49; changing the
guard to `\\iftrue` restores the backup appendix.  The source and lock-release
commits are `2135b31` and `97cc381` in the manuscript repository.

-- Codex

## 2026-09-17 - Slide lock hook portability (Codex)

The Overleaf bridge normalized `.githooks/pre-commit` from executable to
non-executable during a sync, which caused Git to skip the lock check.  The
manuscript workflow now restores the executable bit and documents
`chmod +x .githooks/pre-commit` as part of one-time setup.  This keeps the
frame checkout guard active in local clones while the tracked lock registry
remains the shared source of truth.

The repair is tracked in manuscript commit `49caf37`.

-- Codex

## 2026-09-17 - Slide data audit completed (Codex)

The slide claims were reconciled with the current `manuscript/prelim/prelim.tex`
and its numerical-study appendices.  The baseline simulation now uses 200
replications at n=(10,000,20,000,40,000,80,000), with pooled n x MSE values
461.08, 43.20, and 42.58 and RMSE slopes -0.50, -0.53, and -0.52 for fixed
eight-block, role-rotated, and full-sample implementations.  The taxi slide now
reports the current residual-only estimate near 5.3 dollars, its 5.0--7.6 dollar
sensitivity range, and the 4.6 dollar partial-linear screen.  The outdated
placebo/old coverage claims were removed.

The backup section now mirrors the Appendix~4 replication grid and completed
(t_5)/standardized-mixture diagnostics, including the explicit statement that
bootstrap coverage is not reported.  The backup model notation uses
X^circ consistently with the current outcome regression.  The slide source
commit is `c5aaf3c`; the deck lock was released in `c71105e`.

-- Codex

## 2026-09-17 - Slide TODO pass completed (Codex)

The inline `[TODO:]` notes in `manuscript/prelim/slides.tex` have been
resolved.  The deck now has section roadmaps, an explicit score/outcome
decomposition, a three-step estimator overview, related work in backup, and a
five-frame future-work sequence covering differing slopes, flexible nuisances,
online updates, performative fixed points, and multiple scores.  The redundant
OLS-only spline and closing frames were removed.  The Beamer source uses the
standard Computer Modern Beamer fonts with the built-in AnnArbor theme and
builds to a 49-page PDF without overfull boxes in the edited frames.

The source commit is `dc82b78`; the deck lock was released in `0dcd441`.  The
manuscript build completed to 35 pages, but still reports the pre-existing
undefined `tab:taxi-2x2` references and layout warnings in `prelim.tex`; these
are outside the slide change and are left for a separate manuscript cleanup.

-- Codex

## 2026-09-17 - Prelim camera-readiness review (Codex)

The current `manuscript/prelim/prelim.tex` builds to a 37-page PDF with no
undefined citations or references.  The main narrative is coherent and the
same-slope hard-trimmed theorem is clearly separated from the deferred
differing-slopes theory.  The document is not yet camera ready as a written
prelim because Appendices 1--3 still expose internal proof-status language,
including the Appendix 2 status table, the Appendix 3 ``Remaining author
checks'' section, and instructions about moving material to a later
supplement.  The theorem remains conditional on the high-level R5 criterion
and derivative condition, with the primitive spline-rate and outer-support
eligibility checks still listed as author work.  The taxi discussion also
needs softer causal wording and the application treatment rule should be made
consistent with the strict inequality used in the theory.  No manuscript
source was edited in this review.

-- Codex

## 2026-09-17 - Prelim prose cleanup completed (Codex)

The intended manuscript title was already present in `prelim.tex`, and the
running head now uses that title with the three-author byline.  The abstract
was rewritten for grammar, spelling, and a clearer description of the
covariate-index and latent-residual setup.  The remaining author-check and
incomplete-bootstrap language was removed from the theorem discussion and
diagnostics appendices, and the proof dependency label was recast as a proof
roadmap.  The stray comma in the differing-slopes future-work sentence is
gone.  The manuscript rebuilds to 35 pages with no undefined citation or
reference warnings.  Manuscript changes were pushed to Overleaf as commit
`ef82c7a`.

-- Codex

## 2026-09-17 - Final oral-prelim readiness check (Codex)

The current remote manuscript compiles to a 35-page PDF.  The compiled PDF
has no visible TODOs, author-check labels, placeholders, undefined citations,
or undefined references.  The title and three-author byline are present, the
abstract reads cleanly, and sampled theorem, appendix, and bibliography pages
have no visible clipping.  The document is ready to send for the oral prelim,
with two substantive caveats for the author to be prepared to state aloud:
the main theorem remains conditional on the high-level R5 feasible
criterion/derivative condition, and the theory uses (D=1\{Q>\phi_0\}) while
the taxi application uses (D=1\{Q\geq\phi_0\}).  These are not new edits in
this check.

-- Codex

## 2026-09-16 - Final proof-consistency pass (Codex)

The final source audit found and repaired three notation/implementation
consistency issues: the main stacked outcome-regression map now uses the
nonconstant covariates `X^circ` (the baseline spline already spans the
constant), endpoint variance notation defines `sigma_p^2 = Var(xi_p)`
before giving its `X independent eta` simplification, and the
feasible-consistency maximum-index bound is correctly written as
`O_p(n_gammaU^(-1/2) n_U^(1/4)) = o_p(1)`.  The forced LaTeX build still
succeeds after these edits;
focused hard-trim tests pass 8/8.  No additional proof-breaking issue was found
under the stated R1--R7 conditions and the explicit high-level R5 support/rate
checks.

-- Codex

## 2026-09-16 - Final notation audit (Codex)

The trim fraction \(\epsilon\) and the outcome error \(\varepsilon=\nu-b(\eta)\)
were being denoted by the same symbol in the manuscript.  The model,
regularity conditions, influence function, and simulation equations now use
\(\epsilon\) only for trimming and \(\varepsilon\) only for the outcome error.
The Gaussian baseline simulation also now labels its three nonconstant
covariates \(X^\circ\), matching the intercept convention in the theorem.

-- Codex

## 2026-09-16 - Final theorem audit and support/design-matrix repairs (Codex)

The final manuscript audit found and repaired a rank-deficiency ambiguity:
the score projection uses the full covariate vector \(X=(1,X^\circ)\), while
the outcome spline regression uses only \(X^\circ\), because the baseline
spline span contains a constant.  The theorem now also states nonsingularity
of \(E(XX^\top)\), a finite \(R_W^2\) moment, continuity of
\(m_\eta(t)=E[X\mid\eta=t]\) at the hard boundaries, and the local
generated-index quantile conditions needed by the uniform Bahadur lemma.
Criterion-level uniform consistency is explicit in (R5).

One remaining support issue is now stated rather than hidden: selecting the
outcome sample by the generated residual \(\hat\eta^\alpha\in\mathcal J\)
can create an outer-boundary term even when \(\mathcal J\) is fixed.  The
theorem therefore conditions on a negligible outer-support eligibility
remainder (a strict-margin compact support for \(\eta\) is sufficient); absent
that condition, the outer-boundary contribution must be added to the
influence function.  The density basis is explicitly extended by zero outside
its fixed buffered interval.  The feasible-consistency proof now uses the
correct \(O_p(n_U^{1/4})\) maximum bound under a fourth moment.

Verification: the focused hard-trim Gaussian and cross-fit suites pass (8
tests); fold counts sum to \(n\) for \(n=1000,1001,10000\); the forced
prelim LaTeX build exits 0 with 37 pages, no undefined citations/references,
51 unique labels, and no tracked conflict markers.  The only author-level
theory work left is a primitive spline criterion/derivative-rate check and
documentation of the computational bootstrap; no bootstrap validity theorem
or reused-sample variance claim is made.

-- Codex

## 2026-09-16 - Presentation font decision (Codex)

The presentation remains a standard Beamer deck using the built-in `AnnArbor`
theme and Michigan color overrides. The Fira Sans and Fira Mono packages were
removed from `manuscript/prelim/slides.tex` at the author's request. A successful
`latexmk -pdf -interaction=nonstopmode -halt-on-error slides.tex` build and
`pdffonts` audit confirm that the compiled deck now uses Beamer's standard
Computer Modern sans-serif (`CMSS`) and Computer Modern math fonts.

-- Codex

## 2026-09-16 - Frame-level slide collaboration workflow (Codex)

Added a tracked collaboration protocol for `manuscript/prelim/slides.tex`.
`prelim/tools/slide_lock.py` assigns stable frame ids, records one active owner,
session, branch, and source range in `prelim/SLIDE_LOCKS.json`, and provides
`checkout`, `list`, `release`, and `validate-staged` commands. The manuscript
repository's pre-commit hook rejects unclaimed slide edits, wrong-branch edits,
and commits that change more than one frame; deck-wide changes use the `deck`
lock. Claims must be committed and pushed before content edits, and releases
follow the content commit. Existing uncommitted slide work remains untouched;
the checkout command refuses to claim while `slides.tex` is dirty.

-- Codex

## 2026-09-16 - Equal-split tradeoff documented (Codex)

Added the equal-allocation tradeoff to the prelim.  For the finalized eight
blocks, n_q=n/8+O(1), n_min is asymptotically proportional to n, and every
lambda_{n,q} tends to sqrt(8).  Thus equal splitting changes variance
constants but introduces no asymptotic centering or rate loss under the
decoupled theorem.  The finite-sample cost is noisier spline, endpoint, and
evaluation components because each uses about one eighth of the data.  An
ideal unequal allocation would depend on unknown block influence variances,
so equal blocks remain the transparent default; rotations and full-sample
reuse are diagnostics outside the theorem.

-- Codex

## 2026-09-16 - Final fold-size and proof consistency check (Codex)

Rechecked the eight-block implementation at n=1000, 1001, and 10000; every
role block is nonempty and the counts sum exactly to n.  The focused Gaussian
baseline and regularization suites pass (8 tests).  Rebuilt
`manuscript/prelim/prelim.tex` after replacing the generic Bahadur prose's
sample-size symbol by m (with m=n_b in the boundary lemma); the 37-page PDF
has no undefined citations, duplicate labels, or LaTeX errors.

The fold audit found no proof-breaking mismatch.  Evaluation, outcome,
density, and three first-stage source sums use their own n_q^{-1/2} CLTs;
the lower and upper endpoint terms use n_l^{-1/2} and n_u^{-1/2}, with the
same-fold quantile/OLS covariance retained in sigma_p^2.  The theorem's
lambda_{n,q}=sqrt(n/n_q) factors produce the total-sample normalization, and
the n_min spline/moving-set rates are equivalent to the old total-n rates
under positive limiting fold fractions.  The only substantive theorem-level
check still conditional is a primitive unregularized-spline derivative rate
that implies (R5); bootstrap validity remains intentionally unclaimed.

-- Codex

## 2026-09-16 - Explicit fold-size bookkeeping and theory audit (Codex)

Verified the finalized theorem-facing implementation in
`experiments/scripts/hard_trim_gaussian_baseline.py`: the eight roles
`gamma_alpha`, `gamma_g`, `gamma_U`, `boundary_l`,
`boundary_u`, `outcome`, `density`, and `utility` form a partition,
each with limiting fraction 1/8 up to integer rounding.  A direct check at
n=1000,1001,10000 confirmed that the fold counts sum to n; the existing
cross-fit regularization tests also pass (`python3 -m unittest -q
experiments.tests.test_hard_trim_crossfit_regularization`, 3 tests).

Updated the prelim theorem and estimator map to use explicit block sizes
n_q, total n=sum_q n_q, n_min, limiting fractions pi_q, and
lambda_{n,q}=sqrt(n/n_q).  The CLT now displays six non-boundary block sums
and the two endpoint terms with their own fold weights; the variance is the
corresponding sum of pi_q^{-1}-weighted variances.  Spline and moving-set
remainders use n_min, which is equivalent to the old total-n rates when all
fractions are positive.  The remaining
theorem-level author check is the primitive spline derivative-rate implication
behind high-level condition (R5); the fold bookkeeping, Riesz, Bahadur,
moving-set, density-boundary, and decoupled CLT assembly checks are explicit.

-- Codex

## 2026-09-16 - Reference audit cleanup (Codex)

Applied the reference-audit fixes in `manuscript/references.bib` and the synced
`manuscript/prelim/references.bib`. The Mao and Montreuil entries now use their
published PMLR records, with the Montreuil key renamed to `Montreuil2025`. The
Mukherjee, Banerjee, and Ritov entry now uses the published 2026 key. Canonical
RD references, the IV identification reference,
Newey's series-rate result, and the van der Vaart asymptotic-statistics record
were added or cited. DOI and URL fields were added to the relevant existing
entries. The generic Lousdal IV primer and unused Abadie-Imbens entry were
removed. The sharp RD wording now refers to the treatment effect at the cutoff,
and the prelim builds with zero BibTeX warnings.

The changes preserve the existing uncommitted manuscript edits and generated
files from the other proof and numerical-study work.

-- Codex

## 2026-09-15 — Final hard-trim proof and clarity audit (Codex)

Audited `manuscript/prelim/prelim.tex` from the supported target through the
decoupled-split CLT. The moving-set signs are internally consistent: the lower
endpoint contributes `-b_l Delta_l` to the criterion and `+b_l Delta_l` to the
score, while the upper endpoint has the opposite pair; both quantile-to-endpoint
maps carry a minus sign. The fixed-buffer density repair uses zero outer trace
and weak integration by parts, retaining the two internal hard-trim masses. The
generated-index Bahadur argument is stated under the continuous and mixed-
covariate conditions, with the OLS moment `E||X eta||^2 < infinity` now listed
explicitly.

Clarity edits make the identification lower bound and feasible derivative
consistency explicit as (R1) and (R5), distinguish the continuous theorem
optimizer from a grid implementation, and point readers to Appendices 2--3 for
the complete proof modules. The proof map and author-check list now reflect the
closed endpoint-sign audit. The manuscript compiles to a 35-page PDF with no
undefined or duplicate references and no overfull boxes from the edited proof
displays. Remaining author-level checks are the primitive derivative-rate proof,
fold-size bookkeeping, bootstrap protocol, and the empirical convention that
the taxi data use a weak threshold (`Q >= 15`) while the theorem is written with
`Q > phi` and continuous projection density.

— Codex

## 2026-09-15 — Fixed-support operational clarification (Codex)

Follow-up to the proof audit: the manuscript now states explicitly that the
outer nuisance interval `J` is chosen a priori from a scientific or measurement
range, while the theorem assumes that this fixed interval covers the unknown
population trim window `[l_0,u_0]` with positive margin. This resolves the
operational ambiguity without making `J` data-dependent. The corresponding
manuscript commits are `7367de6` and `aa1f931`.

— Codex

## 2026-09-15 — Full manuscript reference audit (Codex)

Audited `../manuscript/references.bib` and every citation in
`../manuscript/prelim/prelim.tex`. The bibliography has 35 entries, 26 of
which are cited in the prelim. All 26 cited keys resolve to real works through
publisher, DOI, Crossref, NBER, or arXiv records. There are no missing citation
keys, duplicate keys, or BibTeX warnings in the current prelim build.

Metadata follow-ups are needed before finalizing the bibliography. The
`Montreuil2024` entry does not match the source: the verified title is
“A Two-Stage Learning-to-Defer Approach for Multi-Task Learning,” with authors
Yannis Montreuil, Shu Heng Yeo, Axel Carlier, Lai Xing Ng, and Wei Tsang Ooi.
The work has a 2024 arXiv version and a later ICML/PMLR version. The
`MaoMohri2024` entry also has a published ICML/PMLR version that should replace
the arXiv-only metadata if the online paper cites the final version. The
`Dep2021` key is a legacy key for the journal version published in 2026.

Recommended additions or citation placements for the prelim are Hahn, Todd,
and van der Klaauw (2001) for canonical RD identification; Imbens and Lemieux
(2008) and Cattaneo and Titiunik (2022) for RD reviews; Porter (2003), already
in the bibliography, for the RD convergence-rate statement; van der Vaart
(1998), already in the bibliography, for argmax/M-estimation; and Newey (1997)
for series and regression-spline rates. The IV contrast currently uses the
generic Lousdal (2018) primer, which is real but weaker than a core econometric
reference such as Imbens and Angrist (1994). No bibliography edits were made in
this audit.

— Codex

## 2026-09-15 — Theorem-matched role-rotation rerun (Codex)

Completed the requested comparison with the spline-density DGP: one fixed
eight-block theorem split (`decoupled_8block`), all eight cyclic role rotations
(`rotated_8block`), and full-sample reuse (`full_sample`). Each rotated
criterion is validly decoupled within its assignment; the eight criteria are
pooled before maximizing. In 200 replications at
`n={10000,20000,40000,80000}`, pooled `n*MSE` is `461.1` (fixed), `43.2`
(rotated), and `42.6` (full sample). RMSE slopes are `-0.499`, `-0.526`, and
`-0.518`. The rotation closes the fixed split's one-eighth-sample precision
loss in this DGP; all boundary rates are zero. It remains an empirical
implementation diagnostic because the current CLT covers only one fixed split
and not the cross-rotation covariance. Outputs are in
`experiments/runs/fixed_rotated_full_spline_20260915/`; code and tests were
committed and pushed in `0d7f112`.

— Codex

---

## 2026-09-15 — Fixed, role-rotated, and full-sample hard-trim comparison (Codex)

The hard-trim simulation driver now distinguishes the three requested sample-use
cases. `decoupled_8block` is one fixed theorem-aligned assignment of eight
disjoint role blocks and five first-stage fits. `rotated_8block` cycles all eight
role assignments over the same physical partition and maximizes the pooled
held-out criterion. `full_sample` fits every nuisance and evaluates on all rows;
positive ridge values remain optional `full_ridge_*` diagnostics. The former
five-fold row was an ordinary cross-fitting benchmark, not the theorem-facing
estimator, and is no longer part of the primary comparison.

Added `make_role_rotated_folds` and tests that verify every rotation is a valid
partition and that each physical block serves each role once. The short Gaussian
check (`n={1000,2000,4000}`, 30 replications) had RMSE at `n=4000` of `0.322`
(fixed), `0.443` (rotated), and `0.106` (full sample); at `n=8000` in a 20-
replication follow-up, the values were `0.198`, `0.074`, and `0.072`. This
finite-sample crossover is consistent with noisy one-eighth nuisance fits at
small `n`; the role rotation becomes competitive once block sizes are adequate.
The rotated average has cross-rotation covariance and remains outside the
single-split CLT. Targeted tests and syntax checks pass. Code changes are in the
working tree pending commit/push.

— Codex

---

## 2026-09-15 — Clarify theorem-matched spline nuisance in manuscript (Codex)

Section 5 now explicitly states that the outcome and density nuisances are
fixed-support cubic-spline projections, matching the final simulation command
(`--density spline`).  The clarification was compiled successfully and pushed
to Overleaf in commits `c598d3f` and `f5082e9`.

— Codex

---

## 2026-09-15 — Spline-density theorem-matched rerun and manuscript update (Codex)

Because the stated hard-trim theorem uses a fixed-support spline density
nuisance, reran the eight-block construction with `--density spline` for 200
replications at `n={10000,20000,40000,80000}`.  The decoupled RMSEs are
`0.223, 0.146, 0.102, 0.079`; `n*MSE` is `496, 429, 418, 502`; bias is
`0.030, 0.014, 0.006, 0.002`; and boundary rates are `0.005, 0, 0, 0`.
The log--log RMSE slope is `-0.499` and pooled `n*MSE` is `461.1`.  Ordinary
five-fold and full-sample diagnostic slopes are `-0.521` and `-0.518`.

Section 5 of the prelim now reports this spline-density run, explicitly names
the eight-block/five-first-stage construction, and replaces the earlier
shared-first-stage table.  The manuscript changes were compiled successfully
and pushed to Overleaf in commits `179a51b` and `f2734f6`.  The task-board item
for the theorem-aligned simulation is closed.

— Codex

---

## 2026-09-15 — Theorem-aligned eight-block hard-trim simulation (Codex)

The previous baseline's six role blocks reused one main first-stage projection.
To match the hard-trim CLT, `experiments/scripts/hard_trim_gaussian_baseline.py`
now provides `make_theory_folds`, an eight-block seeded partition with distinct
`gamma_alpha`, `gamma_g`, and `gamma_U` source folds for the outcome, density,
and evaluation blocks, plus independent lower- and upper-boundary blocks and
their associated outcome/density/utility blocks.  The five first-stage fits are
recorded in each replication.  The regularization comparison script now reports
this estimator as `decoupled_8block`; ordinary five-fold cross-fitting and
full-sample reuse remain diagnostics only.

The known-target Gaussian simulation was rerun with 200 replications at
`n={1000,2500,5000,10000}` and 100 replications at
`n={10000,20000,40000,80000}`.  The decoupled RMSEs in the larger run are
`0.232, 0.147, 0.104, 0.075` with `n*MSE` `538, 432, 433, 452`, bias
`0.014, -0.000, 0.010, 0.009`, and boundary rates `0,0,0,0`; the log--log
RMSE slope is `-0.538`.  The smaller run has boundary rates `0.420, 0.095,
0.020, 0.005`, illustrating the finite-sample cost of one-eighth blocks.
The five-fold and full-sample comparison slopes are `-0.505` and `-0.499`.
Outputs are in the ignored `experiments/runs/theorem_decoupled_20260915*`
folders; the durable methods/results note is
`experiments/datasets/simulations/THEOREM_DECOUPLED_HARD_TRIM_20260915.md`.

Verification: targeted baseline/cross-fit tests pass; the full experiment test
suite passes except for three pre-existing registry tests requiring the optional
`pyarrow` parquet engine for the local taxi data.  No taxi files were staged.

— Codex

---

## 2026-09-15 — Defer differing-slopes theory from the prelim (Codex)

Author decision: the formal differing-slopes theory is not part of the prelim.
The former Appendix 4 block has been moved from
`../manuscript/prelim/prelim.tex` to
`../manuscript/storage/differing_slopes_theory.tex`. The storage document is
not included by the prelim and should be treated as archived theory for later
work.

The prelim retains the empirical taxi comparison, differing-slopes simulation
evidence, and simulation implementation details. The remaining paper
appendices were renumbered so the simulation implementation is Appendix 4 and
the additional diagnostics are Appendix 5. Future sessions should not restore
the DS assumptions, lemmas, or CLT to the prelim unless the author explicitly
reopens this decision.

— Codex

---

## 2026-09-15 — Complete remaining Luna differing-slopes proof audits (Codex)

In \`../manuscript/prelim/prelim.tex\`, completed the remaining Appendix 4
Luna TODOs.  DS--B now gives the feasible criterion as an explicit five-term
decomposition (evaluation empirical process, outcome nuisance, scalar survival
nuisance, vector weighted-tail product, and moving trim/generated-residual
term) and invokes the inherited crossing-band bound plus the compact argmax
theorem.  DS--E now restates the moving-set expansion with
\`F_{\phi,\mathrm{DS}}\`, verifies the lower/upper signs from the two boundary
integrals and \`\bar G'=-g\`, restates the local generated-index Bahadur
representation, and distinguishes genuine internal weak point masses from the
removed artificial outer trace.  DS--F/G now display the signed loadings,
retain within-block covariance and inverse fold fractions, and state the
fixed-neighborhood derivative and curvature consistency bounds for both
\`g'\` and \`\rho_X'\`.  The Appendix 4 dependency table, audit status, and
\`manuscript/TODO.md\` proof checklist were updated accordingly.  The result
remains conditional on the stated DS1--DS5 and inherited hard-trim
assumptions, and remains deliberately decoupled; no full-sample-reuse or
ordinary-cross-fitting theorem is introduced.

Verification: \`latexmk -pdf -interaction=nonstopmode -halt-on-error
prelim.tex\` completed successfully in \`../manuscript/prelim\`, producing a
48-page PDF with no unresolved references.  Existing overfull/underfull box
warnings are layout warnings only; pages 35, 40, 42, and 43 were visually
checked.

— Codex

---

## 2026-09-15 — Complete page-36 augmented-outcome proof audit (Codex)

The latest manuscript commit `5181dc5` closes the Appendix 4 DS--C TODO on
page 36. The augmented outcome block now verifies the uniform empirical-Gram
bound along the sieve sequence, including the oracle-sample and generated-index
pieces, transfers the DS2 eigenvalue bound to the sample inverse, carries the
$K^{-3}$ spline approximation through the fixed outcome loading, and displays
the exact outcome-fold factor $\pi_o^{-1/2}$ in the score expansion. It also
states that the finite-dimensional $DX^\circ$ block adds no sieve bias and no
new generated-index derivative.

Verification: the current 47-page `../manuscript/prelim/prelim.pdf` contains
the completed DS--C calculation on page 36 and has no unresolved references or
citations after the standard `latexmk` build. Existing overfull-box warnings
remain layout warnings only.

— Codex

---

## 2026-09-15 — Complete page-37 vector-density proof audit (Codex)

In `../manuscript/prelim/prelim.tex`, completed the DS--D vector weighted-
density proof block. The lemma now gives componentwise cubic-spline
approximation constants, normalized quasi-uniform B-spline level/derivative
envelopes, the bounded coefficient-matrix argument for the vector representer,
and the conditional second- and fourth-moment bounds. The proof explicitly
verifies the triangular-array Lindeberg condition through the Lyapunov ratio
`O(L^2/n_rho)` and retains the positive finite-sieve loading before the
fixed-buffer weak integration-by-parts limit. The DS--D TODO is marked closed.

Verification: `latexmk -pdf -interaction=nonstopmode -halt-on-error
prelim.tex` completed successfully in `../manuscript/prelim`, with no
unresolved references or citations. Existing overfull-box warnings remain
layout warnings only.

— Codex

---

## 2026-09-15 — Complete page-35 proof-expansion TODO (Codex)

In `../manuscript/prelim/prelim.tex`, replaced the Appendix 2 collection
placeholder for `\(\Psi_{{\rm main},\epsilon}\)` with an explicit score
decomposition. The draft now displays the direct score block, outcome and
density residual blocks, the evaluation-fold moving-set loading, the finite-
sieve-to-limit outcome loading, and the fixed-buffer weak integration-by-parts
density loading. It also records the signs of the generated-index terms and
states how designated honest folds and fold-fraction rescaling enter the
notation. The corresponding item in “Remaining small proof obligations” is
marked closed. This closes the page-35 TODO only; the other proof obligations
remain open.

Verification: `latexmk -pdf -interaction=nonstopmode -halt-on-error
prelim.tex` completed successfully in `../manuscript/prelim`, producing a
46-page PDF with no unresolved references or citations. Existing overfull-box
warnings remain layout warnings only.

— Codex

---

## 2026-09-15 — Appendix 1 bridge draft (Codex)

Replaced the Appendix 1 TODO boxes with a first draft that records the latent-index
and outcome-model identification conditions, supported target, estimator map, spline
and fixed-support conventions, rate window, and the distinction between theorem and
numerical ridge fitting. Added a crosswalk separating inherited score-explained
ingredients from the new threshold-selection, hard-trimming, boundary, and foldwise
CLT modules. The detailed expansions remain in Appendices 2--3.

The clean worktree based on manuscript commit `e6a493b` compiled with
`latexmk -pdf -interaction=nonstopmode -halt-on-error prelim.tex` after the final
run, with no unresolved citations or references. The manuscript change was pushed
as commits `9c759e2` and `1567d5d`. The local in-progress manuscript build also
completed successfully.

— Codex

## 2026-09-15 — Prelim presentation structure and slide audit (Codex)

The current Sections 1--4 support a four-act, approximately 50-minute presentation:
question and target, estimator, large-sample theory, and evidence with the
differing-slopes extension. The main talk in `../manuscript/prelim/slides.tex` now
implements that structure. Filled frames use the current theorem, simulation values,
and taxi figures from the manuscript. Backup frames remain explicit placeholders for
committee questions.

The deck follows the checked slide-design guidance that each frame carries one main
message, the title states the message, and technical detail is concentrated in the
spoken explanation and a small number of readable equations. `latexmk -pdf
-interaction=nonstopmode -halt-on-error` completed successfully in a clean worktree,
producing a 32-page deck. Visual inspection covered the theory and empirical frames.

The slide source and changelog were pushed to the Overleaf repository in commits
`ed52453` and `28f50ca`.

— Codex

## 2026-09-15 — Fixed buffered support adopted after Mukherjee check (Codex)

Re-read Mukherjee--Banerjee--Ritov, *Estimation of a score-explained
non-randomized treatment effect in fixed and high dimensions*, including Remark
2.2 on the printed page 8 and the associated supplement. Their main estimator
restricts the latent residual to a fixed compact interval, explicitly noting
that this loses efficiency. They say that a slowly increasing interval could
recover the lost tail observations, but would require a density bounded away
from zero over the growing interval, a known rate for the minimum density,
stronger global conditional-mean derivative bounds, and further tail
bookkeeping; they defer that analysis as methodologically uninformative.

The same conclusion applies here, with one important target-specific check.
Because $\Phi$ and the retained latent-index interval $\mathcal J$ are fixed
and compact, choose the deterministic outer density window with strict margin
so that every relevant argument $\phi-\eta$ lies in its interior. If $T$ is
unbounded, the omitted upper survival tail is then the same constant for every
candidate $\phi$; setting that unknown constant to zero changes utility levels
but not its score, curvature, or maximizer. Thus the fixed buffered window is
not an asymptotic approximation to the policy choice and does not require the
extra tail/density conditions in Remark 2.2. A bound chosen too narrowly would
be different: the omitted tail would become $\phi$-dependent and would change
the target, so the strict buffer condition is part of the theorem.

The manuscript now uses only this fixed-support route. The optional growing-
window subsection and its unresolved rate conditions were removed from the
preliminary document and task board; the historical comparison is retained in
this log for provenance.

— Codex

## 2026-09-15 — Final moving-set and density-boundary audit (Codex)

The remaining theory audit found three assumption/bookkeeping issues and no new
conceptual obstruction. The moving-set lemma now explicitly assumes a uniformly
bounded second derivative of its score integrand \(F_\phi(t)\) on a neighborhood
of the retained residual support; this is what justifies the \(O_p(n^{-1})\)
smooth-factor Taylor remainder (for the current \(F_\phi\), it follows from
local \(C^2\) smoothness of \(\alpha\) and \(g\)). The boundary-integral
remainder was weakened from \(O_p(n^{-1})\) to \(o_p(n^{-1/2})\), which follows
from the stated endpoint-density and conditional-mean continuity and is
sufficient for the root-\(n\) expansion. Finally, the density derivative and
value rates are stated on the fixed interior argument region
\(\mathcal T_{\mathrm{rel}}\Subset\mathcal T\), because zero-trace splines
cannot uniformly approximate a generally nonzero density at the artificial
outer endpoints; quasi-uniform knots are now explicit.

These changes leave the estimator, supported estimand, and rate window
unchanged. The manuscript compiles to 48 pages with no LaTeX errors or
undefined citations/references. The corresponding manuscript edits remain in
the active shared worktree for the concurrent theory pass and were not bundled
into a separate commit.

— Codex

## 2026-09-15 — Introduction framing refinements implemented (Codex)

At the author's direction, the introduction now describes the RDD estimand as a
local average treatment effect at the deployed cutoff and states the local rate as
sqrt(nh). It clarifies that unconfoundedness given X alone is not assumed, describes
the policy as a cutoff applied to the observed score, makes the deployed-versus-
counterfactual distinction explicit, and rewrites the contribution list around the
score decomposition, nonparametric nuisance functions, overlap-supported CLT, and
the appendix-only differing-slopes extension. The model terminology in the setup
paragraph and the overlap sentence were intentionally left unchanged pending the
author's decision on the suggested wording. Manuscript commits are `328b50e` and
`f2a7cbc`; the restored 48-page worktree builds without undefined citations.

— Codex

## 2026-09-15 — Supported-optimum identification tail condition repaired (Codex)

The identification lemma in Appendix 3 previously claimed that strict
log-concavity alone makes the trimmed tilt concentrate at the lower and upper
endpoints as the candidate cutoff tends to minus or plus infinity. That claim
is not valid for all strictly log-concave densities: exponential-tail examples
can leave a nondegenerate limiting tilt. The lemma now adds the steep-tail
condition `lim_{t->-infinity}(log g)'(t)=+infinity` and
`lim_{t->+infinity}(log g)'(t)=-infinity`. This makes the endpoint-ratio
argument valid and preserves the stated primitive sign condition
`alpha(l_0)<c<alpha(u_0)`. An alternative is to assume the finite-Φ signs of
the tilted mean directly; the manuscript retains the more interpretable
steep-tail route.

Manuscript commits `f369f82` and `3ce44d0` were pushed to Overleaf after a
successful 48-page LaTeX build with no errors or undefined references.

— Codex

## 2026-09-15 — Boundary bookkeeping and theorem-scope audit (Codex)

The main hard-trim theory had two small but consequential notation gaps. The
endpoint score loadings are now defined as
`b_l=b_{l_0}(phi_epsilon^*)` and `b_u=b_{u_0}(phi_epsilon^*)`, and
`c_bdry` is defined as the combined limiting fraction of the two equal-size
boundary folds. The feasible-consistency discussion now includes crossing due
to the generated residual, not only crossing due to estimated endpoints.

The audit also records the remaining substantive theory task: the theorem's
`Ψ_main,epsilon` is still a deliberate collection placeholder and must be
expanded into the fold-specific score and generated-index loadings before the
hard-trim CLT is presented as fully self-contained. The omitted tail constant
in the fixed density window is now described correctly as unknown and set to
zero, since it shifts utility by a phi-independent constant only.

Manuscript commits `dabff22` and `28fa525` were pushed to Overleaf after a
successful 48-page LaTeX build with no errors or undefined references.

— Codex

## 2026-09-14 — Riesz representation spaces clarified (Codex)

The hard-trimming Riesz argument in `manuscript/prelim/prelim.tex` now identifies
the represented objects and their spaces explicitly. The outcome block is a
weighted Hilbert space on the fixed outer nuisance region, with inner product
`E[S_J h_1 h_2]` and the intercept removed from the covariate block because the
baseline spline contains a constant. Its Riesz functional is the treatment-effect
component of the negative policy score, not the function `alpha` itself; the
finite-sieve representer is the exact `Q_{Z,K,J}^{-1}` projection and converges in
the selected `L_2` norm. The density block uses Lebesgue `L_2(T)`, with the hard
trim indicator in the representer; its finite-sieve version is the exact `L_2`
projection and only undifferentiated convergence is used for the boundary step.

The manuscript changes are in commits `4721161` (content) and `174274b`
(changelog), pushed to the Overleaf remote. `latexmk -pdf -interaction=nonstopmode
-halt-on-error prelim.tex` succeeds with no LaTeX errors or undefined citations or
references; the 47-page worktree build's Riesz pages were visually inspected.
The deliberate split and hard-supported target are unchanged. Concurrent Claude
edits remain uncommitted in the manuscript worktree and were preserved.

— Codex

## 2026-09-14 — Introduction reference audit implemented (Codex)

Follow-up to the audit above: the two verified metadata corrections and the
overlap-trimming wording correction were applied to both manuscript bibliography
copies and the active introduction. The Mukherjee--Banerjee--Ritov record now cites
the 2026 *Bernoulli* publication; Lousdal now uses article number 1 and its DOI;
the Wibisono typo is corrected; and the Crump comparison now describes target-
population trimming. The published manuscript commits are `645c3a9` and
`c3acfb2`; the latest 47-page worktree build has no undefined citations.

— Codex

## 2026-09-14 — Introduction reference audit (Codex)

Audited all 22 bibliography keys cited in `manuscript/prelim/prelim.tex` §1.1--1.2
against publisher/DOI metadata and, for the three closest papers, the local source
PDFs. The document builds to 46 pages with no undefined citations. The two bibliography
copies (`references.bib` and `prelim/references.bib`) are currently byte-identical.

Two records need correction. `Dep2021` is the 2021 arXiv record for Mukherjee,
Banerjee & Ritov, but the paper is now published in *Bernoulli* 32(4), 2569--2593
(2026), DOI `10.3150/24-BEJ1832`; the entry should become an article record and its
year should be updated. `Lousdal2018` is *Emerging Themes in Epidemiology* 15,
article 1, DOI `10.1186/s12982-018-0069-7`; `pages={1--7}` incorrectly treats the
seven-page PDF length as a journal page range.

Two citation sentences overstate what their sources support. Robinson (1988),
Schick (1986), and Yatchew (1997) support partial-linear/semiparametric estimation,
but not the draft's joint claim that the outcome model “is fit by spline or series
methods”: Robinson uses residualization/nonparametric smoothing and Yatchew uses
differencing. Abadie & Imbens (2016) studies propensity-score matching for ATE/ATET
under unconfoundedness, not an RDD estimator targeting a fixed-cutoff effect; the
relevant connection is that Wibisono et al.'s fixed-cutoff ATT procedure uses residual
matching inspired by that literature. The Crump et al. trimming analogy is valid, but
the contrast should say that their propensity-score rule trims to a precision-oriented
ATE target population; it is not naturally characterized as reweighting the welfare
integrand.

The remaining cited-paper summaries and core bibliographic metadata are supported,
including the Mukherjee/Wibisono score-explained results, Marinescu et al.'s threshold
optimization, Dong & Lewbel's marginal threshold treatment effect, policy learning,
triangular models, and the IV exclusion-restriction comparison. Many otherwise-correct
records omit DOI fields; that is a normalization opportunity, not a substantive error.
No manuscript or bibliography edits were made because this was a review-only request
and the introduction files are actively claimed by Claude.

— Codex

## 2026-09-14 — Mixed-covariate Bahadur extension formalized (Codex)

Appendix 2 now contains a formal mixed-covariate projection lemma rather than
only an extension remark. Write `X=(1,R,Z')'`, allow `Z` to have any fixed-
dimensional distribution, and choose `R` so that `gamma_R != 0`. Conditional
on `Z`, it is sufficient that `R` have a continuous density along the local
projection boundary, dominated by `B(Z)` with
`E[(1+||Z||)B(Z)]<infinity`, and that the induced projection density remain
locally positive. Conditioning on `Z` turns the continuous-covariate integral
into an expectation and yields the same derivative
`partial_a F=-f_{a'X}(q)E[X|a'X=q]`.

The fixed-dimensional halfspace VC argument, local stochastic
equicontinuity, quantile inversion, and same-fold boundary CLT are
distribution-free beyond those population smoothness inputs, so they carry
over without modification. The ordinary result still excludes a projection
with an atom at a trimming quantile. This extension matches applications with
continuous quantities plus indicators or counts; a fully atomic index would
require different quantile asymptotics. Manuscript content commit: `e63177e`.
The 46-page prelim compiled without undefined references, and pages 26--28
were visually inspected.

— Codex

## 2026-09-14 — Bahadur proof recast for continuous covariates (Codex)

At the author's direction, Appendix 2 now states the generated-index quantile
proof first under a regular joint density for all nonconstant covariates. Writing
`X=(1,R,V')'` and choosing a continuous coordinate with `gamma_R != 0`, the
proof expresses the projection cdf as an integral up to the hyperplane boundary
`s(a,q,v)`. Dominated differentiation yields the projection density and the
identity `partial_a F(a,q)=-f_{a'X}(q)E[X|a'X=q]`, which supplies the population
linearization used by the uniform Bahadur argument. The intercept is explicitly
exempt from the continuity condition.

The earlier mixed-covariate result is retained only as an extension remark. If
some remaining coordinates are discrete, conditioning on them reduces the
derivation to the same calculation, provided one coordinate with a nonzero
index coefficient has a continuous dominated conditional density. The VC,
stochastic-equicontinuity, quantile-inversion, and same-fold covariance pieces
are unchanged. Manuscript content commit: `cd41a19`. The 45-page prelim builds
without undefined references and the revised pages 25--27 were visually checked.

— Codex

## 2026-09-14 — Generated-index Bahadur lemma completed (Codex)

Appendix 2, Block D of `manuscript/prelim/prelim.tex` now proves rather than
assumes the local uniform expansion for the empirical quantile of
`a'X`, uniformly over `||a-gamma|| <= C/sqrt(n_b)`. The proof uses four pieces:
(i) a continuously differentiable local projection cdf with positive density,
(ii) the derivative identity `partial_a F = -f E[X | a'X=q]`, (iii) the
VC/Donsker property and local L2 continuity of halfspace indicators, and (iv)
monotone cdf inversion after a uniform root-n localization. A primitive
sufficient condition is also recorded: conditional on the other, possibly
discrete, covariates, one continuously distributed covariate has a continuous
dominated density and a nonzero index coefficient.

The boundary CLT is now written for the actual same-fold OLS construction. Its
influence function is the ordinary quantile score plus
`m_p' H_X^{-1} X eta`; the cross-covariance is zero under `E(eta | X)=0`.
Without that restriction the general variance is `Var(xi_p)` and must retain
the covariance. The signed endpoint influence functions are displayed
separately for the lower and upper boundary folds.

Crossref metadata and the original-source landing pages were checked for the
four citations used: Bahadur (1966), DOI `10.1214/aoms/1177699450`; Kiefer
(1967), DOI `10.1214/aoms/1177698690`; Ghosh (1971), DOI
`10.1214/aoms/1177693063`; and van der Vaart & Wellner (1996), DOI
`10.1007/978-1-4757-2545-2`. The classical papers establish fixed-distribution
quantile representations; they are not cited as proving generated-index
uniformity. Ghosh's Theorem 1 gives the `o_p(n^{-1/2})` remainder needed for a
CLT, while van der Vaart--Wellner Sections 2.5--2.6 supply the empirical-process
and VC machinery. Manuscript content commit: `89aefc5`. The 45-page prelim
compiled with no errors or undefined references,
and the new pages were visually inspected.

— Codex

## 2026-09-14 — Prune completed material from `To discuss.` (Codex)

The top advisor-notes block in `prelim/prelim.tex` now points to the completed
fixed-support and moving-boundary arguments in Appendices 2--3 and retains only
open questions: the generated-index Bahadur lemma, final endpoint-sign
bookkeeping, the optional expanding-support route, feasible consistency/Riesz
convergence, and the substantive trimming choice. The revised prelim compiles
to 43 pages with no LaTeX errors or undefined citations.

— Codex

## 2026-09-14 — Reflow prelim main text around hard-trim theorem (Codex)

The written prelim now removes the deferred discrete-policy section from the
main narrative, condenses the general hard-trim robustness/inference program,
and moves its full diagnostic matrix plus the taxi logit-tip robustness check to
new Appendix 6. The differing-slopes simulations and the original-versus-
differing-slopes taxi comparison remain in the main text as the requested
empirical segue and motivation; detailed differing-slopes assumptions and proof
remain in Appendices 4--5. The revised prelim compiles to 44 pages with no
LaTeX errors or undefined citations.

— Codex

## 2026-09-14 — Empirical differing-slopes segue retained in prelim (Codex)

Follow-up clarification: the original same-slope hard-trimmed result remains the
primary theorem, but the main prelim should retain the differing-slopes simulation
evidence and taxi discussion. Those sections are the segue from the original
method to the broader model, showing that level-dependent effects occur in a
real application and that the extension is useful across settings. Only the
differing-slopes assumptions, detailed proof, and formal asymptotic development
are deferred to the appendix; the empirical motivation is not appendix-only.

— Codex

## 2026-09-14 — Prelim scope narrowed to original hard trimming (Codex)

The author confirms that the prelim's primary theory and narrative should remain
the original same-slope setting with $X\perp(W,\eta)$, deliberately decoupled
sample splitting, and hard trimming. The main document may retain the baseline
simulations and the original residual-indexed taxi analysis as evidence for this
method. The differing-slopes model, its additional assumptions and proof sketch,
and the fare-level taxi correction are deferred to appendix material and should
not be presented as a co-equal main result. The presentation follows the same
priority, with only a brief deferred-extension preview.

— Codex

---

## 2026-09-12 — Paragraph 1 drafting workflow follow-up (Codex)

The author asked to reverse the candidate-prose insertion recorded below. The
prelim now again contains only the concise Paragraph 1 TODO: the author will
write the initial text, and Codex will subsequently flesh it out through the
paragraph-by-paragraph editorial workflow. The three candidate boxes and their
task-board selection item were removed in manuscript commit `fd0d379`; this
follow-up supersedes the earlier entry's proposed next step without altering
that historical entry.

— Codex

---

## 2026-09-14 — Nonlinear treatment-effect simulation (Codex)

Tested whether the linear `D*X` differing-slopes correction remains valid when
the treated effect is nonlinear in the threshold score.  The new runner
`experiments/scripts/nonlinear_slopes_simulation.py` generates

\[
Y=b_0+b_1\eta+X^\top\beta_1
 +D\{a_0+a_1\eta+\delta(T^2-1)\},\qquad T=X_1,
\]

with a hard 10% trim and known `eta`/normal score law.  It compares the
alpha-only model, the existing linear `D*X` model, and a correctly augmented
model containing `D*(T^2-1)`.  The population truth calculation now integrates
directly over the trim interval using Gauss--Legendre quadrature, avoiding the
boundary error from applying a hard indicator to Gauss--Hermite nodes.

For the moderate design (`delta=0.40`, 300 replications at
`n={500,1000,2000,4000}`), the full target is `0.424625`.  Alpha-only bias is
about `0.60` at the larger sample sizes and its optimizer moves to the upper
policy bound.  The linear differing-slopes estimator has stable bias about
`-0.06`, so it is also inconsistent under nonlinear score heterogeneity.  The
quadratic augmentation has biases `0.001, -0.003, -0.004` at
`n={1000,2000,4000}`, RMSE tail slope about `-0.48`, variance ratios
`1.01,1.08,0.93`, and coverage `0.963,0.957,0.940`.  The `n=500` quadratic
variance cell is noisy because of a few poorly conditioned threshold fits.

The null (`delta=0`) control leaves all specifications approximately centered,
while the stronger curvature design (`delta=0.80`) gives linear-model bias near
`-0.20` and quadratic-model bias below `0.002` in absolute value for the larger
cells, with variance ratios near one and coverage about 0.92--0.95.

Conclusion: the original reduction fails under nonlinear score heterogeneity,
and the current linear differing-slopes extension is not a general nonlinear
solution.  A sieve/spline treated-interaction basis is the natural next step;
its basis-growth, regularization, and generated-index inference are not tested
here.  Durable report: `experiments/datasets/simulations/NONLINEAR_SLOPES_SIMULATION_20260914.md`.

— Codex

## 2026-09-12 — Introduction Paragraph 1 candidates (Codex)

At the author's request, reviewed the opening pages of the local Mukherjee,
Banerjee & Ritov and Wibisono et al. PDFs before drafting §1.1. Both papers begin
with a concrete score-threshold allocation, explain the substantive outcome,
and only then introduce the statistical limitation. Following that pacing but
orienting it toward PerfRDD's distinct policy question, added three candidate
openings to `../manuscript/prelim/prelim.tex`. They use the examination--
scholarship--college setting and differ in emphasis: the direct welfare
question, the assignment mechanism and policy tradeoff, and evaluation versus
policy design (manuscript source commit `05aa105`). The candidates remain review
alternatives; the author still needs to select and finalize one before Paragraph
2 is drafted.

— Codex

---

## 2026-09-12 — Focused assumption-swap simulation (Codex)

Tested the hypothesis that the differing-slopes estimator is centered when the
data-generating process satisfies $W=a(\eta)+X^\top\beta_2+R_W$, while the
original $\alpha(\eta)$-only reduction is inconsistent for the full policy
target when $\beta_2\ne0$.  The known-target runner
`experiments/scripts/differing_slopes_simulation.py` was run with 300
replications at $n\in\{500,1000,2000,4000\}$ for baseline, null-interaction,
and strong-interaction designs, plus a 200-replication baseline follow-up at
$n\in\{8000,16000\}$.  The durable report is
`experiments/datasets/simulations/DIFFERING_SLOPES_ASSUMPTION_SWAP_20260912.md`.

In the baseline nonzero-interaction DGP, the true differing-slopes target is
$\phi^*=-0.120064$ while the alpha-only formula's target is $-0.339292$.
The differing-slopes estimator has biases $(0.0014,-0.0008,-0.0026,-0.0004)$
at $n=(500,1000,2000,4000)$ and $(0.0016,0.0009)$ at $n=(8000,16000)$;
its RMSE log--log slope is $-0.504$.  The original estimator remains about
$0.43$--$0.47$ below the full target, with normal coverage falling from
$0.947$ at $n=500$ to $0$ at $n=16000$ even though its estimated variance
tracks its own Monte Carlo dispersion.  This is the expected pattern for a
misspecified but increasingly precise estimator.

The null-interaction control sets $\beta_2=0$; both estimators then have small
bias (absolute values below $0.021$), RMSE slopes near $-1/2$, variance ratios
near one, and coverage about $0.93$--$0.97$.  Under the stronger interaction
$\beta_2=(1.60,0.50)$, the true target is $-0.073010$; differing-slopes bias
stays below $0.003$ in absolute value with coverage $0.927$--$0.950$, while
alpha-only bias ranges from $-1.24$ to $-1.94$ and coverage is $0.573$--$0.723$.

Conclusion: the focused simulation supports the hypothesis and separates
identification bias from variance estimation.  It is conditional on known
$\eta$, fixed trim bounds, and the known Gaussian tail; generated-index,
estimated-endpoint, and moving-boundary terms remain outside this check.  The
known-target unit tests pass (3 tests).  The broader full-pipeline test module
could not import in the system Python because `matplotlib` is not installed;
this is an environment dependency issue, not a failure of the focused runner.

— Codex

---

## 2026-09-12 — Align Beamer scaffold with prelim presentation plan (Codex)

The slide source `manuscript/prelim/slides.tex` now mirrors the approved
50-minute talk: motivation/related work, same-slope setup and theorem,
estimation and hard trimming, simulations, taxi, a three-slide differing-slopes
extension/proof sketch, and conclusion.  The stale performativity subtitle and
40-minute timing comment were removed; the title and author use the manuscript's
placeholder convention.  The deck remains audience-facing scaffolding with
TODO blocks, while the differing-slopes section is explicitly secondary to the
same-slope theorem.

— Codex

## 2026-09-12 — Prelim written-document and presentation scope (Codex)

The author confirms that the written prelim is a **superset** of the oral
presentation.  The written document due Wednesday should be organized around
the introduction and related work, then the main same-slope contribution under
the maintained $X\perp(W,\eta)$ assumption, followed by simulations and the
taxi application.  The taxi results motivate the differing-slopes model; that
extension should explain the failure of the scalar reduction, state why the
$DX^\circ$ correction is useful, and give a proof sketch, but it is not to
displace the same-slope theorem as the primary contribution.

The Thursday presentation is a compressed **50-minute** version of this
material: motivation/related work; setup, target, and identification; estimator,
hard trimming, and the main CLT; simulations; taxi; and finally the differing-
slopes motivation and proof sketch.  Slides should not introduce a separate
estimand or claim a fully completed differing-slopes theorem.  Keep the
deliberately decoupled split and the distinction between theorem-backed
simulations and illustrative taxi diagnostics explicit.

— Codex

## 2026-09-09 — Attempted differing-slopes lemma, extension, and bookkeeping (Codex)

The manuscript Appendix 4 now contains three concrete additions for the
deliberately decoupled split.  First, the augmented outcome normal equations
are expanded around the population sieve coefficient; the generated-index
loading is $Q_{\mathrm{DS},K}^{-1}E[z_{\mathrm{DS},K}X^\top
(\partial_\eta z_{\mathrm{DS},K})^\top\theta_K]$, and the finite-dimensional
$X^\circ$ and $DX^\circ$ blocks contribute no derivative because only the two
spline blocks depend on $\eta$.  Second, the vector weighted-density lemma now
records a Lyapunov/Lindeberg bound $O(L^2/n)$ and the two remainders
$O(\sqrt nL^{-3})$ and $O_p(L^{5/2}/\sqrt n)$ under the existing DS5 window.
Third, the four main block scores and two endpoint scores are assembled with
explicit fold fractions, endpoint signs, and within-block covariance; no
ordinary-cross-fitting or full-sample-reuse theorem is claimed.  The argument
still conditions on the inherited generated-index Bahadur and moving-boundary
lemmas and leaves the final signed loading substitution for the next proof pass.

## 2026-09-09 — Author choice and proof scaffold for differing slopes (Codex)

The author elects to retain the stronger `$X\perp\eta$` condition rather than
use the weaker index-sufficiency condition recorded below.  The author also
accepts finite moments, a unique interior maximizer with nonzero curvature,
reasonable uniform augmented-rank assumptions, and an additional honest fold
for the vector weighted-density nuisance.  Under those choices, the only
substantive new model/regularity conditions are the conditional linear effect
model `$E(W\mid X,\eta)=a(\eta)+X^{\circ\top}\beta_2$` and smoothness/weak-
derivative conditions for `$\rho_X(t)=E(X^\circ\mid T=t)f_T(t)$` and
`$h_\rho(t)=E(XX^{\circ\top}\mid T=t)f_T(t)$`.

Appendix 4 of `manuscript/prelim/prelim.tex` now provides a Luna-ready proof
scaffold: DS1--DS5, a seven-block dependency map, a conditional decoupled-split
CLT theorem, a five-step proof skeleton, and inline completion instructions for
feasible consistency, augmented outcome normal equations/Riesz convergence,
the multiplier weighted-density lemma, inherited moving-boundary substitutions,
signed influence-function/fold bookkeeping, and curvature.  The original local
uniform endpoint Bahadur lemma remains inherited unfinished work, not a new
differing-slopes assumption.

— Codex

---

## 2026-09-09 — Introduction positioning: precedents and competitor (Claude)

Reference/positioning findings for the Introduction rewrite (manuscript commit
448b699; §1.1 restructured, `references.bib` extended).

- **Wibisono, Mukherjee, Banerjee & Ritov (2025)**, arXiv:2504.17126, is the
  heterogeneous-effect extension of the score-explained line. Model
  `$Y=\alpha_0(X,\eta)\mathbb 1\{Q\ge\tau_0\}+X^\top\beta_0+\ell(\eta)+\epsilon$`,
  `$Q=Z^\top\gamma+\eta$`; estimand is the **ATT** `$\E[\alpha_0(X,\eta)\mid Q\ge\tau_0]$`
  (plus CATE/ITE) via first-order differencing + residual matching, three-fold split.
  Assumptions: `$\E(\eta\mid Z)=0$`, `$\E(\epsilon\mid X,\eta)=0$` (mean-independence, not
  full independence), compact supports, density-ratio overlap. PDF saved in
  `manuscript/Papers/`.
- **Mukherjee, Banerjee & Ritov (2021)** (`Dep2021`) is the homogeneous-`$\alpha_0$`,
  fixed-cutoff, global root-$n$ precedent under `$(\eta,v)\perp(X,Z)$` (Assumption 1.1);
  its contribution #1 is explicitly "use of the entirety of data … not just observations
  in a small vicinity of the boundary" — so the RDD **local→global** framing is honest,
  not a strawman.
- **Competitor flag:** Marinescu, Triantafillou & Kording (2022), *PLOS ONE*
  (`Marinescu2022`), is the namesake "RD threshold optimization" and **applies to the same
  Haggag (2014) taxi tip data**. It is design-based (LATE + Dong–Lewbel marginal-threshold
  effect, Gaussian-process regression, cost/conservatism constraints) with **no
  score-explained latent-residual model and no root-$n$ limit theory**. §1.1/§1.2 must
  differentiate it explicitly. PDF given to the author (`~/Downloads`), not committed.
- Both precedents' reference lists were mined; verified entries added to
  `references.bib` (RDD lineage, partial-linear, triangular/IV, and — new to this line —
  policy learning `KitagawaTetenov2018`/`AtheyWager2021`/`Manski2004`, threshold change
  `DongLewbel2015`, overlap trimming `CrumpHotzImbensMitnik2009`).

— Claude

---

## 2026-09-08 — Weakened full covariate--residual independence (Codex)

A final assumption audit showed that the unrestricted differing-slopes target
does not require full `$X\perp\eta$`.  It is enough to assume
`$T\perp\eta$` together with the conditional-mean restriction
`$E(X^\circ\mid T,\eta)=E(X^\circ\mid T)$`.  These conditions imply both the
scalar survival factorization and
`$E[X^\circ 1\{T>s\}\mid\eta]=H_X(s)$`; with centered covariates they also
imply `$E(X^\circ\mid\eta)=0$`, so `$a(\eta)=E(W\mid\eta)$` retains its old
interpretation.  Full `$X\perp\eta$` remains a simpler sufficient primitive.
Dropping either part of this weaker index-sufficiency condition would make
`$\bar G$` or `$H_X$` depend on `eta` and require a higher-dimensional nuisance
analysis.  Appendix 4 and the task board now use the weaker exact condition.

— Codex

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

## 2026-09-15 — Simulation-section literature benchmark (Codex)

Reviewed closely related threshold/RD and policy-inference papers. Marinescu,
Triantafillou, and Kording (2022) use one small illustrative simulation figure and
then a taxi application with 100 bootstrap samples; they do not present a
conventional Monte Carlo section. Mukherjee, Banerjee, and Ritov (2021) state that
no-splitting behavior is corroborated by simulations but explicitly leave those
results out of the manuscript. Dong and Lewbel (2015) are primarily analytical
plus an application, without a substantial Monte Carlo section. By contrast,
Calonico, Cattaneo, and Titiunik (2014) include a dedicated coverage/interval-length
Monte Carlo section and put additional DGP and implementation details in the
supplement. Andrews, Kitagawa, and McCloskey's ``Inference on Winners'' uses
several pages of simulations, multiple calibrated scenarios, competing procedures,
and $10^4$ draws because simulation evidence is central to its inference contribution.

Decision for PerfRDD: keep a compact main-text simulation package (baseline hard
trim, differing-slopes misspecification, nonlinear negative control, and
variance/coverage/rate checks) at roughly 700--1,000 words plus one or two
tables/figures. Move the full robustness grid, implementation details, and extra
diagnostics to the appendix or supplement. This is more validation than most
threshold-optimization papers report, but materially lighter than the simulation
sections of dedicated inference papers. Sources: Marinescu et al.
<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276755>;
Mukherjee et al. <https://arxiv.org/abs/2102.11229>; Dong--Lewbel
<https://doi.org/10.1162/rest_a_00510>; Calonico--Cattaneo--Titiunik
<https://mdcattaneo.github.io/papers/Calonico-Cattaneo-Titiunik_2014_ECMA.pdf>;
Andrews--Kitagawa--McCloskey
<https://scholar.harvard.edu/files/iandrews/files/inference_on_winners.pdf>.

## 2026-09-15 — Main-text simulation scope decision (Codex)

Following the literature benchmark and author direction, the prelim's pre-taxi
Numerical study now answers only centering and the root-$n$ rate. The variance,
coverage, bootstrap, and broader robustness diagnostics remain appendix material.
The fold-construction TODO is resolved by calling the theorem-facing comparison
the ``decoupled block split'' and distinguishing it from five-fold cross-fitting
and full-sample reuse. The differing-slopes simulation evidence is placed after
the taxi application, where the level-dependent extension is motivated, and is
limited in the main text to the correctly specified linear comparison and the
nonlinear negative control. The manuscript change was compiled successfully and
pushed in commits `69090e1` and `34910ae` in the Overleaf repository.

## 2026-09-16 — Retire ordinary five-fold cross-fitting (Codex)

Per author direction, ordinary five-fold cross-fitting is no longer an active
estimator. The hard-trim API now rejects `crossfit_folds=5`; application runners
use full-sample ridge fits, while the simulation comparison remains
`decoupled_8block`, `rotated_8block`, and `full_sample`. The differing-slopes
battery no longer contains five-way variants, and the bootstrap CLI no longer
accepts a five-way estimator. Active dataset notes and manuscript text were
updated accordingly; historical run records are preserved elsewhere in this log
for provenance. Focused hard-trim, differing-slopes, and variance tests pass;
the full test discovery still reports the pre-existing missing `pyarrow`
dependency for the taxi parquet registry tests.

## 2026-09-17 — Slide notation for the outcome residual (Codex)

The prelim's deployed assignment is deterministic, $D=\mathbf 1\{\gamma^\top
X+\eta>\phi_0\}$. Therefore $\sigma(\eta,X,D)=\sigma(\eta,X)$, and the
slide-level condition $\E(\varepsilon\mid\eta,X)=0$ is equivalent to the
fuller $\E(\varepsilon\mid\eta,X,D)=0$ used in the theorem-facing manuscript
statement. The weaker condition $\E(\varepsilon\mid\eta)=0$ alone would not
control residual means across treated and untreated observations after the
covariate adjustment.

For the response-surface slide, the stacked regression directly estimates the
treated-effect spline, untreated baseline spline, and partially linear
adjustment in one fit. The spline basis $\tilde N_K$ and coefficient vectors
$\omega_\alpha,\omega_b$ are now defined on-slide; separate fits or matching
remain possible alternatives, but the stacked fit preserves the covariance
between the two spline blocks and is the cleaner presentation.

## 2026-09-17 — Stack the score-distribution definitions on the slide (Codex)

The estimation slide now presents the distribution chain vertically:
$G(t)=\mathbb P(T\le t)$, $\bar G(t)=1-G(t)$, and
$e_\phi(\eta)=\mathbb P\{T>\phi-\eta\mid\eta\}=\bar G(\phi-\eta)$.
This is only a presentation change; the underlying estimator continues to
estimate the score distribution from fitted index values and evaluate its
survival function at candidate thresholds.

## 2026-09-17 — Correct the scope of presentation TODO 20 (Codex)

The prior TODO 20 edit stacked the score-distribution definitions, but the
author intended TODO 20 for the stacked partially linear response regression.
The score-distribution frame was restored. The response-surface frame now shows
$Y_i\approx Z_i^\top\theta+\varepsilon_i$ with the treated spline, baseline
spline, and $X^\circ$ blocks stacked inside $Z_i$, and the corresponding
coefficient blocks stacked inside $\theta$.

## 2026-09-17 — Verify the Wibisono comparison slide (Codex)

The Wibisono et al. paper models the individual treatment effect as
$\alpha_0(X,\eta)$, so it may depend on observed background covariates $X$ as
well as the latent score residual $\eta$. Its ATT estimator first estimates the
linear adjustment by first-order differences among controls, then matches each
treated observation to the nearest control in estimated residual $\hat\eta$, and
averages covariate-adjusted treated--control outcome differences. This is a
residual-matching estimator, not a nonparametric outcome regression for
$\alpha(\eta)$. Source: Wibisono et al., arXiv:2504.17126,
<https://arxiv.org/abs/2504.17126> (Sections 1--2).

## 2026-09-23 — Keep the structural index uncentered (Codex)

The differing-slopes note now states the structural definition
$T=\gamma^\top X$ explicitly. The notation $X^\circ=X-\E[X]$ is reserved
for the centered covariates in the outcome slope block and does not redefine
the treatment/index component. Consequently the generated residual error is
$-X^\top(\widehat\gamma-\gamma)$; centering the outcome covariates alone
does not remove its loading. A centered-index representation requires the
intercept $\gamma^\top\E[X]$ to be handled explicitly.

## 2026-09-23 — Nonlinear alpha/b outcome-flexibility diagnostic (Codex)

Ran `experiments/scripts/differing_slopes_nonlinear_outcome.py` to test the
current differing-slopes pipeline when both the treated response
$\alpha(\eta)$ and the baseline $b(\eta)$ are quadratic in the latent score:
$a_0+a_1\eta+a_2(\eta^2-1)$ and $b_0+b_1\eta+b_2(\eta^2-1)$, with
$(a_2,b_2)=(0.35,0.50)$ in the nonlinear scenario. The DGP uses two observed
covariates, the full $D X$ block, an OLS-generated index, hard trimming at the
known Gaussian 10\% tails, and the known Gaussian score tail in the final
objective. This is deliberately an outcome-flexibility diagnostic, not a full
generated-index/density-Riesz theorem validation.

The 500-replication battery at $n\in\{800,1600,3200,6400\}$ completed in
roughly 30 seconds. It compares the current linear OLS response fit with a
correctly specified quadratic fit and a 10-basis spline fit. In the linear DGP,
all three estimators center on the same target (at $n=6400$, biases are
$-0.0008$, $-0.0007$, and $-0.0011$ for linear, quadratic, and spline). In the
nonlinear DGP, the linear fit has persistent pseudo-target bias of about
$-0.033$ to $-0.036$ across $n=800$--$6400$, whereas the quadratic fit reduces
it to $-0.006$ to $-0.008$ and the spline to $-0.008$ to $-0.015$. At $n=6400$
the RMSEs are $0.0446$ (linear), $0.0299$ (quadratic), and $0.0382$ (spline);
the corresponding $n$-scaled variances are $4.63$, $5.35$, and $8.89$.

Interpretation: nonlinear $\alpha$ and $b$ invalidate the current linear
outcome reduction through a non-vanishing bias, even though the full differing-
slopes block is present. A correctly specified low-dimensional quadratic fit
recovers most of the target at modest variance cost; the flexible spline also
removes the bias but is noisier with ten basis functions. Results are stored in
`outputs/differing_slopes_nonlinear_outcome_20260923/nonlinear_outcome_results_500.json`.

## 2026-09-23 — Percentile bootstrap for nonlinear alpha/b fits (Codex)

Added `experiments/scripts/differing_slopes_nonlinear_bootstrap.py` and ran a
full-sample re-estimation percentile bootstrap: 50 outer samples and 199
bootstrap draws per outer sample, for both the linear benchmark and nonlinear
$\alpha,b$ DGPs at $n\in\{800,1600,3200\}$, plus a separate nonlinear $n=6400$
run. Every draw re-estimates the index, hard trimming, outcome coefficients,
and policy optimum. There were no failed bootstrap draws.

For the nonlinear DGP, percentile coverage for the linear fit was $0.94$,
$0.88$, and $0.94$ at $n=800,1600,3200$, but fell to $0.64$ at $n=6400$.
The corresponding persistent point biases were about $-0.040$, $-0.029$,
$-0.033$, and $-0.040$. Thus the bootstrap can look acceptable at moderate
sample sizes while failing to cover the structural target once the interval
shrinks around the misspecified pseudo-target. The correctly specified
quadratic fit had coverage $0.92$, $0.96$, $1.00$, and $0.92$ across those four
sample sizes; its bias stayed between $-0.002$ and $-0.017$. The spline fit had
coverage $0.98$, $0.94$, $0.96$, and $0.92$, with higher bootstrap dispersion
than the quadratic fit. In the linear DGP, coverage was $0.98$, $0.94$, and
$0.92$ for the linear fit at $n=800,1600,3200$ (Monte Carlo standard errors are
about $0.02$--$0.04$ with 50 outer samples).

These are finite-sample diagnostics, not a bootstrap validity theorem for the
fully decoupled generated-index estimator. Results are in
`outputs/differing_slopes_nonlinear_bootstrap_20260923/summary.json` and
`outputs/differing_slopes_nonlinear_bootstrap_20260923_n6400/summary.json`.

## 2026-09-23 — Higher-replication bootstrap attempt stopped at author deadline (Codex)

Attempted a larger nonlinear-only run with 200 outer samples, 399 bootstrap
draws, four sample sizes ($n=800,1600,3200,6400$), and four workers. The run
was still computing at the author's 2:50 hard stop and was terminated before
writing a summary; no incomplete output is used in the manuscript. The
completed 50-by-199 pilot remains the verified bootstrap result. A larger
replication should be rerun on the cluster or with a more efficient estimator
implementation before making publication-grade coverage claims.
