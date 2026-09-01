# PerfRDD research log — shared between Claude and Codex

This tracked file is the single source of truth for **verified findings and decisions**.
The operating rules are in `COLLABORATION.md`; forward-looking work is tracked in
`../manuscript/TODO.md`.

Add new entries immediately below the divider, newest first. Date and sign every entry.
Never edit or delete another agent's entry; add a follow-up when a conclusion changes.

---

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
