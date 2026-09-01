# PerfRDD research log — shared between Claude and Codex

This tracked file is the single source of truth for **verified findings and decisions**.
The operating rules are in `COLLABORATION.md`; forward-looking work is tracked in
`../manuscript/TODO.md`.

Add new entries immediately below the divider, newest first. Date and sign every entry.
Never edit or delete another agent's entry; add a follow-up when a conclusion changes.

---

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
