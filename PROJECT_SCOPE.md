# PerfRDD project scope

The current agreed scope and methodology of the paper, in one place. Start every new
session by reading this file, then the newest entries in `RESEARCH_LOG.md` and
`../manuscript/TODO.md`.

- **Last reviewed with the author:** 2026-10-20 (Claude)
- **Covers decisions up to:** research log entries of 2026-10-21 ("Follow-up: residual and latent ability")

This file holds author decisions only. Evidence and derivations stay in `RESEARCH_LOG.md`;
tasks stay in `../manuscript/TODO.md`. Every item below can be traced through the
decision history at the end. How to change this file is described in "Updating this file".

## Goal

- **Deliverable:** a journal submission. Write it in *Biometrika* style; the venue is not
  final and there is no date.
- **Paper of record:** `../manuscript/differing_slopes.tex`. The main text carries the
  argument; complete proofs go in the supplement.
- **Frozen:** `prelim/prelim.tex` and `prelim/slides.tex` are the prelim record. Do not edit
  them unless the author asks.
- **Name:** "Performative RDD" is a legacy name. The paper does not model units reacting to
  the cutoff.
- **Main contribution** (2026-10-20): the cutoff itself as the estimand leads. The paper
  point-identifies the utility-maximizing cutoff from data under one deterministic cutoff
  and gives a root-n limit for it. Two things support that headline:
  - an effect that varies with the covariates as well as the latent type, so the utility
    follows the covariate composition of each counterfactual margin (2026-10-21: no
    longer described as a "correction");
  - the limit theory for trimming with a generated regressor.

## Model and target

- **Score:** Q = gamma' X~ + eta, a linear index plus a latent residual. X is independent of
  eta.
- **Effect:** given (X, eta), W has mean alpha(eta) + X' beta2. X is centered, so
  alpha(eta) = E(W | eta), the average effect at latent type eta. Differing slopes is the
  model and common slope is a special case. The effect is linear in X.
- **Positioning in the text** (2026-10-21): the paper presents differing slopes as its
  setting, not as an extension of a common-slope model. Common slope appears only as
  the beta2 = 0 special case (Corollary "Common slope" and the sentence in §2.4 that
  points to it). The outcome model is written without potential outcomes:
  Y = b(eta) + X' beta1 + D W + eps.
- **Notation:** alpha, not a, is the treatment-effect function, in the paper and the
  notebook.
- **Paper notation** (2026-10-21): the assumptions are (A1)-(A7), formerly (DS1)-(DS7);
  the conditions of Proposition 2 (unique interior optimum) are (U1)-(U4); symbols carry
  no "DS" tag (R_Y, r_out, xi, F*_eps, sigma^2, ...); the negative utility derivative
  -U_eps'(phi) is Upsilon(phi), so that Z stays free for score covariates. eta is called the
  "latent ability", not "latent type", and "the residual" (not "latent residual")
  where the score decomposition or its estimate is meant. The notebook
  and the log keep the old labels as a record.
- **Target:** the hard-trimmed utility maximizer phi*_eps. The trim level eps is fixed and
  chosen by the user. Present phi*_eps as the optimal cutoff for the population the data
  supports. Add a remark relating it to phi*, and show sensitivity to eps in the
  application.
- **Interior optimum:** state a proposition with simple sufficient conditions, as a
  single-crossing condition on alpha(eta) - c + beta2' E(X | T = phi - eta) over the trim
  window. It extends "Uniqueness of the maximizer" in `../manuscript/goodstuff.tex`.
  Done (2026-10-19): Proposition 2 of the paper, with (A1)(iii) as the
  single-crossing condition.

## Estimator

- **Split:** four roles with one shared first stage: first stage, outcome, index (density
  and trim endpoints), and evaluation.
- **Fixed and rotated:** the paper states a theorem for one fixed assignment and a
  corollary for the rotated estimator. The rotated estimator is the average of the four
  per-rotation thresholds. The paper keeps the average and adds a remark on its
  second-order bias, with the pooled and combined alternatives (2026-10-19).
- **Nuisances:**
  - one spline series projection of h = (g, rho_X) on one full clamped basis (2026-10-20;
    the pairwise empirical-tail route is not used);
  - a stacked spline outcome regression (equivalently, one partially linear fit per arm);
  - an ordinary-least-squares first stage.
- **Spline sizes** (2026-10-20): a fixed deterministic rule inside the (A5) range (the
  n^{11/60} rule of the simulations). Cross-validation over a grid inside that range is a
  robustness check only.
- **Full-sample reuse:** a simulation remark only, with no theory.
- **No efficiency or debiasing in the paper** (2026-10-19): the paper keeps the plug-in
  estimator with the OLS first stage and makes no efficiency claim. The debiased
  evaluation score and the efficient first stage stay in the notebook as a record.

## Theory

- **Rigor:** the theory must be complete. Assumption (A5), uniform consistency of U-hat and
  its first two derivatives, is derived from primitive assumptions, not assumed.
  Done (2026-10-19): Proposition 3 of the paper; in the paper (A5) now names the
  sieve and rate condition.
- **Asymptotics:** pointwise, for a fixed data-generating process and a fixed eps.
- **Proof route** (2026-10-20): the first-stage effects are found by conditioning on the
  first-stage fold, under the index-family assumption (A4); X is centred without loss of
  generality (exact shift invariance); fourth moments of X~; spline sizes
  n^{1/7} << K << n^{1/4} and n^{1/8} << L << n^{1/4}. The old Taylor-expansion route is
  kept only in a draft-only note in the paper.
- **Boundary optima** (always treat or never treat) are outside the CLT.

## Inference

- Bootstrap confidence intervals, supported by coverage simulations. There is no bootstrap
  validity theorem.
- The paper reports percentile intervals, with 999 draws in the coverage simulations
  (2026-10-20).
- The analytic four-role variance is used only as a cross-check in simulations.

## Simulations

- **Comparisons:** our estimator against the truth, plus an oracle with known nuisances.
  Nothing else. The linear-in-eta negative control in the paper's §5.3 table is dropped
  (2026-10-20).
- **Gaps closed** (2026-10-19 and 2026-10-20): the four-role run with the spline outcome
  fit; the displayed design at its own optimum (phi* ~ 0.545103); bootstrap coverage for
  the rotated estimator; the nonlinear alpha, b design.
- **To do** (2026-10-20):
  - rerun the coverage tables with 999-draw percentile intervals;
  - in the nonlinear design, n Var stays 15-19% above the first-order 4.73. Check the
    analytic value first, then decide whether larger n is needed.

## Application

- NYC taxi tip menus only. The purpose is to show the method working, with an estimate and a
  bootstrap CI, not to make a policy claim. More datasets may be added later if needed.
- **Sample:** January 2009 VTS credit-card rides with the Haggag–Paci restrictions except
  their $5–$25 fare range, which selects on the running variable. The restricted sample is
  kept for a comparison subsection.
- **Taxi model:** the treatment effect is nonlinear in eta and linear in the covariates (no
  functions of distance in the D*X block), and eta should carry meaning.
- **Taxi specifications** (2026-10-20): §6 reports two specifications side by side, with
  no single headline. Both have the effect linear in distance (E1), on the dollar scale:
  - S1-E1, a linear-distance score: $11.87, cell $12.10;
  - S2-E1, a distance + log-distance score: $11.76, cell $12.10.

  S2 is included because its eta-hat is closer to independent of the covariates
  (R^2 0.026 against 0.092). Both carry the independence diagnostics. The theorem needs a
  short extension for different covariates in the score and the effect. Under S2,
  distance enters the score through two columns, which the quantile lemma must cover.

## Positioning

- **Mukherjee, Banerjee & Ritov (Bernoulli 2026):** the paper builds on it, with the same
  Q decomposition and the same ideas. The target is larger (the threshold, plus differing
  slopes) and needs more machinery.
- **Wibisono et al. (2025):**
  - What they allow: a nonparametric alpha0(X, eta), without assuming X is independent of
    eta.
  - What they target: the ATT at the fixed cutoff.
  - Our position: our setting is harder, because counterfactual cutoffs need X independent
    of eta and an effect linear in X.
  - Do not discuss their differencing estimator.
- **Root-n rate:** no remark explaining it relative to threshold-learning papers.
- **Closest prior work (decided 2026-10-20; the paper's §1 follows this):**
  - Zhang, Ben-Michael & Imai, and Ben-Michael et al. (JASA 2025): they bound the
    unidentified counterfactual and give safety and regret guarantees. Our score model
    buys point identification and a root-n limit, at the price of X independent of eta
    and an effect linear in X.
  - Eckles et al. (Biometrika 2025): the identifying idea is framed through them. The
    covariate index randomizes treatment given eta, and it is observed and may enter
    outcomes.
  - Carneiro, Heckman & Vytlacil (2010): the first-order condition is a
    marginal-effect-equals-cost condition in their sense.
  - Crippa: not cited.
  - Novelty claim: no existing method point-identifies the utility-maximizing cutoff from
    one deterministic cutoff and gives a root-n limit for it.

  Reading brief: https://claude.ai/code/artifact/ba824b8e-f3cc-4d05-8c5b-0112bc3e5706.

## Out of scope (future work)

Discrete Q, online or adaptive thresholds, multiple scores, the Chile grade-retention
application, and other candidate datasets. Nonparametric first stage or outcome in X.

## How we work

- **Direction:** set by the advisors (Banerjee, Ritov).
- **Agents:** Claude is the main agent, and the author assigns tasks per request. Codex
  works under the same protocol (`COLLABORATION.md`).
- **Notebook to paper:** `../manuscript/this_week.tex` is the working notebook. Once a result
  is verified and the author approves it, port it into `differing_slopes.tex`.
- **Prose:** plan first, then write one paragraph at a time (see `COLLABORATION.md`).

## Open questions for the author

- Nonlinear-design variance (Simulations, "To do"): checked 2026-10-21. The analytic
  value 4.73 is right, and the excess is a small-n effect from the 8-function density
  basis, gone by n = 12,800 (log, "Nonlinear design: the analytic variance is right").
  Add larger-n cells to the §5 table, or only reword the small-n excess?

## Updating this file

- **When a decision changes:** whenever the author makes or changes a scope or
  methodology decision, the agent in that session:
  - edits the affected line here;
  - adds a row to the decision history below;
  - writes the supporting research-log entry;
  - commits all of these together.
- **Weekly review:** once a week, an agent reads the research-log entries and task-board
  changes since "Covers decisions up to". It lists the proposed changes to this file for
  the author. After the author confirms, it applies them, updates both dates at the top,
  and commits and pushes. If nothing changed, it updates only "Last reviewed".
- **Tools (Claude):** both skills live in the workspace's `.claude/skills/`, which is not
  tracked in git.
  - `/perfrdd-start` briefs a new session from this file.
  - `/perfrdd-scope-review` runs the weekly review as a question-and-answer session with
    the author.
  - A desktop-app scheduled task, `perfrdd-weekly-scope-review`, starts the review every
    Monday at about 9:00. It runs only while the app is open, or on its next launch.
- **Authority:** only the author's decisions change this file. An agent's own findings go
  to the research log and are listed under "Open questions" until the author decides.

## Drift check (the author asked to be told)

At the start of every session, and before any scope or methodology change, the agent
checks the three conditions below. If any one holds, it tells the author plainly before
doing other work.

1. **Stale review:** "Last reviewed" is more than 7 days old.
2. **Unrecorded decision:** a research-log entry dated after "Covers decisions up to"
   records an author decision that this file does not reflect.
3. **Conflicting work:** the work about to be done, or recent commits, conflicts with this
   file. Examples: a different estimator, a dropped requirement, or an edit to a parked or
   frozen item.

When the author asks for something that conflicts with this file, the agent says so and
asks whether to update the file first.

## Decision history

| Date | Decision | Source |
| --- | --- | --- |
| 2026-10-21 | "Latent residual" becomes "residual"; "latent ability" where eta is a unit characteristic | Log: "Follow-up: residual and latent ability" (2026-10-21) |
| 2026-10-21 | eta is called the "latent ability" in the paper | Log: "Follow-up: eta is the latent ability" (2026-10-21) |
| 2026-10-21 | Positioning: differing slopes is the setting, common slope only the beta2 = 0 corollary; no "correction"/"extension" framing; no potential-outcome notation | Log: "Positioning: differing slopes is the setting" (2026-10-21) |
| 2026-10-21 | Paper notation: no DS tags; (DS1)-(DS7) renamed Assumptions (A1)-(A7); Proposition 2 conditions (U1)-(U4); Z(phi) renamed Upsilon(phi) | Log: "Notation: no DS tags; assumptions (A1)-(A7)" (2026-10-21) |
| 2026-10-20 | Weekly review: the cutoff as estimand leads the contribution; taxi §6 reports S1-E1 and S2-E1 side by side; simulations oracle-only (linear-fit contrast dropped); percentile intervals with 999 draws; a fixed spline-size rule with CV as a check; GPA stays out | Log: "Weekly scope review" (2026-10-20) |
| 2026-10-20 | All eight proof simplifications integrated: one density block, closed-form outcome representer, conditioning on the first-stage fold (old route kept in a draft-only note), centred X, least-squares centring, wider spline-size range, pruned conditions, pathwise remark | Log: "Proof simplifications integrated into differing_slopes.tex" |
| 2026-10-20 | Positioning: abstract and introduction rewritten against the four closest papers (Crippa not cited); eps sensitivity reported in §5.3 | Log: "Abstract and introduction: positioning and novelty" |
| 2026-10-19 | Notebook §5 theory ported into the paper; no efficiency or debiasing in the paper; rotated average kept with a second-order-bias remark; §6 taxi port deferred | Log: "Paper brought up to date: identification and consistency theory ported" |
| 2026-10-15 | Taxi effect nonlinear in eta and linear in covariates; eta must carry meaning; nine-way comparison in `this_week.tex` §7 | Log: "Taxi: nine-way comparison of score and effect specifications" |
| 2026-10-15 | Taxi main sample without the $5–$25 fare range; the restricted sample as a comparison | Log: "Taxi application with the four-role estimator" |
| 2026-10-14 | Treatment-effect function written alpha, with alpha(eta) = E(W \| eta) for centered X | Log: "Notation: the treatment effect is alpha(eta) = E(W \| eta)" |
| 2026-10-12 | Debiased-score alternative noted as an open question; plug-in kept for now | Log: "Why MBR's nuisance folds add no variance" |
| 2026-10-12 | Spline densities, constrained CV, rotated average, (DS5) derived, fixed eps framing, single-crossing interior optimum, oracle-only comparison, Wibisono position | Log: "Methodology decisions and prior-work check" |
| 2026-10-12 | Journal target, `differing_slopes.tex` as the paper, four roles, bootstrap, taxi only, side projects parked | Log: "Paper scope consensus with the author" |
| 2026-10-06 | Prelim stays frozen, with its fixed-assignment-only statement | `../manuscript/TODO.md` |
| 2026-09-21 | Differing slopes becomes the primary model of the final paper | `../manuscript/TODO.md`, "Final-paper direction" |
