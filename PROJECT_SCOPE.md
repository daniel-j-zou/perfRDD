# PerfRDD project scope

The current agreed scope and methodology of the paper, in one place. Start every new
session by reading this file, then the newest entries in `RESEARCH_LOG.md` and
`../manuscript/TODO.md`.

- **Last reviewed with the author:** 2026-10-13 (Claude)
- **Covers decisions up to:** research log entries of 2026-10-12

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
- **Main contribution:** the combination of three things, with no single one leading:
  - the cutoff itself as the estimand;
  - the differing-slopes correction;
  - the limit theory for trimming with a generated regressor.

## Model and target

- **Score:** Q = gamma' X~ + eta, a linear index plus a latent residual. X is independent of
  eta.
- **Effect:** given (X, eta), W has mean alpha(eta) + X' beta2. X is centered, so
  alpha(eta) = E(W | eta), the average effect at latent type eta. Differing slopes is the
  model and common slope is a special case. The effect is linear in X.
- **Notation:** alpha, not a, is the treatment-effect function, in the paper and the
  notebook.
- **Target:** the hard-trimmed utility maximizer phi*_eps. The trim level eps is fixed and
  chosen by the user. Present phi*_eps as the optimal cutoff for the population the data
  supports. Add a remark relating it to phi*, and show sensitivity to eps in the
  application.
- **Interior optimum:** state a proposition with simple sufficient conditions, as a
  single-crossing condition on alpha(eta) - c + beta2' E(X | T = phi - eta) over the trim
  window. It extends "Uniqueness of the maximizer" in `../manuscript/goodstuff.tex`.
  Done (2026-10-19): Proposition 2 of the paper, with (DS1)(iii) as the
  single-crossing condition.

## Estimator

- **Split:** four roles with one shared first stage: first stage, outcome, index (density
  and trim endpoints), and evaluation.
- **Fixed and rotated:** the paper states a theorem for one fixed assignment and a
  corollary for the rotated estimator. The rotated estimator is the average of the four
  per-rotation thresholds. The paper keeps the average and adds a remark on its
  second-order bias, with the pooled and combined alternatives (2026-10-19).
- **Nuisances:**
  - spline series estimates of g and rho_X (the pairwise empirical-tail route is not used);
  - a stacked spline outcome regression;
  - an ordinary-least-squares first stage.
- **Spline sizes:** chosen by cross-validation over a grid inside the theorem's rate range.
- **Full-sample reuse:** a simulation remark only, with no theory.
- **No efficiency or debiasing in the paper** (2026-10-19): the paper keeps the plug-in
  estimator with the OLS first stage and makes no efficiency claim. The debiased
  evaluation score and the efficient first stage stay in the notebook as a record.

## Theory

- **Rigor:** the theory must be complete. Condition (DS5), uniform consistency of U-hat and
  its first two derivatives, is derived from primitive assumptions, not assumed.
  Done (2026-10-19): Proposition 3 of the paper; in the paper (DS5) now names the
  sieve and rate condition.
- **Asymptotics:** pointwise, for a fixed data-generating process and a fixed eps.
- **Boundary optima** (always treat or never treat) are outside the CLT.

## Inference

- Bootstrap confidence intervals, supported by coverage simulations. There is no bootstrap
  validity theorem.
- The analytic four-role variance is used only as a cross-check in simulations.

## Simulations

- **Comparisons:** our estimator against the truth, plus an oracle with known nuisances.
  Nothing else.
- **Gaps to close:**
  - the four-role differing-slopes run with the spline outcome fit;
  - rerun the displayed differing-slopes design with its own optimum (phi* ~ 0.545103);
  - bootstrap coverage for the rotated estimator;
  - a design with nonlinear alpha and b.

## Application

- NYC taxi tip menus only. The purpose is to show the method working, with an estimate and a
  bootstrap CI, not to make a policy claim. More datasets may be added later if needed.
- **Sample:** January 2009 VTS credit-card rides with the Haggag–Paci restrictions except
  their $5–$25 fare range, which selects on the running variable. The restricted sample is
  kept for a comparison subsection.
- **Taxi model:** the treatment effect is nonlinear in eta and linear in the covariates (no
  functions of distance in the D*X block), and eta should carry meaning. The candidates are
  the nine specifications compared in `this_week.tex` §7 (fare in dollars).

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
- **Closest prior work (pending author review):**
  - Zhang, Ben-Michael & Imai;
  - Eckles et al. (Biometrika 2025);
  - Ben-Michael et al. (JASA 2025);
  - Carneiro, Heckman & Vytlacil (2010);
  - Crippa.

  A reading brief is at https://claude.ai/code/artifact/ba824b8e-f3cc-4d05-8c5b-0112bc3e5706.
  Nothing is added to `references.bib` until the author decides.

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

- Positioning against the five closest papers, and which of them to cite (see the reading
  brief).
- Which of the checked proof simplifications (log 2026-10-13) to integrate.
- Which of the nine taxi specifications in `this_week.tex` §7 to adopt and port to the
  paper's §6. Suggested: a score with distance and log distance, and the effect linear in
  distance. It gives $11.76 (cell $12.10, 90% [$11.36, $12.17], 20 draws). The spline score
  gives the $13.30 cell, and the common slope always selects the $6 boundary. Two short
  theory extensions are needed: different covariates in the score and the effect, and the
  quantile lemma with distance entering the score through several columns.

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
| 2026-10-19 | Notebook §5 theory ported into the paper; no efficiency or debiasing in the paper; rotated average kept with a second-order-bias remark; §6 taxi port deferred | Log: "Paper brought up to date: identification and consistency theory ported" |
| 2026-10-15 | Taxi effect nonlinear in eta and linear in covariates; eta must carry meaning; nine-way comparison in `this_week.tex` §7 | Log: "Taxi: nine-way comparison of score and effect specifications" |
| 2026-10-15 | Taxi main sample without the $5–$25 fare range; the restricted sample as a comparison | Log: "Taxi application with the four-role estimator" |
| 2026-10-14 | Treatment-effect function written alpha, with alpha(eta) = E(W \| eta) for centered X | Log: "Notation: the treatment effect is alpha(eta) = E(W \| eta)" |
| 2026-10-12 | Debiased-score alternative noted as an open question; plug-in kept for now | Log: "Why MBR's nuisance folds add no variance" |
| 2026-10-12 | Spline densities, constrained CV, rotated average, (DS5) derived, fixed eps framing, single-crossing interior optimum, oracle-only comparison, Wibisono position | Log: "Methodology decisions and prior-work check" |
| 2026-10-12 | Journal target, `differing_slopes.tex` as the paper, four roles, bootstrap, taxi only, side projects parked | Log: "Paper scope consensus with the author" |
| 2026-10-06 | Prelim stays frozen, with its fixed-assignment-only statement | `../manuscript/TODO.md` |
| 2026-09-21 | Differing slopes becomes the primary model of the final paper | `../manuscript/TODO.md`, "Final-paper direction" |
