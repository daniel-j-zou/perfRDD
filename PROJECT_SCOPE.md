# PerfRDD project scope

The current agreed scope and methodology of the paper, in one place. Start every new
session by reading this file, then the newest entries in `RESEARCH_LOG.md` and
`../manuscript/TODO.md`.

- **Last reviewed with the author:** 2026-11-04, a full refresh pass (Claude)
- **Covers decisions up to:** research log entries of 2026-11-04 ("Taxi theory plan; R3-E1 is weakly identified and its score is not monotone")

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
- **Abstract** (2026-11-04): states the novelty claim, folded into its root-n sentence:
  the utility-maximizing threshold is point identified from data collected under a
  single deterministic threshold.

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
  and the log keep the old labels as a record. Since 2026-11-04 (author, "Yes, all"):
  §3's regressor is \hat z_i as in the appendices; density sieve Pi_L, calG_L, breve N_L;
  bias constant calB; sample size n throughout; \hat I_i defined in §3. One name each:
  "outcome interval J", "trim window [l_0,u_0]", "trim endpoints", "supported
  population", "folds". Applied once the taxi theory landed (manuscript 558099d): no eps
  subscript on representers and loadings (r_out, r_T, L_T, A_U, A_T, F*, s*, ...), Gram
  matrix Sigma_K, B of Lemma ds-rout = ell_2, Block D's G_a(q) (G_gamma = G), quantile
  influence zeta_p, f_T -> g, bare S. "Threshold", never "cutoff" (author, 2026-11-04).
- **Target:** the hard-trimmed utility maximizer phi*_eps. The trim level eps is fixed and
  chosen by the user. Present phi*_eps as the optimal cutoff for the population the data
  supports. Add a remark relating it to phi*, and show sensitivity to eps in the
  application. Where the eps endpoints fall outside the data, the window is the eps
  window capped at fixed residual quantiles (taxi: the central 99%), and the theory
  covers that cap (2026-11-04). Confirmed 2026-11-04 against two alternatives, a window
  fixed before the analysis and a window at fixed residual shares (a trimmed mean in
  eta-hat, which would drop (A6) and Block D): the eps window stays because it assures
  the propensity floor on the target's population by definition.
- **Interior optimum:** state a proposition with simple sufficient conditions, as a
  single-crossing condition on alpha(eta) - c + beta2' E(X | T = phi - eta) over the trim
  window. It extends "Uniqueness of the maximizer" in `../manuscript/goodstuff.tex`.
  Done (2026-10-19): Proposition 2 of the paper, with (A1)(iii) as the
  single-crossing condition.
  Rewritten (2026-11-04): uniqueness is argued by "the margin moves up" (log-concave g
  moves the marginal units' latent ability up, log-concave f_eta on the window moves
  their index up; a local form of Efron 1965), with the three-term decomposition kept
  in Appendix 3 for the curvature. (U1) asks for log-concave f_eta only when lambda is
  not constant on V_Phi.

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
- **Taxi extension** (2026-11-04): the supplement extends Theorem 1 to the §6 use
  cases, with each step reviewed before it goes in, and the hedge in §6 becomes a
  pointer to it. Three gaps: the effect uses a subset of the score covariates (a
  restricted D·X block, with its rank condition); a covariate enters the score through
  several columns, as a smooth increasing function (extends (Q1m)); and the window is
  capped at residual quantiles (endpoint terms a trimmed mean in eta-hat). Plan agreed
  (2026-11-04): notebook derivation with numerical checks, proof review, then the
  appendix.
- **Outcome region J in §6** (2026-11-04): J is chosen once from the full-sample
  residuals and held fixed in every fit and bootstrap draw; §6 says so in one sentence,
  and the theory treats J as given. No proof for an estimated J.

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
  - rerun the coverage tables with 999-draw percentile intervals (the run finished on
    2026-10-23; the port into §5 is left);
  - in the nonlinear design, reword the small-n excess in §5 (2026-10-22): the analytic
    4.73 is right and the excess is a small-n density-basis effect, gone by n = 12,800.
    No larger-n cells are added.

## Application

- NYC taxi tip menus only. The purpose is to show the method working, with an estimate and a
  bootstrap CI, not to make a policy claim. More datasets may be added later if needed.
- **Sample:** January 2009 VTS credit-card rides with the Haggag–Paci restrictions except
  their $5–$25 fare range, which selects on the running variable. The restricted sample is
  kept for a comparison subsection.
- **Taxi model:** the treatment effect is nonlinear in eta and linear in the covariates (no
  functions of distance in the D*X block), and eta should carry meaning.
- **Taxi specification** (2026-10-27, "for now"): the paper's use case is R2-E1. The fare
  is on the square-root scale (Box-Cox 1/2), the score has distance and log distance plus
  the indicators, and the effect is linear in distance. It replaces the 2026-10-20 plan to
  report S1-E1 and S2-E1 (dollar scale) side by side. The theory for it is the taxi
  extension under "Theory".
- **Spline score R3-E1** (2026-11-04, revised the same day): a robustness row, not a
  second use case; R2-E1 stays the single use case. R3-E1 keeps the square-root fare and
  the effect linear in distance, with a spline in log distance plus the indicators in
  the score. Its row carries the identification caveat: D×distance keeps 0.48% of its
  variance after partialling.
- **§6 comparisons** (2026-11-04): the dollar-scale S1-E1 and S2-E1, the common slope and
  the $5–$25 sample are robustness rows; CMT is an external benchmark only.
- **Location-scale first stage** (2026-10-22): a robustness check for the taxi
  application only (notebook §8). The model keeps X independent of eta; the paper's
  theory is not extended to the location-scale model.

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
- **Revised 2026-11-12 (author: "Fix all prior work"; manuscript e6da428).** §1 now:
  - credits Marinescu et al. (2022) for the same objective, the zero-local-effect
    condition, and an optimal threshold from one cutoff with bootstrap SEs, on the same
    January 2009 taxi rides (\$11.11; compared with \$12.87 in §6.4);
  - states the policy-learning contrast as "needs overlap" (the score is observed);
  - says Wibisono et al. have a nonparametric effect at the deployed threshold, without
    X ⊥ eta;
  - adds Rokkanen (2015), Yata (2021) and Bertanha (2020);
  - words the novelty claim "under a score model".

  §1 order: target, then difficulty, gap, idea, purpose, optimum and why root-n, price,
  prior work grouped by what it gives up, tools, contributions, taxi.

## Out of scope (future work)

Discrete Q, online or adaptive thresholds, multiple scores, the Chile grade-retention
application, and other candidate datasets. Nonparametric first stage or outcome in X.

## Side tracks (next paper)

Groundwork for a later paper, outside this paper's scope (2026-11-04). Agents work on it
only when the author asks; it does not change the sections above.
- **Two scores:** the two-score estimator and its datasets
  (`experiments/datasets/MULTI_RULE_DATASETS_20261027.md`). Continuous outcomes only
  (author, 2026-10-28).
- **Dose schedules:** the continuous-dose estimator on real data
  (`experiments/datasets/DOSE_SCHEDULE_DATASETS_20261028.md`). Chile SEP is on hold. The
  CWBH data asks users to contact its maintainers for a formal agreement, which needs
  an author decision before any publication.
- **Binary outcomes** (2026-11-11): a generalized linear outcome model for a binary Y.
  The routes are in `../manuscript/binary_outcome_sketch.tex`; the separable expansion
  of the link keeps the paper's one-projection density block. Results so far
  (`experiments/datasets/BINARY_OUTCOME_ROUTES_20261110.md`): no outcome model gives an
  interior optimum on GPA or Chile; very low birth weight (`experiments/datasets/vlbw/`,
  NCHS 1995-2002 cohorts, 1500 g, infant survival) gives one under the logit once a per-infant
  cost is set (2002: 1508 g at c = 0.02, bootstrap [1476, 1543] g; pooled 1995-2002,
  n = 2.1M: 1545 g at c = 0.02, bootstrap [1529, 1555] g, 1500 g optimal near c = 0.025; with the policy set widened to 2200 g the
  logit effect crosses zero at about 1870 g, so c = 0 is interior there). The write-up of this setting with its method is
  `../manuscript/binary_outcome_vlbw.tex` (2026-11-11). Decision 2026-11-12: the method is the
  logit on both arms (spline tails or empirical tails, the latter the default); the linear probability
  model is kept only as the contrast and the additive-effect route is dropped. The note's Section 6
  states the theory for the logit routes (expansion bias, four-role CLT with modified role scores,
  U-statistic route, single crossing with a link) with arguments, not full proofs, and the linear probability
  model fails there. The Chile 485 loan line (PAES scale) does not bind and is out. Next: more data sets with a binary outcome, selected for a
  treatment that is meaningful (a threshold someone chooses, a documented effect, a
  plausible cost, an effect that can cross the cost inside the score range); candidate
  lists go in `experiments/datasets/BINARY_OUTCOME_DATASETS_<date>.md`, and the author
  picks before anything is built.

## How we work

- **Direction:** set by the advisors (Banerjee, Ritov).
- **Agents:** Claude is the main agent, and the author assigns tasks per request. Codex
  works under the same protocol (`COLLABORATION.md`).
- **Notebook to paper:** `../manuscript/this_week.tex` is the working notebook. Once a result
  is verified and the author approves it, port it into `differing_slopes.tex`.
- **Prose:** plan first, then write one paragraph at a time (see `COLLABORATION.md`).
- **No files in the paper's text** (2026-10-21): the paper names no files, scripts,
  run directories or commits. References kept for the agents go in LaTeX comments.
- **811 outline** (2026-10-27): `../manuscript/811.tex` is the author's outline of the
  paper for the 811 writing course. A manuscript push that changes the paper's sections,
  contents or status also updates it.
- **Statistics vocabulary** (2026-11-10): the author prefers terms from statistics to
  terms from econometrics, in every manuscript file outside `prelim/`. Approved
  mapping:
  - Q is "the score" (not "running variable"; §2.1 glosses the RD term once);
  - T = gamma'X-tilde is "the linear predictor" (not "index"): "estimated linear
    predictor" (not "generated index"), "score decomposition" (not "latent-index
    decomposition"), "density fold" for the fold that fits g, rho_X and the trim
    endpoints (not "index fold");
  - "spline approximation/basis/coefficient" (not "sieve");
  - "gradient" for a derivative in gamma, else "coefficient"/"component" (not
    "loading");
  - "utility" (not "welfare"); "independent noise" (not "exogenous noise");
    "confounding" (not "endogeneity").
  Second pass (2026-11-11): "score regression" (not "first stage"; "score-regression
  fold" for I_gamma, "the error in gamma-hat"); "estimated residual / covariate" (not
  "generated residual / regressor"; "data generated under ..." stays); "units at the
  threshold" (not "marginal units"; "marginal density" stays; Proposition 1 is the
  "Threshold-derivative representation"); "set of candidate thresholds" for Phi (not
  "policy set") and "class of threshold rules" (not "policy class"). Section 1 cites
  the statistics literature on individualized treatment rules (Murphy 2003; Qian and
  Murphy 2011; Zhao et al. 2012) beside policy learning.
  Third pass (2026-11-11): the outcome-fold regression is the "partially linear fit"
  (or "partially linear regression"); "outcome model", "outcome equation" and "outcome
  fold" stay.
  Kept: "representer", "latent ability", "Error in variables" and "Policy learning"
  (keywords), and literature names such as "single-index model", "policy learning",
  "marginal policy change". LaTeX labels keep their old keys. Propose any further
  rename to the author first.

## Open questions for the author

- None open. The R3-E1 identification question was answered on 2026-11-04 (robustness
  row).

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
| 2026-11-12 | Introduction trimmed to about 1,400 printed words: proof-tools sentence moved to the §4 opening; policy learning folded into the bounds group (they need overlap); the "covariate composition" item dropped from §1's contributions list (the supporting point in "Main contribution" above is no longer listed there) | Author: "do all of those" (2026-11-12); manuscript a4bd465 |
| 2026-11-12 | Introduction and abstract rewritten (admissions running example, taxi at the end); prior-work comparisons corrected (Marinescu et al. credited and compared; policy-learning contrast is overlap; Rokkanen, Yata, Bertanha added); abstract follows the body, no symbols, alphabetical key words | Author: "Yes, fix and rewrite this whole thing ... Fix all prior work. Abstract should match the body"; "Do the same with the abstract"; "yes" (2026-11-11/12); manuscript e6da428 |
| 2026-11-11 | Second length pass from a whole-paper cut review: all main-text cuts (incl. abstract sentence 2 and the §1 confounding sentence; §2.3 trimming sentences moved to §6.2; knot rules to Appendix 2), supplement cuts except Appendix 1's list and the Appendix 2 outline; Appendix 7 kept. Main text ends on page 18 (about 10,000 printed words); PDF 65 pages. Removed text in storage/trimmed2_w*_20261111.tex | Author: "Yes, all in main. Keep lsit of appendices and the outline. Do all the others." (2026-11-11); manuscript c16103a, 3efca18 |
| 2026-11-11 | Future work and Appendix 8 removed (saved in storage/); length pass: Remarks 3-4 to Appendix 2, Section 5.3 table to Appendix 7, Section 4.1 and 6.5 discussion shortened (assumption remarks to Appendix 4), Section 1 literature tightened, Appendices 1 and 6 shortened; Appendix 7's numerical checks kept for now. Main text now ends on page 20 (about 12,000 printed words) | Author: "please remove any future work"; "trim all of those in the main and trim. G and H. keep F for now" (2026-11-11); manuscript 76983c7, 31918a5, c8030d5 |
| 2026-11-11 | Section 2.4 shortened: Proposition 2 ((U1)-(U4)) moved to Appendix 3 as Proposition C1, intuition and Remark 1 tightened, original saved in storage/; "outcome fit/regression" renamed "partially linear fit/regression" in all manuscript files outside prelim/ | Author: "Do the 2.4 stuff that you recommend. Then, rename all to partially linear regression/fit." (2026-11-11); manuscript 2d130cf, e5baf6c |
| 2026-11-11 | Second statistics-terms pass in all manuscript files outside prelim/: first stage -> score regression, generated -> estimated, marginal units -> units at the threshold, policy set -> set of candidate thresholds; cite Murphy 2003, Qian-Murphy 2011, Zhao et al. 2012 in Section 1; keep "latent ability" and the keywords | Author: "1,2,3,4. I'm fine with ability adn error in variables" (2026-11-11); manuscript 0ef0a95, 751904b |
| 2026-11-10 | Econometrics terms replaced in all manuscript files outside prelim/: index -> linear predictor, sieve -> spline, loading -> gradient/coefficient, welfare -> utility, and cutoff -> threshold, running variable -> score in the notebook, sketches and storage too (mapping under "How we work") | Author: "use linear predictor", "Do all the recommended wording changes", "implement to all files except the locked prelim ones", "Also change cutoff to threshold in those files" (2026-11-10); manuscript 5dc0f8d, merge 55c63e1 |
| 2026-11-10 | Statistics terms preferred to econometrics terms; Q is "the score" throughout (was "running variable" in seven places) | Author, 2026-11-10; manuscript 01b3fd2 |
| 2026-11-09 | Main-text material moved to the supplement (D1), originals kept in storage/; supplement now eight appendices (Appendix 8: further directions) | Log: "Follow-up: main-text material moved to the supplement (D1)" (2026-11-09) |
| 2026-11-09 | Supplement in seven appendices (old 3 split: identification; consistency and endpoint terms; extensions); unneeded old Appendix 4-5 material set aside in storage/, not deleted; (A1)(ii) and (A2) lose clauses other assumptions imply; (A4)(i) gets a sufficient-condition lemma; H stays the curvature | Log: "Submission-readiness review and pass; first E results" (2026-11-09) |
| 2026-11-11 | Binary outcomes recorded as a next-paper side track: the GLM routes, the VLBW result, and the rule that further data sets are selected for a meaningful treatment with the author choosing before a build | Log: "Binary outcomes: very low birth weight ..." (2026-11-11) and its follow-up (2026-11-11) |
| 2026-11-04 | Notation reread: renames and one name per object (outcome interval, trim window, trim endpoints, supported population); renames that touch the taxi-theory lemmas held until that port lands | Log: "Notation reread: writing pass and approved renames" (2026-11-04) |
| 2026-11-04 | R3-E1 becomes a robustness row with an identification caveat, not a co-equal use case (revises the same day's decision); J in §6 stated as fixed, no proof for an estimated J; theory-extension plan agreed | Log: "Taxi theory plan; R3-E1 is weakly identified and its score is not monotone" (2026-11-04) |
| 2026-11-04 | §2.4 rewritten around "the margin moves up"; (U1) needs log-concave f_eta only when lambda is not constant | Log: "§2.4 rewritten: the margin moves up" (2026-11-04) |
| 2026-11-04 | Taxi §6: R3-E1 (spline score, square-root fare) co-equal with R2-E1; dollar-scale S1/S2, common slope and $5–$25 sample as robustness rows; CMT an external benchmark only | Log: "Weekly scope review" (2026-11-04) |
| 2026-11-04 | Taxi theory: write out the extension of Theorem 1 for all three gaps (restricted effect block, covariate in several score columns, window capped at residual quantiles), reviewed before it goes in; the target wording covers the cap | Log: "Weekly scope review" (2026-11-04) |
| 2026-11-04 | Abstract states the novelty claim (point identification from a single deterministic threshold), folded into the root-n sentence | Log: "Weekly scope review" (2026-11-04) |
| 2026-11-04 | Two-score and dose-schedule work recorded as next-paper side tracks, worked on only when the author asks; this paper's out-of-scope list unchanged | Log: "Weekly scope review" (2026-11-04) |
| 2026-11-04 | "How we work" records the 811 rule: `811.tex` tracks the paper's sections and status | Log: "Weekly scope review" (2026-11-04); "811 course outline tracks the paper" (2026-10-27) |
| 2026-11-04 | "Threshold" throughout the paper, never "cutoff"; held notation renames applied | Log: "Held notation renames applied; \"threshold\" throughout" (2026-11-04) |
| 2026-10-27 | Taxi use case for the paper, for now: square-root fare, distance + log-distance score, effect linear in distance (R2-E1); replaces S1-E1 and S2-E1 side by side | Log: "Taxi use case: square-root fare, distance and log distance" (2026-10-27) |
| 2026-10-22 | Location-scale first stage kept as a taxi robustness check only; X independent of eta stays | Log: "Weekly scope review, short pass" (2026-10-22) |
| 2026-10-22 | Nonlinear design: reword the small-n variance excess in §5; no larger-n cells | Log: "Weekly scope review, short pass" (2026-10-22) |
| 2026-10-21 | The paper's text names no files, scripts or commits; such references become LaTeX comments | Log: "Paper text names no files" (2026-10-21) |
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
