# Review of `differing_slopes.tex` (Fable 5.1, 2026-10-27)

Branch `review/fable-20261027`, base commit e4fd7ae. Line numbers below refer to
the base file. Scope: the whole paper, front matter, Sections 1-7 and Appendices
1-5. Sections 6 and Appendices 4-5, unread on 2026-10-21, were read in full.

The branch contains ONE commit on top of e4fd7ae, the writing cleanup (7c38869).
No mechanical fix was applied; every non-writing change is a proposal in
section (b). Nothing was pushed.

## (a) Technical findings

No blocking finding. The proofs of Sections 2.4, 3, 4 and Appendices 1-3 were
followed again step by step after the 2026-10-21 corrections: the
marginal-effect representation and its two derivative forms (Lemma ds-tau-deriv,
both tilting routes and the Leibniz boundary term), the covariance inequalities
of Propositions ds-aligned and ds-opposing, the Gaussian identity (eq:ds-gauss)
and the Brascamp-Lieb step of the bimodal example, the implicit-function
derivative of Corollary ds-eps (signs of dl_0/d eps and du_0/d eps included),
Proposition ds-untrimmed, the identification lower bound and Gram bounds of
Lemma ds-gram (constants kappa_id, c_Q, C_Q), the closed-form outcome
representer (both test directions, membership in H_J, the bounded-variation
rate), the density representer identities, the bias and remainder orders of
Step 2 against (A5), the conditional-mean argument of Lemmas ds-dens-mean and
ds-dens-block, the rates of Lemmas ds-outcome, ds-outer and ds-3a (every
exponent recomputed; the approximation piece of Lemma ds-outer is in fact
K^{-5/2}n^{-1/2}, smaller than the stated K^{-2}n^{-1/2}, which comes from the
fit-error piece, so the statement is consistent), the sign bookkeeping of
Steps 4-5 and (eq:ds-Abar), the location-invariance cancellation, the
uncorrelatedness proof, the rotation corollary, the second-order bias
arithmetic of Remark ds-bias, the evaluation-fold lemma (integration by parts
of the density shift, the window term) and the Bahadur block (read for
consistency, not re-derived). All consistent with the assumptions they cite.
The newly read parts (Sections 6, Appendices 4-5) contain no proofs; their
numbers were checked against the run summary and by quadrature.

| id | location (base lines) | severity | confidence | finding | suggested fix |
|---|---|---|---|---|---|
| T1 | 1462-1464, 1183-1184 | minor | sure | Section 5.4 says "whether this gap closes at larger $n$ is not settled" and the todobox repeats the question. The research log (2026-10-21, "the analytic variance is right; the excess is a small-n density-basis effect": control-variate estimates of $n\Var$ are 5.39, 5.14, 4.74, 4.71 at $n$ = 3,200 to 25,600, so the 5.63 at 12,800 is Monte Carlo noise) and the author's decision of 2026-10-22 (reword the small-$n$ excess, add no larger-$n$ cells) settle it. Scope drift: a recorded decision not yet applied to the text. | Proposal P1. |
| T2 | 4346, 4419-4423, 4589-4592 | minor | sure | Appendices 4 and 5 name scripts, run directories and the research log in the text (`differing_slopes_full_pipeline.py`, `code/experiments/runs/`, `code/RESEARCH_LOG.md`, the three `four_role_*` scripts). The scope rule of 2026-10-21 says the paper names no files; the move of that date explicitly left Appendices 4-5 out. | Proposal P2. |
| T3 | 1424-1425, 1429-1431, 1440-1449, 1465-1468 | minor | sure | The linear-in-$\eta$ negative control is still in Section 5.4 (T1/P6 of the 2026-10-21 review, deferred by the author to the 999-draw rerun). Not a new finding; listed because it remains open against the scope file (oracle-only simulations). | Pending the rerun; no new proposal. |
| T4 | 1568-1571, 1535-1538 | minor | probable | The taxi window is capped at the central 99% of $\hat\eta$, re-estimated in every fit. The endpoints of the estimand are then quantiles of the fitted residual, not the quantiles of the fitted index that (A6) and Block D expand; the endpoint scores and loadings ($z_p$, $m_p$, $b_l$, $b_u$) are different objects. The same half-space argument should cover them, but the theory-gap sentence of Section 6.2 names only the two-column score. | Proposal P3. |
| T5 | 1652, 1625-1626 | minor | sure | Table 7 rows $\epsilon=0.075$ and $0.10$: with the 99% cap in place, those windows keep 84% and 40% of rides (run summary `window_share` 0.840 and 0.403, against 0.990 for the main row). The rows change the target population substantially, and the text presents them as design variations that move the estimate by at most 21 cents. Appendix 3 (line 3380) itself recommends reporting the trimmed share. | Proposal P4. |
| T6 | 1730-1733, 4381-4407, 4435-4475 | minor | sure | Section 6.6 says the full robustness battery (regularization, non-Gaussian errors, weak curvature, boundary optima, clustering) "remains in Appendix 4". Appendix 4's Table 9 lists those designs and replication counts but reports no result for them, and Appendix 5's Table 10 lists six diagnostic questions of which only two (non-Gaussian $T$, spline density) have reported results. The paper promises results it does not contain. | Proposal P5. |
| T7 | 4450-4452, 4478 | cosmetic | sure | Appendix 5's $t_5$ design: the stated target 0.587810 reproduces only with a $t_5$ index scaled to unit variance (a raw $t_5$ index gives 0.642004; `check_numbers.out`). The text attaches "standardized" to the mixture only. | Proposal P6. |
| T8 | 4357-4366, 1693-1694 | cosmetic | sure | Appendix 4 quadratic design: on the search interval $[-3,3]$, $\tau_\epsilon$ has a second, downward zero at $-2.385$ (a local minimum of $U_\epsilon$); 0.484416 is the global maximizer on $[-3,3]$, but (A1)(iii) fails on that $\Phi$. Also, with $\Phi=[-3,3]$ and the spline density on $[-3,3]$, the argument set $[-4.28,4.28]$ is not inside $\mathcal T$, so (A1)(i) fails for the spline variants. Legacy pipeline, outside the theorem; worth one sentence. | Proposal P7 (optional). |
| T9 | 434 | cosmetic | sure | "Identification is discussed in a later section." is a vague pointer; the identification of the optimum is Section 2.4, which follows immediately, and the stacked-regression bound is Lemma ds-gram. | Proposal P8. |
| T10 | 1286-1288 | cosmetic | sure | "with that many [199 draws], even an exact normal bootstrap has percentile coverage of about 0.940": reproduces (0.9407) for the interpolated percentile (`numpy.percentile` default); the order-statistic interval (5th, 195th) would cover 0.950. Consistent with the code's convention; recorded as checked. | none (optionally "interpolated percentile"). |
| T11 | 1509-1511 | cosmetic | probable | "it only chooses the scale on which the covariates and $\eta$ are taken to be independent": the scale also fixes where the score is linear in the covariates (the first stage), which is the other maintained assumption that depends on it. | Proposal P9. |
| T12 | 4361-4364, 4561-4563 | cosmetic | sure | The Gauss-Hermite provenance remarks ($-0.120064$, $0.4955$, bias shift $2.9\times10^{-4}$) survive twice in Appendices 4 and 5; the same remark was cut from Appendix 3 on 2026-10-21 (P15 there). Appendix 5's tables report variances only, so the remark is moot there. | Proposal P10. |
| T13 | 1595-1596 | cosmetic | unsure | "so the deployed cutoff is too high" reads as a policy statement; the scope file says the application's purpose is to show the method, not to make a policy claim. The sentence is about the estimate and sits next to its interval, so this may be acceptable. | Proposal P11 (optional). |

Checks of cited claims:

- Mukherjee, Banerjee and Ritov (2026) and Wibisono et al. (2025), text
  copies: the sentences that cite them (lines 149-154, 909-910, 2071-2072,
  3013-3021) are unchanged since the 2026-10-21 check, which found them
  consistent with the sources (score decomposition, three-fold rotation,
  final-fold re-estimation of $b$; heterogeneous $\alpha_0(X,\eta)$, ATT
  target, continuity in $\delta$ of index-conditional functions). Not
  re-checked.
- Haggag and Paci (2014) (Section 6: VTS menu \$2/\$3/\$4 below \$15 and
  20/25/30% from \$15, CMT 15/20/25% at every fare, the \$5-\$25 restriction,
  daytime rides, no tolls or surcharges, the 2009 meter of \$2.50 plus \$0.40
  per fifth of a mile or two minutes): knowledge only; agrees with my
  recollection of the paper and of the 2009 TLC rate card. Not available
  locally.
- Eckles et al. (2025), Carneiro, Heckman and Vytlacil (2010), Marinescu et al.
  (2022), Dong and Lewbel (2015), Ben-Michael et al. (2025), Zhang, Ben-Michael
  and Imai (2022), Crump et al. (2009), Yang and Ding (2018), Shadrin (2001),
  de Boor (2001), van der Vaart (1998), van der Vaart and Wellner (1996),
  Bahadur, Kiefer, Ghosh: knowledge only, as on 2026-10-21; no inconsistency
  noticed.

Numerical claims reproduced (`check_numbers.py`, `check_numbers.out`, one
process, one-dimensional quadrature):

- Displayed design: $\phi_\epsilon^*=0.545103$, $H=-0.107534$,
  $\Var\psi_U=0.01452$ (paper 0.0145), $\Var\psi_T=0.04339$ (paper 0.0434; new
  this time, using the Gaussian conditional law of $X$ given $T$),
  $\Var\psi_U/H^2=1.2557$ (paper 1.256), $\sigma^2/H^2=31.08$ from the four
  printed variances (paper 31.09), $4\sigma^2/H^2=124.3$ (paper 124.4).
- Nonlinear design (Section 5.4): $\phi_\epsilon^*=0.152143$, $H=-0.219539$,
  $\Var\psi_U=0.001809$, oracle $\Var\psi_U/H^2=0.0375$ (paper 0.0375),
  $\Var\psi_T=0.008753$ (research log 0.008753), $\sigma^2/H^2=4.729$ from the
  log's four variances (paper 4.73).
- Appendix 5 common-slope baseline: $\phi_\epsilon^*=0.73129$ (paper 0.7313),
  $H=-0.07202$ (paper $-0.0720$), $\Var\psi_U=0.01846$ (paper 0.0185),
  $\Var\psi_T=0.03628$ (paper 0.0363), $\sigma^2$ sum 0.2266 (paper 0.2265),
  $4\sigma^2/H^2=174.8$ (paper 174.7), $\sigma^2/H^2=43.71$ (paper 43.68);
  table ratios 4.13/4.33/4.06 against printed 4.14/4.32/4.06 and
  3.88/4.07/4.10/4.08/3.97/3.98 against printed 3.88/4.07/4.11/4.08/3.98/3.98,
  i.e. within rounding of unrounded inputs.
- Appendix 4: baseline target $-0.119770$ with $H=-0.2180$ (Table 8 of
  Appendix 3), window $\pm1.28155$, quadratic-design target 0.484416 (global
  maximizer on $[-3,3]$); $t_5$ target 0.587810 with the unit-variance $t_5$
  index (T7).
- Section 5 arithmetic: Table 1 ratios 4.01 and 4.13, deviations $-2.5\%$ and
  $-3.2\%$ from 31.09; bias 0.074 sd; $4B/\sqrt n=0.683, 0.342$ and
  $B/\sqrt n=0.171, 0.085$ for $B=21.6$; combination biases 0.183 and 0.063;
  Table 2 ratios 1.163, 0.983, 1.019 and coverage standard errors 0.0097,
  0.0154; Table 3 slope 1.70 (paper 1.71 from unrounded 1.706), sevenfold rise
  7.13, 0.106 sd, the seven $n=64{,}000$ deviations $3.9$-$4.8\%$ below $V$
  against a Monte Carlo standard error of 4.5%, step standard deviations
  0.0462 and 0.0216; Table 4 ratios 1.161, 1.152, 1.190, standard errors 6.3%
  and 8.2%, 0.21/\sqrt{12800}=0.0019.
- 199-draw percentile coverage of an exact normal bootstrap: 0.9407
  (interpolated), see T10.
- Section 6 against `experiments/runs/taxi_section6/summary.json` (read only):
  $\hat\phi_\epsilon=12.8666$, cell 12.90, pooled 12.873, rotations
  12.579-13.182, 95% percentile interval [12.562, 13.175], 527+465=992 of 999
  draws in the 12.90 and 13.30 cells, no draw at the boundary, all below \$15,
  gain 0.650 cents, $R^2$ 0.0926 and 0.2032, spread ratio 1.480, own-score cell
  12.90, $\epsilon=0.075/0.10$ estimates 13.004/13.078, $\mathcal J$-tail
  variations 12.880-12.937, joint variations 12.920/12.926, fold seeds
  12.866-12.889, common slope 6.000 at the boundary with no interior local
  maximum, dollar-scale 11.870 and 11.764 with intervals [11.333, 12.512] and
  [11.292, 12.206], dollar-scale independence 0.026/0.064/2.30; $\phi_0$ on the
  $Q$ scale $5.746=2(\sqrt{15}-1)$. The fare grid \$2.50 + \$0.40k puts every
  printed estimate in the printed cell (12.87 to 12.90, 13.00 and 13.08 to
  13.30, 12.94 to 13.30, 11.87 and 11.76 to 12.10, 6.00 to 6.10, 13.15 to
  13.30), and 15 lies between the grid points 14.90 and 15.30. The 21-cent
  maximum move and 11.0% treated share are consistent across the section.
- Scope consistency: notation (latent ability, residual, $\alpha(\eta)$,
  (A1)-(A7), no DS tags, $\Upsilon$), no em dashes, no "extension" or
  "correction" framing of differing slopes, common slope only as the
  $\beta_2=0$ case, the R2-E1 taxi specification, the fixed spline-size rule,
  percentile intervals with 999 draws in Section 6, full-sample reuse as a
  simulation remark only: all as the scope file records. The exceptions are
  T1 (an unapplied decision), T2 (file names in Appendices 4-5) and T3 (the
  deferred negative control).

## (b) Proposed non-writing changes (none applied)

Each item: base lines, current text, proposed text, reason. Severity and
confidence are those of the linked finding.

P1 (T1, minor, sure). Lines 1462-1464.
Current: `The $n$-scaled variance stays 15--19\% above 4.73 up to $n=12{,}800$,
with Monte Carlo standard error 6--8\%; whether this gap closes at larger $n$
is not settled.`
Proposed: `The $n$-scaled variance is 15--19\% above 4.73 in these rows, with
Monte Carlo standard error 6--8\%. A control-variate estimate on 2000 further
data sets per $n$ puts it at 1.14, 1.09 and 1.00 times 4.73 at $n=3{,}200$,
$6{,}400$ and $12{,}800$: the excess is a small-sample effect of the density
basis, which the rule fixes at 8 functions for $n\le6{,}400$, and the 5.63 at
$n=12{,}800$ is Monte Carlo noise.` (Numbers from the research log entry of
2026-10-21; the author chose the "reword" option on 2026-10-22.)
And lines 1183-1184 (todobox).
Current: `\todobox{Pending: the 999-draw rerun of the coverage tables, and
whether the variance excess in the nonlinear design closes at larger $n$.}`
Proposed: `\todobox{Pending: the 999-draw rerun of the coverage tables.}`
Reason: the decision of 2026-10-22 is not yet in the text.

P2 (T2, minor, sure). Lines 4345-4347.
Current: `The simulations in Subsection~\ref{ssec:ds-sim} are generated by
\texttt{experiments/scripts/differing\_slopes\_full\_pipeline.py}.  The
baseline design is`
Proposed: `The simulations in Subsection~\ref{ssec:ds-sim} use the baseline
design` followed by the display, with the script name moved to a LaTeX comment.
Lines 4419-4423.
Current: `The JSON and CSV run records are written under
\texttt{code/experiments/runs/}. The scenario definitions, estimator variants,
and diagnostic calculations are all in the runner cited above.  The summary
values reported in the main text are additionally recorded in the dated
simulation entries of \texttt{code/RESEARCH\_LOG.md}.  No bootstrap validity
claim is made.`
Proposed: `No bootstrap validity claim is made.` with the three deleted
sentences moved to a LaTeX comment.
Lines 4589-4592.
Current: `Sources: \texttt{code/experiments/scripts/four\_role\_shared\_gamma\_check.py},
\texttt{four\_role\_rotation\_average\_check.py} and
\texttt{differing\_slopes\_four\_role\_check.py}, with the saved results under
\texttt{code/experiments/runs/}.`
Proposed: the paragraph as a LaTeX comment.
Reason: scope rule "the paper's text names no files, scripts or commits".

P3 (T4, minor, probable). Lines 1535-1538.
Current: `The theory of Section~\ref{ssec:ds-conditions} is stated for the
unrestricted block and for a score in which each covariate enters once, while
distance enters here through two columns. We expect it to carry over with
notational changes, but we have not written that out.`
Proposed: `The theory of Section~\ref{ssec:ds-conditions} is stated for the
unrestricted block, for a score in which each covariate enters once, while
distance enters here through two columns, and for trim endpoints that are
quantiles of the fitted index, while the window of
Section~\ref{ssec:taxi-design} is capped at quantiles of the fitted residual.
We expect it to carry over with notational changes, but we have not written
that out.` (Section 6.3 has no label; `ssec:taxi-design` would have to be
added, or "the window below" used instead.)
Reason: the capped window changes the endpoint objects of the expansion.

P4 (T5, minor, sure). Lines 1625-1626, after `The design variations move the
estimate by about 21 cents at most.`
Proposed addition: `At $\epsilon=0.075$ and $0.10$ the window keeps 84\% and
40\% of rides, against 99\% in the main specification, so those two rows change
the target population as well as the estimate.`
Reason: the run summary's `window_share`; Appendix 3 recommends reporting the
trimmed share.

P5 (T6, minor, sure). Lines 1730-1733 (after the cleanup, the sentence reads
`The full robustness battery, with regularization, non-Gaussian errors, weak
curvature, boundary optima and clustering, remains in Appendix~4; the main
simulation message does not need it.`).
Proposed: `Appendix~4 lists the designs of a wider robustness battery, with
regularization, non-Gaussian errors, weak curvature, boundary optima and
clustering; their results are not reported here.`
And Appendix 5, lines 4430-4433: either report the four unreported rows of
Table 10 (bootstrap under the ten-block split, support choices, structural
violations, optimization stability) or cut them, keeping the two rows with
results. Appendix 4's Table 9 likewise: keep only the rows whose results the
paper reports, or add the results.
Reason: the text promises results the paper does not contain.

P6 (T7, cosmetic, sure). Lines 4450-4452.
Current: `$t_5$ and a standardized two-component normal mixture, with the
population target computed from the known density.`
Proposed: `$t_5$ scaled to unit variance and a standardized two-component
normal mixture, with the population target computed from the known density.`
Line 4478.
Current: `the exact $t_5$ target is $0.587810$`
Proposed: `the exact target for the unit-variance $t_5$ index is $0.587810$`
Reason: the target reproduces only for the unit-variance $t_5$ index.

P7 (T8, cosmetic, sure; optional). After line 4366.
Proposed addition: `On $[-3,3]$ the marginal effect of the quadratic design also
crosses zero downwards at $-2.38$, a local minimum of $U_\epsilon$; $0.484416$
is the global maximizer on that interval, and the single-crossing condition
(A1)(iii) does not hold there.`
Reason: quadrature (`check_numbers.out`).

P8 (T9, cosmetic, sure). Line 434.
Current: `Identification is discussed in a later section.`
Proposed: `Section~\ref{ssec:ds-unique} gives conditions for a unique interior
optimum; the stacked-regression condition on $\mathcal J$ is
Lemma~\ref{lem:ds-gram} in Appendix~3.`
Reason: a precise pointer in place of "a later section".

P9 (T11, cosmetic, probable). Lines 1509-1511.
Current: `it only chooses the scale on which the covariates and $\eta$ are
taken to be independent.`
Proposed: `it only chooses the scale on which the score is linear in the
covariates and $\eta$ is independent of them.`
Reason: both maintained assumptions depend on the scale.

P10 (T12, cosmetic, sure). Lines 4361-4364.
Current: `An earlier Gauss--Hermite routine applied the trim indicator at its
nodes and gave $-0.120064$ and $0.4955$. The saved baseline runs use
$-0.120064$, which shifts their reported biases by $2.9\times10^{-4}$.`
Proposed: delete.
Lines 4561-4563.
Current: `The saved runs used $-0.120064$ from a Gauss--Hermite rule. That
changes their bias by $2.9\times10^{-4}$, which the table does not report, and
leaves the variances in the table unchanged.`
Proposed: delete.
Reason: provenance of superseded numbers, as cut from Appendix 3 on
2026-10-21 (P15 there); the Appendix 5 table reports variances only.

P11 (T13, cosmetic, unsure; optional). Lines 1595-1596.
Current: `so the deployed cutoff is too high.`
Proposed: `so the estimate lies below the deployed cutoff.`
Reason: the scope file's "no policy claim".

Noted without a proposal: line 1464, "An earlier full-sample fit covered from
$n=800$" (scope: a simulation remark only), left as is; the commented
proof-status box (1007-1027) could record this review in one line if the
author keeps the box.

## (c) Writing cleanup (commit 7c38869)

Sentence-level edits only; no paragraph moved, no statement, display, number,
citation key, label, figure or table content changed. Diff hunks lie in lines
390-391, 1684-1689, 1726-1733, 4412-4414, 4497-4501 and 4512 of the base file.

- Front matter, Abstract: unchanged (author-set on 2026-10-27).
- Section 1: nothing to clean.
- Section 2.2: the boundary-optima sentence recast ("optimal to provide
  everyone or noone treatment" was a typo and a long sentence): "Boundary
  optima are meaningful choices: at a given cost $c$ it may be optimal to
  treat everyone or no one. Their inference differs ..."
- Sections 2.3-2.4, 3, 4, 5: nothing worth cleaning after the 2026-10-21 pass;
  the vague pointer at line 434 is a cross-reference change (P8), not an edit.
- Section 6.1-6.5: read in full; the prose is already in the author's register
  after the two §6 review rounds. No edit.
- Section 6.6: the two "hypothesis" sentences merged into "First, when ...
  should recover its utility target, whereas ... Second, as a negative
  control, if ... should display the projection bias rather than conceal it";
  "The results are the pattern suggested by the application" became "The
  results match the pattern the application suggests"; the last sentence
  shortened, keeping its claim (see T6/P5 for the claim itself).
- Section 7, Future work: unchanged.
- Appendices 1-3: unchanged.
- Appendix 4: "It intentionally excludes ... which is why it is described as
  conditional" became "It excludes the generated-index and estimated-endpoint
  terms, which is why it is called conditional".
- Appendix 5: "retaining decoupling within each role assignment" became
  "keeping the roles decoupled within each assignment"; "Optional ridge fits
  remain numerical stabilization diagnostics rather than additional theorem
  cases" became "Ridge fits are numerical stabilization diagnostics, not
  additional theorem cases"; "Two existing simulations" became "Two
  simulations".

No draft-history text was cut this time; the remaining provenance remarks
carry numbers (P10) and the file references are a scope matter (P2).

## (d) Todoboxes

- Lines 1183-1184 (Section 5 pending): first clause current (the 999-draw
  rerun is still on the task board); second clause STALE, settled by the
  2026-10-21 finding and the 2026-10-22 decision (T1, P1).
- Lines 1007-1027 (proof status): commented out, so not rendered; current as
  of 2026-10-21.
- Lines 2699-2716 (original route): a draft-only record that the scope file
  says to keep; not stale.
- Lines 1817-1818 (acknowledgements): commented out; for submission.

## (e) Not checked

- $\Var\psi_o$ and $\Var\psi_\gamma$ in every design (need the outcome
  representer's Schur complements and the summed loading), hence the printed
  0.2181, 0.0834, 0.1700, 0.0018 and $B=21.6$; only their arithmetic
  consequences were checked. $\Var\psi_U$ and $\Var\psi_T$ are now checked.
- The constants $\vartheta_{\mathcal J}$ ($1.1\times10^{-3}$,
  $1.9\times10^{-4}$), Table ds-outcome-mc, the pseudo-target projections
  (0.408, 0.471, 2.01, 2.13), the simulation tables themselves beyond their
  internal arithmetic, and the Appendix 5 mixture target 0.793023 (the mixture
  is not specified in the text) and RMSE sequences.
- The taxi computations beyond the saved run summary: the data were not
  reprocessed, the figure was not regenerated, and the 999 draws were not
  reread. The 2026-10-27 §6 review did those checks.
- Claims about Haggag and Paci (2014) and the other works listed under "Checks
  of cited claims": knowledge only.
- The Bahadur block (Lemmas prelim-mixed-projection, prelim-uniform-bahadur,
  prelim-boundary) was read for consistency of signs and the use of
  equicontinuity, not re-derived line by line, as on 2026-10-21.
- The bounded-variation approximation rate $O(K^{-1/2})$ and the $O(L^{-1})$
  quasi-interpolant bound for the jump representer: standard, not checked
  against a source.
- Whether the half-space argument of Block D extends to quantiles of the
  fitted residual (T4): not attempted.

## (f) Commits and build

- e4fd7ae  base (unchanged).
- 7c38869  "Review: writing cleanup (Fable 5.1)" (the only commit on the
  branch; no mechanical-fix commit was made).
- Build before (e4fd7ae): `latexmk` exit 0; 2 LaTeX warnings (the
  `prelim/biometrika` class and `prelim/ntheorem` package path notices), 0
  overfull hboxes, 114 overfull vboxes (all "while \output is active", the
  class's page boxes), no undefined or multiply defined references, no errors.
  Logs: `build_base.log`, `build_base/differing_slopes.log`.
- Build after (7c38869): identical counts: exit 0, 2 warnings, 0 overfull
  hboxes, 114 overfull vboxes, no undefined references. Logs:
  `build_after.log`, `build/differing_slopes.log`.
- Scripts and outputs in this directory: `check_numbers.py`,
  `check_numbers.out`.
- Nothing pushed; nothing outside the worktree and this directory touched.

## (g) Second review (Fable 5.1, extra-high effort) of the uncertain findings

An independent reviewer answered the findings marked probable or unsure (T4, T11, T13),
posed as questions without the first reviewer's conclusions (base file e4fd7ae).

| question | finding | verdict | confidence | proposed outcome |
|---|---|---|---|---|
| Q1 | T4, taxi trim window vs the theory's endpoints | valid. At eps = 0.05 the index-fold eps-endpoints never bind, so the window is the central 99% of eta-hat on the evaluation fold. Lemma ds-eval's premise (endpoints independent of I_U) fails, and the endpoint terms become trimmed-mean terms in eta-hat with first-stage loading e_1 instead of the index-quantile terms of Block D. u0 = 3.497 also lies above J (2.065), so both endpoints are outside J. | sure | Third gap named in the 6.2 theory sentence; 6.3 says both endpoints lie outside J and the eps-endpoints never bind |
| Q2 | T11, "only chooses the scale on which ... independent" | valid. The scale also fixes what eta is: where the index is linear, where the effect and baseline are additive in eta, and which rides the window selects. The untrimmed optimum is scale-free; the supported target is not. | sure | Reworded |
| Q3 | T13, "the deployed cutoff is too high" | partly valid. It is a correct reading of the point estimate, but it is an unqualified verdict, tau-hat(15) has no interval, and the effect below $15 is extrapolated. | probable | Conditioned on the model and tied to the bootstrap interval |

Check script: second/check_window.py.

## (h) Author decisions (2026-10-27)

"all, with 6a". Applied in manuscript 7c38869 (writing cleanup) and 4271d93 (proposals),
using the second reviewer's wording for Q1 (T4, the window gap), Q2 (T11, the scale
sentence) and Q3 (T13, the "too high" sentence).

Under 6a, the rows without reported results are cut: Appendix 4's replication table
keeps the baseline and quadratic designs, and Appendix 5's table keeps the non-Gaussian
index and spline-density questions. Two knock-on edits follow from 6a:
- the §6.6 sentence on the robustness battery is removed rather than reworded as in P5,
  since its designs are cut;
- the sentences describing the clustered design (Appendix 4) and the support,
  curvature and error-law diagnostics (Appendix 5) are removed.

T3 (the §5.4 linear-fit comparison) remains deferred to the 999-draw rerun. Changelog
in manuscript (commit after 4271d93).
