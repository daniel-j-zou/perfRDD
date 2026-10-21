# Review of `differing_slopes.tex` (Fable 5.1, 2026-10-21)

Branch `review/fable-20261021`, base commit 469b481. Line numbers below refer to
the base file. Scope: lines 1-1490, 1735-1831, 1832-4347. Section 6 and
Appendices 4-5 were not read for review and are untouched.

Per the author's change of brief, the branch contains ONE commit on top of
469b481, the writing cleanup (a5b827b). No mechanical fix was applied; every
non-writing change is a proposal in section (b) below.

## (a) Technical findings

No blocking finding. Every proof step in Sections 2.4, 3, 4 and Appendices 1-3
was followed; the sign bookkeeping of the loadings and endpoint terms
(Steps 3-5, Lemma ds-eval, Block D), the representer identities (Lemma
ds-rout, Step 1b), the location-invariance cancellation (Lemma
ds-zero-intercept), the three channels of tau' (Lemma ds-tau-deriv) and the
rate bookkeeping in (A5) against Lemmas ds-outcome, ds-outer, ds-3a,
ds-dens-bias, ds-weighted-consistency and the product bounds were checked and
are consistent. Numerical claims that depend only on one-dimensional
quadrature were recomputed (`check_identification.py`,
`check_identification.out`, `check_bimodal_window.py` in this directory).

| id | location (base lines) | severity | confidence | finding | suggested fix |
|---|---|---|---|---|---|
| T1 | 1442-1443, 1447-1451, 1457-1471, 1483-1486 | minor | sure | Section 5.4 still reports the linear-in-eta "negative control" (table columns "Linear fit", the sentence "As a negative control we also fit ...", and the paragraph on its pseudo-target bias). `PROJECT_SCOPE.md` (decision 2026-10-20) says the linear-fit contrast is dropped and simulations are oracle-only. Numbers left untouched as instructed. | Drop the two "Linear fit" columns and the two text passages when the table is replaced by the 999-draw rerun (proposal P6). |
| T2 | 4216-4217 | minor | sure | "Section~\ref{ssec:ds-model} describes what the estimator targets when the conditional effect is not linear in X." Section 2.1 contains no such description; only the Discussion (lines 1753-1755) mentions the pseudo-target. | Proposal P1. |
| T3 | 3199-3203 | cosmetic | sure | Bimodal counterexample: the text gives the window as [-1.28, 1.28], but the quoted numbers (C_lambda = -1.557, B_eps = 1.436, tau'(0) = -0.121, Var_0 = 1.121, zeros +/-0.585) are reproduced only with the window [-1.2816, 1.2816] (the eps = 0.1 window of Section 5). With [-1.28, 1.28] quadrature gives -1.578, 1.458, -0.120, 1.120, +/-0.583. | Proposal P2. |
| T4 | 3176-3183 and 589-590 | minor | probable | Corollary "Common slope" says "No shape condition is placed on f_eta" and Section 2.4 says "no condition on f_eta is needed", but the corollary inherits the regularity of Lemma ds-tau-deriv, which asks f_eta > 0 and C^1 on [l_0, u_0]. With beta_2 = 0 only the g-route of the lemma is used, which needs f_eta positive and continuous; the C^1 requirement is not needed but is not removed. | Proposal P4. |
| T5 | 2267-2281, 2309-2311 | minor | sure | Lemma ds-rout assumes only (A1)(ii) and E\|X\|^2 < infinity, but the bound \|r_out\| <= C(1+\|X\|) in its proof uses "g is bounded on V_Phi", which is part of (A3). | Proposal P5. |
| T6 | 2648-2658 | minor | probable | Lemma ds-dens-block lists (A2)-(A4), independence of delta and L -> infinity, but the representer identity in its first term needs the hard-trim interval [phi*-u_0, phi*-l_0] inside T, i.e. (A1)(i) (used silently via Step 1b). | Proposal P11. |
| T7 | 3234 | cosmetic | sure | "Condition (eq:ds-endpoint) needs Phi containing [-7.24, 10.13]": quadrature gives the right end 10.124, which rounds to 10.12. | Proposal P3. |
| T8 | 3031 | cosmetic | sure | "The utility-score integrand is F_epsilon": F_epsilon is never defined (the paper uses F_phi and F*_epsilon). | Removed in the writing cleanup (listed under (c)); no further action. |
| T9 | 2032-2033 | cosmetic | unsure | "with smoothness only near V_Phi, this argument would give only L_n >> n^{1/6}": the reasoning behind the exponent 1/6 is not given and I could not reconstruct it (local smoothness would seem to lose the sup-norm bound O(L^{-3}) altogether, or, with the Cauchy-Schwarz route, give n^{1/7}). | Proposal P16: author to verify the exponent or drop the clause. |
| T10 | 2542-2543 | cosmetic | probable | "Replacing the inverse empirical Gram matrix costs O_p(K^{3/2}/n), which is o_p(n^{-1/2}) for K << n^{1/3}" holds for the oracle fit (Gram error O_p(K/sqrt n)). With generated regressors Lemma ds-outcome(a) gives O_p(K^{3/2}/sqrt n), so the cost is O_p(K^2/n) and needs K << n^{1/4}, which (A5) provides; the feasible case is handled through Lemma ds-3a, so there is no gap. | Proposal P14 (one clarifying clause), optional. |
| T11 | 1017-1019, 2027-2033 | cosmetic | sure | "the outcome and density spline biases, of orders K^{-7/2} and L^{-4}" is consistent with Step 2 (K^{-1/2} K^{-3} and L^{-1} L^{-3}) and with the lower limits n^{1/7}, n^{1/8}. Recorded as checked. | none |
| T12 | 3477-3482 | cosmetic | unsure | Table ds-id-checks "min_Phi tau'": my 121-point finite-difference grid gives 0.297 (displayed design) and -0.120 (opposing) against the table's 0.300 and -0.116. The differences are at the level of the grid and derivative step, not a discrepancy in the design. | none |

Checks of cited claims:

- Mukherjee, Banerjee and Ritov (2026), text copy: the score decomposition
  Q = Z'gamma_0 + eta (their eq. (1.2)); three folds with rotation and averaging
  ("by rotating the samples ... three asymptotically independent ... estimates,
  which are then averaged", p. 6); the final-fold regression carries
  Z~_i = b^'(eta^_i) Z_i with coefficient gamma^_n - gamma_0 and re-estimates b
  on that fold (their eqs. (2.5)-(2.7): "we estimate ... b(.) twice ... This
  second re-estimation of b is quite critical"). The paper's "Relation to MBR"
  paragraph (3014-3022) matches. Their K range is n^{1/8} << K << n^{1/3}
  (Remark 2.1), not used by the paper.
- Wibisono et al. (2025), text copy: heterogeneous effect alpha_0(X, eta) and
  the ATT E(alpha_0(X, eta) | Q >= tau) as target (their Section 2); Assumption
  3 asks continuity in delta of conditional functions of eta - Z'delta, which
  supports "Conditions of this kind are common in index models" (926).
- Shadrin (2001), de Boor (2001), van der Vaart (1998, Thm 5.7), van der Vaart
  and Wellner (1996, Sections 2.5-2.6), Bahadur (1966), Kiefer (1967), Ghosh
  (1971): the claims attributed to them (L_infinity boundedness of the L_2
  spline projector uniformly in the knots; cubic approximation orders and the
  Riesz property of normalized B-splines; argmax consistency; half-spaces as a
  VC/Donsker class; the Bahadur-Kiefer-Ghosh representations) agree with my
  knowledge of those works. Not checked against the sources (not available
  locally).

Numerical claims reproduced (quadrature, `check_identification.out`):
displayed design phi*_eps = 0.545103, H = -0.107534, phi* = 0.42618,
tau_eps(-3) = -1.42, tau_eps(3) = 0.92, (U4) on [-1.5, 1.5] (-0.91 < 0 < 0.41),
C_alpha, C_lambda, B_eps = 0.337, 0.058, 0.057, pi(phi*) = 0.917,
tau_out(phi*) = 0.599, gap 0.1189 / bound 0.1195, chi(l_0) = -1.215,
chi(u_0) = 0.904, dphi*/deps = 1.707 (finite difference 1.707), endpoint
condition -7.24; eps-sensitivity population columns (u_0, phi*_eps, -H) for all
seven rows; four-role check design phi*_eps = -0.11977, U'' = -0.2180, min tau'
= 0.815, channels 0.314/0.279/0.242, phi* = -0.11765, tau_out = -0.023, gap =
bound = 0.00212, dphi*/deps = -0.0248, endpoint sides -0.121 and 0.321;
nonlinear design phi*_eps = 0.152143, H = -0.219539; opposing example zeros
-2.47, -0.963, 3.97 with U = 0.0793, 0.0822, ~0 and treat-all 0.0800; bimodal
index example tied maxima at +/-0.999. Arithmetic in Section 5: sigma^2/H^2 =
0.3594/0.107534^2 = 31.08, 4 x 31.09 = 124.4, Var psi_U/H^2 = 1.254 (paper
1.256), 4B/sqrt n and B/sqrt n for B = 21.6 (0.683, 0.341, 0.171, 0.085), the
percentages quoted from Tables 2-4 (3%, 16%, 2%, 4-5%, 15-19%, 6-8%), 0.07 and
0.11 standard deviations, sevenfold variance rise, slope 1.71 between eps =
0.075 and 0.125 (unrounded 1.706). Table entries in Sections 5.2-5.4 agree
with `DS_FOUR_ROLE_COVERAGE_20261019.md` and `DS_EPS_SENSITIVITY_20261020.md`;
the pathwise check (-0.01350126, relative error 3e-11) and the 1000-replication
run (30.49, 119.4) agree with `DS_SIMPLIFICATION_CHECKS_20261013.md`.

## (b) Proposed non-writing changes (none applied)

Each item: base lines, current text, proposed text, reason. Severity and
confidence are those of the linked finding.

P1 (T2, minor, sure). Lines 4216-4217.
Current: `Section~\ref{ssec:ds-model} describes what the estimator targets when
the conditional effect is not linear in $X$. Drop`
Proposed: `We now describe what the estimator targets when the conditional
effect is not linear in $X$. Drop`
Reason: Section 2.1 contains no such description.

P2 (T3, cosmetic, sure). Line 3200.
Current: `window $[-1.28,1.28]$, a bimodal`
Proposed: `window $[-1.2816,1.2816]$ of Section~\ref{sec:ds-numerics}, a bimodal`
Reason: the quoted numbers reproduce with this window, not with +/-1.28.

P3 (T7, cosmetic, sure). Line 3234.
Current: `needs $\Phi\supseteq[-7.24,10.13]$,`
Proposed: `needs $\Phi\supseteq[-7.24,10.12]$,`
Reason: quadrature gives 10.124.

P4 (T4, minor, probable). Line 589.
Current: `With a common slope, $\beta_2=0$, no condition on $f_\eta$ is needed`
Proposed: `With a common slope, $\beta_2=0$, no shape condition on $f_\eta$ is needed`
And lines 3181-3182 (statement of Corollary cor:ds-common).
Current: `No shape condition is placed on $f_\eta$.`
Proposed: `No shape condition is placed on $f_\eta$; only its positivity and
continuity on $[l_0,u_0]$ are used.`
Reason: the corollary inherits the lemma's regularity (f_eta > 0, C^1), of which
only positivity and continuity are used when beta_2 = 0.

P5 (T5, minor, sure). Line 2269.
Current: `Suppose (A1)(ii) holds and $\E\|X\|^2<\infty$. Then`
Proposed: `Suppose (A1)(ii) holds, $\E\|X\|^2<\infty$ and $g$ is bounded on
$\mathcal V_\Phi$. Then`
Reason: the bound |r_out| <= C(1+|X|) uses boundedness of g.

P6 (T1, minor, sure). Lines 1442-1443, 1447-1451 (caption), table columns 9-10
(lines 1457-1467), 1483-1486.
Current (1442-1443): `As a negative control we also fit an outcome model linear
in $\eta$ on each arm.`
Proposed: delete the sentence.
Current (1447-1449, caption): `the rotated average with the spline outcome fit,
the oracle, and a fit linear in $\eta$.`
Proposed: `the rotated average with the spline outcome fit and the oracle.`
Current (table): the `Linear fit` column group and its two columns.
Proposed: remove them (to be done with the 999-draw rerun).
Current (1483-1486): `The linear fit has a stable pseudo-target bias of about
$-0.035$ on the threshold scale. Its coverage falls from 0.90 at $n=800$ to
0.69 at $n=6{,}400$ and 0.34 at $n=12{,}800$, as the interval shrinks around
the wrong target.`
Proposed: delete.
Reason: scope decision of 2026-10-20 (oracle-only comparison). If the author
keeps the control, Appendix 3's pseudo-target bullet (4286-4291) still
refers to the Section 6 negative control, which is outside this review.

P7 (stale todobox). Lines 1198-1201.
Current: `\todobox{Pending: the choice between normal intervals and percentile
intervals with more draws; spline sizes by constrained cross-validation instead
of the fixed exponent rule; and whether the variance excess in the nonlinear
design closes at larger $n$.}`
Proposed: `\todobox{Pending: the rerun of the coverage tables with 999-draw
percentile intervals, and whether the variance excess in the nonlinear design
closes at larger $n$.}`
Reason: the scope file records the decisions (percentile intervals with 999
draws; fixed spline-size rule with cross-validation as a check).

P8 (todobox record, optional). Lines 1038-1039.
Current: `An independent review of the written proofs on 2026-10-20 found no
errors; its minor findings (gaps in wording and notation) await the authors.`
Proposed: append `A second independent review on 2026-10-21 found no blocking
error; its findings are in the review report.`
Reason: keeps the status box current.

P9 (hardcoded cross-reference). Line 1044 and line 1296.
Current (1044): `\subsection{Limit distribution and inference}`
Proposed: `\subsection{Limit distribution and inference}\label{ssec:ds-limit}`
Current (1296): `checks the bootstrap of Section~4.4.`
Proposed: `checks the bootstrap of Section~\ref{ssec:ds-limit}.`
Reason: the number is hardcoded; adding a label is a non-writing change, hence
only proposed. (Sections 3, 4, 5.1, 5.2 and Appendices 1-3 are also hardcoded
throughout; all are currently correct.)

P10 (formula wording). Lines 1772-1773 (Future work, item 2).
Current: `the outcome through a partially linear adjustment $\beta^\top X$.`
Proposed: `the outcome through the partially linear adjustment
$X^\top\beta_1+DX^\top\beta_2$.`
Reason: beta is not a symbol of the paper; the adjustment has two slopes.

P11 (T6, minor, probable). Line 2650.
Current: `Assume (A2)--(A4), $\delta$ independent of $\mathcal I_T$ with`
Proposed: `Assume (A1)(i) and (A2)--(A4), $\delta$ independent of $\mathcal I_T$ with`
Reason: the representer identity needs the hard-trim interval inside T.

P12 (typography). Line 3227.
Current: `\item \emph{Four-role check design} (Appendix 5):`
Proposed: `\item \emph{Four-role check design} (Appendix~5):`
Reason: matches `Appendix~5` on line 3223.

P14 (T10, cosmetic, probable). Lines 2542-2543.
Current: `Replacing the inverse empirical Gram matrix therefore costs
$\Op(K^{3/2}/n)$, which is $\op(n^{-1/2})$ for $K\ll n^{1/3}$.`
Proposed: `Replacing the inverse empirical Gram matrix therefore costs
$\Op(K^{3/2}/n)$ for the oracle fit, which is $\op(n^{-1/2})$ for
$K\ll n^{1/3}$; with generated regressors the Gram error of
Lemma~\ref{lem:ds-outcome}(a) makes it $\Op(K^2/n)$, which needs $K\ll n^{1/4}$.`
Reason: clarifies which fit the rate applies to.

P15 (draft-history sentences with numbers, left in place). Lines 3240-3244
and 4284.
Current (3240-3244): `An earlier value for the first design, $-0.120064$, was
off in the fourth decimal. It came from Gauss--Hermite quadrature with the hard
trim indicator applied at the nodes, and only 12 of the 240 nodes fall inside
the window. Gauss--Legendre quadrature on the window gives $-0.1197699$
(Table~\ref{tab:ds-id-checks}).`
Current (4284): `; an earlier Gauss--Hermite computation gave $0.4955$.`
Proposed: delete both, and in the checks list (line 3459) delete `, and the
Gauss--Hermite target`.
Reason: provenance of superseded numbers; adds nothing for a reader. Not cut in
the cleanup because the passages carry numbers and are cross-referenced by the
checks list.

P16 (T9, cosmetic, unsure). Lines 2032-2033.
Current: `with smoothness only near $\mathcal V_\Phi$, this argument would give
only $L_n\gg n^{1/6}$.`
Proposed: verify the exponent, or delete the clause.
Reason: the derivation of 1/6 is not given and I could not reconstruct it.

Noted without a proposal: line 1482, "An earlier full-sample fit covered from
$n=800$; ..." respects the scope ("full-sample reuse: a simulation remark
only") and was left as is.

## (c) Writing cleanup (commit a5b827b)

Sentence-level edits only; no paragraph moved, no statement, display, number,
citation key or label changed. Diff hunks lie in lines 143-1477, 1737-1790 and
1839-4327 of the base file.

- Abstract: unchanged.
- Section 1: one sentence recast ("More importantly for threshold selection"
  -> "For threshold selection the larger limitation is").
- Section 2.1: shorter definition of D_i(phi); the three model assumptions and
  the derivation of alpha(eta) = E(W | eta) in two sentences; intercept and
  exclusion-restriction sentences tightened; "A new cutoff changes ...".
- Section 2.2: the interior-optimum sentence and the boundary-optima sentence
  recast; "our theorem" for "the stated theorem".
- Section 2.3: the trimming paragraph compressed (the two sentences saying
  that the covariate term is not a propensity weight merged); a pointer to
  Remark rem:ds-untrimmed added after "phi*_eps can differ from phi*"; the
  nuisance-region paragraph compressed ("candidate cutoffs" for "candidate
  policies"; "keeps the nuisance fits from becoming moving-set objects").
- Section 2.4: "towards" -> "toward"; "Log-concave densities cannot reverse".
- Section 3: "deliberately decoupled" dropped; the outcome-regression
  paragraph and the density paragraph tightened ("the essential point is").
- Section 4: roadmap paragraph compressed; "The proof has five steps"; items
  2 and 3 of the roadmap lightly tightened; the paragraph after the
  assumptions ends with "Uniform consistency ... is not assumed;
  Proposition 3 below derives it"; a colon in "The first-stage error is not
  negligible: it enters ...".
- Section 5: line breaks only, plus "reruns", "overcover", "much more
  precise". No number changed.
- Discussion: compressed; the sentence on hard trimming now reads "Hard
  trimming to an overlap-supported set, together with the identification of
  the two nonparametric functions alpha and b and the slopes beta_1, beta_2 on
  a fixed region, yields ..." (the base sentence said trimming "allows
  identification" of alpha and b, which the proofs do not say; the author
  should confirm the new wording). "The limit theory is an interior-optimum
  theory."
- Future work: minor tightening; "a harder direction" for "a natural and
  more challenging direction".
- Appendix 1: model paragraph and estimator-map paragraph tightened;
  data-driven spline-size paragraph and "Organization of the proofs"
  compressed.
- Appendix 2: proof-structure paragraph, Step 1 commentary, Step 1a remark
  ("the derivative of Upsilon with respect to the outcome regression" for
  "the derivative of the negative of the utility score"), Step 1b commentary,
  Step 2 opening, the sentence after Lemma ds-dens-block, and the MBR
  paragraph lightly tightened.
- Appendix 3: introduction, the paragraph after Corollary cor:ds-common,
  the reading of Proposition prop:ds-untrimmed, the Bahadur paragraph and the
  closing "Remarks on the regularity conditions" tightened.

Cuts of draft-history or signposting text (each restorable from 469b481):

1. Lines 643-645: "Earlier versions of this construction used ten roles, with a
   separate first-stage fit for each nuisance block and separate endpoint
   folds."
2. Lines 930-933: "Earlier versions of this paper assumed uniform consistency
   of the feasible criterion and of its first two derivatives as a high-level
   condition." (replaced by "Uniform consistency ... is not assumed;
   Proposition 3 below derives it from (A1)-(A7)").
3. Lines 2051-2053: "Ridge stabilization may be used in numerical work when the
   empirical Gram matrix is ill conditioned. It is not part of the asymptotic
   estimator." (duplicate of lines 1939-1941, which stay).
4. Line 3031: "The utility-score integrand is F_epsilon, and" (undefined
   symbol; see T8).
5. Line 4329: "Earlier versions assumed two conditions at a high level."
   (replaced by "Two conditions that could be assumed at a high level are
   derived instead", and "now imposed" -> "imposed").
6. Lines 2446-2447: "This is the population--sample versus approximation
   distinction, written explicitly." (signposting).
7. Lines 2076-2078: "retained as cross-reference labels, not as an instruction
   to read the algebra alphabetically" -> "are cross-reference labels, not a
   reading order".

Not cut, by design: the draft-history remarks that carry numbers (3240-3244,
4284; see P15), and the todoboxes.

## (d) Todoboxes

- Lines 1024-1042 (proof status): current as of 2026-10-20; P8 offers an
  optional one-line update for this review.
- Lines 1198-1201 (Section 5 pending): STALE. Two of its three items are
  decided in the scope file (percentile intervals with 999 draws; fixed
  spline-size rule, cross-validation as a check). See P7.
- Lines 2701-2717 (original route): a draft-only record that the scope file
  says to keep; not stale.

## (e) Not checked

- The simulation tables themselves (not re-derived, as instructed); only their
  internal arithmetic and agreement with the three summary files.
- The role-score variances 0.0145, 0.2181, 0.0434, 0.0834 and B = 21.6 (need
  the representers and the second-order expansion; only their arithmetic
  consequences were checked).
- The constants theta_J = 1.1e-3 and 1.9e-4 (Lemma ds-gram text), Table
  ds-outcome-mc, and the pseudo-target projections (0.408, 0.471 and the
  quadrature values): need simulation or multi-dimensional computation.
- Claims about Shadrin (2001), de Boor (2001), van der Vaart (1998), van der
  Vaart and Wellner (1996), Bahadur, Kiefer, Ghosh: assessed from knowledge
  only, not against the sources.
- The bounded-variation approximation rate O(K^{-1/2}) (lines 2317-2318) and
  the O(L^{-1}) quasi-interpolant bound for the jump representer: standard
  but not checked against a source.
- The uniform Bahadur argument of Block D was read for consistency of the
  signs and the use of equicontinuity, not re-derived line by line.
- Section 6 and Appendices 4-5 (out of scope), including the Appendix 3
  cross-references into them (ssec:ds-sim, Appendix 4, Appendix 5), which
  resolve but were not checked for content.

## (f) Commits and build

- 469b481  base (unchanged).
- a5b827b  "Review: writing cleanup (Fable 5.1)" (the only commit on the
  branch; no mechanical-fix commit was made).
- Build after a5b827b: `latexmk` exit 0; 14 warnings (the float overfull
  vboxes of the base build), 0 overfull hboxes, no undefined or multiply
  defined references, no errors. Logs: `build_base.log`, `build_after.log`;
  LaTeX log in `../fable_build/differing_slopes.log`.
- Nothing pushed; nothing outside the worktree and this directory touched.

## (g) Second review (Fable 5.1, extra-high effort) of the uncertain findings

An independent reviewer answered the five findings marked probable or unsure,
posed as questions without the first reviewer's conclusions (base file 469b481).

| question | finding | verdict | confidence | outcome |
|---|---|---|---|---|
| Q1 | T4, f_eta regularity for beta_2 = 0 | partly valid: the case uses f_eta > 0 on [l_0,u_0] (strictness of C_alpha, p_eps > 0), not its differentiability; "No shape condition" in the corollary is accurate, "no condition on f_eta is needed" in §2.4 overstates | sure | §2.4 and the corollary reworded |
| Q2 | T6, (A1)(i) in Lemma ds-dens-block | partly valid: the inclusion of the trim interval in T is used, but it is implied by (A3); made explicit for consistency with Lemma ds-weighted-consistency | probable | (A1)(i) added |
| Q3 | T9, "L_n >> n^{1/6}" in Appendix 1 | partly valid: 1/6 is the exponent of an O(L^{-3}) bias from the direct bound ||r||_{L_1} sup_V|P_L h - h|, not of the L_1 x L_infinity argument named; the local sup-norm bound would also need the locality of the spline projector | probable | clause dropped |
| Q4 | T10, Gram replacement cost in Step 2 | no gap: Step 2 is the oracle calculation; the fit at generated residuals is covered by Lemmas ds-3a and ds-outer at the same order | sure | clarifying clause added |
| Q5 | T12, min_Phi tau' in Table ds-id-checks | valid: the minimum is at phi = +-3; the table's values are tau' at +-2.95 | sure | corrected |

Further check (Claude, `logistic_min_tau.py` in the review scratch directory): the
logistic design's minimum is 0.321 at phi = -3 (table had 0.323 = tau'(-2.95)); the
four-role check design's 0.815 is correct. The same setup reproduces the table's
phi*_eps = 0.25486 for the logistic design. The script that made the table
(`experiments/scripts/ds_identification_check.py`) probably evaluates tau' on a
grid that stops short of the ends of Phi; not changed.

## (h) Author decisions (2026-10-21)

Writing cleanup merged (a5b827b). Proposals approved, numbered as in the chat list:
1 (cleanup), 2 (Discussion wording), 3 (P-table values), 4 (P2), 5 (P3), 6 (P4,
with the second reviewer's wording), 7 (P5), 8 (P11), 9 (P16, clause dropped),
10 (P1), 11 (P14, second reviewer's wording), 12 (P10), 13 (P9), 14 (P12), 15 (P15),
16 (P7), 17 (P8, with the stale "await the authors" line corrected). Item 18 (P6,
the §5.4 linear-fit comparison) is deferred to the 999-draw rerun. Applied in
manuscript 51f5dbb.
