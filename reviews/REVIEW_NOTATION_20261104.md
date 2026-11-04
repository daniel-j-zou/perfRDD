# Notation-focused reread of differing_slopes.tex (2026-11-04)

Reviewer: Fable 5.1 (high), read-only, on manuscript c1474a8; writing edits committed by the
reviewer as 862bd53, rebased by Claude onto 0631229 as 326ea6e and pushed (8deaf84).
Author decision on the proposals: "Yes, all" (2026-11-04). Applied in manuscript 245dbf9;
items touching the taxi-theory lemmas are held on the board. Line numbers below refer to
c1474a8. Item N12 (\Psi(\delta)) and N9 (generic functional L) had already been resolved by
the proof-shortening pass (5e294ad, 54bb36b) before this pass ran.

Second reviewer (Fable 5.1, xhigh) on item (e)2, (A3) smoothness: f_eta needs only C^2 (one
use of a bounded f_eta'', Lemma ds-eval, density shift); alpha and b do need a bounded third
derivative (the K^{-3} spline bias behind the n^{1/7} lower limit on K). Author: yes
(2026-11-04); applied in manuscript e23fed5.

Held items applied in manuscript 558099d (after 4c715f8 and 943cdc3). A second Fable 5.1
(high) review of that diff found every changed site on the intended object, no new clashes,
and five leftovers outside the script's patterns (bare F, partial F in Block D and Lemma
ds-zero-intercept; a bare Gram Q in Lemma ds-outcome, orphaned by removing the local
shorthand), all fixed before the push; it also suggested defining ell_2 once, done. N3
(1_{a,q}) and N13(i) were no longer needed: the Block D rewrite removed H_{a,q} and states
hat Q_p = hat q_p(hat gamma). Item N16's "cutoff" vs "threshold": author decision "use
threshold" (manuscript 4067554).

---

Worktree: branch `review/notation-20261104`, base c1474a8. Writing-edit commit: **862bd53** ("Review: notation-focused reread, writing edits (Fable 5.1)"), not pushed, one file, 26 insertions / 28 deletions. Build after commit: exit 0, 14 warning lines (same set as the base: class/package path notices x2, font-shape substitutions x9, hyperref Unicode tokens x3, plus the overfull vboxes from floats), 0 overfull hboxes, 0 undefined references. PDF: `/private/tmp/claude-505/-Users-zoudj-Documents-Codex-PerfRDD/3e96b884-102e-40fc-bc48-861da49d2619/scratchpad/notation_build/differing_slopes.pdf`.

Line numbers refer to the base file (c1474a8); my commit shifts lines only after 4474 (by -2). Settled decisions respected: (A1)-(A7), (U1)-(U4), (Q1m)-(Q2m), no DS tags, Upsilon = -U_eps', eta naming, outcome model without potential outcomes, beta_1 = "baseline slope", differing slopes as the setting. No notation changed in the commit.

---

## (a) Notation proposals (Task A), ordered by value

"Other session" = the session editing Lemmas ds-arm, ds-rout, ds-eval and (Q1m).

**N1. Fold regressor `Z_i` (Section 3) versus `\hat z`, `z_K` (App. 2-3); Gram matrices `Q_{Z,K,\mathcal J}`, `\hat Q` versus the score `Q` and quantile `\hat Q_p`.**
- Used: `Z_i` 628-638; `z_K` 30x (App. 2-3, true residual); `\hat z` 39x (App. 3, generated residual); `Q_{Z,K,\mathcal J}` 14x; `\hat Q` (empirical Gram) 8x at 3592-3697 next to `\hat Q_p` (605, 1893, 2730).
- Problem: two symbols for one object (`Z_i` is exactly `\hat z_i`); scope reserves Z for score covariates, and Block D (4022) uses `Z` for a covariate sub-vector; `\hat Q` (Gram) next to `\hat Q_p` (quantile) and `Q` (score).
- Proposal: Section 3: `\hat z_i`, `\hat\theta=(\sum\hat z_i\hat z_i^\top)^{-1}\sum\hat z_iY_i`. Gram matrices: `Q_{Z,K,\mathcal J}` -> `M_K`, `\hat Q` -> `\hat M_K` (matching the half-space second-moment matrices `M_d(e)` of (A1)(ii)), or `\Sigma_K`, `\hat\Sigma_K` (matching `\Sigma_{\tilde X}`, `\Sigma^{(d)}`, `\Sigma_\gamma`).
- Changes: 628-638 (3); `Q_{Z,K,\mathcal J}` at 2155, 2306, 2313, 2326, 2337, 2492, 2494, 2510-2512, 3546-3547, 3632, 3737, 3790; `\hat Q` at 3592-3697. The local shorthand `Q=Q_{Z,K,\mathcal J}` (3632) disappears.
- Risk: low in Section 3; moderate in App. 3 proofs (ds-outcome, ds-outer, ds-3a). Local `M(\phi)` (3128) and `M` (4168) are unaffected but nearby; `\Sigma_K` avoids that.

**N2. `F` is an integrand in Blocks A and C and a distribution function in Block D.**
- Used: `F_\phi`, `F=F_{\phi^*_\epsilon}`, `\hat F`, `F^{(j)}_\phi` (App. 2 Steps 1, 3; Block A 3853-3869; Block C 3926-3998); `F^*_\epsilon` (1001, 2752); `F(a,q)=PH_{a,q}` 9x at 4017-4187; `F_{R\mid Z}` (4087).
- Proposal: `F(a,q)` -> `G_a(q)`, extending Section 2's `G` (cdf of `T`), so `G_\gamma=G`, in the pattern of `h_a`, `f_{a^\top\tilde X}` in (A4); (A6) at 836-837 then names it.
- Changes: 4017, 4065-4077, 4084-4108 (9) and (A6).
- Risk: Block D is next to (Q1m), other session; coordinate.

**N3. `H` curvature, `H_X` weighted tail, `H_{a,q}` half-space indicator, `\mathcal H_{\mathcal J}` Hilbert space.**
- Used: `H` at 994, 1164, 1315-1335, 1379, 2029-2030, 2857-2858, 2961-2969, 4521; `H_{a,q}` 6x at 4016-4191.
- Proposal: `H_{a,q}(x)` -> `\ind_{a,q}(x)` (it is an indicator; the paper writes `\ind` everywhere else). Keep `H`, `H_X`.
- Changes: 4016, 4132, 4149, 4157, 4163, 4171, 4182-4183, 4191. Risk: low.

**N4. The `\epsilon` (and `\eta`) subscripts on representers, functionals and loadings.**
- Used: `r_{\mathrm{out},\epsilon}`, `r_{\mathrm{out},K,\epsilon}`, `r_{T,\epsilon}`, `r_{T,L,\epsilon}`, `r_{g,\epsilon}`, `r_{\rho,\epsilon}`, `L_{\mathrm{out},\epsilon}`, `L_{T,\epsilon}`, `\ell_{\alpha,K,\epsilon}`, `\ell_{2,K,\epsilon}`, `A_{U,\epsilon}`, `A_{\mathrm{out},\eta,\epsilon}`, `A_{T,\eta,\epsilon}`, `F^*_\epsilon`, `s^*_\epsilon` (100+ occurrences, App. 2-3, role scores 1001-1016).
- Problem: `\epsilon` is needed on `U_\epsilon`, `\phi^*_\epsilon`, `\tau_\epsilon`, `p_\epsilon`, `\hat\phi_\epsilon` (untrimmed versions exist). On representers/loadings no untrimmed version exists, and the slot is used as "limit as K or L grows" (`A_{\mathrm{out},\eta,K}\to A_{\mathrm{out},\eta,\epsilon}`, `r_{T,L,\epsilon}\to r_{T,\epsilon}`). The paper already drops it at 4332 and 4336 (`r_{\rm out}`). The `\eta` in `A_{\mathrm{out},\eta,\cdot}`, `A_{T,\eta,\cdot}` distinguishes nothing.
- Proposal: `r_{\mathrm{out}}`, `r_{\mathrm{out},K}`, `r_T`, `r_{T,L}`, `r_g`, `r_\rho`, `L_{\mathrm{out}}`, `L_T`, `\ell_{\alpha,K}`, `\ell_2` (no K dependence; equals `B` of Lemma ds-rout, see N8), `A_U`, `A_{\mathrm{out}}`, `A_{\mathrm{out},K}`, `A_T`, `A_{T,L}`, `F^*`, `s^*`. Keep `\tau_\epsilon`, `U_\epsilon`, `p_\epsilon`, `\phi^*_\epsilon`, `\hat\phi_\epsilon`, `B_\epsilon`, `v_\epsilon`.
- Risk: largest mechanical change; touches Lemmas ds-rout and ds-eval (other session). Do by scripted replacement after that session lands, then rebuild.

**N5. `\rm` versus `\mathrm` (source only).** `{\rm out` 29x (`\tau_{\rm out}`, `r_{\rm out,\epsilon}` 3741-3805, `A_{\rm out,\eta,\epsilon}` 3796, 3805) vs `\mathrm{out}` 46x; `\kappa_{\rm id}` (3527, 3543, 3546) vs `\kappa_{\mathrm{id}}` (2162, 2166); `{\rm avg}`, `{\rm pool}`, `{\rm grid}`, `{\rm prod}`, `{\rm all}`, `{\rm dist}` vs `\mathrm{dist}` (3871). Output identical; `\rm` is deprecated and the mix invites `A_{\rm out,\eta,\epsilon}` vs `A_{{\rm out},\eta,\epsilon}`. Proposal: `\mathrm{}` everywhere, one regex pass. Risk: none to output; merge conflicts in 3741-3805.

**N6. `G_L` (Lebesgue Gram) vs `G` (cdf of T); `P_L` (projection) vs `P_T` (fold average).** `G_L` 6x (681-684, 1958, 2393, 2610, 2705); `P_L` 19x (2390-2716). `P_L` reads as "average over fold L". Proposal: `P_L` -> `\Pi_L` (paper already uses `\Pi_\delta` for a projection at 2879-2884); `G_L` -> `\mathcal G_L` or `\Gamma_L`. Risk: low; no other-session lemma.

**N7. Sample size `N` in Appendix 5, `n` elsewhere.** 4512-4513, 4523, 4529, 4536, 4565-4566, 4573 (8 places incl. two table headers). Proposal: `n`. Risk: none.

**N8. `B` denotes four objects.** Second-order bias constant (1115-1127, 1222-1226); `B=\E[I_0\rho_X(\phi^*_\epsilon-\eta)]` in Lemma ds-rout (2227-2271), which is `B_{\phi^*_\epsilon}` of 2114 and equals `\ell_{2,K,\epsilon}` (2300); `B_\epsilon(\phi)` trim-boundary term (3100-3146); `B(Z)`, `B_0` envelopes (4040-4061). Proposal: Lemma ds-rout: write `\ell_2` (or `B_{\phi^*_\epsilon}`); Remark ds-bias and Section 5: `\mathcal B` (6 places). Leave `B_\epsilon`, `B(Z)`. Risk: Lemma ds-rout is the other session's.

**N9. `L` = density dimension and a generic linear functional.** "Let `L` be a bounded linear functional" (3734-3744) next to `L`, `L_n`, `L_{T,\epsilon}`, `L_\lambda`. Proposal: `\mathcal L` at 3734-3744 (4 places). Risk: none.

**N10. `\hat I` used before defined.** Used at 3725-3726 (proof of ds-outer), 3867-3868, 3887 (Block A); defined at 3925 (Block C). Proposal: define `\hat I_i=\ind\{\hat l\le\hat\eta_i\le\hat u\}` in Section 3 beside (eq:prelim-supported-estimator) and use it there, in Remark ds-check (913) and App. 1 (1903); Block C's definition becomes a reminder. Risk: the display 719-726 changes typographically, not the estimator.

**N11. `e_K` (approximation error, 23x from 2328) vs `e_L` (orthonormalized basis, 2705-2714, 5x).** Proposal: `e_L` -> `\breve N_L`. `e` (argument), `e_1`, `e_j` can stay. Risk: none.

**N12. `\Psi`:** `\hat U'_\epsilon(\phi^*_\epsilon)` in Remark ds-bias (1111-1130) and the map `\Psi(\delta)` in Lemma ds-zero-intercept (2890-2896). Proposal: 2890-2896 -> `\Upsilon_\delta` (3 places). Risk: none.

**N13. Quantiles and fold label.** `\hat Q_p` (605, 1893-1895, 2730-2736) vs `\widehat q_p(a)`, `\widehat q_p(\hat\gamma)` (4120-4221, 8x; only use of `\widehat`); `\hat Q_p=\widehat q_p(\hat\gamma)` never stated. Fold label `q\in\{\gamma,o,T,U\}` (573-574, 2029-2030, 2089-2090, 2957-2965) next to `q_p=Q_p(T)` (994) and the scalar argument `q` of Block D. Proposal: (i) one sentence after Lemma prelim-boundary (4225) stating `\hat Q_p=\widehat q_p(\hat\gamma)`, or write `\hat Q_p(a)` in Block D; (ii) fold label `q` -> `\iota` (9 places). Risk: (ii) touches `\sum_q` in the theorem's expansion; low.

**N14. `S_{\mathcal J}` (27x) vs bare `S` (from 3594/3632, also Prop. ds-pseudo 4281-4331); `\hat S_{\mathcal J}` (2584-2586) vs `\hat S` (15x).** Bare `S` is declared "local" at 3632 but used from 3594 and in 4281-4331. Proposal: `S`, `\hat S` throughout once defined (the `\mathcal J` already rides on `\mathcal H_{\mathcal J}`, `\|\cdot\|_{\mathcal J}`), or subscript everywhere. Risk: 2133-2170 is near Lemma ds-arm (other session).

**N15. `g` vs `f_T`.** `g=f_T` stated at 426 and 3003; `f_T` still at 366, 676, 822, 838, 995, 1859, 2465, 2797, 4076, 4133, 4192, 4211. Proposal: `g` everywhere except inside the index family (`f_{a^\top\tilde X}`) of (A4)/(Q1m)-(Q2m), with one reminder `f_T=f_{\gamma^\top\tilde X}=g`. Risk: `z_p` at 995 and 4211 would read `/g(q_p)`; low.

**N16. Term inconsistencies (word replacements).**
- `\mathcal J`: "fixed region" (624, 1715, 1855), "outer outcome region" (1906), "fixed outer nuisance region" (2134), "outer region" (2742), "nuisance region" (3000, 3017), "fitting region" (3814); `\mathcal T` is consistently "the density interval". Proposal: "the outcome interval `\mathcal J`" (about 9 places).
- Window: "overlap endpoints" (599, 1845), "trim endpoints" (12x), "trimming boundaries" (165), "overlap interval" (1972-1973), "overlap set" (1713-1720, 1770), "overlap-supported set" (1713), "trim window" (10x), "supported population" (4x). Proposal: "trim window" `[l_0,u_0]`, "trim endpoints" `l_0,u_0`, "supported population" for the units; drop "overlap endpoints/interval/set" and "trimming boundaries".
- "blocks" for folds in the bootstrap description (1240 "four new blocks", 1250 "within its blocks") vs "folds" elsewhere and the historical "ten-block split" in App. 5. Proposal: "folds" at 1240, 1250 (Section 5, not 6).
- Weighted objects: `\rho_X` "weighted density" (569, 673, 738, 2894), "weighted index density" (867); `H_X` "weighted tail" (364); estimate "weighted-tail density estimate" (275), "weighted-tail estimate" (697), "weighted-tail functional" (1739). Proposal: `\rho_X` = weighted density, `H_X` = weighted tail; 275 -> "vector-valued weighted-density estimate".
- "cutoff" (76) vs "threshold" (50): title, abstract, theorem name and "threshold selection" use threshold (the problem); the body uses cutoff (the value). Reads as deliberate; no change proposed except "candidate threshold" (1664) -> "candidate cutoff".
- "latent type" at 175 describes Eckles et al.'s model, not ours: keep. "covariate index"/"linear score index"/"latent-index decomposition": fine as prose.

**N17. `\pi(\phi)` (537-545, 3374-3395) vs `\pi^{(d)}(e)` (2223-2283).** Both probabilities; superscript separates. Optional `\pi(\phi)` -> `\varpi(\phi)` (8 places). Low value.

**N18. `z_p` (quantile influence: 995, 1007, 2730-2740, 2823, 2834, 4209-4230; 10x) vs `z_K`, `z`, `\hat z` (regressors, 70x).** Proposal: `z_p` -> `\zeta_p`, then rename the local `\zeta_0,\zeta_1` in the proof of Lemma ds-gram (3558-3566) to `\varsigma_0,\varsigma_1`; or resolve via N1 by renaming the regressor. Risk: touches (eq:ds-role-scores).

**N19. Minor (no change unless wanted).** `h` bandwidth in the Introduction (139, 142) and `h_n` grid mesh (1913-1928) vs `h` weighted density (mesh could be `\varpi_n`); `\theta_t` in Remark ds-pathwise (1079-1080) next to `\hat\theta`; `\varphi_\gamma` (996-1005, 2551, 2826, 2839) visually close to `\phi`; `c_1,c_2` slope directions (2138-2167, 3525-3558) next to the cost `c`, while the proof of ds-rout uses `k_d` for the same role (2256-2271) and `k` is also the generic function in `L_{T,\epsilon}(k)`; `\hat\nu` (2484, 2533, 2564, 3995) never written out, add `\hat\nu=(\hat\alpha-c,\hat\beta_2^\top)^\top` at 2564; `\ell_{2,K,\epsilon}` (2300) carries a `K` it does not depend on; `R` coordinate of `\tilde X` in Block D vs remainder `R` (2030, 2652); `\rho(r,z)` (4058) next to `\rho_X`; `\varepsilon_0` (4217-4221) next to the error `\varepsilon`; `m` a count (1175) and a grid size (2026) next to `m_0`, `m_p`, `m_K`; `V` the (A3) neighborhood and a table column at 1303, 1315 (rename the column); `s` for six things (`H_X(s)`, path score, `s^*_\epsilon`, `s_T,s_\eta`, `s(a,q,z)`, `s_\beta`), `s^*_\epsilon=F'` could be `\dot F^*` with N4; the argument set is written out at 668 and 1844 though named `\mathcal V_\Phi` at 457, 794 (use the name); macros `\Pn`, `\Gn` (20-21) defined, never used (Block D writes `\mathbb P_{n_b}`, `\mathbb G_{n_b}`), delete; hard-coded "Section~3/4/5.1/5" at 577, 747, 787, 901, 1173, 1293, 1377, 1381, 1876, 2180, 3227, 3444, 3450, 4496 (Sections 3 and 4 have no `\label`; adding labels was outside my brief).

---

## (b) Writing edits applied (commit 862bd53)

Sentence-level only; no theorem, assumption, display, number, citation, label or notation changed.

- §1, 183-184: "model its conditional mean as" -> "model the conditional mean of $W$ as".
- §3, 598-599: "Under the OLS rate this error is first order, and it is retained in the influence function." -> "Since $\hat\gamma-\gamma=\Op(n^{-1/2})$, this error is first order and is retained in the influence function."
- §4, 770-774: "The proof first expands ... We first treat ... and then restore" -> "The proof expands ... It first treats ... and then restores".
- §4 Remark ds-pathwise, 1093-1094: "Corollary ... then says that the three factors are orthogonal directions, together with this property of $\psi_\gamma$." -> "Corollary ... thus combines the orthogonality of the three factors with this property of $\psi_\gamma$."
- §5, Discussion: no edits.
- App. 1, 1842-1843: "The distribution of $T=\gamma^\top\tilde X$ has distribution function $G$, density $g$, and survival function $\bar G$." -> "The index $T=\gamma^\top\tilde X$ has distribution function $G$, density $g$ and survival function $\bar G$."
- App. 1, 1855-1861: "The supported target is then identified as [display] with supported utility [display]" -> "The supported target is then identified. With [display] the supported utility and target are [display]" (displays unchanged; the old sentence said the target is identified "as" the definition of $H_X$).
- App. 1, 2014-2015: "Coefficientwise convergence of the sieve coefficients" -> "Convergence of the individual sieve coefficients".
- App. 2: no edits.
- App. 3, 3178: "A sufficient condition for (U4) is" -> "First, a sufficient condition for (U4) is" (paragraph "Two complements" already says "Second").
- App. 3, 3362-3363: "A bimodal index, such as a strong binary covariate in the score produces, breaks the case $\beta_2=0$ too:" -> "A bimodal index, which a strong binary covariate in the score produces, breaks even the case $\beta_2=0$:".
- App. 3, 4110, 4142: "halfspace(s)" -> "half-space(s)" (six other places already hyphenate; label untouched).
- App. 3, 4346: "The appendix separates ..." -> "The appendices separate ..." (representer identities and the assembly are in App. 2).
- App. 4, 4439-4440: "The Monte Carlo validation battery therefore reports" -> "These simulations therefore report".
- App. 5, 4447-4448: "None of these diagnostics enlarges the continuous-threshold theorem or changes its scope." -> "None of them changes the scope of Theorem~\ref{thm:prelim-trim}."
- App. 5, 4480-4482: "so the mixture constant is visibly noisier while the rate remains close to $n^{-1/2}$" -> "so the mixture has a visibly larger variance constant while the rate remains close to $n^{-1/2}$".
- App. 5, 4496-4497: "not additional theorem cases" -> "not cases covered by the theorem".

Whole sentence cut (one): App. 5, 4474: "The non-Gaussian and spline-density batches are complete." (progress note; the next sentence stands alone).

---

## (c) Section 6 suggestions (not applied)

1. 1492-1493. Current: "with a common slope the estimator fails, as shown below." Proposed: "with a common slope the estimator selects a boundary cutoff, as shown below."
2. 1496-1503. Current: "The theory of Section~\ref{ssec:ds-conditions} is stated for the unrestricted block, for a score in which each covariate enters once, while distance enters here through two columns, and for trim endpoints that are index quantiles on the index fold. Here the window is instead the central 99\% of the residuals, estimated on the evaluation fold (below), so the endpoint terms of Theorem~\ref{thm:prelim-trim} take the form of a trimmed mean in $\hat\eta$ rather than the index quantiles of Appendix~3. We expect all three to carry over with notational changes, but we have not written that out." Proposed: "The theory of Section~\ref{ssec:ds-conditions} is stated for an unrestricted $DX$ block, for score covariates that are not functions of one another (distance enters here twice, as distance and log distance), and for trim endpoints that are index quantiles. Here the window is the central 99\% of $\hat\eta$ on the evaluation fold, so the endpoint terms of Theorem~\ref{thm:prelim-trim} become those of a trimmed mean in $\hat\eta$." Keep or replace the last sentence once the taxi theory (scope, 2026-11-04) is written. Note: "each covariate enters once" is not a condition of the theory; the gap is the restricted block and the quantile cap.
3. 1533-1540. `\mathcal J` is used two sentences before it is defined. Proposed order: "We trim at $\epsilon=0.05$. The outcome model is fitted on the central 99.5\% of $\hat\eta$, which is $\mathcal J$. Only 11\% of rides are treated, and both $\epsilon$-endpoints $l_0$ and $u_0$ lie outside $\mathcal J$ and outside the central 99\% of the residuals. The trim window is therefore the central 99\% of $\hat\eta$ on the evaluation fold, re-estimated in every fit, so the window lies inside $\mathcal J$ and the $\epsilon$-endpoints never bind." Also "overlap endpoints" -> "trim endpoints".
4. 1664. Current: "the candidate threshold is searched over a fixed compact interval." Proposed: "the candidate cutoff is searched over a fixed compact interval $\Phi=[-3,3]$."
5. 1682-1684 (table). Current: "The estimator with a $D X$ block is essentially centered, with RMSE declining at about $n^{-1/2}$. The common-slope fit misses the full target by roughly $0.40$--$0.52$." Proposed: "The estimator with a $DX$ block is centered, with RMSE declining at about $n^{-1/2}$. The common-slope fit misses the target by $0.40$ to $0.52$."
6. 1689-1691 (table). Current: "applying the projection characterization of Proposition~\ref{prop:ds-pseudo} to its model gives the pseudo-target $0.408$." Proposed: "Proposition~\ref{prop:ds-pseudo} applied to its model gives the pseudo-target $0.408$."
7. 1697-1701. Current: "a linear $D X$ block does not recover the optimum, which marks a limit of the extrapolation claim." Proposed: "a linear $DX$ block does not recover the optimum; this is the limit of the extrapolation."
8. 1560-1564. Split the long sentence: "... is $+\$0.25$ at \$15: under the model, the marginal rides there would tip about 25 cents more under the percentage menu. Lowering the cutoff from \$15 therefore raises the expected tip on the supported population, which is why the estimate and every bootstrap draw lie below \$15."

---

## (d) Todo boxes

One active `\todobox` (969-989 and 1785 are commented out): 2667-2684 "Original route, replaced on 2026-10-20." A record of the superseded Step 3 argument, not an open task: stale as a TODO; belongs in the log or a LaTeX comment (2682-2683 already points at the manuscript commit). Contents kept.

---

## (e) Technical observations (report only)

Checked and consistent: signs of the role scores vs Steps 4-5 and Lemma prelim-boundary; Schur-complement bound in Lemma ds-rout; bias orders $K^{-7/2}$, $L^{-4}$ vs (A5); rates in Lemmas ds-weighted-consistency, ds-outcome, ds-outer, ds-3a; bias combination in Remark ds-bias; $\sigma^2/H^2=31.09$ and $\Var\psi_U/H^2=1.256$ from the stated variances; window equivalences 389-391; likelihood-ratio argument of Lemma ds-margin.

Worth a look:
1. Lemma ds-outer (3687): the approximation piece in the proof is $\Op(K^{-5/2}n^{-1/2})$, smaller than the $K^{-2}n^{-1/2}$ in the statement (which comes from the fit-error piece). Statement correct, not tight.
2. (A3) asks $f_\eta\in C^3$ on a neighborhood of $\mathcal J$; I found no step using more than a bounded $f_\eta''$ (Lemma ds-eval) and $f_\eta'$ (log-concavity). Not checked exhaustively; if nothing needs $f_\eta'''$, (A3) could ask for $C^2$.
3. App. 3, 3000-3027 restates $l_0,u_0$, $U_\epsilon$, $\hat l,\hat u$ for the third time (after §2, §3, App. 1). Cutting it is structural, so left; the one obvious redundancy left.
4. Remark ds-check (911-922) and App. 3 "A consistent check of (A1)" (3901-3917) share four near-identical sentences; one could be a pointer.
5. §6, 1496-1503: "each covariate enters once" is not a condition of the theory (see (c) 2).
6. Lemma ds-eval assumes $\E X=0$; consistent with the centering convention and Lemma ds-shift, and the zero intercept coordinate of $\bar A$ depends on it as Lemma ds-zero-intercept says. No issue.

---

## (f) Commit and build

Commit 862bd53 on `review/notation-20261104` (local only). `latexmk -pdf` exit 0; 14 warning lines, all pre-existing (class/package path x2, font-shape substitutions x9, hyperref Unicode x3) plus the float overfull vboxes; 0 overfull hboxes; 0 undefined references or citations.
