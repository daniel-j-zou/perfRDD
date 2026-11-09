# Submission-readiness review of differing_slopes.tex (2026-11-09)

Four independent Fable 5.1 (high) referee reads of manuscript master eb1825b, one per
part: front matter and §§1-2, 5-7; §§3-4; Appendices 1-2; Appendices 3-5. Claude
checked each report's main claims against the text before relaying them. Line numbers
below refer to eb1825b. Bracketed notes in the reports were added when they were used
as briefs for the edit, to mark items the author had not approved.

**Verdict (as relayed to the author).** Not ready yet; the gaps are presentation, length
and the application's robustness, not the mathematics: no reviewer found a wrong step.
The research gap is clear and fairly positioned. The main text was 23 pages (about
13,100 words) in the Biometrika class, against the template's typical 15 and rarely
much more than 20.

**Author decisions (2026-11-09).** "Fix all the writing. Do D3 and D4, and send out
subagents for all the Es." On old Appendices 4-5: "take those out for now, but don't
delete them permanently. Do all the other decisions too."
- Applied in manuscript f31dfa2, 3fe1acf, 0294b96 (three Fable editors), merged with
  cross-references in 616a1ff, review fixes in 39281e1, changelog c5751a6/73af42d.
  Three independent reviews of the merged diff (main text; appendices; an xhigh
  check of the new mathematics) found no mathematical error. Their fixes: a dropped
  (A1)(ii) clause still cited in Prop E3's proof; the spline-size sentence needed the
  rate m = cn + O_p(n^{1/2}); a grid refined by a local optimizer needed one sentence in
  Appendix 2; half-finished direction renames; pointers.
- Set aside (not deleted): `../manuscript/storage/set_aside_20261109.tex`.
- E items: see the research log entry of 2026-11-09.

**Not done, not approved: D1, moving material out of the main text.** After the pass the
main text is 25 pages (about 14,700 words). Candidates from the reports: the §4.3
proof roadmap; the §5.2 second-order bias paragraph and the stray n = 32,000 run; most
of §5.4 (keep one row of its table); §6.6 and its table; the §6 mapping table; three
Future-work paragraphs (sequential and performative thresholds); the eligibility
clause of Proposition 3; the finite-difference check in the pathwise remark; the
expansion in the bias remark.

## Claude's assessment of old Appendices 4-5 (given to the author before the decision)

- Appendix 4 (now 6), needed: the design, parameters and targets of §6.6, the note that
  the quadratic design breaks single crossing on [-3,3], sample sizes and replications.
  Not needed: the oracle variant, the Gaussian-tail/spline/ridge variants beyond the
  two fits §6.6 reports, the cross-reference to the ten-block runs, the boundary flag,
  the replication table's "Main purpose" column, the conditional-variance and coverage
  paragraph. All set aside except a reworded sentence on the two fits and the grid.
- Appendix 5 (now 7), "Additional hard-trim diagnostics": the programme table (a research
  plan whose second row is never reported), the t_5/mixture runs (full-sample reuse,
  not the four-role estimator: set aside and rerun), the ten-block spline-density
  comparison and the bootstrap paragraph: all set aside.
- Four-role split check: the common-slope table kept (a second design matching the
  analytic prediction); the differing-slopes table (linear outcome fit, full-sample
  reuse, no analytic level) and the pooled-against-average paragraph (already in §5
  Table 1) set aside.
- Identification examples and checks: the examples and the log-concavity discussion
  are cited and kept; the numerical-checks bullet list shortened to one paragraph that
  keeps every value the proofs cite.


---

# Referee report: abstract, §1, §2, §5–§7 (Fable 5.1, 2026-11-09; line numbers refer to master eb1825b)

**Verdict.** The paper has a real and well-motivated target (the utility-maximizing threshold as estimand, root-n inference for it), the introduction positions it accurately against the named prior work, and §5 supports the first-order theory convincingly (Tables 1–3). Its weaknesses are presentation and robustness, not substance. An RD reader is lost first in the dense "three reasons" paragraph (l.152–163), which uses "index", "supported population" and "estimated trim endpoints" before any of them is defined. §2 is readable but Proposition 2 borrows conditions from an appendix lemma, and the target defined in §2.3 (the epsilon window) is not the one used in §6 (a 99% residual cap), which §2 never mentions. §6 rests on two assumptions the paper's own diagnostics show fail (X independent of eta; linearity of the effect in distance for untreated fares), the gain is 0.65 cents per ride with no interval, and no simulation probes either failure. At 23 pages / ~13,100 words of main text the paper is too long for Biometrika; 5–6 pages should move to the supplement.

**Must fix (ranked)**

1. *Robustness to the failures the application exhibits* (§5–§6, §7). §6.5 reports R² = 0.203 for eta-hat² on covariates and a residual spread ratio 1.48 across distance deciles (l.1581–1587), i.e. X ⊥ eta fails, yet no result or simulation says what the estimator converges to then, and §7 lists X ⊥ eta only as a maintained assumption (l.1718). Add one simulation with heteroskedastic eta of the size seen in the data (and, if possible, a Proposition-C-style pseudo-target statement), and name this in §7 as the main limitation. [Being handled separately by computation agents; NOT for the writing pass.]
2. *The gain has no uncertainty* (l.1564–1566; Fig. 1 left). [Interval being computed separately; NOT for the writing pass.] The left panel's y-range (−30 to +1) makes the gain invisible; at print size the figure fonts are too small. [Figure: not for the writing pass.]
3. *The §6 target is not the §2 target.* §2.3 defines phi*_eps via the epsilon window (l.380–396); §6 says both endpoints "lie outside J" and replaces the window by "the central 99% of eta-hat" (l.1534–1539), so epsilon = 0.05 is nominal there and the target becomes "the tip-maximizing threshold for 99% of rides". Add one sentence to §2.3 that the window is capped at fixed residual quantiles when the epsilon endpoints fall outside the data and that the capped-window results in the appendix cover the cap. (The capped window is a settled decision; the gap is only that §2 does not announce it.)
4. *Proposition 2 cites appendix conditions* (l.498: "the regularity of Lemma C2"). A main-text proposition must be self-contained; state the smoothness it needs in-line.
5. *Extrapolation and weak identification* (l.1610–1613, l.1658–1660). The threshold below $15 depends entirely on beta2 = $0.24/mile applied to untreated fares; the distance column keeps only 4.5% of its variance after partialling. [beta2 interval being computed separately.] Move the extrapolation caveat (l.1658) up to the Results subsection.
6. *Overreach in one sentence* (l.216–218): "point identification of the optimal threshold" should read "of the supported optimum", as l.224 and contribution 1 (l.264) correctly say.
7. *Notation clashes* a referee will circle: H is the weighted tail H_X (l.360) and the curvature H = U'' (l.1163, Table 3 "−H"); Q is the running variable and Q_p(T) a quantile (l.382–383); g = f_T is introduced twice (l.335, l.422). [Decision: H and Q_p stay; add one clause where H is introduced that it is unrelated to H_X; g=f_T at l.422 is a deliberate reminder.]

**Should fix**

- *Intro clarity* (l.152–163): defer the three reasons until after the model sketch at l.181–188, or define "supported population" in one clause at first use. "aligned heterogeneity" (l.267) and "four roles" (l.274) are undefined in §1.
- *Positioning vs Wibisono et al.* (l.151–152): say what they target (the average effect at the deployed threshold) so the reader sees why allowing X-dependence is not the novelty; otherwise the contrast with "the effect must therefore vary with X" (l.156) is confusing. The .bbl entry for Wibisono2025 has no venue.
- *Keywords/title*: "Error in variables" (l.70) does not describe the paper; the title lacks an article and "score-explained" is in-house jargon. Abstract says "subpopulation with overlap" while the body says "supported population". [Title, keywords and abstract wording are the author's; do not change.]
- *§5.2 second-order bias paragraph* (l.1219–1234) and the stray n = 32,000 run (l.1216) are supplement material; keep one sentence. [Moving material out of the main text was NOT approved; do not move.]
- *§6.6 and Table 7* (l.1663–1708): a simulation placed inside the empirical section, with a table of prose, no numbers and a pointer to Appendix 4. [Moving NOT approved.]
- *Table 6*: "four comparisons" (l.1593) counts five rows after the design variations; the row "eps = 0.075; eps = 0.10 | 13.00; 13.08" packs two cases into one.
- *§7*: honest on stability, linearity, pseudo-target and boundary optima, but omits that bootstrap validity is unproved (l.1134–1136), driver clustering and the fare grid (l.1660). [Add these to the limitations.] The sequential and performative paragraphs (l.1740–1753) are far from the paper and can go. [Cutting NOT approved.]
- Marked as a settled decision that hurts: "latent ability" for a taxi residual equal to "how slow it was" (l.1484, Table 5) reads oddly; in §6 a parenthetical "the ride's slowness" at first use would help.
- Minor: "11.0%" (l.1455) vs "11%" (l.1534); six-digit targets (l.1162–1163) beside four-digit ones; the estimate "$12.87" is a Q-scale value mapped back to dollars, say so once.

**Works well.** The identifying idea in l.165–179 is the clearest paragraph in the paper. Proposition 1 and the "margin moves up" explanation (l.481–494) are intuitive. Remark 1's closed form for phi*_eps − phi* and the eps-sensitivity table (Table 3) answer the obvious objection to trimming. Tables 1–2 match the predicted variances to within 3% and the four-fold variance ratio is visible. §6's robustness table and the two external benchmarks are the right kind of evidence. The accounts of Marinescu et al., Dong–Lewbel, Zhang–Ben-Michael–Imai, Ben-Michael et al., Eckles et al. and Carneiro–Heckman–Vytlacil are fair; the novelty claim is correctly hedged ("to our knowledge", l.223).

---

# Referee report: Sections 3–4 (Fable 5.1, 2026-11-09; line numbers refer to master eb1825b)

**Verdict.** The theory is complete, internally consistent and honest about its limits: the endpoint signs in the role scores (1005) agree with Appendix 2 (2718-2736), the bootstrap is plainly marked unproved (1135-1136), and the second-order remark says "is not a theorem" (1109). The statement of Theorem 1 (1021-1036) is clean. What a reader cannot say after one read is what sigma^2 is made of: the six ingredients of the role scores are "all defined in Appendix 2" (1014-1015). (A4) and (A6) are non-primitive and one is stated in appendix-internal terms ("Block D", 834); several clauses are redundant. Section 3 gives the estimator but not the tuning or maximization choices, which sit in Section 5 and Appendix 1.

**Must fix (ranked)**

1. *Role scores by pointer (999-1015).* F* and r_T are one-liners in the appendix (2747-2750, 2357-2361) and b_v = F*(v) f_eta(v) (2767) is a phrase; display them in Section 4.4. For r_out, give the defining Riesz identity in words plus the equation number of its closed form (2218); for A-bar, write the sum "A_U - A_out - A_T - b_l m_{1-eps} + b_u m_eps" (2797-2801) with one clause per term (evaluation shift, outcome fit, density block, endpoints). Then the variance becomes readable and Corollary 1's "intercept coordinate of A-bar is zero" (1044-1045) has a referent.

2. *(A4), (A6) are not primitive and lean on the appendix.* (A4)(i) asks L2-Frechet differentiability of a -> h_a with derivative "-h'_{X-tilde} delta" (818-819); the sufficient condition at 868-871 ("holds when (a,t) -> h_a(t) is continuously differentiable...") is asserted without a lemma. Make it a displayed lemma in the appendix and cite it. In (A6), "m_p := ... is the continuous version of E(X-tilde | T = q_p)" (832-833) mixes a definition with an identity; say whether the identity is assumed or proved, and replace "Appendix 3, Block D" (834) by a named lemma. Also state which norm "|| . ||_{L2(T')}" (818) means for a matrix-valued function.

3. *Redundant clauses.* "E(X-tilde X-tilde^T) is nonsingular" in (A2) (799) follows from (A1)(ii) since M_0(e)+M_1(e) is that matrix (791-792). "T has a continuous distribution" (789) follows from (A3). "f_eta is bounded, and c_f <= f_eta <= C_f on J" (808-809): say where the global bound is used or drop it. "In particular, E(W | X, eta) = ..." (802) is a consequence, not an assumption. (A1)(i) repeats Section 3's choices (614-616, 663-664); keep but say "as in Section 3".

4. *Implementation gaps in Section 3.* (a) theta-hat sums over all i in I_o (632-634) although z-hat_i is defined only when eta-hat_i in J (621-622): put the eligibility indicator in the sum. (b) K, L: only a range (729-731) and "deterministic knots" (621); the rule used, max{3,[m^{11/60}]} interior knots with m the smaller arm count (1174-1177), appears in Section 5 and is data-dependent, while (A5) says "deterministic quasi-uniform knots" (824) and the Appendix 1 grid argument covers a fixed finite grid (2005-2016). State the rule in Section 3 and say why it is covered. (c) Maximization: "in argmax" (724-726); the grid-plus-refinement rule and the smallest-maximizer tie rule are in Appendix 1 (1901-1905). One sentence here. (d) alpha-hat(eta-hat_i) in (719) is undefined for evaluation units with eta-hat_i outside J when I-hat_i = 1 in finite samples; state the convention. (e) Define "<<" (730). (f) Say the folds are formed at random (557-559).

5. *Theorem statement.* Add: the result holds for any measurable selection from the argmax (or the grid rule); H is nonzero by (A1)(iii), with a pointer to 839-843; either assume sigma^2 > 0 or say the limit may be degenerate; U-hat_eps is twice differentiable in phi because G-bar-hat and H-hat_X are integrals of splines (not said in Section 3).

6. *Bootstrap.* Section 4 says only "re-estimates every fit ... in each draw" (1134-1135); the scheme (resample n observations, split anew, refit) is in Section 5 (1238-1241). Move it here and note that re-splitting in each draw departs from the fixed folds of (A7), which is one reason validity is unproved.

**Should fix**

- *Roadmap (919-965).* The five steps mirror the appendix table (2040-2056) but read as notes for someone who knows the proof: "fixed population memberships" (926-927), "Representer convergence handles the changing weights inside centered averages" (934-935), "need no separate moving-set argument" (951-952). End each step with the term of (1026) it produces: steps 1-2 the three sampling averages and the o_p remainder, step 3 P_gamma psi_gamma through A-bar, step 4 the zeta terms in psi_T, step 5 division by H. The roadmap precedes the display it explains; forward-reference (1026) or swap the order.
- *Undefined terms at first use:* "loading" (574, 1013), "representer" (737, 934), "role score" (747) are used in Section 3 before Section 4 explains them.
- *Move to appendix:* the eligibility-indicator clause of Proposition 3 (889-892); the finite-difference check in the pathwise remark (1093-1100); the expansion in the bias remark (1109-1126). [Moving NOT approved; leave these.]
- *Motivation gaps:* (A3)'s three derivatives of g and rho_X are explained only in Appendix 1 (1983-1986); (A2), (A5), (A7) have no motivating sentence.
- *Notation collisions:* H for the curvature (992) vs H_X (360) and the appendix space H_J (2198) [keep H; add a clause]; q as fold index (571) vs q_p quantile (827, 992) [kept by author decision]; pi(phi) (533) vs pi^{(d)} (2203); V (807) vs calV_Phi (787); "phi-hat_{eps,pool}" is named for the first time at 1124; Section 3 describes it unnamed (761) [name it there]. nu is used in r_T (2357) but defined in the main text only inside a remark (1084) [define it with the role scores]. A half-page symbol table in Appendix 1 with the line of definition would help.

**Works well.** The theorem statement itself; the explanation of (A1)(iii) and the curvature constant (839-854); the constant-shift argument for U^T (698-712); the four-role rationale and the factor four (744-751, 1038); the pathwise-derivative remark and the explicit "no efficiency claim" (1092-1093); the clear flags on what is unproved (1109, 1135-1136).

---

# Referee report: Appendices 1–2 (Fable 5.1, 2026-11-09; line numbers refer to master eb1825b)

**Verdict.** Appendices 1–2 are technically in good shape: the five-step plan of §4.4 (925–957) is followed step for step in Appendix 2, every object named in the role scores is defined in Appendix 2, the signs of A-bar (2798–2801) and the zero-intercept computation (2870–2897) check out, and no step is wrong. The weaknesses are organizational and notational.

**Must fix**

1. *"Organization of the proofs" is empty* (2018–2024). It says only "Table~\ref{tab:proof-map} locates each block" and the table is in Appendix 2 (2039–2056). Move the table under this heading, and add rows for what it omits: Lemma ds-shift and the grid/data-driven arguments (Appendix 1), Lemma ds-zero-intercept and the two corollaries (2859–2954), and the proofs of Propositions 1–2 and Remark 2 (Appendix 3). A referee asking "where is Corollary 2 proved" currently has to scan.
2. *Blocks B and E do not exist.* Line 2024 says "letters A, C and D … are labels for cross-reference only"; Appendix 3 has Blocks A (3710), C (3773), D (3856). Leftover of a superseded route.
3. *Appendix 1 uses results proved later.* The grid argument (1902–1921) and data-driven sizes (2005–2016) invoke Theorem 1 and Proposition 3 ("so it is consistent by Proposition~\ref{prop:ds-consistency}", 1908–1909). Not circular, but a reader meets corollaries of the theorem before its proof. Move both paragraphs after the proof of Corollary 2 (after 2954), or label them "consequences, proved in Appendix 2".
4. *Forward references inside Appendix 2 to definitions in Step 5.* Step 3 writes "with b_v and A_U as in Step~5" (2580); Step 3a cites \eqref{eq:prelim-Aalpha-loading} (2601); Lemma ds-dens-block states its conclusion with "A_T of \eqref{eq:prelim-Ag-loading}" (2617), defined at 2785–2790. Also Lemma ds-weighted-consistency (2683) is used at 2500, 180 lines earlier. Either define A_U, A_out, A_T, b_v at the top of Step 3 (they are four displays) or state them inside the lemmas that produce them.
5. *Step 2's remainder bounds are unlabeled prose* (2454–2552). "Outcome bias", "Density representer replacement", "Evaluation interaction", "Outcome remainders", "Products", "Feasible products" each carry a rate but no statement to check against. Package them as one lemma with parts (a)–(f), each with its hypotheses and rate, e.g. (d) ||P_o[S z_K z_K^T] - Sigma_K|| = O_p(K/sqrt n) under (A2), (A5). The sentence "is paired with centered noise in Lemma~\ref{lem:ds-outcome}(e) and costs the same order" (2520–2521) is unverifiable from Appendix 2 alone; state the resulting bound.
6. *Overloaded alpha, b in the Hilbert-space definitions* (2115–2117, 2139–2141, 2192–2195). The true functions alpha, b are reused as generic directions: "alpha, b in L_2(P_eta)". Lemma ds-rout's proof already uses u_d, k_d (2236); use those (or another pair) throughout Step 1a, especially in the bound "||alpha I_0||_{L_2(P_eta)}" (2194), where alpha is a direction.

**Should fix**

- *Redundancy with the main text.* "Latent index and outcome model" (1803–1862) restates §2: gamma, eta, the model, l_0, u_0, I_0, H_X, rho_X, U_eps (cf. 359–395). "Estimator map" (1864–1896) restates §3 (578–727). Spline sizes (1974–1979) restate (A5). Cut to pointers and keep only what is new: the stability-condition sentence (1825–1830), Lemma ds-shift, the grid and data-driven paragraphs. The density-bias explanation appears twice: 1982–1985 and 2451–2452.
- *Vague restatement of (A3)* at 1833–1835: "positive wherever the endpoint and utility expansions require it". Cite (A3) and V instead.
- *Identification inequality called a "condition"* (2136–2137, 1843–1845: "supplies the remaining identification condition") although Lemma ds-gram derives it from (A1)(ii). Call it a consequence.
- *Duplicate symbols.* B_phi (2091) is used once and equals ell_2 (2207) at phi*; keep one. h_delta = h_{gamma+delta} defined at 2379 and again at 2605. b_v defined as F*(v) f_eta(v) (2767) and as F(v) f_eta(v) in Lemma ds-eval (3797); equal at the closed endpoints, but say so or pick one. P_q defined at 570 and 2066–2068.
- *Other overloads:* v is a shift (1929), a sieve coefficient (2265, 2273), an endpoint label in b_v (2767) and a path direction (1095); h_n for the grid mesh (1904) collides with the h family; C is a generic constant and also C_k, C-hat, C_delta coefficient matrices (2394–2400, 2700); m_0 (regression function, 2303) next to m_p and M_0(e); pi^{(d)} (2203) vs pi(phi) (533). The F family needs one sentence: F_phi carries no I_0, F* = F_{phi*} I_0 does (2072, 2747).
- *Notation table:* yes. About 35 appendix-only symbols are introduced across 900 lines (nu, S, z_K, Sigma_K, ell_K, ell_2, ell_{alpha,K}, r_out, r_{out,K}, r_T, r_{T,L}, Pi_L, breve N_L, T_delta, eta_delta, h_delta, h_{X-tilde}, h_rho, xi, s*, A_U, A_out, A_T, A-bar, b_v, m_p, zeta_p, E_o, E_T, R_prod, R_U, L_out, L_T, pi^{(d)}, mu^{(d)}, Sigma^{(d)}, Gamma^{(d)}, omega^{(d)}). A table with the defining place would replace most of the forward pointers.
- *Key ideas before use.* (i) why A_U has only an intercept component (2764–2770 asserts "carries no slope loading"); one line — E[X-tilde d_e{I_0 F}(eta)] = e_1 E[d_e{I_0 F}] by X independent of eta — would make it predictable; (ii) S is used at 2062 ("the oracle outcome-fitting indicator S") before its definition at 2110.
- *One-line link missing in Step 1a:* state E_o = L_out(m-hat^0 - m_0) explicitly before 2319.
- *Table 1 row 5* lists only "Block A"; add Lemma ds-zero-intercept and the two corollaries.

**Works well.** Step order matches §4.4 exactly, and each step opens with a one-sentence purpose. Lemma ds-dens-mean is a clean device. Lemma ds-rout's closed form and proof are checkable line by line. Sign bookkeeping between Upsilon and U_eps' is explicit. Lemma ds-dens-bias's L1 x L_inf pairing is explained. The proof of Corollary 1 is short and complete. Honest flags: "which we do not prove" (2467), "Neither argument covers full-sample reuse" (2964).

---

# Referee report: Appendices 3–5 (Fable 5.1, 2026-11-09; line numbers refer to master eb1825b)

**Verdict.** Appendix 3 (PDF pp. 36–56) is mathematically careful and, read lemma by lemma, well written: results are stated before they are used, interpretive paragraphs follow the hard results, and every step checked is right. Its weakness is navigation: 1,860 lines with a five-line preamble that names only half the contents, block letters A, C, D with no B, simulation tables and design names inside proofs, and a 700-line "extensions" tail (pseudo-target plus taxi) that a Section 6 reader cannot find from the main text. Appendices 4–5 mix two valuable checks (four-role variance ratio, identification identities) with lab-protocol material whose results the paper never reports.

**Must fix (ranked)**

1. *Map and labels for Appendix 3* (2970–2975, 2023–2025). The opening omits the opposing/untrimmed/eps-sensitivity results, Lemma ds-outcome, the pseudo-target part and names the taxi part only as "Near its end". Add an itemized map. Letter every part or none: "Block A" first appears at 3710, after four unlettered parts, and there is no Block B. Split into Appendix 3 (identification, 2985–3375), Appendix 4 (consistency, Blocks A/C/D, 3376–4102) and Appendix 5 (extensions: pseudo-target and taxi, 4104–4803), renumbering the current 4–5. [Approved as D3.]
2. *Design names used before definition.* "In the four-role check design" (3366, 3372, 3439, 3672) and "the logistic design" (3367) are first defined in Appendix 5 at 5056 and 5103; "displayed Section 5 design" (3364, 3370, 5061) is an odd label. At 5056, "Four-role check design (below)" points the wrong way: that design is above, at 5003–5017. Fix: name the designs once and cross-reference.
3. *Overloaded Lambda.* Lambda(phi) = E_phi lambda(T_phi) (3017, 3046, used through 3130) versus "let Lambda be a bounded linear functional" (3515). Rename the one at 3515.
4. *Simulations inside proofs.* 3669–3708 ("The outcome-fit lemmas were checked by simulation", Table tab:ds-outcome-mc) and the numerical paragraphs 3364–3374, 3439–3443 belong with the numerical checks; the gap/bound numbers 0.1189/0.1195 and 0.00212/0.00212 at 3370–3374 are repeated verbatim in Table tab:ds-id-checks (5142–5143), and 1.707 appears at 549, 1332 and 3366. Also 3675 says the basis grows "like n^{0.185}" while the paper's rule is 11/60 (1174); state the rule. Move these to the numerical-checks appendix and cite them from the lemma.
5. *Appendices 4–5 lab-notebook content.* [Report only, NOT for the editing pass: the author asked for a separate report on this.] Also 1682 sends the reader to "Appendix 4" for "Additional diagnostics", but that title belongs to Appendix 5.

**Should fix**

- *Taxi part and Section 6.* Section 6.2 (1493–1499) says only that Cor ds-taxi "extends the theorem to this specification". Name there the extra conditions, (C1)–(C3) for the capped window and (S1)–(S3) for distance in two columns, and carry the caveat "We have not checked this cell by cell" (4790) into Section 6.5. The four-way list at 4214–4231 is clear; say at 4236 which result handles each difference (1 → Prop ds-restricted, 2 → Lemma ds-columns, 3 → Prop ds-capped, 4 → assumed).
- *Pseudo-target placement* (4104–4210) interrupts the flow between Block D and the taxi part; it serves Section 6.6, so group it with the extensions. Prop ds-pseudo (4127) imports smoothness of alpha-dagger, b-dagger implicitly via "conditions of Proposition ds-consistency with (alpha-dagger, b-dagger) in place"; since they are projections, state the smoothness as an assumption. [Stating an assumption explicitly that the statement already imports is allowed; do not add new conditions.]
- *Block D sufficient conditions* (3876–3911) are a long unnumbered paragraph; (A6)'s pointer "Sufficient conditions … Appendix 3, Block D" (834) should land on a numbered lemma parallel to Lemma ds-columns.
- *Lemma ds-outcome setup* (3445–3475) packs theta_K-circ, theta-bar_K, the basis continuation and the max residual error into one paragraph; itemize. In (e) the key idea, "bounded separately, each would only be O_p(K^{-3}) and would need K >> n^{1/6}" (3641–3643), deserves a lead sentence. Bare "Step 1a", "Step 2" (3462, 3639) should read "Appendix 2, Step 2" as at 3379.
- *Wording.* 3112–3113 "positive wherever one of the two is positive": say C_alpha or C_lambda + B_eps. 4668–4672 (tie share 61.7%) is data detail; one sentence in Section 6.5 suffices.
- *Redundancy.* "Target and trim window" (2977–2983) restates Sections 2.3/3.2; "Remarks on the regularity conditions" (4805–4827) repeats 836–870. Either cut or merge into the map of item 1.

**Works well.** Interpretive paragraphs after each identification result and the two explicit counterexamples (3224–3258, 3260–3287) make the single-crossing condition concrete. Block C states its key idea plainly. The Bahadur argument (4011–4097) is self-contained and uniform in a. The capped-window lemma's trimmed-mean reading (4664–4666) and the change table in Prop ds-capped (4682–4694) make the taxi scores easy to audit, and the four-role variance check is a genuine test of Cor ds-rotation.
