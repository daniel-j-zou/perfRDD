# PerfRDD research log — shared between Claude and Codex

This tracked file is the single source of truth for **verified findings and decisions**.
The operating rules are in `COLLABORATION.md`; forward-looking work is tracked in
`../manuscript/TODO.md`.

Add new entries immediately below the divider, newest first. Date and sign every entry.
Never edit or delete another agent's entry; add a follow-up when a conclusion changes.

---

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
