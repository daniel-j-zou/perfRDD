# GPA hard-trim application status

## Reproduction

Run from the code-repository root with the project Python environment:

```sh
/Users/zoudj/miniconda3/envs/srsd/bin/python \
  -m experiments.scripts.gpa_redesign_hard_trim
```

The command writes the complete machine-readable results and diagnostic figures to
`experiments/runs/gpa_redesign_hard_trim/`. That directory is intentionally ignored by
Git; this file records the durable result summary and the script reproduces it.

## Work completed on 2026-08-31

- Replaced the earlier smooth-gate application with the exact hard indicator
  `1{l_hat <= eta_hat <= u_hat}`.
- Kept the outcome redesign that separates full-population persistence, the selected
  observed-next-GPA diagnostic, and explicit full-population composite sensitivities.
- Used one locked nuisance support, `(-2, 0)`, for every outcome and specification.
- Ran four full-sample ridge scales—0, 0.0001, 0.001, and 0.01—and an unregularized
  five-fold cross-fit robustness estimate for all 14 outcomes (70 fits total).
- Retained every specification in `summary.json`; ridge 0.001 organizes the display
  plots but is not selected by its results.
- Re-ran the local-linear RD outcome-definition check with Q-clustered standard errors.

## Locked analysis choices

| Choice | Value | Status |
|---|---:|---|
| Hard-trim probability | `eps = 0.1` | Prespecified |
| Nuisance support | `(-2, 0)` | Pilot-derived and rounded outward |
| Policy grid | `[-0.6, 0.6]`, 241 points | Provisional, matches the central RD bandwidth |
| Direct treatment cost | `c = 0` | Cost omitted pending an author-specified value |
| Primary display | Full sample, ridge 0.001 | Organizational only |
| Robustness | Full ridge grid + five-fold unregularized cross-fit | All retained |

The support came from the August 2026 pilot diagnostic. These results are exploratory,
not confirmatory, because the support and policy grid have not yet been justified on
independent scientific grounds.

## Verified point estimates

The full population has 44,362 observations; the selected observed-next-GPA diagnostic
has 40,582. The hard window retains 35.34% of the full population and 31.85% of the
selected diagnostic. `RD` and `RD SE` below are the local-linear validation estimate and
its Q-clustered standard error. `Hard alpha` is the hard-window average treatment-effect
nuisance estimate. `CF alpha` is its five-fold unregularized counterpart.

| Outcome | RD | RD SE | Hard alpha | CF alpha | phi* |
|---|---:|---:|---:|---:|---:|
| Fall return | -0.0830 | 0.0132 | -0.0833 | -0.0852 | -0.6 |
| Not left voluntarily | -0.0177 | 0.0069 | -0.0165 | -0.0178 | -0.6 |
| Subsequent GPA recorded | -0.0348 | 0.0110 | -0.0277 | -0.0293 | -0.6 |
| Composite: no-record GPA 0.0 | 0.1384 | 0.0290 | 0.1433 | 0.1447 | 0.6 |
| Composite: no-record GPA 0.8 | 0.1662 | 0.0245 | 0.1662 | 0.1681 | 0.6 |
| Composite: no-record GPA 0.9 | 0.1697 | 0.0241 | 0.1691 | 0.1711 | 0.6 |
| Composite: no-record GPA 1.1 | 0.1766 | 0.0235 | 0.1748 | 0.1769 | 0.6 |
| Composite: no-record GPA 1.5 | 0.1905 | 0.0227 | 0.1863 | 0.1886 | 0.6 |
| Composite: zero GPA, penalty 2 | 0.0689 | 0.0462 | 0.0860 | 0.0861 | 0.6 |
| Composite: zero GPA, penalty 4 | -0.0006 | 0.0663 | 0.0287 | 0.0276 | 0.6 |
| Composite: zero GPA, penalty 5 | -0.0354 | 0.0768 | -0.0000 | -0.0017 | -0.6 |
| Composite: zero GPA, penalty 6 | -0.0702 | 0.0874 | -0.0287 | -0.0310 | -0.6 |
| Composite: zero GPA, penalty 8 | -0.1397 | 0.1089 | -0.0860 | -0.0896 | -0.6 |
| Observed next GPA, selected diagnostic | 0.2333 | 0.0260 | 0.2329 | 0.2333 | 0.6 |

All displayed `phi*` values are grid-boundary solutions in both the primary display and
cross-fit specifications. Thus the current no-cost criterion establishes stable signs
and close agreement between full-sample and cross-fit point estimates, but it does **not**
identify an interior welfare-optimal threshold. A meaningful interior policy comparison
requires a defensible cost and policy domain.

## Remaining limitations

- The hard-trim application currently reports point estimates only. It does not yet
  implement the theorem's boundary-aware influence-function variance.
- The observed-next-GPA outcome conditions on a post-treatment event and remains a
  diagnostic rather than a full-population policy estimand.
- The composite outcomes encode explicit values for the no-subsequent-record state; they
  are sensitivity analyses, not recovered missing GPAs.
- The pilot-derived nuisance support and provisional policy grid must be justified or
  replaced before the application is described as confirmatory.
