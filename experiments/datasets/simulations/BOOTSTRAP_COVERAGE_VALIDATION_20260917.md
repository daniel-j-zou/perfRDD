# Full re-estimation bootstrap coverage validation

## Question

Does the nominal 95% full re-estimation bootstrap confidence interval cover the
known hard-trim policy threshold when coverage is estimated with enough outer
replications to be informative?

## Design

The runner is `experiments/scripts/hard_trim_bootstrap_coverage.py`.  We use
the full-sample hard-trim estimator (`eps=0.1`, `cost=2.25`) with independent
draws, 200 outer Monte Carlo samples per cell, and 199 bootstrap
re-estimations per outer sample.  The four cells match the earlier pilot:
`n in {1200, 2400}` and Student-t(5) or two-component-mixture errors.  Both
percentile and normal intervals are evaluated against the known population
threshold.  The run produced 159,200 successful bootstrap fits and zero
failures.

Reproduce with:

```sh
cd /Users/zoudj/Documents/Codex/PerfRDD/code
PYTHONPATH=. python3 -m experiments.scripts.hard_trim_bootstrap_coverage \
  --n 1200 2400 --outer-reps 200 --bootstrap-reps 199 --workers 8 \
  --laws t5 mixture --estimators full \
  --out experiments/runs/bootstrap_coverage_validation_20260917
```

## Results

| Error law | n | Known target | Mean bootstrap SD | Percentile coverage | Normal coverage |
|---|---:|---:|---:|---:|---:|
| t5 | 1,200 | 0.5878 | 0.2249 | 0.935 (187/200) | 0.980 (196/200) |
| t5 | 2,400 | 0.5878 | 0.1228 | 0.950 (190/200) | 0.975 (195/200) |
| mixture | 1,200 | 0.7930 | 0.3591 | 0.945 (189/200) | 0.925 (185/200) |
| mixture | 2,400 | 0.7930 | 0.2595 | 0.945 (189/200) | 0.925 (185/200) |

The percentile intervals are close to nominal in every cell.  Their binomial
95% Wilson intervals are approximately `[0.89, 0.96]` for 0.935 coverage,
`[0.91, 0.97]` for 0.950, and `[0.90, 0.97]` for 0.945, so none rejects a
95% coverage benchmark at this resolution.  Normal intervals are conservative
under t5 and mildly under-cover for the mixture (0.925), suggesting that the
percentile interval is the safer default for this heavy-tailed diagnostic.

This is strong finite-sample evidence that the implementation behaves
sensibly, but it remains a simulation validation rather than a bootstrap
validity theorem.  The experiment does not address clustering, data-dependent
bandwidth selection, or the theorem's fully decoupled influence-function
construction.

Raw outputs are retained in
`experiments/runs/bootstrap_coverage_validation_20260917/summary.json` and
`replications.csv`.

-- Codex, 2026-09-17
