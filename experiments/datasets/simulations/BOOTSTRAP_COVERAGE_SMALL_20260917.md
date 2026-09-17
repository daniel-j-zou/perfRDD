# Full re-estimation bootstrap coverage pilot

## Question

Do nominal 95% full re-estimation bootstrap intervals cover the known hard-trim
policy threshold in the theorem-aligned simulation DGP?

## Design

The runner is `experiments/scripts/hard_trim_bootstrap_coverage.py`.  This pilot
uses the full-sample estimator (`--estimators full`) with hard trimming
(`eps=0.1`, `cost=2.25`), independent draws, 20 outer Monte Carlo samples per
cell, and 99 bootstrap re-estimations per outer sample.  We vary
`n in {1200, 2400}` and use Student-t(5) and two-component-mixture
running-variable laws.  The
percentile and normal bootstrap intervals both use the 2.5% and 97.5% bootstrap
quantiles.  No bootstrap fit failed in any cell.

Reproduce with:

```sh
cd /Users/zoudj/Documents/Codex/PerfRDD/code
PYTHONPATH=. python3 -m experiments.scripts.hard_trim_bootstrap_coverage \
  --n 1200 2400 --outer-reps 20 --bootstrap-reps 99 --workers 4 \
  --laws t5 mixture --estimators full \
  --out experiments/runs/bootstrap_coverage_small_20260917
```

## Results

| Running-variable law | n | Known target | Mean bootstrap SD | Percentile coverage | Normal coverage |
|---|---:|---:|---:|---:|---:|
| t5 | 1,200 | 0.5878 | 0.2197 | 0.95 | 1.00 |
| t5 | 2,400 | 0.5878 | 0.1254 | 0.95 | 1.00 |
| mixture | 1,200 | 0.7930 | 0.3570 | 0.95 | 0.95 |
| mixture | 2,400 | 0.7930 | 0.2506 | 0.95 | 0.95 |

The coverage values are proportions over only 20 outer samples, so their Monte
Carlo standard error is about 0.05 near 95%.  The pilot is therefore a useful
sanity check (intervals are centered and numerically stable), not evidence of a
bootstrap validity theorem.  A larger replication study is needed for a
publication-quality coverage claim.

Raw outputs are retained in
`experiments/runs/bootstrap_coverage_small_20260917/summary.json` and
`replications.csv`.

-- Codex, 2026-09-17
