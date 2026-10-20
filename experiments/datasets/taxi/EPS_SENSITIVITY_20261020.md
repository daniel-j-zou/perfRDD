# Taxi: sensitivity of the four-role estimate to the trim level eps (2026-10-20, Claude)

At the author's request (talk about how the estimator changes with eps, in the
simulations and the taxi data). Script: `experiments/scripts/taxi_eps_sensitivity.py`;
results in the ignored `experiments/runs/taxi_eps_sensitivity/summary.json`.

## Setup

The nine-way comparison's setup (`taxi_four_role_grid`, see `FOUR_ROLE_APPLICATION.md`),
with effect E1 (linear in distance) and the three scores S1-S3:
- fare in dollars; J = central 99.5% of eta-hat; trim window capped at the central 99%;
- policy set $6-$24; deployed cutoff $15; 61-point grid plus refinement;
- one four-role split (seed 1) shared across eps;
- S2 only: 20 full re-estimation bootstrap draws, 90% interval = estimate +- 1.645 sd.

At every eps the upper trim endpoint lies beyond the observed residuals, so the cap
sets it ($5.28-5.39). Only the lower endpoint moves: raising eps drops the rides with
the lowest eta-hat, the trips charged least time for their length and hour.

## Results

Estimate phi-hat in dollars (fare cell in brackets) and the share of rides in the
trim window.

| eps | share (S2) | S1 | S2 | S2 bootstrap sd | S3 |
|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.990 | 11.87 (12.10) | 11.76 (12.10) | 0.25 | 13.02 (13.30) |
| 0.075 | 0.961 | 12.22 (12.50) | 11.75 (12.10) | 0.26 | 12.99 (13.30) |
| 0.100 | 0.508 | 11.77 (12.10) | 11.75 (12.10) | 0.30 | 11.49 (11.70) |
| 0.125 | 0.158 | 11.71 (12.10) | 10.61 (10.90) | 0.40 | 7.13 (7.30), edge |
| 0.150 | 0.065 | 11.62 (11.70) | 9.83 (10.10) | 0.85 | 6.00, edge |
| 0.200 | 0.019 | 10.57 (10.90) | 9.30 (9.70) | 0.93 | 8.19 (8.50) |

Shares under S1 and S3 are within 0.11 of S2's. The eps = 0.05 rows reproduce the
nine-way comparison (S2 interval 11.36-12.17).

- **eps <= 0.075.** At least 94% of rides stay in the window, and each score's
  estimate moves by at most $0.35.
- **eps = 0.10.** 40-55% of rides remain. S1 and S2 stay within $0.50 of their
  eps = 0.05 values (S2 moves by $0.01); S3 drops by $1.53 to $11.49.
- **eps >= 0.125.** Fewer than 20% of rides remain. S2 falls to $9.30 and its
  bootstrap sd rises to $0.93; S3 reaches the $6 lower edge of the policy set in
  two of four rotations at 0.125 and in all four at 0.15; S1 stays near $11.7
  until 0.20.
- The gain over the deployed $15 is 0.1-2.2 cents per ride throughout.

## Reading

While the window keeps most rides, the estimate is stable given the score. Once
eps removes more than half of them, the target population becomes the slow trips
only, and the estimate follows it, first under S3 and then under S2. The specification choice (S2 vs S3)
moves the estimate more than eps does in the stable range. Nothing in the paper
changes; §6 stays separate until the author chooses the taxi specification.

```bash
python -m experiments.scripts.taxi_eps_sensitivity --boot 20 --workers 5   # about 10 min
```
