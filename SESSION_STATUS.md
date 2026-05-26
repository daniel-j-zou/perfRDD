# Trimmed-estimator session status — 2026-05-26

Snapshot of where we are in the trimmed-estimator empirical work.

## What's done (committed and pushed)

| commit | content |
|---|---|
| `3e38edd` | `perfrdd.py` intercept fix (unpenalised intercept in PLM design). |
| `990f6b2` | New `experiments/methods/perfrdd_trim.py` + `experiments/scripts/four_datasets_trim_compare.py`. Trimmed estimator + side-by-side comparison with standard, on the 4 linear-Q datasets (GPA, NHANES, OULAD, Lending Club). |
| `7811273` | Five prior-session analysis scripts recovered (`four_l2_*`, `lc_b_dip_*`). |
| `5fd4ae8` | `experiments/scripts/four_datasets_trim_eps_sweep.py` — first ε sweep (8 values, 0.02–0.4). |
| `c960155` | Three sensitivity scripts: `three_datasets_trim_eps_fine.py`, `synthetic_trim_overlap.py`, `four_datasets_trim_bootstrap.py`. |

## What's run (outputs in `experiments/runs/`)

- `four_datasets_trim_compare.{png,json}` — main comparison, ε=0.1, 4 datasets.
- `four_datasets_trim_eps_sweep.{png,json}` — first ε sweep, 8 ε values, 4 datasets.
- `three_datasets_trim_eps_fine.{png,json}` — finer ε sweep (0.005–0.2), 3 datasets (GPA/NHANES/Lending Club only — OULAD too unstable here).
- `synthetic_trim_overlap.{png,json}` — synthetic DGP study v2 with α(η)=η centred. True φ*(c) computed by 200k-MC. 25 seeds × 7 ε values × 2 scenarios.
- `synthetic_trim_overlap_2x2.png` — replotted 2×2 layout (scenarios × {c=0, c=0.5}) from the same JSON. **Best synth figure — uses this.**

## What's running (background)

Bootstrap CIs (`four_datasets_trim_bootstrap.py`):
- Started 2026-05-26 ~19:51 (local).
- Background task ID: `bfha7byw2` (output `/private/tmp/claude-505/.../bfha7byw2.output`).
- B=30 bootstrap resamples per (dataset, ε), 4 datasets × (1 std + 6 ε values).
- Expected total: 4 × 7 × 30 = 840 fits, ~20 min wall.
- Outputs: `experiments/runs/four_datasets_trim_bootstrap.{png,json}`.

## What's pending (not yet committed)

- `experiments/scripts/_replot_synthetic.py` — replots the synth study with the cleaner 2×2 layout (scenarios × cost levels). Run and produced `synthetic_trim_overlap_2x2.png`.
- Once bootstrap finishes:
  - Inspect `four_datasets_trim_bootstrap.png`.
  - Possibly polish the figure (similar layout issues likely).
  - Commit + push the bootstrap figure / refined script and the replot script.
  - Compose final summary report for the user (the three tasks they requested).

## Key empirical findings so far

1. **Standard estimator's `α̂` is biased upward at no-overlap regions on three of four datasets** (GPA, NHANES, OULAD — see the original `four_datasets_trim_compare.png`).
2. **The trimmed estimator converges as ε → 0 but NOT necessarily to the standard estimator's φ*.** For GPA at ε = 0.005 the gap is ~0.35; for Lending Club at c=0 it's exactly 0 (both at the grid boundary).
3. **Synthetic study confirms the trimmed estimator is closer to the TRUE φ*** especially on the bad-overlap scenario:
   - bad scenario, c=0: trim med 0.01 (truth 0.01); std med −1.8. **Trim ≫ std.**
   - bad scenario, c=0.5: trim med 0.95 (truth 0.53); std med 0.89. Close call.
   - good scenario, c=0: trim med −0.03 (truth 0.005); std med −0.41. Trim better.
4. **OULAD's wild oscillations in φ* across ε** are likely sampling noise — the bootstrap will quantify this.

## Resume instructions if interrupted

If I (the assistant) get disconnected:

1. Check whether `experiments/runs/four_datasets_trim_bootstrap.png` exists. If yes, bootstrap finished — proceed to compose the summary.
2. If not, check whether the bg process is still running: `ps aux | grep -E "python.*bootstrap" | grep -v grep`. If running, wait. If not (and no png), rerun with `python -m experiments.scripts.four_datasets_trim_bootstrap`.
3. Commit `experiments/scripts/_replot_synthetic.py` (not yet committed) before pushing the bootstrap.
4. Produce the final summary: the three figures the user wanted are
   - `three_datasets_trim_eps_fine.png` (task 1)
   - `synthetic_trim_overlap_2x2.png` (task 2)
   - `four_datasets_trim_bootstrap.png` (task 3, pending)
