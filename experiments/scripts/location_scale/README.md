# Location-scale first stage: simulation checks (2026-10-21)

These scripts support `this_week.tex`, "A location-scale first stage". They were written
by independent reviewer agents and by Claude during the theory check. They are plain
scripts, not modules: each imports `common.py` from its own folder and writes its
output to the current directory. Run them from the ignored runs folder, which holds the
outputs used in the write-up:

```bash
cd experiments/runs/location_scale_sim
PY=/Users/zoudj/miniconda3/envs/srsd/bin/python
S=../../scripts/location_scale
$PY $S/target.py                                    # target_results.json: phi*, window, identity check
$PY $S/mc.py mc_main.jsonl normal:4000:200:0 normal:16000:100:1 normal:64000:50:1
$PY $S/mc.py mc_extra.jsonl normal:64000:50:1:50    # second half of n = 64000
$PY $S/mc.py mc_t5.jsonl t5:16000:100:0
$PY $S/summarize.py mc_main.jsonl mc_extra.jsonl mc_t5.jsonl
$PY $S/fix_checks.py scale                          # Harvey vs Poisson scale fit
$PY $S/fix_checks.py smooth > fix_smooth.log        # smoothed vs step-function criterion
$PY $S/bign.py 4e5 0 bign                           # one run at n = 400,000
```

## Design (`common.py`)

| piece | value |
|---|---|
| covariates | X1 ~ N(0,1); X2 = Bernoulli(1/2) - 1/2 |
| index | T = X1 + 0.5 X2 |
| scale | s = exp(0.4 X1 + 0.2 X2) |
| standardized residual | zeta ~ N(0,1), independent of X (also t(5), standardized) |
| score and treatment | Q = T + s zeta, phi0 = 1, D = 1(Q > phi0) |
| outcome | Y = sin(zeta) + D alpha(zeta) + X'(0.5, 0.2) + D X'(0.3, 0) + N(0, 0.25) |
| effect function | alpha(zeta) = 0.4 zeta - 0.35 + 0.1 tanh(2 zeta) |
| trim and policy set | eps = 0.05; Phi = [-0.2, 2.2] |

The estimators in `common.py` are the four-role estimator with location-scale and with
additive (s = 1) first stages. Both use:
- a Harvey scale fit;
- arm-wise cubic B-splines with about n^{1/5} interior knots on the 0.25% arm-trimmed J;
- the empirical index law;
- grid search plus refinement for the argmax;
- the average over the four rotations.

`fix_checks.py` adds two pieces:
- the Poisson pseudo-likelihood scale fit;
- a smoothed criterion: Gaussian kernel density of the standardized residual on the
  utility fold, bandwidth sd * n^{-1/3}, with an analytic derivative.

Note: the trim window here is not capped inside J. So alpha-hat is extrapolated over
about 1% of the window above the untreated arm's fitted range, and the estimator's
large-n limit is phi* + 0.006.
