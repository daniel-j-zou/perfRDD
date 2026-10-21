# Nonlinear alpha, b design: is the analytic variance right? (2026-10-20, Claude)

Requested by the author in the weekly scope review of 2026-10-20: "check the analytic
value first". In the nonlinear design of `differing_slopes.tex` §5 (script
`experiments/scripts/ds_four_role_coverage.py`, design `nonlinear`), n Var of the
rotated estimator stayed 15-19% above the first-order value 4.73 for n = 3,200 to
12,800 in the coverage run (500 data sets per n, 300 at 12,800).

The check was done by an independent agent that did not write the variance code. Its
scripts are in `experiments/scripts/ds_nonlinear_variance_check/`, with small JSON
outputs beside them. The point-estimate Monte Carlo rows (`decomp_n*.jsonl`, about
3 MB each) are in the ignored `experiments/runs/ds_nonlinear_variance_check/`.

## 1. The analytic value is right

`indep_pieces.py` recomputes every piece from the formulas of the paper (§4.4 and
Appendix 2), by adaptive quadrature with its own truncated-normal arm moments. It does
not import the project code.

| piece | independent value | `population_summary()` |
|---|---|---|
| phi* | 0.15214329 | same to about 1e-15 |
| H | -0.21953920 (finite-difference U'' agrees to 3e-9) | same |
| Var psi_U | 0.00180924 | same |
| Var psi_o (E r_out^2 = 0.8692550) | 0.217314 | same |
| Var psi_T | 0.00875272 | same |
| Var psi_gamma (A_bar = (0, -0.004643, -0.006446)) | 6.311e-5 | same |
| sigma^2 | 0.227939 | same |
| sigma^2 / H^2 | **4.7293** | 4.7293 |

Further checks:
- `mc_scores.py`: Monte Carlo of the four scores with N = 2e7 gives 0.0018098,
  0.21759 (s.e. about 1.2e-4), 0.0087551 and 6.31e-5. All six cross-covariances
  are below 1.5e-5, as Corollary 1 says.
- The outcome representer without the closed form: the finite-sieve Riesz
  representer gives ||r_K||^2 = 0.869254997, which agrees to 1e-11.
- A_U, A_T and the endpoint loadings match central finite differences of the
  population map delta -> Upsilon(phi*) to 1e-9.

ψ_o carries 95% of sigma^2.

## 2. Where the excess comes from

`mc_decomp.py` re-implements the spline arm of `ds_four_role_coverage.estimate`.
It uses the same calls and seeds and reproduces the stored estimates exactly. On the
same data sets it also evaluates the first-order proxy
phi* - H^{-1} P_all(psi_U + psi_o + psi_T + psi_gamma), whose n Var is exactly 4.729.

The proxy correlates 0.96-0.99 with the estimator. So the control-variate estimate
4.729 + n {Var(est) - Var(proxy)} has a much smaller Monte Carlo error than the raw
n Var (`paired.py`, 2,000 point-estimate data sets per n, no bootstrap):

| n | raw n Var (avg) | proxy n Var | control-variate estimate | ratio to 4.729 |
|---:|---:|---:|---:|---:|
| 3,200 | 5.53 (0.19) | 4.88 | 5.39 (0.08) | 1.14 |
| 6,400 | 4.91 (0.16) | 4.50 | 5.14 (0.05) | 1.09 |
| 12,800 | 4.95 (0.15) | 4.93 | 4.74 (0.04) | 1.00 |
| 25,600 | 4.50 (0.15) | 4.52 | 4.71 (0.03) | 0.995 |

- At n = 25,600 the fixed assignment gives 19.04 against 4 x 4.729 = 18.92.
- The cross-rotation correlation is about 0, as for the proxy.
- Of the stored coverage data sets, n = 12,800 (reps 0-299) gives an estimator
  n Var of 5.63. The first-order proxy alone gives 5.75 on those same data sets, so
  that excess is Monte Carlo noise.

**Cause: the density basis is small at small n.** The rule gives L = 8 density basis
functions for n <= 6,400 and 9 at 12,800. With L = 8 the curvature is biased:

| L | sieve H | mean of H-hat |
|---:|---:|---|
| 8 | -0.2147 | -0.2138 (3,200), -0.2142 (6,400) |
| 9 | -0.2207 | -0.2204 (12,800) |
| 10 | -0.2197 | -0.2194 (25,600) |

The true H is -0.2195.

`sieve_variance.py` gives the first-order variance at the script's sieve sizes:

| L (n) | sigma^2/H^2 |
|---|---:|
| 8 (n <= 6,400) | 4.92 |
| 9 (12,800) | 4.62 |
| 10 (25,600) | 4.70 |
| 11 (51,200) | 4.72 |

The outcome knots (3 or 4) change it by about 1%.

The curvature conversion adds about 10%, 7.5% and 0% per rotation at 3,200, 6,400
and 12,800, including the L = 8 bias. The score itself is within 3.6%, 1.5% and 0% of
its linear approximation.

**Ruled out:**
- heavy tails (excess kurtosis about 0 for n >= 3,200);
- cross-rotation correlation;
- the pooled-versus-average second-order term, because the pooled maximizer has a
  smaller excess (1.10 and 1.07).

## Reading

The first-order theory and its variance formula are confirmed in this design. The
15-19% excess in the coverage table is a finite-sample effect at n <= 6,400, mostly
from the 8-function density basis, plus Monte Carlo noise at 12,800. A larger-n
bootstrap run is not needed to settle the variance question. Whether to add one is
the author's call.
