# Proof simplifications for `differing_slopes.tex`: checks (2026-10-13, Claude)

On 2026-10-13 Claude reviewed the proof of Theorem 1 in `differing_slopes.tex`
(upstream master after a591181) and proposed simplifications. At the author's
request ("make sure these actually work"), each proposal was checked before the
author decides which to integrate. Nothing in the paper or the notebook changes.

Script: `experiments/scripts/ds_simplification_checks.py`
(results in `runs/ds_simplification_checks/checks.json` and
`simulation_summary.json`; tests in `tests/test_ds_simplification_checks.py`).
The checks use quadrature with closed forms for Gaussian-mixture X wherever
possible, so they do not rely on the sieve expansions they check.

Designs: `paper` (the paper's displayed design: Gaussian X in 3 dimensions,
Gaussian eta, phi* = 0.545103); `mixx` (X a two-component Gaussian mixture in 2
dimensions, so not elliptical; eta a skewed normal mixture; nonlinear a;
phi0 = 0.2; phi* = 0.518761); `mixx_raw` (`mixx` with E X = (0.8, -0.5), same
target); `kink` (paper design with a, b in C^{2,1}, third derivatives jumping at
the trim points, the least favourable placement for the bias).

Notation: h1(t) = E[X~ | T=t] f_T(t) = (g, rho_X')', h2(t) = E[X~X~' | T=t] f_T(t),
w(e) = (a(e) - c, beta2')', merged representer
r(t) = w(phi*-t) f_eta(phi*-t) 1{phi*-u0 <= t <= phi*-l0}, A_dens = -int h2' r = A_g + A_rho.

## Verdicts

| # | Proposal | Verdict |
|---|---|---|
| 1 | Condition on the first-stage fold; first-stage effects are derivatives of population maps | Verified numerically and by an independent re-derivation. Covers the full clamped basis the code uses. Needs two conditions the paper lacks, (C1) and (C2) below. |
| 2 | Merge g and rho_X into one block with multiplier X~ | Verified (exact identity); the code already does this |
| 3 | Arm-wise outcome fit and closed-form representer | Verified numerically, Gaussian and non-Gaussian X |
| 4 | Center X; use location invariance first | Verified: exact invariance needs one shared density basis |
| 5 | Center the outcome expansion at the population LS projection | Verified (identity checked to 1e-15) |
| 6 | Rate range | n^{1/7} << K << n^{1/4}, n^{1/8} << L << n^{1/4} confirmed by the remainder list below, provided three lemmas are rewritten (listed there) and g, rho_X are C^3 on all of the density interval. K >> n^{1/8} is suggested by the numerics but not established. Not yet written as a proof. Simulation: see below. |
| 7 | Condition housekeeping | Confirmed by reading; the 8th-moment claim refined below |
| C | Role scores as pathwise derivatives | Verified for the law-of-X direction (the only non-trivial one) |

## 1. Conditioning on the first-stage fold

Given delta = gamma_hat - gamma (independent of the other folds), each later fold
is an iid sample at a fixed shift. Three facts carry the argument:

* **Exact conditional mean.** E[Omega_hat | delta] = G_L^{-1} int N_L h1_delta', the
  projection of the X~-weighted density of the shifted index. Monte Carlo (L = 12,
  n = 2000, 2000 reps): max |z| = 2.0 over 48 coefficients.
* **Density-block loading at finite L.** Theta_L(delta) = -int_T r_L' h1_delta has
  gradient int_T (dh2/dt) r_L, with no Taylor expansion of the basis. Central
  differences agree to 4e-12 (paper) and 6e-12 (mixx) for L = 8 to 96, full and
  zero-trace bases. With the full basis, the paper's Taylor coefficient
  -int h2 r_L' differs from it by the boundary term [h2 r_L]_A^B (an
  integration-by-parts identity); the term is 1.2e-3 at L = 8 and 2.6e-5 at
  L = 32 in the paper design. With the zero-trace basis it vanishes, as the
  paper says. r_L at the interval ends falls roughly geometrically in L (0.12 at
  L = 8, 7e-7 at L = 96), so the Taylor route could also be repaired by bounding
  that term; the conditioning route does not need it. The second-order remainder
  |Theta_L(delta) - Theta_L(0) - grad'delta| / |delta|^2 is 0.10 to 0.14 for every
  L, flat in L, so the paper's O(L^{5/2}|delta|^2) bound is not attained and no
  upper limit on L comes from this step. The full-basis loading converges to its
  limit faster than the zero-trace one (2e-7 against 3e-6 at L = 24, paper design).
* **Population maps (no sieve).** The index-fold map (densities and quantiles of the
  shifted index, a and beta2 fixed) has gradient -A_dens - b_l m_{1-eps} + b_u m_eps:
  agreement 5e-12 (paper), 8e-12 (mixx), 1e-11 (mixx_raw). The evaluation-fold map
  -int_{l0}^{u0} F(e) f_{eta - X~'delta}(e) de has gradient A_U (2e-12, 1e-12,
  6e-12); its covariate block is zero when E X = 0 and nonzero in mixx_raw.
* **Sampling term.** V_L(delta) = E[{(r_L(T + X~'delta) - r_L(T))'X~}^2] satisfies
  V / (L |delta|^2) <= 0.009 for L up to 128 and |delta| from 0.003 to 0.03, so the
  centred sampling difference is O_p(n^{-1/2} (L/n + n^{-1/2})^{1/2}) and imposes
  no rate condition.

What the route removes: Step 3b's Taylor expansion and its K^{5/2}/n remainder,
Block E and the fixed-support density repair, the zero-outer-trace construction,
the matrix Block E listed as sketched, most of Block C and Step 3c (and with them
the need for uniform consistency of a_hat', which Step 3c used). The zero trace
is used in Step 3b, the density consistency lemma, Block E and its proposition,
the notebook's weighted Block E, and the text of Section 3 and Appendix 1; it is
not used by the U^T constant, Block A, Blocks C and D or location invariance.
Conditioning replaces every use, since ||P_L(h1_delta - h1)||_2 <= ||h1_delta - h1||_{L2(T)}.

What it needs, beyond the paper's conditions:

* **(C1)** delta -> h1_delta, the X~-weighted density of (gamma + delta)'X~, is
  differentiable at 0 as a map into L2 of the density interval, up to and
  including its endpoints, with derivative -dh2/dt and an O(|delta|^2) remainder
  uniformly for small delta. A sufficient version is the notebook's (A5) on a
  neighbourhood of the interval. (DS4) and (DS6) constrain only delta = 0 objects
  or the trim quantiles. Mukherjee et al. (Assumption 3.5) and Wibisono et al.
  (Assumption 3) assume the same kind of condition.
* **(C2)** The bound on crossings of the interval ends needs
  f_T(t) E[||X~||^4 | T = t] bounded on a neighbourhood of the interval, not only
  on it.

All test designs are Gaussian or Gaussian mixtures, smooth across the interval
ends, so they cannot show a failure of (C1). Block D's population expansion uses
the same map; its uniform Bahadur argument stays. The evaluation-fold step needs
only f_eta twice differentiable near the window (already in (DS3)) and
E||X~||^2.

## 2. Merged density block

A_g + A_rho = A_dens to 1e-16 (paper design, against `ds_local_efficiency`). The
code (`methods/weighted_tails.py`) already fits g and rho_X on one basis, so the
merged estimator is the implemented one.

## 3. Arm-wise outcome fit and closed-form representer

* Stacked fit (D N, N, X, DX) and two arm-wise fits give fitted values equal to
  2e-13 and a = w1 - w0, beta2 = d1 - d0 to 4e-14.
* On arm d, r = S_J{s_d I0 g(phi*-eta)/pi_d(eta) + (X - m_d(eta))' omega_d},
  omega_d = s_d M_d^{-1}(B - E[I0 g m_d]). Riesz identity <r, h>_J = L_out(h) by
  Monte Carlo (3e6 draws, 16 directions including X, DX, D sin, cos and 12 random
  ones): max |z| = 1.55 (paper), 1.48 (mixx). r vanishes off J.
* The population sieve representer converges to r at rate K^{-0.46} (paper) and
  K^{-0.55} (mixx), as the jumps imply; A_out,K - A_out is 1e-12 and 9e-11 at K = 64.

## 4. Centering and location invariance

* Shifting X by c = (1.3, -0.7, 2.1) in all folds: with one shared density basis the
  criterion is unchanged to 1e-14 and phi_hat to optimizer tolerance (at most
  1.6e-7 of a fixed-assignment s.d.) in every rotation. With separate bases for g
  (9) and rho_X (12), phi_hat moves by 1 to 5% of a s.d. So exact invariance needs
  the shared basis, which item 2 gives. The paper currently allows different
  dimensions for the two blocks; with them, only first-order equivalence holds.
  A ridge penalty (used in some empirical runs as stabilization) also breaks
  exact invariance.
* mixx_raw (E X != 0) against mixx (centred): Abar_X equal to 1e-15, intercept of
  Abar zero in both (1e-16), A_U covariate block (0.026, -0.016) against 0. The
  individual loadings A_dens,X and A_out,X differ between the two; their sum does
  not. So with E X = 0 the evaluation fold carries no slope loading.

## 5-6. Bias rates and rate range

Slopes below are log-log fits over L, K >= 16; the biases change sign, so they
are indicative only.

Density block (full clamped basis): for these smooth designs the bias
int r'(P_L h1 - h1) falls faster than any rate at issue (slope about -6). The test
that matters uses a C^{2,1} nuisance whose third derivative jumps where r jumps
(least favourable placement): slope -4.2 (paper) and -4.3 (mixx), -3.8 to -4.8 as
the fitting window moves, consistent with the notebook's L^{-4} lemma
(L >> n^{1/8}). ||r - P_L r||_{L1} has slope -1.0 for both bases and
||r - P_L r||_{L2} slope -0.5. The lemma uses a global sup-norm bound, so it needs
g and rho_X in C^3 on the whole density interval; the paper assumes this only
near the evaluation regions.

Outcome block (`kink` design, population-LS centering, K = 6 to 128): the bias
L(m_bar_K - m) equals <r, m_bar_K - m> to all digits shown (item 5). Measured
rates: ||r - Pi_K r||_{L1} slope -1.05, ||r - Pi_K r||_{L2} slope -0.53, sup
approximation error slope -3.28, bias slope -4.1. The bias falls faster than the
Cauchy-Schwarz bound with these rates (-3.8) and about as fast as the L1 bound
(-4.3). The Lebesgue constant of the weighted, X-partialled arm-wise projection
stays between 3.5 and 7.8, the key input of a weighted L1 lemma. This suggests
K >> n^{1/8}, but the sign changes in the tail (K = 48 to 128) make it
suggestive, not decisive, and the weighted lemma is not proved. K >> n^{1/7} is
proved.

Remainders under the conditioning route (from Appendix 2 and the notebook's
outcome-fit, outer-crossing and generated-input lemmas):

| Remainder | Order | Constraint |
|---|---|---|
| Outcome bias | K^{-7/2} (proved); K^{-4} (numerics) | K >> n^{1/7} (n^{1/8}) |
| Outcome Gram at generated residuals | K^{3/2} n^{-1/2} | K << n^{1/3} |
| Generated-input coefficient (Lemma 3a), times delta | K^{3/2} n^{-1/2} + (K/n)^{1/2} | K << n^{1/3} |
| Outer crossing of J, as the notebook writes it (sup norm of a_hat) | K^2 n^{-1} | K << n^{1/4} |
| Outer crossing of J, through the bounded functional l_K'(theta_hat - theta_hat^S) | K^{1/2} n^{-3/4} + K^{3/2} n^{-1} | K << n^{1/3} (needs that rewrite) |
| Basis continuation, max_i |eta_hat_i - eta_i| << 1/K | o_p(n^{-1/2+1/p}) under E||X~||^p | K << n^{1/4} (p = 4), n^{1/3} (p = 6) |
| Density bias | L^{-4} | L >> n^{1/8} |
| Density sampling difference, first-stage mean term | see item 1 | none |
| Curvature: sup |rho_hat' - rho'| on V | L^2 n^{-1/2} + L n^{-1/2} + L^{-2} | L << n^{1/4} |
| Products of outcome and density errors | (sqrt(K/n) + K^{-3})(sqrt(L/n) + n^{-1/2} + L^{-3}) | KL >> n^{1/6}, implied |
| Sup consistency of a_hat, rho_hat (bounded integrand, argmax) | K n^{-1/2}, L n^{-1/2} | K, L << n^{1/2} |

So n^{1/7} << K << n^{1/4} and n^{1/8} << L << n^{1/4} are sufficient, against
the paper's n^{1/6} << K, L << n^{1/5}, once three steps are rewritten:

* the notebook's Lemma 3a bounds each approximation piece by O(K^{-3}), which needs
  K >> n^{1/6}; bound their difference instead, O(K^{-3/2} n^{-1/2});
* the paper's Step 2 bounds ||P_o[S z e_K]|| by O(K^{-3}); use the population-LS
  centering of item 5 (bias identity plus a centred term);
* (DS5) drops the uniform consistency of a_hat', which only Step 3c used.

K << n^{1/3} additionally needs sixth moments and the outer-crossing rewrite.
This list is a derivation, checked independently, not a written proof. The
curvature step is what forces L << n^{1/4}; an equicontinuity argument like the
debiased proof's could relax it.

**Simulation.** The four-role estimator as implemented (OLS first stage, arm-wise
spline outcome fit on J, one full clamped basis for g and rho_X, empirical-quantile
endpoints, rotated average), paper design, 1000 replications per cell, the same
data across configurations. Outcome interior knots n_eff^k (paper's rule
k = 11/60); density dimension 2 n_T^l (paper's rule l = 11/60). Predictions:
rotated N Var 31.09, fixed 124.4. Monte Carlo s.e. of N Var about 4.5%; of
sqrt(N) bias about 0.18 (rotated).

| Configuration | N = 16,000 | N = 32,000 | N = 64,000 |
|---|---|---|---|
| paper rule (k = l = 11/60; L = 9, 10, 12) | 33.38 / +0.84 | 30.49 / +0.52 | 32.82 / +0.58 |
| l = 0.24 (L = 15, 17, 20) | 33.44 / +0.78 | 30.19 / +0.52 | 32.90 / +0.58 |
| l = 0.30 (L = 24, 30, 36; beyond n^{1/4}) | 33.35 / +0.76 | 30.32 / +0.51 | 32.85 / +0.58 |
| k = 0.30 (beyond n^{1/4}) | 33.61 / +0.84 | 30.71 / +0.51 | 32.66 / +0.57 |
| k = 0.30, l = 0.24 | 33.68 / +0.78 | 30.39 / +0.51 | 32.75 / +0.57 |

Entries: rotated N Var / sqrt(N) bias. Fixed-assignment N Var is 140.0 to 140.7,
118.4 to 119.4 and 127.9 to 129.2 at the three N (prediction 124.4).

Reading:

* Dimensions well above the paper's upper limit change nothing: variances
  differ across configurations by less than 1%, inside Monte Carlo error of the
  prediction at every N.
* The upper limits in the table above come from the proof, not from anything
  visible at these N.
* Every configuration, including the paper's own rule, has a small positive
  sqrt(N) bias: +0.84, +0.52 and +0.58 (about 3 s.e. each). In phi that is
  0.0066, 0.0029 and 0.0023, about a tenth of a rotated standard deviation at
  N = 64,000. The fixed assignment shows the same (+0.89, +0.39, +0.61, s.e.
  about 0.36).
* The bias does not depend on K or L, so it is not caused by the dimension
  choices. These N cannot tell an O(1/n) bias (predicted 0.84, 0.59, 0.42) from
  a constant. The local-efficiency simulation reported the same +0.003 at
  N = 32,000. This is a property of the current estimator, not of the proposed
  simplifications, and is left as an open item.

## 7. Conditions

* E||X~||^8 is used only in the notebook's outcome-fit section, to get
  max_i |eta_hat_i - eta_i| = o_p(n^{-1/2} n^{1/8}) << 1/K. Under E||X~||^4 the same
  bound holds for K << n^{1/4}, which covers the paper's range and the one above.
  The paper itself uses at most fourth moments (Block A, Gram concentration).
  E(R_W^4), E(epsilon^4) and E eta^4 have no use in the paper or notebook that I
  found; bounded conditional variance and E eta^2 suffice where errors appear.
* (Q1)-(Q2) imply (Q1m)-(Q2m): with Z = V, f_{R|V}(r|v) = rho(r, v)/f_V(v) is
  continuous in r, the envelope B(v)/f_V(v) has E[(1 + ||V||) B(V)/f_V(V)] =
  int (1 + ||v||) B(v) dv < infinity, and the two projection-density formulas
  coincide. So (DS6) can keep only (Q1m)-(Q2m).
* (DS3) asks rho_X in C^3 near its evaluation region; (DS4) asks it on the whole
  buffered region, which implies (DS3)'s clause. m_eta(t) = E[X~ | eta = t] = E X~
  under X independent of eta, so (DS4)'s continuity clause is automatic.
* (I1)-(I4) are defined in `prelim/prelim.tex` (lines 594-602) but not in
  `differing_slopes.tex`, whose Appendix 3 proof still cites (I2) and (I3).

## C. Pathwise derivative in the law of X

Shift the mean of X along v = (0.6, -0.3, 0.5) (paper design), holding a, beta2,
gamma and the law of eta fixed. The central difference of U_eps'(phi*) is
-0.0135012601494; E[psi_T v'Sigma^{-1}(X - EX)] is -0.0135012601490 (difference
4e-13). This checks psi_T, including the quantile terms and their signs. The
other directions are linear in f_eta (psi_U), the Riesz identity of item 3
(psi_o), and the index map of item 1 (slopes).

## Independent review

A reviewer agent re-derived claims 1 to 4 by hand, read Appendices 2 and 3 and
the notebook lemmas, and checked `idx_upper`, `idx_h2` and the arm moments by
Monte Carlo (4e6 draws, mixx design; all differences at Monte Carlo noise) and
dh2/dt by finite differences (3e-11). It also recomputed the outcome biases with
knot-aligned quadrature (same to four digits). It found no wrong formula or sign.
Its additions are in the text above: conditions (C1) and (C2), C^3 on the whole
density interval, the three rewrites, the ridge caveat, and the downgrade of
K >> n^{1/8} to suggestive. Its script notes (a dead assignment; slopes fitted
through the pre-asymptotic range; one identity that holds by construction) are
fixed or annotated in the script.

## Not checked

* Block D's uniform Bahadur argument is unchanged; it was not re-derived.
* The weighted L1 projection bound for the outcome block (K >> n^{1/8}).
* The remainder list above has not been written as a proof.
* No design violates (C1), so the checks cannot show what fails without it.
