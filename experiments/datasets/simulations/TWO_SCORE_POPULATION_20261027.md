# Two-score threshold rules: population check (2026-10-27, Claude)

This is an exploratory check, started from the main session after a chat
discussion of the two-score extension. Multiple scores stay out of scope for the
paper (`PROJECT_SCOPE.md`, "Out of scope"), and nothing in the paper or the
notebook changes.

- **Script:** `experiments/scripts/two_score_population.py`.
- **Command** (run from `code/`):
  `python -m experiments.scripts.two_score_population --mc-check`. It takes about
  12 min on 4-5 cores.
- **Outputs:** in the ignored `experiments/runs/two_score_population/`
  (`designs.json`, `sweep.json`, `index_scale_check.json`, `run.log`).

## Model and the objects computed

- **Scores.** Q_j = T_j + eta_j with T_j = gamma_j'X, j = 1, 2. X ~ N(0, I_3) is
  independent of eta, and eta is bivariate normal with unit variances and
  correlation rho.
- **Effect.** W = alpha(eta) + X'beta2. Because X is Gaussian,
  lambda(t) = beta2'E(X | T = t) = kappa't with kappa = Var(T)^{-1} G beta2.
  Every utility below depends on beta2 only through kappa, and beta2 = G'kappa is
  the minimal-norm beta2 for a given kappa.
- **Deployed rule.** D0 = 1{Q_1 > 0, Q_2 > 0}.
- **Supported population.** I0 = 1{eps <= pi0(eta) <= 1 - eps} with
  pi0(e) = P(T_1 > -e_1, T_2 > -e_2) and eps = 0.10. Since pi0 increases in both
  coordinates, for each eta_1 the supported eta_2 form an interval. The quadrature
  is iterated Gauss-Legendre over eta_1 and then over that interval, with eta_2
  standardized given eta_1. Under I0, each eta_j is at least -1.28: P(T_j > -eta_j)
  must reach eps.
- **Utility of a rule A.** U(A) = E[(W - c) 1{Q in A} I0]. Write p(q) for the
  supported density of Q and tau(q) = E(W - c | Q = q, I0 = 1). Then m = p tau and
  U(A) = int_A m(q) dq.
  - Conjunctive: closed form given eta (bivariate-normal orthant and Tallis
    moments).
  - Compensatory 1{w'Q > phi}, with w = (cos theta, sin theta): closed form given
    eta.
  - Single-score: theta = 0 or pi/2.
  - First best A* = {tau > 0}: U* = int m^+ on a 0.04 grid. The grid error is
    removed by adding the grid difference U*_grid - U_comp,grid to the closed-form
    U_comp.

**Face conditions and curvature of the conjunctive rule.** With
U(phi) = int_{q_1 > phi_1, q_2 > phi_2} m(q) dq:

- **First derivatives.** dU/dphi_1 = -f_1(phi) tau_1(phi).
  - f_1 is the density of Q_1 at phi_1 among supported units with Q_2 > phi_2.
  - tau_1 = E[W - c | Q_1 = phi_1, Q_2 > phi_2, I0 = 1].
  - Face 2 is the same with the roles swapped. This is Proposition 1 face by face.
    At an interior optimum, tau_1 = tau_2 = 0.
- **Cross partial.** d^2U/dphi_1 dphi_2 = m(phi) = p(phi) tau(phi): the density at
  the corner times its net effect. When tau increases along each face, faces that
  average zero force tau(corner) < 0, so H_12 < 0.
- **Diagonal.** Where tau_j = 0, H_jj = -f_j d tau_j / d phi_j. A critical point is
  therefore a maximum if and only if:
  - both face averages increase in their own cutoff;
  - f_1 f_2 (d tau_1/d phi_1)(d tau_2/d phi_2) > m(phi)^2.

  A strongly negative corner can turn a critical point into a saddle.

## Designs

All designs share these settings:
- gamma_1 = (0.8, 0.6, 0) and gamma_2 = (0, 0.6, 0.8), so Var T_j = 1 and
  Corr(T_1, T_2) = 0.36;
- unit-variance eta and rho = 0.5, except in the sweep;
- phi0 = (0, 0), eps = 0.10, c = 2.25, and the policy set Phi = [-4, 4]^2;
- P(I0) = 0.721.

`softmin(e_1, e_2) = -log(exp(-e_1) + exp(-e_2)) + log 2` increases in both
arguments and equals e on the diagonal.

| design | alpha(eta) | beta2 | kappa |
|---|---|---|---|
| paper_beta2 | 2 + 0.8 e1 + 0.5 e2 | (0.4, -0.25, 0.15), the paper's §5 slopes | (0.208, -0.105) |
| symmetric | 2 + 0.7 e1 + 0.7 e2 | (0.16, 0.24, 0.16) | (0.2, 0.2) |
| symmetric_paper_beta2 | 2 + 0.7 e1 + 0.7 e2 | (0.4, -0.25, 0.15) | (0.208, -0.105) |
| complementary | 2 + 1.4 softmin(e1, e2) | (0.16, 0.24, 0.16) | (0.2, 0.2) |

## The best conjunctive rule

| design | phi* | Hessian at phi* | eigenvalues | face means at phi* | tau(corner) | other critical points in Phi |
|---|---|---|---|---|---|---|
| paper_beta2 | (0.014, **-4**), bound | [[-0.0974, 0], [0, -0.0001]] | -0.097, -0.0001 | -5e-13; **+0.27** | +0.011 | none |
| symmetric | (-0.496, -0.496) | [[-0.0529, -0.0257], [-0.0257, -0.0529]] | -0.079, -0.027 | -1e-11; -1e-11 | -0.438 | none |
| symmetric_paper_beta2 | (-0.089, -2.328) | [[-0.0879, -0.0022], [-0.0022, -0.0004]] | -0.088, -0.0004 | -4e-11; 4e-11 | -0.288 | saddle at (-0.079, -3.864) |
| complementary | (-0.272, -0.272) | [[-0.0603, -0.0250], [-0.0250, -0.0603]] | -0.085, -0.035 | 7e-11; 1e-9 | -0.370 | none |

- **paper_beta2: the second hurdle never binds.**
  - At phi_2 = -4 the face-2 average is still +0.27, so U keeps rising as phi_2
    falls. The supremum is the single-score rule 1{Q_1 > 0.0145}.
  - The cause is the trimmed tail. Under I0, eta_2 >= -1.28, so a very low Q_2
    means a very low T_2. With kappa_2 < 0, low T_2 raises lambda(T), and the net
    effect on face 2 stays positive.
- **symmetric and complementary: interior and well curved.**
  - The eigenvectors are (1, 1)/sqrt 2, moving both cutoffs together (strong
    curvature), and (1, -1)/sqrt 2, trading one cutoff for the other (weak
    curvature).
  - The corner net effect is strongly negative, -0.44 and -0.37.
- **symmetric_paper_beta2: interior but flat, with a second critical point.**
  - The maximum has |lambda_min| = 0.0004 because f_2 = 0.009 and
    d tau_2 / d phi_2 = 0.05 are both small far in the Q_2 tail.
  - tau_2 crosses zero a second time at phi_2 = -3.864, which gives a saddle. U
    there equals the single-score value (0.225442) to 1e-6, and the maximum is
    only 3e-5 above it.
  - So with kappa_2 < 0, single crossing of the face average fails in the tail.
- **Identities, checked at every phi*:**
  - H_12 - m(phi*) is at most 6e-11;
  - H_jj + f_j d tau_j / d phi_j is at most 3e-10 wherever tau_j = 0;
  - the face means at interior optima are at most 1.2e-9.

## Comparison of rules (utilities U = E[(W - c) 1{rule} I0])

| design | deployed | best conjunctive | best single | best compensatory (w) | first best | 2nd score as hurdle | 2nd score in a linear rule | loss: conjunctive | loss: compensatory |
|---|---|---|---|---|---|---|---|---|---|
| paper_beta2 | 0.17745 | 0.21893 | 0.21893 (Q1) | 0.22239 (0.972, 0.234) | 0.22274 | 0.0000 | 0.0035 | 0.0038 | 0.0004 |
| symmetric | 0.24674 | 0.26680 | 0.24815 | 0.28798 (0.707, 0.707) | 0.28809 | 0.0187 | 0.0398 | 0.0213 | 0.0001 |
| symmetric_paper_beta2 | 0.19529 | 0.22547 | 0.22544 (Q1) | 0.23643 (0.901, 0.434) | 0.23678 | 0.0000 | 0.0110 | 0.0113 | 0.0004 |
| complementary | 0.20911 | 0.21539 | 0.18797 | 0.22775 (0.707, 0.707) | 0.22829 | 0.0274 | 0.0398 | 0.0129 | 0.0005 |

Column definitions:
- "2nd score as hurdle" = best conjunctive - best single;
- "in a linear rule" = best compensatory - best single;
- "loss" = first best - rule.

Share of the first-best gain over the deployed rule:

| design | conjunctive | compensatory |
|---|---|---|
| paper_beta2 | 92% | 99% |
| symmetric | 49% | 99.7% |
| symmetric_paper_beta2 | 73% | 99% |
| complementary | 33% | 97% |

## Checks

- **Quadrature.**
  - Raising the eta nodes from 10.8k to 110k changes U by at most 2.5e-7 and the
    gradient by at most 9e-8 at rho = 0.5. At rho = -0.5 and 0.95 the changes are at
    most 9.5e-7.
  - The sweep's 3k-node rule is within 4e-6.
  - The bivariate-normal orthant matches `scipy.stats.multivariate_normal.cdf` to
    2e-16.
  - P(I0) matches a 2e6-draw simulation within 2 s.e.
- **First-best grid.** On the 0.04 grid:
  - the closed-form compensatory and conjunctive utilities are reproduced within
    5e-6;
  - the mass P(I0) is reproduced within 2e-7.
- **Monte Carlo** (2e7 draws, seed 20261027; band +-0.02 for the face means):

| design | U(phi0) MC / quad | U(phi*) MC / quad | U(comp) MC / quad | face 1 mean (MC) | face 2 mean (MC) | tau(corner) MC (box +-0.08) / quad |
|---|---|---|---|---|---|---|
| paper_beta2 | 0.17746 / 0.17745 | 0.21889 / 0.21893 | 0.22226 / 0.22239 | 0.0005 (0.0017) | 0.263 (0.088), quad 0.271 | — (37 units) |
| symmetric | 0.24686 / 0.24674 | 0.26682 / 0.26680 | 0.28796 / 0.28798 | 0.0010 (0.0018) | -0.0000 (0.0018) | -0.440 / -0.438 |
| symmetric_paper_beta2 | 0.19534 / 0.19529 | 0.22547 / 0.22547 | 0.23637 / 0.23643 | -0.0033 (0.0019) | -0.010 (0.008) | -0.302 / -0.288 |
| complementary | 0.20917 / 0.20911 | 0.21534 / 0.21539 | 0.22776 / 0.22775 | -0.0036 (0.0017) | 0.0023 (0.0017) | -0.370 / -0.370 |

The MC standard errors of U are about 1.3e-4, and every utility agrees within 1
s.e. Of the eight face means, the largest deviation is 2.1 s.e.

## Sweep over rho and beta2 (symmetric design, kappa_1 = 0.2)

kappa_2 is the loading of lambda(t) = kappa't on T_2, and beta2 = G'(0.2, kappa_2):
- kappa_2 = -0.4: beta2 = (0.16, -0.12, -0.32);
- kappa_2 = 0: beta2 = (0.16, 0.12, 0);
- kappa_2 = 0.4: beta2 = (0.16, 0.36, 0.32).

Each cell gives phi* and |lambda_min|, the curvature in the weak direction.
"Slack" means that phi_2* sits at the box edge -4 and the best conjunctive rule is
the single-score rule on Q_1.

| rho \ kappa_2 | -0.4 | -0.2 | -0.1 | 0 | +0.1 | +0.2 | +0.4 |
|---|---|---|---|---|---|---|---|
| -0.5 | slack | slack | slack | (0.28, -1.98) 0.0009 | (-0.10, -0.72) 0.0095 | (-0.39, -0.39) 0.0146 | (-0.80, -0.14) 0.0181 |
| 0 | slack | slack | slack | (-0.15, -1.37) 0.0057 | (-0.35, -0.81) 0.0155 | (-0.54, -0.54) 0.0210 | (-0.85, -0.26) 0.0225 |
| 0.5 | slack | slack | (-0.10, -2.13) 0.0008 | (-0.21, -1.10) 0.0112 | (-0.35, -0.72) 0.0215 | (-0.50, -0.50) 0.0272 | (-0.78, -0.23) 0.0261 |
| 0.8 | slack | slack | (-0.05, -1.66) 0.0026 | (-0.14, -0.97) 0.0133 | (-0.27, -0.63) 0.0246 | (-0.41, -0.41) 0.0310 | (-0.72, -0.15) 0.0269 |
| 0.95 | slack | slack | (-0.00, -1.65) 0.0021 | (-0.08, -0.93) 0.0126 | (-0.20, -0.58) 0.0252 | (-0.35, -0.35) 0.0325 | (-0.69, -0.09) 0.0256 |

The strong eigenvalue is 0.044-0.12 in every cell. Every interior cell has exactly
one critical point in Phi (a maximum); slack cells have none.

Value of the second score as a hurdle / loss of the conjunctive rule / loss of the
compensatory rule (both losses against the first best), all x 1000:

| rho \ kappa_2 | -0.4 | -0.2 | -0.1 | 0 | +0.1 | +0.2 | +0.4 |
|---|---|---|---|---|---|---|---|
| -0.5 | 0.0 / 6.6 / 1.8 | 0.0 / 4.9 / 2.5 | 0.0 / 11.8 / 2.0 | 0.2 / 21.4 / 1.5 | 6.3 / 26.9 / 1.1 | 16.5 / 29.7 / 0.8 | 11.6 / 32.8 / 0.4 |
| 0 | 0.0 / 3.0 / 2.9 | 0.0 / 8.6 / 1.8 | 0.0 / 15.8 / 1.3 | 1.6 / 23.4 / 0.9 | 8.3 / 27.2 / 0.7 | 17.7 / 29.3 / 0.5 | 11.8 / 31.4 / 0.3 |
| 0.5 | 0.0 / 1.8 / 1.7 | 0.0 / 5.6 / 0.6 | 0.1 / 11.7 / 0.3 | 3.4 / 16.5 / 0.2 | 9.9 / 19.5 / 0.2 | 18.7 / 21.3 / 0.1 | 11.4 / 22.9 / 0.1 |
| 0.8 | 0.0 / 2.1 / 1.0 | 0.0 / 2.6 / 0.2 | 0.4 / 7.2 / 0.1 | 3.6 / 11.0 / 0.0 | 9.7 / 13.6 / 0.0 | 18.0 / 15.2 / 0.0 | 9.6 / 16.2 / 0.0 |
| 0.95 | 0.0 / 3.5 / 0.7 | 0.0 / 1.2 / 0.1 | 0.3 / 4.8 / 0.0 | 3.0 / 8.4 / 0.0 | 8.5 / 10.8 / 0.0 | 16.5 / 12.3 / 0.0 | 7.6 / 12.8 / 0.0 |

The best conjunctive rule gains 0.008-0.059 over the deployed cutoffs. The gain is
largest in slack cells, where the deployed hurdle Q_2 > 0 is the main mistake.

**When the optimum is interior and well curved.**
- **kappa_2 <= -0.2: never interior.** The second hurdle is slack for every rho.
- **kappa_2 = -0.1: interior only at rho >= 0.5, and flat.**
  |lambda_min| <= 0.0026, phi_2* < -1.6, and the hurdle is worth at most 4e-4.
- **kappa_2 = 0: interior, well curved only at rho >= 0.5.** |lambda_min| is
  0.011-0.013 there, falls to 0.006 at rho = 0, and is 0.0009 at rho = -0.5.
- **kappa_2 >= 0.1: interior and well curved for every rho.**
  - |lambda_min| is 0.0095-0.033.
  - Curvature mostly grows with rho: at kappa_2 = 0.2 it rises from 0.015 to
    0.033.
  - At rho >= 0.5 it is largest at kappa_2 = 0.2, where the two scores carry equal
    weight (kappa_2 = kappa_1).
- **Practical size.** The n-scaled variance along the weak eigenvector scales
  as 1/lambda_min^2. Going from 0.027 (symmetric) to 0.001 multiplies it by about
  700.

## Index-noise check (`index_scale_check.json`)

Shrinking the index noise (gamma -> s gamma, beta2 -> s beta2, so kappa is fixed)
makes Q track eta more closely. The share of the first-best gain stays about the
same:

| design | s | P(I0) | phi* | conjunctive | compensatory |
|---|---|---|---|---|---|
| symmetric | 1.0 / 0.6 / 0.4 | 0.72 / 0.52 / 0.37 | (-0.50, -0.50) / (-0.43, -0.43) / (-0.41, -0.41) | 49% / 48% / 48% | 99.7% / 99% / 98% |
| complementary | 1.0 / 0.6 / 0.4 | 0.72 / 0.52 / 0.37 | (-0.27, -0.27) / (-0.22, -0.22) / (-0.19, -0.19) | 33% / 35% / 32% | 97% / 97% / 99% |

## Reading

1. **The two-score target has the structure of Proposition 1.**
   - Each face condition says that the marginal units on that face have zero
     average net effect.
   - The cross partial is the corner's density times its net effect.
   - These were checked by quadrature identities and by simulation.
2. **An interior optimum is not automatic.**
   - Each face average must cross zero exactly once and increase through it.
   - The trim makes the deep tail of Q_j consist of units with extreme T_j. There
     the covariate part kappa_j T_j decides the sign of the face average.
   - With the paper's beta2 (kappa_2 < 0), the hurdle on the second score is slack
     or nearly flat, and a second critical point (a saddle) can appear.
   - A two-score analogue of Proposition 2 would need a single-crossing condition
     on each face average and the determinant condition above.
3. **The weak direction is the trade-off between the two cutoffs.** Even in the
   well-curved cells (kappa_2 >= 0.1), |lambda_min| is only 0.16-0.37 of
   |lambda_max|.
4. **Conjunctive rules lose much more than compensatory rules.**
   - The best compensatory rule is within 0.003 of the first best in every cell,
     and within 1e-4 at rho >= 0.8 with kappa_2 >= -0.1.
   - The best conjunctive rule loses 0.001-0.033 and captures 24-97% of the
     first-best gain over the deployed rule. It does best where the second
     hurdle is slack, because there the first best is close to a single-score
     rule.
   - With linear alpha and Gaussian (X, eta), the untrimmed tau is affine in q, so
     the untrimmed first best is a half-plane. The trim bends its boundary only
     slightly.
   - Complementary abilities (softmin alpha) and less noisy indices do not change
     this. The noise from T and the trim smooth E(W | Q) toward an affine
     function, so the first-best region stays close to a half-plane.
5. **Caveat.** The design is all-Gaussian with linear lambda, and these results
   concern population quantities only; no estimator was run. Before any claim
   about two-score rules in general, the comparison should be repeated with
   non-Gaussian eta or X, or with an alpha that has a hard threshold in eta.
