"""What does within-arm heteroskedasticity cost?  Population calculation.

Companion to ``ds_local_efficiency.py``.  Condition (H) of ``this_week.tex``
(section "Local efficiency at Gaussian residuals") asks the outcome noise to be
homoskedastic within each arm.  Here the arm-d noise variance is
sigma_d^2(eta) = sigma_d^2 exp(kappa eta), a function of eta only, and three
rotated N-scaled threshold variances are compared:

* ``paper``: unweighted outcome fit and OLS first stage (the paper's estimator);
* ``stacked``: unweighted outcome fit and the stacked first stage with arm-wise
  constant weights 1/sbar_d^2 (efficient only under H).  Under heteroskedasticity
  its first-stage and outcome scores are correlated, and the rotated variance
  includes that covariance;
* ``weighted_ols``: outcome fit weighted by 1/sigma_D^2(eta), OLS first stage;
* ``bound``: the efficiency bound of M_J, attained by the outcome fit and the
  stacked first-stage outcome rows both weighted by 1/sigma_D^2(eta).  Its outcome term is
  -sigma^{-2} R_Y r^w, where r^w represents L_out in the weighted inner product
  E[S_J m1 m2 / sigma^2]; on arm d,
      r^w = s_d I0 g(phi* - eta) sigma_d^2(eta) / pi_d(eta) + (X - m_d)'k_d,
      k_d = s_d (M_d^w)^{-1} (B - E[I0 g m_d]),  M_d^w = E[S_J pi_d V_d / sigma_d^2].
  Its first-stage term uses the weighted residual R_xi^w, the weighted information
  I_w = Var(X)/var_eta + E[S_J R^w R^w' / sigma^2], and the loading
  Abar_X + E[S_J r_out R_xi^w].

Gaussian eta, so the score rows are least squares.  Attaining the bound would
need weights 1/sigma_hat^2(D, X, eta) in the outcome fold as well; that estimator
is not implemented.

Run:
    python -m experiments.scripts.ds_local_efficiency_hetero
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.scripts.ds_local_efficiency import DESIGNS, Population

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency"


def hetero_variances(pop: Population, kappa: float) -> dict:
    d = pop.d
    base = pop.decomposition()                      # psi_U, psi_T, Abar do not change
    phi, l0, u0 = pop.phi, pop.l0, pop.u0
    eJ, wJ = pop._eta_nodes(d.J[0], d.J[1], extra=(l0, u0))
    inI0 = (eJ >= l0) & (eJ <= u0)
    gphi = np.where(inI0, pop.g(phi - eJ), 0.0)
    B = np.sum((wJ * inI0)[:, None] * pop.rho(phi - eJ), axis=0)
    AX = np.asarray(base["Abar"])[1:]
    V_X = pop.Sigma
    dim = pop.dim

    var_o_paper = 0.0
    var_o_bound = 0.0
    cross_bound = np.zeros(dim)                     # E[S r_out R_xi^w]
    Jw = np.zeros((dim, dim))
    G_ls = V_X / pop.var_eta
    var_psiw = V_X / pop.var_eta
    cov_ls = np.zeros(dim)                          # E[S r_out R_xi sigma^2 / sbar^2]
    for arm, s in ((0, -1.0), (1, 1.0)):
        sig2 = d.sigma_arm[arm] ** 2 * np.exp(kappa * eJ)
        pi, m, V = pop.pi(arm, eJ), pop.m_arm(arm, eJ), pop.V_arm(arm, eJ)
        dc = pop.dc_fun[arm](eJ)
        rhs = B - np.sum((wJ * gphi)[:, None] * m, axis=0)
        # unweighted (paper) representer
        M = np.einsum("i,ijk->jk", wJ * pi, V)
        N = np.einsum("i,ijk->jk", wJ * pi * dc, V)
        P = np.einsum("i,ijk->jk", wJ * pi * dc ** 2, V)
        omega = s * np.linalg.solve(M, rhs)
        var_o_paper += (float(np.sum(wJ * gphi ** 2 * sig2 / pi))
                        + float(omega @ np.einsum("i,ijk->jk", wJ * pi * sig2, V) @ omega))
        # weighted representer and residual (bound)
        Mw = np.einsum("i,ijk->jk", wJ * pi / sig2, V)
        Nw = np.einsum("i,ijk->jk", wJ * pi * dc / sig2, V)
        Pw = np.einsum("i,ijk->jk", wJ * pi * dc ** 2 / sig2, V)
        k = s * np.linalg.solve(Mw, rhs)
        var_o_bound += float(np.sum(wJ * gphi ** 2 * sig2 / pi)) + float(k @ Mw @ k)
        Jw += Pw - Nw @ np.linalg.solve(Mw, Nw)
        Om_w = np.linalg.solve(Mw, Nw)              # weighted projection coefficients
        cross_bound += (N - Om_w.T @ M) @ omega
        # least-squares stacked first stage with arm-constant weight sbar^2
        sbar2 = float(np.sum(wJ * pi * sig2) / np.sum(wJ * pi))
        Om = np.linalg.solve(M, N)
        RR_sig = (np.einsum("i,ijk->jk", wJ * pi * sig2 * dc ** 2, V)
                  - Om.T @ np.einsum("i,ijk->jk", wJ * pi * sig2 * dc, V)
                  - np.einsum("i,ijk->jk", wJ * pi * sig2 * dc, V) @ Om
                  + Om.T @ np.einsum("i,ijk->jk", wJ * pi * sig2, V) @ Om)
        G_ls = G_ls + (P - N @ Om) / sbar2
        var_psiw = var_psiw + RR_sig / sbar2 ** 2
        RV = (np.einsum("i,ijk->jk", wJ * pi * sig2 * dc, V)
              - Om.T @ np.einsum("i,ijk->jk", wJ * pi * sig2, V))
        cov_ls += RV @ omega / sbar2

    common = base["variances"]["psi_U"] + base["variances"]["psi_T"]
    H2 = pop.H ** 2
    var_g_ols = base["variances"]["psi_gamma_ols"]
    paper = (common + var_o_paper + var_g_ols) / H2
    Ginv_A = np.linalg.solve(G_ls, AX)
    var_g_ls = float(Ginv_A @ var_psiw @ Ginv_A)
    # psi_o = -r_out R_Y and the outcome part of psi_gamma^w = -Abar' G^{-1} S R R_Y / sbar^2
    cov_o_g = float(Ginv_A @ cov_ls)
    stacked = (common + var_o_paper + var_g_ls + 2.0 * cov_o_g) / H2
    I_w = V_X / pop.var_eta + Jw
    c = AX + cross_bound          # = Abar_X of the weighted plug-in
    bound = (common + var_o_bound + float(c @ np.linalg.solve(I_w, c))) / H2
    # weighted outcome fit with the OLS first stage
    weighted_ols = (common + var_o_bound + pop.var_eta * float(c @ np.linalg.solve(V_X, c))) / H2
    slope_cov = np.linalg.solve(G_ls, np.linalg.solve(G_ls, var_psiw).T)
    return {"kappa": kappa, "paper": paper, "stacked": stacked, "bound": bound,
            "weighted_ols": weighted_ols, "weighted_stacked": bound,
            "fixed_paper": 4 * paper,
            "fixed_stacked": 4 * (common + var_o_paper + var_g_ls) / H2,
            "two_cov_outcome_first_stage": 2.0 * cov_o_g,
            "stacked_slope_var": np.diag(slope_cov).tolist(),
            "var_o_paper": var_o_paper, "var_o_bound": var_o_bound,
            "gain_stacked": 1 - stacked / paper, "gain_bound": 1 - bound / paper}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="paper_baseline")
    parser.add_argument("--kappa", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    pop = Population(DESIGNS[args.design], allow_hetero=True)
    rows = [hetero_variances(pop, k) for k in args.kappa]
    for r in rows:
        print(f"kappa={r['kappa']:.2f}: paper={r['paper']:.3f} stacked={r['stacked']:.3f} "
              f"weighted+OLS={r['weighted_ols']:.3f} weighted+stacked=bound={r['bound']:.3f} "
              f"(gain stacked {100 * r['gain_stacked']:.1f}%, bound {100 * r['gain_bound']:.1f}%)")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"hetero_{args.design}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
