"""Simulation check of the first-order variance with an evaluation-fold capped trim window.

Variant of the four-role estimator of ``differing_slopes.tex`` (Theorem 1, Appendix 2
Steps 4-5, Block C) in which the index-quantile window [l_hat, u_hat] is additionally
capped at residual quantiles computed on the EVALUATION fold:

    window_hat = [max(l_hat, xi_hat_lo), min(u_hat, xi_hat_hi)],

with xi_hat_p the empirical p-quantile of eta_hat over the evaluation fold.  The
population window is [l*, u*] = [max(l0, xi_lo), min(u0, xi_hi)], xi_p = Q_p(eta).
When both caps bind strictly the claimed first-order result is

* psi_U = -F*(eta) + b_l zeta_lo(eta) - b_u zeta_hi(eta),
  zeta_p = {p - 1(eta <= xi_p)} / f_eta(xi_p), b_v = F(v) f_eta(v);
* psi_T = -r_g(T) - r_1(T) beta2'X (no endpoint terms; capped window in r_T);
* psi_o as in the uncapped case with I0* = 1(l* <= eta <= u*);
* psi_gamma = Abar' E[X~X~']^{-1} X~ eta with
  Abar = A_U - A_out - A_T - b_l e_1 + b_u e_1, A_U = e_1 {E[s* I0*] + b_l - b_u};
* sigma^2 = sum of the four variances; n Var = sigma^2/H^2 (rotated average, pooled),
  4 sigma^2/H^2 (one fixed assignment).

Design: ``paper_baseline`` (X ~ N(0, I_3), T ~ N(0, 1), eta ~ N(0, 1), c = 2.25,
J = [-1.75, 1.75], Phi = [-1.5, 1.5]) with eps = 0.10 and caps p = (0.15, 0.85), so
both caps bind and [l*, u*] equals the eps = 0.15 window.

Arms recorded per data set (same data, same split, same nuisance fits):

* ``capped_{avg,pooled,fixed}``: the capped estimator;
* ``eps15_{avg,pooled,fixed}``: the plain index-quantile estimator at eps = 0.15;
* ``oracle_{avg,pooled}``: true gamma, alpha, beta2, g, rho_X and the true window [l*, u*];
* ``oracle_capq_{avg,pooled}`` (diagnostic): true nuisances, but the caps are the
  empirical quantiles of the true eta on the evaluation fold (isolates the zeta terms).

Run (from code/):
    python -m experiments.scripts.ds_capped_window_check --population
    python -m experiments.scripts.ds_capped_window_check --n 16000 64000 --reps 1000 500 --workers 3
    python -m experiments.scripts.ds_capped_window_check --report
Raw rows go to experiments/runs/ds_capped_window_check/ (git-ignored); reruns resume.
Larger run used in the notebook (extends the 500 data sets above with the same seeds;
copy n64000.jsonl into the new directory first):
    python -m experiments.scripts.ds_capped_window_check --n 64000 --reps 2000 --workers 3 \
        --out experiments/runs/ds_capped_window_check/supp_2000

Notation: the notebook (this_week.tex, section sec:week-taxi-theory, revised
2026-11-04) writes q^eta_p for xi_p, z^eta_p for zeta_p and [l_c, u_c], I_c for
[l*, u*], I0*.
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.polynomial import Polynomial
from scipy import integrate, optimize
from scipy.stats import norm

from experiments.methods.trimmed_quadrature import trimmed_normal_mean
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_four_role_coverage import (
    SplineTailPieces, TrueTails, _avg_and_pooled, _criterion,
)
from experiments.scripts.ds_local_efficiency import DESIGNS, Design, Population, _gl_nodes
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, _xt, generate, ols_first_stage, outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_capped_window_check"
DESIGN = replace(DESIGNS["paper_baseline"], eps=0.10)
EPS_PLAIN = 0.15
P_LO, P_HI = 0.15, 0.85


# ------------------------------------------------------------ population

class CappedPopulation(Population):
    """Population objects with the window [l*, u*] = [max(l0, xi_lo), min(u0, xi_hi)].

    Every inherited method that reads ``self.l0``/``self.u0`` (target, F, r_g, r_1)
    then uses the capped window.  Only both-sides-binding caps are handled.
    """

    def __init__(self, design: Design, p_lo: float, p_hi: float):
        if tuple(design.eta_law) != ("normal",):
            raise ValueError("eta quantiles implemented for normal eta only")
        super().__init__(design)
        self.p_lo, self.p_hi = float(p_lo), float(p_hi)
        self.l_eps, self.u_eps = self.l0, self.u0
        self.xi_lo = self.sig_eta * float(norm.ppf(p_lo))
        self.xi_hi = self.sig_eta * float(norm.ppf(p_hi))
        if not (self.l_eps < self.xi_lo < self.xi_hi < self.u_eps):
            raise ValueError("this check assumes both caps bind strictly")
        self.l0, self.u0 = self.xi_lo, self.xi_hi
        self.phi, self.H = self._target()


def capped_decomposition(pop: CappedPopulation) -> Dict[str, object]:
    """Four role scores of the capped estimator (formulas in the module docstring)."""
    d, dim = pop.d, pop.dim
    phi, l0, u0 = pop.phi, pop.l0, pop.u0
    J0, J1 = d.J
    f_lo, f_hi = float(pop.f_eta(l0)), float(pop.f_eta(u0))
    b_l = float(pop.F(l0) * f_lo)
    b_u = float(pop.F(u0) * f_hi)

    # psi_U = -F*(eta) + b_l zeta_lo - b_u zeta_hi
    e_w, w_w = pop._eta_nodes(l0, u0)
    F_w = pop.F(e_w)
    mean_F = float(np.sum(w_w * F_w))                       # = Z(phi*) = 0
    EF2 = float(np.sum(w_w * F_w ** 2))
    plo, phi_ = pop.p_lo, pop.p_hi
    # zeta parts: Var(b_l z_lo - b_u z_hi), and Cov(-F*, .) via E[F* 1(eta <= xi)]
    var_z = (b_l ** 2 * plo * (1 - plo) / f_lo ** 2 + b_u ** 2 * phi_ * (1 - phi_) / f_hi ** 2
             - 2 * b_l * b_u * plo * (1 - phi_) / (f_lo * f_hi))
    EF_below_lo = 0.0                                        # F* = 0 below l* = xi_lo
    EF_below_hi = mean_F                                     # window lies below xi_hi
    # Cov(-F*, b_l z_lo) = b_l/f_lo (E[F* 1(eta<=xi_lo)] - plo E F*), similarly hi
    cov_Fz = (b_l / f_lo * (EF_below_lo - plo * mean_F)
              - b_u / f_hi * (EF_below_hi - phi_ * mean_F))
    var_U = EF2 - mean_F ** 2 + var_z + 2 * cov_Fz

    # psi_T = -r_g(T) - r_1(T) beta2'X, no endpoint terms
    tlo, thi = d.gamma0 - 9 * pop.sT, d.gamma0 + 9 * pop.sT
    t, wt = _gl_nodes([tlo, phi - u0, phi - l0, thi])
    wt = wt * pop.g(t)
    slope = float(pop.beta2 @ pop.kappa) / pop.sT ** 2
    v_perp = float(pop.beta2 @ pop.Sigma @ pop.beta2
                   - (pop.beta2 @ pop.kappa) ** 2 / pop.sT ** 2)
    m_T = -pop.r_g(t) - pop.r_1(t) * slope * (t - d.gamma0)
    mean_T = float(np.sum(wt * m_T))
    var_T = float(np.sum(wt * m_T ** 2) + v_perp * np.sum(wt * pop.r_1(t) ** 2) - mean_T ** 2)

    # outcome representer with I0* (as in Population.decomposition)
    eJ, wJ = pop._eta_nodes(J0, J1, extra=(l0, u0))
    inI0 = (eJ >= l0) & (eJ <= u0)
    gphi = np.where(inI0, pop.g(phi - eJ), 0.0)
    B = np.sum((wJ * inI0)[:, None] * pop.rho(phi - eJ), axis=0)
    omega, M, N = {}, {}, {}
    var_o, A_out_X = 0.0, np.zeros(dim)
    A_out_0 = float(np.sum(wJ * gphi * pop.da(eJ)))
    for arm, s in ((0, -1.0), (1, 1.0)):
        pi = pop.pi(arm, eJ)
        m = pop.m_arm(arm, eJ)
        V = pop.V_arm(arm, eJ)
        dc = pop.dc_fun[arm](eJ)
        M[arm] = np.einsum("i,ijk->jk", wJ * pi, V)
        N[arm] = np.einsum("i,ijk->jk", wJ * pi * dc, V)
        rhs = B - np.sum((wJ * gphi)[:, None] * m, axis=0)
        omega[arm] = s * np.linalg.solve(M[arm], rhs)
        sig2 = d.sigma_arm[arm] ** 2
        var_o += sig2 * (float(np.sum(wJ * gphi ** 2 / pi))
                         + float(omega[arm] @ M[arm] @ omega[arm]))
        A_out_X += (s * np.sum((wJ * dc * gphi)[:, None] * m, axis=0)
                    + N[arm] @ omega[arm])

    # density loadings (capped window inside r_g, r_1)
    tt, wtt = _gl_nodes([phi - u0, phi - l0])
    rg, r1 = pop.r_g(tt), pop.r_1(tt)
    A_g = -np.concatenate([[np.sum(wtt * pop.dg(tt) * rg)],
                           np.sum((wtt * rg)[:, None] * pop.drho(tt), axis=0)])
    kk = np.outer(pop.kappa, pop.kappa)
    dev = tt - d.gamma0
    base = pop.Sigma - kk / pop.sT ** 2
    dEXXg = (base[None] * pop.dg(tt)[:, None, None]
             + kk[None] / pop.sT ** 4
             * (2 * dev * pop.g(tt) + dev ** 2 * pop.dg(tt))[:, None, None])
    A_rho = -np.concatenate([[np.sum(wtt * r1 * (pop.drho(tt) @ pop.beta2))],
                             np.einsum("i,ijk,k->j", wtt * r1, dEXXg, pop.beta2)])

    # loadings: cap endpoints load on e_1 = E X~ (E X = 0 here)
    e1 = np.zeros(dim + 1)
    e1[0] = 1.0
    Es_star = float(np.sum(w_w * pop.s_star(e_w)))
    A_U = e1 * (Es_star + b_l - b_u)
    A_out = np.concatenate([[A_out_0], A_out_X])
    Abar = A_U - A_out - A_g - A_rho - b_l * e1 + b_u * e1
    Sig_t = np.zeros((dim + 1, dim + 1))                     # E[X~ X~'] with E X = 0
    Sig_t[0, 0] = 1.0
    Sig_t[1:, 1:] = pop.Sigma
    var_gamma = float(pop.var_eta * Abar @ np.linalg.solve(Sig_t, Abar))

    H2 = pop.H ** 2
    s2 = var_U + var_o + var_T + var_gamma
    return {
        "phi_star": phi, "H": pop.H, "window": [l0, u0],
        "eps_window": [pop.l_eps, pop.u_eps], "xi": [pop.xi_lo, pop.xi_hi],
        "b_l": b_l, "b_u": b_u, "f_eta_lo": f_lo, "f_eta_hi": f_hi,
        "means": {"F_star": mean_F, "psi_T": mean_T},
        "variances": {"psi_U": var_U, "psi_U_Fstar_part": EF2 - mean_F ** 2,
                      "psi_U_zeta_part": var_z, "psi_U_cross": 2 * cov_Fz,
                      "psi_o": var_o, "psi_T": var_T, "psi_gamma": var_gamma},
        "Abar": Abar.tolist(), "Abar_intercept": float(Abar[0]),
        "A_U": A_U.tolist(), "A_out": A_out.tolist(), "A_g": A_g.tolist(),
        "A_rho": A_rho.tolist(), "E_s_star_I": Es_star,
        "omega": {str(k): v.tolist() for k, v in omega.items()},
        "sigma2": s2, "V_rotated": s2 / H2, "V_fixed": 4 * s2 / H2,
        "V_oracle": (EF2 - mean_F ** 2) / H2,               # true window, true nuisances
        "V_oracle_capq": var_U / H2,                         # true nuisances, estimated caps
    }


def score_mc(pop: CappedPopulation, dec: dict, plain: dict, n: int = 2_000_000,
             seed: int = 7, chunk: int = 500_000) -> dict:
    """Monte Carlo of the four role scores at the same draw W = (X, eta, D, noise).

    Also the influence function of the plain eps-window estimator (``plain`` is its
    ``Population.decomposition``; same window, so psi_o and the density part of psi_T
    coincide), for the paired difference capped - plain.
    """
    d = pop.d
    rng = np.random.default_rng(seed)
    Abar = np.asarray(dec["Abar"])
    Sig_t = np.diag(np.concatenate([[1.0], np.asarray(d.sigma_x, dtype=float)]))
    load = np.linalg.solve(Sig_t, Abar)
    load_pl = np.linalg.solve(Sig_t, np.asarray(plain["Abar"]))
    eps_pl = float(plain["design"]["eps"])
    q_lo, q_hi = pop.q(eps_pl), pop.q(1.0 - eps_pl)
    if abs(plain["l0"] - pop.l0) > 1e-12 or abs(plain["u0"] - pop.u0) > 1e-12:
        raise ValueError("paired difference assumes the plain window equals [l*, u*]")
    Sd = np.zeros(2)                                         # sum, sum of squares of diff
    omega = {int(k): np.asarray(v) for k, v in dec["omega"].items()}
    l0, u0, phi = pop.l0, pop.u0, pop.phi
    b_l, b_u = dec["b_l"], dec["b_u"]
    f_lo, f_hi = dec["f_eta_lo"], dec["f_eta_hi"]
    S1 = np.zeros(4)
    S2 = np.zeros((4, 4))
    done = 0
    while done < n:
        m = min(chunk, n - done)
        X = rng.standard_normal((m, pop.dim)) * np.sqrt(np.asarray(d.sigma_x))
        eta = d.sigma_eta * rng.standard_normal(m)
        T = d.gamma0 + X @ pop.gx
        D = (T + eta > d.phi0).astype(int)
        u = np.where(D == 1, d.sigma_arm[1], d.sigma_arm[0]) * rng.standard_normal(m)
        inW = (eta >= l0) & (eta <= u0)
        z_lo = (pop.p_lo - (eta <= pop.xi_lo)) / f_lo
        z_hi = (pop.p_hi - (eta <= pop.xi_hi)) / f_hi
        psi_U = -np.where(inW, pop.F(eta), 0.0) + b_l * z_lo - b_u * z_hi
        inJ = (eta >= d.J[0]) & (eta <= d.J[1])
        r_out = np.zeros(m)
        for arm, s in ((0, -1.0), (1, 1.0)):
            on = (D == arm) & inJ
            e = eta[on]
            gw = np.where(inW[on], pop.g(phi - e), 0.0)
            r_out[on] = (s * gw / pop.pi(arm, e)
                         + np.sum((X[on] - pop.m_arm(arm, e)) * omega[arm], axis=1))
        psi_o = r_out * u
        psi_T = -pop.r_g(T) - pop.r_1(T) * (X @ pop.beta2)
        psi_g = (_xt(X) @ load) * eta
        P = np.column_stack((psi_U, psi_o, psi_T, psi_g))
        S1 += P.sum(axis=0)
        S2 += P.T @ P
        # plain eps-window estimator at the same draw
        zt_hi = ((1.0 - eps_pl) - (T <= q_hi)) / pop.g(q_hi)
        zt_lo = (eps_pl - (T <= q_lo)) / pop.g(q_lo)
        psi_pl = (-np.where(inW, pop.F(eta), 0.0) + psi_o + psi_T
                  - b_l * zt_hi + b_u * zt_lo + (_xt(X) @ load_pl) * eta)
        diff = P.sum(axis=1) - psi_pl
        Sd += [diff.sum(), (diff ** 2).sum()]
        done += m
    mean = S1 / n
    cov = S2 / n - np.outer(mean, mean)
    sd = np.sqrt(np.diag(cov))
    names = ["psi_U", "psi_o", "psi_T", "psi_gamma"]
    var_diff = float(Sd[1] / n - (Sd[0] / n) ** 2)
    return {"n": n, "names": names, "mean": mean.tolist(),
            "var": dict(zip(names, np.diag(cov).tolist())),
            "corr": (cov / np.outer(sd, sd)).tolist(),
            "var_of_sum": float(np.sum(cov)), "sum_of_var": float(np.trace(cov)),
            "V_rotated_mc": float(np.sum(cov)) / pop.H ** 2,
            "var_diff_capped_minus_plain": var_diff,
            "D_rotated": var_diff / pop.H ** 2, "D_fixed": 4 * var_diff / pop.H ** 2}


def diff_variance(pop: CappedPopulation, dec: dict, plain: dict) -> float:
    """Closed form of Var(IF_capped - IF_plain) when the two windows coincide.

    The difference is (b_l zeta_lo - b_u zeta_hi)(eta) + (b_l z_hi - b_u z_lo)(T)
    + (Abar_cap - Abar_plain)' E[X~X~']^{-1} X~ eta; the three parts are uncorrelated.
    """
    eps = float(plain["design"]["eps"])
    b_l, b_u = dec["b_l"], dec["b_u"]
    g_lo, g_hi = pop.g(pop.q(eps)), pop.g(pop.q(1.0 - eps))
    var_zT = (b_l ** 2 * eps * (1 - eps) / g_hi ** 2 + b_u ** 2 * eps * (1 - eps) / g_lo ** 2
              - 2 * b_l * b_u * eps ** 2 / (g_lo * g_hi))
    dA = np.asarray(dec["Abar"]) - np.asarray(plain["Abar"])
    Sig_t = np.diag(np.concatenate([[1.0], np.asarray(pop.d.sigma_x, dtype=float)]))
    return float(dec["variances"]["psi_U_zeta_part"] + var_zT
                 + pop.var_eta * dA @ np.linalg.solve(Sig_t, dA))


def population_summary() -> dict:
    pop = CappedPopulation(DESIGN, P_LO, P_HI)
    dec = capped_decomposition(pop)
    # target and curvature by adaptive quadrature on [l*, u*]
    a = Polynomial(DESIGN.a_coef)
    b2 = np.asarray(DESIGN.beta2, dtype=float)

    def Z(phi):
        return trimmed_normal_mean(
            lambda e: (a(e) - DESIGN.cost) * pop.g(phi - e) + pop.rho(phi - e) @ b2,
            pop.l0, pop.u0, sd=DESIGN.sigma_eta)

    phi_q = float(optimize.brentq(Z, pop.phi - 0.05, pop.phi + 0.05, xtol=1e-13))
    h = 1e-4
    H_q = -(Z(phi_q + h) - Z(phi_q - h)) / (2 * h)
    if abs(phi_q - dec["phi_star"]) > 1e-8:
        raise RuntimeError(f"targets disagree: {phi_q} vs {dec['phi_star']}")
    # uncapped eps = 0.15 estimator (existing code)
    plain = Population(replace(DESIGN, eps=EPS_PLAIN)).decomposition()
    # Var psi_U by adaptive quadrature as an independent check of the closed form
    F2 = integrate.quad(lambda e: float(pop.F(np.array([e]))[0]) ** 2 * norm.pdf(e),
                        pop.l0, pop.u0, epsabs=1e-13, epsrel=1e-12)[0]
    mc = score_mc(pop, dec, plain)
    var_diff = diff_variance(pop, dec, plain)
    return {
        "design": repr(DESIGN), "p_lo": P_LO, "p_hi": P_HI, "eps_plain": EPS_PLAIN,
        "capped": dec, "phi_star_quad": phi_q, "H_quad_fd": H_q,
        "E_Fstar2_quad": F2,
        "plain_eps15": {k: plain[k] for k in ("phi_star", "H", "l0", "u0", "b_l", "b_u",
                                              "variances", "Abar", "Abar_intercept",
                                              "V_rotated_ols", "V_fixed_ols")},
        "plain_eps15_V_oracle": plain["variances"]["psi_U"] / plain["H"] ** 2,
        "score_mc": mc,
        "var_diff_capped_minus_plain": var_diff,
        "D_rotated": var_diff / pop.H ** 2, "D_fixed": 4 * var_diff / pop.H ** 2,
    }


def print_population(pop: dict) -> None:
    c, p, mc = pop["capped"], pop["plain_eps15"], pop["score_mc"]
    v = c["variances"]
    print(f"capped window [l*, u*] = [{c['window'][0]:.4f}, {c['window'][1]:.4f}] "
          f"(eps window [{c['eps_window'][0]:.4f}, {c['eps_window'][1]:.4f}])")
    print(f"phi* = {c['phi_star']:.6f} (quadrature {pop['phi_star_quad']:.6f}; eps=0.15 "
          f"{p['phi_star']:.6f}), H = {c['H']:.6f} (finite diff {pop['H_quad_fd']:.6f}; "
          f"eps=0.15 {p['H']:.6f})")
    print(f"b_l = {c['b_l']:.6f}, b_u = {c['b_u']:.6f}")
    print(f"Var psi_U = {v['psi_U']:.6f} (F* part {v['psi_U_Fstar_part']:.6f} [quad "
          f"{pop['E_Fstar2_quad']:.6f}], zeta part {v['psi_U_zeta_part']:.6f}, cross "
          f"{v['psi_U_cross']:.1e})")
    print(f"Var psi_o = {v['psi_o']:.6f}, Var psi_T = {v['psi_T']:.6f}, "
          f"Var psi_gamma = {v['psi_gamma']:.6f}")
    print(f"Abar = {np.round(c['Abar'], 6)}; intercept {c['Abar_intercept']:.2e}")
    print(f"sigma^2 = {c['sigma2']:.6f}: V_rotated = {c['V_rotated']:.3f}, V_fixed = "
          f"{c['V_fixed']:.2f}, V_oracle = {c['V_oracle']:.4f}, V_oracle_capq = "
          f"{c['V_oracle_capq']:.4f}")
    pv = p["variances"]
    print(f"plain eps=0.15: Var psi_U={pv['psi_U']:.6f} psi_o={pv['psi_o']:.6f} "
          f"psi_T={pv['psi_T']:.6f} psi_gamma={pv['psi_gamma_ols']:.6f}; V_rotated = "
          f"{p['V_rotated_ols']:.3f}, V_fixed = {p['V_fixed_ols']:.2f}, Abar = "
          f"{np.round(p['Abar'], 6)}")
    print(f"score MC (n={mc['n']:,}): means {np.round(mc['mean'], 5)}; vars "
          + ", ".join(f"{k}={x:.6f}" for k, x in mc["var"].items()))
    print("  corr:\n" + "\n".join("   " + " ".join(f"{x:+.4f}" for x in row) for row in mc["corr"]))
    print(f"  Var(sum)={mc['var_of_sum']:.6f} vs sum Var={mc['sum_of_var']:.6f}; "
          f"V_rotated_mc = {mc['V_rotated_mc']:.3f}")
    print(f"paired capped - eps=0.15: Var(IF diff) = {pop['var_diff_capped_minus_plain']:.6f} "
          f"(MC {mc['var_diff_capped_minus_plain']:.6f}); n Var(diff) rotated = "
          f"{pop['D_rotated']:.3f}, fixed = {pop['D_fixed']:.3f}")


# ------------------------------------------------------------ estimator

def estimate(s, pop: dict, rng) -> dict:
    """Capped, plain eps=0.15 and oracle arms on one data set and one random split."""
    d = DESIGN
    n = len(s.Q)
    blocks = np.array_split(rng.permutation(n), 4)
    grid = np.linspace(d.policy[0], d.policy[1], 61)
    gamma_true = np.concatenate(([d.gamma0], np.asarray(d.gamma_x)))
    a_true, b2_true = Polynomial(d.a_coef), np.asarray(d.beta2, dtype=float)
    ttails = TrueTails(d)
    l_star, u_star = pop["capped"]["window"]
    l_eps0, u_eps0 = pop["capped"]["eps_window"]
    eta_true = s.Q - _xt(s.X) @ gamma_true
    arms = ("capped", "eps15")
    crit = {k: [] for k in arms + ("oracle", "oracle_capq")}
    vals = {k: [] for k in arms}
    diag = {"cap_binds": [], "window_capped": [], "window_eps15": []}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        n_u = len(folds["utility"])
        gamma_hat = ols_first_stage(s, folds["gamma"])
        eta_hat = s.Q - _xt(s.X) @ gamma_hat
        T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], d.T_support)
        l_hat = d.phi0 - float(np.quantile(T_idx, 1.0 - d.eps))
        u_hat = d.phi0 - float(np.quantile(T_idx, d.eps))
        l15 = d.phi0 - float(np.quantile(T_idx, 1.0 - EPS_PLAIN))
        u15 = d.phi0 - float(np.quantile(T_idx, EPS_PLAIN))
        e_all = eta_hat[folds["utility"]]
        xlo, xhi = np.quantile(e_all, [P_LO, P_HI], method="inverted_cdf")
        lc, uc = max(l_hat, float(xlo)), min(u_hat, float(xhi))
        diag["cap_binds"].append(bool(l_hat < xlo and u_hat > xhi))
        diag["window_capped"].append([lc, uc])
        diag["window_eps15"].append([l15, u15])
        fit = outcome_fit(s, folds["outcome"], eta_hat, d.J)
        b2 = np.asarray(fit.beta2)
        for arm, (lo, hi) in (("capped", (lc, uc)), ("eps15", (l15, u15))):
            e = e_all[(e_all >= lo) & (e_all <= hi)]
            pieces = SplineTailPieces(tails, e, n_u)
            amc = fit.a(e) - d.cost
            vals[arm].append([pieces.value(x, amc, b2) for x in grid])
            crit[arm].append(lambda x, p=pieces, a=amc, b=b2: p.value(x, a, b))
        eo_all = eta_true[folds["utility"]]
        eo = eo_all[(eo_all >= l_star) & (eo_all <= u_star)]
        crit["oracle"].append(_criterion(eo, a_true(eo) - d.cost, b2_true, ttails, n_u))
        qlo, qhi = np.quantile(eo_all, [P_LO, P_HI], method="inverted_cdf")
        lq, uq = max(l_eps0, float(qlo)), min(u_eps0, float(qhi))
        eq = eo_all[(eo_all >= lq) & (eo_all <= uq)]
        crit["oracle_capq"].append(_criterion(eq, a_true(eq) - d.cost, b2_true, ttails, n_u))
    out = {}
    for arm in arms:
        out[f"{arm}_avg"], out[f"{arm}_pooled"], rot = _avg_and_pooled(
            crit[arm], grid, np.array(vals[arm]))
        out[f"{arm}_fixed"] = rot[0]
    for arm in ("oracle", "oracle_capq"):
        out[f"{arm}_avg"], out[f"{arm}_pooled"], _ = _avg_and_pooled(crit[arm], grid)
    return {"est": out, "diag": diag}


def replicate(task) -> dict:
    n, rep, seed, pop = task
    t0 = time.perf_counter()
    s = generate(DESIGN, n, seed)
    res = estimate(s, pop, np.random.default_rng([seed, 1]))
    return {"rep": rep, "seed": seed, **res, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------ summaries

PRED = {"capped_avg": "V_rotated", "capped_pooled": "V_rotated", "capped_fixed": "V_fixed",
        "eps15_avg": "P_rotated", "eps15_pooled": "P_rotated", "eps15_fixed": "P_fixed",
        "oracle_avg": "V_oracle", "oracle_pooled": "V_oracle",
        "oracle_capq_avg": "V_oracle_capq", "oracle_capq_pooled": "V_oracle_capq",
        "diff_avg": "D_rotated", "diff_pooled": "D_rotated", "diff_fixed": "D_fixed"}


def predictions(pop: dict) -> dict:
    c, p = pop["capped"], pop["plain_eps15"]
    return {"V_rotated": c["V_rotated"], "V_fixed": c["V_fixed"], "V_oracle": c["V_oracle"],
            "V_oracle_capq": c["V_oracle_capq"], "P_rotated": p["V_rotated_ols"],
            "P_fixed": p["V_fixed_ols"], "D_rotated": pop["D_rotated"],
            "D_fixed": pop["D_fixed"]}


def summarize(rows: List[dict], n: int, pop: dict) -> dict:
    phi = pop["capped"]["phi_star"]
    pred = predictions(pop)
    R = len(rows)
    out = {"n": n, "reps": R, "arms": {},
           "cap_binds_all": float(np.mean([all(r["diag"]["cap_binds"]) for r in rows])),
           "mean_seconds": float(np.mean([r["seconds"] for r in rows]))}
    get = lambda key: np.array([r["est"][key] for r in rows])
    errs = {key: get(key) - phi for key in rows[0]["est"]}
    for v in ("avg", "pooled", "fixed"):          # paired difference, target 0
        errs[f"diff_{v}"] = get(f"capped_{v}") - get(f"eps15_{v}")
    for key, err in errs.items():
        nv = float(n * err.var(ddof=1))
        c = err - err.mean()
        m4 = float(np.mean(c ** 4))
        s2 = float(err.var(ddof=1))
        se_robust = n * np.sqrt(max(m4 - s2 ** 2 * (R - 3) / (R - 1), 0.0) / R)
        V = pred[PRED[key]]
        se = nv * np.sqrt(2.0 / (R - 1))
        out["arms"][key] = {
            "sqrt_n_bias": float(np.sqrt(n) * err.mean()),
            "sqrt_n_bias_se": float(np.sqrt(n) * err.std(ddof=1) / np.sqrt(R)),
            "n_var": nv, "n_var_se": float(se), "n_var_se_kurtosis": float(se_robust),
            "n_mse": float(n * np.mean(err ** 2)), "pred": V,
            "z_vs_pred": float((nv - V) / se),
        }
    return out


def _load(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def run_cell(n: int, reps: int, seed: int, workers: int, out: Path, pop: dict) -> dict:
    path = out / f"n{n}.jsonl"
    rows = _load(path)
    done = {r["rep"] for r in rows}
    tasks = [(n, r, seed + 104_729 * n + r, pop) for r in range(reps) if r not in done]
    if tasks:
        print(f"n={n}: {len(done)} done, running {len(tasks)}", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers) as pool, path.open("a") as fh:
            futs = [pool.submit(replicate, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if i % 50 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}, {time.perf_counter() - t0:.0f}s", flush=True)
    rows = sorted(rows, key=lambda r: r["rep"])[:reps]
    return summarize(rows, n, pop)


LABELS = [("capped_avg", "capped, rotated average"), ("capped_pooled", "capped, pooled"),
          ("capped_fixed", "capped, fixed assignment"), ("oracle_avg", "oracle, average"),
          ("oracle_pooled", "oracle, pooled"),
          ("oracle_capq_avg", "oracle + estimated caps, avg"),
          ("eps15_avg", "eps=0.15, rotated average"), ("eps15_pooled", "eps=0.15, pooled"),
          ("eps15_fixed", "eps=0.15, fixed assignment"),
          ("diff_avg", "capped - eps=0.15, avg (paired)"),
          ("diff_fixed", "capped - eps=0.15, fixed (paired)")]


def report(result: dict) -> str:
    lines = ["| n | estimator | sqrt(n) bias (s.e.) | n Var (MC s.e.) | predicted | z |",
             "|---:|---|---:|---:|---:|---:|"]
    for n, cell in sorted(result["cells"].items(), key=lambda kv: int(kv[0])):
        for key, lab in LABELS:
            c = cell["arms"].get(key)
            if c is None:
                continue
            lines.append(f"| {int(n):,} | {lab} | {c['sqrt_n_bias']:+.2f} ({c['sqrt_n_bias_se']:.2f}) "
                         f"| {c['n_var']:.2f} ({c['n_var_se']:.2f}) | {c['pred']:.2f} "
                         f"| {c['z_vs_pred']:+.2f} |")
    lines.append("")
    lines.append("; ".join(f"n = {int(n):,}: {c['reps']} data sets, caps bind in all 4 "
                           f"rotations in {100 * c['cap_binds_all']:.1f}%"
                           for n, c in sorted(result["cells"].items(), key=lambda kv: int(kv[0]))))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, nargs="+", default=[16000, 64000])
    ap.add_argument("--reps", type=int, nargs="+", default=[1000, 500])
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--population", action="store_true", help="population numbers only")
    ap.add_argument("--report", action="store_true", help="print the table of a finished run")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    if args.report:
        res = json.loads(summary_path.read_text())
        print_population(res["population"])
        print(report(res))
        return
    pop_path = args.out / "population.json"
    pop = population_summary()
    pop_path.write_text(json.dumps(pop, indent=1) + "\n")
    print_population(pop)
    if args.population:
        return
    reps = args.reps if len(args.reps) == len(args.n) else [args.reps[0]] * len(args.n)
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result.update({"population": pop, "settings": {"seed": args.seed}})
    result.setdefault("cells", {})
    for n, R in zip(args.n, reps):
        result["cells"][str(n)] = run_cell(n, R, args.seed, args.workers, args.out, pop)
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
        print(report({"cells": {str(n): result["cells"][str(n)]}}), flush=True)
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
