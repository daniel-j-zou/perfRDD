"""Local efficiency of the four-role differing-slopes estimator (Gaussian designs).

Population side of the efficiency calculation in ``this_week.tex``, section
"Local efficiency at Gaussian eta".  For a design with

    X ~ N(0, diag(sigma_x)),  T = gamma0 + gamma_x'X,  eta ~ N(0, sigma_eta^2),
    Q = T + eta,  D = 1{Q > phi0},
    Y = b(eta) + X'beta1 + D {a(eta) + X'beta2} + sigma_D e,  e ~ N(0, 1),

with polynomial a and b, every object in the paper's four-role theorem has a
closed form or a one-dimensional integral:

* the target phi*_eps and curvature H = U_eps''(phi*_eps);
* the four role scores psi_U, psi_o, psi_T, psi_gamma and their variances;
* the outcome Riesz representer r_out, arm by arm: on arm d,
      r_out = s_d I0(eta) g(phi* - eta) / pi_d(eta) + (X - m_d(eta))' omega_d,
  with s_1 = +1, s_0 = -1, pi_d = P(D = d | eta), m_d = E(X | eta, D = d);
* the summed first-stage loading Abar (its intercept must vanish);
* the outcome equation's information about the first-stage slopes,
      J_Y = sum_d sigma_d^{-2} E[S_J 1{D=d} R_xi R_xi'],
  where R_xi is the residual of c_d'(eta) X from {w(eta) + X'omega} on arm d,
  c_0 = b and c_1 = a + b.

The OLS first stage contributes sigma_eta^2 Abar_X' Var(X)^{-1} Abar_X to the
summed score variance; the efficient first stage contributes
Abar_X' (Var(X)/sigma_eta^2 + J_Y)^{-1} Abar_X.  With the other three role
scores unchanged, these give the rotated (N-scaled) variances

    V_ols = sigma_DS^2 / H^2   and   V_eff = sigma_eff^2 / H^2,

and V_eff is the efficiency bound of the model that imposes the outcome
restriction on the fitting window J (this_week.tex, Proposition E1).

Run:
    python -m experiments.scripts.ds_local_efficiency
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency"


@dataclass(frozen=True)
class Design:
    name: str
    sigma_x: Tuple[float, ...]          # diagonal of Var(X); E X = 0
    gamma0: float
    gamma_x: Tuple[float, ...]
    sigma_eta: float
    a_coef: Tuple[float, ...]           # a(e) = sum_k a_coef[k] e^k
    b_coef: Tuple[float, ...]
    beta1: Tuple[float, ...]
    beta2: Tuple[float, ...]
    sigma_arm: Tuple[float, float]      # outcome noise s.d. on arms 0 and 1
    cost: float
    eps: float = 0.10
    phi0: float = 0.0
    J: Tuple[float, float] = (-1.75, 1.75)          # outcome fitting window
    T_support: Tuple[float, float] = (-3.0, 3.0)    # density interval
    policy: Tuple[float, float] = (-1.5, 1.5)       # Phi
    note: str = ""
    # Law of eta, always with mean 0 and variance sigma_eta^2:
    #   ("normal",), ("t", nu) a rescaled t_nu, or ("mixture", lam) the
    #   equal mixture of N(-lam s, (1 - lam^2) s^2) and N(lam s, (1 - lam^2) s^2).
    eta_law: Tuple = ("normal",)
    # Within-arm heteroskedasticity: noise s.d. sigma_d exp(noise_kappa * eta / 2).
    # Population.decomposition assumes (H) and refuses noise_kappa != 0; use
    # ds_local_efficiency_hetero.hetero_variances for those designs.
    noise_kappa: float = 0.0


DESIGNS: Dict[str, Design] = {
    # Displayed design of differing_slopes.tex Section 5 (target 0.5451).
    "paper_baseline": Design(
        name="paper_baseline",
        sigma_x=(1.0, 1.0, 1.0), gamma0=0.0,
        gamma_x=tuple(np.ones(3) / np.sqrt(3.0)), sigma_eta=1.0,
        a_coef=(2.0, 1.0), b_coef=(0.0, 0.0, 0.5),
        beta1=(0.3, -0.2, 0.1), beta2=(0.4, -0.25, 0.15),
        sigma_arm=(0.5, 0.5), cost=2.25,
        note="differing_slopes.tex Section 5 displayed design",
    ),
    # The paper design with non-Gaussian eta (variance 1 in both cases).
    "paper_t5": Design(
        name="paper_t5",
        sigma_x=(1.0, 1.0, 1.0), gamma0=0.0,
        gamma_x=tuple(np.ones(3) / np.sqrt(3.0)), sigma_eta=1.0,
        a_coef=(2.0, 1.0), b_coef=(0.0, 0.0, 0.5),
        beta1=(0.3, -0.2, 0.1), beta2=(0.4, -0.25, 0.15),
        sigma_arm=(0.5, 0.5), cost=2.25, eta_law=("t", 5.0),
        note="paper design with eta a rescaled t_5",
    ),
    "paper_mix": Design(
        name="paper_mix",
        sigma_x=(1.0, 1.0, 1.0), gamma0=0.0,
        gamma_x=tuple(np.ones(3) / np.sqrt(3.0)), sigma_eta=1.0,
        a_coef=(2.0, 1.0), b_coef=(0.0, 0.0, 0.5),
        beta1=(0.3, -0.2, 0.1), beta2=(0.4, -0.25, 0.15),
        sigma_arm=(0.5, 0.5), cost=2.25, eta_law=("mixture", 0.8),
        note="paper design with eta an equal two-normal mixture (lam = 0.8)",
    ),
    # The paper design with smaller outcome noise (largest gain in the sensitivity grid).
    "paper_lownoise": Design(
        name="paper_lownoise",
        sigma_x=(1.0, 1.0, 1.0), gamma0=0.0,
        gamma_x=tuple(np.ones(3) / np.sqrt(3.0)), sigma_eta=1.0,
        a_coef=(2.0, 1.0), b_coef=(0.0, 0.0, 0.5),
        beta1=(0.3, -0.2, 0.1), beta2=(0.4, -0.25, 0.15),
        sigma_arm=(0.25, 0.25), cost=2.25,
        note="paper design with sigma_0 = sigma_1 = 0.25",
    ),
    # The paper design with heteroskedastic arms, sigma_d(eta) = 0.5 exp(eta / 2).
    "paper_hetero": Design(
        name="paper_hetero",
        sigma_x=(1.0, 1.0, 1.0), gamma0=0.0,
        gamma_x=tuple(np.ones(3) / np.sqrt(3.0)), sigma_eta=1.0,
        a_coef=(2.0, 1.0), b_coef=(0.0, 0.0, 0.5),
        beta1=(0.3, -0.2, 0.1), beta2=(0.4, -0.25, 0.15),
        sigma_arm=(0.5, 0.5), cost=2.25, noise_kappa=1.0,
        note="paper design with within-arm heteroskedastic noise (kappa = 1)",
    ),
    # this_week.tex differing-slopes simulation design: linear a and b.
    "ds_linear": Design(
        name="ds_linear",
        sigma_x=(1.0, 1.0), gamma0=0.0, gamma_x=(1.0, 0.0), sigma_eta=1.0,
        a_coef=(0.35, 0.90), b_coef=(0.20, 0.60),
        beta1=(0.30, -0.20), beta2=(0.80, 0.25),
        sigma_arm=(0.5, 0.5), cost=0.25, policy=(-3.0, 3.0), J=(-2.0, 2.0),
        note="this_week.tex four-role DS check design; linear a, b",
    ),
}


# ---------------------------------------------------------------- eta laws

def _eta_parts(design: "Design"):
    """Return (pdf, score rho = f'/f) for the law of eta in ``design``."""
    sd, law = float(design.sigma_eta), tuple(design.eta_law)
    if law[0] == "normal":
        pdf = lambda e: norm.pdf(np.asarray(e) / sd) / sd
        rho = lambda e: -np.asarray(e) / sd ** 2
        return pdf, rho
    if law[0] == "t":
        from scipy.stats import t as student
        nu = float(law[1])
        s = sd * np.sqrt((nu - 2.0) / nu)
        pdf = lambda e: student.pdf(np.asarray(e) / s, nu) / s
        rho = lambda e: -(nu + 1.0) * np.asarray(e) / (nu * s ** 2 + np.asarray(e) ** 2)
        return pdf, rho
    if law[0] == "mixture":
        lam = float(law[1])
        mu, s = lam * sd, np.sqrt(1.0 - lam ** 2) * sd
        def pdf(e):
            e = np.asarray(e)
            return 0.5 * (norm.pdf((e - mu) / s) + norm.pdf((e + mu) / s)) / s
        def rho(e):
            e = np.asarray(e)
            w1, w2 = norm.pdf((e - mu) / s), norm.pdf((e + mu) / s)
            return -((e - mu) * w1 + (e + mu) * w2) / (s ** 2 * (w1 + w2))
        return pdf, rho
    raise ValueError(f"unknown eta law {law!r}")


def sample_eta(design: "Design", rng, n: int) -> np.ndarray:
    sd, law = float(design.sigma_eta), tuple(design.eta_law)
    if law[0] == "normal":
        return sd * rng.standard_normal(n)
    if law[0] == "t":
        nu = float(law[1])
        return sd * np.sqrt((nu - 2.0) / nu) * rng.standard_t(nu, n)
    if law[0] == "mixture":
        lam = float(law[1])
        sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        return sd * (lam * sign + np.sqrt(1.0 - lam ** 2) * rng.standard_normal(n))
    raise ValueError(f"unknown eta law {law!r}")


# ---------------------------------------------------------------- quadrature

def _gl_nodes(breaks, order: int = 48, pieces: int = 12):
    """Composite Gauss--Legendre nodes and Lebesgue weights on sorted breaks."""
    x, w = leggauss(order)
    nodes, weights = [], []
    breaks = np.unique(np.asarray(breaks, dtype=float))
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        edges = np.linspace(lo, hi, pieces + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            nodes.append(0.5 * (b - a) * x + 0.5 * (b + a))
            weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


# ---------------------------------------------------------------- population

class Population:
    """Closed forms and integrals for one Gaussian design."""

    def __init__(self, design: Design, allow_hetero: bool = False):
        if design.noise_kappa and not allow_hetero:
            raise ValueError("decomposition assumes (H); see ds_local_efficiency_hetero")
        self.d = design
        self.Sigma = np.diag(np.asarray(design.sigma_x, dtype=float))
        self.gx = np.asarray(design.gamma_x, dtype=float)
        self.dim = len(self.gx)
        self.kappa = self.Sigma @ self.gx                 # Cov(X, T)
        self.sT = float(np.sqrt(self.gx @ self.Sigma @ self.gx))
        self.beta2 = np.asarray(design.beta2, dtype=float)
        self.a = Polynomial(design.a_coef)
        self.b = Polynomial(design.b_coef)
        self.da, self.db = self.a.deriv(), self.b.deriv()
        self.c_fun = (self.b, self.a + self.b)            # arm-wise baseline c_d
        self.dc_fun = (self.db, self.da + self.db)
        self.sig_eta = float(design.sigma_eta)
        self.var_eta = self.sig_eta ** 2
        self._f_eta, self.rho_eta = _eta_parts(design)
        e, w = _gl_nodes([-12 * self.sig_eta, 12 * self.sig_eta], pieces=48)
        self.I_eta = float(np.sum(w * self._f_eta(e) * self.rho_eta(e) ** 2))
        self.q = lambda p: design.gamma0 + self.sT * float(norm.ppf(p))
        self.l0 = design.phi0 - self.q(1.0 - design.eps)
        self.u0 = design.phi0 - self.q(design.eps)
        if not (design.J[0] < self.l0 < self.u0 < design.J[1]):
            raise ValueError("trim window must lie inside the fitting window J")
        self.phi, self.H = self._target()

    # -- index law ------------------------------------------------------------
    def g(self, t):
        return norm.pdf((np.asarray(t) - self.d.gamma0) / self.sT) / self.sT

    def dg(self, t):
        t = np.asarray(t)
        return -(t - self.d.gamma0) / self.sT ** 2 * self.g(t)

    def rho(self, t):
        """rho_X(t) = E[X | T=t] g(t), shape (len(t), dim)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return ((t - self.d.gamma0) / self.sT ** 2 * self.g(t))[:, None] * self.kappa

    def drho(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        s = (self.g(t) + (t - self.d.gamma0) * self.dg(t)) / self.sT ** 2
        return s[:, None] * self.kappa

    def f_eta(self, e):
        return self._f_eta(e)

    # -- arm-wise selection quantities ----------------------------------------
    def _z(self, e):
        return (self.d.phi0 - np.asarray(e) - self.d.gamma0) / self.sT

    def pi(self, arm, e):
        z = self._z(e)
        return norm.sf(z) if arm == 1 else norm.cdf(z)

    def _lam(self, arm, e):
        z = self._z(e)
        return (norm.pdf(z) / norm.sf(z) if arm == 1
                else -norm.pdf(z) / norm.cdf(z))

    def m_arm(self, arm, e):
        """E[X | eta = e, D = arm], shape (len(e), dim)."""
        lam = np.atleast_1d(self._lam(arm, e))
        return (lam / self.sT)[:, None] * self.kappa

    def V_arm(self, arm, e):
        """Var(X | eta = e, D = arm), shape (len(e), dim, dim)."""
        z = np.atleast_1d(self._z(e))
        lam = np.atleast_1d(self._lam(arm, e))
        vstar = 1.0 + z * lam - lam ** 2                  # truncated-normal variance
        kk = np.outer(self.kappa, self.kappa) / self.sT ** 2
        return self.Sigma[None] - kk[None] * (1.0 - vstar)[:, None, None]

    # -- target ---------------------------------------------------------------
    def _eta_nodes(self, lo, hi, extra=()):
        e, w = _gl_nodes([lo, hi, *[x for x in extra if lo < x < hi]])
        return e, w * self.f_eta(e)

    def Z(self, phi):
        e, w = self._eta_nodes(self.l0, self.u0)
        F = (self.a(e) - self.d.cost) * self.g(phi - e) + self.rho(phi - e) @ self.beta2
        return float(np.sum(w * F))

    def dZ(self, phi):
        e, w = self._eta_nodes(self.l0, self.u0)
        F = (self.a(e) - self.d.cost) * self.dg(phi - e) + self.drho(phi - e) @ self.beta2
        return float(np.sum(w * F))

    def _target(self):
        lo, hi = self.d.policy
        grid = np.linspace(lo, hi, 601)
        vals = np.array([self.Z(x) for x in grid])
        roots = [brentq(self.Z, grid[i], grid[i + 1])
                 for i in range(len(grid) - 1)
                 if vals[i] < 0.0 <= vals[i + 1]]          # U' = -Z crosses + to -
        if len(roots) != 1:
            raise ValueError(f"expected one interior maximizer, found {roots}")
        phi = float(roots[0])
        return phi, -self.dZ(phi)                          # H = U'' = -Z'

    # -- role scores ----------------------------------------------------------
    def F(self, e):
        e = np.asarray(e)
        return ((self.a(e) - self.d.cost) * self.g(self.phi - e)
                + self.rho(self.phi - e) @ self.beta2)

    def s_star(self, e):
        e = np.asarray(e)
        return (self.da(e) * self.g(self.phi - e)
                - (self.a(e) - self.d.cost) * self.dg(self.phi - e)
                - self.drho(self.phi - e) @ self.beta2)

    def r_g(self, t):
        e = self.phi - np.asarray(t)
        inside = (e >= self.l0) & (e <= self.u0)
        return np.where(inside, (self.a(e) - self.d.cost) * self.f_eta(e), 0.0)

    def r_1(self, t):
        e = self.phi - np.asarray(t)
        inside = (e >= self.l0) & (e <= self.u0)
        return np.where(inside, self.f_eta(e), 0.0)

    def decomposition(self) -> Dict[str, object]:
        d, dim = self.d, self.dim
        phi, l0, u0 = self.phi, self.l0, self.u0
        J0, J1 = d.J
        b_l = float(self.F(l0) * self.f_eta(l0))
        b_u = float(self.F(u0) * self.f_eta(u0))

        # psi_U = -I0 F(eta)
        e_w, w_w = self._eta_nodes(l0, u0)
        var_U = float(np.sum(w_w * self.F(e_w) ** 2))
        mean_U = float(np.sum(w_w * self.F(e_w)))           # = Z(phi*) = 0

        # psi_T = -r_g(T) - r_1(T) beta2'X - b_l z_{1-eps}(T) + b_u z_eps(T)
        q_lo, q_hi = self.q(d.eps), self.q(1.0 - d.eps)
        tlo, thi = d.gamma0 - 9 * self.sT, d.gamma0 + 9 * self.sT
        t, wt = _gl_nodes([tlo, phi - u0, phi - l0, q_lo, q_hi, thi])
        wt = wt * self.g(t)
        slope = float(self.beta2 @ self.kappa) / self.sT ** 2
        v_perp = float(self.beta2 @ self.Sigma @ self.beta2
                       - (self.beta2 @ self.kappa) ** 2 / self.sT ** 2)
        z_hi = ((1.0 - d.eps) - (t <= q_hi)) / self.g(q_hi)
        z_lo = (d.eps - (t <= q_lo)) / self.g(q_lo)
        m_T = (-self.r_g(t) - self.r_1(t) * slope * (t - d.gamma0)
               - b_l * z_hi + b_u * z_lo)
        mean_T = float(np.sum(wt * m_T))
        var_T = float(np.sum(wt * m_T ** 2) + v_perp * np.sum(wt * self.r_1(t) ** 2)
                      - mean_T ** 2)

        # outcome representer, arm by arm
        eJ, wJ = self._eta_nodes(J0, J1, extra=(l0, u0))
        inI0 = (eJ >= l0) & (eJ <= u0)
        gphi = np.where(inI0, self.g(phi - eJ), 0.0)
        B = np.sum((wJ * inI0)[:, None] * self.rho(phi - eJ), axis=0)
        omega, M, N, P = {}, {}, {}, {}
        var_o, A_out_X = 0.0, np.zeros(dim)
        A_out_0 = float(np.sum(wJ * gphi * self.da(eJ)))
        for arm, s in ((0, -1.0), (1, 1.0)):
            pi = self.pi(arm, eJ)
            m = self.m_arm(arm, eJ)
            V = self.V_arm(arm, eJ)
            dc = self.dc_fun[arm](eJ)
            M[arm] = np.einsum("i,ijk->jk", wJ * pi, V)
            N[arm] = np.einsum("i,ijk->jk", wJ * pi * dc, V)
            P[arm] = np.einsum("i,ijk->jk", wJ * pi * dc ** 2, V)
            rhs = B - np.sum((wJ * gphi)[:, None] * m, axis=0)
            omega[arm] = s * np.linalg.solve(M[arm], rhs)
            sig2 = d.sigma_arm[arm] ** 2
            var_o += sig2 * (float(np.sum(wJ * gphi ** 2 / pi))
                             + float(omega[arm] @ M[arm] @ omega[arm]))
            A_out_X += (s * np.sum((wJ * dc * gphi)[:, None] * m, axis=0)
                        + N[arm] @ omega[arm])

        # density loadings A_g = -int h' r_g, A_rho = -int h_rho' beta2 r_1
        tt, wtt = _gl_nodes([phi - u0, phi - l0])
        rg, r1 = self.r_g(tt), self.r_1(tt)
        A_g = -np.concatenate([[np.sum(wtt * self.dg(tt) * rg)],
                               np.sum((wtt * rg)[:, None] * self.drho(tt), axis=0)])
        kk = np.outer(self.kappa, self.kappa)
        dev = tt - d.gamma0
        base = self.Sigma - kk / self.sT ** 2
        dEXXg = (base[None] * self.dg(tt)[:, None, None]
                 + kk[None] / self.sT ** 4
                 * (2 * dev * self.g(tt) + dev ** 2 * self.dg(tt))[:, None, None])
        A_rho = -np.concatenate([[np.sum(wtt * r1 * (self.drho(tt) @ self.beta2))],
                                 np.einsum("i,ijk,k->j", wtt * r1, dEXXg, self.beta2)])

        # evaluation loading (E X = 0) and endpoint loadings
        A_U = np.zeros(dim + 1)
        A_U[0] = float(np.sum(w_w * self.s_star(e_w))) + b_l - b_u
        m_p = lambda p: np.concatenate([[1.0], self.kappa * (self.q(p) - d.gamma0)
                                        / self.sT ** 2])
        A_out = np.concatenate([[A_out_0], A_out_X])
        Abar = A_U - A_out - A_g - A_rho - b_l * m_p(1.0 - d.eps) + b_u * m_p(d.eps)
        AX = Abar[1:]

        # first stage: OLS against the efficient (stacked) first stage
        V_X = self.Sigma
        var_gamma_ols = float(self.var_eta * AX @ np.linalg.solve(V_X, AX))
        J_Y = sum((P[arm] - N[arm] @ np.linalg.solve(M[arm], N[arm]))
                  / d.sigma_arm[arm] ** 2 for arm in (0, 1))
        I_gamma = self.I_eta * V_X + J_Y
        var_gamma_eff = float(AX @ np.linalg.solve(I_gamma, AX))
        # stacked first stage with least-squares score rows (efficient only for
        # Gaussian eta): Abar_X' (Var X / var_eta + J_Y)^{-1} Abar_X for any law
        var_gamma_stack = float(AX @ np.linalg.solve(V_X / self.var_eta + J_Y, AX))

        H2 = self.H ** 2
        common = var_U + var_o + var_T
        s2_ols, s2_eff = common + var_gamma_ols, common + var_gamma_eff
        return {
            "design": asdict(d),
            "phi_star": phi, "H": self.H, "l0": l0, "u0": u0,
            "b_l": b_l, "b_u": b_u,
            "means": {"psi_U": mean_U, "psi_T": mean_T},
            "variances": {"psi_U": var_U, "psi_o": var_o, "psi_T": var_T,
                          "psi_gamma_ols": var_gamma_ols,
                          "psi_gamma_eff": var_gamma_eff,
                          "psi_gamma_stack_ls": var_gamma_stack},
            "Abar": Abar.tolist(), "Abar_intercept": float(Abar[0]),
            "A_U": A_U.tolist(), "A_out": A_out.tolist(),
            "A_g": A_g.tolist(), "A_rho": A_rho.tolist(),
            "omega": {str(k): v.tolist() for k, v in omega.items()},
            "J_Y": J_Y.tolist(), "I_gamma": I_gamma.tolist(),
            "sigma2_ols": s2_ols, "sigma2_eff": s2_eff,
            "V_rotated_ols": s2_ols / H2, "V_rotated_eff": s2_eff / H2,
            "V_rotated_stack_ls": (common + var_gamma_stack) / H2,
            "V_fixed_ols": 4 * s2_ols / H2, "V_fixed_eff": 4 * s2_eff / H2,
            "first_stage_share_ols": var_gamma_ols / s2_ols,
            "efficiency_gain": 1.0 - s2_eff / s2_ols,
            "gamma_variance_ratio": (np.diag(np.linalg.inv(I_gamma))
                                     / np.diag(self.var_eta * np.linalg.inv(V_X))).tolist(),
            "I_eta_times_var_eta": self.I_eta * self.var_eta,
        }


def mc_projection_check(design: Design, n: int = 400_000, seed: int = 1,
                        n_spline: int = 14) -> Dict[str, object]:
    """Independent check of the tangent-space projection by least squares.

    On a Monte Carlo sample, the slope score S_gamma (Gaussian eta) is regressed
    on generic bases for the nuisance tangent space: splines in eta (T_eta),
    polynomials in X (T_X), and (u / sigma_D^2) times arm-wise splines in eta
    and X on J, plus a rich arm-wise basis off J (T_mu of the model M_J).  The
    residual covariance estimates I_gamma; regressing psi_gamma on the same
    bases plus S_gamma estimates Var(Pi_T psi_gamma).  Neither step uses the
    truncated-normal formulas of ``Population.decomposition``.
    """
    from scipy.interpolate import BSpline
    pop = Population(design)
    res = pop.decomposition()
    rng = np.random.default_rng(seed)
    dim = pop.dim
    X = rng.standard_normal((n, dim)) * np.sqrt(np.asarray(design.sigma_x))
    eta = sample_eta(design, rng, n)
    T = design.gamma0 + X @ pop.gx
    D = (T + eta > design.phi0).astype(float)
    sig2 = np.where(D == 1, design.sigma_arm[1], design.sigma_arm[0]) ** 2
    u = np.sqrt(sig2) * rng.standard_normal(n)
    xi = np.where(D == 1, pop.dc_fun[1](eta), pop.dc_fun[0](eta))
    inJ = ((eta >= design.J[0]) & (eta <= design.J[1])).astype(float)

    def bsp(x, lo, hi, k):
        t = np.concatenate([[lo] * 4, np.linspace(lo, hi, k - 2)[1:-1], [hi] * 4])
        return BSpline.design_matrix(np.clip(x, lo, hi), t, 3).toarray()

    w = u / sig2
    Be_all = bsp(eta, -6 * pop.sig_eta, 6 * pop.sig_eta, 2 * n_spline)
    BeJ = bsp(eta, design.J[0], design.J[1], n_spline)
    cols = [Be_all,                                   # T_eta (with constant)
            X, X ** 2]                                # part of T_X
    for arm in (0.0, 1.0):
        on = (D == arm).astype(float)
        cols.append((w * on * inJ)[:, None] * BeJ)
        cols.append((w * on * inJ)[:, None] * X)
        off = on * (1.0 - inJ)                        # M_J: free off the window
        cols.append((w * off)[:, None] * Be_all)
        cols.append((w * off)[:, None] * X)
        cols.append((w * off)[:, None] * (X * eta[:, None]))
    Nmat = np.column_stack(cols)
    S = (-pop.rho_eta(eta))[:, None] * X - (w * inJ * xi)[:, None] * X
    coefN = np.linalg.lstsq(Nmat, S, rcond=None)[0]
    S_eff = S - Nmat @ coefN
    I_mc = S_eff.T @ S_eff / n
    AX = np.asarray(res["Abar"])[1:]
    psi_g = (X @ np.linalg.solve(pop.Sigma, AX)) * eta
    full = np.column_stack((Nmat, S))
    fit = full @ np.linalg.lstsq(full, psi_g, rcond=None)[0]
    return {
        "I_gamma_formula": res["I_gamma"], "I_gamma_mc": I_mc.tolist(),
        "var_proj_formula": res["variances"]["psi_gamma_eff"],
        "var_proj_mc": float(np.var(fit)),
        "var_psi_gamma_ols_formula": res["variances"]["psi_gamma_ols"],
        "var_psi_gamma_ols_mc": float(np.var(psi_g)),
    }


def sensitivity(base: Design) -> list:
    """Gain of the efficient first stage as the design moves away from ``base``."""
    from dataclasses import replace
    variants = []
    for s in (0.25, 0.5, 1.0):
        variants.append((f"sigma_arm={s}", replace(base, sigma_arm=(s, s))))
    for h in (1.5, 2.5, 3.5):
        variants.append((f"J=+-{h}", replace(base, J=(-h, h))))
    variants.append(("b linear (b=0)", replace(base, b_coef=(0.0,))))
    variants.append(("b cubic (eta^2/2 + eta^3/6)", replace(base, b_coef=(0.0, 0.0, 0.5, 1 / 6))))
    variants.append(("a quadratic (+0.3 eta^2)", replace(base, a_coef=(2.0, 1.0, 0.3))))
    variants.append(("beta2 = 0 (common slope truth)", replace(base, beta2=(0.0, 0.0, 0.0))))
    rows = []
    for label, design in variants:
        try:
            r = Population(design).decomposition()
        except ValueError as exc:
            rows.append({"variant": label, "error": str(exc)})
            continue
        rows.append({
            "variant": label, "phi_star": r["phi_star"], "H": r["H"],
            "V_rotated_ols": r["V_rotated_ols"], "V_rotated_eff": r["V_rotated_eff"],
            "first_stage_share_ols": r["first_stage_share_ols"],
            "efficiency_gain": r["efficiency_gain"],
            "gamma_variance_ratio": float(np.mean(r["gamma_variance_ratio"])),
            "Abar_intercept": r["Abar_intercept"],
        })
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", nargs="+",
                        default=sorted(k for k, v in DESIGNS.items() if not v.noise_kappa))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mc-check", action="store_true",
                        help="also run the least-squares projection check")
    parser.add_argument("--sensitivity", action="store_true",
                        help="vary the paper design one feature at a time")
    args = parser.parse_args(argv)
    if args.sensitivity:
        rows = sensitivity(DESIGNS["paper_baseline"])
        for r in rows:
            if "error" in r:
                print(f"{r['variant']:32s} {r['error']}")
                continue
            print(f"{r['variant']:32s} phi*={r['phi_star']:+.4f} V_ols={r['V_rotated_ols']:8.3f} "
                  f"V_eff={r['V_rotated_eff']:8.3f} gain={100 * r['efficiency_gain']:5.2f}% "
                  f"share={100 * r['first_stage_share_ols']:5.2f}% "
                  f"gamma ratio={r['gamma_variance_ratio']:.3f} A0={r['Abar_intercept']:.1e}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "sensitivity.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"[wrote] {args.out / 'sensitivity.json'}")
        return
    results = {}
    for name in args.designs:
        if DESIGNS[name].noise_kappa:
            print(f"[{name}] skipped: heteroskedastic; use ds_local_efficiency_hetero")
            continue
        res = Population(DESIGNS[name]).decomposition()
        if args.mc_check:
            res["mc_projection_check"] = mc_projection_check(DESIGNS[name])
            mc = res["mc_projection_check"]
            print(f"[{name}] MC check: Var(Pi psi_gamma) formula={mc['var_proj_formula']:.5f} "
                  f"mc={mc['var_proj_mc']:.5f}; diag I_gamma formula="
                  f"{np.round(np.diag(mc['I_gamma_formula']), 4)} mc={np.round(np.diag(mc['I_gamma_mc']), 4)}")
        results[name] = res
        v = res["variances"]
        print(f"[{name}] phi*={res['phi_star']:.6f} H={res['H']:.6f} "
              f"Abar_0={res['Abar_intercept']:.2e}")
        print(f"   Var psi_U={v['psi_U']:.5f} psi_o={v['psi_o']:.5f} psi_T={v['psi_T']:.5f} "
              f"gamma_ols={v['psi_gamma_ols']:.6f} gamma_eff={v['psi_gamma_eff']:.6f}")
        print(f"   I_eta*var={res['I_eta_times_var_eta']:.3f} V_rot stack_ls={res['V_rotated_stack_ls']:.3f}")
        print(f"   V_rot ols={res['V_rotated_ols']:.3f} eff={res['V_rotated_eff']:.3f} "
              f"gain={100 * res['efficiency_gain']:.2f}% "
              f"first-stage share={100 * res['first_stage_share_ols']:.2f}%")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "population.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"[wrote] {args.out / 'population.json'}")


if __name__ == "__main__":
    main()
