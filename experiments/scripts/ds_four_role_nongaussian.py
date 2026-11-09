"""Four-role estimator on the two non-Gaussian index designs of Appendix 5.

``differing_slopes.tex`` Appendix 5 used to report a ``t_5 scaled to unit
variance'' index and a ``standardized two-component normal mixture'' index,
run with full-sample reuse (``hard_trim_robustness_monte_carlo.py`` on the
design of ``hard_trim_robustness_smoke.make_sample``; targets 0.587810 and
0.793023).  This script reruns those two designs with the paper's own
estimator: the four-role split of Section 3, a fixed assignment and the two
rotated estimators of Table ``tab:ds-four-role-paper`` (average of the four
rotated thresholds and maximizer of the averaged criterion).

Design (identical to ``make_sample``; the RNG call order is kept, so a seed
gives the same data set):

    X = (X1, X2, X3),  X1 ~ law (mean 0, variance 1),  X2, X3 ~ N(0, 1),
    T = X1,  eta ~ N(0, 1),  Q = T + eta,  D = 1{Q > 0},
    Y = D (2 + eta) + eta^2 / 2 + X'(0.3, -0.2, 0.1) + N(0, 0.5^2),
    c = 2.25,  eps = 0.10,  trim window from the T-quantiles.

    law ``t5``:      X1 = t_5 / sqrt(5/3);
    law ``mixture``: X1 = standardized 0.7 N(-0.4, 0.7^2) + 0.3 N(1.0, 0.8^2).

So alpha(eta) = 2 + eta, b(eta) = eta^2 / 2, beta1 = (0.3, -0.2, 0.1) and
beta2 = 0 (common slopes); the estimator still fits the full D X block and
rho_X.  Estimator settings are those of Section 5.1: least-squares first
stage, J = [-1.75, 1.75], T = [-3, 3], Phi = [-1.5, 1.5] on a 61-point grid
refined by a bounded scalar optimizer, knot rules max{3, [m^{11/60}]} and
max{8, [2 n_T^{11/60}]}.  The fold code is that of ``ds_bias_decomposition``
(which produced Table tab:ds-four-role-paper) and ``ds_local_efficiency_sim``.

Population side.  ``ds_local_efficiency.Population`` assumes a Gaussian X.
``NonGaussianPopulation`` ports its four-role decomposition (target, H, the
four role-score variances, the summed first-stage loading Abar) to a general
law of T = X1 with X2, X3 independent N(0, 1); the truncated-normal arm
moments are replaced by closed-form partial moments of the index law.  With
``law="normal"`` it must reproduce ``Population`` (checked by
``--population-only``), and a large-n Monte Carlo of the role scores checks
the quadrature and Corollary ``cor:ds-uncorrelated``.

Run (from the code directory):
    python -m experiments.scripts.ds_four_role_nongaussian --population-only
    python -m experiments.scripts.ds_four_role_nongaussian \
        --n 4000 16000 64000 --reps 400 200 100 --workers 2
    python -m experiments.scripts.ds_four_role_nongaussian --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm, t as student_t

from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_local_efficiency import _gl_nodes
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, Sample, _xt, ols_first_stage, outcome_fit,
)
from experiments.scripts.hard_trim_robustness_smoke import (
    BETA as BETA1_OLD, GAMMA as GAMMA_OLD, MIXTURE_COMPONENTS, MIXTURE_MEAN,
    MIXTURE_SD, _running_variable, population_truth, running_quantile,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "four_role_nongaussian"
LAWS = ("t5", "mixture")
EPS = 0.10
COST = 2.25
PHI0 = 0.0
J = (-1.75, 1.75)
T_SUPPORT = (-3.0, 3.0)
POLICY = (-1.5, 1.5)
SIGMA_Y = 0.5
A_COEF = (2.0, 1.0)            # alpha(eta) = 2 + eta
B_COEF = (0.0, 0.0, 0.5)       # b(eta) = eta^2 / 2
BETA2 = np.zeros(3)
T5_NU = 5.0
T5_SCALE = float(np.sqrt(T5_NU / (T5_NU - 2.0)))
EDGE_TOL = 1e-3
ESTIMATORS = ("fixed", "average", "pooled")


# ---------------------------------------------------------------- index laws

class IndexLaw:
    """Vectorized density, derivative, cdf and upper partial moments of T = X1."""

    def __init__(self, law: str):
        if law not in ("t5", "mixture", "normal"):
            raise ValueError(f"unknown index law {law!r}")
        self.law = law

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        if self.law == "normal":
            return norm.pdf(t)
        if self.law == "t5":
            return T5_SCALE * student_t.pdf(T5_SCALE * t, T5_NU)
        raw = MIXTURE_MEAN + MIXTURE_SD * t
        return MIXTURE_SD * sum(w * norm.pdf(raw, m, s) for w, m, s in MIXTURE_COMPONENTS)

    def dpdf(self, t):
        t = np.asarray(t, dtype=float)
        if self.law == "normal":
            return -t * norm.pdf(t)
        if self.law == "t5":
            u = T5_SCALE * t
            return (T5_SCALE ** 2 * student_t.pdf(u, T5_NU)
                    * (-(T5_NU + 1.0) * u / (T5_NU + u ** 2)))
        raw = MIXTURE_MEAN + MIXTURE_SD * t
        return MIXTURE_SD ** 2 * sum(
            w * norm.pdf(raw, m, s) * (-(raw - m) / s ** 2) for w, m, s in MIXTURE_COMPONENTS)

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        if self.law == "normal":
            return norm.cdf(t)
        if self.law == "t5":
            return student_t.cdf(T5_SCALE * t, T5_NU)
        raw = MIXTURE_MEAN + MIXTURE_SD * t
        return sum(w * norm.cdf(raw, m, s) for w, m, s in MIXTURE_COMPONENTS)

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        if self.law == "normal":
            return norm.sf(t)
        if self.law == "t5":
            return student_t.sf(T5_SCALE * t, T5_NU)
        raw = MIXTURE_MEAN + MIXTURE_SD * t
        return sum(w * norm.sf(raw, m, s) for w, m, s in MIXTURE_COMPONENTS)

    def quantile(self, p: float) -> float:
        if self.law == "normal":
            return float(norm.ppf(p))
        return running_quantile(p, self.law)

    def upper_moments(self, s):
        """(E[T 1{T > s}], E[T^2 1{T > s}]) in closed form."""
        s = np.asarray(s, dtype=float)
        if self.law == "normal":
            return norm.pdf(s), s * norm.pdf(s) + norm.sf(s)
        if self.law == "t5":
            nu, a = T5_NU, T5_SCALE * s
            f, sbar = student_t.pdf(a, nu), student_t.sf(a, nu)
            m1 = (nu + a ** 2) / (nu - 1.0) * f / T5_SCALE
            m2 = (a * (nu + a ** 2) * f + nu * sbar) / (nu - 2.0) / T5_SCALE ** 2
            return m1, m2
        m1 = np.zeros_like(s)
        m2 = np.zeros_like(s)
        raw = MIXTURE_MEAN + MIXTURE_SD * s
        for w, m, sd in MIXTURE_COMPONENTS:
            alpha = (raw - m) / sd
            delta = m - MIXTURE_MEAN
            phi, sbar = norm.pdf(alpha), norm.sf(alpha)
            m1 = m1 + w * (delta * sbar + sd * phi)
            m2 = m2 + w * (sd ** 2 * (alpha * phi + sbar) + 2 * sd * delta * phi
                           + delta ** 2 * sbar)
        return m1 / MIXTURE_SD, m2 / MIXTURE_SD ** 2


# ---------------------------------------------------------------- population

class NonGaussianPopulation:
    """Four-role first-order variance for T = X1 with a general law.

    A port of ``ds_local_efficiency.Population.decomposition`` (OLS first
    stage) to X = (T, X2, X3) with X2, X3 ~ N(0, 1) independent of T, Gaussian
    eta and beta2 = 0.  Then E(X | T=t) = (t, 0, 0), rho_X(t) = (t g(t), 0, 0),
    and given eta = e and arm d the law of X is that of T on the half-line
    {T > -e} (d = 1) or {T <= -e} (d = 0), with X2, X3 unchanged.
    """

    def __init__(self, law: str):
        self.law = law
        self.T = IndexLaw(law)
        self.a = Polynomial(A_COEF)
        self.b = Polynomial(B_COEF)
        self.da, self.db = self.a.deriv(), self.b.deriv()
        self.dc_fun = (self.db, self.da + self.db)      # c_0 = b, c_1 = a + b
        self.dim = 3
        self.q_lo = self.T.quantile(EPS)
        self.q_hi = self.T.quantile(1.0 - EPS)
        self.l0 = PHI0 - self.q_hi
        self.u0 = PHI0 - self.q_lo
        if not (J[0] < self.l0 < self.u0 < J[1]):
            raise ValueError("trim window must lie inside J")
        self.phi, self.H = self._target()

    # -- index law pieces ------------------------------------------------------
    def g(self, t):
        return self.T.pdf(t)

    def dg(self, t):
        return self.T.dpdf(t)

    def rho(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros((len(t), self.dim))
        out[:, 0] = t * self.g(t)
        return out

    def drho(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros((len(t), self.dim))
        out[:, 0] = self.g(t) + t * self.dg(t)
        return out

    @staticmethod
    def f_eta(e):
        return norm.pdf(e)

    def pi(self, arm, e):
        s = PHI0 - np.asarray(e, dtype=float)
        return self.T.sf(s) if arm == 1 else self.T.cdf(s)

    def _arm_moments(self, arm, e):
        s = np.atleast_1d(PHI0 - np.asarray(e, dtype=float))
        m1, m2 = self.T.upper_moments(s)
        if arm == 1:
            p = self.T.sf(s)
            mean, second = m1 / p, m2 / p
        else:
            p = self.T.cdf(s)
            mean, second = -m1 / p, (1.0 - m2) / p      # E T = 0, E T^2 = 1
        return mean, second - mean ** 2

    def m_arm(self, arm, e):
        mean, _ = self._arm_moments(arm, e)
        out = np.zeros((len(mean), self.dim))
        out[:, 0] = mean
        return out

    def V_arm(self, arm, e):
        _, var = self._arm_moments(arm, e)
        out = np.zeros((len(var), self.dim, self.dim))
        out[:, 0, 0] = var
        out[:, 1, 1] = 1.0
        out[:, 2, 2] = 1.0
        return out

    # -- target ------------------------------------------------------------------
    def _eta_nodes(self, lo, hi, extra=()):
        e, w = _gl_nodes([lo, hi, *[x for x in extra if lo < x < hi]])
        return e, w * self.f_eta(e)

    def Z(self, phi):
        e, w = self._eta_nodes(self.l0, self.u0)
        return float(np.sum(w * (self.a(e) - COST) * self.g(phi - e)))

    def dZ(self, phi):
        e, w = self._eta_nodes(self.l0, self.u0)
        return float(np.sum(w * (self.a(e) - COST) * self.dg(phi - e)))

    def _target(self):
        grid = np.linspace(POLICY[0], POLICY[1], 601)
        vals = np.array([self.Z(x) for x in grid])
        roots = [brentq(self.Z, grid[i], grid[i + 1], xtol=1e-14)
                 for i in range(len(grid) - 1) if vals[i] < 0.0 <= vals[i + 1]]
        if len(roots) != 1:
            raise ValueError(f"expected one interior maximizer, found {roots}")
        phi = float(roots[0])
        return phi, -self.dZ(phi)

    # -- role-score pieces -------------------------------------------------------
    def F(self, e):
        e = np.asarray(e, dtype=float)
        return (self.a(e) - COST) * self.g(self.phi - e)

    def s_star(self, e):
        e = np.asarray(e, dtype=float)
        return self.da(e) * self.g(self.phi - e) - (self.a(e) - COST) * self.dg(self.phi - e)

    def r_g(self, t):
        e = self.phi - np.asarray(t, dtype=float)
        inside = (e >= self.l0) & (e <= self.u0)
        return np.where(inside, (self.a(e) - COST) * self.f_eta(e), 0.0)

    def m_T(self, t):
        """psi_T(t) = -r_g(t) - b_l zeta_{1-eps}(t) + b_u zeta_eps(t) (beta2 = 0)."""
        t = np.asarray(t, dtype=float)
        z_hi = ((1.0 - EPS) - (t <= self.q_hi)) / self.g(self.q_hi)
        z_lo = (EPS - (t <= self.q_lo)) / self.g(self.q_lo)
        return -self.r_g(t) - self.b_l * z_hi + self.b_u * z_lo

    def decomposition(self) -> dict:
        phi, l0, u0, dim = self.phi, self.l0, self.u0, self.dim
        self.b_l = b_l = float(self.F(l0) * self.f_eta(l0))
        self.b_u = b_u = float(self.F(u0) * self.f_eta(u0))

        # psi_U = -I0 F(eta)
        e_w, w_w = self._eta_nodes(l0, u0)
        var_U = float(np.sum(w_w * self.F(e_w) ** 2))
        mean_U = float(np.sum(w_w * self.F(e_w)))

        # psi_T on the law of T; m_T is constant beyond [tlo, thi]
        tlo, thi = -40.0, 40.0
        t, wt = _gl_nodes([tlo, phi - u0, phi - l0, self.q_lo, self.q_hi, thi], pieces=24)
        wt = wt * self.g(t)
        mt = self.m_T(t)
        p_lo, p_hi = float(self.T.cdf(tlo)), float(self.T.sf(thi))
        m_lo, m_hi = float(self.m_T(tlo - 1.0)), float(self.m_T(thi + 1.0))
        mean_T = float(np.sum(wt * mt) + p_lo * m_lo + p_hi * m_hi)
        var_T = float(np.sum(wt * mt ** 2) + p_lo * m_lo ** 2 + p_hi * m_hi ** 2
                      - mean_T ** 2)

        # outcome representer, arm by arm
        eJ, wJ = self._eta_nodes(J[0], J[1], extra=(l0, u0))
        inI0 = (eJ >= l0) & (eJ <= u0)
        gphi = np.where(inI0, self.g(phi - eJ), 0.0)
        B = np.sum((wJ * inI0)[:, None] * self.rho(phi - eJ), axis=0)     # ell_2
        omega, M, N = {}, {}, {}
        var_o, A_out_X = 0.0, np.zeros(dim)
        A_out_0 = float(np.sum(wJ * gphi * self.da(eJ)))
        for arm, sgn in ((0, -1.0), (1, 1.0)):
            pi = self.pi(arm, eJ)
            m = self.m_arm(arm, eJ)
            V = self.V_arm(arm, eJ)
            dc = self.dc_fun[arm](eJ)
            M[arm] = np.einsum("i,ijk->jk", wJ * pi, V)                    # Gamma_d
            N[arm] = np.einsum("i,ijk->jk", wJ * pi * dc, V)
            rhs = B - np.sum((wJ * gphi)[:, None] * m, axis=0)
            omega[arm] = sgn * np.linalg.solve(M[arm], rhs)
            var_o += SIGMA_Y ** 2 * (float(np.sum(wJ * gphi ** 2 / pi))
                                     + float(omega[arm] @ M[arm] @ omega[arm]))
            A_out_X += (sgn * np.sum((wJ * dc * gphi)[:, None] * m, axis=0)
                        + N[arm] @ omega[arm])
        self.omega = omega

        # density loading A_T = -int h' r_g (the rho_X part carries beta2 = 0)
        tt, wtt = _gl_nodes([phi - u0, phi - l0])
        rg = self.r_g(tt)
        A_g = -np.concatenate([[np.sum(wtt * self.dg(tt) * rg)],
                               np.sum((wtt * rg)[:, None] * self.drho(tt), axis=0)])

        A_U = np.zeros(dim + 1)
        A_U[0] = float(np.sum(w_w * self.s_star(e_w))) + b_l - b_u
        m_p = lambda q: np.array([1.0, q, 0.0, 0.0])                        # E(X~ | T = q)
        A_out = np.concatenate([[A_out_0], A_out_X])
        Abar = A_U - A_out - A_g - b_l * m_p(self.q_hi) + b_u * m_p(self.q_lo)
        self.AX = AX = Abar[1:]
        var_gamma = float(AX @ AX)                                          # Var X = I, Var eta = 1

        H2 = self.H ** 2
        s2 = var_U + var_o + var_T + var_gamma
        return {
            "law": self.law, "phi_star": phi, "H": self.H, "l0": l0, "u0": u0,
            "q_eps": self.q_lo, "q_1meps": self.q_hi, "b_l": b_l, "b_u": b_u,
            "means": {"psi_U": mean_U, "psi_T": mean_T},
            "variances": {"psi_U": var_U, "psi_o": var_o, "psi_T": var_T,
                          "psi_gamma": var_gamma},
            "Abar": Abar.tolist(), "Abar_intercept": float(Abar[0]),
            "sigma2": s2, "V_rotated": s2 / H2, "V_fixed": 4.0 * s2 / H2,
            "V_oracle": var_U / H2,
            "argument_set": [POLICY[0] - u0, POLICY[1] - l0],
        }

    def mc_role_scores(self, n_draw: int, seed: int = 1, chunk: int = 250_000) -> dict:
        """Large-n Monte Carlo of the four role scores on the design's own draws."""
        if not hasattr(self, "AX"):
            self.decomposition()
        rng = np.random.default_rng(seed)
        sums = np.zeros(4)
        cross = np.zeros((4, 4))
        done = 0
        while done < n_draw:
            m = min(chunk, n_draw - done)
            if self.law == "normal":
                X1 = rng.standard_normal(m)
            else:
                X1 = _running_variable(rng, m, self.law)
            X = np.column_stack((X1, rng.standard_normal((m, 2))))
            eta = rng.standard_normal(m)
            R = SIGMA_Y * rng.standard_normal(m)
            D = (X1 + eta > PHI0).astype(int)
            I0 = (eta >= self.l0) & (eta <= self.u0)
            S = (eta >= J[0]) & (eta <= J[1])
            psi_U = -np.where(I0, self.F(eta), 0.0)
            r_out = np.zeros(m)
            for arm, sgn in ((0, -1.0), (1, 1.0)):
                on = S & (D == arm)
                e = eta[on]
                lead = sgn * np.where(I0[on], self.g(self.phi - e), 0.0) / self.pi(arm, e)
                r_out[on] = lead + np.sum((X[on] - self.m_arm(arm, e)) * self.omega[arm], axis=1)
            psi_o = -r_out * R
            psi_T = self.m_T(X1)
            psi_g = (X @ self.AX) * eta
            P = np.column_stack((psi_U, psi_o, psi_T, psi_g))
            sums += P.sum(axis=0)
            cross += P.T @ P
            done += m
        mean = sums / n_draw
        cov = cross / n_draw - np.outer(mean, mean)
        sd = np.sqrt(np.diag(cov))
        names = ("psi_U", "psi_o", "psi_T", "psi_gamma")
        return {
            "n_draw": n_draw,
            "mean": dict(zip(names, mean.tolist())),
            "variance": dict(zip(names, np.diag(cov).tolist())),
            "correlation": (cov / np.outer(sd, sd)).tolist(),
            "V_rotated_mc": float(np.trace(cov) / self.H ** 2),
            "V_rotated_mc_sum": float(cov.sum() / self.H ** 2),
        }


def gaussian_port_check() -> dict:
    """The port with a Gaussian index must reproduce ds_local_efficiency.Population."""
    from experiments.scripts.ds_local_efficiency import Design, Population
    design = Design(
        name="smoke_gaussian_index", sigma_x=(1.0, 1.0, 1.0), gamma0=0.0,
        gamma_x=(1.0, 0.0, 0.0), sigma_eta=1.0, a_coef=A_COEF, b_coef=B_COEF,
        beta1=tuple(BETA1_OLD.tolist()), beta2=(0.0, 0.0, 0.0),
        sigma_arm=(SIGMA_Y, SIGMA_Y), cost=COST, eps=EPS, phi0=PHI0, J=J,
        T_support=T_SUPPORT, policy=POLICY)
    ref = Population(design).decomposition()
    mine = NonGaussianPopulation("normal").decomposition()
    keys = ("psi_U", "psi_o", "psi_T")
    return {
        "phi_star": (ref["phi_star"], mine["phi_star"]),
        "H": (ref["H"], mine["H"]),
        **{k: (ref["variances"][k], mine["variances"][k]) for k in keys},
        "psi_gamma": (ref["variances"]["psi_gamma_ols"], mine["variances"]["psi_gamma"]),
        "Abar": (ref["Abar"], mine["Abar"]),
        "V_rotated": (ref["V_rotated_ols"], mine["V_rotated"]),
    }


# ---------------------------------------------------------------- simulation

def generate(law: str, n: int, seed: int) -> Sample:
    """``hard_trim_robustness_smoke.make_sample`` (correct outcome), same RNG order."""
    rng = np.random.default_rng(seed)
    X = np.column_stack((_running_variable(rng, n, law), rng.standard_normal((n, 2))))
    eta = rng.standard_normal(n)
    T = X @ GAMMA_OLD
    Q = T + eta
    D = (Q > 0.0).astype(float)
    Y = D * (2.0 + eta) + 0.5 * eta ** 2 + X @ BETA1_OLD
    Y += rng.normal(0.0, 0.5, size=n)
    return Sample(X=X, Q=Q, D=D, Y=Y)


def rotation_criterion(s: Sample, folds: dict, gamma_hat: np.ndarray):
    """Fitted criterion U-hat of one role assignment (paper Section 3)."""
    eta_hat = s.Q - _xt(s.X) @ gamma_hat
    fit = outcome_fit(s, folds["outcome"], eta_hat, J)
    T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
    tails = fit_weighted_tails(T_idx, s.X[folds["index"]], T_SUPPORT)
    l_hat = PHI0 - float(np.quantile(T_idx, 1.0 - EPS))
    u_hat = PHI0 - float(np.quantile(T_idx, EPS))
    n_u = len(folds["utility"])
    e = eta_hat[folds["utility"]]
    e = e[(e >= l_hat) & (e <= u_hat)]
    amc = fit.a(e) - COST
    b2 = fit.beta2

    def value(phi):
        t = phi - e
        return float(np.sum(amc * tails.survival(t) + tails.weighted_tail(t, b2)) / n_u)

    info = {"K_interior": int(len(fit.info["t"]) - 2 * (fit.info["degree"] + 1)),
            "L": int(tails.n_basis), "l_hat": l_hat, "u_hat": u_hat,
            "beta2_hat_1": float(b2[0]), "gamma_hat_1": float(gamma_hat[1])}
    return value, info


def argmax(value) -> float:
    lo, hi = POLICY
    grid = np.linspace(lo, hi, 61)
    vals = np.array([value(x) for x in grid])
    k = int(np.argmax(vals))
    a_, b_ = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: -value(x), bounds=(a_, b_), method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x)


def at_edge(x: float) -> bool:
    return min(abs(x - POLICY[0]), abs(x - POLICY[1])) < EDGE_TOL


def replicate(task):
    law, n, seed = task
    s = generate(law, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    phis, values, infos = [], [], []
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        gamma_hat = ols_first_stage(s, folds["gamma"])
        value, info = rotation_criterion(s, folds, gamma_hat)
        phis.append(argmax(value))
        values.append(value)
        infos.append(info)
    pooled = argmax(lambda x: sum(v(x) for v in values) / 4.0)
    return {
        "law": law, "n": n, "seed": seed, "rotations": phis,
        "fixed": phis[0], "average": float(np.mean(phis)), "pooled": pooled,
        "edge_rotations": int(sum(at_edge(p) for p in phis)),
        "edge_fixed": bool(at_edge(phis[0])), "edge_pooled": bool(at_edge(pooled)),
        "info0": infos[0],
    }


# ---------------------------------------------------------------- summaries

def summarize_cell(rows: list, n: int, pop: dict) -> dict:
    phi = pop["phi_star"]
    reps = len(rows)
    out = {"n": n, "reps": reps, "n_var_rel_mc_se": float(np.sqrt(2.0 / (reps - 1))),
           "seed_range": [int(min(r["seed"] for r in rows)), int(max(r["seed"] for r in rows))]}
    for key in ESTIMATORS:
        err = np.array([r[key] for r in rows]) - phi
        out[key] = {
            "sqrtN_bias": float(np.sqrt(n) * err.mean()),
            "sqrtN_bias_se": float(np.sqrt(n) * err.std(ddof=1) / np.sqrt(reps)),
            "n_var": float(n * err.var(ddof=1)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "n_mse": float(n * np.mean(err ** 2)),
            "sd_units_bias": float(err.mean() / err.std(ddof=1)),
        }
    out["ratio_fixed_over_average"] = out["fixed"]["n_var"] / out["average"]["n_var"]
    out["ratio_fixed_over_pooled"] = out["fixed"]["n_var"] / out["pooled"]["n_var"]
    out["boundary"] = {
        "rotation_thresholds_at_edge": int(sum(r["edge_rotations"] for r in rows)),
        "rotation_thresholds_total": 4 * reps,
        "fixed_at_edge": int(sum(r["edge_fixed"] for r in rows)),
        "pooled_at_edge": int(sum(r["edge_pooled"] for r in rows)),
        "datasets_with_any_edge": int(sum((r["edge_rotations"] > 0) or r["edge_pooled"]
                                          for r in rows)),
    }
    out["basis"] = {"K_interior": sorted({r["info0"]["K_interior"] for r in rows}),
                    "L": sorted({r["info0"]["L"] for r in rows})}
    out["prediction"] = {"V_rotated": pop["V_rotated"], "V_fixed": pop["V_fixed"],
                         "ratio_avg_to_pred": out["average"]["n_var"] / pop["V_rotated"],
                         "ratio_pooled_to_pred": out["pooled"]["n_var"] / pop["V_rotated"],
                         "ratio_fixed_to_pred": out["fixed"]["n_var"] / pop["V_fixed"]}
    return out


def rmse_slopes(cells: dict) -> dict:
    ns = sorted(int(k) for k in cells)
    if len(ns) < 2:
        return {}
    x = np.log(ns)
    return {key: float(np.polyfit(x, np.log([cells[str(n)][key]["rmse"] for n in ns]), 1)[0])
            for key in ESTIMATORS}


def markdown(summary: dict) -> str:
    lines = ["# Four-role estimator on the Appendix 5 non-Gaussian index designs", ""]
    for law, block in summary["laws"].items():
        pop = block["population"]
        lines += [f"## {law}: phi* = {pop['phi_star']:.6f}, H = {pop['H']:.6f}, "
                  f"[l0, u0] = [{pop['l0']:.4f}, {pop['u0']:.4f}]",
                  f"First-order n Var: rotated {pop['V_rotated']:.2f}, fixed {pop['V_fixed']:.2f} "
                  f"(Var psi_U {pop['variances']['psi_U']:.4f}, psi_o {pop['variances']['psi_o']:.4f}, "
                  f"psi_T {pop['variances']['psi_T']:.4f}, psi_gamma {pop['variances']['psi_gamma']:.4f})",
                  "",
                  "| n | reps | nVar fixed | nVar avg | nVar pooled | fixed/avg | sqrt(n) bias avg | "
                  "sqrt(n) bias pooled | RMSE fixed | RMSE avg | RMSE pooled | edge (rot/fixed/pooled) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in sorted(block["cells"], key=int):
            c = block["cells"][n]
            b = c["boundary"]
            lines.append(
                f"| {int(n):,} | {c['reps']} | {c['fixed']['n_var']:.1f} | {c['average']['n_var']:.1f} | "
                f"{c['pooled']['n_var']:.1f} | {c['ratio_fixed_over_average']:.2f} | "
                f"{c['average']['sqrtN_bias']:.2f} ({c['average']['sqrtN_bias_se']:.2f}) | "
                f"{c['pooled']['sqrtN_bias']:.2f} ({c['pooled']['sqrtN_bias_se']:.2f}) | "
                f"{c['fixed']['rmse']:.4f} | {c['average']['rmse']:.4f} | {c['pooled']['rmse']:.4f} | "
                f"{b['rotation_thresholds_at_edge']}/{b['fixed_at_edge']}/{b['pooled_at_edge']} |")
        lines.append(f"| prediction | | {pop['V_fixed']:.1f} | {pop['V_rotated']:.2f} | "
                     f"{pop['V_rotated']:.2f} | 4 | 0 | 0 | | | | |")
        if block.get("rmse_slopes"):
            sl = block["rmse_slopes"]
            lines.append("")
            lines.append(f"log-log RMSE slope: fixed {sl['fixed']:.3f}, average {sl['average']:.3f}, "
                         f"pooled {sl['pooled']:.3f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_summary(out_dir: Path, pops: dict, meta: dict) -> dict:
    summary = {"meta": meta, "laws": {}}
    for law, pop in pops.items():
        cells = {}
        for path in sorted(out_dir.glob(f"rows_{law}_*.json")):
            rows = json.loads(path.read_text())
            n = int(rows[0]["n"])
            cells[str(n)] = summarize_cell(rows, n, pop)
        summary["laws"][law] = {"population": pop, "cells": cells,
                                "rmse_slopes": rmse_slopes(cells)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "summary.md").write_text(markdown(summary))
    return summary


def populations(laws, mc_draws: int) -> dict:
    pops = {}
    for law in laws:
        P = NonGaussianPopulation(law)
        dec = P.decomposition()
        old = population_truth(law)
        dec["old_target_quad_policy_pm3"] = old["phi_star"]
        dec["old_trim_bounds"] = [old["trim_lower"], old["trim_upper"]]
        if mc_draws:
            dec["mc_role_scores"] = P.mc_role_scores(mc_draws, seed=20261109)
        pops[law] = dec
    return pops


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--laws", nargs="+", default=list(LAWS), choices=LAWS)
    ap.add_argument("--n", type=int, nargs="+", default=[4000, 16000, 64000])
    ap.add_argument("--reps", type=int, nargs="+", default=[400, 200, 100])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--mc-draws", type=int, default=2_000_000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--population-only", action="store_true")
    ap.add_argument("--summarize", action="store_true",
                    help="rebuild summary.json/summary.md from the saved rows")
    args = ap.parse_args(argv)
    if len(args.reps) != len(args.n):
        raise SystemExit("--reps needs one value per --n")
    args.out.mkdir(parents=True, exist_ok=True)

    pop_path = args.out / "population.json"
    if args.summarize and pop_path.exists():
        pops = json.loads(pop_path.read_text())
    else:
        t0 = time.time()
        check = gaussian_port_check()
        print("[port check, Gaussian index: (Population, port)]", flush=True)
        for k, v in check.items():
            print(f"  {k}: {v}", flush=True)
        pops = populations(LAWS, args.mc_draws)      # both laws, so summaries keep every cell
        pops_out = {**pops, "gaussian_port_check": check}
        pop_path.write_text(json.dumps(pops_out, indent=1) + "\n")
        for law, p in pops.items():
            print(f"[{law}] phi*={p['phi_star']:.6f} (old quad {p['old_target_quad_policy_pm3']:.6f}) "
                  f"H={p['H']:.6f} [l0,u0]=[{p['l0']:.4f},{p['u0']:.4f}] "
                  f"Abar0={p['Abar_intercept']:.2e} V_rot={p['V_rotated']:.3f} "
                  f"V_fixed={p['V_fixed']:.2f} var={p['variances']}", flush=True)
            if "mc_role_scores" in p:
                mc = p["mc_role_scores"]
                print(f"   MC({mc['n_draw']}): var={mc['variance']} mean={mc['mean']} "
                      f"V_rot_mc={mc['V_rotated_mc']:.3f} (with cross terms "
                      f"{mc['V_rotated_mc_sum']:.3f})", flush=True)
        print(f"[population done in {time.time() - t0:.1f}s]", flush=True)
    pops = {k: v for k, v in pops.items() if k in LAWS}
    meta = {"script": "experiments/scripts/ds_four_role_nongaussian.py", "seed": args.seed,
            "n": args.n, "reps": args.reps, "workers": args.workers}
    if args.population_only:
        return
    if args.summarize:
        build_summary(args.out, pops, meta)
        print((args.out / "summary.md").read_text())
        return

    for i_n, (n, reps) in enumerate(zip(args.n, args.reps)):
        for i_law, law in enumerate(args.laws):
            t0 = time.time()
            base = args.seed + 10_000_000 * LAWS.index(law) + 100_000 * i_n
            tasks = [(law, n, base + r) for r in range(reps)]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(replicate, tasks, chunksize=1))
            (args.out / f"rows_{law}_{n}.json").write_text(json.dumps(rows))
            c = summarize_cell(rows, n, pops[law])
            print(f"[{law} n={n} reps={reps} {time.time() - t0:.0f}s] nVar fixed/avg/pooled = "
                  f"{c['fixed']['n_var']:.1f}/{c['average']['n_var']:.1f}/{c['pooled']['n_var']:.1f} "
                  f"(pred {pops[law]['V_fixed']:.1f}/{pops[law]['V_rotated']:.2f}) "
                  f"sqrt(n) bias avg {c['average']['sqrtN_bias']:.2f} ({c['average']['sqrtN_bias_se']:.2f}) "
                  f"edge {c['boundary']}", flush=True)
            build_summary(args.out, pops, meta)
    print((args.out / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
