"""Checks of the proposed simplifications of the differing-slopes proof.

The 2026-10-13 review of ``differing_slopes.tex`` proposed to shorten the proof
of Theorem 1 (research log, "Proof simplifications: checks").  This script
checks each proposal numerically, by quadrature with closed forms wherever
possible, so that no step relies on the sieve expansions being checked.

Designs.  ``paper`` is the displayed design of the paper's Section 5
(Gaussian X and eta).  ``mixx`` has a two-component Gaussian-mixture X in
two dimensions (not elliptical), a skewed two-component mixture eta, a
nonlinear a, and phi0 != 0.  ``mixx_raw`` is ``mixx`` with E X != 0.
``kink`` is the paper design with a and b in C^{2,1} but not C^3, used for
the bias rates.

Notation.  X~ = (1, X'), T = gamma' X~, h1(t) = E[X~ | T=t] f_T(t) = (g, rho_X')',
h2(t) = E[X~ X~' | T=t] f_T(t), w(e) = (a(e) - c, beta2')' and the merged
density representer r(t) = w(phi*-t) f_eta(phi*-t) 1{phi*-u0 <= t <= phi*-l0}.
The paper's A_g + A_rho is A_dens = -int dh2/dt r dt.

Checks (keys of the JSON output):
  validate      generic code against ds_local_efficiency.Population
  item1_mean    E[rho_hat | delta] = P_L h1_delta (Monte Carlo)
  item1_sieve   finite-L derivative of the density-block mean, full and
                zero-trace bases; second-order remainder against L
  item1_maps    gradients of the index-fold and evaluation-fold maps
  item1_var     variance of the centred sampling difference
  item3_ls      stacked = arm-wise least squares
  item3_riesz   Riesz identity of the closed-form outcome representer (MC)
  item3_sieve   sieve representer and loading convergence
  item4_shift   the four-role estimator under X -> X + c
  item4_center  Abar_X with raw and centred X
  item5_dens    density-block bias rate, full and zero-trace bases
  item5_out     outcome-block bias rate, L1/L2 projection errors, and the
                Lebesgue constant of the weighted arm-wise projection
  item_c_path   d/dt of U_eps'(phi*) under a shift in the law of X = E[psi_T S]

``--simulate`` runs the end-to-end four-role simulation with spline
dimensions outside the paper's range (item 6).

Run (from code/):
    python -m experiments.scripts.ds_simplification_checks
    python -m experiments.scripts.ds_simplification_checks --simulate --reps 400 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

from experiments.methods.perfrdd import _basis_params
from experiments.methods.spline_density import (
    _basis_info,
    evaluate_basis_zero_outside,
    lebesgue_gram,
)
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts import ds_local_efficiency_sim as les
from experiments.scripts.ds_local_efficiency import DESIGNS, Population, _gl_nodes

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_simplification_checks"


# ====================================================================== laws

@dataclass(frozen=True)
class MixX:
    """Finite mixture of Gaussians for X."""

    w: tuple
    mu: tuple
    cov: tuple

    def comps(self):
        for w, m, S in zip(self.w, self.mu, self.cov):
            yield float(w), np.asarray(m, float), np.asarray(S, float)

    @property
    def dim(self) -> int:
        return len(self.mu[0])

    def mean(self) -> np.ndarray:
        return sum(w * m for w, m, _ in self.comps())

    def second(self) -> np.ndarray:
        return sum(w * (S + np.outer(m, m)) for w, m, S in self.comps())

    def shifted(self, v) -> "MixX":
        v = np.asarray(v, float)
        return MixX(self.w, tuple(tuple(np.asarray(m, float) + v) for m in self.mu), self.cov)

    def sample(self, rng, n: int) -> np.ndarray:
        k = rng.choice(len(self.w), size=n, p=np.asarray(self.w, float))
        out = np.empty((n, self.dim))
        for c, (_, m, S) in enumerate(self.comps()):
            idx = np.flatnonzero(k == c)
            out[idx] = rng.multivariate_normal(m, S, size=len(idx))
        return out


@dataclass(frozen=True)
class EtaMix:
    """Finite mixture of normals for eta."""

    w: tuple
    m: tuple
    s: tuple

    def comps(self):
        for w, m, s in zip(self.w, self.m, self.s):
            yield float(w), float(m), float(s)

    def pdf(self, e):
        e = np.asarray(e, float)
        return sum(w * norm.pdf((e - m) / s) / s for w, m, s in self.comps())

    def dpdf(self, e):
        e = np.asarray(e, float)
        return sum(-w * (e - m) / s ** 2 * norm.pdf((e - m) / s) / s
                   for w, m, s in self.comps())

    def sample(self, rng, n: int) -> np.ndarray:
        k = rng.choice(len(self.w), size=n, p=np.asarray(self.w, float))
        m, s = np.asarray(self.m, float)[k], np.asarray(self.s, float)[k]
        return m + s * rng.standard_normal(n)

    def shifted_pdf(self, X: MixX, d0: float, dx: np.ndarray, e):
        """Density of eta - X~'delta at e, delta = (d0, dx), X independent of eta."""
        e = np.asarray(e, float)
        out = np.zeros_like(e)
        for wc, mc, Sc in X.comps():
            mshift = d0 + dx @ mc
            vshift = float(dx @ Sc @ dx)
            for wj, mj, sj in self.comps():
                sd = np.sqrt(sj ** 2 + vshift)
                out += wc * wj * norm.pdf((e - (mj - mshift)) / sd) / sd
        return out


# ============================================================ index functions

def _idx(m, S, a0, ax):
    mT = a0 + ax @ m
    kap = S @ ax
    sT = float(np.sqrt(ax @ S @ ax))
    return mT, sT, kap


def idx_f(X: MixX, a0, ax, t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for w, m, S in X.comps():
        mT, sT, _ = _idx(m, S, a0, ax)
        out = out + w * norm.pdf((t - mT) / sT) / sT
    return out


def idx_cdf(X: MixX, a0, ax, t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for w, m, S in X.comps():
        mT, sT, _ = _idx(m, S, a0, ax)
        out = out + w * norm.cdf((t - mT) / sT)
    return out


def idx_quantile(X: MixX, a0, ax, p: float) -> float:
    return brentq(lambda q: float(idx_cdf(X, a0, ax, q)) - p, -60.0, 60.0, xtol=1e-14)


def idx_h2(X: MixX, a0, ax, t, deriv: bool = False):
    """h2(t) = E[X~X~' | a'X~ = t] f(t), shape (n, 1+d, 1+d); optionally d/dt."""
    t = np.atleast_1d(np.asarray(t, float))
    d = X.dim
    H = np.zeros((len(t), d + 1, d + 1))
    dH = np.zeros_like(H)
    for w, m, S in X.comps():
        mT, sT, kap = _idx(m, S, a0, ax)
        z = (t - mT) / sT
        base = w * norm.pdf(z) / sT
        mu = m[None, :] + np.outer(z / sT, kap)
        C = S - np.outer(kap, kap) / sT ** 2
        M = np.empty((len(t), d + 1, d + 1))
        M[:, 0, 0] = 1.0
        M[:, 0, 1:] = mu
        M[:, 1:, 0] = mu
        M[:, 1:, 1:] = C[None] + mu[:, :, None] * mu[:, None, :]
        H += base[:, None, None] * M
        if deriv:
            dbase = -z * base / sT
            dmu = kap / sT ** 2
            dM = np.zeros_like(M)
            dM[:, 0, 1:] = dmu
            dM[:, 1:, 0] = dmu
            dM[:, 1:, 1:] = dmu[None, :, None] * mu[:, None, :] + mu[:, :, None] * dmu[None, None, :]
            dH += dbase[:, None, None] * M + base[:, None, None] * dM
    return (H, dH) if deriv else H


def idx_h1(X: MixX, a0, ax, t):
    """h1(t) = E[X~ | a'X~ = t] f(t) = (g, rho_X')', shape (n, 1+d)."""
    return idx_h2(X, a0, ax, t)[:, :, 0]


def idx_dh1(X: MixX, a0, ax, t):
    return idx_h2(X, a0, ax, t, deriv=True)[1][:, :, 0]


def idx_upper(X: MixX, a0, ax, s, second: bool = True):
    """P(T > s), E[X 1{T>s}] and (optionally) E[XX' 1{T>s}] at points s."""
    s = np.atleast_1d(np.asarray(s, float))
    d = X.dim
    P = np.zeros(len(s))
    E1 = np.zeros((len(s), d))
    E2 = np.zeros((len(s), d, d)) if second else None
    for w, m, S in X.comps():
        mT, sT, kap = _idx(m, S, a0, ax)
        z = (s - mT) / sT
        sf, pdf = norm.sf(z), norm.pdf(z)
        P += w * sf
        E1 += w * (sf[:, None] * m[None, :] + (pdf / sT)[:, None] * kap[None, :])
        if second:
            C = S - np.outer(kap, kap) / sT ** 2
            E2 += w * (sf[:, None, None] * (C + np.outer(m, m))[None]
                       + (pdf / sT)[:, None, None] * (np.outer(m, kap) + np.outer(kap, m))[None]
                       + ((z * pdf + sf) / sT ** 2)[:, None, None] * np.outer(kap, kap)[None])
    return P, E1, E2


# ================================================================= design

@dataclass(frozen=True)
class GDesign:
    name: str
    X: MixX
    eta: EtaMix
    gamma0: float
    gamma_x: tuple
    a: Callable
    da: Callable
    b: Callable
    db: Callable
    beta2: tuple
    cost: float
    eps: float = 0.10
    phi0: float = 0.0
    J: tuple = (-1.75, 1.75)
    Tsup: tuple = (-3.0, 3.0)
    policy: tuple = (-1.5, 1.5)


def _poly(c):
    p = np.polynomial.Polynomial(c)
    return p, p.deriv()


_pa, _pda = _poly((2.0, 1.0))
_pb, _pdb = _poly((0.0, 0.0, 0.5))
PAPER = GDesign(
    name="paper",
    X=MixX((1.0,), ((0.0, 0.0, 0.0),), (tuple(map(tuple, np.eye(3))),)),
    eta=EtaMix((1.0,), (0.0,), (1.0,)),
    gamma0=0.0, gamma_x=tuple(np.ones(3) / np.sqrt(3.0)),
    a=_pa, da=_pda, b=_pb, db=_pdb,
    beta2=(0.4, -0.25, 0.15), cost=2.25,
)

_MIX_X = MixX(
    (0.6, 0.4),
    ((-0.6, 0.3), (0.9, -0.45)),
    (((0.5, 0.2), (0.2, 0.8)), ((1.2, -0.3), (-0.3, 0.6))),
)
_MIX_ETA = EtaMix((0.7, 0.3), (-0.3, 0.7), (0.8, 0.6))
MIXX = GDesign(
    name="mixx", X=_MIX_X, eta=_MIX_ETA,
    gamma0=0.2, gamma_x=(0.8, 0.5),
    a=lambda e: 0.3 + 0.7 * np.asarray(e) + 0.15 * np.sin(2 * np.asarray(e)),
    da=lambda e: 0.7 + 0.3 * np.cos(2 * np.asarray(e)),
    b=lambda e: 0.2 + 0.4 * np.asarray(e) + 0.3 * np.asarray(e) ** 2,
    db=lambda e: 0.4 + 0.6 * np.asarray(e),
    beta2=(0.5, -0.3), cost=0.45, phi0=0.2,
    J=(-2.2, 2.2), Tsup=(-3.5, 3.5), policy=(-1.5, 1.5),
)

# mixx with E X = mu != 0.  The structural model is fixed in the raw X:
# W = a_raw(eta) + X'beta2 with a_raw = a_mixx - mu'beta2, so the centred
# version of this design is exactly mixx.
_MU_SHIFT = np.array([0.8, -0.5])
_MIX_RAW = _MIX_X.shifted(_MU_SHIFT)
MIXX_RAW = GDesign(
    name="mixx_raw", X=_MIX_RAW, eta=_MIX_ETA,
    gamma0=0.2 - float(np.array([0.8, 0.5]) @ _MU_SHIFT), gamma_x=(0.8, 0.5),
    a=lambda e: MIXX.a(e) - float(_MU_SHIFT @ np.array(MIXX.beta2)),
    da=MIXX.da,
    b=lambda e: MIXX.b(e) - float(_MU_SHIFT @ np.array([0.1, 0.2])),
    db=MIXX.db,
    beta2=MIXX.beta2, cost=MIXX.cost, phi0=MIXX.phi0,
    J=MIXX.J, Tsup=MIXX.Tsup, policy=MIXX.policy,
)

# Paper design with a, b in C^{2,1}: third derivatives jump at one point, so
# cubic splines approximate them at rate K^-3, as the theorem assumes.  The
# jumps sit at the trim points u0 and l0, where the representer jumps: the
# projection error of r is localized there, so this is the least favourable
# placement for the bias (a kink away from the trim points gives a much
# smaller bias).
_U0 = float(norm.ppf(0.9))
_K1, _K2 = _U0, -_U0
KINK = GDesign(
    name="kink", X=PAPER.X, eta=PAPER.eta, gamma0=0.0, gamma_x=PAPER.gamma_x,
    a=lambda e: 2.0 + np.asarray(e) + 0.2 * np.abs(np.asarray(e) - _K2) ** 3,
    da=lambda e: 1.0 + 0.6 * np.sign(np.asarray(e) - _K2) * (np.asarray(e) - _K2) ** 2,
    b=lambda e: 0.5 * np.asarray(e) ** 2 + 0.25 * np.abs(np.asarray(e) - _K1) ** 3,
    db=lambda e: np.asarray(e) + 0.75 * np.sign(np.asarray(e) - _K1) * (np.asarray(e) - _K1) ** 2,
    beta2=PAPER.beta2, cost=2.25,
)


# ============================================================== population

class Pop:
    """Population objects of the four-role theorem for a GDesign."""

    def __init__(self, d: GDesign):
        self.d = d
        self.X = d.X
        self.g0 = float(d.gamma0)
        self.gx = np.asarray(d.gamma_x, float)
        self.b2 = np.asarray(d.beta2, float)
        self.dim = self.X.dim
        self.qlo = idx_quantile(self.X, self.g0, self.gx, d.eps)
        self.qhi = idx_quantile(self.X, self.g0, self.gx, 1.0 - d.eps)
        self.l0 = d.phi0 - self.qhi
        self.u0 = d.phi0 - self.qlo
        if not (d.J[0] < self.l0 < self.u0 < d.J[1]):
            raise ValueError("trim window must lie inside J")
        self.phi, self.H = self._target()

    # -- basic pieces ------------------------------------------------------
    def f_eta(self, e):
        return self.d.eta.pdf(e)

    def w_vec(self, e):
        e = np.atleast_1d(np.asarray(e, float))
        return np.column_stack((self.d.a(e) - self.d.cost, np.tile(self.b2, (len(e), 1))))

    def h1(self, t, delta=None):
        a0, ax = self._coef(delta)
        return idx_h1(self.X, a0, ax, t)

    def _coef(self, delta):
        if delta is None:
            return self.g0, self.gx
        delta = np.asarray(delta, float)
        return self.g0 + delta[0], self.gx + delta[1:]

    def _win(self, lo=None, hi=None, order=48, pieces=12):
        lo = self.l0 if lo is None else lo
        hi = self.u0 if hi is None else hi
        return _gl_nodes([lo, hi], order=order, pieces=pieces)

    def Z(self, phi):
        e, w = self._win()
        F = np.sum(self.w_vec(e) * self.h1(phi - e), axis=1)
        return float(np.sum(w * F * self.f_eta(e)))

    def dZ(self, phi):
        e, w = self._win()
        dF = np.sum(self.w_vec(e) * idx_dh1(self.X, self.g0, self.gx, phi - e), axis=1)
        return float(np.sum(w * dF * self.f_eta(e)))

    def _target(self):
        lo, hi = self.d.policy
        grid = np.linspace(lo, hi, 601)
        vals = np.array([self.Z(x) for x in grid])
        roots = [brentq(self.Z, grid[i], grid[i + 1], xtol=1e-14)
                 for i in range(len(grid) - 1) if vals[i] < 0.0 <= vals[i + 1]]
        if len(roots) != 1:
            raise ValueError(f"{self.d.name}: expected one interior maximizer, found {roots}")
        phi = float(roots[0])
        return phi, -self.dZ(phi)

    def F(self, e):
        return np.sum(self.w_vec(e) * self.h1(self.phi - np.atleast_1d(e)), axis=1)

    def s_star(self, e):
        e = np.atleast_1d(np.asarray(e, float))
        dh1 = idx_dh1(self.X, self.g0, self.gx, self.phi - e)
        return self.d.da(e) * self.h1(self.phi - e)[:, 0] - np.sum(self.w_vec(e) * dh1, axis=1)

    @property
    def b_l(self):
        return float(self.F(self.l0)[0] * self.f_eta(self.l0))

    @property
    def b_u(self):
        return float(self.F(self.u0)[0] * self.f_eta(self.u0))

    def r(self, t):
        """Merged density representer, shape (n, 1+d)."""
        t = np.atleast_1d(np.asarray(t, float))
        e = self.phi - t
        inside = (e >= self.l0) & (e <= self.u0)
        return np.where(inside[:, None], self.w_vec(e) * self.f_eta(e)[:, None], 0.0)

    @property
    def r_support(self):
        return self.phi - self.u0, self.phi - self.l0

    def m_p(self, q):
        h = self.h1(q)[0]
        return h / h[0]

    # -- loadings ------------------------------------------------------------
    def A_U(self):
        e, w = self._win()
        core = float(np.sum(w * self.s_star(e) * self.f_eta(e))) + self.b_l - self.b_u
        return np.concatenate(([1.0], self.X.mean())) * core

    def A_dens(self):
        lo, hi = self.r_support
        t, w = _gl_nodes([lo, hi])
        _, dH = idx_h2(self.X, self.g0, self.gx, t, deriv=True)
        return -np.einsum("i,ijk,ik->j", w, dH, self.r(t))

    def index_part(self):
        """Index-fold first-stage loading: -A_dens - b_l m_{1-eps} + b_u m_eps."""
        return (-self.A_dens() - self.b_l * self.m_p(self.qhi)
                + self.b_u * self.m_p(self.qlo))

    def B(self):
        e, w = self._win()
        return np.sum((w * self.f_eta(e))[:, None] * self.h1(self.phi - e)[:, 1:], axis=0)

    def arm_moments(self, e, second=True):
        """pi_d, m_d, V_d at eta = e for d = 0, 1."""
        e = np.atleast_1d(np.asarray(e, float))
        P1, E1, S1 = idx_upper(self.X, self.g0, self.gx, self.d.phi0 - e, second)
        P0 = 1.0 - P1
        E0 = self.X.mean()[None, :] - E1
        out = {1: {"pi": P1, "m": E1 / P1[:, None]}, 0: {"pi": P0, "m": E0 / P0[:, None]}}
        if second:
            S0 = self.X.second()[None] - S1
            for arm, S in ((1, S1), (0, S0)):
                m = out[arm]["m"]
                out[arm]["V"] = S / out[arm]["pi"][:, None, None] - m[:, :, None] * m[:, None, :]
        return out

    def _J_nodes(self):
        J0, J1 = self.d.J
        e, w = _gl_nodes([J0, self.l0, self.u0, J1])
        return e, w * self.f_eta(e)

    def gI0(self, e):
        e = np.asarray(e, float)
        inside = (e >= self.l0) & (e <= self.u0)
        return np.where(inside, idx_f(self.X, self.g0, self.gx, self.phi - e), 0.0)

    def c_prime(self, arm, e):
        return self.d.db(e) + (self.d.da(e) if arm == 1 else 0.0)

    def outcome(self):
        """Closed-form arm-wise representer coefficients and A_out."""
        e, wJ = self._J_nodes()
        mom = self.arm_moments(e)
        gI = self.gI0(e)
        B = self.B()
        omega, M = {}, {}
        A0 = float(np.sum(wJ * gI * self.d.da(e)))
        AX = np.zeros(self.dim)
        for arm, s in ((0, -1.0), (1, 1.0)):
            pi, m, V = mom[arm]["pi"], mom[arm]["m"], mom[arm]["V"]
            M[arm] = np.einsum("i,ijk->jk", wJ * pi, V)
            rhs = B - np.sum((wJ * gI)[:, None] * m, axis=0)
            omega[arm] = s * np.linalg.solve(M[arm], rhs)
            dc = self.c_prime(arm, e)
            AX += (s * np.sum((wJ * dc * gI)[:, None] * m, axis=0)
                   + np.einsum("i,ijk->jk", wJ * pi * dc, V) @ omega[arm])
        return {"omega": omega, "M": M, "B": B, "A_out": np.concatenate(([A0], AX))}

    def Abar(self):
        out = self.outcome()
        return self.A_U() - out["A_out"] + self.index_part()

    def r_out(self, D, eta, X, omega):
        """Closed-form outcome representer at observations."""
        J0, J1 = self.d.J
        S = (eta >= J0) & (eta <= J1)
        mom = self.arm_moments(eta, second=False)
        gI = self.gI0(eta)
        out = np.zeros(len(eta))
        for arm, s in ((0, -1.0), (1, 1.0)):
            sel = S & (D == arm)
            pi, m = mom[arm]["pi"][sel], mom[arm]["m"][sel]
            out[sel] = s * gI[sel] / pi + np.sum((X[sel] - m) * omega[arm], axis=1)
        return out

    # -- population maps -----------------------------------------------------
    def M_T(self, delta):
        """Index-fold map: U_eps'(phi*) with g, rho_X and quantiles of the shifted index."""
        a0, ax = self._coef(delta)
        qlo = idx_quantile(self.X, a0, ax, self.d.eps)
        qhi = idx_quantile(self.X, a0, ax, 1.0 - self.d.eps)
        lo, hi = self.d.phi0 - qhi, self.d.phi0 - qlo
        e, w = self._win(lo, hi)
        F = np.sum(self.w_vec(e) * idx_h1(self.X, a0, ax, self.phi - e), axis=1)
        return -float(np.sum(w * F * self.f_eta(e)))

    def M_U(self, delta):
        """Evaluation-fold map: -E[F(eta_hat) 1{l0 <= eta_hat <= u0}], eta_hat = eta - X~'delta."""
        delta = np.asarray(delta, float)
        e, w = self._win()
        dens = self.d.eta.shifted_pdf(self.X, delta[0], delta[1:], e)
        return -float(np.sum(w * self.F(e) * dens))


def fd_grad(fun, dim: int, h: float = 1e-5) -> np.ndarray:
    g = np.zeros(dim)
    for j in range(dim):
        e = np.zeros(dim)
        e[j] = h
        g[j] = (fun(e) - fun(-e)) / (2 * h)
    return g


# ============================================================ density sieve

class DensityBasis:
    """Cubic B-spline basis on the density interval, full clamped or zero trace."""

    def __init__(self, L: int, support, zero_trace: bool = False):
        self.info = _basis_info(int(L), tuple(support))
        self.L = int(L)
        self.lo, self.hi = map(float, support)
        self.keep = np.arange(1, L - 1) if zero_trace else np.arange(L)
        gram = lebesgue_gram(self.info)
        self.G = gram[np.ix_(self.keep, self.keep)]
        self.knots = np.unique(self.info["t"])

    def design(self, t):
        return evaluate_basis_zero_outside(np.asarray(t, float), self.info)[:, self.keep]

    def nodes(self, extra=(), order=20, pieces=2):
        br = np.concatenate((self.knots, [x for x in extra if self.lo < x < self.hi]))
        return _gl_nodes(br, order=order, pieces=pieces)

    def project(self, fun, extra=()):
        t, w = self.nodes(extra)
        vals = np.atleast_2d(fun(t))
        if vals.shape[0] != len(t):
            vals = vals.T
        return np.linalg.solve(self.G, self.design(t).T @ (w[:, None] * vals))

    def spline(self, coef):
        full = np.zeros((self.L,) + coef.shape[1:])
        full[self.keep] = coef
        return BSpline(self.info["t"], full, 3, extrapolate=False)

    def evaluate(self, coef, t, nu=0):
        t = np.asarray(t, float)
        sp = self.spline(coef)
        if nu:
            sp = sp.derivative(nu)
        out = np.zeros((len(t),) + coef.shape[1:])
        inside = (t >= self.lo) & (t <= self.hi)
        out[inside] = sp(t[inside])
        return out


# ================================================================== item 1

def check_item1_mean(pop: Pop, L=12, n=2000, reps=2000, seed=11) -> dict:
    """E[Omega_hat | delta] equals the projection of h1 of the shifted index."""
    rng = np.random.default_rng(seed)
    basis = DensityBasis(L, pop.d.Tsup)
    delta = np.array([0.03, 0.02, -0.015, 0.025][: pop.dim + 1])
    a0, ax = pop._coef(delta)
    exact = basis.project(lambda t: idx_h1(pop.X, a0, ax, t))
    acc = np.zeros_like(exact)
    acc2 = np.zeros_like(exact)
    for _ in range(reps):
        X = pop.X.sample(rng, n)
        Xt = np.column_stack((np.ones(n), X))
        That = Xt @ np.concatenate(([a0], ax))
        est = np.linalg.solve(basis.G, basis.design(That).T @ Xt / n)
        acc += est
        acc2 += est ** 2
    mean = acc / reps
    se = np.sqrt(np.maximum(acc2 / reps - mean ** 2, 1e-300) / reps)
    z = (mean - exact) / se
    return {"L": L, "n": n, "reps": reps, "max_abs_z": float(np.max(np.abs(z))),
            "mean_abs_z": float(np.mean(np.abs(z)))}


def check_item1_sieve(pop: Pop, Ls=(8, 12, 16, 24, 32, 48, 64, 96), seed=3) -> dict:
    """Finite-L derivative of Theta_L(delta) = -int r_L' h1_delta, full and zero-trace bases."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((6, pop.dim + 1))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rho_step = 0.05
    limit = -pop.A_dens()                    # = int dh2 r, the L -> infinity gradient
    out = {"limit_grad": limit.tolist(), "cases": []}
    A, Bnd = pop.d.Tsup
    for zero_trace in (False, True):
        for L in Ls:
            basis = DensityBasis(L, pop.d.Tsup, zero_trace=zero_trace)
            coef = basis.project(pop.r, extra=pop.r_support)
            t, w = basis.nodes(order=20, pieces=2)
            rL = basis.evaluate(coef, t)
            drL = basis.evaluate(coef, t, nu=1)
            H, dH = idx_h2(pop.X, pop.g0, pop.gx, t, deriv=True)

            def theta(delta):
                a0, ax = pop._coef(delta)
                return -float(np.sum(w[:, None] * rL * idx_h1(pop.X, a0, ax, t)))

            g_fd = fd_grad(theta, pop.dim + 1)
            g_pop = np.einsum("i,ijk,ik->j", w, dH, rL)
            g_taylor = -np.einsum("i,ijk,ik->j", w, H, drL)
            rA = basis.evaluate(coef, np.array([A]))[0]
            rB = basis.evaluate(coef, np.array([Bnd - 1e-12]))[0]
            HA = idx_h2(pop.X, pop.g0, pop.gx, np.array([A]))[0]
            HB = idx_h2(pop.X, pop.g0, pop.gx, np.array([Bnd]))[0]
            boundary = HB @ rB - HA @ rA
            th0 = theta(np.zeros(pop.dim + 1))
            rem = max(abs(theta(rho_step * u) - th0 - rho_step * g_fd @ u) / rho_step ** 2
                      for u in dirs)
            out["cases"].append({
                "basis": "zero_trace" if zero_trace else "full", "L": L,
                "fd_vs_population": float(np.max(np.abs(g_fd - g_pop))),
                "fd_vs_taylor": float(np.max(np.abs(g_fd - g_taylor))),
                "boundary_term": float(np.max(np.abs(boundary))),
                # g_pop - g_taylor = boundary is integration by parts on the
                # quadrature; the substantive check is fd_vs_population
                "fd_minus_taylor_minus_boundary": float(np.max(np.abs(g_fd - g_taylor - boundary))),
                "dist_to_limit": float(np.max(np.abs(g_fd - limit))),
                "second_order_ratio": float(rem),
                "r_L_at_ends": float(max(np.max(np.abs(rA)), np.max(np.abs(rB)))),
            })
    return out


def check_item1_maps(pop: Pop) -> dict:
    gT_fd = fd_grad(pop.M_T, pop.dim + 1)
    gT = pop.index_part()
    gU_fd = fd_grad(pop.M_U, pop.dim + 1)
    gU = pop.A_U()
    return {
        "index_map_fd": gT_fd.tolist(), "index_map_formula": gT.tolist(),
        "index_map_max_diff": float(np.max(np.abs(gT_fd - gT))),
        "eval_map_fd": gU_fd.tolist(), "eval_map_formula": gU.tolist(),
        "eval_map_max_diff": float(np.max(np.abs(gU_fd - gU))),
        "E_X": pop.X.mean().tolist(),
        "A_U_covariate_block": gU[1:].tolist(),
    }


def check_item1_var(pop: Pop, Ls=(8, 16, 32, 64, 128), dnorms=(0.003, 0.01, 0.03),
                    n=400_000, seed=5) -> dict:
    """V_L(delta) = E[{(r_L(T + X~'delta) - r_L(T))'X~}^2]."""
    rng = np.random.default_rng(seed)
    X = pop.X.sample(rng, n)
    Xt = np.column_stack((np.ones(n), X))
    T = Xt @ np.concatenate(([pop.g0], pop.gx))
    u = rng.standard_normal(pop.dim + 1)
    u /= np.linalg.norm(u)
    rows = []
    for L in Ls:
        basis = DensityBasis(L, pop.d.Tsup)
        coef = basis.project(pop.r, extra=pop.r_support)
        base = np.sum(basis.evaluate(coef, T) * Xt, axis=1)
        sup_d = float(np.max(np.abs(basis.evaluate(coef, np.linspace(*pop.d.Tsup, 20001), nu=1))))
        for dn in dnorms:
            shifted = np.sum(basis.evaluate(coef, T + Xt @ (dn * u)) * Xt, axis=1)
            V = float(np.mean((shifted - base) ** 2))
            rows.append({"L": L, "delta_norm": dn, "V": V, "V_over_delta": V / dn,
                         "V_over_L_delta2": V / (L * dn ** 2), "sup_abs_rL_prime": sup_d})
    return {"n_mc": n, "rows": rows}


# ================================================================== item 3

def check_item3_ls(seed=21) -> dict:
    design = DESIGNS["paper_baseline"]
    s = les.generate(design, 20000, seed)
    g = les.ols_first_stage(s, np.arange(5000))
    eta = s.Q - les._xt(s.X) @ g
    idx = np.arange(5000, 20000)
    keep = idx[(eta[idx] >= design.J[0]) & (eta[idx] <= design.J[1])]
    info = _basis_params(6, design.J)
    e, X, D, Y = eta[keep], s.X[keep], s.D[keep], s.Y[keep]
    Bm = BSpline.design_matrix(e, info["t"], 3).toarray()
    stacked = np.column_stack((D[:, None] * Bm, Bm, X, D[:, None] * X))
    coef_s = np.linalg.lstsq(stacked, Y, rcond=None)[0]
    fit_s = stacked @ coef_s
    armwise, nb = les._design_matrix(e, X, D, info)
    coef_a = np.linalg.lstsq(armwise, Y, rcond=None)[0]
    fit_a = armwise @ coef_a
    dim = X.shape[1]
    w0, d0 = coef_a[:nb], coef_a[nb:nb + dim]
    w1, d1 = coef_a[nb + dim:2 * nb + dim], coef_a[2 * nb + dim:]
    return {
        "max_fitted_diff": float(np.max(np.abs(fit_s - fit_a))),
        "a_coef_diff": float(np.max(np.abs(coef_s[:nb] - (w1 - w0)))),
        "beta2_diff": float(np.max(np.abs(coef_s[2 * nb + dim:] - (d1 - d0)))),
        "stacked_rank": int(np.linalg.matrix_rank(stacked)), "stacked_cols": stacked.shape[1],
    }


def check_item3_riesz(pop: Pop, n=3_000_000, n_dirs=12, seed=31) -> dict:
    """<r_out, h>_J = L_out(h) by Monte Carlo for random directions h."""
    rng = np.random.default_rng(seed)
    X = pop.X.sample(rng, n)
    eta = pop.d.eta.sample(rng, n)
    T = pop.g0 + X @ pop.gx
    D = (T + eta > pop.d.phi0).astype(float)
    out = pop.outcome()
    r = pop.r_out(D, eta, X, out["omega"])
    e_w, w_w = pop._win()
    gw = w_w * pop.f_eta(e_w) * pop.h1(pop.phi - e_w)[:, 0]
    J0, J1 = pop.d.J
    rows = []

    def direction(k):
        if k == 0:
            return "X_1", lambda e: 0 * e, lambda e: 0 * e, np.eye(pop.dim)[0], np.zeros(pop.dim)
        if k == 1:
            return "D X_1", lambda e: 0 * e, lambda e: 0 * e, np.zeros(pop.dim), np.eye(pop.dim)[0]
        if k == 2:
            return "D sin", lambda e: np.sin(1.3 * e), lambda e: 0 * e, np.zeros(pop.dim), np.zeros(pop.dim)
        if k == 3:
            return "cos", lambda e: 0 * e, lambda e: np.cos(0.9 * e), np.zeros(pop.dim), np.zeros(pop.dim)
        p = rng.standard_normal(4)
        q = rng.standard_normal(4)
        fa = lambda e, p=p: p[0] + p[1] * e + p[2] * e ** 2 + p[3] * np.sin(2 * e)
        fb = lambda e, q=q: q[0] + q[1] * e + q[2] * e ** 3 + q[3] * np.cos(e)
        return f"random {k - 3}", fa, fb, rng.standard_normal(pop.dim), rng.standard_normal(pop.dim)

    for k in range(4 + n_dirs):
        name, a1, b1, c1, c2 = direction(k)
        h = D * a1(eta) + b1(eta) + X @ c1 + D * (X @ c2)
        prod = r * h
        mc = float(np.mean(prod))
        se = float(np.std(prod) / np.sqrt(n))
        exact = float(np.sum(gw * a1(e_w)) + out["B"] @ c2)
        rows.append({"direction": name, "mc": mc, "exact": exact, "se": se,
                     "z": (mc - exact) / se if se > 0 else 0.0})
    support = (eta >= J0) & (eta <= J1)
    return {"n_mc": n, "rows": rows, "max_abs_z": float(max(abs(r_["z"]) for r_ in rows)),
            "r_zero_off_J": bool(np.all(r[~support] == 0.0))}


def _arm_sieve(pop: Pop, K: int, e, wJ, mom):
    """Population sieve representer on each arm (spline in eta plus X'k)."""
    info = _basis_info(K, pop.d.J)
    Bm = evaluate_basis_zero_outside(e, info)
    gI = pop.gI0(e)
    B = pop.B()
    res = {}
    for arm, s in ((0, -1.0), (1, 1.0)):
        pi, m, V = mom[arm]["pi"], mom[arm]["m"], mom[arm]["V"]
        ww = wJ * pi
        G = np.block([[Bm.T @ (ww[:, None] * Bm), Bm.T @ (ww[:, None] * m)],
                      [m.T @ (ww[:, None] * Bm),
                       np.einsum("i,ijk->jk", ww, V + m[:, :, None] * m[:, None, :])]])
        ell = s * np.concatenate((Bm.T @ (wJ * gI), B))
        v = np.linalg.solve(G, ell)
        res[arm] = {"G": G, "v": v, "Bm": Bm, "nb": Bm.shape[1]}
    return res


def check_item3_sieve(pop: Pop, Ks=(6, 8, 12, 16, 24, 32, 48, 64)) -> dict:
    e, wJ = pop._J_nodes()
    mom = pop.arm_moments(e)
    out = pop.outcome()
    gI = pop.gI0(e)
    rows = []
    for K in Ks:
        sv = _arm_sieve(pop, K, e, wJ, mom)
        dist2 = 0.0
        A = np.zeros(pop.dim + 1)
        for arm, s in ((0, -1.0), (1, 1.0)):
            pi, m, V = mom[arm]["pi"], mom[arm]["m"], mom[arm]["V"]
            nb, v, Bm = sv[arm]["nb"], sv[arm]["v"], sv[arm]["Bm"]
            vw, vk = v[:nb], v[nb:]
            om = out["omega"][arm]
            fun_K = Bm @ vw                       # function part of r_K on this arm
            fun = s * gI / pi - m @ om            # function part of r
            u = fun_K - fun
            Dk = vk - om
            dist2 += float(np.sum(wJ * pi * ((u + m @ Dk) ** 2 + np.einsum("j,ijk,k->i", Dk, V, Dk))))
            dc = pop.c_prime(arm, e)
            mean_rK = fun_K + m @ vk
            A[0] += float(np.sum(wJ * pi * dc * mean_rK))
            A[1:] += np.sum((wJ * pi * dc)[:, None] * (m * fun_K[:, None]
                                                      + np.einsum("ijk,k->ij", V + m[:, :, None] * m[:, None, :], vk)),
                            axis=0)
        rows.append({"K": K, "dist_rK_r": float(np.sqrt(dist2)),
                     "A_out_K_minus_A_out": float(np.max(np.abs(A - out["A_out"])))})
    Ks_arr = np.array([r_["K"] for r_ in rows], float)
    slope = float(np.polyfit(np.log(Ks_arr[2:]), np.log([r_["dist_rK_r"] for r_ in rows[2:]]), 1)[0])
    return {"rows": rows, "slope_dist": slope, "A_out": out["A_out"].tolist()}


# ================================================================== item 4

def _rotation(s, design, folds, gamma_hat, Lg=None, Lr=None, knots_exp=11 / 60,
              grid=None):
    """One rotation of the four-role estimator; returns the criterion on a grid and phi_hat."""
    J, eps, phi0 = design.J, design.eps, design.phi0
    eta_hat = s.Q - les._xt(s.X) @ gamma_hat
    idx = folds["outcome"]
    e = eta_hat[idx]
    fit_idx = idx[(e >= J[0]) & (e <= J[1])]
    n1 = int(s.D[fit_idx].sum())
    info = _basis_params(max(3, int(round(min(n1, len(fit_idx) - n1) ** knots_exp))), J)
    Z, nb = les._design_matrix(eta_hat[fit_idx], s.X[fit_idx], s.D[fit_idx], info)
    coef = les._ls(Z, s.Y[fit_idx])
    dim = s.X.shape[1]
    w0, d0 = coef[:nb], coef[nb:nb + dim]
    w1, d1 = coef[nb + dim:2 * nb + dim], coef[2 * nb + dim:]
    a_spl = BSpline(info["t"], w1 - w0, 3, extrapolate=False)
    b2 = d1 - d0
    T_idx = les._xt(s.X[folds["index"]]) @ gamma_hat
    tails_g = fit_weighted_tails(T_idx, s.X[folds["index"]], design.T_support, n_basis=Lg)
    tails_r = tails_g if Lr == Lg else fit_weighted_tails(
        T_idx, s.X[folds["index"]], design.T_support, n_basis=Lr)
    l_hat = phi0 - float(np.quantile(T_idx, 1.0 - eps))
    u_hat = phi0 - float(np.quantile(T_idx, eps))
    e = eta_hat[folds["utility"]]
    e = e[(e >= l_hat) & (e <= u_hat)]
    amc = a_spl(np.clip(e, info["lo"], info["hi"])) - design.cost

    def value(phi):
        t = phi - e
        return float(np.mean(amc * tails_g.survival(t) + tails_r.weighted_tail(t, b2)))

    lo, hi = design.policy
    gvals = np.array([value(x) for x in grid])
    k = int(np.argmax(gvals))
    res = minimize_scalar(lambda x: -value(x), bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                          method="bounded", options={"xatol": 1e-10})
    return gvals, float(res.x)


def check_item4_shift(n=16000, seed=41, shift=(1.3, -0.7, 2.1)) -> dict:
    design = DESIGNS["paper_baseline"]
    s = les.generate(design, n, seed)
    s_shift = les.Sample(X=s.X + np.asarray(shift), Q=s.Q, D=s.D, Y=s.Y)
    rng = np.random.default_rng(seed + 1)
    blocks = np.array_split(rng.permutation(n), 4)
    grid = np.linspace(*design.policy, 61)
    pop = Population(design).decomposition()
    sd_fixed = float(np.sqrt(pop["V_fixed_ols"] / n))
    rows = []
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(les.ROLES)}
        g_raw = les.ols_first_stage(s, folds["gamma"])
        g_sh = les.ols_first_stage(s_shift, folds["gamma"])
        for label, Lg, Lr in (("shared basis", 10, 10), ("separate bases", 9, 12)):
            v_raw, p_raw = _rotation(s, design, folds, g_raw, Lg, Lr, grid=grid)
            v_sh, p_sh = _rotation(s_shift, design, folds, g_sh, Lg, Lr, grid=grid)
            rows.append({"rotation": k, "case": label,
                         "max_criterion_diff": float(np.max(np.abs(v_raw - v_sh))),
                         "phi_diff": abs(p_raw - p_sh),
                         "phi_diff_over_sd": abs(p_raw - p_sh) / sd_fixed})
    return {"n": n, "shift": list(shift), "sd_fixed_assignment": sd_fixed, "rows": rows}


def check_item4_center() -> dict:
    raw, cen = Pop(MIXX_RAW), Pop(MIXX)
    A_raw, A_cen = raw.Abar(), cen.Abar()
    return {
        "phi_raw": raw.phi, "phi_centered": cen.phi,
        "Abar_raw": A_raw.tolist(), "Abar_centered": A_cen.tolist(),
        "Abar_X_max_diff": float(np.max(np.abs(A_raw[1:] - A_cen[1:]))),
        "A_U_X_raw": raw.A_U()[1:].tolist(), "A_U_X_centered": cen.A_U()[1:].tolist(),
        "A_dens_X_raw": raw.A_dens()[1:].tolist(), "A_dens_X_centered": cen.A_dens()[1:].tolist(),
        "A_out_X_raw": raw.outcome()["A_out"][1:].tolist(),
        "A_out_X_centered": cen.outcome()["A_out"][1:].tolist(),
    }


# ================================================================== item 5

def _slope(xs, ys, lo=16):
    """Log-log slope of |ys| on xs >= lo (the pre-asymptotic range is excluded).

    The biases change sign, so slopes are indicative only; the tables carry the
    values themselves.
    """
    xs, ys = np.asarray(xs, float), np.abs(np.asarray(ys, float))
    keep = xs >= lo
    return float(np.polyfit(np.log(xs[keep]), np.log(ys[keep]), 1)[0])


def check_item5_dens(pop: Pop, Ls=(8, 12, 16, 24, 32, 48, 64, 96, 128)) -> dict:
    out = {}
    for zero_trace in (False, True):
        rows = []
        for L in Ls:
            basis = DensityBasis(L, pop.d.Tsup, zero_trace=zero_trace)
            t, w = basis.nodes(extra=pop.r_support, order=24, pieces=3)
            h = pop.h1(t)
            hL = basis.evaluate(basis.project(pop.h1, extra=pop.r_support), t)
            r = pop.r(t)
            rL = basis.evaluate(basis.project(pop.r, extra=pop.r_support), t)
            bias = float(np.sum(w[:, None] * r * (hL - h)))
            rows.append({"L": L, "bias": bias,
                         "r_proj_L1": float(np.sum(w[:, None] * np.abs(rL - r))),
                         "r_proj_L2": float(np.sqrt(np.sum(w[:, None] * (rL - r) ** 2))),
                         "h_proj_sup_interior": float(np.max(np.abs((hL - h)[(t > pop.r_support[0]) & (t < pop.r_support[1])])))})
            # least favourable C^{2,1} nuisance: third derivative jumps where r jumps
            kink = pop.r_support[1]
            kfun = lambda tt, kink=kink: np.abs(np.asarray(tt) - kink)[:, None] ** 3
            kL = basis.evaluate(basis.project(kfun, extra=pop.r_support), t)
            rows[-1]["bias_kink_nuisance"] = float(np.sum(w * r[:, 0] * (kL - kfun(t))[:, 0]))
        Ls_ = [r_["L"] for r_ in rows]
        out["zero_trace" if zero_trace else "full"] = {
            "rows": rows,
            "slope_bias": _slope(Ls_, [r_["bias"] for r_ in rows]),
            "slope_bias_kink_nuisance": _slope(Ls_, [r_["bias_kink_nuisance"] for r_ in rows]),
            "slope_r_L1": _slope(Ls_, [r_["r_proj_L1"] for r_ in rows]),
            "slope_r_L2": _slope(Ls_, [r_["r_proj_L2"] for r_ in rows]),
        }
    return out


def check_item5_out(pop: Pop, Ks=(6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)) -> dict:
    """Outcome bias L(m_bar_K - m) with population-LS centering, per-arm projections."""
    J0, J1 = pop.d.J
    # extra breaks at the kinks of the C^{2,1} design for accurate quadrature
    e, wq = _gl_nodes([J0, pop.l0, pop.u0, _K1, _K2, J1], order=40, pieces=30)
    wJ = wq * pop.f_eta(e)
    mom = pop.arm_moments(e)
    out = pop.outcome()
    gI = pop.gI0(e)
    rows = []
    for K in Ks:
        sv = _arm_sieve(pop, K, e, wJ, mom)
        bias_L = 0.0
        bias_riesz = 0.0
        l1 = 0.0
        l2 = 0.0
        sup_err = 0.0
        lebesgue = []
        for arm, s in ((0, -1.0), (1, 1.0)):
            pi, m, V = mom[arm]["pi"], mom[arm]["m"], mom[arm]["V"]
            G, nb, Bm, v = sv[arm]["G"], sv[arm]["nb"], sv[arm]["Bm"], sv[arm]["v"]
            c = pop.d.b(e) + (pop.d.a(e) if arm == 1 else 0.0)
            ww = wJ * pi
            rhs = np.concatenate((Bm.T @ (ww * c), m.T @ (ww * c)))
            coef = np.linalg.solve(G, rhs)
            spl = Bm @ coef[:nb]
            kK = coef[nb:]
            # m_bar_K - m on this arm: (spl - c) + X'kK
            fun_err = spl - c
            bias_L += s * float(np.sum(wJ * gI * fun_err)) + s * float(out["B"] @ kK)
            om = out["omega"][arm]
            rfun = s * gI / pi - m @ om
            bias_riesz += float(np.sum(ww * (rfun + m @ om) * (fun_err + m @ kK)
                                       + ww * np.einsum("j,ijk,k->i", om, V, kK)))
            sup_err = max(sup_err, float(np.max(np.abs(fun_err))))
            # projection residual of r on this arm: function part only matters for L1
            vw, vk = v[:nb], v[nb:]
            u = Bm @ vw - rfun
            Dk = vk - om
            l2 += float(np.sum(ww * ((u + m @ Dk) ** 2 + np.einsum("j,ijk,k->i", Dk, V, Dk))))
            # L1 norm of the conditional mean of r_K - r given (eta, D): a lower
            # bound on the L1 norm, dominated by the jumps of the function part
            l1 += float(np.sum(ww * np.abs(u + m @ Dk)))
            # Lebesgue constant of the spline part of the weighted, X-partialled projection
            Ginv = np.linalg.inv(G)
            ev = np.linspace(J0 + 1e-9, J1 - 1e-9, 301)
            Bev = evaluate_basis_zero_outside(ev, _basis_info(K, pop.d.J))
            kern = Bev @ Ginv[:nb] @ np.vstack((Bm.T, m.T))          # (301, nodes)
            lebesgue.append(float(np.max(np.sum(np.abs(kern) * ww[None, :], axis=1))))
        rows.append({"K": K, "bias": bias_L, "bias_via_riesz": bias_riesz,
                     "r_minus_PiK_r_L1": l1, "r_minus_PiK_r_L2": float(np.sqrt(l2)),
                     "sup_m_minus_mbar_function_part": sup_err,
                     "lebesgue_constant_max_arm": max(lebesgue)})
    Ks_ = [r_["K"] for r_ in rows]
    return {
        "design": pop.d.name, "rows": rows,
        "slope_bias": _slope(Ks_, [r_["bias"] for r_ in rows]),
        "slope_r_L1": _slope(Ks_, [r_["r_minus_PiK_r_L1"] for r_ in rows]),
        "slope_r_L2": _slope(Ks_, [r_["r_minus_PiK_r_L2"] for r_ in rows]),
        "slope_sup_m": _slope(Ks_, [r_["sup_m_minus_mbar_function_part"] for r_ in rows]),
    }


# ================================================================== item C

def check_item_c_path(base: GDesign = PAPER) -> dict:
    """d/dt U_eps'(phi*) under X ~ law shifted by t v (Gaussian X) vs E[psi_T v'Sigma^{-1}(X - EX)]."""
    pop = Pop(base)
    if len(base.X.w) != 1:
        raise ValueError("Gaussian X only")
    Sigma = np.asarray(base.X.cov[0], float)
    v = np.array([0.6, -0.3, 0.5][: pop.dim])
    phi = pop.phi

    def theta(t):
        Xt = base.X.shifted(t * v)
        qlo = idx_quantile(Xt, pop.g0, pop.gx, base.eps)
        qhi = idx_quantile(Xt, pop.g0, pop.gx, 1 - base.eps)
        lo, hi = base.phi0 - qhi, base.phi0 - qlo
        e, w = _gl_nodes([lo, hi])
        F = np.sum(pop.w_vec(e) * idx_h1(Xt, pop.g0, pop.gx, phi - e), axis=1)
        return -float(np.sum(w * F * pop.f_eta(e)))

    h = 1e-5
    deriv = (theta(h) - theta(-h)) / (2 * h)
    # E[psi_T S], S = v' Sigma^{-1} (X - E X); psi_T = -r(T)'X~ - b_l z_hi(T) + b_u z_lo(T)
    c = np.linalg.solve(Sigma, v)
    lo_r, hi_r = pop.r_support
    t, w = _gl_nodes([lo_r, hi_r])
    H2 = idx_h2(pop.X, pop.g0, pop.gx, t)
    EX = pop.X.mean()
    # E[-r(T)'X~ S] = -int r(t)' E[X~ (X - EX)' | T=t] f(t) dt c
    cross = H2[:, :, 1:] - H2[:, :, :1] * EX[None, None, :]
    part_r = -float(np.einsum("i,ij,ijk,k->", w, pop.r(t), cross, c))

    def z_term(p, q):
        # E[z_p(T) (X - EX)] = -E[(X - EX) 1{T <= q}] / g(q)
        P1, E1, _ = idx_upper(pop.X, pop.g0, pop.gx, q, second=False)
        lower = (EX - E1[0]) - EX * (1.0 - P1[0])
        return -lower / float(idx_f(pop.X, pop.g0, pop.gx, q))

    part_q = (-pop.b_l * z_term(1 - base.eps, pop.qhi) @ c
              + pop.b_u * z_term(base.eps, pop.qlo) @ c)
    return {"fd_derivative": deriv, "E_psiT_S": part_r + part_q,
            "abs_diff": abs(deriv - part_r - part_q)}


# ============================================================ validation

def check_validate() -> dict:
    ref = Population(DESIGNS["paper_baseline"]).decomposition()
    pop = Pop(PAPER)
    A_g, A_rho = np.asarray(ref["A_g"]), np.asarray(ref["A_rho"])
    return {
        "phi": [pop.phi, ref["phi_star"]], "H": [pop.H, ref["H"]],
        "b_l": [pop.b_l, ref["b_l"]], "b_u": [pop.b_u, ref["b_u"]],
        "A_dens_minus_Ag_Arho": float(np.max(np.abs(pop.A_dens() - (A_g + A_rho)))),
        "A_U_diff": float(np.max(np.abs(pop.A_U() - np.asarray(ref["A_U"])))),
        "A_out_diff": float(np.max(np.abs(pop.outcome()["A_out"] - np.asarray(ref["A_out"])))),
        "Abar_diff": float(np.max(np.abs(pop.Abar() - np.asarray(ref["Abar"])))),
        "Abar_intercept": float(pop.Abar()[0]),
    }


# ============================================================ simulation

SIM_CONFIGS = {
    # name: (outcome interior-knot exponent, density basis exponent)
    "paper": (11 / 60, 11 / 60),
    "L_0.24": (11 / 60, 0.24),
    "L_0.30": (11 / 60, 0.30),
    "K_0.30": (0.30, 11 / 60),
    "K_0.30_L_0.24": (0.30, 0.24),
}


def _n_basis(n_fit: int, exponent: float) -> int:
    return max(8, int(round(2.0 * n_fit ** exponent)))


def sim_replicate(task):
    n, seed = task
    design = DESIGNS["paper_baseline"]
    s = les.generate(design, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    grid = np.linspace(*design.policy, 61)
    out = {name: [] for name in SIM_CONFIGS}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(les.ROLES)}
        g = les.ols_first_stage(s, folds["gamma"])
        nT = len(folds["index"])
        for name, (kexp, lexp) in SIM_CONFIGS.items():
            L = _n_basis(nT, lexp)
            _, phi = _rotation(s, design, folds, g, L, L, knots_exp=kexp, grid=grid)
            out[name].append(phi)
    return {name: {"fixed": v[0], "avg": float(np.mean(v))} for name, v in out.items()}


def simulate(ns, reps, workers, seed) -> dict:
    pop = Population(DESIGNS["paper_baseline"]).decomposition()
    phi = pop["phi_star"]
    result = {"prediction": {"V_rotated": pop["V_rotated_ols"], "V_fixed": pop["V_fixed_ols"],
                             "phi_star": phi}, "cells": {}}
    for n in ns:
        tasks = [(n, seed + 104_729 * n + r) for r in range(reps)]
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(sim_replicate, tasks, chunksize=2))
        cell = {}
        nT = n // 4
        for name, (kexp, lexp) in SIM_CONFIGS.items():
            entry = {"L": _n_basis(nT, lexp)}
            for key in ("fixed", "avg"):
                err = np.array([r_[name][key] for r_ in rows]) - phi
                entry[key] = {"n_var": float(n * np.var(err, ddof=1)),
                              "sqrt_n_bias": float(np.sqrt(n) * np.mean(err)),
                              "sqrt_n_bias_se": float(np.sqrt(n * np.var(err, ddof=1) / len(err)))}
            cell[name] = entry
        cell["reps"] = reps
        cell["relative_mc_se_var"] = float(np.sqrt(2.0 / (reps - 1)))
        result["cells"][str(n)] = cell
        print(f"N={n} reps={reps}")
        for name in SIM_CONFIGS:
            c = cell[name]
            print(f"  {name:>14} L={c['L']:>3}: rotated N Var={c['avg']['n_var']:.2f} "
                  f"(pred {pop['V_rotated_ols']:.2f}), sqrtN bias={c['avg']['sqrt_n_bias']:+.2f} "
                  f"(se {c['avg']['sqrt_n_bias_se']:.2f}); fixed N Var={c['fixed']['n_var']:.1f} "
                  f"(pred {pop['V_fixed_ols']:.1f})")
    return result


# ================================================================== main

def run_checks(fast: bool = False) -> dict:
    res = {}
    pops = {"paper": Pop(PAPER), "mixx": Pop(MIXX)}
    res["validate"] = check_validate()
    res["designs"] = {k: {"phi": p.phi, "H": p.H, "l0": p.l0, "u0": p.u0} for k, p in pops.items()}
    res["item1_mean"] = check_item1_mean(pops["paper"], reps=300 if fast else 2000)
    res["item1_sieve"] = {k: check_item1_sieve(p, Ls=(8, 16, 32) if fast else
                                               (8, 12, 16, 24, 32, 48, 64, 96))
                          for k, p in pops.items()}
    res["item1_maps"] = {k: check_item1_maps(p) for k, p in pops.items()}
    res["item1_maps"]["mixx_raw"] = check_item1_maps(Pop(MIXX_RAW))
    res["item1_var"] = check_item1_var(pops["paper"], n=50_000 if fast else 400_000)
    res["item3_ls"] = check_item3_ls()
    res["item3_riesz"] = {k: check_item3_riesz(p, n=200_000 if fast else 3_000_000,
                                               n_dirs=4 if fast else 12)
                          for k, p in pops.items()}
    res["item3_sieve"] = {k: check_item3_sieve(p) for k, p in pops.items()}
    res["item4_shift"] = check_item4_shift(n=8000 if fast else 16000)
    res["item4_center"] = check_item4_center()
    res["item5_dens"] = {k: check_item5_dens(p) for k, p in pops.items()}
    res["item5_out"] = check_item5_out(Pop(KINK))
    res["item_c_path"] = check_item_c_path()
    return res


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--n", type=int, nargs="+", default=[16000, 32000])
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.simulate:
        res = simulate(args.n, args.reps, args.workers, args.seed)
        path = args.out / "simulation_summary.json"
    else:
        res = run_checks(fast=args.fast)
        path = args.out / ("checks_fast.json" if args.fast else "checks.json")
    path.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"[wrote] {path}")


if __name__ == "__main__":
    main()
