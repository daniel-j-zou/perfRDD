"""Two-score threshold rules: population utilities by quadrature (exploratory).

Multiple scores are out of scope for the paper (``code/PROJECT_SCOPE.md``); this
is an exploratory population check of the two-score extension.  Design:

    X ~ N(0, I_p),  T_j = gamma_j'X,  Q_j = T_j + eta_j  (j = 1, 2),
    eta ~ N(0, [[s1^2, rho s1 s2], [rho s1 s2, s2^2]]),  X independent of eta,
    W = alpha(eta) + X'beta2,  deployed rule D0 = 1{Q_1 > phi01, Q_2 > phi02}.

The supported population is I0 = 1{eps <= pi0(eta) <= 1 - eps} with
pi0(e) = P(T_1 > phi01 - e_1, T_2 > phi02 - e_2).  For a rule A (a set of score
values) the supported utility is U(A) = E[(W - c) 1{Q in A} I0].  Because X is
Gaussian, lambda(t) = beta2'E(X | T = t) = kappa't with kappa = Var(T)^{-1} G beta2
(G the 2 x p matrix of index slopes), so given eta = e

    E[(W - c) 1{T in A - e}] = (alpha(e) - c) P(T in A - e) + kappa'E[T 1{T in A - e}],

which is closed form for quadrants (bivariate normal orthant and Tallis moments)
and half-planes.  The eta integral over the supported region uses iterated
Gauss--Legendre nodes: for each eta_1 the supported eta_2 form an interval,
because pi0 increases in both coordinates.

Rules compared:
  * conjunctive 1{Q_1 > phi_1, Q_2 > phi_2}: maximizer, Hessian, all critical
    points, face conditions E[W - c | Q_1 = phi_1, Q_2 > phi_2, I0 = 1] = 0
    (and the analogue on face 2);
  * first best A* = {q : tau(q) > 0}, tau(q) = E(W - c | Q = q, I0 = 1);
  * best compensatory 1{w_1 Q_1 + w_2 Q_2 > phi}, w = (cos theta, sin theta);
  * best single-score 1{Q_j > phi}.

The cross partial of the conjunctive utility is the supported density of Q at
the corner times tau there: d^2 U / d phi_1 d phi_2 = p(phi) tau(phi) = m(phi).

Run (from ``code/``):
    python -m experiments.scripts.two_score_population            # designs, noise check, sweep
    python -m experiments.scripts.two_score_population --mc-check # adds Monte Carlo checks
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize, minimize_scalar, root
from scipy.special import ndtr
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "two_score_population"

_SQ2PI = np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class Design:
    name: str
    gamma: Tuple[Tuple[float, ...], Tuple[float, ...]]   # index slopes, E X = 0, Var X = I
    sigma_eta: Tuple[float, float]
    rho: float                                            # Corr(eta_1, eta_2)
    # alpha(e) = a0 + a1 e1 + a2 e2 + a_min * softmin(e1, e2),
    # softmin(e1, e2) = -log(exp(-e1) + exp(-e2)) + log 2 (increasing in both).
    alpha: Tuple[float, float, float]
    beta2: Tuple[float, ...]
    cost: float
    a_min: float = 0.0
    eps: float = 0.10
    phi0: Tuple[float, float] = (0.0, 0.0)
    box: Tuple[float, float] = (-4.0, 4.0)                # policy set Phi = box^2
    note: str = ""


_G1, _G2 = (0.8, 0.6, 0.0), (0.0, 0.6, 0.8)              # Var T_j = 1, Cov = 0.36
_BETA2_PAPER = (0.4, -0.25, 0.15)                        # paper Section 5 slopes
# beta2 = G'kappa gives lambda(t) = kappa't; kappa = (0.2, 0.2) here
_BETA2_SYM = (0.16, 0.24, 0.16)

DESIGNS: Dict[str, Design] = {
    "paper_beta2": Design(
        name="paper_beta2", gamma=(_G1, _G2), sigma_eta=(1.0, 1.0), rho=0.5,
        alpha=(2.0, 0.8, 0.5), beta2=_BETA2_PAPER, cost=2.25,
        note="alpha = 2 + 0.8 e1 + 0.5 e2, paper beta2 (kappa_2 < 0), c = 2.25",
    ),
    "symmetric": Design(
        name="symmetric", gamma=(_G1, _G2), sigma_eta=(1.0, 1.0), rho=0.5,
        alpha=(2.0, 0.7, 0.7), beta2=_BETA2_SYM, cost=2.25,
        note="alpha = 2 + 0.7 e1 + 0.7 e2, kappa = (0.2, 0.2), c = 2.25",
    ),
    "symmetric_paper_beta2": Design(
        name="symmetric_paper_beta2", gamma=(_G1, _G2), sigma_eta=(1.0, 1.0), rho=0.5,
        alpha=(2.0, 0.7, 0.7), beta2=_BETA2_PAPER, cost=2.25,
        note="symmetric alpha with the paper beta2 (kappa_2 < 0): flat interior max and a saddle",
    ),
    "complementary": Design(
        name="complementary", gamma=(_G1, _G2), sigma_eta=(1.0, 1.0), rho=0.5,
        alpha=(2.0, 0.0, 0.0), a_min=1.4, beta2=_BETA2_SYM, cost=2.25,
        note="alpha = 2 + 1.4 softmin(e1, e2), both abilities needed; kappa = (0.2, 0.2)",
    ),
}


# ---------------------------------------------------------------- normal helpers

_GL_R = leggauss(32)


def bvn_sf(a, b, r: float):
    """P(Z_1 > a, Z_2 > b) for standard bivariate normal Z with correlation r.

    Drezner--Wesolowsky: Phi(-a)Phi(-b) + (2 pi)^{-1} int_0^{asin r}
    exp{-(a^2 - 2ab sin t + b^2) / (2 cos^2 t)} dt, by 32-point Gauss--Legendre
    (accurate to ~1e-13 for |r| <= 0.9).
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    base = ndtr(-a) * ndtr(-b)
    if r == 0.0:
        return base
    x, w = _GL_R
    th = 0.5 * np.arcsin(r) * (x + 1.0)
    st, c2 = np.sin(th), np.cos(th) ** 2
    out = np.zeros(a.shape)
    a2b2, ab = a * a + b * b, a * b
    for k in range(len(x)):
        out += w[k] * np.exp(-(a2b2 - 2.0 * ab * st[k]) / (2.0 * c2[k]))
    return base + out * 0.5 * np.arcsin(r) / (2.0 * np.pi)


def _gl_nodes(breaks, order: int = 24, pieces: int = 4):
    """Composite Gauss--Legendre nodes and weights on sorted breaks."""
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

class TwoScorePopulation:
    """Supported utilities of two-score rules for one Gaussian design."""

    def __init__(self, design: Design, order: int = 24, pieces: int = 4,
                 halfwidth: float = 8.0):
        self.d = design
        G = np.asarray(design.gamma, dtype=float)
        self.G = G
        self.ST = G @ G.T                                   # Var(T)
        self.sT = np.sqrt(np.diag(self.ST))
        self.rT = float(self.ST[0, 1] / (self.sT[0] * self.sT[1]))
        if abs(self.rT) > 0.9:
            raise ValueError("bvn_sf is tuned for |Corr(T_1, T_2)| <= 0.9")
        self.beta2 = np.asarray(design.beta2, dtype=float)
        self.kappa = np.linalg.solve(self.ST, G @ self.beta2)   # lambda(t) = kappa't
        se = np.asarray(design.sigma_eta, dtype=float)
        self.Seta = np.array([[se[0] ** 2, design.rho * se[0] * se[1]],
                              [design.rho * se[0] * se[1], se[1] ** 2]])
        self.phi0 = np.asarray(design.phi0, dtype=float)
        self.e, self.w = self._support_nodes(order, pieces, halfwidth)
        self.alpha_e = self.alpha(self.e)
        self.P_I0 = float(self.w.sum())

    # -- primitives -------------------------------------------------------------
    def alpha(self, e):
        e = np.atleast_2d(e)
        a0, a1, a2 = self.d.alpha
        out = a0 + a1 * e[:, 0] + a2 * e[:, 1]
        if self.d.a_min:
            out = out + self.d.a_min * (-np.logaddexp(-e[:, 0], -e[:, 1]) + np.log(2.0))
        return out

    def f_eta(self, e):
        e = np.atleast_2d(e)
        Si = np.linalg.inv(self.Seta)
        qf = np.einsum("ij,jk,ik->i", e, Si, e)
        return np.exp(-0.5 * qf) / (2.0 * np.pi * np.sqrt(np.linalg.det(self.Seta)))

    def g_T(self, t):
        t = np.atleast_2d(t)
        Si = np.linalg.inv(self.ST)
        qf = np.einsum("ij,jk,ik->i", t, Si, t)
        return np.exp(-0.5 * qf) / (2.0 * np.pi * np.sqrt(np.linalg.det(self.ST)))

    def orthant(self, s):
        """P(T_1 > s_1, T_2 > s_2), s of shape (..., 2)."""
        s = np.asarray(s, dtype=float)
        return bvn_sf(s[..., 0] / self.sT[0], s[..., 1] / self.sT[1], self.rT)

    def tallis(self, s):
        """E[T 1{T_1 > s_1, T_2 > s_2}], shape (..., 2)."""
        s = np.asarray(s, dtype=float)
        a, b, r = s[..., 0] / self.sT[0], s[..., 1] / self.sT[1], self.rT
        q = np.sqrt(1.0 - r * r)
        pa, pb = norm.pdf(a), norm.pdf(b)
        Fa, Fb = ndtr(-(a - r * b) / q), ndtr(-(b - r * a) / q)
        m1 = pa * Fb + r * pb * Fa
        m2 = pb * Fa + r * pa * Fb
        return np.stack([self.sT[0] * m1, self.sT[1] * m2], axis=-1)

    def pi0(self, e):
        e = np.atleast_2d(e)
        return self.orthant(self.phi0[None, :] - e)

    # -- supported eta nodes --------------------------------------------------
    def _support_nodes(self, order, pieces, halfwidth):
        """Nodes on {eps <= pi0 <= 1 - eps} inside a +-halfwidth sd box."""
        eps, (p1, p2) = self.d.eps, self.phi0
        se = np.sqrt(np.diag(self.Seta))
        B1, B2 = halfwidth * se
        pi = lambda e1, e2: float(self.pi0(np.array([[e1, e2]]))[0])
        # supported eta_1 range: P(T_1 > phi01 - e1) must reach eps
        a_lo = p1 - self.sT[0] * norm.isf(eps)
        a_hi = p1 - self.sT[0] * norm.isf(1.0 - eps)
        lo1 = max(a_lo, -B1)
        breaks = [lo1, B1]
        if lo1 < a_hi < B1:
            breaks.append(a_hi)
        # kinks of the e2 interval at the box edges
        for e2_edge, level in ((-B2, eps), (B2, 1.0 - eps), (-B2, 1.0 - eps)):
            f = lambda x: pi(x, e2_edge) - level
            if f(lo1) * f(B1) < 0:
                breaks.append(brentq(f, lo1, B1, xtol=1e-13))
        e1, w1 = _gl_nodes(sorted(breaks), order=order, pieces=pieces)
        c21 = self.Seta[0, 1] / self.Seta[0, 0]
        s21 = np.sqrt(self.Seta[1, 1] - self.Seta[0, 1] ** 2 / self.Seta[0, 0])
        nodes, weights = [], []
        for x1, v1 in zip(e1, w1):
            lo = -B2 if pi(x1, -B2) >= eps else (
                np.nan if pi(x1, B2) < eps else
                brentq(lambda x: pi(x1, x) - eps, -B2, B2, xtol=1e-13))
            hi = B2 if pi(x1, B2) <= 1.0 - eps else (
                np.nan if pi(x1, -B2) > 1.0 - eps else
                brentq(lambda x: pi(x1, x) - (1.0 - eps), -B2, B2, xtol=1e-13))
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                continue
            # standardize eta_2 given eta_1 so that the nodes resolve the
            # conditional density when |rho| is large
            mu, sc = c21 * x1, s21
            zlo, zhi = max((lo - mu) / sc, -halfwidth), min((hi - mu) / sc, halfwidth)
            if zhi <= zlo:
                continue
            z, wz = _gl_nodes([zlo, zhi], order=order, pieces=pieces)
            nodes.append(np.column_stack([np.full_like(z, x1), mu + sc * z]))
            weights.append(v1 * sc * wz)
        e = np.concatenate(nodes)
        return e, np.concatenate(weights) * self.f_eta(e)

    # -- conjunctive rule ------------------------------------------------------
    def U_conj(self, phi):
        """Supported utility of 1{Q_1 > phi_1, Q_2 > phi_2}; phi of shape (..., 2)."""
        phi = np.asarray(phi, dtype=float)
        flat = phi.reshape(-1, 2)
        out = np.empty(len(flat))
        net = self.alpha_e - self.d.cost
        for i, p in enumerate(flat):
            s = p[None, :] - self.e
            out[i] = np.sum(self.w * (net * self.orthant(s) + self.tallis(s) @ self.kappa))
        return out.reshape(phi.shape[:-1])

    def _face_parts(self, phi, j):
        """Face j of the quadrant at phi: (density, -dU/dphi_j) for each phi.

        density = d/dphi_j P(Q_j <= phi_j, Q_k > phi_k, I0 = 1), and
        -dU/dphi_j = density * E[W - c | Q_j = phi_j, Q_k > phi_k, I0 = 1].
        """
        k = 1 - j
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        sj, sk = self.sT[j], self.sT[k]
        beta = self.ST[j, k] / sj ** 2
        sc = np.sqrt(self.ST[k, k] - self.ST[j, k] ** 2 / sj ** 2)
        net = self.alpha_e - self.d.cost
        dens, flux = np.empty(len(phi)), np.empty(len(phi))
        for i, p in enumerate(phi):
            tj = p[j] - self.e[:, j]
            tk = p[k] - self.e[:, k]
            gj = np.exp(-0.5 * (tj / sj) ** 2) / (_SQ2PI * sj)
            mu = beta * tj
            z = (tk - mu) / sc
            Sz, pz = ndtr(-z), np.exp(-0.5 * z * z) / _SQ2PI
            dens[i] = np.sum(self.w * gj * Sz)
            flux[i] = np.sum(self.w * gj * ((net + self.kappa[j] * tj + self.kappa[k] * mu) * Sz
                                            + self.kappa[k] * sc * pz))
        return dens, flux

    def grad_conj(self, phi):
        phi = np.atleast_2d(np.asarray(phi, dtype=float))
        g = np.empty_like(phi)
        for j in (0, 1):
            g[:, j] = -self._face_parts(phi, j)[1]
        return g

    def face_means(self, phi):
        """(face density, E[W - c | on face, I0 = 1]) for faces 1 and 2."""
        out = {}
        for j in (0, 1):
            dens, flux = self._face_parts(phi, j)
            out[f"face{j + 1}"] = {"density": float(dens[0]), "mean_net": float(flux[0] / dens[0])}
        return out

    def hess_conj(self, phi, h: float = 1e-4):
        phi = np.asarray(phi, dtype=float)
        H = np.empty((2, 2))
        for j in (0, 1):
            dp = np.zeros(2)
            dp[j] = h
            H[:, j] = (self.grad_conj(phi + dp)[0] - self.grad_conj(phi - dp)[0]) / (2 * h)
        return H

    # -- density and net effect of Q on the supported population -------------
    def m_p(self, q, chunk: int = 2000):
        """m(q) = p(q) tau(q) and p(q), the supported density of Q; q shape (n, 2)."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        Si = np.linalg.inv(self.ST)
        cst = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(self.ST)))
        net = self.alpha_e - self.d.cost - self.e @ self.kappa
        m, p = np.empty(len(q)), np.empty(len(q))
        for i0 in range(0, len(q), chunk):
            qq = q[i0:i0 + chunk]
            t0 = qq[:, None, 0] - self.e[None, :, 0]
            t1 = qq[:, None, 1] - self.e[None, :, 1]
            gk = cst * np.exp(-0.5 * (Si[0, 0] * t0 * t0 + 2 * Si[0, 1] * t0 * t1
                                      + Si[1, 1] * t1 * t1)) * self.w[None, :]
            p[i0:i0 + chunk] = gk.sum(axis=1)
            m[i0:i0 + chunk] = gk @ net + (qq @ self.kappa) * p[i0:i0 + chunk]
        return m, p

    # -- compensatory rule -------------------------------------------------------
    def U_comp(self, theta, phi):
        """Supported utility of 1{cos(theta) Q_1 + sin(theta) Q_2 > phi}."""
        w = np.array([np.cos(theta), np.sin(theta)])
        sw = float(np.sqrt(w @ self.ST @ w))
        z = (phi - self.e @ w) / sw
        lam = float(self.kappa @ self.ST @ w) / sw
        net = self.alpha_e - self.d.cost
        return float(np.sum(self.w * (net * ndtr(-z) + lam * np.exp(-0.5 * z * z) / _SQ2PI)))

    def best_phi_comp(self, theta):
        lo, hi = 3.0 * np.array(self.d.box)
        grid = np.linspace(lo, hi, 121)
        vals = np.array([self.U_comp(theta, x) for x in grid])
        i = int(np.argmax(vals))
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(lambda x: -self.U_comp(theta, x), bounds=(a, b),
                              method="bounded", options={"xatol": 1e-10})
        return float(res.x), float(-res.fun)


# ---------------------------------------------------------------- analysis

def _classify(H):
    ev = np.linalg.eigvalsh(H)
    if np.all(ev < 0):
        return "max"
    if np.all(ev > 0):
        return "min"
    return "saddle"


def critical_points(pop: TwoScorePopulation, step: float = 0.1):
    """All zeros of grad U_conj on the policy box, by sign changes and Newton."""
    lo, hi = pop.d.box
    xs = np.arange(lo, hi + 1e-9, step)
    P1, P2 = np.meshgrid(xs, xs, indexing="ij")
    grid = np.column_stack([P1.ravel(), P2.ravel()])
    gr = pop.grad_conj(grid).reshape(len(xs), len(xs), 2)
    s1, s2 = np.sign(gr[..., 0]), np.sign(gr[..., 1])
    cands = []
    for i in range(len(xs) - 1):
        for k in range(len(xs) - 1):
            c1 = s1[i:i + 2, k:k + 2]
            c2 = s2[i:i + 2, k:k + 2]
            if c1.min() < c1.max() and c2.min() < c2.max():
                cands.append([xs[i] + step / 2, xs[k] + step / 2])
    found: List[np.ndarray] = []
    for x0 in cands:
        sol = root(lambda x: pop.grad_conj(x)[0], x0, method="hybr", tol=1e-12)
        x = sol.x
        if not sol.success or np.any(x < lo) or np.any(x > hi):
            continue
        if np.max(np.abs(pop.grad_conj(x)[0])) > 1e-9:
            continue
        if all(np.max(np.abs(x - y)) > 1e-5 for y in found):
            found.append(x)
    out = []
    for x in found:
        H = pop.hess_conj(x)
        out.append({"phi": x.tolist(), "U": float(pop.U_conj(x)), "type": _classify(H),
                    "hess_eig": np.linalg.eigvalsh(H).tolist()})
    return out, {"grid_step": step, "grad_grid_max_abs": float(np.abs(gr).max())}


def best_conjunctive(pop: TwoScorePopulation, crit):
    """Box-constrained maximizer (L-BFGS-B from the best coarse-grid points)."""
    lo, hi = pop.d.box
    xs = np.linspace(lo, hi, 33)
    P1, P2 = np.meshgrid(xs, xs, indexing="ij")
    grid = np.column_stack([P1.ravel(), P2.ravel()])
    U = pop.U_conj(grid)
    starts = [grid[i] for i in np.argsort(-U)[:4]] + [np.array(c["phi"]) for c in crit]
    best = None
    for x0 in starts:
        res = minimize(lambda x: -pop.U_conj(x), x0, jac=lambda x: -pop.grad_conj(x)[0],
                       method="L-BFGS-B", bounds=[(lo, hi), (lo, hi)],
                       options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 500})
        if best is None or -res.fun > best[1]:
            best = (res.x, float(-res.fun))
    x, Ux = best
    at_bound = [bool(min(v - lo, hi - v) < 1e-6) for v in x]
    return x, Ux, at_bound


def first_best(pop: TwoScorePopulation, h: float = 0.04, half: float = 7.0):
    """U(A*) = int m^+ dq on a grid, with the grid error of the best compensatory
    and conjunctive rules reported for calibration."""
    xs = np.arange(-half, half + 1e-9, h)
    Q1, Q2 = np.meshgrid(xs, xs, indexing="ij")
    q = np.column_stack([Q1.ravel(), Q2.ravel()])
    m, p = pop.m_p(q)
    w = np.full(len(xs), h)
    w[0] = w[-1] = h / 2                                   # trapezoid
    W2 = np.outer(w, w).ravel()
    return q, m, p, W2


def analyse(design: Design, verbose: bool = True, fb_h: float = 0.04,
            nodes: Tuple[int, int] = (20, 3)) -> Dict[str, object]:
    pop = TwoScorePopulation(design, order=nodes[0], pieces=nodes[1])
    d = design
    U0 = float(pop.U_conj(pop.phi0))
    crit, crit_info = critical_points(pop)
    x, Ux, at_bound = best_conjunctive(pop, crit)
    H = pop.hess_conj(x)
    ev, evec = np.linalg.eigh(H)
    faces = pop.face_means(x)
    m_corner, p_corner = pop.m_p(x)
    # d tau_j / d phi_j along face j; where tau_j = 0, H_jj = -f_j d tau_j / d phi_j
    dtau = []
    for j in (0, 1):
        dp = np.zeros(2)
        dp[j] = 1e-4
        key = f"face{j + 1}"
        dtau.append((pop.face_means(x + dp)[key]["mean_net"]
                     - pop.face_means(x - dp)[key]["mean_net"]) / 2e-4)
    # compensatory and single-score rules
    thetas = np.linspace(0.0, 2.0 * np.pi, 145)[:-1]
    comp_grid = [(t, *pop.best_phi_comp(t)) for t in thetas]
    t0, _, _ = max(comp_grid, key=lambda r: r[2])
    res = minimize(lambda v: -pop.U_comp(v[0], v[1]),
                   [t0, pop.best_phi_comp(t0)[0]], method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 4000})
    th_c, ph_c = float(res.x[0]) % (2 * np.pi), float(res.x[1])
    U_comp = float(-res.fun)
    single = {}
    for j, th in ((1, 0.0), (2, np.pi / 2)):
        ph, Us = pop.best_phi_comp(th)
        single[f"Q{j}"] = {"phi": ph, "U": Us}
    j_best = max(single, key=lambda k: single[k]["U"])
    # first best on a grid; calibrate the grid against the closed forms
    q, m, p, W2 = first_best(pop, h=fb_h)
    wc = np.array([np.cos(th_c), np.sin(th_c)])
    U_fb_grid = float(np.sum(W2 * np.maximum(m, 0.0)))
    U_comp_grid = float(np.sum(W2 * m * (q @ wc > ph_c)))
    U_conj_grid = float(np.sum(W2 * m * ((q[:, 0] > x[0]) & (q[:, 1] > x[1]))))
    U_fb = U_comp + (U_fb_grid - U_comp_grid)              # difference integrand is local
    mass_grid = float(np.sum(W2 * p))
    tau_at_phi0 = pop.m_p(pop.phi0)
    out = {
        "design": asdict(d),
        "kappa": pop.kappa.tolist(),
        "corr_T": pop.rT,
        "P_I0": pop.P_I0,
        "n_eta_nodes": int(len(pop.w)),
        "deployed": {"phi0": pop.phi0.tolist(), "U": U0,
                     "faces": pop.face_means(pop.phi0),
                     "tau_corner": float(tau_at_phi0[0][0] / tau_at_phi0[1][0])},
        "conjunctive": {
            "phi": x.tolist(), "U": Ux, "at_box_bound": at_bound,
            "gain_over_deployed": Ux - U0,
            "hessian": H.tolist(), "hess_eig": ev.tolist(),
            "hess_eigvec": evec.tolist(), "neg_def": bool(np.all(ev < 0)),
            "cond": float(abs(ev).max() / abs(ev).min()),
            "faces": faces,
            "dtau_face": dtau,
            "Hjj_plus_f_dtau": [float(H[j, j] + faces[f"face{j + 1}"]["density"] * dtau[j])
                                for j in (0, 1)],
            "corner": {"m": float(m_corner[0]), "p": float(p_corner[0]),
                       "tau": float(m_corner[0] / p_corner[0]),
                       "H12_minus_m": float(0.5 * (H[0, 1] + H[1, 0]) - m_corner[0]),
                       "H12_asym": float(H[0, 1] - H[1, 0])},
        },
        "critical_points": crit,
        "critical_grid": crit_info,
        "compensatory": {"theta": th_c, "w": wc.tolist(), "phi": ph_c, "U": U_comp},
        "single": single, "single_best": j_best,
        "first_best": {"U": U_fb, "U_grid": U_fb_grid, "grid_h": fb_h,
                       "grid_err_comp": U_comp_grid - U_comp,
                       "grid_err_conj": U_conj_grid - Ux,
                       "grid_mass_err": mass_grid - pop.P_I0,
                       "share_tau_pos": float(np.sum(W2 * p * (m > 0)) / mass_grid)},
    }
    out["summary"] = {
        "value_second_score": Ux - single[j_best]["U"],
        "loss_conj_vs_fb": U_fb - Ux,
        "loss_comp_vs_fb": U_fb - U_comp,
        "conj_minus_comp": Ux - U_comp,
        "share_of_fb_gain_conj": (Ux - U0) / (U_fb - U0),
        "share_of_fb_gain_comp": (U_comp - U0) / (U_fb - U0),
        "share_of_fb_gain_single": (single[j_best]["U"] - U0) / (U_fb - U0),
        "interior": bool(not any(at_bound) and np.all(ev < 0)),
    }
    if verbose:
        _print(out)
    return out


def _print(o):
    c, s = o["conjunctive"], o["summary"]
    print(f"== {o['design']['name']}: rho={o['design']['rho']}, beta2={o['design']['beta2']}, "
          f"kappa={np.round(o['kappa'], 4).tolist()}, P(I0)={o['P_I0']:.4f}")
    print(f"  deployed phi0={o['deployed']['phi0']}  U={o['deployed']['U']:.6f}")
    print(f"  conjunctive phi*={np.round(c['phi'], 5).tolist()}  U={c['U']:.6f}  "
          f"gain={c['gain_over_deployed']:.6f}  bound={c['at_box_bound']}")
    print(f"    Hessian eig={np.round(c['hess_eig'], 5).tolist()}  cond={c['cond']:.2f}  "
          f"H12-m={c['corner']['H12_minus_m']:.2e}")
    print(f"    face means: {c['faces']['face1']['mean_net']:.2e}, "
          f"{c['faces']['face2']['mean_net']:.2e};  tau(corner)={c['corner']['tau']:.4f}")
    for cp in o["critical_points"]:
        print(f"    critical point {np.round(cp['phi'], 4).tolist()} {cp['type']} U={cp['U']:.6f}")
    print(f"  compensatory w={np.round(o['compensatory']['w'], 4).tolist()} "
          f"phi={o['compensatory']['phi']:.4f} U={o['compensatory']['U']:.6f}")
    for k, v in o["single"].items():
        print(f"  single {k}: phi={v['phi']:.4f} U={v['U']:.6f}")
    fb = o["first_best"]
    print(f"  first best U={fb['U']:.6f} (grid {fb['U_grid']:.6f}; grid err comp "
          f"{fb['grid_err_comp']:.1e}, conj {fb['grid_err_conj']:.1e})")
    print(f"  value of 2nd score={s['value_second_score']:.6f}  loss conj={s['loss_conj_vs_fb']:.6f}"
          f"  loss comp={s['loss_comp_vs_fb']:.6f}  interior={s['interior']}")


# ---------------------------------------------------------------- Monte Carlo

def mc_check(design: Design, phi_star, comp, n: int = 20_000_000, band: float = 0.02,
             seed: int = 20261027, chunk: int = 2_000_000):
    """Independent check by simulation: U at phi0, phi* and the best compensatory
    rule comp = (theta, phi); the face means at phi*; tau near the corner."""
    pop = TwoScorePopulation(design)
    rng = np.random.default_rng(seed)
    G = pop.G
    L = np.linalg.cholesky(pop.Seta)
    phi_star = np.asarray(phi_star, dtype=float)
    wc = np.array([np.cos(comp[0]), np.sin(comp[0])])
    acc = {k: [] for k in ("U0", "Us", "Uc", "f1", "f2", "c1")}
    for i0 in range(0, n, chunk):
        X = rng.standard_normal((chunk, G.shape[1]))
        eta = rng.standard_normal((chunk, 2)) @ L.T
        Q = X @ G.T + eta
        I0 = pop.pi0(eta)
        I0 = (I0 >= design.eps) & (I0 <= 1.0 - design.eps)
        net = pop.alpha(eta) + X @ pop.beta2 - design.cost
        acc["U0"].append(net * ((Q[:, 0] > pop.phi0[0]) & (Q[:, 1] > pop.phi0[1]) & I0))
        acc["Us"].append(net * ((Q[:, 0] > phi_star[0]) & (Q[:, 1] > phi_star[1]) & I0))
        acc["Uc"].append(net * ((Q @ wc > comp[1]) & I0))
        f1 = I0 & (np.abs(Q[:, 0] - phi_star[0]) < band) & (Q[:, 1] > phi_star[1])
        f2 = I0 & (np.abs(Q[:, 1] - phi_star[1]) < band) & (Q[:, 0] > phi_star[0])
        acc["f1"].append(net[f1])
        acc["f2"].append(net[f2])
        c = I0 & (np.abs(Q[:, 0] - phi_star[0]) < 4 * band) & (np.abs(Q[:, 1] - phi_star[1]) < 4 * band)
        acc["c1"].append(net[c])
    res = {}
    for k in ("U0", "Us", "Uc"):
        v = np.concatenate(acc[k])
        res[k] = (float(v.mean()), float(v.std() / np.sqrt(len(v))))
    for k in ("f1", "f2", "c1"):
        v = np.concatenate(acc[k])
        res[k] = (float(v.mean()), float(v.std() / np.sqrt(len(v))), int(len(v)))
    faces = pop.face_means(phi_star)
    return {"n": n, "band": band, "seed": seed,
            "mc": {"U_deployed": res["U0"], "U_star": res["Us"], "U_comp": res["Uc"],
                   "face1_mean": res["f1"], "face2_mean": res["f2"],
                   "corner_tau_box": res["c1"]},
            "quadrature": {"U_deployed": float(pop.U_conj(pop.phi0)),
                           "U_star": float(pop.U_conj(phi_star)),
                           "U_comp": pop.U_comp(*comp),
                           "face1_mean": faces["face1"]["mean_net"],
                           "face2_mean": faces["face2"]["mean_net"],
                           "corner_tau": float(np.divide(*pop.m_p(phi_star))[0])}}


# ---------------------------------------------------------------- sweep

def sweep(base: Design, rhos, kappa2s, verbose: bool = False):
    """Vary rho and beta2.  beta2 = G'kappa (minimal norm) keeps kappa_1 at its
    baseline value and sets kappa_2, the loading of lambda(t) = kappa't on T_2."""
    pop = TwoScorePopulation(base)
    G = pop.G
    k1 = float(pop.kappa[0])
    rows = []
    for rho in rhos:
        for k2 in kappa2s:
            kap = np.array([k1, k2])
            beta2 = tuple(float(v) for v in G.T @ kap)
            dsg = replace(base, name=f"{base.name}_rho{rho}_k2{k2}", rho=rho, beta2=beta2)
            o = analyse(dsg, verbose=verbose, fb_h=0.05, nodes=(16, 2))
            c, s = o["conjunctive"], o["summary"]
            rows.append({
                "rho": rho, "kappa": kap.tolist(), "beta2": list(beta2),
                "phi": c["phi"], "at_bound": c["at_box_bound"], "U_conj": c["U"],
                "U_deployed": o["deployed"]["U"], "hess_eig": c["hess_eig"],
                "cond": c["cond"], "neg_def": c["neg_def"],
                "n_crit": len(o["critical_points"]),
                "crit_types": [cp["type"] for cp in o["critical_points"]],
                "U_comp": o["compensatory"]["U"], "w_comp": o["compensatory"]["w"],
                "single_best": o["single_best"],
                "U_single": o["single"][o["single_best"]]["U"],
                "U_fb": o["first_best"]["U"], "P_I0": o["P_I0"],
                **s,
            })
            r = rows[-1]
            print(f"rho={rho:5.2f} k2={k2:5.2f}  phi*=({r['phi'][0]:6.3f},{r['phi'][1]:6.3f})"
                  f" bound={r['at_bound']}  eig=({r['hess_eig'][0]:.4f},{r['hess_eig'][1]:.4f})"
                  f" crit={r['crit_types']}  gain={r['U_conj'] - r['U_deployed']:.4f}"
                  f" 2nd={r['value_second_score']:.4f} lossconj={r['loss_conj_vs_fb']:.4f}"
                  f" losscomp={r['loss_comp_vs_fb']:.5f}", flush=True)
    return rows


def index_scale_check(scales=(1.0, 0.6, 0.4)):
    """Shrink the index noise T (gamma -> s gamma, beta2 -> s beta2 keeps kappa):
    does the conjunctive rule close the gap to the first best when Q tracks eta
    more closely?  Linear (symmetric) and complementary alpha."""
    rows = []
    for name in ("symmetric", "complementary"):
        base = DESIGNS[name]
        for sc in scales:
            G = sc * np.asarray(base.gamma)
            dsg = replace(base, name=f"{name}_scale{sc}", gamma=tuple(map(tuple, G)),
                          beta2=tuple(sc * np.asarray(base.beta2)))
            o = analyse(dsg, verbose=False, fb_h=0.04, nodes=(16, 2))
            c, s = o["conjunctive"], o["summary"]
            rows.append({"design": name, "scale": sc, "P_I0": o["P_I0"], "phi": c["phi"],
                         "hess_eig": c["hess_eig"], "U_deployed": o["deployed"]["U"],
                         "U_conj": c["U"], "U_comp": o["compensatory"]["U"],
                         "U_single": o["single"][o["single_best"]]["U"],
                         "U_fb": o["first_best"]["U"], **s})
            r = rows[-1]
            print(f"{name:13s} scale={sc}: P(I0)={r['P_I0']:.3f} phi*=({r['phi'][0]:.3f},{r['phi'][1]:.3f})"
                  f" share of first-best gain: conj {r['share_of_fb_gain_conj']:.3f},"
                  f" comp {r['share_of_fb_gain_comp']:.3f}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mc-check", action="store_true")
    ap.add_argument("--mc-n", type=int, default=20_000_000)
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {name: analyse(dsg) for name, dsg in DESIGNS.items()}
    if args.mc_check:
        for name, dsg in DESIGNS.items():
            comp = (results[name]["compensatory"]["theta"], results[name]["compensatory"]["phi"])
            mc = mc_check(dsg, results[name]["conjunctive"]["phi"], comp, n=args.mc_n)
            results[name]["mc_check"] = mc
            print(f"MC {name}: {json.dumps(mc, indent=1)}")
    (args.out / "designs.json").write_text(json.dumps(results, indent=1))
    (args.out / "index_scale_check.json").write_text(json.dumps(index_scale_check(), indent=1))
    if not args.no_sweep:
        rows = sweep(DESIGNS["symmetric"], rhos=(-0.5, 0.0, 0.5, 0.8, 0.95),
                     kappa2s=(-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4))
        (args.out / "sweep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
