"""Two-score threshold rules: a pairwise estimator of the supported utility (exploratory).

Multiple scores are future work in ``PROJECT_SCOPE.md``; this module is the exploratory
estimator agreed with the author on 2026-10-27 (research log, same date).

Model: Q = T + eta with T = Gamma X~ (two scores), X independent of eta = (eta_1, eta_2),
effect W = alpha(eta) + X'beta2, outcome Y = b(eta) + X'beta1 + D W + error, deployed
rule D = 1{Q in A0}. For a rule A the supported utility is

    U(A) = E[ I(eta) {alpha(eta) - c + X'beta2} 1{T + eta in A} ],
    I(eta) = 1{eps <= pi0(eta) <= 1 - eps},  pi0(eta) = P(T + eta in A0).

Because T and eta are independent and separately observed after the first stage, U(A)
is estimated by the pairwise (two-sample V-statistic) average

    U_hat(A) = (n_u n_x)^{-1} sum_i sum_j I_i {alpha_hat(eta_i) - c + beta2_hat'X_j} 1{T_j + eta_i in A},

with i in the evaluation fold and j in the index fold (``manuscript/pairwise_theory.tex``
gives the one-score version). No density of T is estimated. The roles follow the paper:
first stage (OLS per score), outcome (stacked spline fit, alpha and b functions of
(eta_1, eta_2), tensor-product or additive), index (T_j, X_j, and pi0), evaluation.

Rules are written on sign-adjusted scores s * Q (s = +1 or -1 per score), so that every
rule treats high values:
  * ``halfplane``: lam (s1 Q1) + (1 - lam)(s2 Q2) >= phi   (a weighted average, theta = (lam, phi));
  * ``both``:      s1 Q1 >= phi1 and s2 Q2 >= phi2          (theta = (phi1, phi2));
  * ``either``:    s1 Q1 >= phi1 or  s2 Q2 >= phi2.
U_hat over a grid of theta is computed at once: the pairwise sum is a convolution of the
(linearly binned) weighted point masses of eta and T, followed by a survival transform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

ROLES = ("gamma", "outcome", "index", "utility")


# ---------------------------------------------------------------- rules

@dataclass(frozen=True)
class Rule:
    kind: str                         # "halfplane" | "both" | "either"
    theta: Tuple[float, float]        # (lam, phi) or (phi1, phi2), on sign-adjusted scores
    signs: Tuple[int, int] = (1, 1)

    def treat_adj(self, Z: np.ndarray) -> np.ndarray:
        """Treatment for sign-adjusted points Z (n, 2)."""
        a, b = self.theta
        if self.kind == "halfplane":
            return a * Z[:, 0] + (1.0 - a) * Z[:, 1] >= b
        if self.kind == "both":
            return (Z[:, 0] >= a) & (Z[:, 1] >= b)
        if self.kind == "either":
            return (Z[:, 0] >= a) | (Z[:, 1] >= b)
        raise ValueError(self.kind)

    def treat(self, Q: np.ndarray) -> np.ndarray:
        return self.treat_adj(Q * np.asarray(self.signs, float))


# ---------------------------------------------------------------- binning helpers

def _bin1(x, w, lo, delta, size):
    """Linear (cloud-in-cell) binning of weighted points on nodes lo + k delta."""
    f = (x - lo) / delta
    k = np.clip(np.floor(f).astype(int), 0, size - 2)
    r = np.clip(f - k, 0.0, 1.0)
    out = np.bincount(k, w * (1 - r), minlength=size)
    out += np.bincount(k + 1, w * r, minlength=size)
    return out[:size]


def _bin2(x, w, lo, delta, size):
    f0 = (x[:, 0] - lo[0]) / delta[0]
    f1 = (x[:, 1] - lo[1]) / delta[1]
    k0 = np.clip(np.floor(f0).astype(int), 0, size[0] - 2)
    k1 = np.clip(np.floor(f1).astype(int), 0, size[1] - 2)
    r0 = np.clip(f0 - k0, 0.0, 1.0)
    r1 = np.clip(f1 - k1, 0.0, 1.0)
    out = np.zeros(size[0] * size[1])
    for d0, w0 in ((0, 1 - r0), (1, r0)):
        for d1, w1 in ((0, 1 - r1), (1, r1)):
            out += np.bincount((k0 + d0) * size[1] + (k1 + d1), w * w0 * w1,
                               minlength=out.size)[:out.size]
    return out.reshape(size)


def _survival2(m):
    return np.flip(np.cumsum(np.cumsum(np.flip(m, (0, 1)), 0), 1), (0, 1))


def _survival1(m):
    return np.flip(np.cumsum(np.flip(m)))


# ---------------------------------------------------------------- pi0 (deployed rule)

def prob_in_rule(rule: Rule, e_adj: np.ndarray, T_adj: np.ndarray, G: int = 1024) -> np.ndarray:
    """pi(e) = share of index points T_j with T_j + e in the rule (sign-adjusted inputs)."""
    a, b = rule.theta
    if rule.kind == "halfplane":
        s = np.sort(a * T_adj[:, 0] + (1 - a) * T_adj[:, 1])
        v = a * e_adj[:, 0] + (1 - a) * e_adj[:, 1]
        return 1.0 - np.searchsorted(s, b - v, side="left") / len(s)
    # bivariate survival table of T on a fine grid, bilinear lookup
    lo = T_adj.min(0)
    hi = T_adj.max(0)
    delta = (hi - lo) / (G - 1)
    size = (G, G)
    S = _survival2(_bin2(T_adj, np.full(len(T_adj), 1.0 / len(T_adj)), lo, delta, size))
    S = np.pad(S, ((0, 1), (0, 1)))          # zero beyond the top node

    def surv(qa, qb):
        f0 = np.clip((qa - lo[0]) / delta[0], 0, G)
        f1 = np.clip((qb - lo[1]) / delta[1], 0, G)
        k0 = np.minimum(np.floor(f0).astype(int), G - 1)
        k1 = np.minimum(np.floor(f1).astype(int), G - 1)
        r0, r1 = f0 - k0, f1 - k1
        return ((1 - r0) * (1 - r1) * S[k0, k1] + r0 * (1 - r1) * S[k0 + 1, k1]
                + (1 - r0) * r1 * S[k0, k1 + 1] + r0 * r1 * S[k0 + 1, k1 + 1])

    qa, qb = a - e_adj[:, 0], b - e_adj[:, 1]
    both = surv(qa, qb)
    if rule.kind == "both":
        return both
    m1 = surv(qa, np.full_like(qb, -np.inf))
    m2 = surv(np.full_like(qa, -np.inf), qb)
    return m1 + m2 - both


# ---------------------------------------------------------------- splines

def _basis1(x, lo, hi, m):
    t = np.concatenate([np.repeat(lo, 4), np.linspace(lo, hi, m + 2)[1:-1], np.repeat(hi, 4)])
    return BSpline.design_matrix(np.clip(x, lo, hi), t, 3).toarray()


def n_interior(n_min: int) -> int:
    """Interior knots per coordinate (fixed rule for this exploratory code)."""
    return max(2, int(round(n_min ** (1.0 / 7.0))))


@dataclass
class EtaBasis:
    box: np.ndarray                  # (2, 2): [[lo1, hi1], [lo2, hi2]]
    m: int
    form: str                        # "tensor" | "additive"

    def __call__(self, e):
        B1 = _basis1(e[:, 0], *self.box[0], self.m)
        B2 = _basis1(e[:, 1], *self.box[1], self.m)
        if self.form == "tensor":
            return (B1[:, :, None] * B2[:, None, :]).reshape(len(e), -1)
        return np.column_stack([B1, B2[:, 1:]])


@dataclass
class OutcomeFit:
    basis: EtaBasis
    w0: np.ndarray
    w1: np.ndarray
    d0: np.ndarray
    d1: np.ndarray
    pen: float = 0.0

    def alpha(self, e):
        return self.basis(e) @ (self.w1 - self.w0)

    @property
    def beta2(self):
        return self.d1 - self.d0


def _diff_penalty(m: int, form: str) -> np.ndarray:
    """Second-difference (P-spline) penalty on the spline coefficients of one arm."""
    k = m + 4
    D2 = np.diff(np.eye(k), 2, axis=0)
    P1 = D2.T @ D2
    if form == "tensor":
        return np.kron(P1, np.eye(k)) + np.kron(np.eye(k), P1)
    P = np.zeros((2 * k - 1, 2 * k - 1))
    P[:k, :k] = P1
    P[k:, k:] = P1[1:, 1:]
    return P


PEN_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def _penalized(ZtZ, ZtY, P, nb, p, pen):
    A = ZtZ.copy()
    if pen > 0:
        for start in (0, nb + p):
            sl = slice(start, start + nb)
            A[sl, sl] += pen * float(np.mean(np.diag(ZtZ)[sl])) * P
    return np.linalg.lstsq(A, ZtY, rcond=None)[0]


def outcome_fit(e, X, D, Y, form="tensor", m: Optional[int] = None,
                pen="cv", cv_folds: int = 5, seed: int = 0) -> OutcomeFit:
    """Arm-wise stacked spline fit: columns ((1-D)B, (1-D)X, DB, DX).

    In two dimensions some tensor cells hold data from one arm only (the deployed rule
    separates the arms), so each arm's spline coefficients get a second-difference
    (P-spline) penalty, pen x (mean diagonal of that arm's Gram block). pen = 0 gives
    least squares; pen = "cv" picks pen from PEN_GRID by ``cv_folds``-fold
    cross-validation of the squared prediction error of Y."""
    box = np.column_stack([np.quantile(e, 0.005, axis=0), np.quantile(e, 0.995, axis=0)])
    if m is None:
        n1 = int(D.sum())
        m = n_interior(min(n1, len(D) - n1))
    basis = EtaBasis(box=box, m=m, form=form)
    B = basis(e)
    Dc = D[:, None]
    Z = np.column_stack(((1 - Dc) * B, (1 - Dc) * X, Dc * B, Dc * X))
    nb, p = B.shape[1], X.shape[1]
    ZtZ, ZtY = Z.T @ Z, Z.T @ Y
    P = _diff_penalty(m, form)
    if pen == "cv":
        k = np.random.default_rng(seed).integers(0, cv_folds, len(Y))
        parts = [(Z[k == f], Y[k == f]) for f in range(cv_folds)]
        sse = np.zeros(len(PEN_GRID))
        for Zf, Yf in parts:
            A, b = ZtZ - Zf.T @ Zf, ZtY - Zf.T @ Yf
            for g, pg in enumerate(PEN_GRID):
                sse[g] += float(np.sum((Yf - Zf @ _penalized(A, b, P, nb, p, pg)) ** 2))
        pen = PEN_GRID[int(np.argmin(sse))]
    coef = _penalized(ZtZ, ZtY, P, nb, p, float(pen))
    return OutcomeFit(basis, coef[:nb], coef[nb + p:2 * nb + p], coef[nb:nb + p],
                      coef[2 * nb + p:], pen=float(pen))


# ---------------------------------------------------------------- utility surfaces

@dataclass
class Pairs:
    """Evaluation-side and index-side pieces of the pairwise utility (sign-adjusted)."""
    e: np.ndarray          # (n_u, 2) eta of the evaluation fold (trimmed units have weight 0)
    u: np.ndarray          # I_i (alpha(eta_i) - c)
    I: np.ndarray          # I_i
    T: np.ndarray          # (n_x, 2) index-fold T
    x: np.ndarray          # beta2'X_j
    n_u: int
    n_x: int
    _cache: Dict = field(default_factory=dict, repr=False)

    def value(self, rule: Rule) -> float:
        """Exact U_hat(rule) for one rule (O(n_u n_x) only for 'both'/'either'; use sparingly)."""
        keep = self.I > 0
        e, u, I = self.e[keep], self.u[keep], self.I[keep]
        if rule.kind == "halfplane":
            lam, phi = rule.theta
            s = lam * self.T[:, 0] + (1 - lam) * self.T[:, 1]
            o = np.argsort(s)
            s, cx = s[o], np.concatenate([np.cumsum(self.x[o][::-1])[::-1], [0.0]])
            k = np.searchsorted(s, phi - (lam * e[:, 0] + (1 - lam) * e[:, 1]), side="left")
            return float(np.sum(u * (len(s) - k) + I * cx[k]) / (self.n_u * self.n_x))
        a, b = rule.theta
        tot = 0.0
        for c0 in range(0, len(e), 500):
            ee = e[c0:c0 + 500]
            Z0 = self.T[None, :, 0] + ee[:, 0:1]
            Z1 = self.T[None, :, 1] + ee[:, 1:2]
            inA = (Z0 >= a) & (Z1 >= b) if rule.kind == "both" else (Z0 >= a) | (Z1 >= b)
            tot += float(u[c0:c0 + 500] @ inA.sum(1) + I[c0:c0 + 500] @ (inA @ self.x))
        return tot / (self.n_u * self.n_x)

    # -- 2-D surfaces for 'both' and 'either' -----------------------------------------
    def surface2(self, G: int = 512):
        if G not in self._cache:
            self._cache[G] = self._surface2(G)
        return self._cache[G]

    def _surface2(self, G: int):
        keep = self.I > 0
        e, u, I = self.e[keep], self.u[keep], self.I[keep]
        lo_e, hi_e = e.min(0), e.max(0)
        lo_t, hi_t = self.T.min(0), self.T.max(0)
        delta = np.maximum(hi_e - lo_e, hi_t - lo_t) / (G - 1)
        se = (np.ceil((hi_e - lo_e) / delta).astype(int) + 2)
        st = (np.ceil((hi_t - lo_t) / delta).astype(int) + 2)
        He_u = _bin2(e, u, lo_e, delta, tuple(se))
        He_I = _bin2(e, I, lo_e, delta, tuple(se))
        Ht_1 = _bin2(self.T, np.ones(self.n_x), lo_t, delta, tuple(st))
        Ht_x = _bin2(self.T, self.x, lo_t, delta, tuple(st))
        conv = fftconvolve(He_u, Ht_1) + fftconvolve(He_I, Ht_x)
        S = _survival2(conv) / (self.n_u * self.n_x)
        # a node's mass stands for the cell around it, so the survival at node k is the
        # survival at node k - delta/2
        nodes = [lo_e[k] + lo_t[k] + delta[k] * (np.arange(S.shape[k]) - 0.5) for k in (0, 1)]
        return nodes, S

    # -- 1-D profile for a halfplane with weight lam ---------------------------------
    def curve1(self, lam: float, G: int = 4096):
        keep = self.I > 0
        v = lam * self.e[keep, 0] + (1 - lam) * self.e[keep, 1]
        s = lam * self.T[:, 0] + (1 - lam) * self.T[:, 1]
        delta = max(np.ptp(v), np.ptp(s)) / (G - 1)
        sv = int(np.ceil(np.ptp(v) / delta)) + 2
        ss = int(np.ceil(np.ptp(s) / delta)) + 2
        conv = (fftconvolve(_bin1(v, self.u[keep], v.min(), delta, sv), _bin1(s, np.ones(self.n_x), s.min(), delta, ss))
                + fftconvolve(_bin1(v, self.I[keep], v.min(), delta, sv), _bin1(s, self.x, s.min(), delta, ss)))
        S = _survival1(conv) / (self.n_u * self.n_x)
        return v.min() + s.min() + delta * (np.arange(len(S)) - 0.5), S


# ---------------------------------------------------------------- maximizers

def _quad_peak_1d(x, y, k):
    if 0 < k < len(x) - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            off = 0.5 * (y0 - y2) / den
            h = x[1] - x[0]
            return x[k] + off * h, y1 - 0.25 * (y0 - y2) * off
    return x[k], y[k]


def _quad_peak_2d(n0, n1, Z, k0, k1, r: int = 2):
    i0 = np.arange(max(k0 - r, 0), min(k0 + r + 1, len(n0)))
    i1 = np.arange(max(k1 - r, 0), min(k1 + r + 1, len(n1)))
    if len(i0) < 3 or len(i1) < 3:
        return np.array([n0[k0], n1[k1]]), Z[k0, k1], False
    A0, A1 = np.meshgrid(n0[i0] - n0[k0], n1[i1] - n1[k1], indexing="ij")
    z = Z[np.ix_(i0, i1)].ravel()
    a0, a1 = A0.ravel(), A1.ravel()
    M = np.column_stack([np.ones_like(a0), a0, a1, 0.5 * a0 ** 2, a0 * a1, 0.5 * a1 ** 2])
    c = np.linalg.lstsq(M, z, rcond=None)[0]
    H = np.array([[c[3], c[4]], [c[4], c[5]]])
    g = c[1:3]
    if np.all(np.linalg.eigvalsh(H) < 0):
        d = -np.linalg.solve(H, g)
        if abs(d[0]) <= r * (n0[1] - n0[0]) and abs(d[1]) <= r * (n1[1] - n1[0]):
            return np.array([n0[k0] + d[0], n1[k1] + d[1]]), float(c[0] + 0.5 * g @ d), True
    return np.array([n0[k0], n1[k1]]), Z[k0, k1], False


def best_two_cutoffs(pairs: Pairs, kind: str, box, G: int = 512) -> Dict:
    """Maximize U_hat over (phi1, phi2) in box = ((lo1, hi1), (lo2, hi2)) for 'both' or 'either'."""
    (n0, n1), S = pairs.surface2(G)
    if kind == "either":
        Z = S[:, :1] + S[:1, :] - S
    else:
        Z = S
    m0 = (n0 >= box[0][0]) & (n0 <= box[0][1])
    m1 = (n1 >= box[1][0]) & (n1 <= box[1][1])
    n0, n1, Z = n0[m0], n1[m1], Z[np.ix_(m0, m1)]
    k0, k1 = np.unravel_index(int(np.argmax(Z)), Z.shape)
    theta, val, interior = _quad_peak_2d(n0, n1, Z, k0, k1)
    edge = k0 in (0, len(n0) - 1) or k1 in (0, len(n1) - 1)
    return {"theta": theta.tolist(), "U": float(val), "edge": bool(edge),
            "grid": (n0, n1, Z)}


def best_halfplane(pairs: Pairs, lam_range, phi_box, lam_fixed: Optional[float] = None,
                   n_lam: int = 41, G: int = 4096) -> Dict:
    """Maximize U_hat over (lam, phi); lam_fixed gives the fixed-weight (one-cutoff) problem."""
    def profile(lam):
        x, S = pairs.curve1(lam, G)
        m = (x >= phi_box[0]) & (x <= phi_box[1])
        x, S = x[m], S[m]
        k = int(np.argmax(S))
        phi, val = _quad_peak_1d(x, S, k)
        return float(phi), float(val), k in (0, len(x) - 1)

    if lam_fixed is not None:
        phi, val, edge = profile(lam_fixed)
        return {"theta": [lam_fixed, phi], "U": val, "edge": edge}
    lams = np.linspace(lam_range[0], lam_range[1], n_lam)
    prof = [profile(l) for l in lams]
    vals = np.array([p[1] for p in prof])
    k = int(np.argmax(vals))
    lo, hi = lams[max(k - 1, 0)], lams[min(k + 1, n_lam - 1)]
    res = minimize_scalar(lambda l: -profile(l)[1], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    lam = float(res.x)
    phi, val, edge = profile(lam)
    return {"theta": [lam, phi], "U": val, "edge": bool(edge or k in (0, n_lam - 1)),
            "profile": (lams, vals)}


# ---------------------------------------------------------------- the estimator

@dataclass
class TwoScoreProblem:
    Q: np.ndarray                 # (n, 2) raw scores
    X: np.ndarray                 # (n, p) covariates
    Y: np.ndarray
    deployed: Rule                # on sign-adjusted scores
    cost: float = 0.0
    eps: float = 0.05
    signs: Tuple[int, int] = (1, 1)

    def __post_init__(self):
        self.D = self.deployed.treat(self.Q).astype(float)
        self.Xc = self.X - self.X.mean(0)
        self.Z = self.Q * np.asarray(self.signs, float)


def _ols(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


def rotation_pairs(prob: TwoScoreProblem, folds: Dict[str, np.ndarray], form: str = "tensor",
                   m: Optional[int] = None, pen="cv") -> Tuple[Pairs, OutcomeFit, Dict]:
    """First stage, trim, outcome fit and the pairwise pieces for one role assignment."""
    Xt = np.column_stack([np.ones(len(prob.Xc)), prob.Xc])
    Gam = _ols(Xt[folds["gamma"]], prob.Z[folds["gamma"]])
    T = Xt @ Gam
    eta = prob.Z - T
    T_idx = T[folds["index"]]
    # trim: pi0 from the index fold; the outcome fit uses a wider band
    po = prob_in_rule(prob.deployed, eta[folds["outcome"]], T_idx)
    keep_o = (po >= prob.eps / 2) & (po <= 1 - prob.eps / 2)
    o = folds["outcome"][keep_o]
    fit = outcome_fit(eta[o], prob.Xc[o], prob.D[o], prob.Y[o], form=form, m=m, pen=pen)
    pu = prob_in_rule(prob.deployed, eta[folds["utility"]], T_idx)
    I = ((pu >= prob.eps) & (pu <= 1 - prob.eps)).astype(float)
    e_u = eta[folds["utility"]]
    u = np.zeros(len(e_u))
    u[I > 0] = fit.alpha(e_u[I > 0]) - prob.cost
    pairs = Pairs(e=e_u, u=u * I, I=I, T=T_idx, x=prob.Xc[folds["index"]] @ fit.beta2,
                  n_u=len(e_u), n_x=len(T_idx))
    info = {"gamma": Gam, "trim_share": float(I.mean()), "m": fit.basis.m, "pen": fit.pen}
    return pairs, fit, info


def make_folds(n: int, seed: int):
    rng = np.random.default_rng(seed)
    blocks = np.array_split(rng.permutation(n), 4)
    return [{role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)} for k in range(4)]


def estimate(prob: TwoScoreProblem, classes: Dict[str, Dict], seed: int = 0,
             form: str = "tensor", m: Optional[int] = None, rotations: int = 4,
             pen="cv") -> Dict:
    """Rotated estimator: per rotation, the argmax for each rule class; then the average.

    ``classes`` maps a label to {"kind": "halfplane"|"both"|"either", "box": ...,
    "lam_range": ... (halfplane), "lam_fixed": ... (optional)}. Also returns U_hat of the
    deployed rule and of each per-rotation optimum (on that rotation's pairs)."""
    per = {k: [] for k in classes}
    U_dep = []
    infos = []
    for folds in make_folds(len(prob.Y), seed)[:rotations]:
        pairs, fit, info = rotation_pairs(prob, folds, form=form, m=m, pen=pen)
        infos.append(info)
        U_dep.append(pairs.value(prob.deployed) if prob.deployed.kind == "halfplane"
                     else _value_from_grid(pairs, prob.deployed))
        for label, spec in classes.items():
            if spec["kind"] == "halfplane":
                r = best_halfplane(pairs, spec.get("lam_range", (0.0, 1.0)), spec["box"],
                                   lam_fixed=spec.get("lam_fixed"))
            else:
                r = best_two_cutoffs(pairs, spec["kind"], spec["box"])
            r.pop("grid", None)
            r.pop("profile", None)
            per[label].append(r)
    out = {"U_deployed": float(np.mean(U_dep)), "trim_share": float(np.mean([i["trim_share"] for i in infos])),
           "m": infos[0]["m"], "pen": [i["pen"] for i in infos]}
    for label, rs in per.items():
        th = np.array([r["theta"] for r in rs])
        out[label] = {"theta": th.mean(0).tolist(), "theta_rot": th.tolist(),
                      "U": float(np.mean([r["U"] for r in rs])),
                      "edge_rotations": int(sum(r["edge"] for r in rs))}
    return out


def _value_from_grid(pairs: Pairs, rule: Rule) -> float:
    """U_hat at one 'both'/'either' rule, read from the binned surface."""
    (n0, n1), S = pairs.surface2()
    Z = S[:, :1] + S[:1, :] - S if rule.kind == "either" else S
    from scipy.interpolate import RegularGridInterpolator
    return float(RegularGridInterpolator((n0, n1), Z)([rule.theta])[0])
