"""Optimal dose schedule from a fixed score (Model 2 of the continuous-dose draft).

Model, for doses a in the fitted range (manuscript ``dose_schedule_sketch.tex``):

    E{Y(a) | X, eta} = beta'X + g(eta) + f(eta, a) + (a - a_ref) tau'Z,   f(e, a_ref) = 0,

with score Q = gamma'(1, X) + eta and historical dose A = s(Q, group). ``a_ref`` is the
dose of the flat region that plays the untreated role (0, or a benefit floor).

Estimator, one role rotation:
  1. fold F1: OLS of Q on (1, X); eta_hat = Q - gamma_hat'(1, X).
  2. fold F2, units with A in the fitted range: OLS of Y on P_K(eta_hat), X, the tensor
     P_k(eta_hat) x B(A), and (A - a_ref) Z. For a continuous dose, B is a clamped cubic
     B-spline basis on [a_ref, a_bar] without its first function, so f(e, a_ref) = 0; for
     a step schedule, B is the indicators of the non-reference levels.
  3. fold F3: every unit's predicted gain at every dose,
     W_i(a) = f_hat(eta_hat_i, a) + (a - a_ref) tau_hat'Z_i, averaged within score bins:
     psi_hat(q, a). The three rotations are averaged.
  4. d_hat(q) = argmax_a psi_hat(q, a) - c(a).

Also: ``readiness`` (support diagnostics), ``modifier_test`` (adds (A - a_ref) T_hat,
T_hat = gamma_hat'(1, X), the score-direction modifier that a one-to-one schedule cannot
identify), and ``best_linear`` (the best linear schedule with a tapered weight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

ROTATIONS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))      # (first stage, outcome fit, evaluation)


def bspline(x, lo, hi, m):
    """Clamped cubic B-spline basis with m interior knots on [lo, hi]; x is clipped."""
    t = np.concatenate([np.repeat(lo, 4), np.linspace(lo, hi, m + 2)[1:-1], np.repeat(hi, 4)])
    return BSpline.design_matrix(np.clip(np.asarray(x, float), lo, hi), t, 3).toarray()


@dataclass
class DoseBasis:
    """B(a) with B(a_ref) = 0: splines on [a_lo, a_bar], or indicators of the levels.

    With a_lo = a_ref (the default) the clamped basis without its first function vanishes at
    a_ref. With a_lo < a_ref (doses below the reference), the basis is B(a) - B(a_ref) without
    its first column; the full set would be linearly dependent, since B-splines sum to one.
    """
    a_ref: float
    a_bar: float
    l: int = 3
    levels: Optional[np.ndarray] = None       # non-reference doses of a step schedule
    a_lo: Optional[float] = None

    def __call__(self, a):
        a = np.atleast_1d(np.asarray(a, float))
        if self.levels is not None:
            return (np.abs(a[:, None] - self.levels[None, :]) < 1e-9).astype(float)
        lo = self.a_ref if self.a_lo is None else self.a_lo
        B = bspline(a, lo, self.a_bar, self.l)
        if lo < self.a_ref:
            B = B - bspline(np.array([self.a_ref]), lo, self.a_bar, self.l)
        return B[:, 1:]

    def grid(self, size: int = 61):
        if self.levels is not None:
            return np.sort(np.append(self.levels, self.a_ref))
        lo = self.a_ref if self.a_lo is None else self.a_lo
        return np.linspace(lo, self.a_bar, size)


@dataclass
class DoseProblem:
    Q: np.ndarray
    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    Z: np.ndarray                              # effect modifiers (n, r); may have r = 0
    basis: DoseBasis
    a_lo: float                                # fitted dose range [a_lo, a_bar]

    @property
    def in_range(self):
        return (self.A >= self.a_lo - 1e-9) & (self.A <= self.basis.a_bar + 1e-9)


def make_folds(n, seed=0, cluster=None):
    """Three folds; with ``cluster`` (e.g. an entity id in a panel), whole clusters share a fold."""
    rng = np.random.default_rng(seed)
    if cluster is None:
        return rng.permutation(n) % 3
    _, inv = np.unique(cluster, return_inverse=True)
    return (rng.permutation(inv.max() + 1) % 3)[inv]


def first_stage(Q, X, idx):
    Xt = np.column_stack([np.ones(len(Q)), X])
    gam = np.linalg.lstsq(Xt[idx], Q[idx], rcond=None)[0]
    return gam, Q - Xt @ gam, Xt @ gam


def _hc1_se(D, resid, j, cluster=None):
    """Robust (HC1, or cluster-robust when ``cluster`` is given) standard error of coefficient j."""
    Qr, R = np.linalg.qr(D)
    Rinv = np.linalg.pinv(R)
    G = Qr * resid[:, None]
    if cluster is not None:
        _, inv = np.unique(cluster, return_inverse=True)
        Gc = np.zeros((inv.max() + 1, G.shape[1]))
        np.add.at(Gc, inv, G)
        G = Gc
    V = Rinv @ (G.T @ G) @ Rinv.T * len(resid) / max(len(resid) - D.shape[1], 1)
    return float(np.sqrt(max(V[j, j], 0.0)))


@dataclass
class OutcomeFit:
    ebox: tuple
    K: int
    k: int
    Wf: np.ndarray                             # (k-basis, dose-basis) coefficients of f
    tau: np.ndarray
    basis: DoseBasis
    extra: Dict

    def W(self, eta):
        """Rows give f_hat(eta_i, .) in dose-basis coordinates."""
        return bspline(eta, *self.ebox, self.k) @ self.Wf

    def f(self, eta, a):
        return np.sum(self.W(eta) * self.basis(a), axis=1)


def outcome_fit(prob: DoseProblem, eta, idx, K, k, extra_cols=None, cluster=None) -> OutcomeFit:
    use = idx & prob.in_range
    e = eta[use]
    ebox = tuple(np.quantile(eta[prob.in_range], [0.005, 0.995]))
    PK, Pk = bspline(e, *ebox, K), bspline(e, *ebox, k)
    B = prob.basis(prob.A[use])
    T = (Pk[:, :, None] * B[:, None, :]).reshape(use.sum(), -1)
    dA = (prob.A[use] - prob.basis.a_ref)[:, None]
    blocks = [PK, prob.X[use], T, dA * prob.Z[use]]
    if extra_cols is not None:
        blocks.append(extra_cols[use])
    D = np.column_stack(blocks)
    coef, *_ = np.linalg.lstsq(D, prob.Y[use], rcond=None)
    nP, p = PK.shape[1], prob.X.shape[1]
    nT, r = T.shape[1], prob.Z.shape[1]
    Wf = coef[nP + p:nP + p + nT].reshape(Pk.shape[1], B.shape[1])
    tau = coef[nP + p + nT:nP + p + nT + r]
    extra = {}
    if extra_cols is not None:
        j0 = nP + p + nT + r
        resid = prob.Y[use] - D @ coef
        cl = None if cluster is None else cluster[use]
        extra = {"coef": coef[j0:], "se": np.array([_hc1_se(D, resid, j, cl) for j in range(j0, D.shape[1])]),
                 "n_fit": int(use.sum())}
    return OutcomeFit(ebox, K, k, Wf, tau, prob.basis, extra)


def bin_edges(Q, n_bins):
    return np.unique(np.quantile(Q, np.linspace(0, 1, n_bins + 1)))


def psi_bins(fit: OutcomeFit, eta, Z, Q, edges, a_grid, a_ref):
    """psi_hat(bin, a): bin mean of f_hat(eta_i, a) + (a - a_ref) tau_hat'Z_i (linear in W)."""
    b = np.clip(np.searchsorted(edges, Q, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    cnt = np.bincount(b, minlength=nb).astype(float)
    W = fit.W(eta)
    Wbar = np.zeros((nb, W.shape[1]))
    np.add.at(Wbar, b, W)
    zt = Z @ fit.tau if Z.shape[1] else np.zeros(len(Q))
    zbar = np.bincount(b, weights=zt, minlength=nb)
    c = np.maximum(cnt, 1)[:, None]
    return (Wbar / c) @ fit.basis(a_grid).T + (zbar / c[:, 0])[:, None] * (a_grid - a_ref)[None, :], cnt


def estimate(prob: DoseProblem, n_bins=20, seed=0, K=None, k=None, a_grid=None, rotations=ROTATIONS,
             score_edges=None, cluster=None):
    """Rotation-averaged psi_hat on score bins, plus per-rotation copies."""
    n = len(prob.Y)
    fold = make_folds(n, seed, cluster)
    n_fit = (prob.in_range & (fold == 1)).sum()
    K = K or max(3, int(round(n_fit ** 0.2)))
    k = k or max(3, int(round(n_fit ** (1 / 7))))
    a_grid = prob.basis.grid() if a_grid is None else a_grid
    edges = bin_edges(prob.Q, n_bins) if score_edges is None else score_edges
    out = []
    for r0, r1, r2 in rotations:
        _, eta, _ = first_stage(prob.Q, prob.X, fold == r0)
        fit = outcome_fit(prob, eta, fold == r1, K, k)
        ev = fold == r2
        psi, cnt = psi_bins(fit, eta[ev], prob.Z[ev], prob.Q[ev], edges, a_grid, prob.basis.a_ref)
        out.append((psi, fit))
    psi = np.mean([o[0] for o in out], axis=0)
    return {"psi": psi, "psi_rot": np.array([o[0] for o in out]), "fits": [o[1] for o in out],
            "edges": edges, "a_grid": a_grid, "K": K, "k": k}


def coverage(eta, A, Q, edges, a_grid, a_ref, levels=None, band=0.05, probs=(0.025, 0.975)):
    """Share of each score bin's units whose eta lies inside the eta range of units given dose a.

    The transported psi_hat(q, a) is an extrapolation where this share is low: the dose a was
    only given to units whose latent part differs from that of the units at score q. For a
    continuous dose, "given a" means |A - a| <= band * (range of the observed doses).
    """
    b = np.clip(np.searchsorted(edges, Q, side="right") - 1, 0, len(edges) - 2)
    width = band * (np.max(A) - np.min(A)) if levels is None else 0.0
    out = np.zeros((len(edges) - 1, len(a_grid)))
    for j, a in enumerate(a_grid):
        m = np.abs(A - a) <= max(width, 1e-9)
        if m.sum() < 30:
            continue
        lo, hi = np.quantile(eta[m], probs)
        inside = (eta >= lo) & (eta <= hi)
        out[:, j] = np.bincount(b, weights=inside, minlength=len(edges) - 1) / np.maximum(
            np.bincount(b, minlength=len(edges) - 1), 1)
    return out


def gain(fit: OutcomeFit, eta, Z, a, a_ref):
    """Unit-level f_hat(eta_i, a_i) + (a_i - a_ref) tau_hat'Z_i for a vector of doses a."""
    zt = Z @ fit.tau if Z.shape[1] else np.zeros(len(eta))
    return fit.f(eta, a) + (np.asarray(a, float) - a_ref) * zt


def support_share(eta_fit, A_fit, eta_eval, a_new, n_cells=20, probs=(0.025, 0.975)):
    """Share of units whose new dose lies in a dose cell where their eta is inside the 1-99% eta range."""
    edges = np.unique(np.quantile(A_fit, np.linspace(0, 1, n_cells + 1)))
    lo = np.full(len(edges) - 1, np.nan)
    hi = np.full(len(edges) - 1, np.nan)
    cell = np.clip(np.searchsorted(edges, A_fit, side="right") - 1, 0, len(edges) - 2)
    for j in range(len(edges) - 1):
        m = cell == j
        if m.sum() >= 30:
            lo[j], hi[j] = np.quantile(eta_fit[m], probs)
    cn = np.clip(np.searchsorted(edges, a_new, side="right") - 1, 0, len(edges) - 2)
    return float(np.mean((eta_eval >= lo[cn]) & (eta_eval <= hi[cn])))


def family_value(prob: DoseProblem, dose_fn: Callable, thetas, seed=0, cluster=None, K=None, k=None,
                 cost: Optional[Callable] = None, weight=None):
    """Change in mean outcome (net of cost) from the status-quo doses to d_theta, per rotation.

    dose_fn(theta, Q, i_eval) gives the new dose of each evaluation unit. Returns (values, support):
    arrays (3, len(thetas)) with the evaluation-fold mean of gain(new) - gain(old) - cost(new) +
    cost(old), and the share of units whose new dose is inside the eta support of that dose.
    """
    n = len(prob.Y)
    fold = make_folds(n, seed, cluster)
    n_fit = (prob.in_range & (fold == 1)).sum()
    K = K or max(3, int(round(n_fit ** 0.2)))
    k = k or max(3, int(round(n_fit ** (1 / 7))))
    cost = cost or (lambda a: 0.0 * np.asarray(a, float))
    w = np.ones(n) if weight is None else weight
    vals, sup = np.zeros((3, len(thetas))), np.zeros((3, len(thetas)))
    a_ref = prob.basis.a_ref
    for r, (r0, r1, r2) in enumerate(ROTATIONS):
        _, eta, _ = first_stage(prob.Q, prob.X, fold == r0)
        fit = outcome_fit(prob, eta, fold == r1, K, k)
        ev = np.where((fold == r2) & prob.in_range)[0]
        fi = np.where((fold == r1) & prob.in_range)[0]
        g0 = gain(fit, eta[ev], prob.Z[ev], prob.A[ev], a_ref) - cost(prob.A[ev])
        for j, th in enumerate(thetas):
            a1 = dose_fn(th, prob.Q[ev], ev)
            g1 = gain(fit, eta[ev], prob.Z[ev], a1, a_ref) - cost(a1)
            vals[r, j] = np.sum(w[ev] * (g1 - g0)) / np.sum(w[ev])
            sup[r, j] = support_share(eta[fi], prob.A[fi], eta[ev], a1)
    return vals, sup


def d_hat(psi, a_grid, cost: Callable):
    return a_grid[np.argmax(psi - cost(a_grid)[None, :], axis=1)]


def readiness(Q, X, A, a_ref, levels=None, n_dec=10):
    """R^2 of Q on X, sd(eta), share at a_ref, and eta quantiles by dose level / decile."""
    _, eta, T = first_stage(Q, X, np.ones(len(Q), bool))
    at_ref = np.abs(A - a_ref) < 1e-9
    rows = []
    if levels is None:
        pos = ~at_ref
        cuts = np.quantile(A[pos], np.linspace(0, 1, n_dec + 1)) if pos.any() else []
        groups = [("ref", at_ref)] + [(f"d{j + 1}:{cuts[j]:.3g}-{cuts[j + 1]:.3g}",
                                       pos & (A >= cuts[j]) & (A <= cuts[j + 1])) for j in range(len(cuts) - 1)]
    else:
        groups = [(f"{a:g}", np.abs(A - a) < 1e-9) for a in np.sort(np.unique(np.append(levels, a_ref)))]
    for name, m in groups:
        if m.sum():
            rows.append((name, int(m.sum()), *np.quantile(eta[m], [0.1, 0.5, 0.9])))
    return {"n": len(Q), "R2": float(np.var(T) / np.var(Q)), "sd_eta": float(eta.std()),
            "sd_T": float(T.std()), "share_ref": float(at_ref.mean()), "by_dose": rows}


def modifier_test(prob: DoseProblem, seed=0, K=None, k=None, cluster=None):
    """Coefficient on (A - a_ref) T_hat, T_hat = gamma_hat'(1, X): zero under Model 2.

    With a one-to-one schedule, T = s^{-1}(A) - eta is a function of (eta, A), so this column is
    absorbed by f and the test has no power (``r2_absorbed`` near 1). Steps or several schedules
    make it identifiable.
    """
    n = len(prob.Y)
    fold = make_folds(n, seed, cluster)
    gam, eta, _ = first_stage(prob.Q, prob.X, fold == 0)
    T = np.column_stack([np.ones(n), prob.X]) @ gam
    col = ((prob.A - prob.basis.a_ref) * (T - T.mean()))[:, None]
    use = fold != 0
    n_fit = (prob.in_range & use).sum()
    K = K or max(3, int(round(n_fit ** 0.2)))
    k = k or max(3, int(round(n_fit ** (1 / 7))))
    fit = outcome_fit(prob, eta, use, K, k, extra_cols=col, cluster=cluster)
    # how much of the column the rest of the design explains
    u = use & prob.in_range
    e = eta[u]
    ebox = fit.ebox
    Pk = bspline(e, *ebox, k)
    B = prob.basis(prob.A[u])
    Tm = (Pk[:, :, None] * B[:, None, :]).reshape(u.sum(), -1)
    D0 = np.column_stack([bspline(e, *ebox, K), prob.X[u], Tm, (prob.A[u] - prob.basis.a_ref)[:, None] * prob.Z[u]])
    cu = col[u, 0]
    resid = cu - D0 @ np.linalg.lstsq(D0, cu, rcond=None)[0]
    r2 = 1 - resid.var() / cu.var()
    return {"coef": float(fit.extra["coef"][0]), "se": float(fit.extra["se"][0]),
            "r2_absorbed": float(r2), "n_fit": fit.extra["n_fit"], "sd_T": float(T.std())}


def taper(q, lo, hi):
    inside = (q >= lo) & (q <= hi)
    return np.where(inside, np.sin(np.pi * (q - lo) / (hi - lo)) ** 2, 0.0)


def best_linear(prob: DoseProblem, cost: Callable, window, seed=0, K=None, k=None, q_bar=None, start=None,
                cluster=None):
    """Best schedule a_ref + clip(theta0 + theta1 (q - q_bar)) within the fitted range, tapered welfare.

    Returns theta_hat per rotation (no rotation averaging of the argmax)."""
    n = len(prob.Y)
    fold = make_folds(n, seed, cluster)
    n_fit = (prob.in_range & (fold == 1)).sum()
    K = K or max(3, int(round(n_fit ** 0.2)))
    k = k or max(3, int(round(n_fit ** (1 / 7))))
    lo, hi = window
    q_bar = 0.5 * (lo + hi) if q_bar is None else q_bar
    res = []
    for r0, r1, r2 in ROTATIONS:
        _, eta, _ = first_stage(prob.Q, prob.X, fold == r0)
        fit = outcome_fit(prob, eta, fold == r1, K, k)
        ev = (fold == r2) & (prob.Q >= lo) & (prob.Q <= hi)
        W, q, w = fit.W(eta[ev]), prob.Q[ev], taper(prob.Q[ev], lo, hi)
        zt = prob.Z[ev] @ fit.tau if prob.Z.shape[1] else np.zeros(ev.sum())
        a_ref, a_bar = prob.basis.a_ref, prob.basis.a_bar

        def negV(t):
            d = np.clip(t[0] + t[1] * (q - q_bar), a_ref, a_bar)
            return -np.mean(w * (np.sum(W * prob.basis(d), 1) + (d - a_ref) * zt - cost(d)))

        x0 = np.array(start if start is not None else [0.5 * (a_ref + a_bar), 0.0])
        r = minimize(negV, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=4000))
        res.append(r.x)
    return np.array(res)
