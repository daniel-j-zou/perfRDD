"""Binary outcomes: the routes of manuscript/binary_outcome_sketch.tex on real data.

Exploratory (Claude, 2026-11-10, at the author's request). The scope file keeps binary
outcomes outside the paper; nothing here changes the paper.

Settings with a binary outcome and data on this machine:
  * GPA (Lindo, Sanders and Oreopoulos 2010): graduation within 6 and 5 years,
    fall registration in year 2, not leaving; mirrored score as in gpa_four_role.py.
  * Chile PAES 2024 university-application eligibility (Q >= 458): university
    enrollment 2024, any enrollment 2025, university enrollment 2025.

Routes (one four-role estimator each, same folds, average of the four rotations):
  lpm        the paper's estimator on the binary Y (linear probability model);
  additive   logit baseline, additive effect on the probability scale (note, Section 3);
  logit_sp   logit on both arms; separable Chebyshev sieve in the link's intercept with
             the paper's spline tails (note, Section 5);
  logit_emp  the same logit fit with empirical tails on the index fold, which is the
             pairwise double average of the note's Section 4 computed through the sieve.

Run (from code/):
    python -m experiments.scripts.binary_outcome_routes --settings gpa chile --out experiments/runs/binary_outcome_routes
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
from numpy.polynomial import chebyshev as Cheb
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts import gpa_four_role as G
from experiments.scripts import taxi_four_role as M

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "binary_outcome_routes"
ROUTES = ("lpm", "additive", "logit_sp", "logit_emp")


def expit(u):
    return 1.0 / (1.0 + np.exp(-u))


# ----------------------------------------------------------------------------- data

@dataclass
class Data:
    Q: np.ndarray
    Z: np.ndarray          # score covariates (standardized); also the baseline
    X: np.ndarray          # effect covariates (standardized)
    Y: np.ndarray
    phi0: float
    Zb: np.ndarray | None = None

    @property
    def D(self):
        return (self.Q >= self.phi0).astype(float)

    @property
    def baseline(self):
        return self.Z if self.Zb is None else self.Zb

    def subset(self, idx):
        return Data(self.Q[idx], G._standardize(self.Z[idx]), G._standardize(self.X[idx]),
                    self.Y[idx], self.phi0, None if self.Zb is None else G._standardize(self.Zb[idx]))


def make_design(data: Data, eps: float, policy: tuple, tail: float = 0.0025,
                window_tail: float = 0.005, margin_sd: float = 0.7) -> M.Design:
    """As taxi_four_role.make_design, with the caps re-estimated on the evaluation fold
    and J stopping where each arm thins out (as in the GPA design)."""
    A = M._xt(data.Z)
    T = A @ np.linalg.lstsq(A, data.Q, rcond=None)[0]
    eta = data.Q - T
    D = data.D.astype(bool)
    l0 = data.phi0 - float(np.quantile(T, 1 - eps))
    u0 = data.phi0 - float(np.quantile(T, eps))
    sd = float(eta.std())
    J = (max(l0 - margin_sd * sd, float(np.quantile(eta[D], tail))),
         min(u0 + margin_sd * sd, float(np.quantile(eta[~D], 1 - tail))))
    pad = 0.5 * sd
    tlo, thi = np.quantile(T, [0.0005, 0.9995])
    T_support = (float(min(tlo, policy[0] - J[1]) - pad), float(max(thi, policy[1] - J[0]) + pad))
    window = (float(np.quantile(eta[D], window_tail)), float(np.quantile(eta[~D], 1 - window_tail)))
    return M.Design(J=J, eps=eps, phi0=data.phi0, T_support=T_support, policy=tuple(policy),
                    effect=True, window=window)


def build_gpa(y: str, score: str = "flex", effect: str = "hsgrade", eps: float = 0.1):
    df = G.load_frame()
    gd, _ = G.make_data(df, y, score, effect, "raw")
    data = Data(gd.Q, gd.Z, gd.X, gd.Y, gd.s0)
    design = G.make_design(gd, eps, "raw", effect=True)
    return data, design, {"n": len(data.Q), "cell": lambda s: G.dist_cell(G.to_dist(s, "raw"),
                                                                            df["dist_from_cut"].to_numpy(float))}


CHILE_POLICY = (420.0, 520.0)


def build_chile(y_col: str, effect: str = "nem_ranking", eps: float = 0.1):
    from experiments.datasets.chile_admission import adapter as C
    df = C.build()
    df = df[df[y_col].notna()]
    Z = G._standardize(df[C.X_COLS].to_numpy(float))
    cols = {"nem": ["nem"], "nem_ranking": ["nem", "ranking"]}[effect]
    X = G._standardize(df[cols].to_numpy(float))
    data = Data(df.Q.to_numpy(float), Z, X, df[y_col].to_numpy(float), C.THRESHOLD)
    design = make_design(data, eps, CHILE_POLICY)
    return data, design, {"n": len(data.Q), "cell": lambda s: float(np.ceil(s * 2) / 2)}


VLBW_POLICY_GRAMS = (1200.0, 1800.0)


def build_vlbw(variant: str = "low", eps: float = 0.1, effect: str = "gestat"):
    """NCHS 2002 cohort, mirrored score Q = -grams, treated iff weight < 1500 g.
    ``low``: births of 500-2500 g; ``all``: 500-5000 g."""
    from experiments.datasets.vlbw import adapter as V
    weights = (500.0, 2500.0) if variant == "low" else (500.0, 5000.0)
    Q, X, Y_death, phi0, d = V.mirrored(weights)
    Y = 1.0 - Y_death          # survival, so that a beneficial treatment has a positive effect
    Z = G._standardize(X)
    cols = {"gestat": ["gestat"], "gestat_plural": ["gestat", "twin", "triplet_plus"]}[effect]
    Xe = G._standardize(d[cols].to_numpy(float))
    data = Data(Q, Z, Xe, Y, phi0)
    policy = (-VLBW_POLICY_GRAMS[1], -VLBW_POLICY_GRAMS[0])
    design = make_design(data, eps, policy)
    return data, design, {"n": len(Q), "cell": lambda s: float(np.floor(-s)) + 1.0}


def build_macrosomia(eps: float = 0.1, weights: tuple = (3500.0, 5500.0)):
    """The 5000 g macrosomia line in the same NCHS 2002 file: treat above (no mirroring),
    Y = survival to one year, effect covariate gestation."""
    from experiments.datasets.vlbw import adapter as V
    d = V.sample_frame(weights)
    Q = d.weight.to_numpy(float)
    X = d[V.X_COLS].to_numpy(float)
    data = Data(Q, G._standardize(X), G._standardize(d[["gestat"]].to_numpy(float)),
                1.0 - d.death.to_numpy(float), 4999.5)
    design = make_design(data, eps, (4500.0, 5400.0))
    return data, design, {"n": len(Q), "cell": lambda s: float(np.ceil(s))}


def build_sample(sample, policy: tuple, effect_cols: list, eps: float = 0.1,
                 mirror: bool = False, survival: bool = False):
    """Generic builder from an RDDSample (one score column). ``mirror`` for rules that
    treat below the threshold; ``survival`` flips a binary Y so that larger is better."""
    Q = np.asarray(sample.Q, float).reshape(-1)
    X = np.asarray(sample.X, float)
    Y = np.asarray(sample.Y, float)
    names = list(sample.feature_names)
    ok = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(1)
    Q, X, Y = Q[ok], X[ok], Y[ok]
    if survival:
        Y = 1.0 - Y
    thr = float(sample.threshold)
    if mirror:
        Q, thr, policy = -Q, -thr, (-policy[1], -policy[0])
    idx = [names.index(c) for c in effect_cols]
    data = Data(Q, G._standardize(X), G._standardize(X[:, idx]), Y, thr)
    design = make_design(data, eps, tuple(policy))
    sign = -1.0 if mirror else 1.0
    return data, design, {"n": len(Q), "cell": lambda s: sign * float(s)}


# ----------------------------------------------------------------------------- outcome fits

def _design_matrix(data: Data, eta, idx, J):
    e = eta[idx]
    keep = idx[(e >= J[0]) & (e <= J[1])]
    e, D = eta[keep], data.D[keep]
    n1 = int(D.sum())
    info = _basis_params(max(3, int(round(min(n1, len(keep) - n1) ** (11 / 60)))), J)
    B = BSpline.design_matrix(np.clip(e, info["lo"], info["hi"]), info["t"], info["degree"]).toarray()
    return keep, D, B, info


@dataclass
class ArmFit:
    """Both arms on the link scale: u_d(e, zb, x) = c_d(e) + zb'beta1 + d x'beta2."""
    info: dict
    w0: np.ndarray
    w1: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray

    def _spline(self, w):
        return BSpline(self.info["t"], w, self.info["degree"], extrapolate=False)

    def c(self, e, d):
        e = np.clip(e, self.info["lo"], self.info["hi"])
        return self._spline(self.w1 if d else self.w0)(e)

    def effect(self, e, zb, x):
        v = zb @ self.beta1
        return expit(self.c(e, 1) + v + x @ self.beta2) - expit(self.c(e, 0) + v)


def fit_logit(data: Data, eta, idx, design: M.Design, ridge: float = 1e-3, iters: int = 40) -> ArmFit:
    keep, D, B, info = _design_matrix(data, eta, idx, design.J)
    Zb, X, Y = data.baseline[keep], data.X[keep], data.Y[keep]
    A = np.column_stack([(1 - D)[:, None] * B, D[:, None] * B, Zb, D[:, None] * X])
    n, p = A.shape
    lam = ridge * n / p
    theta = np.zeros(p)
    # intercept-like start: logit of the arm means, spread over the spline columns
    nb = B.shape[1]
    for d, sl in ((0, slice(0, nb)), (1, slice(nb, 2 * nb))):
        m = float(np.clip(Y[D == d].mean(), 1e-3, 1 - 1e-3)) if np.any(D == d) else 0.5
        theta[sl] = np.log(m / (1 - m))
    ll_old = -np.inf
    for _ in range(iters):
        u = A @ theta
        m = expit(u)
        ll = float(np.sum(Y * u - np.logaddexp(0, u))) - 0.5 * lam * theta @ theta
        grad = A.T @ (Y - m) - lam * theta
        w = m * (1 - m)
        H = (A * w[:, None]).T @ A + lam * np.eye(p)
        step = np.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-4:
            th = theta + t * step
            u2 = A @ th
            ll2 = float(np.sum(Y * u2 - np.logaddexp(0, u2))) - 0.5 * lam * th @ th
            if ll2 >= ll - 1e-10:
                break
            t /= 2
        theta = th
        if abs(ll2 - ll_old) < 1e-8 * (1 + abs(ll2)):
            break
        ll_old = ll2
    q = Zb.shape[1]
    return ArmFit(info, theta[:nb], theta[nb:2 * nb], theta[2 * nb:2 * nb + q], theta[2 * nb + q:])


@dataclass
class AdditiveFit:
    """m = F(c_0(e) + zb'beta1) + D {alpha(e) + x'beta2}: logit baseline, additive effect."""
    info: dict
    w0: np.ndarray
    beta1: np.ndarray
    wa: np.ndarray
    beta2: np.ndarray

    def alpha(self, e):
        e = np.clip(e, self.info["lo"], self.info["hi"])
        return BSpline(self.info["t"], self.wa, self.info["degree"], extrapolate=False)(e)


def fit_additive(data: Data, eta, idx, design: M.Design, start: tuple | None = None) -> AdditiveFit:
    keep, D, B, info = _design_matrix(data, eta, idx, design.J)
    Zb, X, Y = data.baseline[keep], data.X[keep], data.Y[keep]
    nb, q, r = B.shape[1], Zb.shape[1], X.shape[1]
    DB, DX = D[:, None] * B, D[:, None] * X

    def mean(th):
        return expit(B @ th[:nb] + Zb @ th[nb:nb + q]) + DB @ th[nb + q:2 * nb + q] + DX @ th[2 * nb + q:]

    def resid(th):
        return mean(th) - Y

    def jac(th):
        m0 = expit(B @ th[:nb] + Zb @ th[nb:nb + q])
        w = (m0 * (1 - m0))[:, None]
        return np.column_stack([w * B, w * Zb, DB, DX])

    if start is None:
        start = np.zeros(2 * nb + q + r)
    res = least_squares(resid, start, jac=jac, method="lm", max_nfev=200)
    th = res.x
    return AdditiveFit(info, th[:nb], th[nb:nb + q], th[nb + q:2 * nb + q], th[2 * nb + q:])


# ----------------------------------------------------------------------------- separable sieve

class Sieve:
    """Chebyshev interpolant of c -> F(c + v) on [c_lo, c_hi] with M terms:
    F(c + v) ~ sum_k phi_k(c) psi_k(v)."""

    def __init__(self, c_lo: float, c_hi: float, M: int = 12):
        self.lo, self.hi, self.M = float(c_lo), float(c_hi), int(M)
        x = np.cos(np.pi * (np.arange(M) + 0.5) / M)        # nodes on [-1, 1]
        self.nodes = self.lo + (x + 1) * (self.hi - self.lo) / 2
        Tk = np.array([Cheb.chebval(x, np.eye(M)[k]) for k in range(M)])   # M x M
        self.proj = (2.0 / M) * Tk
        self.proj[0] /= 2.0

    def scale(self, c):
        return 2 * (np.asarray(c, float) - self.lo) / (self.hi - self.lo) - 1

    def phi(self, c):
        """phi_k(c), shape (len(c), M)."""
        x = np.clip(self.scale(c), -1, 1)
        return np.column_stack([Cheb.chebval(x, np.eye(self.M)[k]) for k in range(self.M)])

    def psi(self, v):
        """psi_k(v), shape (len(v), M)."""
        return expit(self.nodes[None, :] + np.asarray(v, float)[:, None]) @ self.proj.T

    def error(self, v_lo, v_hi, n=121):
        cs = np.linspace(self.lo, self.hi, n)
        vs = np.linspace(v_lo, v_hi, n)
        approx = self.phi(cs) @ self.psi(vs).T
        return float(np.max(np.abs(approx - expit(cs[:, None] + vs[None, :]))))


class EmpiricalTails:
    """Index-fold empirical tails of a multiplier matrix: H_k(s) = mean_j Psi_jk 1{T_j > s}."""

    def __init__(self, T, Psi):
        o = np.argsort(T)
        self.T = T[o]
        self.cum = np.cumsum(Psi[o][::-1], axis=0)[::-1] / len(T)   # cum[i] = sum over j >= i
        self.cum = np.vstack([self.cum, np.zeros((1, Psi.shape[1]))])

    def tails(self, s):
        """Shape (len(s), K): mean_j Psi_jk 1{T_j > s}."""
        pos = np.searchsorted(self.T, np.asarray(s, float), side="right")
        return self.cum[pos]


# ----------------------------------------------------------------------------- rotations

@dataclass
class Rot:
    value: object
    marginal: object
    own: object
    effect_at_cutoff: float
    effect_in_window: float
    window: tuple
    window_share: float
    extra: dict


def _window(data, design, eta, folds, T_idx):
    if design.window_estimated:
        e_idx = eta[folds["utility"]]
        caps = (float(np.quantile(e_idx, design.window_tail)),
                float(np.quantile(e_idx, 1 - design.window_tail)))
    else:
        caps = design.window if design.window is not None else design.J
    l_hat = max(design.phi0 - float(np.quantile(T_idx, 1 - design.eps)), caps[0])
    u_hat = min(design.phi0 - float(np.quantile(T_idx, design.eps)), caps[1])
    return l_hat, u_hat


def rotation(data: Data, design: M.Design, folds: dict, route: str, M_sieve: int = 12) -> Rot:
    g = np.linalg.lstsq(M._xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
    eta = data.Q - M._xt(data.Z) @ g
    T_idx = M._xt(data.Z[folds["index"]]) @ g
    l_hat, u_hat = _window(data, design, eta, folds, T_idx)
    U = folds["utility"]
    e_all = eta[U]
    inside = (e_all >= l_hat) & (e_all <= u_hat)
    e = e_all[inside]
    n_u = len(U)
    near = np.abs(data.Q[U] - design.phi0) <= 0.05 * float(np.std(data.Q))
    extra = {}

    if route in ("lpm", "additive"):
        if route == "lpm":
            fit = M.outcome_fit(data, eta, folds["outcome"], design)
        else:
            fit = fit_additive(data, eta, folds["outcome"], design)
        tails = fit_weighted_tails(T_idx, data.X[folds["index"]], design.T_support)
        amc = fit.alpha(e) - design.cost
        b2 = fit.beta2

        def value(phi):
            t = phi - e
            return float(np.sum(amc * tails.survival(t) + tails.weighted_tail(t, b2)) / n_u)

        def marginal(phi):
            t = phi - e
            return float(np.sum(amc * tails.density(t) + tails.weighted_density(t, b2))
                         / np.sum(tails.density(t)))

        eff_all = fit.alpha(e_all) + data.X[U] @ b2
        extra["beta2"] = np.asarray(b2).tolist()
    else:
        fit = fit_logit(data, eta, folds["outcome"], design)
        Zb_i, X_i = data.baseline[folds["index"]], data.X[folds["index"]]
        v1 = Zb_i @ fit.beta1 + X_i @ fit.beta2
        v0 = Zb_i @ fit.beta1
        c_vals = np.concatenate([fit.c(e, 0), fit.c(e, 1)])
        sieve = Sieve(c_vals.min() - 0.25, c_vals.max() + 0.25, M_sieve)
        Psi = np.column_stack([sieve.psi(v1), sieve.psi(v0)])          # n_T x 2M
        Phi1, Phi0 = sieve.phi(fit.c(e, 1)), sieve.phi(fit.c(e, 0))   # inside x M
        extra["sieve_error"] = sieve.error(float(min(v0.min(), v1.min())), float(max(v0.max(), v1.max())))
        extra["c_range"] = [sieve.lo, sieve.hi]
        extra["beta2"] = np.asarray(fit.beta2).tolist()
        if route == "logit_sp":
            tails = fit_weighted_tails(T_idx, Psi, design.T_support)
            K = Psi.shape[1]

            def _H(t):
                return np.column_stack([tails.weighted_tail(t, np.eye(K)[k]) for k in range(K)])

            def _rho(t):
                return np.column_stack([tails.weighted_density(t, np.eye(K)[k]) for k in range(K)])

            def value(phi):
                t = phi - e
                H = _H(t)
                w = np.sum(Phi1 * H[:, :M_sieve], axis=1) - np.sum(Phi0 * H[:, M_sieve:], axis=1)
                return float(np.sum(w - design.cost * tails.survival(t)) / n_u)

            def marginal(phi):
                t = phi - e
                R = _rho(t)
                num = np.sum(Phi1 * R[:, :M_sieve], axis=1) - np.sum(Phi0 * R[:, M_sieve:], axis=1)
                return float(np.sum(num - design.cost * tails.density(t)) / np.sum(tails.density(t)))
        else:
            emp = EmpiricalTails(T_idx, np.column_stack([Psi, np.ones(len(T_idx))]))

            def value(phi):
                t = phi - e
                H = emp.tails(t)
                w = np.sum(Phi1 * H[:, :M_sieve], axis=1) - np.sum(Phi0 * H[:, M_sieve:2 * M_sieve], axis=1)
                return float(np.sum(w - design.cost * H[:, -1]) / n_u)

            def marginal(phi):          # finite differences of the step criterion
                h = 0.02 * float(np.std(e_all))
                num = -(value(phi + h) - value(phi - h)) / (2 * h)
                den = np.sum(emp.tails(phi - h - e)[:, -1] - emp.tails(phi + h - e)[:, -1]) / (2 * h * n_u)
                return float(num / den) if den > 0 else float("nan")

        eff_all = fit.effect(e_all, data.baseline[U], data.X[U])

    def own(phi):
        return float(np.mean(inside * (eff_all - design.cost) * (data.Q[U] >= phi)))

    # the model's effect near the deployed threshold by terciles of the first effect
    # covariate, to compare with the local RD split (Chile: NEM terciles)
    x1 = data.X[U][:, 0]
    q1, q2 = np.quantile(x1, [1 / 3, 2 / 3])
    extra["effect_at_cutoff_terciles"] = [
        float(np.mean(eff_all[near & m])) if (near & m).any() else float("nan")
        for m in (x1 <= q1, (x1 > q1) & (x1 <= q2), x1 > q2)]

    return Rot(value, marginal, own, float(np.mean(eff_all[near])) if near.any() else float("nan"),
               float(np.mean(eff_all[inside])), (l_hat, u_hat), float(inside.mean()), extra)


def four_role(data: Data, design: M.Design, route: str, seed: int, curves: np.ndarray,
              grid_points: int = 121, M_sieve: int = 12) -> dict:
    blocks = np.array_split(np.random.default_rng(seed).permutation(len(data.Q)), 4)
    rots, phis = [], []
    for k in range(4):
        folds = {r: blocks[(j + k) % 4] for j, r in enumerate(M.ROLES)}
        rot = rotation(data, design, folds, route, M_sieve)
        rots.append(rot)
        phis.append(M._argmax(rot.value, *design.policy, m=grid_points)[0])
    pooled = lambda x: float(np.mean([r.value(x) for r in rots]))
    phi_pool, at_bound, n_local = M._argmax(pooled, *design.policy, m=grid_points)
    cuts = np.unique(data.Q)
    cuts = cuts[(cuts >= design.policy[0]) & (cuts <= design.policy[1])]
    if len(cuts) > 400:
        cuts = np.quantile(cuts, np.linspace(0, 1, 400))
    own = np.array([np.mean([r.own(c) for r in rots]) for c in cuts])
    return {
        "route": route, "phi_avg": float(np.mean(phis)), "phi_pooled": phi_pool,
        "phi_rotations": phis, "at_boundary": at_bound, "local_maxima": n_local,
        "gain_vs_deployed": pooled(phi_pool) - pooled(design.phi0),
        "own_cutoff": float(cuts[int(np.argmax(own))]),
        "effect_at_cutoff": float(np.mean([r.effect_at_cutoff for r in rots])),
        "effect_in_window": float(np.mean([r.effect_in_window for r in rots])),
        "window": np.mean([r.window for r in rots], axis=0).tolist(),
        "window_share": float(np.mean([r.window_share for r in rots])),
        "U": (np.array([pooled(x) for x in curves]) - pooled(design.phi0)).tolist(),
        "tau": [float(np.mean([r.marginal(x) for r in rots])) for x in curves],
        "extra": rots[0].extra,
    }


# ----------------------------------------------------------------------------- driver

SETTINGS = {
    "gpa": [("gpa", y) for y in ("grad6", "grad5", "return2", "stay")],
    "chile": [("chile", y) for y in ("univ_2024", "any_2025", "univ_2025")],
    "vlbw": [("vlbw", "low")],      # "all" (500-5000 g) leaves no supported window; see the README
    "macro": [("macro", "5000g")],  # the 5000 g macrosomia line in the same file
    "sba": [("sba", "chargeoff")],
    "nyc": [("nyc", "next_score"), ("nyc", "closed")],
    "cps": [("cps", "medicaid"), ("cps", "uninsured")],
}

# Generic settings through build_sample: module, loader, policy set (score units),
# effect covariates (feature names), mirror (treat below), survival (flip a bad outcome).
GENERIC = {
    ("sba", "chargeoff"): dict(module="experiments.datasets.sba_7a.adapter", loader="load",
                               policy=(100_000.0, 250_000.0), effect=["term_months"],
                               mirror=True, survival=True),
    ("nyc", "next_score"): dict(module="experiments.datasets.nyc_restaurants.adapter", loader="load",
                                policy=(5.0, 25.0), effect=["prior_score"], mirror=True, survival=True),
    ("nyc", "closed"): dict(module="experiments.datasets.nyc_restaurants.adapter", loader="load_exit",
                            policy=(5.0, 25.0), effect=["prior_score"], mirror=True, survival=True),
    ("cps", "medicaid"): dict(module="experiments.datasets.cps_medicaid.adapter", loader="load",
                              policy=(80.0, 220.0), effect=["age"], mirror=True, survival=False),
    ("cps", "uninsured"): dict(module="experiments.datasets.cps_medicaid.adapter", loader="load_uninsured",
                               policy=(80.0, 220.0), effect=["age"], mirror=True, survival=True),
}


def build(setting: str, y: str, eps: float):
    if setting == "gpa":
        return build_gpa(y, eps=eps)
    if setting == "chile":
        return build_chile(y, eps=eps)
    if setting == "macro":
        return build_macrosomia(eps=eps)
    if (setting, y) in GENERIC:
        import importlib
        g = GENERIC[(setting, y)]
        sample = getattr(importlib.import_module(g["module"]), g["loader"])()
        return build_sample(sample, g["policy"], g["effect"], eps=eps, mirror=g["mirror"],
                            survival=g["survival"])
    return build_vlbw(y, eps=eps)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["gpa", "chile"])
    ap.add_argument("--routes", nargs="+", default=list(ROUTES))
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--cost", type=float, default=0.0,
                    help="per-treated-unit cost c on the outcome's probability scale")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sieve", type=int, default=12)
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in a.settings:
        for setting, y in SETTINGS[s]:
            data, design, meta = build(setting, y, a.eps)
            design.cost = a.cost
            curves = np.linspace(*design.policy, 61)
            print(f"\n== {setting} {y}: n={meta['n']}, treated={data.D.mean():.3f}, "
                  f"mean Y={data.Y.mean():.3f}, J={np.round(design.J, 3).tolist()}, "
                  f"window caps={np.round(design.window, 3).tolist()}, policy={design.policy}", flush=True)
            for route in a.routes:
                t0 = time.time()
                r = four_role(data, design, route, a.seed, curves, a.grid, a.sieve)
                r.update(setting=setting, outcome=y, n=meta["n"], eps=a.eps, cost=a.cost,
                         cell=meta["cell"](r["phi_pooled"]), seconds=round(time.time() - t0, 1),
                         curves=curves.tolist(), phi0=design.phi0)
                rows.append(r)
                ex = r["extra"]
                print(f"  {route:9s} phi_avg={r['phi_avg']:8.3f} pooled={r['phi_pooled']:8.3f} "
                      f"cell={r['cell']:7.3f} bound={r['at_boundary']!s:5s} maxima={r['local_maxima']} "
                      f"own={r['own_cutoff']:8.3f} eff@cut={r['effect_at_cutoff']:+.3f} "
                      f"eff(win)={r['effect_in_window']:+.3f} gain={r['gain_vs_deployed']:+.4f} "
                      f"win={r['window_share']:.2f}"
                      + (f" sieve_err={ex['sieve_error']:.1e}" if "sieve_error" in ex else "")
                      + f"  [{r['seconds']}s]", flush=True)
                with open(a.out / "summary.json", "w") as f:
                    json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
