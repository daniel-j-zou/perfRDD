"""Debiased (orthogonal) evaluation score against the plug-in estimator.

Companion to ``manuscript/this_week.tex`` (debiased evaluation score).  The
score adds the estimated outcome, index and first-stage scores to the plug-in
evaluation average,

    mu = -(a(eta)-c) g(phi-eta) I(eta)                       plug-in
         - r_out(D,eta,X) {Y - m(D,eta,X)}                    outcome
         - {r_g(T) - int r_g g}                               density
         - b_l z_{1-eps}(T) + b_u z_eps(T)                    endpoints
         + A' Sigma^{-1} X~ (Q - gamma' X~)                   first stage,

with r_g(t) = (a(phi-t)-c) f_eta(phi-t) 1{l <= phi-t <= u}, b_v =
(a(v)-c) g(phi-v) f_eta(v) and z_p(t) = {p - 1(t <= q_p)} / g(q_p).  The
estimator is the one-step update phi_DR = phi_tilde - Psi(phi_tilde) / H_hat,
where phi_tilde maximizes the plug-in criterion on the same evaluation fold.

All nuisances, including gamma, are fitted on one training sample.  The outcome
representer is the sieve representer z_r' Q_r^{-1} l_r on a spline basis whose
knots refine the outcome fit's knots by ``--rep-mult`` (nested sieves), or, with
``--rep-mult 0``, the closed-form representer of the partially linear outcome
space under X independent of eta (the version covered by the proof in
``this_week.tex``).  The
first-stage loading A is a central difference of the training-sample plug-in
score in gamma, refitting every nuisance; its intercept coordinate is zero by
location invariance and is skipped.

Estimators (``cell`` keys):
  pi4_fixed, pi4_rot   four-role plug-in, one assignment / pooled rotation
  pi2_one, dr2_one     training half / evaluation half, one split
  pi2_cf, dr2_cf       average over both halves
  pi4f_one, dr4f_one   one fold of a 4-fold cross-fit (train on 3/4)
  pi4f_cf, dr4f_cf     average over the four folds

Designs:
  baseline   alpha(eta) = 2 + eta (the hard-trim Gaussian baseline).
             First-order predictions from the analytic role scores:
             pi4_fixed 174.7; pi2_one = dr2_one = 87.4; pi4f_one 67.8;
             dr4f_one 174.7; every rotated or cross-fit estimator 43.68.
  nonlinear  alpha(eta) = 2 + eta + 0.5 sin(2 eta), for the bias check with a
             coarse outcome spline (``--outcome-knots``).

Run:
    python -m experiments.scripts.debiased_score_check --design baseline --n 20000 40000 --reps 500
    python -m experiments.scripts.debiased_score_check --design nonlinear --outcome-knots 2 \
        --estimators pi4f_cf dr4f_cf --rep-mult 1 3 --n 20000 80000 --reps 400
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.methods.spline_density import SplineDensityFit, fit_spline_density
from experiments.scripts.eight_role_variance_decomposition import decomposition
from experiments.scripts.four_role_shared_gamma_check import component as four_role_component
from experiments.scripts.hard_trim_crossfit_regularization import (
    T_DENSITY_SUPPORT,
    EvaluationComponent,
    _boundaries_from_T,
    _maximize_components,
)
from experiments.scripts.hard_trim_gaussian_baseline import (
    BETA,
    COST,
    EPS,
    GAMMA,
    L0,
    NUISANCE_SUPPORT,
    P,
    PHI_0,
    POLICY_BOUNDS,
    SIGMA_Y,
    U0,
    GeneratedData,
    KNOT_EXPONENT,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "debiased_score_check"
ETA_DENSITY_SUPPORT = (-3.0, 3.0)
A_STEP = 0.05
ALL_ESTIMATORS = ("pi4_fixed", "pi4_rot", "pi2_one", "dr2_one", "pi2_cf", "dr2_cf",
                  "pi4f_one", "dr4f_one", "pi4f_cf", "dr4f_cf")


# ---------------------------------------------------------------- designs

def alpha_baseline(eta):
    return 2.0 + np.asarray(eta)


def alpha_nonlinear(eta):
    eta = np.asarray(eta)
    return 2.0 + eta + 0.5 * np.sin(2.0 * eta)


ALPHAS: Dict[str, Callable] = {"baseline": alpha_baseline, "nonlinear": alpha_nonlinear}


def generate(design: str, n: int, seed: int) -> GeneratedData:
    """Same DGP as the Gaussian baseline, with the design's alpha."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, P))
    eta = rng.standard_normal(n)
    T = X @ GAMMA
    Q = T + eta
    D = (Q > PHI_0).astype(float)
    Y = D * ALPHAS[design](eta) + 0.5 * eta ** 2 + X @ BETA + rng.normal(0.0, SIGMA_Y, n)
    return GeneratedData(X=X, eta=eta, T=T, Q=Q, D=D, Y=Y)


def truth(design: str) -> Dict[str, float]:
    a = ALPHAS[design]

    def score(phi):
        return quad(lambda e: -(a(e) - COST) * norm.pdf(phi - e) * norm.pdf(e),
                    L0, U0, epsabs=1e-12, limit=200)[0]

    def curvature(phi):
        return quad(lambda e: (a(e) - COST) * (phi - e) * norm.pdf(phi - e) * norm.pdf(e),
                    L0, U0, epsabs=1e-12, limit=200)[0]

    phi = float(brentq(score, *POLICY_BOUNDS))
    return {"phi_star": phi, "curvature": float(curvature(phi))}


def predictions() -> Dict[str, float]:
    """First-order N Var predictions for the baseline from the role scores."""
    d = decomposition()
    H2 = d["truth"]["hard_curvature"] ** 2
    cov = np.asarray(d["score_covariance"])
    roles = list(d["roles"])
    var_u = float(cov[roles.index("utility"), roles.index("utility")])
    sigma2 = float(np.sum(cov))
    var_n = sigma2 - var_u
    return {
        "sigma2": sigma2, "H2": H2,
        "pi4_fixed": 4 * sigma2 / H2, "pi4_rot": sigma2 / H2,
        "pi2_one": (2 * var_u + 2 * var_n) / H2, "dr2_one": 2 * sigma2 / H2,
        "pi2_cf": sigma2 / H2, "dr2_cf": sigma2 / H2,
        "pi4f_one": (4 * var_u + 4 / 3 * var_n) / H2, "dr4f_one": 4 * sigma2 / H2,
        "pi4f_cf": sigma2 / H2, "dr4f_cf": sigma2 / H2,
    }


# ---------------------------------------------------------------- nuisances

@dataclass
class OutcomeFit:
    info: dict
    coef: np.ndarray
    n_basis: int

    def design(self, X, D, eta):
        Phi = _eval_basis(eta, self.info)
        return np.column_stack((X, Phi, D[:, None] * Phi))

    def predict(self, X, D, eta):
        return self.design(X, D, eta) @ self.coef

    def effect(self, eta):
        return _eval_basis(eta, self.info) @ self.coef[P + self.n_basis:]


def in_support(eta):
    return (eta >= NUISANCE_SUPPORT[0]) & (eta <= NUISANCE_SUPPORT[1])


def default_knots(data, idx, eta_hat) -> int:
    keep = idx[in_support(eta_hat[idx])]
    treated = int(data.D[keep].sum())
    n_eff = min(treated, len(keep) - treated)
    return max(3, int(round(n_eff ** KNOT_EXPONENT)))


def fit_outcome(data, idx, eta_hat, n_knots) -> OutcomeFit:
    keep = idx[in_support(eta_hat[idx])]
    info = _basis_params(n_knots, NUISANCE_SUPPORT)
    n_basis = n_knots + 4
    fit = OutcomeFit(info=info, coef=np.zeros(P + 2 * n_basis), n_basis=n_basis)
    design = fit.design(data.X[keep], data.D[keep], eta_hat[keep])
    fit.coef, *_ = np.linalg.lstsq(design, data.Y[keep], rcond=None)
    return fit


@dataclass
class Nuisances:
    gamma: np.ndarray
    outcome: OutcomeFit
    gdens: SplineDensityFit
    l: float
    u: float
    knots: int


def fit_gamma(data, idx):
    design = np.column_stack((np.ones(len(idx)), data.X[idx]))
    return np.linalg.lstsq(design, data.Q[idx], rcond=None)[0]


def fit_nuisances(data, idx, gamma, knots=None, density_basis=None) -> Nuisances:
    T_hat = gamma[0] + data.X @ gamma[1:]
    eta_hat = data.Q - T_hat
    k = default_knots(data, idx, eta_hat) if knots is None else int(knots)
    outcome = fit_outcome(data, idx, eta_hat, k)
    gdens = fit_spline_density(T_hat[idx], T_DENSITY_SUPPORT, n_basis=density_basis)
    l, u = _boundaries_from_T(T_hat[idx])
    return Nuisances(gamma, outcome, gdens, l, u, k)


def plug_in_score(nu: Nuisances, data, idx, phi) -> float:
    """Sample plug-in utility derivative on ``idx`` (normalized by |idx|)."""
    eta = data.Q[idx] - nu.gamma[0] - data.X[idx] @ nu.gamma[1:]
    inside = (eta >= nu.l) & (eta <= nu.u)
    a = nu.outcome.effect(eta)
    return -float(np.mean((a - COST) * nu.gdens.density(phi - eta) * inside))


def first_stage_loading(data, idx, nu: Nuisances, phi, density_basis=None) -> np.ndarray:
    """Central-difference A: refit every nuisance at gamma +- step e_j."""
    A = np.zeros(P + 1)
    for j in range(1, P + 1):
        vals = []
        for sign in (1.0, -1.0):
            g = nu.gamma.copy()
            g[j] += sign * A_STEP
            shifted = fit_nuisances(data, idx, g, knots=nu.knots, density_basis=density_basis)
            vals.append(plug_in_score(shifted, data, idx, phi))
        A[j] = (vals[0] - vals[1]) / (2 * A_STEP)
    return A


def component_for(nu: Nuisances, data, idx) -> EvaluationComponent:
    eta = data.Q[idx] - nu.gamma[0] - data.X[idx] @ nu.gamma[1:]
    return EvaluationComponent(
        eta=eta,
        hard_weights=((eta >= nu.l) & (eta <= nu.u)).astype(float),
        treatment_effect=nu.outcome.effect(eta),
        T_density=nu.gdens,
    )


def debiased_one_step(data, train, evalf, rep_mult=3, knots=None, density_basis=None,
                      iterations=1):
    """Return (plug-in phi_tilde, debiased phi, diagnostics).

    ``rep_mult`` may be a list; then the debiased estimate is a dict keyed by
    representer choice (0 = closed form, m >= 1 = sieve refined m times),
    sharing every other nuisance and correction.  ``iterations`` > 1 repeats
    the Newton step phi <- phi - Psi(phi) / H_hat with H_hat and A_hat held
    fixed, which converges to the root of the debiased score; the root has the
    same first-order limit as the one-step estimator and does not need a
    consistent curvature estimate.
    """
    rep_list = list(rep_mult) if isinstance(rep_mult, (list, tuple)) else [rep_mult]
    gamma = fit_gamma(data, train)
    nu = fit_nuisances(data, train, gamma, knots=knots, density_basis=density_basis)
    phi_t = _maximize_components([component_for(nu, data, evalf)])[0]

    T_all = gamma[0] + data.X @ gamma[1:]
    eta_all = data.Q - T_all
    eta_tr, eta_ev = eta_all[train], eta_all[evalf]
    T_ev = T_all[evalf]
    fdens = fit_spline_density(eta_tr, ETA_DENSITY_SUPPORT)
    a = nu.outcome.effect
    g = nu.gdens.density
    inside_ev = (eta_ev >= nu.l) & (eta_ev <= nu.u)
    inside_tr = (eta_tr >= nu.l) & (eta_tr <= nu.u)
    S_ev, S_tr = in_support(eta_ev), in_support(eta_tr)
    H_hat = -float(np.mean((a(eta_ev) - COST) * nu.gdens.density_derivative(phi_t - eta_ev)
                           * inside_ev))
    resid = np.where(S_ev, data.Y[evalf] - nu.outcome.predict(data.X[evalf], data.D[evalf],
                                                                eta_ev), 0.0)

    # Pieces of the outcome representers that do not depend on phi.
    rep_pieces = {}
    for rep_m in rep_list:
        if rep_m == 0:
            # Closed form (common slope): r = S_J{(D - p) lam + X_c' kappa},
            # lam = (w - H_c' kappa) / {p (1 - p)}, kappa solving
            # {P(J) Sigma_c - E[S_J H_c H_c' / v]} kappa = -E[S_J w H_c / v].
            T_tr = T_all[train]
            X_bar = data.X[train].mean(axis=0)
            Xc_tr = data.X[train] - X_bar
            order = np.argsort(T_tr)
            T_sorted = T_tr[order]
            tail_x = np.vstack((np.cumsum(Xc_tr[order][::-1], axis=0)[::-1],
                                np.zeros((1, P)))) / len(train)

            def tails(eta, T_sorted=T_sorted, tail_x=tail_x):
                pos = np.searchsorted(T_sorted, PHI_0 - eta, side="right")
                return (len(train) - pos) / len(train), tail_x[pos]

            p_tr, H_tr = tails(eta_tr)
            v_tr = np.clip(p_tr * (1 - p_tr), 1e-3, None)
            lhs = S_tr.mean() * (Xc_tr.T @ Xc_tr / len(train)) \
                - (H_tr * (S_tr / v_tr)[:, None]).T @ H_tr / len(train)
            p_ev, H_ev = tails(eta_ev)
            v_ev = np.clip(p_ev * (1 - p_ev), 1e-3, None)
            rep_pieces[0] = (lhs, H_tr, v_tr, p_ev, H_ev, v_ev, data.X[evalf] - X_bar)
        else:
            k_r = rep_m * (nu.knots + 1) - 1
            info_r = _basis_params(k_r, NUISANCE_SUPPORT)
            nb_r = k_r + 4

            def z_r(X, D, eta, info_r=info_r):
                Phi = _eval_basis(eta, info_r)
                return np.column_stack((X, Phi, D[:, None] * Phi))

            Z_tr = z_r(data.X[train], data.D[train], eta_tr) * S_tr[:, None]
            Q_r = Z_tr.T @ Z_tr / len(train)
            Phi_tr = _eval_basis(eta_tr, info_r)
            Z_ev = z_r(data.X[evalf], data.D[evalf], eta_ev) * S_ev[:, None]
            rep_pieces[rep_m] = (Q_r, Phi_tr, nb_r, Z_ev)

    def outcome_representer(rep_m, phi):
        w_tr = inside_tr * g(phi - eta_tr)
        if rep_m == 0:
            lhs, H_tr, v_tr, p_ev, H_ev, v_ev, Xc_ev = rep_pieces[0]
            kappa = np.linalg.solve(lhs, -np.mean((S_tr * w_tr / v_tr)[:, None] * H_tr, axis=0))
            lam_ev = (inside_ev * g(phi - eta_ev) - H_ev @ kappa) / v_ev
            return S_ev * ((data.D[evalf] - p_ev) * lam_ev + Xc_ev @ kappa)
        Q_r, Phi_tr, nb_r, Z_ev = rep_pieces[rep_m]
        ell = np.zeros(P + 2 * nb_r)
        ell[P + nb_r:] = np.mean(w_tr[:, None] * Phi_tr, axis=0)
        return Z_ev @ np.linalg.solve(Q_r, ell)

    nodes, weights = leggauss(200)
    e_nodes = 0.5 * (nu.u - nu.l) * nodes + 0.5 * (nu.u + nu.l)
    e_w = 0.5 * (nu.u - nu.l) * weights
    q_hi, q_lo = PHI_0 - nu.l, PHI_0 - nu.u          # q_{1-eps}, q_eps
    z_hi = ((1 - EPS) - (T_ev <= q_hi)) / float(g(np.array([q_hi]))[0])
    z_lo = (EPS - (T_ev <= q_lo)) / float(g(np.array([q_lo]))[0])

    # First-stage correction: A' Sigma^{-1} X~ (Q - gamma' X~), Sigma on eval fold.
    A_hat = first_stage_loading(data, train, nu, phi_t, density_basis)
    Xt_ev = np.column_stack((np.ones(len(evalf)), data.X[evalf]))
    gamma_corr = (Xt_ev @ np.linalg.solve(Xt_ev.T @ Xt_ev / len(evalf), A_hat)) * eta_ev

    def pieces(phi):
        plug = -(a(eta_ev) - COST) * g(phi - eta_ev) * inside_ev
        e = phi - T_ev
        r_g = (a(e) - COST) * fdens.density(e) * ((e >= nu.l) & (e <= nu.u))
        C_g = float(np.sum(e_w * (a(e_nodes) - COST) * g(phi - e_nodes) * fdens.density(e_nodes)))
        density_corr = -(r_g - C_g)

        def b(v_):
            return float((a(np.array([v_]))[0] - COST) * g(np.array([phi - v_]))[0]
                         * fdens.density(np.array([v_]))[0])

        endpoint_corr = -b(nu.l) * z_hi + b(nu.u) * z_lo
        common = plug + density_corr + endpoint_corr + gamma_corr
        outcome = {m: -outcome_representer(m, phi) * resid for m in rep_list}
        return common, outcome, (plug, density_corr, endpoint_corr)

    phi_dr = {}
    for m in rep_list:
        phi = phi_t
        for _ in range(max(1, int(iterations))):
            common, outcome, _ = pieces(phi)
            phi = phi - float(np.mean(common + outcome[m])) / H_hat
        phi_dr[m] = phi
    common, outcome, (plug, density_corr, endpoint_corr) = pieces(phi_t)
    diag = {"H_hat": H_hat, "A_hat": A_hat[1:].tolist(),
            "corr_means": [float(np.mean(x)) for x in
                           (plug, outcome[rep_list[0]], density_corr, endpoint_corr,
                            gamma_corr)]}
    if not isinstance(rep_mult, (list, tuple)):
        phi_dr = phi_dr[rep_mult]
    return phi_t, phi_dr, diag


# ---------------------------------------------------------------- replication

def replicate(task):
    design, n, seed, estimators, rep_mults, knots, density_basis, *rest = task
    iterations = rest[0] if rest else 1
    data = generate(design, n, seed)
    rng = np.random.default_rng(seed + 7_919)
    out: Dict[str, float] = {}

    if {"pi4_fixed", "pi4_rot"} & set(estimators):
        blocks = np.array_split(rng.permutation(n), 4)
        roles = ("gamma", "outcome", "index", "utility")
        parts = [four_role_component(data, {r: blocks[(k + s) % 4] for k, r in enumerate(roles)})
                 for s in range(4)]
        out["pi4_fixed"] = _maximize_components([parts[0]])[0]
        out["pi4_rot"] = _maximize_components(parts)[0]

    for n_folds, tag in ((2, "2"), (4, "4f")):
        wanted = [e for e in estimators if e.endswith(("_one", "_cf")) and tag in e]
        if not wanted:
            continue
        folds = np.array_split(rng.permutation(n), n_folds)
        pis, drs = [], []
        for k in range(n_folds):
            train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            phi_t, phi_dr, _ = debiased_one_step(data, train, folds[k], list(rep_mults),
                                                 knots, density_basis, iterations)
            pis.append(phi_t)
            drs.append(phi_dr)
        out[f"pi{tag}_one"] = pis[0]
        out[f"pi{tag}_cf"] = float(np.mean(pis))
        for m in rep_mults:
            suffix = "" if len(rep_mults) == 1 else ("_exact" if m == 0 else f"_m{m}")
            out[f"dr{tag}_one{suffix}"] = drs[0][m]
            out[f"dr{tag}_cf{suffix}"] = float(np.mean([d[m] for d in drs]))
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=sorted(ALPHAS), default="baseline")
    ap.add_argument("--n", type=int, nargs="+", default=[20000, 40000])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--estimators", nargs="+", default=list(ALL_ESTIMATORS))
    ap.add_argument("--rep-mult", type=int, nargs="+", default=[3])
    ap.add_argument("--outcome-knots", type=int, default=None,
                    help="interior outcome knots (default: the theorem's rate rule)")
    ap.add_argument("--density-basis", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=1,
                    help="Newton steps on the debiased score (1 = one-step estimator)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    tr = truth(args.design)
    result = {"args": vars(args), "truth": tr, "cells": {}}
    if args.design == "baseline":
        result["prediction"] = predictions()
    print(f"design={args.design} phi*={tr['phi_star']:.6f} H={tr['curvature']:.6f}")
    for n in args.n:
        tasks = [(args.design, n, args.seed + 104_729 * n + r, args.estimators,
                  args.rep_mult, args.outcome_knots, args.density_basis, args.iterations)
                 for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(replicate, tasks, chunksize=2))
        keys = sorted(rows[0])
        cell = {}
        for key in keys:
            err = np.array([row[key] for row in rows]) - tr["phi_star"]
            cell[key] = {
                "n_var": float(n * np.var(err, ddof=1)),
                "n_mse": float(n * np.mean(err ** 2)),
                "root_n_bias": float(np.sqrt(n) * np.mean(err)),
                "root_n_bias_se": float(np.sqrt(n) * np.std(err, ddof=1) / np.sqrt(len(err))),
            }
        result["cells"][str(n)] = cell
        print(f"N={n} (reps={args.reps}; N Var has ~{100 * np.sqrt(2 / (args.reps - 1)):.0f}% MC error)")
        for key in keys:
            c = cell[key]
            pred = result.get("prediction", {}).get(key.split("_m")[0])
            ptxt = f"  pred {pred:6.1f}" if pred else ""
            print(f"  {key:14s} N Var {c['n_var']:7.1f}  N MSE {c['n_mse']:7.1f}  "
                  f"sqrt(N) bias {c['root_n_bias']:+6.2f} (se {c['root_n_bias_se']:.2f}){ptxt}")
    out = DEFAULT_OUT / (args.tag or f"{args.design}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {out / 'summary.json'}")


if __name__ == "__main__":
    main()
