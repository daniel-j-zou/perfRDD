"""Differing slopes: outcome spline fit at generated residuals, uniform consistency.

Monte Carlo companion to the consistency lemmas in ``this_week.tex``
(2026-10-13).  One first-stage fold gives gamma-hat; an independent outcome
fold of the same size fits the stacked regression

    Y on (D N_K(eta-hat), N_K(eta-hat), X, D X)   for eta-hat in J,

with a cubic B-spline basis on the fixed region J.  Reported per n:

* sup |a-hat - a| and sup |a-hat' - a'| on the trim window [l0, u0] and on J;
* |beta2-hat - beta2|;
* the outer-crossing effect sqrt(n) |l_K'(theta-hat - theta-hat^S)|, where
  theta-hat^S uses the oracle membership 1{eta in J} with the same generated
  inputs, and l_K is the utility functional of the outcome fit at phi*_eps.

The lemmas predict that the first three shrink (rates K/sqrt(n) + K^{-5/2},
K^2/sqrt(n) + K^{-3/2}, sqrt(K/n) + K^{-3}) and the last is o_p(1).

Design: the four-role differing-slopes check design with nonlinear a and b,
X ~ N(0, I_2), T = X1, eta ~ N(0,1), D = 1{Q > 0}, trim level 0.10.

Run:
    python -m experiments.scripts.ds_outcome_fit_consistency --reps 200 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_outcome_fit_consistency"

J = (-1.7, 1.7)
EPS = 0.10
L0, U0 = -stats.norm.ppf(1 - EPS), stats.norm.ppf(1 - EPS)
BETA1 = np.array([0.30, -0.20])
BETA2 = np.array([0.80, 0.25])
PHI_STAR = -0.1197699  # supported optimum of the linear design; any fixed phi works here


def a_fun(e):
    return 0.35 + 0.90 * e + 0.30 * np.sin(1.5 * e)


def da_fun(e):
    return 0.90 + 0.45 * np.cos(1.5 * e)


def b_fun(e):
    return 0.20 + 0.60 * e + 0.50 * (e ** 2 - 1)


def n_interior_knots(n: int) -> int:
    # K grows like n^{0.185}, inside n^{1/6} << K << n^{1/5}
    return max(4, int(round(1.6 * n ** 0.185)))


def basis(knots_int: int):
    t_int = np.linspace(J[0], J[1], knots_int + 2)
    t = np.r_[[J[0]] * 3, t_int, [J[1]] * 3]
    nb = len(t) - 4
    return t, nb


def design(e, t, nb, extrapolate=False):
    # Columns are the B-spline basis functions (extrapolate continues the boundary cubic).
    out = np.empty((len(e), nb))
    for j in range(nb):
        c = np.zeros(nb)
        c[j] = 1.0
        out[:, j] = BSpline(t, c, 3, extrapolate=extrapolate)(e)
    return np.nan_to_num(out)


def draw(n, rng):
    X = rng.standard_normal((n, 2))
    eta = rng.standard_normal(n)
    T = X[:, 0]
    Q = T + eta
    D = (Q > 0).astype(float)
    Y = b_fun(eta) + X @ BETA1 + D * (a_fun(eta) + X @ BETA2) + 0.5 * rng.standard_normal(n)
    return X, eta, Q, D, Y


def fit(eta_in, keep, X, D, Y, t, nb, extrapolate=False):
    B = design(eta_in[keep], t, nb, extrapolate)
    Z = np.c_[D[keep, None] * B, B, X[keep], D[keep, None] * X[keep]]
    theta, *_ = np.linalg.lstsq(Z, Y[keep], rcond=None)
    return theta


def ell_vector(t, nb):
    """l_K: E[g(phi*-eta) N_K(eta) I0] on the a-block and E[rho_X(phi*-eta) I0] on the beta2-block."""
    e, w = np.polynomial.legendre.leggauss(400)
    e = 0.5 * (U0 - L0) * e + 0.5 * (U0 + L0)
    w = 0.5 * (U0 - L0) * w
    wt = w * stats.norm.pdf(e) * stats.norm.pdf(PHI_STAR - e)
    la = design(e, t, nb).T @ wt
    # rho_X(s) = E[X | T=s] g(s) = (s, 0) g(s)
    l2 = np.array([np.sum(w * stats.norm.pdf(e) * (PHI_STAR - e) * stats.norm.pdf(PHI_STAR - e)), 0.0])
    return np.r_[la, np.zeros(nb), np.zeros(2), l2]


def replicate(task):
    n, seed = task
    rng = np.random.default_rng(seed)
    Xg, _, Qg, _, _ = draw(n, rng)
    Xt = np.c_[np.ones(n), Xg]
    gamma_hat, *_ = np.linalg.lstsq(Xt, Qg, rcond=None)
    X, eta, Q, D, Y = draw(n, rng)
    eta_hat = Q - np.c_[np.ones(n), X] @ gamma_hat
    k = n_interior_knots(n)
    t, nb = basis(k)
    keep_hat = (eta_hat >= J[0]) & (eta_hat <= J[1])
    keep_or = (eta >= J[0]) & (eta <= J[1])
    th = fit(eta_hat, keep_hat, X, D, Y, t, nb)
    thS = fit(eta_hat, keep_or, X, D, Y, t, nb, extrapolate=True)
    a_hat = BSpline(t, th[:nb], 3)
    grid_w = np.linspace(L0, U0, 801)
    grid_j = np.linspace(J[0] + 1e-9, J[1] - 1e-9, 1601)
    ell = ell_vector(t, nb)
    return {
        "n": n, "K": nb,
        "sup_a_window": float(np.max(np.abs(a_hat(grid_w) - a_fun(grid_w)))),
        "sup_da_window": float(np.max(np.abs(a_hat.derivative()(grid_w) - da_fun(grid_w)))),
        "sup_a_J": float(np.max(np.abs(a_hat(grid_j) - a_fun(grid_j)))),
        "sup_da_J": float(np.max(np.abs(a_hat.derivative()(grid_j) - da_fun(grid_j)))),
        "beta2_err": float(np.linalg.norm(th[2 * nb + 2:] - BETA2)),
        "outer_crossing_scaled": float(np.sqrt(n) * abs(ell @ (th - thS))),
        "n_crossing": int(np.sum(keep_hat != keep_or)),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 8000, 32000, 128000])
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(n, args.seed + 1000 * i + r) for i, n in enumerate(args.sizes) for r in range(args.reps)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(replicate, tasks, chunksize=4))
    summary = {}
    for n in args.sizes:
        rs = [r for r in rows if r["n"] == n]
        summary[str(n)] = {"K": rs[0]["K"], "reps": len(rs)}
        for key in ("sup_a_window", "sup_da_window", "sup_a_J", "sup_da_J", "beta2_err",
                    "outer_crossing_scaled", "n_crossing"):
            v = np.array([r[key] for r in rs], dtype=float)
            summary[str(n)][key] = {"median": float(np.median(v)), "q90": float(np.quantile(v, 0.9))}
    print(json.dumps(summary, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "rows.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
