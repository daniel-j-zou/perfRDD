"""Differing slopes under a nonlinear effect in X: the pseudo-target.

Companion to the misspecification remark in ``this_week.tex`` (2026-10-13).
When E(W | X, eta) is not linear in X, the plug-in estimator converges to
the maximizer phi-dagger of the pseudo-utility built from the population
least-squares projection of E(Y | D, eta, X) on the fitted outcome model,

    U-dagger(phi) = E[{a-dagger(eta) - c + X' beta2-dagger} 1(Q > phi) I0],

not to the maximizer of the true supported utility.  This script computes
both, for the quadratic-misspecification scenario of
``differing_slopes_full_pipeline.py`` (treated effect gains 0.9 (X1^2 - 1)):

* the projection coefficients by chunked Monte Carlo on 2e7 draws;
* phi-dagger for the pipeline's linear-in-eta outcome model, to compare with
  the simulated estimates, which settle near 0.405 (log entry on
  ``differing_slopes_full_pipeline.py``);
* phi-dagger for the spline-in-eta model of the theorem, on J = [-1.7, 1.7];
* the true supported optimum, by Gauss-Legendre quadrature on the window.

Run:
    python -m experiments.scripts.ds_misspecification_target
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_misspecification_target"

A0, A1, B0, B1 = 0.35, 0.90, 0.20, 0.60
BETA1 = np.array([0.30, -0.20])
BETA2 = np.array([0.80, 0.25])
Q2 = 0.90
COST = 0.25
EPS = 0.10
L0, U0 = -stats.norm.ppf(1 - EPS), stats.norm.ppf(1 - EPS)
J = (-1.7, 1.7)
GL_X, GL_W = np.polynomial.legendre.leggauss(400)
E_NODES = 0.5 * (U0 - L0) * GL_X + 0.5 * (U0 + L0)
E_W = 0.5 * (U0 - L0) * GL_W


def draw(n, rng):
    X = rng.standard_normal((n, 2))
    eta = rng.standard_normal(n)
    D = (X[:, 0] + eta > 0).astype(float)
    effect = A0 + A1 * eta + X @ BETA2 + Q2 * (X[:, 0] ** 2 - 1)
    Y = B0 + B1 * eta + X @ BETA1 + D * effect + 0.5 * rng.standard_normal(n)
    return X, eta, D, Y


def spline_basis(e, nint=12):
    t_int = np.linspace(J[0], J[1], nint + 2)
    t = np.r_[[J[0]] * 3, t_int, [J[1]] * 3]
    nb = len(t) - 4
    out = np.empty((len(e), nb))
    for j in range(nb):
        c = np.zeros(nb)
        c[j] = 1.0
        out[:, j] = BSpline(t, c, 3, extrapolate=False)(e)
    return np.nan_to_num(out), t, nb


def projection(n_total, chunk, seed, model):
    rng = np.random.default_rng(seed)
    xtx = None
    xty = None
    for _ in range(n_total // chunk):
        X, eta, D, Y = draw(chunk, rng)
        if model == "linear_eta":
            Z = np.column_stack([np.ones(chunk), eta, X, D, D * eta, D[:, None] * X])
            keep = np.ones(chunk, bool)
        else:
            keep = (eta >= J[0]) & (eta <= J[1])
            B, _, _ = spline_basis(eta[keep])
            Dk = D[keep]
            Z = np.column_stack([Dk[:, None] * B, B, X[keep], Dk[:, None] * X[keep]])
            Y = Y[keep]
        if xtx is None:
            xtx = Z.T @ Z
            xty = Z.T @ Y
        else:
            xtx += Z.T @ Z
            xty += Z.T @ Y
    return np.linalg.solve(xtx, xty)


def tau(phi, a_fun, lam_fun):
    w = stats.norm.pdf(phi - E_NODES) * stats.norm.pdf(E_NODES) * E_W
    return float(np.sum(w * (a_fun(E_NODES) - COST + lam_fun(phi - E_NODES))) / np.sum(w))


def zero(a_fun, lam_fun):
    """Zeros of tau on [-3, 3] where tau crosses from - to + (local maxima of U)."""
    grid = np.linspace(-3, 3, 241)
    t = np.array([tau(p, a_fun, lam_fun) for p in grid])
    ks = np.where((t[:-1] < 0) & (t[1:] >= 0))[0]
    roots = [optimize.brentq(lambda p: tau(p, a_fun, lam_fun), grid[k], grid[k + 1], xtol=1e-12) for k in ks]
    downs = np.where((t[:-1] > 0) & (t[1:] <= 0))[0]
    if len(roots) != 1 or len(downs) > 0:
        print("note: local maxima", roots, "local minima near", [float(grid[k]) for k in downs],
              "tau at ends", float(t[0]), float(t[-1]))
    return roots[0] if len(roots) == 1 else roots


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # True supported optimum: E(X1^2 | T = t) = t^2 because T = X1.
    true_star = zero(lambda e: A0 + A1 * e, lambda t: BETA2[0] * t + Q2 * (t ** 2 - 1))
    linear_star = zero(lambda e: A0 + A1 * e, lambda t: BETA2[0] * t)

    th = projection(args.n, args.chunk, args.seed, "linear_eta")
    a_lin = (th[4], th[5])
    b2_lin = th[6:8]
    lin_dagger = zero(lambda e: a_lin[0] + a_lin[1] * e, lambda t: b2_lin[0] * t)

    th_s = projection(args.n, args.chunk, args.seed + 1, "spline_eta")
    _, t, nb = spline_basis(np.array([0.0]))
    a_spl = BSpline(t, th_s[:nb], 3)
    b2_spl = th_s[2 * nb + 2:]
    spl_dagger = zero(lambda e: a_spl(e), lambda t: b2_spl[0] * t)

    out = {
        "true_supported_optimum": true_star,
        "optimum_without_quadratic_term": linear_star,
        "linear_eta_model": {"a_dagger": [float(a_lin[0]), float(a_lin[1])],
                             "beta2_dagger": b2_lin.tolist(), "phi_dagger": lin_dagger},
        "spline_eta_model": {"beta2_dagger": b2_spl.tolist(), "phi_dagger": spl_dagger,
                             "a_dagger_at": {str(e): float(a_spl(e)) for e in (-1.0, 0.0, 1.0)}},
        "simulated_linear_variants_settle_near": 0.405,
    }
    print(json.dumps(out, indent=1))
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
