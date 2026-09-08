"""Differing-slopes taxi estimator: identify a level-dependent treatment effect.

Baseline PerfRDD models the effect as alpha(eta) only, which assumes T=gamma'X does
not modify the effect (T indep of W). The taxi percentage-menu effect depends on the
fare LEVEL Q=T+eta, violating that, so alpha-only is biased/degenerate for the policy
optimum. Letting the treated/control covariate slopes differ,

    Y = b(eta) + D*alpha(eta) + X'beta1 + (D*X)'beta2 + eps,   effect = alpha(eta)+X'beta2,

lets the effect depend on T and recovers the interior optimum.

Choices for honest stability: knots ~ n_treated^(1/5); ridge on the beta2 and spline
blocks with strength chosen by cross-validation on the outcome (never tuned to a target).
Utility is the empirical U(phi) = mean_{eta in window} (effect)*1{Q>=phi}; c=0.

CMT is external validation only (see DIFFERING_SLOPES.md) and must be compared on the
matched paper-restricted population, not the full sample.

Run:
    python -m experiments.scripts.taxi_differing_slopes
"""
from __future__ import annotations

import numpy as np

from experiments.datasets.taxi.adapter import load_haggag_paci_vendor
from experiments.methods.perfrdd import _basis_params, _eval_basis

EPS = 0.10
THRESHOLD = 15.0
LAMBDA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def _clean(sample):
    """Restricted-sample rows with data-entry errors (tip>fare and tip>=$10) removed."""
    Q = np.asarray(sample.Q, float)
    X = np.asarray(sample.X, float)
    tip = np.asarray(sample.Y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Q > 0, tip / Q, np.nan)
    weird = np.isfinite(ratio) & (ratio > 1.0) & (tip >= 10.0)
    keep = np.isfinite(ratio) & (Q > 0) & (~weird)
    return Q[keep], X[keep], tip[keep]


def _solve(H, y, penalize, lam):
    n = len(y)
    P = np.diag(np.where(penalize, lam / np.sqrt(n), 0.0))
    return np.linalg.solve(H.T @ H + n * P, H.T @ y)


def _design(Q, X, eta, D, info, interact):
    Phi = _eval_basis(eta, info)
    nx = X.shape[1]
    cols = [np.ones(len(Q)), X]
    pen = [False] + [False] * nx
    if interact:
        cols.append(D[:, None] * X)
        pen += [True] * nx
    cols += [Phi, D[:, None] * Phi]
    pen += [True] * Phi.shape[1] * 2
    return np.column_stack(cols), np.array(pen), Phi.shape[1], nx


def fit_optimum(Q, X, Y, eta, D, window, phi_grid, info, interact, rng):
    """CV-select ridge, fit, return (phi_star, effect_on_sample, alpha_on_grid)."""
    H, pen, nb, nx = _design(Q, X, eta, D, info, interact)
    tr = rng.random(len(Y)) < 0.8
    best = (np.inf, None)
    for lam in LAMBDA_GRID:
        c = _solve(H[tr], Y[tr], pen, lam)
        mse = np.mean((Y[~tr] - H[~tr] @ c) ** 2)
        if mse < best[0]:
            best = (mse, lam)
    lam = best[1]
    c = _solve(H, Y, pen, lam)
    omega_treat = c[-nb:]
    eff = _eval_basis(eta, info) @ omega_treat
    if interact:
        eff = eff + X @ c[1 + nx:1 + 2 * nx]
    win = (eta >= window[0]) & (eta <= window[1])
    U = np.array([np.mean(np.where(win, eff * (Q >= p), 0.0)) for p in phi_grid])
    return float(phi_grid[int(np.argmax(U))]), lam


def main() -> None:
    rng = np.random.default_rng(0)
    vts = load_haggag_paci_vendor("VTS")
    Q, X, Y = _clean(vts)                      # outcome = tip dollars
    Xd = np.column_stack((np.ones(len(Q)), X))
    gamma, *_ = np.linalg.lstsq(Xd, Q, rcond=None)
    eta = Q - Xd @ gamma
    T = Q - eta
    l0 = THRESHOLD - np.quantile(T, 1 - EPS)
    u0 = THRESHOLD - np.quantile(T, EPS)
    lo, hi = np.percentile(eta, 0.5), np.percentile(eta, 99.5)
    l0, u0 = max(min(l0, u0), lo), min(max(l0, u0), hi)
    D = (Q >= THRESHOLD).astype(float)
    kn = max(4, int(round(int(D.sum()) ** (1 / 5)))) + 1   # ~ n_treated^{1/5}
    info = _basis_params(kn, (lo, hi))
    phi_grid = np.linspace(0.0, THRESHOLD + 3 * np.std(Q), 301)

    print(f"restricted VTS n={len(Y):,}  n_treated={int(D.sum()):,}  knots={kn}  "
          f"window=[{l0:.2f},{u0:.2f}]")
    pA, lA = fit_optimum(Q, X, Y, eta, D, (l0, u0), phi_grid, info, interact=False, rng=rng)
    pB, lB = fit_optimum(Q, X, Y, eta, D, (l0, u0), phi_grid, info, interact=True, rng=rng)
    print(f"  alpha only        phi* = ${pA:.2f}  (cv lambda={lA})")
    print(f"  differing slopes  phi* = ${pB:.2f}  (cv lambda={lB})")
    print("  external validation (CMT / arithmetic / raw bins, matched population): ~$11-12")


if __name__ == "__main__":
    main()
