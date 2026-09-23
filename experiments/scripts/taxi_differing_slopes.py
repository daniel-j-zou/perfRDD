"""Differing-slopes taxi estimator: identify a level-dependent treatment effect.

Baseline PerfRDD models the effect as alpha(eta) only, which assumes T=gamma'X does
not modify the effect (T indep of W). The taxi percentage-menu effect depends on the
fare LEVEL Q=T+eta, violating that, so alpha-only is biased/degenerate for the policy
optimum. Letting the treated/control covariate slopes differ,

    Y = b(eta) + D*alpha(eta) + X'beta1 + (D*X)'beta2 + eps,   effect = alpha(eta)+X'beta2,

lets the effect depend on T and recovers the interior optimum.

Choices for honest stability: knots ~ n_treated^(1/5); ridge on the beta2 and spline
blocks with strength chosen by cross-validation on the outcome (never tuned to a target).

The threshold maximizes the differing-slopes utility of ``this_week.tex``,

    U_J(phi) = mean_i I_i [ (alpha_hat(eta_i) - c) Gbar_hat(phi - eta_i)
                            + beta2_hat' H_X_hat(phi - eta_i) ],   c = 0,

where Gbar_hat and H_X_hat integrate the Lebesgue-Gram spline projections g_hat
of the density of T_hat and p_X_hat(t) = E(X | T=t) f_T(t)
(``experiments.methods.weighted_tails``).  The alpha-only fit uses the same
utility with beta2 = 0.  As a robustness check the script also reports the
direct own-fare objective mean_i I_i (effect_i - c) 1{Q_i >= phi}.  Fares lie on
a $0.40 lattice, so each optimum is also reported as the implied fare cutoff.

CMT is external validation only (see DIFFERING_SLOPES.md) and must be compared on the
matched paper-restricted population, not the full sample.

Run:
    python -m experiments.scripts.taxi_differing_slopes
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from experiments.datasets.taxi.adapter import load_haggag_paci_vendor
from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.methods.weighted_tails import fit_weighted_tails, uj_utility

EPS = 0.10
THRESHOLD = 15.0
COST = 0.0
DENSITY_PAD = 0.5   # fixed density interval = observed T_hat range +/- this pad
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


def fit_effect(Q, X, Y, eta, D, info, interact, rng):
    """CV-select ridge and fit; return (alpha_hat(eta_i), beta2_hat, lambda)."""
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
    alpha = _eval_basis(eta, info) @ c[-nb:]
    beta2 = c[1 + nx:1 + 2 * nx] if interact else np.zeros(nx)
    return alpha, beta2, lam


def maximize_uj(eta, window, alpha, beta2, tails, phi_grid):
    """Grid search, then a bounded refinement, of the smooth U_J(phi)."""
    win = ((eta >= window[0]) & (eta <= window[1])).astype(float)
    alpha_minus_c = alpha - COST

    def U(phi):
        return uj_utility(phi, eta, win, alpha_minus_c, beta2, tails)

    values = np.array([U(p) for p in phi_grid])
    j = int(np.argmax(values))
    lo = phi_grid[max(j - 1, 0)]
    hi = phi_grid[min(j + 1, len(phi_grid) - 1)]
    res = minimize_scalar(lambda p: -U(p), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    phi = float(res.x) if -res.fun >= values[j] else float(phi_grid[j])
    boundary = j in (0, len(phi_grid) - 1)
    return phi, boundary


def own_fare_optimum(Q, eta, window, effect, phi_grid):
    """Robustness check: the direct objective mean_i I_i (effect_i - c) 1{Q_i >= phi}."""
    win = (eta >= window[0]) & (eta <= window[1])
    U = np.array([np.mean(np.where(win, (effect - COST) * (Q >= p), 0.0)) for p in phi_grid])
    return float(phi_grid[int(np.argmax(U))])


def fare_cutoff(phi, fares):
    """Smallest observed fare treated by the rule Q >= phi."""
    treated = fares[fares >= phi]
    return float(treated.min()) if treated.size else float("inf")


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
    support = (float(T.min()) - DENSITY_PAD, float(T.max()) + DENSITY_PAD)
    tails = fit_weighted_tails(T, X, support)          # g_hat and p_X_hat of T_hat
    fares = np.unique(Q)

    print(f"restricted VTS n={len(Y):,}  n_treated={int(D.sum()):,}  knots={kn}  "
          f"window=[{l0:.2f},{u0:.2f}]  density support=[{support[0]:.2f},{support[1]:.2f}] "
          f"({tails.n_basis} basis)  c={COST}")
    for label, interact in (("alpha only", False), ("differing slopes", True)):
        alpha, beta2, lam = fit_effect(Q, X, Y, eta, D, info, interact, rng)
        phi, boundary = maximize_uj(eta, (l0, u0), alpha, beta2, tails, phi_grid)
        own = own_fare_optimum(Q, eta, (l0, u0), alpha + X @ beta2, phi_grid)
        print(f"  {label:17s} U_J phi* = ${phi:.2f} (treat fares >= ${fare_cutoff(phi, fares):.2f})"
              f"{'  [grid boundary]' if boundary else ''}   own-fare check: ${own:.2f}"
              f" (>= ${fare_cutoff(own, fares):.2f})   cv lambda={lam}")
    print("  external validation (CMT / arithmetic / raw bins, matched population): ~$11-12")


if __name__ == "__main__":
    main()
