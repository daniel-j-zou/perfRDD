"""Cross-dataset differing-slopes screen with the U_J estimator.

For each dataset, compare the alpha(eta)-only effect model with the
differing-slopes model W = alpha(eta) + X'beta2.  Both thresholds maximize

    U_J(phi) = mean_i I_i [ (alpha_hat(eta_i) - c) Gbar_hat(phi - eta_i)
                            + beta2_hat' H_X_hat(phi - eta_i) ],   c = 0,

with beta2 = 0 for the alpha-only model.  Gbar_hat and H_X_hat integrate the
Lebesgue-Gram spline projections g_hat and p_X_hat of the fitted index T_hat
(``experiments.methods.weighted_tails``).  The outcome fit, trim window, knot
rule and cross-validated ridge are those of ``taxi_differing_slopes.py``.
The earlier own-score objective mean_i I_i (effect_i - c) 1{Q_i >= phi} is
reported alongside; under X independent of eta both target the same utility,
so a gap between them flags a failure of that assumption.

Below-cutoff designs (treated when Q < cutoff) are mirrored (Q -> -Q), run
through the above-cutoff pipeline, and mapped back.  Covariates are
standardized, so beta2 is per standard deviation.  The candidate thresholds
are the 0.5%--99.5% range of Q.  Each optimum is reported with the share of
the trim window it treats; a share above 99% or below 1% (or a maximizer at an
end of the range) is flagged as a boundary solution, since U_J can be flat on
a plateau where everyone or no one in the window is treated.

Run:
    python -m experiments.scripts.differing_slopes_screen --out ../outputs/differing_slopes_screen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from experiments._core.registry import load
from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails, uj_utility
from experiments.scripts.taxi_differing_slopes import (
    _clean as _taxi_clean,
    fit_effect,
    own_fare_optimum,
)
from experiments.datasets.taxi.adapter import load_haggag_paci_vendor

EPS = 0.10
COST = 0.0
N_GRID = 241
DENSITY_QUANTILES = (0.0005, 0.9995)
DISCRETE_LIMIT = 200   # report the implied cutoff when Q has at most this many values

# Own-score (1{Q >= phi}) screen from experiments/datasets/taxi/DIFFERING_SLOPES.md.
PREVIOUS = {
    "taxi_full": (4.2, 9.2),
    "taxi_restricted": (0.0, 12.5),
    "oulad": (51.2, 36.0),
    "lending_default": (45.45, 45.45),
    "gpa": (2.6, 2.6),
    "nhanes": (3.4, 3.4),
}


def _registry(name: str) -> Callable[[], tuple]:
    def loader():
        sample = load(name)
        return (np.asarray(sample.Q, float), np.asarray(sample.X, float),
                np.asarray(sample.Y, float), float(sample.threshold))
    return loader


def _taxi_restricted():
    Q, X, Y = _taxi_clean(load_haggag_paci_vendor("VTS"))
    return Q, X, Y, 15.0


DATASETS = {
    "taxi_full": (_registry("taxi"), "above"),
    "taxi_restricted": (_taxi_restricted, "above"),
    "oulad": (_registry("oulad"), "above"),
    "lending_default": (_registry("lending_default"), "above"),
    "gpa": (_registry("gpa"), "below"),
    "nhanes": (_registry("nhanes"), "above"),
}


def _standardize(X: np.ndarray) -> np.ndarray:
    scale = X.std(axis=0)
    keep = scale > 0
    return (X[:, keep] - X[:, keep].mean(axis=0)) / scale[keep]


def _maximize(U, grid):
    values = np.array([U(p) for p in grid])
    j = int(np.argmax(values))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    res = minimize_scalar(lambda p: -U(p), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    phi = float(res.x) if -res.fun >= values[j] else float(grid[j])
    return phi, j in (0, len(grid) - 1)


def screen_dataset(name: str, seed: int = 0) -> dict:
    loader, direction = DATASETS[name]
    Q, X, Y, thr = loader()
    ok = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y = Q[ok], _standardize(X[ok]), Y[ok]
    sign = 1.0 if direction == "above" else -1.0
    Qs, thr_s = sign * Q, sign * thr                  # treated iff Qs >= thr_s

    Xd = np.column_stack((np.ones(len(Qs)), X))
    gamma, *_ = np.linalg.lstsq(Xd, Qs, rcond=None)
    eta = Qs - Xd @ gamma
    T = Qs - eta
    l0 = thr_s - np.quantile(T, 1 - EPS)
    u0 = thr_s - np.quantile(T, EPS)
    lo, hi = np.percentile(eta, 0.5), np.percentile(eta, 99.5)
    l0, u0 = max(min(l0, u0), lo), min(max(l0, u0), hi)
    D = (Qs >= thr_s).astype(float)
    kn = max(4, int(round(int(D.sum()) ** (1 / 5)))) + 1
    info = _basis_params(kn, (lo, hi))
    grid = np.linspace(*np.quantile(Qs, [0.005, 0.995]), N_GRID)
    support = tuple(np.quantile(T, DENSITY_QUANTILES))
    tails = fit_weighted_tails(T, X, support)
    win = ((eta >= l0) & (eta <= u0)).astype(float)
    r2 = 1.0 - eta.var() / Qs.var()

    rng = np.random.default_rng(seed)
    result = {
        "dataset": name, "direction": direction, "cutoff": thr, "n": int(len(Q)),
        "n_treated": int(D.sum()), "distinct_Q": int(len(np.unique(Q))),
        "first_stage_r2": float(r2), "window_eta": [float(l0), float(u0)],
        "window_share": float(win.mean()), "density_support": list(map(float, support)),
        "density_basis": tails.n_basis, "outcome_knots": kn,
    }
    for label, interact in (("alpha_only", False), ("differing_slopes", True)):
        alpha, beta2, lam = fit_effect(Qs, X, Y, eta, D, info, interact, rng)
        U = lambda p, a=alpha, b=beta2: uj_utility(p, eta, win, a - COST, b, tails)
        phi, at_end = _maximize(U, grid)
        own = own_fare_optimum(Qs, eta, (l0, u0), alpha + X @ beta2, grid)
        in_win = win > 0
        share_uj = float(np.mean(tails.survival(phi - eta[in_win])) / max(tails.survival(-np.inf), 1e-12))
        share_own = float(np.mean(Qs[in_win] >= own))
        result[label] = {
            "phi_uj": float(sign * phi),
            "treated_share_uj": share_uj,
            "boundary_uj": bool(at_end or share_uj > 0.99 or share_uj < 0.01),
            "phi_own": float(sign * own),
            "treated_share_own": share_own,
            "boundary_own": bool(own <= grid[0] or own >= grid[-1]
                                 or share_own > 0.99 or share_own < 0.01),
            "ridge_lambda": lam,
            "beta2": [float(v) for v in beta2],
            "sd_alpha_in_window": float(np.std(alpha[in_win])),
            "sd_xbeta2_in_window": float(np.std((X @ beta2)[in_win])),
        }
        if result["distinct_Q"] <= DISCRETE_LIMIT:
            values = np.unique(Qs)
            treated = values[values >= phi]
            first = float(treated.min()) if treated.size else float("inf")
            result[label]["implied_cutoff"] = float(sign * first)
    result["previous_own_score"] = dict(zip(("alpha_only", "differing_slopes"), PREVIOUS[name]))
    return result


def _fmt(item: dict, key: str, bkey: str, skey: str) -> str:
    return f"{item[key]:.2f} [{100 * item[skey]:.0f}%]{' (bnd)' if item[bkey] else ''}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.datasets:
        res = screen_dataset(name)
        (args.out / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        a, d = res["alpha_only"], res["differing_slopes"]
        line = (f"| {name} | {res['direction']} | {res['cutoff']:g} | "
                f"{_fmt(a, 'phi_uj', 'boundary_uj', 'treated_share_uj')} | "
                f"{_fmt(d, 'phi_uj', 'boundary_uj', 'treated_share_uj')} | "
                f"{_fmt(a, 'phi_own', 'boundary_own', 'treated_share_own')} | "
                f"{_fmt(d, 'phi_own', 'boundary_own', 'treated_share_own')} | "
                f"{res['previous_own_score']['alpha_only']:g} / {res['previous_own_score']['differing_slopes']:g} |")
        rows.append(line)
        print(line, flush=True)
    header = ("Entries: optimum [share of trim window treated]; (bnd) = boundary solution.\n\n"
              "| dataset | direction | cutoff | alpha-only U_J | diff. slopes U_J | "
              "alpha-only own-score | diff. slopes own-score | previous screen (own-score) |\n"
              "|---|---|---:|---:|---:|---:|---:|---:|")
    (args.out / "screen_table.md").write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
