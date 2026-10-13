"""Heteroskedasticity-weighted four-role estimator: does full weighting reach the bound?

Companion to ``ds_local_efficiency_sim.py`` and ``ds_local_efficiency_hetero.py``.
Under within-arm heteroskedasticity with variance sigma_D^2(eta) (condition H'
of ``this_week.tex``), the efficient estimator weights both

* the outcome fold's stacked spline regression, by 1/sigma_hat_D^2(eta_hat), and
* the outcome rows of the stacked first stage, by the same weights,

where sigma_hat_d^2 is a per-arm spline regression of squared residuals from an
unweighted pilot fit on eta_hat (floored at 5% of the arm's mean squared residual).
Four estimators are run on the same data and folds:

    paper      unweighted outcome fit, OLS first stage
    stacked    unweighted outcome fit, stacked first stage (arm-constant weights)
    w_ols      weighted outcome fit, OLS first stage
    w_stacked  weighted outcome fit, weighted stacked first stage

Population predictions come from ``hetero_variances``.

Run:
    python -m experiments.scripts.ds_local_efficiency_weighted_sim --design paper_hetero \
        --n 32000 --reps 1000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_local_efficiency import DESIGNS, Design, Population
from experiments.scripts.ds_local_efficiency_hetero import hetero_variances
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES,
    OutcomeFit,
    Sample,
    _design_matrix,
    _n_interior,
    _xt,
    efficient_first_stage,
    generate,
    ols_first_stage,
    outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency_weighted_sim"
VAR_BASIS = 6


@dataclass(frozen=True)
class VarianceFit:
    info: dict
    coef: tuple              # per-arm spline coefficients for E(R^2 | eta)
    floor: tuple

    def __call__(self, D: np.ndarray, e: np.ndarray) -> np.ndarray:
        e = np.clip(e, self.info["lo"], self.info["hi"])
        B = BSpline.design_matrix(e, self.info["t"], self.info["degree"]).toarray()
        out = np.empty(len(e))
        for arm in (0, 1):
            on = D == arm
            out[on] = np.maximum(B[on] @ self.coef[arm], self.floor[arm])
        return out


def _wls(A, y, w):
    Aw = A * w[:, None]
    return np.linalg.solve(A.T @ Aw, Aw.T @ y)


def fit_variance(s: Sample, idx, eta_hat, J, fit: OutcomeFit) -> VarianceFit:
    e = eta_hat[idx]
    keep = (e >= J[0]) & (e <= J[1])
    i = idx[keep]
    e, X, D, Y = eta_hat[i], s.X[i], s.D[i], s.Y[i]
    Z, nb = _design_matrix(e, X, D, fit.info)
    coef = np.concatenate([fit.w0, fit.d0, fit.w1, fit.d1])
    r2 = (Y - Z @ coef) ** 2
    info = _basis_params(VAR_BASIS - 4, J)
    B = BSpline.design_matrix(np.clip(e, J[0], J[1]), info["t"], 3).toarray()
    coefs, floors = [], []
    for arm in (0.0, 1.0):
        on = D == arm
        coefs.append(np.linalg.lstsq(B[on], r2[on], rcond=None)[0])
        floors.append(0.05 * float(np.mean(r2[on])))
    return VarianceFit(info=info, coef=tuple(coefs), floor=tuple(floors))


def weighted_outcome_fit(s: Sample, idx, eta_hat, J):
    """Pilot unweighted fit, variance function, then weighted refit (same fold)."""
    pilot = outcome_fit(s, idx, eta_hat, J)
    vf = fit_variance(s, idx, eta_hat, J, pilot)
    e = eta_hat[idx]
    keep = (e >= J[0]) & (e <= J[1])
    i = idx[keep]
    e, X, D, Y = eta_hat[i], s.X[i], s.D[i], s.Y[i]
    Z, nb = _design_matrix(e, X, D, pilot.info)
    coef = _wls(Z, Y, 1.0 / vf(D, e))
    dim = X.shape[1]
    fit = OutcomeFit(info=pilot.info, w0=coef[:nb], d0=coef[nb:nb + dim],
                     w1=coef[nb + dim:2 * nb + dim], d1=coef[2 * nb + dim:], s2=pilot.s2)
    return fit, vf


def weighted_first_stage(s: Sample, idx, J, rng) -> np.ndarray:
    """Stacked first stage with outcome rows weighted by 1/sigma_hat^2(D, eta_hat)."""
    parts = np.array_split(rng.permutation(idx), 3)
    estimates = []
    for r in range(3):
        pilot, deriv, stack = parts[r], parts[(r + 1) % 3], parts[(r + 2) % 3]
        g_p = ols_first_stage(s, pilot)
        res_p = s.Q[pilot] - _xt(s.X[pilot]) @ g_p
        s_eta2 = float(np.sum(res_p ** 2) / (len(pilot) - len(g_p)))
        eta_hat = s.Q - _xt(s.X) @ g_p
        fit, vf = weighted_outcome_fit(s, deriv, eta_hat, J)
        e, X, D, Y = eta_hat[stack], s.X[stack], s.D[stack], s.Y[stack]
        Xc = X - X.mean(axis=0)
        m = len(stack)
        inJ = (e >= J[0]) & (e <= J[1])
        Dj, ej, Xj = D[inJ], e[inJ], X[inJ]
        n_int = _n_interior(int(min(Dj.sum(), inJ.sum() - Dj.sum())))
        Zy, _ = _design_matrix(ej, Xj, Dj, _basis_params(n_int, J))
        w = 1.0 / vf(Dj, ej)
        xi = np.where(Dj == 1.0, fit.c_prime(1, ej), fit.c_prime(0, ej))
        W = xi[:, None] * Xj
        R = W - Zy @ _wls(Zy, W, w)                 # weighted arm-wise residual
        G = (Xc.T @ Xc) / (m * s_eta2) + (R.T * w) @ R / m
        rhs = (Xc.T @ e) / (m * s_eta2) - (R.T * w) @ Y[inJ] / m
        dX = np.linalg.solve(G, rhs)
        d0 = float(np.mean(e - X @ dX))
        estimates.append(g_p + np.concatenate([[d0], dX]))
    return np.mean(estimates, axis=0)


def threshold(s: Sample, design: Design, folds, gamma_hat, weighted: bool) -> float:
    J, eps, phi0 = design.J, design.eps, design.phi0
    eta_hat = s.Q - _xt(s.X) @ gamma_hat
    fit = (weighted_outcome_fit(s, folds["outcome"], eta_hat, J)[0] if weighted
           else outcome_fit(s, folds["outcome"], eta_hat, J))
    T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
    tails = fit_weighted_tails(T_idx, s.X[folds["index"]], design.T_support)
    l_hat = phi0 - float(np.quantile(T_idx, 1.0 - eps))
    u_hat = phi0 - float(np.quantile(T_idx, eps))
    e = eta_hat[folds["utility"]]
    e = e[(e >= l_hat) & (e <= u_hat)]
    amc, b2 = fit.a(e) - design.cost, fit.beta2

    def value(phi):
        t = phi - e
        return float(np.mean(amc * tails.survival(t) + tails.weighted_tail(t, b2)))

    lo, hi = design.policy
    grid = np.linspace(lo, hi, 61)
    k = int(np.argmax([value(x) for x in grid]))
    res = minimize_scalar(lambda x: -value(x), method="bounded",
                          bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 60)]),
                          options={"xatol": 1e-9})
    return float(res.x)


VARIANTS = ("paper", "stacked", "w_ols", "w_stacked")


def replicate(task):
    name, n, seed = task
    design = DESIGNS[name]
    s = generate(design, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    out = {v: [] for v in VARIANTS}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        g_ols = ols_first_stage(s, folds["gamma"])
        g_st = efficient_first_stage(s, folds["gamma"], design.J, np.random.default_rng(seed + 11 * k + 3))
        g_wst = weighted_first_stage(s, folds["gamma"], design.J, np.random.default_rng(seed + 11 * k + 3))
        out["paper"].append(threshold(s, design, folds, g_ols, False))
        out["stacked"].append(threshold(s, design, folds, g_st, False))
        out["w_ols"].append(threshold(s, design, folds, g_ols, True))
        out["w_stacked"].append(threshold(s, design, folds, g_wst, True))
    row = {f"avg_{v}": float(np.mean(out[v])) for v in VARIANTS}
    row.update({f"fixed_{v}": out[v][0] for v in VARIANTS})
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="paper_hetero", choices=sorted(DESIGNS))
    parser.add_argument("--n", type=int, nargs="+", default=[32000])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    design = DESIGNS[args.design]
    pop = Population(design, allow_hetero=True)
    pred = hetero_variances(pop, design.noise_kappa)
    pred_rot = {"paper": pred["paper"], "stacked": pred["stacked"],
                "w_ols": pred["weighted_ols"], "w_stacked": pred["bound"]}
    result = {"design": args.design, "prediction": pred, "cells": {}}
    for n in args.n:
        tasks = [(args.design, n, args.seed + 104_729 * n + r) for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(replicate, tasks, chunksize=2))
        cell = {"reps": len(rows)}
        for v in VARIANTS:
            a = np.array([r[f"avg_{v}"] for r in rows]) - pop.phi
            f = np.array([r[f"fixed_{v}"] for r in rows]) - pop.phi
            cell[v] = {"rot_n_var": float(n * a.var(ddof=1)), "rot_pred": pred_rot[v],
                       "fixed_n_var": float(n * f.var(ddof=1)), "bias": float(a.mean())}
        base = np.array([r["avg_paper"] for r in rows])
        cell["paired_vs_paper"] = {
            v: float(n * (base.var(ddof=1) - np.array([r[f"avg_{v}"] for r in rows]).var(ddof=1)))
            for v in VARIANTS[1:]}
        result["cells"][str(n)] = cell
        print(f"[{args.design}] N={n} reps={len(rows)}")
        for v in VARIANTS:
            c = cell[v]
            print(f"  {v:9s}: rotated N Var={c['rot_n_var']:.2f} (pred {c['rot_pred']:.2f}); "
                  f"fixed={c['fixed_n_var']:.1f}; bias={c['bias']:+.4f}")
        print(f"  paired (paper minus variant): {cell['paired_vs_paper']}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.design}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / (args.design + '_summary.json')}")


if __name__ == "__main__":
    main()
