"""Adaptive efficient first stage: dropping the Gaussian assumption on eta.

Companion to ``ds_local_efficiency_sim.py``.  The stacked first stage there uses
least-squares score rows, which are efficient only for Gaussian eta.  Here the
score rows are replaced by the estimated location score of eta: on the stacking
part the slope update solves

    G_ad Delta_X = P_C[-rho_hat(eta_hat)(X - Xbar)] - P_C[S_hat R_hat Y / s_D^2],
    G_ad = I_hat P_C[(X - Xbar)(X - Xbar)'] + P_C[S_hat R_hat R_hat' / s_D^2],

where rho_hat = f_hat'/f_hat is a kernel estimate of the score of eta from the
pilot residuals (Bickel 1982), I_hat = P_A rho_hat(eta_hat)^2, and R_hat is the
arm-wise residual of c_hat_d'(eta_hat) X on the spline and slope columns.  For
rho_hat(e) = -e / s_eta^2 this is exactly the least-squares stacking.  The
slopes are adaptively estimable because X is independent of eta; the intercept
is not, and its loading in Abar is zero.

Per replication: first-stage slopes on one fold (OLS, LS-stacked, adaptive) and
the rotation-averaged thresholds of the four-role estimator with each.

Run:
    python -m experiments.scripts.ds_local_efficiency_adaptive_sim --design paper_mix \
        --n 16000 --reps 1000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm

from experiments.methods.perfrdd import _basis_params
from experiments.scripts.ds_local_efficiency import DESIGNS, Design, Population, sample_eta
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES,
    Sample,
    _design_matrix,
    _ls,
    _n_interior,
    _xt,
    efficient_first_stage,
    ols_first_stage,
    outcome_fit,
    rotation_threshold,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency_adaptive_sim"


def generate(design: Design, n: int, seed: int) -> Sample:
    """As ``ds_local_efficiency_sim.generate``, with the design's law of eta."""
    from numpy.polynomial import Polynomial
    rng = np.random.default_rng(seed)
    dim = len(design.gamma_x)
    X = rng.standard_normal((n, dim)) * np.sqrt(np.asarray(design.sigma_x))
    eta = sample_eta(design, rng, n)
    T = design.gamma0 + X @ np.asarray(design.gamma_x)
    Q = T + eta
    D = (Q > design.phi0).astype(float)
    a, b = Polynomial(design.a_coef), Polynomial(design.b_coef)
    sig = np.where(D == 1.0, design.sigma_arm[1], design.sigma_arm[0])
    sig = sig * np.exp(0.5 * design.noise_kappa * eta)
    Y = (b(eta) + X @ np.asarray(design.beta1)
         + D * (a(eta) + X @ np.asarray(design.beta2))
         + sig * rng.standard_normal(n))
    return Sample(X=X, Q=Q, D=D, Y=Y)


class KernelScore:
    """Kernel estimate of rho = f'/f from residuals, tabulated on a grid.

    Gaussian kernel with bandwidth sd * n^{-1/7} (the order suited to a density
    derivative); the score is set to zero where f_hat falls below 1% of its
    maximum, the usual trimming for adaptive estimation.
    """

    def __init__(self, resid: np.ndarray, n_grid: int = 801):
        r = np.asarray(resid, dtype=float)
        sd = float(np.std(r))
        h = sd * len(r) ** (-1.0 / 7.0)
        lo, hi = r.min() - 3 * h, r.max() + 3 * h
        self.grid = np.linspace(lo, hi, n_grid)
        z = (self.grid[:, None] - r[None, :]) / h
        k = norm.pdf(z)
        f = k.mean(axis=1) / h
        df = (-z * k).mean(axis=1) / h ** 2
        keep = f > 0.01 * f.max()
        self.rho = np.where(keep, df / np.where(keep, f, 1.0), 0.0)

    def __call__(self, e: np.ndarray) -> np.ndarray:
        return np.interp(e, self.grid, self.rho, left=0.0, right=0.0)


def adaptive_first_stage(s: Sample, idx: np.ndarray, J, rng) -> np.ndarray:
    parts = np.array_split(rng.permutation(idx), 3)
    estimates = []
    for r in range(3):
        pilot, deriv, stack = parts[r], parts[(r + 1) % 3], parts[(r + 2) % 3]
        g_p = ols_first_stage(s, pilot)
        res_p = s.Q[pilot] - _xt(s.X[pilot]) @ g_p
        score = KernelScore(res_p)
        I_hat = float(np.mean(score(res_p) ** 2))
        eta_hat = s.Q - _xt(s.X) @ g_p
        fit = outcome_fit(s, deriv, eta_hat, J)
        sd = np.sqrt(np.asarray(fit.s2))

        e, X, D, Y = eta_hat[stack], s.X[stack], s.D[stack], s.Y[stack]
        Xc = X - X.mean(axis=0)
        m = len(stack)
        inJ = (e >= J[0]) & (e <= J[1])
        Dj = D[inJ]
        n_int = _n_interior(int(min(Dj.sum(), inJ.sum() - Dj.sum())))
        Zy, _ = _design_matrix(e[inJ], X[inJ], Dj, _basis_params(n_int, J))
        xi = np.where(Dj == 1.0, fit.c_prime(1, e[inJ]), fit.c_prime(0, e[inJ]))
        W = xi[:, None] * X[inJ]
        R = W - Zy @ _ls(Zy, W)                       # arm-wise residual (Zy is arm-blocked)
        wy = 1.0 / np.where(Dj == 1.0, sd[1], sd[0]) ** 2
        G = I_hat * (Xc.T @ Xc) / m + (R.T * wy) @ R / m
        rhs = (-(score(e))[:, None] * Xc).sum(axis=0) / m - (R.T * wy) @ Y[inJ] / m
        dX = np.linalg.solve(G, rhs)
        d0 = float(np.mean(e - X @ dX))
        estimates.append(g_p + np.concatenate([[d0], dX]))
    return np.mean(estimates, axis=0)


def replicate(task):
    design_name, n, seed = task
    design = DESIGNS[design_name]
    s = generate(design, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    out = {"ols": [], "stack": [], "adapt": []}
    gammas = {}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        sub = np.random.default_rng(seed + 11 * k + 3)
        firsts = {
            "ols": ols_first_stage(s, folds["gamma"]),
            "stack": efficient_first_stage(s, folds["gamma"], design.J,
                                           np.random.default_rng(seed + 11 * k + 3)),
            "adapt": adaptive_first_stage(s, folds["gamma"], design.J, sub),
        }
        if k == 0:
            gammas = {key: val.tolist() for key, val in firsts.items()}
        for key, g in firsts.items():
            out[key].append(rotation_threshold(s, design, folds, g))
    row = {f"avg_{key}": float(np.mean(v)) for key, v in out.items()}
    row.update({f"gamma_{key}": v for key, v in gammas.items()})
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="paper_mix", choices=sorted(DESIGNS))
    parser.add_argument("--n", type=int, nargs="+", default=[16000])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    design = DESIGNS[args.design]
    pop = Population(design).decomposition()
    I = np.asarray(pop["I_gamma"])
    J_Y = np.asarray(pop["J_Y"])
    V_X = np.diag(design.sigma_x)
    pred_gamma = {
        "ols": np.diag(design.sigma_eta ** 2 * np.linalg.inv(V_X)),
        "stack": np.diag(np.linalg.inv(V_X / design.sigma_eta ** 2 + J_Y)),
        "adapt": np.diag(np.linalg.inv(I)),
    }
    pred_phi = {"ols": pop["V_rotated_ols"], "stack": pop["V_rotated_stack_ls"],
                "adapt": pop["V_rotated_eff"]}
    result = {"design": args.design, "population": pop, "cells": {}}
    for n in args.n:
        tasks = [(args.design, n, args.seed + 104_729 * n + r) for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(replicate, tasks, chunksize=2))
        m = n // 4
        cell = {"reps": len(rows)}
        for key in ("ols", "stack", "adapt"):
            G = np.array([r[f"gamma_{key}"] for r in rows])[:, 1:] - np.asarray(design.gamma_x)
            ph = np.array([r[f"avg_{key}"] for r in rows]) - pop["phi_star"]
            cell[key] = {"gamma_m_var": (m * G.var(axis=0, ddof=1)).tolist(),
                         "gamma_pred": pred_gamma[key].tolist(),
                         "phi_n_var": float(n * ph.var(ddof=1)), "phi_pred": pred_phi[key],
                         "phi_bias": float(ph.mean())}
        a = np.array([r["avg_stack"] for r in rows]); b = np.array([r["avg_adapt"] for r in rows])
        o = np.array([r["avg_ols"] for r in rows])
        cell["paired"] = {"ols_minus_adapt": float(n * (o.var(ddof=1) - b.var(ddof=1))),
                          "stack_minus_adapt": float(n * (a.var(ddof=1) - b.var(ddof=1))),
                          "pred_ols_minus_adapt": pred_phi["ols"] - pred_phi["adapt"],
                          "pred_stack_minus_adapt": pred_phi["stack"] - pred_phi["adapt"]}
        result["cells"][str(n)] = cell
        print(f"[{args.design}] N={n} reps={len(rows)}")
        for key in ("ols", "stack", "adapt"):
            c = cell[key]
            print(f"  {key:5s}: m Var slopes={np.round(c['gamma_m_var'], 3)} "
                  f"(pred {np.round(c['gamma_pred'], 3)}); N Var phi={c['phi_n_var']:.2f} "
                  f"(pred {c['phi_pred']:.2f}); bias={c['phi_bias']:+.4f}")
        print(f"  paired: ols-adapt={cell['paired']['ols_minus_adapt']:.2f} "
              f"(pred {cell['paired']['pred_ols_minus_adapt']:.2f}); stack-adapt="
              f"{cell['paired']['stack_minus_adapt']:.2f} (pred {cell['paired']['pred_stack_minus_adapt']:.2f})")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.design}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / (args.design + '_summary.json')}")


if __name__ == "__main__":
    main()
