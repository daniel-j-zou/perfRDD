"""Check the cross-fitting variance claims of Mukherjee, Banerjee and Ritov.

Mukherjee, Banerjee and Ritov (Bernoulli 2026, Theorem 2.8) estimate a constant
treatment effect alpha_0 in

    Y = alpha_0 1{Q > 0} + X'beta_0 + b(eta) + eps,   Q = Z'gamma_0 + eta,

with three folds: D1 estimates gamma by OLS, D2 estimates b' by a spline
partial-linear fit, and D3 estimates alpha from the stacked regression (2.7),
which re-estimates (gamma_hat - gamma_0) as a coefficient.  They claim

    sqrt(n)(alpha_hat - alpha_0)     -> N(0, 3 V_tau),   single split,
    sqrt(n)(alpha_bar_hat - alpha_0) -> N(0, V_tau),     three-way rotation,

with V_tau = e1' Omega_tau^{-1} Omega*_tau Omega_tau^{-1} e1, because each
fold's leading term depends only on its own D3 observations.

This script computes V_tau and runs a Monte Carlo of the single-split,
rotated, and no-split estimators, including the correlation between the three
rotated estimates and the dependence of a single-split estimate on its
first-stage error.  Two designs (Z, eta, eps uniform(-1, 1), gamma_0 = (-1, 1),
alpha_0 = 2):

``paper``     their Section 4.1: X = Z, beta_0 = (1, 2), b(x) = x^3/3.  Here the
              first-stage loading is nearly zero, so the design cannot tell
              their corrected estimator from a naive plug-in.
``excluded``  X = Z_1 only (Z_2 is excluded from the outcome), beta_0 = 1,
              b(x) = x + x^3/3.  The first-stage error now matters.

Run:
    python -m experiments.scripts.mbr_crossfit_check --reps 2000 --n 5000 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline

ALPHA0 = 2.0
GAMMA0 = np.array([-1.0, 1.0])
DESIGNS = {
    # name: (dim X, beta_0, b, b')
    "paper": (2, np.array([1.0, 2.0]), lambda e: e ** 3 / 3.0, lambda e: e ** 2),
    "excluded": (1, np.array([1.0]), lambda e: e + e ** 3 / 3.0, lambda e: 1.0 + e ** 2),
}
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "mbr_crossfit_check"


def theoretical_variance(tau: float, design: str = "paper", order: int = 80) -> dict:
    """V_tau = e1' Omega^{-1} Omega* Omega^{-1} e1 by Gauss-Legendre quadrature."""
    x, w = leggauss(order)
    w = w / 2.0                                   # uniform(-1, 1) density
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    WX = np.outer(w, w)
    index = GAMMA0[0] * X1 + GAMMA0[1] * X2
    e_nodes, e_w = leggauss(400)
    e_w = e_w / 2.0
    p1, _, _, b_prime = DESIGNS[design]
    d = 1 + p1 + 2
    omega = np.zeros((d, d))
    for eta, weight in zip(e_nodes, e_w):
        if abs(eta) > tau:
            continue
        S = (index + eta > 0).astype(float)
        comps = [S] + [X1, X2][:p1] + [X1 * b_prime(eta), X2 * b_prime(eta)]
        means = [float(np.sum(WX * c)) for c in comps]
        cov = np.array([[float(np.sum(WX * a * b)) - ma * mb
                         for b, mb in zip(comps, means)]
                        for a, ma in zip(comps, means)])
        omega += weight * cov
    sigma2 = 1.0 / 3.0                             # var(eps) = var(eta) = 1/3
    second = np.zeros((d, d))
    second[1 + p1:, 1 + p1:] = np.eye(2) / 3.0     # cov(Z)
    Omega = omega + second
    Omega_star = sigma2 * omega + (1.0 / 3.0) * second
    inv = np.linalg.inv(Omega)
    return {"tau": tau, "design": design, "V": float((inv @ Omega_star @ inv)[0, 0]),
            "Omega": Omega.tolist()}


def _basis(K: int, tau: float):
    knots = np.concatenate([np.repeat(-tau, 4), np.linspace(-tau, tau, K + 1)[1:-1],
                            np.repeat(tau, 4)])
    nb = len(knots) - 4
    spline = BSpline(knots, np.eye(nb), 3, extrapolate=False)
    return spline, spline.derivative(1)


def generate(n: int, rng, design: str = "paper"):
    p1, beta0, b, _ = DESIGNS[design]
    Z = rng.uniform(-1, 1, (n, 2))
    X = Z[:, :p1]
    eta = rng.uniform(-1, 1, n)
    eps = rng.uniform(-1, 1, n)
    Q = Z @ GAMMA0 + eta
    S = (Q > 0).astype(float)
    Y = ALPHA0 * S + X @ beta0 + b(eta) + eps
    return {"X": X, "Z": Z, "Q": Q, "S": S, "Y": Y}


def estimate(data, i1, i2, i3, K: int, tau: float, plug_in: bool = False):
    """One split: gamma from i1, b' from i2, alpha from the stacked fit on i3.

    ``plug_in=True`` is a contrast that is not in the paper: it drops the
    b'(eta_hat) Z correction column and the second equation, so the D1
    first-stage error is not re-estimated on D3.  Returns (alpha_hat,
    gamma_hat - gamma_0).
    """
    X, Z, Q, S, Y = data["X"], data["Z"], data["Q"], data["S"], data["Y"]
    p1 = X.shape[1]
    gamma = np.linalg.lstsq(Z[i1], Q[i1], rcond=None)[0]
    eta_hat = Q - Z @ gamma
    spline, dspline = _basis(K, tau)

    keep2 = i2[np.abs(eta_hat[i2]) <= tau]
    N2 = spline(eta_hat[keep2])
    design2 = np.column_stack((S[keep2], X[keep2], N2))
    omega_b = np.linalg.lstsq(design2, Y[keep2], rcond=None)[0][1 + p1:]

    keep3 = i3[np.abs(eta_hat[i3]) <= tau]
    N3 = spline(eta_hat[keep3])
    if plug_in:
        design = np.column_stack((S[keep3], X[keep3], N3))
        return float(np.linalg.lstsq(design, Y[keep3], rcond=None)[0][0]), gamma - GAMMA0
    bp = dspline(eta_hat[keep3]) @ omega_b
    top = np.column_stack((S[keep3], X[keep3], bp[:, None] * Z[keep3], N3))
    bottom = np.column_stack((np.zeros((len(i3), 1 + p1)), -Z[i3],
                              np.zeros((len(i3), N3.shape[1]))))
    design = np.vstack((top, bottom))
    response = np.concatenate((Y[keep3], eta_hat[i3]))
    return float(np.linalg.lstsq(design, response, rcond=None)[0][0]), gamma - GAMMA0


def replicate(task):
    n, seed, K, tau, design = task
    rng = np.random.default_rng(seed)
    data = generate(n, rng, design)
    folds = np.array_split(rng.permutation(n), 3)
    rotations = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]     # (gamma, b', alpha) folds
    fits = [estimate(data, folds[a], folds[b], folds[c], K, tau) for a, b, c in rotations]
    single = [f[0] for f in fits]
    everything = np.arange(n)
    no_split = estimate(data, everything, everything, everything, K, tau)[0]
    naive = [estimate(data, folds[a], folds[b], folds[c], K, tau, plug_in=True)[0]
             for a, b, c in rotations]
    # columns: 0-2 single, 3 rotated, 4 no split, 5-6 first-stage error of
    # rotation 0, 7-9 naive plug-in single, 10 naive rotated
    return (single + [float(np.mean(single)), no_split] + list(fits[0][1])
            + naive + [float(np.mean(naive))])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[5000])
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--design", choices=sorted(DESIGNS), default="paper")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    theory = theoretical_variance(args.tau, args.design)
    V = theory["V"]
    result = {"theory": theory, "design": args.design, "K": args.K, "tau": args.tau,
              "reps": args.reps, "cells": {}}
    print(f"design = {args.design}; theory: V_tau = {V:.4f}  "
          f"(single split 3 V = {3 * V:.4f})   tau = {args.tau}")
    for n in args.n:
        tasks = [(n, args.seed + 7919 * n + r, args.K, args.tau, args.design)
                 for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            draws = np.array(list(pool.map(replicate, tasks, chunksize=20)))
        err = draws[:, :5] - ALPHA0
        corr = np.corrcoef(err[:, :3].T)
        dgamma = draws[:, 5:7]
        naive = draws[:, 7:11] - ALPHA0

        def r2_on_first_stage(values):
            design = np.column_stack((np.ones(len(values)), dgamma))
            fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
            return float(1.0 - np.var(values - fitted) / np.var(values))

        se = np.sqrt(2.0 / (args.reps - 1))
        cell = {
            "n_var_single_each": [float(n * np.var(err[:, j], ddof=1)) for j in range(3)],
            "n_var_single_mean": float(n * np.mean(np.var(err[:, :3], axis=0, ddof=1))),
            "n_var_rotated": float(n * np.var(err[:, 3], ddof=1)),
            "n_var_no_split": float(n * np.var(err[:, 4], ddof=1)),
            "bias": [float(v) for v in err.mean(axis=0)],
            "rmse_rotated": float(np.sqrt(np.mean(err[:, 3] ** 2))),
            "rmse_no_split": float(np.sqrt(np.mean(err[:, 4] ** 2))),
            "corr_between_rotations": [float(corr[0, 1]), float(corr[0, 2]), float(corr[1, 2])],
            "relative_mc_se_of_variance": float(se),
            "r2_single_on_first_stage_error": r2_on_first_stage(err[:, 0]),
            "naive_plug_in": {
                "n_var_single_mean": float(n * np.mean(np.var(naive[:, :3], axis=0, ddof=1))),
                "n_var_rotated": float(n * np.var(naive[:, 3], ddof=1)),
                "corr_between_rotations": [float(v) for v in
                                           np.corrcoef(naive[:, :3].T)[np.triu_indices(3, 1)]],
                "r2_single_on_first_stage_error": r2_on_first_stage(naive[:, 0]),
                "bias_rotated": float(naive[:, 3].mean()),
            },
        }
        cell["ratio_single_over_rotated"] = cell["n_var_single_mean"] / cell["n_var_rotated"]
        result["cells"][str(n)] = cell
        print(f"n={n}: N Var single = {cell['n_var_single_mean']:.3f} (theory {3 * V:.3f}); "
              f"rotated = {cell['n_var_rotated']:.3f} (theory {V:.3f}); "
              f"no split = {cell['n_var_no_split']:.3f}; ratio = {cell['ratio_single_over_rotated']:.3f}")
        nv = cell["naive_plug_in"]
        print(f"      R^2 of a single-split estimate on its first-stage error: "
              f"MBR {cell['r2_single_on_first_stage_error']:.3f}, naive plug-in "
              f"{nv['r2_single_on_first_stage_error']:.3f}")
        print(f"      naive plug-in: N Var single = {nv['n_var_single_mean']:.3f}, rotated = "
              f"{nv['n_var_rotated']:.3f}, ratio = {nv['n_var_single_mean'] / nv['n_var_rotated']:.3f}, "
              f"corr = {np.round(nv['corr_between_rotations'], 3)}, bias = {nv['bias_rotated']:+.4f}")
        print(f"      corr between rotations = {np.round(cell['corr_between_rotations'], 3)}; "
              f"bias (rot, no split) = {cell['bias'][3]:+.4f}, {cell['bias'][4]:+.4f}; "
              f"RMSE rot = {cell['rmse_rotated']:.5f}, no split = {cell['rmse_no_split']:.5f}; "
              f"MC s.e. of each variance ~ {100 * se:.1f}%")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
