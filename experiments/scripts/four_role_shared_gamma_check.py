"""Four uncorrelated roles: does a shared first stage give a clean 4x?

The eight-block baseline pays more than 8x for a fixed split (ratio 10.6)
because five separate first-stage fits contribute errors that would cancel
if one fit served every role.  This script runs the same Gaussian hard-trim
estimator with four equal folds:

    gamma    one OLS fit of Q on (1, X), used by every other fold;
    outcome  spline partially linear fit of alpha on eta_hat;
    index    spline density of T_hat and both trim endpoints (T_hat quantiles);
    utility  evaluation average and argmax.

With one shared gamma_hat the intercept loading cancels by location
invariance, and under X independent of eta the four influence functions
(kappa T eta; -r_alpha u_Y; psi_T; -I F) are mutually uncorrelated.  The
prediction is therefore N Var = 4 V for one fixed assignment and V after
rotating the four roles, with V = Var(sum psi) / H^2 = 43.675 from
``eight_role_variance_decomposition.py``.

Run:
    python -m experiments.scripts.four_role_shared_gamma_check --reps 400 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.methods.perfrdd import _eval_basis
from experiments.scripts.eight_role_variance_decomposition import decomposition
from experiments.scripts.hard_trim_crossfit_regularization import (
    EvaluationComponent,
    _boundaries_from_T,
    _fit_T_density,
    _maximize_components,
)
from experiments.scripts.hard_trim_gaussian_baseline import (
    NUISANCE_SUPPORT,
    _fit_gamma,
    _fit_spline_plm,
    _predict_T,
    generate_data,
    population_truth,
)

ROLES = ("gamma", "outcome", "index", "utility")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "four_role_shared_gamma_check"


def component(data, folds) -> EvaluationComponent:
    gamma = _fit_gamma(data, folds["gamma"])
    eta_hat = data.Q - _predict_T(data.X, gamma)
    fit = _fit_spline_plm(data, folds["outcome"], eta_hat, NUISANCE_SUPPORT)
    T_index = _predict_T(data.X[folds["index"]], gamma)
    density = _fit_T_density(T_index, "spline")
    l_hat, u_hat = _boundaries_from_T(T_index)
    eta_eval = eta_hat[folds["utility"]]
    weights = ((eta_eval >= l_hat) & (eta_eval <= u_hat)).astype(float)
    effect = _eval_basis(eta_eval, fit.info) @ fit.omega_treat
    return EvaluationComponent(eta=eta_eval, hard_weights=weights,
                               treatment_effect=effect, T_density=density)


def replicate(task):
    n, seed = task
    data = generate_data(n, seed)
    rng = np.random.default_rng(seed + 31_000_003)
    blocks = np.array_split(rng.permutation(n), len(ROLES))
    parts = []
    for shift in range(len(ROLES)):
        folds = {role: blocks[(k + shift) % len(ROLES)] for k, role in enumerate(ROLES)}
        parts.append(component(data, folds))
    fixed = _maximize_components([parts[0]])[0]
    rotated = _maximize_components(parts)[0]
    return fixed, rotated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[20000, 40000])
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    target = population_truth()["hard_phi_star"]
    V = decomposition()["threshold_variance"]["rotated_or_full_sample"]
    result = {"prediction": {"rotated": V, "fixed_four_role": 4.0 * V}, "cells": {}}
    print(f"prediction: fixed four-role N Var = 4 V = {4 * V:.2f}; rotated = V = {V:.2f}")
    for n in args.n:
        tasks = [(n, args.seed + 104_729 * n + r) for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            draws = np.array(list(pool.map(replicate, tasks, chunksize=4)))
        err = draws - target
        cell = {
            "n_var_fixed": float(n * np.var(err[:, 0], ddof=1)),
            "n_var_rotated": float(n * np.var(err[:, 1], ddof=1)),
            "n_mse_fixed": float(n * np.mean(err[:, 0] ** 2)),
            "n_mse_rotated": float(n * np.mean(err[:, 1] ** 2)),
            "relative_mc_se": float(np.sqrt(2.0 / (args.reps - 1))),
        }
        cell["ratio"] = cell["n_var_fixed"] / cell["n_var_rotated"]
        result["cells"][str(n)] = cell
        print(f"N={n}: fixed N Var = {cell['n_var_fixed']:.1f} (MSE {cell['n_mse_fixed']:.1f}); "
              f"rotated = {cell['n_var_rotated']:.2f} (MSE {cell['n_mse_rotated']:.2f}); "
              f"ratio = {cell['ratio']:.2f}; MC s.e. ~ {100 * cell['relative_mc_se']:.0f}% each")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
