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

``--design five`` moves the two trim endpoints to a fifth fold (still using
the shared gamma_hat).  The endpoint score and the density score are both
functions of T and are correlated on the same observation, so that design
does not give an exact multiple: the prediction is 5 S_5 / H^2 = 214.8 for a
fixed assignment (ratio 4.92), with the rotated variance unchanged.

Run:
    python -m experiments.scripts.four_role_shared_gamma_check --reps 400 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm

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
DESIGNS = {
    "four": ROLES,
    "five": ("gamma", "outcome", "index", "endpoint", "utility"),
}
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "four_role_shared_gamma_check"


def component(data, folds) -> EvaluationComponent:
    gamma = _fit_gamma(data, folds["gamma"])
    eta_hat = data.Q - _predict_T(data.X, gamma)
    fit = _fit_spline_plm(data, folds["outcome"], eta_hat, NUISANCE_SUPPORT)
    T_index = _predict_T(data.X[folds["index"]], gamma)
    density = _fit_T_density(T_index, "spline")
    # Without a separate endpoint fold the endpoints share the index fold.
    endpoint_fold = folds.get("endpoint", folds["index"])
    l_hat, u_hat = _boundaries_from_T(_predict_T(data.X[endpoint_fold], gamma))
    eta_eval = eta_hat[folds["utility"]]
    weights = ((eta_eval >= l_hat) & (eta_eval <= u_hat)).astype(float)
    effect = _eval_basis(eta_eval, fit.info) @ fit.omega_treat
    return EvaluationComponent(eta=eta_eval, hard_weights=weights,
                               treatment_effect=effect, T_density=density)


def predicted_variances(design: str) -> dict:
    """Fixed-split and rotated N Var from the analytic role scores."""
    result = decomposition()
    total = float(np.sum(result["score_covariance"]))
    H2 = result["truth"]["hard_curvature"] ** 2
    if design == "four":
        diagonal = total
    else:
        # Separate endpoint fold: drop the density-endpoint covariance.
        q = result["loadings"]
        variances = dict(zip(result["roles"], np.diag(result["score_covariance"])))
        z = float(norm.ppf(0.9))
        gz2 = float(norm.pdf(z)) ** 2
        kappa = -(q["A_alpha_T"] + q["A_g_T"] + z * (q["b_l"] + q["b_u"]))
        endpoint = ((q["b_l"] ** 2 + q["b_u"] ** 2) * 0.09 - 2 * q["b_l"] * q["b_u"] * 0.01) / gz2
        diagonal = (kappa ** 2 + variances["outcome"] + variances["density"]
                    + endpoint + variances["utility"])
    k = len(DESIGNS[design])
    return {"rotated": total / H2, "fixed": k * diagonal / H2}


def replicate(task):
    n, seed, design = task
    roles = DESIGNS[design]
    data = generate_data(n, seed)
    rng = np.random.default_rng(seed + 31_000_003)
    blocks = np.array_split(rng.permutation(n), len(roles))
    parts = []
    for shift in range(len(roles)):
        folds = {role: blocks[(k + shift) % len(roles)] for k, role in enumerate(roles)}
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
    parser.add_argument("--design", choices=sorted(DESIGNS), default="four")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    target = population_truth()["hard_phi_star"]
    out = args.out or (DEFAULT_OUT if args.design == "four"
                       else DEFAULT_OUT.with_name(f"{args.design}_role_shared_gamma_check"))
    prediction = predicted_variances(args.design)
    result = {"design": args.design, "prediction": prediction, "cells": {}}
    print(f"prediction ({args.design} roles): fixed N Var = {prediction['fixed']:.2f}; "
          f"rotated = {prediction['rotated']:.2f}")
    for n in args.n:
        tasks = [(n, args.seed + 104_729 * n + r, args.design) for r in range(args.reps)]
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
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {out / 'summary.json'}")


if __name__ == "__main__":
    main()
