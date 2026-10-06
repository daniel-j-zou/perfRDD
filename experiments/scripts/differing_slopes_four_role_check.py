"""Differing slopes: do four roles with a shared first stage give a clean 4x?

Same question as ``four_role_shared_gamma_check.py``, now for the
differing-slopes utility U_J.  Four equal folds:

    gamma    one OLS fit of Q on (1, X), used by every other fold;
    outcome  stacked outcome fit giving alpha-hat and beta2-hat together;
    index    Lebesgue-Gram spline fits of g and p_X, and both trim endpoints;
    utility  evaluation of U_J and its argmax.

The differing-slopes extension adds a term to each of the four role scores
but creates no new role: beta2-hat comes from the outcome fit, and p_X-hat
from the same T-fold as g-hat.  Under X independent of eta with exogenous
outcome errors the four scores stay mutually uncorrelated (the index score is
a function of X, the utility score a function of eta, the outcome score is
conditionally centred, and the shared-gamma intercept loading is zero), so the
prediction is N Var(fixed) = 4 N Var(rotated), with the rotated variance equal
to the full-sample variance.

The DGP, linear outcome fit, tail fits and maximizer are those of
``differing_slopes_full_pipeline.py`` (baseline scenario).

Run:
    python -m experiments.scripts.differing_slopes_four_role_check --reps 1000 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.scripts.differing_slopes_full_pipeline import (
    DEFAULT_DGP,
    Component,
    _fit_first_stage,
    _fit_outcome,
    _fit_tail,
    _fit_variant,
    _maximize,
    _predict_t,
    _trim_bounds_from_t,
    generate_sample,
    population_truth,
)

ROLES = ("gamma", "outcome", "index", "utility")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "differing_slopes_four_role_check"


def component(sample, dgp, folds) -> Component:
    gamma_hat = _fit_first_stage(sample, folds["gamma"])
    eta_hat = sample.Q - _predict_t(sample.X, gamma_hat)
    outcome = _fit_outcome(sample, eta_hat, folds["outcome"], include_beta2=True, ridge=0.0)
    t_index = _predict_t(sample.X[folds["index"]], gamma_hat)
    tail = _fit_tail(t_index, sample.X[folds["index"]])
    l_hat, u_hat = _trim_bounds_from_t(t_index, dgp.trim_eps)
    eta_eval = eta_hat[folds["utility"]]
    weights = ((eta_eval >= l_hat) & (eta_eval <= u_hat)).astype(float)
    return Component(eta=eta_eval, hard_weights=weights, outcome=outcome, tail=tail,
                     include_beta2=True, l_hat=l_hat, u_hat=u_hat)


def replicate(task):
    n, seed = task
    dgp = DEFAULT_DGP
    sample = generate_sample(n, seed, dgp)
    rng = np.random.default_rng(seed + 31_000_003)
    blocks = np.array_split(rng.permutation(n), len(ROLES))
    parts = []
    for shift in range(len(ROLES)):
        folds = {role: blocks[(k + shift) % len(ROLES)] for k, role in enumerate(ROLES)}
        parts.append(component(sample, dgp, folds))
    fixed = _maximize([parts[0]], dgp)[0]
    rotated = _maximize(parts, dgp)[0]
    full = _fit_variant(sample, dgp, seed, "full_spline_ols")[0]
    return fixed, rotated, full


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[8000, 16000, 32000])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    target = population_truth(DEFAULT_DGP)["phi_star"]
    result = {"phi_star": target, "cells": {}}
    names = ("fixed", "rotated", "full")
    for n in args.n:
        tasks = [(n, args.seed + 104_729 * n + r) for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            draws = np.array(list(pool.map(replicate, tasks, chunksize=4)))
        err = draws - target
        cell = {f"n_var_{name}": float(n * np.var(err[:, k], ddof=1)) for k, name in enumerate(names)}
        cell.update({f"bias_{name}": float(np.mean(err[:, k])) for k, name in enumerate(names)})
        cell["ratio_fixed_to_rotated"] = cell["n_var_fixed"] / cell["n_var_rotated"]
        cell["ratio_fixed_to_full"] = cell["n_var_fixed"] / cell["n_var_full"]
        cell["relative_mc_se"] = float(np.sqrt(2.0 / (args.reps - 1)))
        result["cells"][str(n)] = cell
        print(f"N={n}: N Var fixed = {cell['n_var_fixed']:.2f}, rotated = {cell['n_var_rotated']:.2f}, "
              f"full = {cell['n_var_full']:.2f}; fixed/rotated = {cell['ratio_fixed_to_rotated']:.2f}, "
              f"fixed/full = {cell['ratio_fixed_to_full']:.2f}; bias = "
              f"{cell['bias_fixed']:+.4f}/{cell['bias_rotated']:+.4f}/{cell['bias_full']:+.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
