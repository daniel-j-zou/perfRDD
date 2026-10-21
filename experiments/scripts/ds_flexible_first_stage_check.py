"""Growing first-stage fits under the hard trim (common-slope Gaussian design).

Question: can the linear first stage Q = gamma'X~ + eta be replaced by a
flexible fit h_hat(X) whose dimension grows with n, and keep the root-n rate?

Theory (``manuscript/this_week.tex``, notes of 2026-10-19): under the hard trim
the first stage enters through level-set averages m_delta(t) = E[delta(X) | T=t]
of its error delta = h_hat - h, at T = phi* - l0, phi* - u0 and at the two trim
quantiles.  An additive spline fit is averaged over each level set, so these
terms stay root-n when no single covariate carries the index.  A fit that can
represent functions of the index itself (index splines, tensor products, neural
networks) turns them into point evaluations, and N Var grows with its
dimension.

Designs (both have T ~ N(0, 1), so phi*, H and the linear-first-stage
prediction sigma^2/H^2 = 43.68 are the same):
  spread  gamma = (1, 1, 1)/sqrt(3)  (the Gaussian baseline)
  single  gamma = (1, 0, 0)          (one covariate carries the index)

First stages (``kind:K``):
  linear      OLS on (1, X)
  additive    OLS on (1, X, cubic B-splines in each X_j with K interior knots)
  index       OLS on (1, X, cubic B-splines in the OLS pilot index with K knots)
  mlp         one-hidden-layer tanh network with K units (sklearn)

Estimator: the shared-training plug-in with 4-fold cross-fitting (``_cf``) and a
single fold (``_one``); every other nuisance uses the default rules of
``debiased_score_check.py``.  The true h is linear, so every spline family
contains it and only the variance of the first stage changes with K.

Run:
    python -m experiments.scripts.ds_flexible_first_stage_check --design spread \
        --n 10000 40000 --reps 300 --configs linear:0 additive:8 additive:32 index:8 index:32
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np

from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.methods.spline_density import fit_spline_density
from experiments.scripts.debiased_score_check import (
    alpha_baseline,
    default_knots,
    fit_outcome,
    truth,
)
from experiments.scripts.hard_trim_crossfit_regularization import (
    T_DENSITY_SUPPORT,
    EvaluationComponent,
    _boundaries_from_T,
    _maximize_components,
)
from experiments.scripts.hard_trim_gaussian_baseline import (
    BETA,
    P,
    PHI_0,
    SIGMA_Y,
    GeneratedData,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_flexible_first_stage"
X_SUPPORT = (-4.5, 4.5)
DESIGN_GAMMA = {"spread": np.ones(P) / np.sqrt(P), "single": np.eye(P)[0]}
N_FOLDS = 4


def generate(design: str, n: int, seed: int) -> GeneratedData:
    """The Gaussian baseline with the design's index; 'spread' is the baseline."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, P))
    eta = rng.standard_normal(n)
    T = X @ DESIGN_GAMMA[design]
    Q = T + eta
    D = (Q > PHI_0).astype(float)
    Y = D * alpha_baseline(eta) + 0.5 * eta ** 2 + X @ BETA + rng.normal(0.0, SIGMA_Y, n)
    return GeneratedData(X=X, eta=eta, T=T, Q=Q, D=D, Y=Y)


def _spline_block(x, k):
    return _eval_basis(x, _basis_params(int(k), X_SUPPORT))


def fit_first_stage(data, idx, kind: str, k: int, seed: int = 0) -> Callable:
    """Fit the first stage on ``idx``; return a predictor of T from X."""
    X, Q = data.X[idx], data.Q[idx]
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor

        net = MLPRegressor(hidden_layer_sizes=(int(k),), activation="tanh", alpha=1e-4,
                           max_iter=500, tol=1e-6, random_state=seed)
        net.fit(X, Q)
        return net.predict
    lin = np.column_stack((np.ones(len(idx)), X))
    pilot = np.linalg.lstsq(lin, Q, rcond=None)[0]

    def features(Z):
        cols = [np.ones(len(Z)), Z]
        if kind == "additive":
            cols += [_spline_block(Z[:, j], k) for j in range(P)]
        elif kind == "index":
            cols.append(_spline_block(Z @ pilot[1:], k))
        elif kind != "linear":
            raise ValueError(f"unknown first stage {kind!r}")
        return np.column_stack(cols)

    coef = np.linalg.lstsq(features(X), Q, rcond=None)[0]
    return lambda Z: features(Z) @ coef


def fold_estimate(data, train, evalf, kind: str, k: int, seed: int = 0) -> float:
    """Shared-training plug-in on one evaluation fold."""
    T_hat = fit_first_stage(data, train, kind, k, seed)(data.X)
    eta_hat = data.Q - T_hat
    outcome = fit_outcome(data, train, eta_hat, default_knots(data, train, eta_hat))
    gdens = fit_spline_density(T_hat[train], T_DENSITY_SUPPORT)
    l, u = _boundaries_from_T(T_hat[train])
    eta = eta_hat[evalf]
    component = EvaluationComponent(
        eta=eta,
        hard_weights=((eta >= l) & (eta <= u)).astype(float),
        treatment_effect=outcome.effect(eta),
        T_density=gdens,
    )
    return _maximize_components([component])[0]


def parse_config(text: str):
    kind, k = text.split(":")
    return kind, int(k)


def replicate(task):
    design, n, seed, configs = task
    data = generate(design, n, seed)
    folds = np.array_split(np.random.default_rng(seed + 7_919).permutation(n), N_FOLDS)
    out: Dict[str, float] = {}
    for kind, k in configs:
        fits = []
        for j in range(N_FOLDS):
            train = np.concatenate([folds[i] for i in range(N_FOLDS) if i != j])
            fits.append(fold_estimate(data, train, folds[j], kind, k, seed + j))
        out[f"{kind}_K{k}_one"] = fits[0]
        out[f"{kind}_K{k}_cf"] = float(np.mean(fits))
    return out


def summarize(draws: Sequence[Dict[str, float]], n: int, phi_star: float):
    cells = {}
    for key in draws[0]:
        x = np.array([d[key] for d in draws])
        err = x - phi_star
        cells[key] = {
            "n_var": float(n * x.var(ddof=1)),
            "n_mse": float(n * np.mean(err ** 2)),
            "root_n_bias": float(np.sqrt(n) * err.mean()),
            "root_n_bias_se": float(np.sqrt(n) * x.std(ddof=1) / np.sqrt(len(x))),
        }
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=sorted(DESIGN_GAMMA), default="spread")
    ap.add_argument("--n", type=int, nargs="+", default=[10000, 40000])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=["linear:0", "additive:8", "index:8"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    configs = [parse_config(c) for c in args.configs]
    tr = truth("baseline")
    result = {"args": vars(args), "truth": tr, "prediction_linear": 43.68, "cells": {}}
    print(f"design={args.design} phi*={tr['phi_star']:.6f} H={tr['curvature']:.6f}")
    for n in args.n:
        tasks = [(args.design, n, args.seed + 1000 * r, configs) for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            draws = list(pool.map(replicate, tasks, chunksize=2))
        cells = summarize(draws, n, tr["phi_star"])
        result["cells"][str(n)] = cells
        print(f"N={n} (reps={args.reps}; N Var has ~{100 * np.sqrt(2 / args.reps):.0f}% MC error)")
        for key, c in cells.items():
            print(f"  {key:16s} N Var {c['n_var']:7.1f}  N MSE {c['n_mse']:7.1f}  "
                  f"sqrt(N) bias {c['root_n_bias']:+6.2f} (se {c['root_n_bias_se']:.2f})")
    tag = args.tag or f"{args.design}"
    out = DEFAULT_OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    print(f"[wrote] {out / 'summary.json'}")


if __name__ == "__main__":
    main()
