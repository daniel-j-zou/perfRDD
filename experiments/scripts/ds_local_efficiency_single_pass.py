"""Does the stacked first stage need its internal three-way split?

The efficient (stacked) first stage of ``this_week.tex`` splits the first-stage fold
into three parts: a pilot OLS fit, a derivative fit giving c_d', and the stacked
least-squares step, rotated and averaged. The single-pass version runs all three
steps on the whole first-stage fold. On the same data and folds this runs

    paper      OLS first stage, unweighted outcome fit
    stack3     stacked first stage, three-part split (ds_local_efficiency_sim.py)
    stack1     stacked first stage, single pass
    w_stack3   both weighted, three-part split, variance fitted once on the whole
               first-stage fold (w_ls_full of ds_local_efficiency_weight_diag.py)
    w_stack1   both weighted, single pass, same variance fit

and records the rotated and fixed thresholds and the first-stage slopes of the
first rotation's fold. ``--fs-only`` instead checks the first stage alone on a
sample of size m.

Run:
    python -m experiments.scripts.ds_local_efficiency_single_pass --design paper_baseline \
        --seed 20261013 --n 16000 32000 --reps 1000 --workers 6
    python -m experiments.scripts.ds_local_efficiency_single_pass --design paper_hetero \
        --n 32000 --reps 1000 --workers 6
    python -m experiments.scripts.ds_local_efficiency_single_pass --fs-only \
        --m 4000 16000 64000 --reps 1000
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

from experiments.methods.perfrdd import _basis_params
from experiments.scripts.ds_local_efficiency import DESIGNS, Population
from experiments.scripts.ds_local_efficiency_hetero import hetero_variances
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES,
    Sample,
    _design_matrix,
    _ls,
    _n_interior,
    _xt,
    efficient_first_stage,
    generate,
    ols_first_stage,
    outcome_fit,
)
from experiments.scripts.ds_local_efficiency_weighted_sim import (
    _wls,
    fit_variance,
    threshold,
    weighted_first_stage,
    weighted_outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency_single_pass"
VARIANTS = ("paper", "stack3", "stack1", "w_stack3", "w_stack1")
SLOPE_KEYS = ("paper", "stack3", "stack1", "w_stack3", "w_stack1")


def single_pass_first_stage(s: Sample, idx: np.ndarray, J) -> np.ndarray:
    """Stacked first stage with pilot, derivative fit and stacking all on ``idx``.

    The same rows and weights as one rotation of ``efficient_first_stage``.
    """
    g_p = ols_first_stage(s, idx)
    res_p = s.Q[idx] - _xt(s.X[idx]) @ g_p
    s_eta = float(np.sqrt(np.sum(res_p ** 2) / (len(idx) - len(g_p))))
    eta_hat = s.Q - _xt(s.X) @ g_p
    fit = outcome_fit(s, idx, eta_hat, J)
    sd = np.sqrt(np.asarray(fit.s2))
    e, X, D, Y = eta_hat[idx], s.X[idx], s.D[idx], s.Y[idx]
    dim = X.shape[1]
    inJ = (e >= J[0]) & (e <= J[1])
    Dj = D[inJ]
    n_int = _n_interior(int(min(Dj.sum(), inJ.sum() - Dj.sum())))
    Zy, _ = _design_matrix(e[inJ], X[inJ], Dj, _basis_params(n_int, J))
    xi = np.where(Dj == 1.0, fit.c_prime(1, e[inJ]), fit.c_prime(0, e[inJ]))
    wy = 1.0 / np.where(Dj == 1.0, sd[1], sd[0])
    n_out = Zy.shape[1]
    rows_q = np.column_stack((np.ones(len(e)), X, np.zeros((len(e), n_out)))) / s_eta
    rows_y = np.column_stack((np.zeros(len(Dj)), -xi[:, None] * X[inJ], Zy)) * wy[:, None]
    coef = _ls(np.vstack((rows_q, rows_y)), np.concatenate((e / s_eta, Y[inJ] * wy)))
    return g_p + coef[:1 + dim]


def single_pass_weighted_first_stage(s: Sample, idx: np.ndarray, J, vf) -> np.ndarray:
    """Weighted single pass: the rows of ``weighted_first_stage`` with all parts = ``idx``."""
    g_p = ols_first_stage(s, idx)
    res_p = s.Q[idx] - _xt(s.X[idx]) @ g_p
    s_eta2 = float(np.sum(res_p ** 2) / (len(idx) - len(g_p)))
    eta_hat = s.Q - _xt(s.X) @ g_p
    fit, vf = weighted_outcome_fit(s, idx, eta_hat, J, vf)
    e, X, D, Y = eta_hat[idx], s.X[idx], s.D[idx], s.Y[idx]
    Xc = X - X.mean(axis=0)
    m = len(idx)
    inJ = (e >= J[0]) & (e <= J[1])
    Dj, ej, Xj = D[inJ], e[inJ], X[inJ]
    n_int = _n_interior(int(min(Dj.sum(), inJ.sum() - Dj.sum())))
    Zy, _ = _design_matrix(ej, Xj, Dj, _basis_params(n_int, J))
    w = 1.0 / vf(Dj, ej)
    xi = np.where(Dj == 1.0, fit.c_prime(1, ej), fit.c_prime(0, ej))
    W = xi[:, None] * Xj
    R = W - Zy @ _wls(Zy, W, w)
    G = (Xc.T @ Xc) / (m * s_eta2) + (R.T * w) @ R / m
    rhs = (Xc.T @ e) / (m * s_eta2) - (R.T * w) @ Y[inJ] / m
    dX = np.linalg.solve(G, rhs)
    d0 = float(np.mean(e - X @ dX))
    return g_p + np.concatenate([[d0], dX])


def whole_fold_variance(s: Sample, idx, J, gamma_hat):
    eta_hat = s.Q - _xt(s.X) @ gamma_hat
    return fit_variance(s, idx, eta_hat, J, outcome_fit(s, idx, eta_hat, J))


def first_stages(s: Sample, idx, J, sub_seed: int) -> dict:
    g_ols = ols_first_stage(s, idx)
    vf = whole_fold_variance(s, idx, J, g_ols)
    return {
        "paper": g_ols,
        "stack3": efficient_first_stage(s, idx, J, np.random.default_rng(sub_seed)),
        "stack1": single_pass_first_stage(s, idx, J),
        "w_stack3": weighted_first_stage(s, idx, J, np.random.default_rng(sub_seed), vf),
        "w_stack1": single_pass_weighted_first_stage(s, idx, J, vf),
    }


def replicate(task):
    name, n, seed = task
    design = DESIGNS[name]
    s = generate(design, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    out = {v: [] for v in VARIANTS}
    slopes = {}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        g = first_stages(s, folds["gamma"], design.J, seed + 11 * k + 3)
        for v in VARIANTS:
            out[v].append(threshold(s, design, folds, g[v], v.startswith("w_")))
        if k == 0:
            slopes = {v: g[v][1:].tolist() for v in SLOPE_KEYS}
    row = {f"avg_{v}": float(np.mean(out[v])) for v in VARIANTS}
    row.update({f"fixed_{v}": out[v][0] for v in VARIANTS})
    row["slopes"] = slopes
    return row


def replicate_fs(task):
    name, m, seed = task
    design = DESIGNS[name]
    s = generate(design, m, seed)
    g = first_stages(s, np.arange(m), design.J, seed + 3)
    return {v: g[v][1:].tolist() for v in SLOPE_KEYS}


def predictions(design):
    if design.noise_kappa:
        pop = Population(design, allow_hetero=True)
        het = hetero_variances(pop, design.noise_kappa)
        return pop.phi, {"paper": het["paper"], "stack3": het["stacked"], "stack1": het["stacked"],
                         "w_stack3": het["bound"], "w_stack1": het["bound"]}
    res = Population(design).decomposition()
    v_eff = res["V_rotated_eff"]
    return res["phi_star"], {"paper": res["V_rotated_ols"], "stack3": v_eff, "stack1": v_eff,
                             "w_stack3": v_eff, "w_stack1": v_eff}


def run_fs(args) -> None:
    design = DESIGNS[args.design]
    if design.noise_kappa:
        raise SystemExit("--fs-only needs a homoskedastic design (the limit uses I_gamma)")
    gamma_x = np.asarray(design.gamma_x)
    res = Population(design).decomposition()
    limit = np.diag(np.linalg.inv(np.asarray(res["I_gamma"])))
    result = {"design": args.design, "limit_eff": limit.tolist(), "cells": {}}
    for m in args.m:
        tasks = [(args.design, m, args.seed + 7919 * m + r) for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(replicate_fs, tasks, chunksize=4))
        cell = {}
        for v in SLOPE_KEYS:
            G = np.array([r[v] for r in rows]) - gamma_x
            cell[v] = {"m_var_mean": float(m * np.mean(np.var(G, axis=0, ddof=1))),
                       "bias_mean": float(np.mean(G))}
        result["cells"][str(m)] = cell
        print(f"[{args.design}] m={m} reps={len(rows)}; limits OLS 1.000, efficient "
              f"{np.mean(limit):.3f}")
        for v in SLOPE_KEYS:
            print(f"  {v:9s}: m Var={cell[v]['m_var_mean']:.3f}; mean error={cell[v]['bias_mean']:+.5f}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.design}_fs_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="paper_baseline", choices=sorted(DESIGNS))
    parser.add_argument("--n", type=int, nargs="+", default=[32000])
    parser.add_argument("--m", type=int, nargs="+", default=[4000, 16000, 64000])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fs-only", action="store_true",
                        help="first stage alone on a sample of size m")
    args = parser.parse_args(argv)
    if args.fs_only:
        run_fs(args)
        return
    design = DESIGNS[args.design]
    phi, pred = predictions(design)
    gamma_x = np.asarray(design.gamma_x)
    result = {"design": args.design, "prediction": pred, "cells": {}}
    for n in args.n:
        tasks = [(args.design, n, args.seed + 104_729 * n + r) for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(replicate, tasks, chunksize=2))
        cell = {"reps": len(rows)}
        base = np.array([r["avg_paper"] for r in rows])
        for v in VARIANTS:
            a = np.array([r[f"avg_{v}"] for r in rows])
            f = np.array([r[f"fixed_{v}"] for r in rows]) - phi
            cell[v] = {"rot_n_var": float(n * a.var(ddof=1)), "rot_pred": pred[v],
                       "fixed_n_var": float(n * f.var(ddof=1)),
                       "bias": float(np.mean(a) - phi),
                       "paired_vs_paper": float(n * (base.var(ddof=1) - a.var(ddof=1)))}
        a1 = np.array([r["avg_stack1"] for r in rows])
        a3 = np.array([r["avg_stack3"] for r in rows])
        cell["corr_stack1_stack3"] = float(np.corrcoef(a1, a3)[0, 1])
        m = len(np.array_split(np.arange(n), 4)[0])
        cell["slope_m_var"] = {
            v: float(m * np.mean(np.var([r["slopes"][v] for r in rows], axis=0, ddof=1)))
            for v in SLOPE_KEYS}
        cell["slope_mean_error"] = {
            v: float(np.mean(np.mean([r["slopes"][v] for r in rows], axis=0) - gamma_x))
            for v in SLOPE_KEYS}
        result["cells"][str(n)] = cell
        print(f"[{args.design}] N={n} reps={len(rows)}")
        for v in VARIANTS:
            c = cell[v]
            print(f"  {v:9s}: rotated N Var={c['rot_n_var']:.2f} (pred {c['rot_pred']:.2f}); "
                  f"fixed={c['fixed_n_var']:.1f}; paired vs paper={c['paired_vs_paper']:+.2f}; "
                  f"bias={c['bias']:+.4f}")
        print(f"  corr(stack1, stack3) = {cell['corr_stack1_stack3']:.3f}")
        print(f"  slopes m Var: {cell['slope_m_var']}")
        print(f"  slopes mean error: {cell['slope_mean_error']}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.design}_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {path}")


if __name__ == "__main__":
    main()
