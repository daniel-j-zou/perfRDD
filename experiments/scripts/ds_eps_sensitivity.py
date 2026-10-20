"""How the supported target and the four-role estimator move with the trim level eps.

Displayed design of differing_slopes.tex Section 5 (``paper_baseline``), with the
policy set Phi = [-1, 1.5] and density interval T = [-3.5, 3.5], so that (DS1)(i)
holds at every eps in the grid: V_Phi = Phi - [l0, u0] stays inside T down to
eps = 0.05, and [l0, u0] stays inside J = [-1.75, 1.75].

For each eps the population part reports the supported target phi*_eps, the
curvature H, the first-order n-scaled variances (rotated and fixed) and the
supported share P(I0 = 1). The Monte Carlo part reuses one data set and one
four-role split across all eps: the first stage, the outcome fit on J and the
density fits do not depend on eps, so only the trim endpoints and the
evaluation trim change. The estimator is the average of the four rotated
thresholds, as in the paper.

Run (from code/):
    python -m experiments.scripts.ds_eps_sensitivity --n 16000 64000 --reps 2000 1000 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_local_efficiency import DESIGNS, Population
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, _xt, generate, ols_first_stage, outcome_fit)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "ds_eps_sensitivity"
EPS_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25)
POLICY = (-1.0, 1.5)
BASE = replace(DESIGNS["paper_baseline"], policy=POLICY, T_support=(-3.5, 3.5))


def population() -> dict:
    out = {}
    for e in EPS_GRID:
        r = Population(replace(BASE, eps=e)).decomposition()
        out[str(e)] = {"phi_star": r["phi_star"], "H": r["H"], "l0": r["l0"], "u0": r["u0"],
                       "V_rotated": r["V_rotated_ols"], "V_fixed": r["V_fixed_ols"]}
    return out


def _argmax(value, lo, hi):
    grid = np.linspace(lo, hi, 61)
    vals = np.array([value(x) for x in grid])
    k = int(np.argmax(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: -value(x), bounds=(a, b), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x), bool(k in (0, len(grid) - 1))


def replicate(task):
    n, seed = task
    d = BASE
    s = generate(d, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    est = {str(e): [] for e in EPS_GRID}
    edge = {str(e): False for e in EPS_GRID}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        g = ols_first_stage(s, folds["gamma"])
        eta_hat = s.Q - _xt(s.X) @ g
        fit = outcome_fit(s, folds["outcome"], eta_hat, d.J)
        T_idx = _xt(s.X[folds["index"]]) @ g
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], d.T_support)
        e_all = eta_hat[folds["utility"]]
        b2 = fit.beta2
        for eps in EPS_GRID:
            l_hat = d.phi0 - float(np.quantile(T_idx, 1.0 - eps))
            u_hat = d.phi0 - float(np.quantile(T_idx, eps))
            e = e_all[(e_all >= l_hat) & (e_all <= u_hat)]
            amc = fit.a(e) - d.cost

            def value(phi, e=e, amc=amc):
                t = phi - e
                return float(np.mean(amc * tails.survival(t) + tails.weighted_tail(t, b2)))

            phi, at_edge = _argmax(value, *POLICY)
            est[str(eps)].append(phi)
            edge[str(eps)] |= at_edge
    return {"avg": {k: float(np.mean(v)) for k, v in est.items()},
            "fixed": {k: v[0] for k, v in est.items()}, "edge": edge}


def summarize(rows, n, pop):
    out = {}
    for e in EPS_GRID:
        k = str(e)
        x = np.array([r["avg"][k] for r in rows])
        f = np.array([r["fixed"][k] for r in rows])
        tgt = pop[k]["phi_star"]
        out[k] = {"phi_star": tgt, "mean": float(x.mean()),
                  "sqrt_n_bias": float(np.sqrt(n) * (x.mean() - tgt)),
                  "sqrt_n_bias_se": float(np.sqrt(n) * x.std(ddof=1) / np.sqrt(len(x))),
                  "n_var": float(n * x.var(ddof=1)), "n_var_fixed": float(n * f.var(ddof=1)),
                  "V_rotated": pop[k]["V_rotated"], "V_fixed": pop[k]["V_fixed"],
                  "n_mse": float(n * np.mean((x - tgt) ** 2)),
                  "edge_share": float(np.mean([r["edge"][k] for r in rows]))}
    # paired differences between consecutive eps, same data and folds
    diffs = {}
    for a, b in zip(EPS_GRID[:-1], EPS_GRID[1:]):
        dx = np.array([r["avg"][str(b)] - r["avg"][str(a)] for r in rows])
        diffs[f"{a}->{b}"] = {"mean": float(dx.mean()), "se": float(dx.std(ddof=1) / np.sqrt(len(dx))),
                              "target": pop[str(b)]["phi_star"] - pop[str(a)]["phi_star"]}
    return {"n": n, "reps": len(rows), "by_eps": out, "paired_steps": diffs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16000])
    ap.add_argument("--reps", type=int, nargs="+", default=[500])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    reps = a.reps if len(a.reps) == len(a.n) else [a.reps[0]] * len(a.n)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pop = population()
    print(json.dumps(pop, indent=1), flush=True)
    result = {"design": "paper_baseline", "policy": POLICY, "J": BASE.J, "T": BASE.T_support,
              "eps_grid": EPS_GRID, "untrimmed_phi_star": 0.4261829480634599,
              "population": pop, "cells": []}
    for n, R in zip(a.n, reps):
        tasks = [(n, a.seed + 1_000 * i + n) for i in range(R)]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(replicate, tasks, chunksize=4))
        cell = summarize(rows, n, pop)
        result["cells"].append(cell)
        with open(a.out / f"rows_n{n}.json", "w") as fh:
            json.dump(rows, fh)
        for k, v in cell["by_eps"].items():
            print(f"n={n} eps={k}: phi*={v['phi_star']:.4f} mean={v['mean']:.4f} "
                  f"sqrt(n)bias={v['sqrt_n_bias']:+.2f} ({v['sqrt_n_bias_se']:.2f}) "
                  f"nVar={v['n_var']:.1f} (pred {v['V_rotated']:.1f}) fixed {v['n_var_fixed']:.1f} "
                  f"edge {v['edge_share']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        with open(a.out / "summary.json", "w") as fh:
            json.dump(result, fh, indent=1)
    print(f"wrote {a.out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
