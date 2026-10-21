"""Cross-validated spline sizes against the fixed rule, in the Section 5 designs.

Robustness check requested by the author (weekly scope review, 2026-10-20): the paper
uses the fixed n^{11/60} rule for the spline sizes, and cross-validation over a grid
inside the (DS5) range is a check only. On each simulated data set and one random
four-block split (the seeds of ``ds_four_role_coverage.py``, so the rule arm repeats
that script's point estimates), the rotated four-role estimator is computed twice:

* ``rule``: outcome knots max(3, round(m^{11/60})) with m the smaller arm count in J,
  and density basis max(8, round(2 n_T^{11/60}));
* ``cv``: in each rotation, the outcome knots minimize the 5-fold cross-validated
  squared error of the arm-wise fit on the outcome fold, over max(3, round(m^a)) for
  a in (0.15, 0.17, 0.19, 0.21, 0.23); the density basis minimizes the least-squares
  cross-validation score of the density block on the index fold, over
  max(8, round(2 n_T^a)) for a in (0.13, 0.155, 0.18, 0.205, 0.23). All exponents lie
  inside the (DS5) range.

Reported per design and n: sqrt(n) bias, n Var and n MSE of both, the sd of the
paired difference, and how often each size is chosen. No bootstrap.

Run (from code/):
    python -m experiments.scripts.ds_spline_size_cv --design displayed --n 16000 64000 \
        --reps 300 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.methods.perfrdd import _basis_params
from experiments.methods.spline_density import spline_basis_dimension
from experiments.methods.spline_size_cv import (
    argmin, density_lscv, density_size_grid, knot_grid, outcome_cv,
)
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts import ds_four_role_coverage as C
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, _design_matrix, _n_interior, _xt, generate, ols_first_stage, outcome_fit,
)

OUT = C.ROOT / "runs" / "ds_spline_size_cv"


def _chosen_sizes(s, folds, eta_hat, design, rng):
    """Cross-validated knots (outcome fold) and density basis size (index fold)."""
    idx = folds["outcome"]
    e = eta_hat[idx]
    keep = idx[(e >= design.J[0]) & (e <= design.J[1])]
    e, X, D, Y = eta_hat[keep], s.X[keep], s.D[keep], s.Y[keep]
    n1 = int(D.sum())
    knots = knot_grid(min(n1, len(keep) - n1))
    cv_k = outcome_cv(len(keep), lambda rows, k: _design_matrix(
        e[rows], X[rows], D[rows], _basis_params(k, design.J))[0], Y, knots, rng)
    T_idx = _xt(s.X[folds["index"]]) @ ols_first_stage(s, folds["gamma"])
    sizes = density_size_grid(len(T_idx))
    cv_l = density_lscv(T_idx, s.X[folds["index"]], design.T_support, sizes)
    return argmin(cv_k), argmin(cv_l)


def estimate(s, design, rng, mode):
    n = len(s.Q)
    blocks = np.array_split(rng.permutation(n), 4)
    grid = np.linspace(design.policy[0], design.policy[1], 61)
    crit, vals, sizes = [], [], []
    cv_rng = np.random.default_rng(int(rng.integers(2 ** 31)))
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        n_u = len(folds["utility"])
        gamma_hat = ols_first_stage(s, folds["gamma"])
        eta_hat = s.Q - _xt(s.X) @ gamma_hat
        T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
        if mode == "cv":
            k_int, L = _chosen_sizes(s, folds, eta_hat, design, cv_rng)
        else:
            k_int, L = None, spline_basis_dimension(len(T_idx))
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], design.T_support, n_basis=L)
        l_hat = design.phi0 - float(np.quantile(T_idx, 1.0 - design.eps))
        u_hat = design.phi0 - float(np.quantile(T_idx, design.eps))
        e = eta_hat[folds["utility"]]
        e = e[(e >= l_hat) & (e <= u_hat)]
        fit = outcome_fit(s, folds["outcome"], eta_hat, design.J, n_interior=k_int)
        sizes.append((int(len(fit.w0)), int(L)))
        pieces = C.SplineTailPieces(tails, e, n_u)
        amc = fit.a(e) - design.cost
        b2 = np.asarray(fit.beta2)
        row = []
        for x in grid:
            sg, sh = pieces.sums(x, [amc])
            row.append((sg[0] + float(sh @ b2)) / n_u)
        crit.append(lambda x, p=pieces, a=amc, b=b2: p.value(x, a, b))
        vals.append(row)
    avg, pooled, rot = C._avg_and_pooled(crit, grid, np.array(vals))
    return {"avg": avg, "pooled": pooled, "sizes": sizes}


def replicate(task):
    design_key, n, rep, seed = task
    design = C.COVERAGE_DESIGNS[design_key]
    t0 = time.perf_counter()
    s = generate(design, n, seed)
    out = {"rep": rep, "seed": seed}
    for mode in ("rule", "cv"):
        out[mode] = estimate(s, design, np.random.default_rng([seed, 1]), mode)
    out["seconds"] = time.perf_counter() - t0
    return out


def summarize(rows, n, phi):
    res = {"n": n, "reps": len(rows)}
    for mode in ("rule", "cv"):
        for arm in ("avg", "pooled"):
            err = np.array([r[mode][arm] for r in rows]) - phi
            res[f"{mode}_{arm}"] = {
                "sqrt_n_bias": float(np.sqrt(n) * err.mean()),
                "sqrt_n_bias_se": float(np.sqrt(n) * err.std(ddof=1) / np.sqrt(len(err))),
                "n_var": float(n * err.var(ddof=1)), "n_mse": float(n * np.mean(err ** 2))}
    diff = np.array([r["cv"]["avg"] - r["rule"]["avg"] for r in rows])
    res["paired_diff_avg"] = {"sqrt_n_mean": float(np.sqrt(n) * diff.mean()),
                              "sqrt_n_sd": float(np.sqrt(n) * diff.std(ddof=1))}
    for mode in ("rule", "cv"):
        ks = Counter(k for r in rows for k, _ in r[mode]["sizes"])
        ls = Counter(L for r in rows for _, L in r[mode]["sizes"])
        res[f"{mode}_outcome_basis"] = {str(k): v for k, v in sorted(ks.items())}
        res[f"{mode}_density_basis"] = {str(k): v for k, v in sorted(ls.items())}
    res["mean_seconds_per_rep"] = float(np.mean([r["seconds"] for r in rows]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", choices=sorted(C.COVERAGE_DESIGNS), default="displayed")
    ap.add_argument("--n", type=int, nargs="+", default=[16000, 64000])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    pop = C.population_summary(C.COVERAGE_DESIGNS[a.design])
    summary_path = a.out / f"{a.design}_summary.json"
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {"cells": {}}
    result.update(design=a.design, phi_star=pop["phi_star"], V_rotated=pop["V_rotated"])
    for n in a.n:
        path = a.out / f"{a.design}_n{n}.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        done = {r["rep"] for r in rows}
        tasks = [(a.design, n, r, a.seed + 104_729 * n + r) for r in range(a.reps) if r not in done]
        t0 = time.perf_counter()
        print(f"[{a.design}] n={n}: {len(done)} done, running {len(tasks)}", flush=True)
        with ProcessPoolExecutor(a.workers) as pool, path.open("a") as fh:
            futures = [pool.submit(replicate, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} reps, {time.perf_counter() - t0:.0f}s", flush=True)
        rows = sorted(rows, key=lambda r: r["rep"])[:a.reps]
        result["cells"][str(n)] = summarize(rows, n, pop["phi_star"])
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["cells"][str(n)], indent=1), flush=True)


if __name__ == "__main__":
    main()
