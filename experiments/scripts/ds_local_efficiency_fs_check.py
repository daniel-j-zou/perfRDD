"""First-stage-only check: do the stacked and adaptive first stages reach their limits?

For a first-stage fold of size m drawn from a design of ``ds_local_efficiency``,
compute three slope estimates of gamma_X on the same data:

* OLS of Q on (1, X):                     m Var -> var_eta Var(X)^{-1};
* stacked first stage, LS score rows:     m Var -> (Var(X)/var_eta + J_Y)^{-1};
* adaptive stacked first stage:           m Var -> (I_eta Var(X) + J_Y)^{-1} = I_gamma^{-1}

(this_week.tex, Lemma "Efficient first stage" and its adaptive remark).  Reports
m times the Monte Carlo variance of each slope, averaged over coordinates, against
these limits, as m grows.

Run:
    python -m experiments.scripts.ds_local_efficiency_fs_check \
        --designs paper_baseline paper_t5 paper_mix --m 4000 16000 64000 --reps 1000
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

from experiments.scripts.ds_local_efficiency import DESIGNS, Population
from experiments.scripts.ds_local_efficiency_adaptive_sim import adaptive_first_stage, generate
from experiments.scripts.ds_local_efficiency_sim import efficient_first_stage, ols_first_stage

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency_fs_check"


def replicate(task):
    name, m, seed = task
    design = DESIGNS[name]
    s = generate(design, m, seed)
    idx = np.arange(m)
    return {
        "ols": ols_first_stage(s, idx)[1:].tolist(),
        "stack": efficient_first_stage(s, idx, design.J, np.random.default_rng(seed + 5))[1:].tolist(),
        "adapt": adaptive_first_stage(s, idx, design.J, np.random.default_rng(seed + 5))[1:].tolist(),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", nargs="+", default=["paper_baseline", "paper_t5", "paper_mix"])
    parser.add_argument("--m", type=int, nargs="+", default=[4000, 16000, 64000])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = {}
    for name in args.designs:
        design = DESIGNS[name]
        pop = Population(design).decomposition()
        V_X = np.diag(design.sigma_x)
        J_Y = np.asarray(pop["J_Y"])
        limits = {
            "ols": float(np.mean(np.diag(design.sigma_eta ** 2 * np.linalg.inv(V_X)))),
            "stack": float(np.mean(np.diag(np.linalg.inv(V_X / design.sigma_eta ** 2 + J_Y)))),
            "adapt": float(np.mean(np.diag(np.linalg.inv(np.asarray(pop["I_gamma"]))))),
        }
        result[name] = {"limits": limits, "I_eta_times_var": pop["I_eta_times_var_eta"], "cells": {}}
        for m in args.m:
            tasks = [(name, m, args.seed + 7919 * m + r) for r in range(args.reps)]
            with ProcessPoolExecutor(args.workers) as pool:
                rows = list(pool.map(replicate, tasks, chunksize=8))
            gx = np.asarray(design.gamma_x)
            cell = {}
            for key in ("ols", "stack", "adapt"):
                G = np.array([r[key] for r in rows]) - gx
                cell[key] = {"m_var_mean": float(np.mean(m * G.var(axis=0, ddof=1))),
                             "bias_mean": float(np.mean(G.mean(axis=0)))}
            result[name]["cells"][str(m)] = cell
            print(f"[{name}] m={m}: " + "; ".join(
                f"{k} {cell[k]['m_var_mean']:.3f} (limit {limits[k]:.3f})" for k in ("ols", "stack", "adapt")))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
