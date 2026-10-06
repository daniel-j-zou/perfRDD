"""Four-role rotation: pooled criterion versus the average of rotated estimates.

Mukherjee, Banerjee and Ritov rotate their three folds, obtain three
asymptotically independent estimates, and average them.  This script checks
the same structure for the four-role shared-first-stage split of
``four_role_shared_gamma_check.py`` in the Gaussian hard-trim baseline:

* each single rotation should have N Var = 4 V (V = 43.675);
* the four rotated estimates should be asymptotically uncorrelated, because
  two rotations give every fold different roles and different roles have
  uncorrelated scores;
* their average and the pooled-criterion maximizer should both have N Var = V
  and agree to first order.

Run:
    python -m experiments.scripts.four_role_rotation_average_check --n 40000 --reps 1000
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.scripts.four_role_shared_gamma_check import ROLES, component, predicted_variances
from experiments.scripts.hard_trim_crossfit_regularization import _maximize_components
from experiments.scripts.hard_trim_gaussian_baseline import generate_data, population_truth

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "four_role_rotation_average_check"


def replicate(task):
    """Return the four single-rotation thresholds followed by the pooled one."""
    n, seed = task
    data = generate_data(n, seed)
    rng = np.random.default_rng(seed + 31_000_003)
    blocks = np.array_split(rng.permutation(n), len(ROLES))
    parts = [
        component(data, {role: blocks[(k + shift) % len(ROLES)] for k, role in enumerate(ROLES)})
        for shift in range(len(ROLES))
    ]
    singles = [_maximize_components([part])[0] for part in parts]
    return singles + [_maximize_components(parts)[0]]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40000)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=777_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    target = population_truth()["hard_phi_star"]
    tasks = [(args.n, args.seed + r) for r in range(args.reps)]
    with ProcessPoolExecutor(args.workers) as pool:
        draws = np.array(list(pool.map(replicate, tasks, chunksize=4))) - target
    singles, pooled = draws[:, :-1], draws[:, -1]
    average = singles.mean(axis=1)
    corr = np.corrcoef(singles.T)
    off_diagonal = corr[~np.eye(len(ROLES), dtype=bool)]
    result = {
        "n": args.n,
        "reps": args.reps,
        "prediction": predicted_variances("four"),
        "n_var_single_rotations": (args.n * singles.var(axis=0, ddof=1)).tolist(),
        "rotation_correlations": corr.tolist(),
        "rotation_correlation_range": [float(off_diagonal.min()), float(off_diagonal.max())],
        "n_var_average_of_rotations": float(args.n * average.var(ddof=1)),
        "n_var_pooled_criterion": float(args.n * pooled.var(ddof=1)),
        "corr_average_pooled": float(np.corrcoef(average, pooled)[0, 1]),
        "n_var_average_minus_pooled": float(args.n * (average - pooled).var(ddof=1)),
    }
    print(f"prediction: single rotation {result['prediction']['fixed']:.1f}; rotated {result['prediction']['rotated']:.2f}")
    print("N Var single rotations:", np.round(result["n_var_single_rotations"], 1))
    print("rotation correlations in [%.2f, %.2f]" % tuple(result["rotation_correlation_range"]))
    print(f"N Var average of four = {result['n_var_average_of_rotations']:.2f}; "
          f"pooled criterion = {result['n_var_pooled_criterion']:.2f}; "
          f"corr = {result['corr_average_pooled']:.4f}; "
          f"N Var(difference) = {result['n_var_average_minus_pooled']:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
