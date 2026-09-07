"""Moderate/long Monte Carlo follow-up to the hard-trim robustness smoke tests.

The default is intentionally smaller than a publication-grade run.  It supplies known
population targets for the non-Gaussian running variables, then checks finite-sample
bias, RMSE, boundary frequency, support sensitivity, an omitted-interaction outcome,
and a small iid bootstrap.  The same harness accepts a larger list of sample sizes and
replications for the longer study.  It does not claim asymptotic coverage because the
point-estimation API still has no feasible standard-error estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from experiments.scripts.hard_trim_robustness_smoke import (
    DEFAULT_SUPPORT,
    estimate_once,
    make_sample,
    population_truth,
    short_bootstrap,
)


def _summarize(
    values: Sequence[float],
    target: float,
    boundary_flags: Sequence[bool],
    n: int | None = None,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    errors = values - float(target)
    result = {
        "target": float(target),
        "mean": float(np.mean(values)),
        "bias": float(np.mean(errors)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "n_times_mse": float(len(values) * np.mean(errors ** 2)),
        "boundary_rate": float(np.mean(np.asarray(boundary_flags, dtype=float))),
    }
    if n is not None:
        result["n_times_variance"] = float(n * np.var(values, ddof=1))
    return result


def _log_rmse_slope(rows: Dict[str, Dict[str, float]]) -> float:
    """Estimate the log-log RMSE slope across the supplied sample sizes."""
    ns = np.asarray([int(value) for value in rows], dtype=float)
    rmses = np.asarray([rows[str(int(n))]["rmse"] for n in ns], dtype=float)
    if len(ns) < 3:
        return float("nan")
    return float(np.polyfit(np.log(ns), np.log(rmses), 1)[0])


def run_monte_carlo(
    n_values: Iterable[int] = (600, 1200, 2400),
    reps: int = 20,
    seed: int = 70_001,
    laws: Iterable[str] = ("t5", "mixture", "skewed"),
    include_auxiliary: bool = True,
) -> Dict[str, Any]:
    """Run a moderate, known-target comparison for non-Gaussian T laws."""
    if reps < 2:
        raise ValueError("reps must be at least two for Monte Carlo SDs")
    output: Dict[str, Any] = {
        "n_values": [int(n) for n in n_values],
        "reps": int(reps),
        "laws": list(laws),
        "include_auxiliary": bool(include_auxiliary),
        "scenarios": {},
    }
    valid_laws = {"t5", "mixture", "skewed"}
    selected_laws = tuple(output["laws"])
    if not selected_laws or not set(selected_laws).issubset(valid_laws):
        raise ValueError(f"laws must be a nonempty subset of {sorted(valid_laws)}")
    for law_index, law in enumerate(selected_laws):
        truth = population_truth(law)
        by_n: Dict[str, Any] = {}
        for n_index, n in enumerate(output["n_values"]):
            full_values = []
            crossfit_values = []
            full_boundary = []
            crossfit_boundary = []
            for rep in range(reps):
                sample = make_sample(
                    n,
                    seed + 10_000 * law_index + 1_000 * n_index + rep,
                    running_law=law,
                )
                full = estimate_once(sample, DEFAULT_SUPPORT, crossfit_folds=1)
                crossfit = estimate_once(
                    sample, DEFAULT_SUPPORT, crossfit_folds=3
                )
                full_values.append(full["phi"])
                crossfit_values.append(crossfit["phi"])
                full_boundary.append(full["grid_boundary"])
                crossfit_boundary.append(crossfit["grid_boundary"])
            by_n[str(n)] = {
                "full_sample": _summarize(
                    full_values, truth["phi_star"], full_boundary, n
                ),
                "crossfit": _summarize(
                    crossfit_values, truth["phi_star"], crossfit_boundary, n
                ),
            }
        output["scenarios"][law] = {
            "population_truth": truth,
            "by_n": by_n,
            "rmse_log_slope": {
                "full_sample": _log_rmse_slope({
                    n: by_n[n]["full_sample"] for n in by_n
                }),
                "crossfit": _log_rmse_slope({
                    n: by_n[n]["crossfit"] for n in by_n
                }),
            },
        }

    if not include_auxiliary:
        return output

    support_n = 1200
    support_reps = min(reps, 12)
    support_values: Dict[str, list[float]] = {
        "-2.5,2.5": [], "-3.0,3.0": [], "-3.5,3.5": []
    }
    for rep in range(support_reps):
        sample = make_sample(support_n, seed + 50_000 + rep, running_law="t5")
        for label, support in (
            ("-2.5,2.5", (-2.5, 2.5)),
            ("-3.0,3.0", (-3.0, 3.0)),
            ("-3.5,3.5", (-3.5, 3.5)),
        ):
            support_values[label].append(estimate_once(sample, support)["phi"])
    reference = np.asarray(support_values["-3.0,3.0"])
    output["support_sensitivity"] = {
        "n": support_n,
        "reps": support_reps,
        "mean": {
            label: float(np.mean(values))
            for label, values in support_values.items()
        },
        "mean_absolute_difference_from_default": {
            label: float(np.mean(np.abs(np.asarray(values) - reference)))
            for label, values in support_values.items()
        },
    }

    misspecified_values = {"full_sample": [], "crossfit": []}
    misspecified_boundary = {"full_sample": [], "crossfit": []}
    misspecified_reps = min(reps, 12)
    for rep in range(misspecified_reps):
        sample = make_sample(
            support_n,
            seed + 60_000 + rep,
            running_law="t5",
            misspecified_outcome=True,
        )
        full = estimate_once(sample)
        crossfit = estimate_once(sample, crossfit_folds=3)
        for label, result in (("full_sample", full), ("crossfit", crossfit)):
            misspecified_values[label].append(result["phi"])
            misspecified_boundary[label].append(result["grid_boundary"])
    output["misspecified_outcome"] = {
        "n": support_n,
        "reps": misspecified_reps,
        "full_sample": _summarize(
            misspecified_values["full_sample"],
            population_truth("t5")["phi_star"],
            misspecified_boundary["full_sample"],
            support_n,
        ),
        "crossfit": _summarize(
            misspecified_values["crossfit"],
            population_truth("t5")["phi_star"],
            misspecified_boundary["crossfit"],
            support_n,
        ),
        "note": "The t5 target is only a reference; the omitted interaction changes the estimand.",
    }

    bootstrap_sample = make_sample(600, seed + 70_000, running_law="t5")
    output["bootstrap_diagnostic"] = short_bootstrap(
        bootstrap_sample, reps=min(15, max(5, reps // 2)), seed=seed + 80_000
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[600, 1200, 2400])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=70_001)
    parser.add_argument(
        "--laws", nargs="+", choices=("t5", "mixture", "skewed"),
        default=["t5", "mixture", "skewed"],
    )
    parser.add_argument(
        "--skip-auxiliary", action="store_true",
        help="skip support, misspecification, and bootstrap side diagnostics",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_monte_carlo(
        args.n,
        args.reps,
        args.seed,
        args.laws,
        include_auxiliary=not args.skip_auxiliary,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
