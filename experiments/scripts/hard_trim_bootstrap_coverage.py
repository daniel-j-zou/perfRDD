"""Monte Carlo coverage check for hard-trim bootstrap inference.

This is a deliberately empirical diagnostic for the hard-trim estimator.  For each
outer synthetic sample we re-estimate the complete point estimator on iid bootstrap
resamples, then form percentile and normal bootstrap intervals for the selected policy
threshold.  The population target is known for the Gaussian, ``t5``, and skewed-mixture
running-variable laws used by :mod:`hard_trim_robustness_smoke`.

The current public estimator exposes point estimates only.  Consequently this script
does *not* claim a bootstrap validity theorem, nor does it implement the manuscript's
fully decoupled six-fold influence-function construction.  It answers the practical
question: does full-sample or ordinary K-fold re-estimation bootstrap behave sensibly
in finite samples before we invest in a feasible analytic variance estimator?

Examples
--------
Short smoke check::

    python -m experiments.scripts.hard_trim_bootstrap_coverage \
        --n 600 --outer-reps 2 --bootstrap-reps 9 --workers 1 \
        --laws t5 --estimators full crossfit3 --out /tmp/bootstrap_smoke

Overnight batch::

    python -m experiments.scripts.hard_trim_bootstrap_coverage \
        --n 1200 2400 4800 --outer-reps 40 --bootstrap-reps 199 \
        --workers 4 --laws t5 mixture --estimators full crossfit3 \
        --out experiments/runs/overnight_bootstrap_coverage
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

import numpy as np

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim
from experiments.scripts.hard_trim_robustness_smoke import (
    DEFAULT_PHI_GRID,
    DEFAULT_SUPPORT,
    make_sample,
    population_truth,
)


DEFAULT_COST = 2.25


def _estimate(sample: RDDSample, *, crossfit_folds: int) -> dict[str, Any]:
    """Return a point estimate without writing per-replication figures."""
    with TemporaryDirectory(prefix="perfrdd_bootstrap_") as directory:
        result = perfrdd_hard_trim(
            sample,
            Path(directory),
            DEFAULT_SUPPORT,
            eps=0.10,
            c_values=(DEFAULT_COST,),
            phi_grid=DEFAULT_PHI_GRID,
            max_n=None,
            crossfit_folds=crossfit_folds,
            write_outputs=False,
            return_curves=False,
        )
    return {
        "phi": float(result["phi_star"][str(DEFAULT_COST)]),
        "boundary": bool(result["phi_star_at_grid_boundary"][str(DEFAULT_COST)]),
        "retention": float(result["hard_retention"]),
    }


def _bootstrap_sample(sample: RDDSample, indices: np.ndarray) -> RDDSample:
    """Construct a bootstrap sample while preserving the RDDSample contract."""
    return RDDSample(
        Q=np.asarray(sample.Q)[indices],
        X=np.asarray(sample.X)[indices],
        Y=np.asarray(sample.Y)[indices],
        threshold=sample.threshold,
        name=sample.name,
        feature_names=list(sample.feature_names),
    )


def _outer_replication(
    task: tuple[str, int, int, int, int, int],
) -> dict[str, Any]:
    """Run one outer sample and all requested bootstrap estimators."""
    law, n, outer_seed, bootstrap_reps, bootstrap_seed, crossfit_folds = task
    sample = make_sample(n, outer_seed, running_law=law)
    point = _estimate(sample, crossfit_folds=crossfit_folds)
    rng = np.random.default_rng(bootstrap_seed)
    values: list[float] = []
    failures = 0
    boundaries = 0
    for _ in range(int(bootstrap_reps)):
        indices = rng.integers(0, n, size=n)
        try:
            result = _estimate(
                _bootstrap_sample(sample, indices),
                crossfit_folds=crossfit_folds,
            )
            values.append(result["phi"])
            boundaries += int(result["boundary"])
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            # A failed bootstrap resample is retained in the diagnostics rather
            # than silently dropped from the denominator.
            failures += 1
    boot = np.asarray(values, dtype=float)
    target = float(population_truth(law)["phi_star"])
    if boot.size:
        q025, q975 = np.quantile(boot, [0.025, 0.975])
        mean = float(np.mean(boot))
        sd = float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0
        normal_low, normal_high = point["phi"] - 1.96 * sd, point["phi"] + 1.96 * sd
        percentile_coverage = bool(q025 <= target <= q975)
        normal_coverage = bool(normal_low <= target <= normal_high)
    else:
        q025 = q975 = mean = sd = normal_low = normal_high = float("nan")
        percentile_coverage = normal_coverage = False
    return {
        "law": law,
        "n": int(n),
        "outer_seed": int(outer_seed),
        "bootstrap_reps_requested": int(bootstrap_reps),
        "bootstrap_reps_successful": int(boot.size),
        "bootstrap_failures": int(failures),
        "bootstrap_boundary_rate": float(boundaries / boot.size) if boot.size else float("nan"),
        "target_phi": target,
        "point_phi": point["phi"],
        "point_boundary": point["boundary"],
        "point_retention": point["retention"],
        "bootstrap_mean": mean,
        "bootstrap_sd": sd,
        "percentile_q025": float(q025),
        "percentile_q975": float(q975),
        "normal_q025": float(normal_low),
        "normal_q975": float(normal_high),
        "percentile_coverage": percentile_coverage,
        "normal_coverage": normal_coverage,
        "crossfit_folds": int(crossfit_folds),
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate coverage, bias, RMSE, and bootstrap failure diagnostics."""
    output: dict[str, Any] = {}
    for law in sorted({str(row["law"]) for row in rows}):
        output[law] = {}
        for n in sorted({int(row["n"]) for row in rows if row["law"] == law}):
            output[law][str(n)] = {}
            for folds in sorted({int(row["crossfit_folds"]) for row in rows
                                 if row["law"] == law and int(row["n"]) == n}):
                subset = [row for row in rows if (
                    row["law"] == law and int(row["n"]) == n
                    and int(row["crossfit_folds"]) == folds
                )]
                point = np.asarray([float(row["point_phi"]) for row in subset])
                target = float(subset[0]["target_phi"])
                output[law][str(n)][str(folds)] = {
                    "outer_replications": len(subset),
                    "target_phi": target,
                    "point_mean": float(np.mean(point)),
                    "point_bias": float(np.mean(point - target)),
                    "point_rmse": float(np.sqrt(np.mean((point - target) ** 2))),
                    "point_boundary_rate": float(np.mean([
                        bool(row["point_boundary"]) for row in subset
                    ])),
                    "bootstrap_mean_sd": float(np.nanmean([
                        float(row["bootstrap_sd"]) for row in subset
                    ])),
                    "percentile_coverage": float(np.mean([
                        bool(row["percentile_coverage"]) for row in subset
                    ])),
                    "normal_coverage": float(np.mean([
                        bool(row["normal_coverage"]) for row in subset
                    ])),
                    "mean_bootstrap_failures": float(np.mean([
                        int(row["bootstrap_failures"]) for row in subset
                    ])),
                    "max_bootstrap_failures": int(max(
                        int(row["bootstrap_failures"]) for row in subset
                    )),
                }
    return output


def run_coverage(
    n_values: Iterable[int],
    outer_reps: int,
    bootstrap_reps: int,
    workers: int,
    laws: Iterable[str],
    estimators: Iterable[str],
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Run the coverage study and write a CSV plus JSON summary."""
    law_values = tuple(laws)
    estimator_values = tuple(estimators)
    fold_map = {"full": 1, "crossfit3": 3, "crossfit5": 5}
    if not law_values or not set(law_values).issubset({"t5", "mixture", "skewed"}):
        raise ValueError("laws must be a nonempty subset of t5, mixture, skewed")
    if not estimator_values or not set(estimator_values).issubset(fold_map):
        raise ValueError("estimators must be a nonempty subset of full, crossfit3, crossfit5")
    if workers < 1 or outer_reps < 1 or bootstrap_reps < 9:
        raise ValueError("workers and outer_reps must be positive; bootstrap_reps must be at least 9")

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, int, int, int, int, int]] = []
    task_index = 0
    for law_index, law in enumerate(law_values):
        for n_index, n in enumerate(n_values):
            for outer in range(outer_reps):
                for estimator in estimator_values:
                    task_index += 1
                    estimator_seed = seed + 1_000_003 * task_index
                    tasks.append((
                        law,
                        int(n),
                        seed + 100_000 * law_index + 10_000 * n_index + outer,
                        int(bootstrap_reps),
                        estimator_seed,
                        fold_map[estimator],
                    ))

    rows: list[dict[str, Any]] = []
    csv_path = out_dir / "replications.csv"
    fieldnames = [
        "law", "n", "outer_seed", "bootstrap_reps_requested",
        "bootstrap_reps_successful", "bootstrap_failures", "bootstrap_boundary_rate",
        "target_phi", "point_phi", "point_boundary", "point_retention",
        "bootstrap_mean", "bootstrap_sd", "percentile_q025", "percentile_q975",
        "normal_q025", "normal_q975", "percentile_coverage", "normal_coverage",
        "crossfit_folds",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if workers == 1:
            completed = enumerate(tasks, start=1)
            for index, task in completed:
                row = _outer_replication(task)
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"[bootstrap] completed {index}/{len(tasks)}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_outer_replication, task) for task in tasks]
                for index, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()
                    print(f"[bootstrap] completed {index}/{len(tasks)}")

    rows.sort(key=lambda row: (
        str(row["law"]), int(row["n"]), int(row["outer_seed"]),
        int(row["crossfit_folds"]),
    ))
    summary = {
        "description": (
            "Full re-estimation iid bootstrap coverage diagnostic for the hard-trim "
            "policy threshold; not a bootstrap validity theorem."
        ),
        "eps": 0.10,
        "cost": DEFAULT_COST,
        "n_values": [int(n) for n in n_values],
        "outer_replications_per_cell": int(outer_reps),
        "bootstrap_replications_per_outer_sample": int(bootstrap_reps),
        "laws": list(law_values),
        "estimators": list(estimator_values),
        "seed": int(seed),
        "summary": _summarize(rows),
        "replications_csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1200, 2400, 4800])
    parser.add_argument("--outer-reps", type=int, default=40)
    parser.add_argument("--bootstrap-reps", type=int, default=199)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--laws", nargs="+", choices=("t5", "mixture", "skewed"),
                        default=["t5", "mixture"])
    parser.add_argument("--estimators", nargs="+", choices=("full", "crossfit3", "crossfit5"),
                        default=["full", "crossfit3"])
    parser.add_argument("--seed", type=int, default=91_001)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_coverage(
        args.n,
        args.outer_reps,
        args.bootstrap_reps,
        args.workers,
        args.laws,
        args.estimators,
        args.seed,
        args.out,
    )
    print(json.dumps(summary["summary"], indent=2))
    print(f"[wrote] {args.out / 'replications.csv'}")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
