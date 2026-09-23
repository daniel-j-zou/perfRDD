"""Bootstrap coverage diagnostic for the feasible differing-slopes estimator.

The point estimator is imported from :mod:`differing_slopes_full_pipeline`.
Every bootstrap resample re-estimates the first-stage index, hard-trim
endpoints, outcome regression (including the full ``D * X`` block), and the
Lebesgue-Gram spline estimates of g and p_X used by the utility U_J.  This is an application-style
full-sample re-estimation bootstrap, not a claim of bootstrap validity for the
fully decoupled theorem.

The diagnostic compares an oracle-index benchmark with the feasible spline-tail
OLS and ridge-stabilized fits.  It reports percentile coverage for the
known population optimum, Monte Carlo bias/RMSE, bootstrap dispersion, failed
resamples, and boundary rates.

Example::

    python -m experiments.scripts.differing_slopes_feasible_bootstrap \
        --scenario baseline --n 800 1600 3200 --outer-reps 20 \
        --bootstrap-reps 99 --workers 4 \
        --out experiments/runs/differing_slopes_bootstrap
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.scripts.differing_slopes_full_pipeline import (
    DEFAULT_DGP,
    DGP,
    SCENARIOS,
    _fit_variant,
    generate_sample,
    population_truth,
)


VARIANTS = (
    "oracle",
    "full_spline_ols",
    "full_spline_ridge",
)


def _bootstrap_sample(sample: Any, indices: np.ndarray) -> Any:
    """Resample all observed variables while preserving the Sample contract."""
    # Importing Sample here keeps the public module import lightweight while
    # ensuring that bootstrap samples retain the exact dataclass interface.
    from experiments.scripts.differing_slopes_full_pipeline import Sample

    return Sample(
        X=np.asarray(sample.X)[indices],
        eta=np.asarray(sample.eta)[indices],
        T=np.asarray(sample.T)[indices],
        Q=np.asarray(sample.Q)[indices],
        D=np.asarray(sample.D)[indices],
        Y=np.asarray(sample.Y)[indices],
        # Cluster labels are irrelevant for the iid bootstrap point estimator;
        # assign unique labels rather than pretending the resample preserves
        # the original cluster structure.
        cluster_id=np.arange(len(indices), dtype=int),
    )


def _estimate(sample: Any, dgp: DGP, variant: str) -> dict[str, Any]:
    phi, boundary, retained, _ = _fit_variant(sample, dgp, 0, variant)
    return {
        "phi": float(phi),
        "boundary": bool(boundary),
        "retained": float(retained),
    }


def _outer_replication(
    task: tuple[str, int, int, int, int, tuple[str, ...]],
) -> dict[str, Any]:
    scenario, n, outer_seed, bootstrap_reps, bootstrap_seed, variants = task
    dgp = SCENARIOS[scenario]
    sample = generate_sample(n, outer_seed, dgp)
    target = float(population_truth(dgp)["phi_star"])
    point = {variant: _estimate(sample, dgp, variant) for variant in variants}
    values: dict[str, list[float]] = {variant: [] for variant in variants}
    boundaries = {variant: 0 for variant in variants}
    failures = {variant: 0 for variant in variants}
    rng = np.random.default_rng(bootstrap_seed)

    for _ in range(int(bootstrap_reps)):
        indices = rng.integers(0, n, size=n)
        boot_sample = _bootstrap_sample(sample, indices)
        for variant in variants:
            try:
                result = _estimate(boot_sample, dgp, variant)
                values[variant].append(float(result["phi"]))
                boundaries[variant] += int(result["boundary"])
            except (ArithmeticError, FloatingPointError, np.linalg.LinAlgError, ValueError):
                failures[variant] += 1

    row: dict[str, Any] = {
        "scenario": scenario,
        "n": int(n),
        "outer_seed": int(outer_seed),
        "bootstrap_reps_requested": int(bootstrap_reps),
        "target_phi": target,
    }
    for variant in variants:
        boot = np.asarray(values[variant], dtype=float)
        point_item = point[variant]
        if boot.size >= 2:
            q025, q975 = np.quantile(boot, [0.025, 0.975])
            boot_mean = float(np.mean(boot))
            boot_sd = float(np.std(boot, ddof=1))
            covered = bool(q025 <= target <= q975)
        else:
            q025 = q975 = boot_mean = boot_sd = float("nan")
            covered = False
        prefix = variant
        row.update({
            f"{prefix}_point_phi": float(point_item["phi"]),
            f"{prefix}_point_boundary": bool(point_item["boundary"]),
            f"{prefix}_point_retained": float(point_item["retained"]),
            f"{prefix}_bootstrap_reps_successful": int(boot.size),
            f"{prefix}_bootstrap_failures": int(failures[variant]),
            f"{prefix}_bootstrap_boundary_rate": (
                float(boundaries[variant] / boot.size) if boot.size else float("nan")
            ),
            f"{prefix}_bootstrap_mean": boot_mean,
            f"{prefix}_bootstrap_sd": boot_sd,
            f"{prefix}_percentile_q025": float(q025),
            f"{prefix}_percentile_q975": float(q975),
            f"{prefix}_percentile_coverage": covered,
        })
    return row


def _summary(rows: Sequence[dict[str, Any]], variants: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        output[scenario] = {}
        for n in sorted({int(row["n"]) for row in rows if row["scenario"] == scenario}):
            subset = [row for row in rows if row["scenario"] == scenario and int(row["n"]) == n]
            block: dict[str, Any] = {"outer_replications": len(subset), "variants": {}}
            target = float(subset[0]["target_phi"])
            for variant in variants:
                point = np.asarray([float(row[f"{variant}_point_phi"]) for row in subset])
                valid_boot = np.asarray([
                    float(row[f"{variant}_bootstrap_sd"])
                    for row in subset
                    if np.isfinite(float(row[f"{variant}_bootstrap_sd"]))
                ])
                empirical_sd = float(np.std(point, ddof=1)) if len(point) > 1 else float("nan")
                coverage = float(np.mean([
                    bool(row[f"{variant}_percentile_coverage"]) for row in subset
                ]))
                block["variants"][variant] = {
                    "target_phi": target,
                    "point_mean": float(np.mean(point)),
                    "point_bias": float(np.mean(point - target)),
                    "point_rmse": float(np.sqrt(np.mean((point - target) ** 2))),
                    "empirical_sd": empirical_sd,
                    "n_times_empirical_variance": (
                        float(n * empirical_sd**2) if np.isfinite(empirical_sd) else float("nan")
                    ),
                    "mean_bootstrap_sd": float(np.mean(valid_boot)) if valid_boot.size else float("nan"),
                    "bootstrap_sd_to_empirical_sd": (
                        float(np.mean(valid_boot) / empirical_sd)
                        if valid_boot.size and np.isfinite(empirical_sd) and empirical_sd > 0
                        else float("nan")
                    ),
                    "percentile_coverage": coverage,
                    "percentile_coverage_mc_se": float(
                        np.sqrt(coverage * (1.0 - coverage) / len(subset))
                    ),
                    "point_boundary_rate": float(np.mean([
                        bool(row[f"{variant}_point_boundary"]) for row in subset
                    ])),
                    "mean_bootstrap_boundary_rate": float(np.nanmean([
                        float(row[f"{variant}_bootstrap_boundary_rate"]) for row in subset
                    ])),
                    "mean_bootstrap_failures": float(np.mean([
                        int(row[f"{variant}_bootstrap_failures"]) for row in subset
                    ])),
                    "max_bootstrap_failures": int(max(
                        int(row[f"{variant}_bootstrap_failures"]) for row in subset
                    )),
                }
            output[scenario][str(n)] = block
    return output


def run_bootstrap(
    scenarios: Iterable[str],
    n_values: Iterable[int],
    outer_reps: int,
    bootstrap_reps: int,
    workers: int,
    seed: int,
    out_dir: Path,
    variants: Sequence[str] = VARIANTS,
) -> dict[str, Any]:
    scenarios = tuple(str(value) for value in scenarios)
    n_values = tuple(int(value) for value in n_values)
    variants = tuple(str(value) for value in variants)
    if not scenarios or not set(scenarios).issubset(SCENARIOS):
        raise ValueError(f"scenarios must be a nonempty subset of {sorted(SCENARIOS)}")
    if not n_values or any(n < 500 for n in n_values):
        raise ValueError("all sample sizes must be at least 500")
    if not variants or not set(variants).issubset(VARIANTS):
        raise ValueError(f"variants must be a nonempty subset of {VARIANTS}")
    if outer_reps < 1 or bootstrap_reps < 9 or workers < 1:
        raise ValueError("outer_reps/workers must be positive and bootstrap_reps >= 9")

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    task_index = 0
    for scenario_index, scenario in enumerate(scenarios):
        for n_index, n in enumerate(n_values):
            for outer in range(outer_reps):
                tasks.append((
                    scenario,
                    n,
                    seed + 100_000 * scenario_index + 10_000 * n_index + outer,
                    int(bootstrap_reps),
                    seed + 1_000_003 * (1 + task_index),
                    variants,
                ))
                task_index += 1
    rows: list[dict[str, Any]] = []
    csv_path = out_dir / "replications.csv"
    with csv_path.open("w", newline="") as handle:
        writer = None
        if workers == 1:
            completed = enumerate(tasks, start=1)
            results = ((index, _outer_replication(task)) for index, task in completed)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            futures = [pool.submit(_outer_replication, task) for task in tasks]
            results = ((index, future.result()) for index, future in enumerate(futures, start=1))
        try:
            for index, row in results:
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"[bootstrap] completed {index}/{len(tasks)}")
        finally:
            if workers != 1:
                pool.shutdown()

    rows.sort(key=lambda row: (str(row["scenario"]), int(row["n"]), int(row["outer_seed"])))
    result = {
        "description": (
            "Full-sample re-estimation bootstrap for the differing-slopes estimator; "
            "diagnostic only, not a bootstrap validity theorem."
        ),
        "scenarios": list(scenarios),
        "n_values": list(n_values),
        "outer_replications_per_cell": int(outer_reps),
        "bootstrap_replications_per_outer_sample": int(bootstrap_reps),
        "seed": int(seed),
        "variants": list(variants),
        "summary": _summary(rows, variants),
        "replications_csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), nargs="+", default=["baseline"])
    parser.add_argument("--n", type=int, nargs="+", default=[800, 1600, 3200])
    parser.add_argument("--outer-reps", type=int, default=40)
    parser.add_argument("--bootstrap-reps", type=int, default=199)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--variant", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_bootstrap(
        args.scenario,
        args.n,
        args.outer_reps,
        args.bootstrap_reps,
        args.workers,
        args.seed,
        args.out,
        args.variant,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"[wrote] {args.out / 'replications.csv'}")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
