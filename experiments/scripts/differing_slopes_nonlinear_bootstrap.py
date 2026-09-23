"""Bootstrap coverage check for the nonlinear alpha/b simulation.

This is a diagnostic for the outcome-flexibility experiment in
``differing_slopes_nonlinear_outcome``. Each bootstrap draw re-estimates the
OLS index, hard-trim endpoints, outcome regression, and policy optimum. The
reported intervals are ordinary percentile intervals for the population
optimum. This is not a claim that the bootstrap is valid for the full
generated-index/density-Riesz theorem.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.scripts.differing_slopes_nonlinear_outcome import (
    SCENARIOS,
    VARIANTS,
    estimate,
    generate_sample,
    population_truth,
)


def _resample(sample: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[indices] for name, values in sample.items()}


def _outer_replication(
    task: tuple[str, int, int, int, int, tuple[str, ...]],
) -> dict[str, Any]:
    scenario, n, outer_seed, bootstrap_reps, bootstrap_seed, variants = task
    dgp = SCENARIOS[scenario]
    sample = generate_sample(n, outer_seed, dgp)
    target = float(population_truth(dgp))
    point = {variant: float(estimate(sample, dgp, variant)) for variant in variants}
    values = {variant: [] for variant in variants}
    failures = {variant: 0 for variant in variants}
    rng = np.random.default_rng(bootstrap_seed)

    for _ in range(int(bootstrap_reps)):
        indices = rng.integers(0, n, size=n)
        boot_sample = _resample(sample, indices)
        for variant in variants:
            try:
                values[variant].append(float(estimate(boot_sample, dgp, variant)))
            except (ArithmeticError, FloatingPointError, np.linalg.LinAlgError, ValueError):
                failures[variant] += 1

    row: dict[str, Any] = {
        "scenario": scenario,
        "n": int(n),
        "outer_seed": int(outer_seed),
        "target_phi": target,
        "bootstrap_reps_requested": int(bootstrap_reps),
    }
    for variant in variants:
        boot = np.asarray(values[variant], dtype=float)
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
            f"{prefix}_point_phi": point[variant],
            f"{prefix}_bootstrap_reps_successful": int(boot.size),
            f"{prefix}_bootstrap_failures": int(failures[variant]),
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
            target = float(subset[0]["target_phi"])
            block: dict[str, Any] = {"outer_replications": len(subset), "variants": {}}
            for variant in variants:
                point = np.asarray([row[f"{variant}_point_phi"] for row in subset], dtype=float)
                boot_sd = np.asarray([
                    row[f"{variant}_bootstrap_sd"] for row in subset
                    if np.isfinite(row[f"{variant}_bootstrap_sd"])
                ], dtype=float)
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
                    "mean_bootstrap_sd": float(np.mean(boot_sd)) if boot_sd.size else float("nan"),
                    "bootstrap_sd_to_empirical_sd": (
                        float(np.mean(boot_sd) / empirical_sd)
                        if boot_sd.size and np.isfinite(empirical_sd) and empirical_sd > 0
                        else float("nan")
                    ),
                    "percentile_coverage": coverage,
                    "percentile_coverage_mc_se": float(
                        np.sqrt(coverage * (1.0 - coverage) / len(subset))
                    ),
                    "mean_bootstrap_failures": float(np.mean([
                        row[f"{variant}_bootstrap_failures"] for row in subset
                    ])),
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
    if outer_reps < 2 or bootstrap_reps < 9 or workers < 1:
        raise ValueError("outer_reps must be at least 2; bootstrap_reps/workers must be positive")

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

    if workers == 1:
        rows = [_outer_replication(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_outer_replication, tasks))
    rows.sort(key=lambda row: (str(row["scenario"]), int(row["n"]), int(row["outer_seed"])))

    csv_path = out_dir / "replications.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "description": "Percentile bootstrap diagnostic for nonlinear alpha/b outcome fits.",
        "scenarios": list(scenarios),
        "n_values": list(n_values),
        "outer_replications_per_cell": int(outer_reps),
        "bootstrap_replications_per_outer_sample": int(bootstrap_reps),
        "workers": int(workers),
        "seed": int(seed),
        "variants": list(variants),
        "summary": _summary(rows, variants),
        "replications_csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), nargs="+", default=["nonlinear"])
    parser.add_argument("--n", type=int, nargs="+", default=[800, 1600, 3200])
    parser.add_argument("--outer-reps", type=int, default=50)
    parser.add_argument("--bootstrap-reps", type=int, default=199)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--variant", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_bootstrap(
        args.scenario, args.n, args.outer_reps, args.bootstrap_reps,
        args.workers, args.seed, args.out, args.variant,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"[wrote] {args.out / 'replications.csv'}")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
