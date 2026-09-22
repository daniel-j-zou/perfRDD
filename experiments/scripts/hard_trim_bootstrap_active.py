"""Bootstrap coverage diagnostic for the active hard-trim implementations.

This runner re-estimates the current three sample-use variants on iid bootstrap
resamples of the Gaussian hard-trimming DGP:

``decoupled_8block``
    One theorem-facing partition into eight disjoint role blocks.
``rotated_8block``
    The eight cyclic role rotations over one fixed partition.
``full_sample``
    Application-style reuse of the complete sample for nuisance fits and the
    policy criterion.

The output is deliberately diagnostic.  It reports percentile-bootstrap
coverage for the known population optimum, empirical Monte Carlo dispersion,
mean bootstrap dispersion, boundary rates, and failed-resample rates.  It does
not establish a bootstrap validity theorem for the generated-index and moving-
boundary terms.

Examples
--------
Smoke test::

    python -m experiments.scripts.hard_trim_bootstrap_active \
        --n 600 --outer-reps 2 --bootstrap-reps 9 --workers 1 \
        --density gaussian --out /tmp/active_bootstrap_smoke

Main grid (submit this through Slurm on Great Lakes)::

    python -m experiments.scripts.hard_trim_bootstrap_active \
        --n 1200 2400 4800 9600 --outer-reps 50 --bootstrap-reps 199 \
        --workers 8 --density gaussian --out experiments/runs/active_bootstrap
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.scripts.hard_trim_crossfit_regularization import (
    DENSITY_METHODS,
    _component,
    _maximize_components,
    _theory_decoupled_component,
    generate_data,
    make_role_rotated_folds,
)
from experiments.scripts.hard_trim_gaussian_baseline import (
    GeneratedData,
    population_truth,
    make_theory_folds,
)


ESTIMATORS = ("decoupled_8block", "rotated_8block", "full_sample")


def _bootstrap_data(data: GeneratedData, indices: np.ndarray) -> GeneratedData:
    """Resample all generated variables together, preserving the DGP contract."""
    return GeneratedData(
        X=np.asarray(data.X)[indices],
        eta=np.asarray(data.eta)[indices],
        T=np.asarray(data.T)[indices],
        Q=np.asarray(data.Q)[indices],
        D=np.asarray(data.D)[indices],
        Y=np.asarray(data.Y)[indices],
    )


def _estimate(
    data: GeneratedData,
    seed: int,
    density_method: str,
) -> dict[str, dict[str, Any]]:
    """Estimate all active variants on one generated sample."""
    n = len(data.Q)
    all_idx = np.arange(n, dtype=int)
    result: dict[str, dict[str, Any]] = {}

    folds = make_theory_folds(n, seed)
    component, diagnostics = _theory_decoupled_component(
        data, folds, density_method
    )
    phi, boundary, retained = _maximize_components([component])
    result["decoupled_8block"] = {
        "phi": float(phi),
        "boundary": bool(boundary),
        "retention": float(retained / len(component.eta)),
        "gamma_error": float(max(diagnostics.values())),
    }

    components = []
    max_gamma_error = 0.0
    for rotation in range(8):
        rotated_folds = make_role_rotated_folds(n, seed, rotation)
        rotated_component, rotated_diag = _theory_decoupled_component(
            data, rotated_folds, density_method
        )
        components.append(rotated_component)
        max_gamma_error = max(max_gamma_error, max(rotated_diag.values()))
    phi, boundary, retained = _maximize_components(components)
    result["rotated_8block"] = {
        "phi": float(phi),
        "boundary": bool(boundary),
        "retention": float(retained / n),
        "gamma_error": float(max_gamma_error),
    }

    component = _component(data, all_idx, all_idx, 0.0, density_method)
    phi, boundary, retained = _maximize_components([component])
    result["full_sample"] = {
        "phi": float(phi),
        "boundary": bool(boundary),
        "retention": float(retained / n),
        "gamma_error": float("nan"),
    }
    return result


def _outer_replication(
    task: tuple[int, int, int, int, str],
) -> dict[str, Any]:
    """Run one outer sample and its iid bootstrap replications."""
    n, outer_seed, bootstrap_reps, bootstrap_seed, density_method = task
    data = generate_data(n, outer_seed)
    # Keep the role assignment fixed across the bootstrap draws.  This is the
    # conditional bootstrap analogue of the estimator actually applied to the
    # outer sample; re-randomizing the split would add a separate Monte Carlo
    # component and is reserved for a sensitivity analysis.
    analysis_seed = outer_seed + 700_001
    point = _estimate(data, analysis_seed, density_method)
    target = float(population_truth()["hard_phi_star"])
    values: dict[str, list[float]] = {label: [] for label in ESTIMATORS}
    boundaries: dict[str, int] = {label: 0 for label in ESTIMATORS}
    failures: dict[str, int] = {label: 0 for label in ESTIMATORS}
    rng = np.random.default_rng(bootstrap_seed)

    for replicate in range(int(bootstrap_reps)):
        indices = rng.integers(0, n, size=n)
        sample = _bootstrap_data(data, indices)
        try:
            boot = _estimate(
                sample,
                analysis_seed,
                density_method,
            )
            for label in ESTIMATORS:
                values[label].append(float(boot[label]["phi"]))
                boundaries[label] += int(boot[label]["boundary"])
        except (ArithmeticError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            # A bootstrap draw is counted once for every estimator because the
            # active variants are evaluated together and share the resample.
            for label in ESTIMATORS:
                failures[label] += 1

    row: dict[str, Any] = {
        "n": int(n),
        "outer_seed": int(outer_seed),
        "bootstrap_reps_requested": int(bootstrap_reps),
        "target_phi": target,
        "density_method": density_method,
        "bootstrap_fold_assignment": "fixed within outer sample",
    }
    for label in ESTIMATORS:
        point_item = point[label]
        boot = np.asarray(values[label], dtype=float)
        if boot.size:
            q025, q975 = np.quantile(boot, [0.025, 0.975])
            boot_mean = float(np.mean(boot))
            boot_sd = float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0
            covered = bool(q025 <= target <= q975)
        else:
            q025 = q975 = boot_mean = boot_sd = float("nan")
            covered = False
        prefix = label
        row.update({
            f"{prefix}_point_phi": float(point_item["phi"]),
            f"{prefix}_point_boundary": bool(point_item["boundary"]),
            f"{prefix}_point_retention": float(point_item["retention"]),
            f"{prefix}_bootstrap_reps_successful": int(boot.size),
            f"{prefix}_bootstrap_failures": int(failures[label]),
            f"{prefix}_bootstrap_boundary_rate": (
                float(boundaries[label] / boot.size) if boot.size else float("nan")
            ),
            f"{prefix}_bootstrap_mean": boot_mean,
            f"{prefix}_bootstrap_sd": boot_sd,
            f"{prefix}_percentile_q025": float(q025),
            f"{prefix}_percentile_q975": float(q975),
            f"{prefix}_percentile_coverage": covered,
        })
    return row


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate point and bootstrap diagnostics by sample size and estimator."""
    target = float(population_truth()["hard_phi_star"])
    output: dict[str, Any] = {}
    for n in sorted({int(row["n"]) for row in rows}):
        subset = [row for row in rows if int(row["n"]) == n]
        block: dict[str, Any] = {"n": n, "outer_replications": len(subset), "estimators": {}}
        for label in ESTIMATORS:
            point = np.asarray([float(row[f"{label}_point_phi"]) for row in subset])
            valid_sd = np.asarray([
                float(row[f"{label}_bootstrap_sd"])
                for row in subset
                if np.isfinite(float(row[f"{label}_bootstrap_sd"]))
            ])
            emp_sd = float(np.std(point, ddof=1)) if len(point) > 1 else float("nan")
            block["estimators"][label] = {
                "target_phi": target,
                "point_mean": float(np.mean(point)),
                "point_bias": float(np.mean(point - target)),
                "point_rmse": float(np.sqrt(np.mean((point - target) ** 2))),
                "empirical_sd": emp_sd,
                "sqrt_n_scaled_bias": float(np.sqrt(n) * np.mean(point - target)),
                "n_times_empirical_variance": float(n * emp_sd ** 2) if np.isfinite(emp_sd) else float("nan"),
                "mean_bootstrap_sd": float(np.mean(valid_sd)) if valid_sd.size else float("nan"),
                "bootstrap_sd_to_empirical_sd": (
                    float(np.mean(valid_sd) / emp_sd)
                    if valid_sd.size and np.isfinite(emp_sd) and emp_sd > 0
                    else float("nan")
                ),
                "percentile_coverage": float(np.mean([
                    bool(row[f"{label}_percentile_coverage"]) for row in subset
                ])),
                "point_boundary_rate": float(np.mean([
                    bool(row[f"{label}_point_boundary"]) for row in subset
                ])),
                "mean_bootstrap_boundary_rate": float(np.nanmean([
                    float(row[f"{label}_bootstrap_boundary_rate"]) for row in subset
                ])),
                "mean_bootstrap_failures": float(np.mean([
                    int(row[f"{label}_bootstrap_failures"]) for row in subset
                ])),
                "max_bootstrap_failures": int(max(
                    int(row[f"{label}_bootstrap_failures"]) for row in subset
                )),
            }
        output[str(n)] = block
    return output


def run_bootstrap(
    n_values: Iterable[int],
    outer_reps: int,
    bootstrap_reps: int,
    workers: int,
    density_method: str,
    seed: int,
    out_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the active-estimator bootstrap study and write CSV/JSON outputs."""
    if density_method not in DENSITY_METHODS:
        raise ValueError(f"density_method must be one of {DENSITY_METHODS}")
    n_values = tuple(int(n) for n in n_values)
    if not n_values or any(n < 500 for n in n_values):
        raise ValueError("all sample sizes must be at least 500")
    if outer_reps < 1 or bootstrap_reps < 9 or workers < 1:
        raise ValueError("outer_reps/workers must be positive and bootstrap_reps >= 9")

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            int(n),
            seed + 100_000 * n_index + outer,
            int(bootstrap_reps),
            seed + 1_000_003 * (1 + n_index * outer_reps + outer),
            density_method,
        )
        for n_index, n in enumerate(n_values)
        for outer in range(int(outer_reps))
    ]
    rows: list[dict[str, Any]] = []
    csv_path = out_dir / "replications.csv"
    fields = [
        "n", "outer_seed", "bootstrap_reps_requested", "target_phi",
        "density_method", "bootstrap_fold_assignment",
    ]
    for label in ESTIMATORS:
        fields.extend([
            f"{label}_point_phi", f"{label}_point_boundary", f"{label}_point_retention",
            f"{label}_bootstrap_reps_successful", f"{label}_bootstrap_failures",
            f"{label}_bootstrap_boundary_rate", f"{label}_bootstrap_mean",
            f"{label}_bootstrap_sd", f"{label}_percentile_q025",
            f"{label}_percentile_q975", f"{label}_percentile_coverage",
        ])

    def parse_saved_row(raw: dict[str, str]) -> dict[str, Any]:
        """Restore the small set of typed fields needed for summarization."""
        parsed: dict[str, Any] = {}
        for key, value in raw.items():
            if key in {"density_method", "bootstrap_fold_assignment"}:
                parsed[key] = value
            elif key.endswith("_point_boundary") or key.endswith("_percentile_coverage"):
                parsed[key] = value.strip().lower() == "true"
            elif key in {"n", "outer_seed", "bootstrap_reps_requested"}:
                parsed[key] = int(value)
            elif key.endswith("_bootstrap_reps_successful") or key.endswith("_bootstrap_failures"):
                parsed[key] = int(value)
            else:
                parsed[key] = float(value)
        return parsed

    completed: set[tuple[int, int, str]] = set()
    if resume and csv_path.exists():
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValueError(
                    f"cannot resume {csv_path}: existing columns do not match "
                    "the current estimator output"
                )
            for raw in reader:
                saved = parse_saved_row(raw)
                rows.append(saved)
                completed.add((int(saved["n"]), int(saved["outer_seed"]), str(saved["density_method"])))
        tasks = [
            task for task in tasks
            if (int(task[0]), int(task[1]), str(task[4])) not in completed
        ]

    file_mode = "a" if resume and csv_path.exists() and csv_path.stat().st_size > 0 else "w"
    with csv_path.open(file_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if file_mode == "w":
            writer.writeheader()
        if resume and completed:
            print(
                f"[bootstrap] resuming {csv_path}: "
                f"{len(completed)} completed outer samples; "
                f"{len(tasks)} remaining",
                flush=True,
            )
        if workers == 1:
            iterator = ((_outer_replication(task), i) for i, task in enumerate(tasks, 1))
            for row, index in iterator:
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"[bootstrap] completed {index}/{len(tasks)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_outer_replication, task) for task in tasks]
                for index, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()
                    print(f"[bootstrap] completed {index}/{len(tasks)}", flush=True)

    rows.sort(key=lambda row: (int(row["n"]), int(row["outer_seed"])))
    summary = {
        "description": (
            "Iid re-estimation percentile bootstrap diagnostic for the active "
            "hard-trim estimators; role folds are fixed within each outer sample; "
            "not a bootstrap validity theorem."
        ),
        "estimators": list(ESTIMATORS),
        "density_method": density_method,
        "n_values": list(n_values),
        "outer_replications_per_cell": int(outer_reps),
        "bootstrap_replications_per_outer_sample": int(bootstrap_reps),
        "seed": int(seed),
        "resume": bool(resume),
        "target_phi": float(population_truth()["hard_phi_star"]),
        "summary": _summarize(rows),
        "replications_csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1200, 2400, 4800, 9600])
    parser.add_argument("--outer-reps", type=int, default=50)
    parser.add_argument("--bootstrap-reps", type=int, default=199)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--density", choices=DENSITY_METHODS, default="gaussian")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from completed rows in the output CSV if it exists",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_bootstrap(
        args.n, args.outer_reps, args.bootstrap_reps, args.workers,
        args.density, args.seed, args.out, args.resume,
    )
    print(json.dumps(summary["summary"], indent=2))
    print(f"[wrote] {args.out / 'replications.csv'}")
    print(f"[wrote] {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
