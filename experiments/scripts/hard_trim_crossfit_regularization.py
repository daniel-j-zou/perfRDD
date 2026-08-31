"""Finite-sample comparison for exact hard-trimming implementations.

All estimators maximize the same hard-support-trimmed population criterion.
They differ only in sample reuse and regularization:

* ``honest_split`` uses the disjoint blocks in the hard-trim baseline;
* ``crossfit_5fold`` evaluates every observation with out-of-fold nuisances;
* ``full_ridge_*`` fits and evaluates on the full sample over a ridge grid.

Every variant fits its outcome nuisance spline on the deterministic
neighborhood J used in the theory audit.  The T distribution can be estimated
either by the original Gaussian location-scale fit or by the manuscript's
least-squares spline projection density.  This isolates cross-fitting and
regularization from the separate application choice of putting spline
boundaries at the estimated trim endpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Protocol, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from experiments.methods.perfrdd import _eval_basis
from experiments.methods.spline_density import SplineDensityFit, fit_spline_density
from experiments.scripts.hard_trim_gaussian_baseline import (
    COST,
    EPS,
    GeneratedData,
    NUISANCE_SUPPORT,
    PHI_0,
    POLICY_BOUNDS,
    _estimate_T_normal,
    _estimate_boundaries,
    _fit_gamma,
    _fit_spline_plm,
    _predict_T,
    generate_data,
    make_folds,
    population_truth,
    population_utility,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "hard_trim_crossfit_regularization"
DEFAULT_SPLINE_OUT = ROOT / "runs" / "hard_trim_spline_density"
DEFAULT_RIDGE_GRID = (0.0, 0.0001, 0.001, 0.01, 0.1)
# The hard target only evaluates T-density arguments in approximately
# [-2.8, 2.8] under the fixed policy and trim windows.  This deterministic
# interval leaves a margin without spending scarce finite-sample basis
# functions on irrelevant Gaussian tails.
T_DENSITY_SUPPORT = (-3.0, 3.0)
DENSITY_METHODS = ("gaussian", "spline")


class TDensity(Protocol):
    """Minimal distribution interface needed by the policy criterion."""

    def survival(self, points: np.ndarray | float) -> np.ndarray:
        ...


@dataclass(frozen=True)
class GaussianTDensity:
    """Gaussian location-scale nuisance retained as the old benchmark."""

    mean: float
    sd: float

    def survival(self, points: np.ndarray | float) -> np.ndarray:
        return np.asarray(norm.sf((np.asarray(points) - self.mean) / self.sd))


@dataclass(frozen=True)
class EvaluationComponent:
    """One held-out fold and the training-fold nuisances used to score it."""

    eta: np.ndarray
    hard_weights: np.ndarray
    treatment_effect: np.ndarray
    T_density: TDensity


def _fit_T_density(T_values: np.ndarray, method: str) -> TDensity:
    """Fit either the legacy Gaussian or manuscript spline T nuisance."""
    if method == "gaussian":
        mean, sd = _estimate_T_normal(T_values)
        return GaussianTDensity(mean, sd)
    if method == "spline":
        return fit_spline_density(T_values, T_DENSITY_SUPPORT)
    raise ValueError(f"unknown density method: {method!r}")


def _density_basis_count(density: TDensity) -> int:
    return density.n_basis if isinstance(density, SplineDensityFit) else 2


def make_crossfit_folds(n: int, seed: int, n_folds: int = 5) -> list[np.ndarray]:
    if n_folds < 2 or n_folds > n:
        raise ValueError("n_folds must lie between 2 and n")
    rng = np.random.default_rng(seed + 87_401_039)
    return [
        np.asarray(chunk, dtype=int)
        for chunk in np.array_split(rng.permutation(n), n_folds)
    ]


def _boundaries_from_T(T_train: np.ndarray) -> tuple[float, float]:
    l_hat = PHI_0 - float(np.quantile(T_train, 1.0 - EPS))
    u_hat = PHI_0 - float(np.quantile(T_train, EPS))
    if not l_hat < u_hat:
        raise ValueError(f"estimated overlap window is invalid: [{l_hat}, {u_hat}]")
    return l_hat, u_hat


def _maximize_components(
    components: Sequence[EvaluationComponent],
) -> tuple[float, bool, float]:
    """Maximize aggregate utility from one or more evaluation folds."""
    denominator = float(sum(np.sum(part.hard_weights) for part in components))
    if denominator < 20.0:
        raise ValueError("too few hard-trimmed evaluation observations")

    def objective(phi: float) -> float:
        numerator = 0.0
        for part in components:
            probability = part.T_density.survival(phi - part.eta)
            numerator += float(np.sum(
                part.hard_weights
                * (part.treatment_effect - COST)
                * probability
            ))
        return numerator / denominator

    result = minimize_scalar(
        lambda value: -objective(float(value)),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    phi_hat = float(result.x)
    tolerance = 2e-4
    boundary = bool(
        phi_hat <= POLICY_BOUNDS[0] + tolerance
        or phi_hat >= POLICY_BOUNDS[1] - tolerance
    )
    return phi_hat, boundary, denominator


def _component(
    data: GeneratedData,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    ridge_scale: float,
    density_method: str,
) -> EvaluationComponent:
    """Fit nuisances on ``train_idx`` and construct held-out policy inputs."""
    gamma_hat = _fit_gamma(data, train_idx)
    eta_hat = data.Q - _predict_T(data.X, gamma_hat)
    T_train = _predict_T(data.X[train_idx], gamma_hat)
    l_hat, u_hat = _boundaries_from_T(T_train)
    fit = _fit_spline_plm(
        data,
        train_idx,
        eta_hat,
        NUISANCE_SUPPORT,
        ridge_scale=ridge_scale,
    )
    T_density = _fit_T_density(T_train, density_method)
    eta_eval = eta_hat[eval_idx]
    weights = ((eta_eval >= l_hat) & (eta_eval <= u_hat)).astype(float)
    effect = _eval_basis(eta_eval, fit.info) @ fit.omega_treat
    return EvaluationComponent(
        eta=eta_eval,
        hard_weights=weights,
        treatment_effect=effect,
        T_density=T_density,
    )


def _ridge_label(ridge_scale: float) -> str:
    value = f"{ridge_scale:g}".replace(".", "p").replace("-", "m")
    return f"full_ridge_{value}"


def _crossfit_label(n_folds: int) -> str:
    return f"crossfit_{n_folds}fold"


def estimator_labels(
    ridge_grid: Sequence[float], n_folds: int = 5,
) -> list[str]:
    return ["honest_split", _crossfit_label(n_folds)] + [
        _ridge_label(float(value)) for value in ridge_grid
    ]


def run_replication(
    n: int,
    seed: int,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    n_folds: int = 5,
    density_method: str = "gaussian",
) -> Dict[str, Any]:
    if density_method not in DENSITY_METHODS:
        raise ValueError(f"density_method must be one of {DENSITY_METHODS}")
    data = generate_data(n, seed)
    target = population_truth()["hard_phi_star"]
    all_idx = np.arange(n)
    result: Dict[str, Any] = {"n": int(n), "seed": int(seed)}

    # Conservative theorem-style construction with mutually disjoint blocks.
    honest = make_folds(n, seed)
    gamma_hat = _fit_gamma(data, honest["first_stage"])
    eta_hat = data.Q - _predict_T(data.X, gamma_hat)
    l_hat, u_hat = _estimate_boundaries(data, honest)
    fit = _fit_spline_plm(data, honest["outcome"], eta_hat, NUISANCE_SUPPORT)
    T_density_values = _predict_T(data.X[honest["density"]], gamma_hat)
    T_density = _fit_T_density(T_density_values, density_method)
    eta_eval = eta_hat[honest["utility"]]
    weights = ((eta_eval >= l_hat) & (eta_eval <= u_hat)).astype(float)
    effect = _eval_basis(eta_eval, fit.info) @ fit.omega_treat
    phi, boundary, retained = _maximize_components(
        [EvaluationComponent(eta_eval, weights, effect, T_density)]
    )
    result.update({
        "honest_split_phi": phi,
        "honest_split_boundary": boundary,
        "honest_split_retention": retained / len(eta_eval),
        "honest_split_density_basis": _density_basis_count(T_density),
    })

    # Standard K-fold cross-fitting: held-out evaluation, training-fold nuisances.
    components = []
    for eval_idx in make_crossfit_folds(n, seed, n_folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[eval_idx] = False
        train_idx = all_idx[train_mask]
        components.append(
            _component(data, train_idx, eval_idx, 0.0, density_method)
        )
    phi, boundary, retained = _maximize_components(components)
    crossfit_label = _crossfit_label(n_folds)
    result.update({
        f"{crossfit_label}_phi": phi,
        f"{crossfit_label}_boundary": boundary,
        f"{crossfit_label}_retention": retained / n,
        f"{crossfit_label}_density_basis": float(np.mean([
            _density_basis_count(part.T_density) for part in components
        ])),
    })

    # Full-sample application-style variants, including its default ridge 0.01.
    for ridge_scale in ridge_grid:
        component = _component(
            data, all_idx, all_idx, float(ridge_scale), density_method
        )
        phi, boundary, retained = _maximize_components([component])
        label = _ridge_label(float(ridge_scale))
        result.update({
            f"{label}_phi": phi,
            f"{label}_boundary": boundary,
            f"{label}_retention": retained / n,
            f"{label}_density_basis": _density_basis_count(component.T_density),
        })

    for label in estimator_labels(ridge_grid, n_folds):
        result[f"{label}_squared_error"] = float(
            (result[f"{label}_phi"] - target) ** 2
        )
    return result


def _worker(task: tuple[int, int, tuple[float, ...], int, str]) -> Dict[str, Any]:
    return run_replication(task[0], task[1], task[2], task[3], task[4])


def summarize(
    rows: Sequence[Dict[str, Any]],
    ridge_grid: Sequence[float],
    n_folds: int = 5,
) -> Dict[str, Any]:
    target = population_truth()["hard_phi_star"]
    target_utility = population_utility(target, True)
    output: Dict[str, Any] = {}
    for n in sorted({int(row["n"]) for row in rows}):
        subset = [row for row in rows if int(row["n"]) == n]
        block: Dict[str, Any] = {
            "n": n,
            "replications": len(subset),
            "estimators": {},
        }
        for label in estimator_labels(ridge_grid, n_folds):
            values = np.asarray([row[f"{label}_phi"] for row in subset])
            errors = values - target
            regrets = np.asarray([
                target_utility - population_utility(float(value), True)
                for value in values
            ])
            block["estimators"][label] = {
                "target": float(target),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "bias": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mae": float(np.mean(np.abs(errors))),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "boundary_rate": float(np.mean([
                    row[f"{label}_boundary"] for row in subset
                ])),
                "mean_retention": float(np.mean([
                    row[f"{label}_retention"] for row in subset
                ])),
                "mean_density_basis": float(np.mean([
                    row[f"{label}_density_basis"] for row in subset
                ])),
                "mean_utility_regret": float(np.mean(regrets)),
            }
        output[str(n)] = block
    return output


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    summary: Dict[str, Any],
    ridge_grid: Sequence[float],
    n_folds: int,
    path: Path,
) -> None:
    ns = np.asarray(sorted(int(value) for value in summary))
    labels = estimator_labels(ridge_grid, n_folds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label in labels:
        rmse = [summary[str(n)]["estimators"][label]["rmse"] for n in ns]
        regret = [
            summary[str(n)]["estimators"][label]["mean_utility_regret"] for n in ns
        ]
        axes[0].plot(ns, rmse, "o-", label=label)
        axes[1].plot(ns, regret, "o-", label=label)
    axes[0].set_title("Hard-threshold RMSE")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("Population utility regret")
    axes[1].set_ylabel("regret")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("n")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    ns: Iterable[int],
    reps: int,
    workers: int,
    out_dir: Path,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    n_folds: int = 5,
    density_method: str = "gaussian",
) -> Dict[str, Any]:
    if density_method not in DENSITY_METHODS:
        raise ValueError(f"density_method must be one of {DENSITY_METHODS}")
    ridge_grid = tuple(float(value) for value in ridge_grid)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (int(n), int(seed), ridge_grid, int(n_folds), density_method)
        for n in ns
        for seed in range(reps)
    ]
    if workers == 1:
        rows = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_worker, tasks, chunksize=2))
    rows.sort(key=lambda row: (int(row["n"]), int(row["seed"])))
    summary = summarize(rows, ridge_grid, n_folds)
    payload = {
        "description": "Exact hard trimming: sample reuse and ridge comparison",
        "target": population_truth(),
        "n_folds": int(n_folds),
        "density_method": density_method,
        "deterministic_T_density_support": (
            list(T_DENSITY_SUPPORT) if density_method == "spline" else None
        ),
        "ridge_grid": list(ridge_grid),
        "ridge_definition": "spline penalty lambda = ridge_scale / sqrt(n_fit)",
        "deterministic_nuisance_support": list(NUISANCE_SUPPORT),
        "replications": int(reps),
        "summary": summary,
    }
    _write_csv(rows, out_dir / "replications.csv")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    _plot(summary, ridge_grid, n_folds, out_dir / "summary.png")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 2500, 5000, 10000])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--ridge", type=float, nargs="+", default=list(DEFAULT_RIDGE_GRID)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--density", choices=DENSITY_METHODS, default="gaussian",
        help="T-distribution nuisance: legacy Gaussian or manuscript spline",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output directory (defaults to a density-method-specific run folder)",
    )
    args = parser.parse_args(argv)
    if args.reps <= 0 or args.workers <= 0:
        parser.error("--reps and --workers must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least two")
    if any(n < 500 for n in args.n):
        parser.error("all sample sizes must be at least 500")
    if any(value < 0.0 for value in args.ridge):
        parser.error("ridge scales must be nonnegative")
    out_dir = args.out or (
        DEFAULT_SPLINE_OUT if args.density == "spline" else DEFAULT_OUT
    )
    payload = run_experiment(
        args.n,
        args.reps,
        args.workers,
        out_dir,
        args.ridge,
        args.folds,
        args.density,
    )
    print(json.dumps({"target": payload["target"], "summary": payload["summary"]}, indent=2))
    print(f"[wrote] {out_dir / 'replications.csv'}")
    print(f"[wrote] {out_dir / 'summary.json'}")
    print(f"[wrote] {out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
