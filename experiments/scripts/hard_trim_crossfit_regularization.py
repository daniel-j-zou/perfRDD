"""Finite-sample comparison for exact hard-trimming implementations.

All estimators maximize the same hard-support-trimmed population criterion.
The primary comparison is deliberately explicit about sample reuse:

* ``decoupled_8block`` uses one theorem-aligned assignment of eight disjoint
  blocks: separate first-stage source folds for the outcome, density, and
  evaluation blocks, plus independent lower- and upper-boundary blocks;
* ``rotated_8block`` repeats that fully decoupled construction over all eight
  cyclic role assignments and maximizes the aggregate held-out criterion;
* ``full_sample`` fits every nuisance and evaluates the criterion on the same
  sample.  ``full_ridge_*`` adds optional regularized versions of that fit.

Every variant fits its outcome nuisance spline on the deterministic
neighborhood J used in the theory audit.  The T distribution can be estimated
either by the original Gaussian location-scale fit or by the manuscript's
least-squares spline projection density.  This isolates sample-use choices and
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
    GAMMA,
    GeneratedData,
    NUISANCE_SUPPORT,
    PHI_0,
    POLICY_BOUNDS,
    THEORY_FOLD_FRACTIONS,
    _estimate_T_normal,
    _estimate_theory_boundaries,
    _fit_gamma,
    _fit_spline_plm,
    _predict_T,
    generate_data,
    make_theory_folds,
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


def make_role_rotated_folds(
    n: int, seed: int, rotation: int,
) -> Dict[str, np.ndarray]:
    """Rotate the eight theorem roles over one fixed eight-way partition.

    A single call returns a valid theorem-facing split.  Across rotations,
    every physical block serves every role exactly once.  The partition itself
    is held fixed so that the comparison changes only role assignment, not the
    random sample split.  The rotated estimator remains an implementation
    diagnostic: averaging the eight criteria introduces cross-rotation
    covariance that is not covered by the single-split CLT.
    """
    names = tuple(THEORY_FOLD_FRACTIONS)
    if not 0 <= int(rotation) < len(names):
        raise ValueError(f"rotation must lie in [0, {len(names) - 1}]")
    base = make_theory_folds(n, seed)
    blocks = [base[name] for name in names]
    shift = int(rotation)
    return {
        name: np.asarray(blocks[(position + shift) % len(names)], dtype=int)
        for position, name in enumerate(names)
    }


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


def _theory_decoupled_component(
    data: GeneratedData,
    folds: Dict[str, np.ndarray],
    density_method: str,
) -> tuple[EvaluationComponent, Dict[str, float]]:
    """Construct one fully decoupled theorem-facing evaluation component.

    The three main score blocks use distinct first-stage fits.  The lower and
    upper boundary blocks use two additional fits, each on its own endpoint
    block.  This is the eight-block analogue of the paper's decoupled split;
    no first-stage estimate is shared across main nuisance/evaluation blocks.
    """
    gamma_alpha = _fit_gamma(data, folds["gamma_alpha"])
    eta_alpha = data.Q - _predict_T(data.X, gamma_alpha)
    fit = _fit_spline_plm(data, folds["outcome"], eta_alpha, NUISANCE_SUPPORT)

    gamma_g = _fit_gamma(data, folds["gamma_g"])
    T_density = _fit_T_density(
        _predict_T(data.X[folds["density"]], gamma_g), density_method
    )

    gamma_U = _fit_gamma(data, folds["gamma_U"])
    eval_idx = folds["utility"]
    eta_eval = data.Q[eval_idx] - _predict_T(data.X[eval_idx], gamma_U)
    l_hat, u_hat, gamma_l, gamma_u = _estimate_theory_boundaries(data, folds)
    weights = ((eta_eval >= l_hat) & (eta_eval <= u_hat)).astype(float)
    effect = _eval_basis(eta_eval, fit.info) @ fit.omega_treat
    component = EvaluationComponent(
        eta=eta_eval,
        hard_weights=weights,
        treatment_effect=effect,
        T_density=T_density,
    )
    diagnostics = {
        "gamma_alpha_error": float(np.linalg.norm(gamma_alpha[1:] - GAMMA)),
        "gamma_g_error": float(np.linalg.norm(gamma_g[1:] - GAMMA)),
        "gamma_U_error": float(np.linalg.norm(gamma_U[1:] - GAMMA)),
        "gamma_l_error": float(np.linalg.norm(gamma_l[1:] - GAMMA)),
        "gamma_u_error": float(np.linalg.norm(gamma_u[1:] - GAMMA)),
        "l_hat": float(l_hat),
        "u_hat": float(u_hat),
    }
    return component, diagnostics


def _ridge_label(ridge_scale: float) -> str:
    value = f"{ridge_scale:g}".replace(".", "p").replace("-", "m")
    return f"full_ridge_{value}"


def estimator_labels(ridge_grid: Sequence[float]) -> list[str]:
    labels = ["decoupled_8block", "rotated_8block", "full_sample"]
    labels.extend(
        _ridge_label(float(value))
        for value in ridge_grid
        if float(value) > 0.0
    )
    return labels


def run_replication(
    n: int,
    seed: int,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    density_method: str = "gaussian",
) -> Dict[str, Any]:
    if density_method not in DENSITY_METHODS:
        raise ValueError(f"density_method must be one of {DENSITY_METHODS}")
    data = generate_data(n, seed)
    target = population_truth()["hard_phi_star"]
    all_idx = np.arange(n)
    result: Dict[str, Any] = {"n": int(n), "seed": int(seed)}

    # The standard theorem-facing estimator uses one fixed eight-block role
    # assignment and five first-stage fits.
    theory_folds = make_theory_folds(n, seed)
    theory_component, theory_diag = _theory_decoupled_component(
        data, theory_folds, density_method
    )
    eta_eval = theory_component.eta
    weights = theory_component.hard_weights
    phi, boundary, retained = _maximize_components(
        [theory_component]
    )
    result.update({
        "decoupled_8block_phi": phi,
        "decoupled_8block_boundary": boundary,
        "decoupled_8block_retention": retained / len(eta_eval),
        "decoupled_8block_density_basis": _density_basis_count(
            theory_component.T_density
        ),
        "theory_fold_counts": {
            name: int(len(index)) for name, index in theory_folds.items()
        },
        "theory_first_stage_diagnostics": theory_diag,
    })

    # Role-rotated fully decoupled implementation.  Each rotation is itself a
    # valid eight-block split; aggregate the held-out criteria before taking
    # the argmax rather than averaging the eight threshold estimates.
    rotated_components = []
    rotated_diagnostics = []
    for rotation in range(len(THEORY_FOLD_FRACTIONS)):
        rotated_folds = make_role_rotated_folds(n, seed, rotation)
        component, diagnostics = _theory_decoupled_component(
            data, rotated_folds, density_method
        )
        rotated_components.append(component)
        rotated_diagnostics.append(diagnostics)
    phi, boundary, retained = _maximize_components(rotated_components)
    result.update({
        "rotated_8block_phi": phi,
        "rotated_8block_boundary": boundary,
        "rotated_8block_retention": retained / n,
        "rotated_8block_density_basis": float(np.mean([
            _density_basis_count(part.T_density)
            for part in rotated_components
        ])),
        "rotated_8block_rotations": len(rotated_components),
        "rotated_8block_max_gamma_error": float(max(
            max(
                diagnostics[key]
                for key in (
                    "gamma_alpha_error", "gamma_g_error", "gamma_U_error",
                    "gamma_l_error", "gamma_u_error",
                )
            )
            for diagnostics in rotated_diagnostics
        )),
    })

    # Full-sample application-style estimator and optional ridge variants.
    component = _component(data, all_idx, all_idx, 0.0, density_method)
    phi, boundary, retained = _maximize_components([component])
    result.update({
        "full_sample_phi": phi,
        "full_sample_boundary": boundary,
        "full_sample_retention": retained / n,
        "full_sample_density_basis": _density_basis_count(component.T_density),
    })
    for ridge_scale in ridge_grid:
        if float(ridge_scale) <= 0.0:
            continue
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

    for label in estimator_labels(ridge_grid):
        result[f"{label}_squared_error"] = float(
            (result[f"{label}_phi"] - target) ** 2
        )
    return result


def _worker(task: tuple[int, int, tuple[float, ...], str]) -> Dict[str, Any]:
    return run_replication(task[0], task[1], task[2], task[3])


def summarize(
    rows: Sequence[Dict[str, Any]],
    ridge_grid: Sequence[float],
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
        for label in estimator_labels(ridge_grid):
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
    path: Path,
) -> None:
    ns = np.asarray(sorted(int(value) for value in summary))
    labels = estimator_labels(ridge_grid)
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
    density_method: str = "gaussian",
) -> Dict[str, Any]:
    if density_method not in DENSITY_METHODS:
        raise ValueError(f"density_method must be one of {DENSITY_METHODS}")
    ridge_grid = tuple(float(value) for value in ridge_grid)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (int(n), int(seed), ridge_grid, density_method)
        for n in ns
        for seed in range(reps)
    ]
    if workers == 1:
        rows = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_worker, tasks, chunksize=2))
    rows.sort(key=lambda row: (int(row["n"]), int(row["seed"])))
    summary = summarize(rows, ridge_grid)
    payload = {
        "description": (
            "Exact hard trimming: fixed decoupling, role-rotated decoupling, "
            "and full-sample reuse"
        ),
        "target": population_truth(),
        "density_method": density_method,
        "deterministic_T_density_support": (
            list(T_DENSITY_SUPPORT) if density_method == "spline" else None
        ),
        "ridge_grid": list(ridge_grid),
        "ridge_definition": "spline penalty lambda = ridge_scale / sqrt(n_fit)",
        "deterministic_nuisance_support": list(NUISANCE_SUPPORT),
        "decoupled_design": "eight disjoint blocks; five role-specific first-stage fits",
        "decoupled_fold_fractions": THEORY_FOLD_FRACTIONS,
        "rotated_design": (
            "all eight cyclic role assignments over one fixed eight-way "
            "partition; aggregate held-out criteria"
        ),
        "rotated_role_count": len(THEORY_FOLD_FRACTIONS),
        "replications": int(reps),
        "summary": summary,
    }
    _write_csv(rows, out_dir / "replications.csv")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    _plot(summary, ridge_grid, out_dir / "summary.png")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 2500, 5000, 10000])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--ridge", type=float, nargs="+", default=list(DEFAULT_RIDGE_GRID)
    )
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
        args.density,
    )
    print(json.dumps({"target": payload["target"], "summary": payload["summary"]}, indent=2))
    print(f"[wrote] {out_dir / 'replications.csv'}")
    print(f"[wrote] {out_dir / 'summary.json'}")
    print(f"[wrote] {out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
