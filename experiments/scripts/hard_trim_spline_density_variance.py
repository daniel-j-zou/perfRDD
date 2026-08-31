"""Asymptotic-variance check for hard trimming with spline T density.

The limiting density influence function is the hard-support Riesz score from
the manuscript.  This script calculates its variance and covariance with the
empirical trim-boundary scores under the known Gaussian DGP.  It also reports
the population pseudo-target induced by each finite spline dimension, so a
Monte Carlo comparison can distinguish sampling variance from projection bias.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from experiments.methods.spline_density import (
    project_known_density,
    spline_basis_dimension,
)
from experiments.scripts.hard_trim_asymptotic_variance import (
    HONEST_FOLD_FRACTIONS,
    _alpha_riesz_variance,
    _eta_expectation,
)
from experiments.scripts.hard_trim_crossfit_regularization import (
    T_DENSITY_SUPPORT,
)
from experiments.scripts.hard_trim_gaussian_baseline import (
    COST,
    L0,
    POLICY_BOUNDS,
    U0,
    population_truth,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = ROOT / "runs" / "hard_trim_spline_density" / "summary.json"
DEFAULT_OUT = ROOT / "runs" / "hard_trim_spline_density" / "variance_benchmark.json"


def _alpha_minus_cost(eta: float) -> float:
    return eta + 2.0 - COST


def _integrate_normal(function, split_points: Sequence[float]) -> float:
    points = sorted(set([-10.0, 10.0, *[
        float(value) for value in split_points if -10.0 < value < 10.0
    ]]))
    return float(sum(
        quad(
            lambda value: function(value) * norm.pdf(value),
            left,
            right,
            epsabs=2e-12,
            limit=300,
        )[0]
        for left, right in zip(points[:-1], points[1:])
    ))


def _boundary_loadings(phi: float, density) -> tuple[float, float]:
    lower = (
        _alpha_minus_cost(L0)
        * float(density(np.asarray(phi - L0)))
        * norm.pdf(L0)
    )
    upper = (
        _alpha_minus_cost(U0)
        * float(density(np.asarray(phi - U0)))
        * norm.pdf(U0)
    )
    return lower, upper


def _joint_density_boundary_variance(
    phi: float,
    density_representer,
    density,
) -> Dict[str, float]:
    """Integrate the shared-sample spline-density and quantile scores."""
    q10 = float(norm.ppf(0.1))
    q90 = float(norm.ppf(0.9))
    quantile_density = float(norm.pdf(q90))
    lower, upper = _boundary_loadings(phi, density)
    representer_mean = _integrate_normal(
        density_representer, [phi - U0, phi - L0]
    )

    def density_if(t: float) -> float:
        # U'(phi) contains minus the estimated density.
        return -(density_representer(t) - representer_mean)

    def boundary_if(t: float) -> float:
        q90_if = (0.9 - float(t <= q90)) / quantile_density
        q10_if = (0.1 - float(t <= q10)) / quantile_density
        return -lower * q90_if + upper * q10_if

    split_points = [q10, q90, phi - U0, phi - L0]
    density_variance = _integrate_normal(
        lambda t: density_if(t) ** 2, split_points
    )
    boundary_variance = _integrate_normal(
        lambda t: boundary_if(t) ** 2, split_points
    )
    joint_variance = _integrate_normal(
        lambda t: (density_if(t) + boundary_if(t)) ** 2, split_points
    )
    covariance = 0.5 * (
        joint_variance - density_variance - boundary_variance
    )
    return {
        "representer_mean": representer_mean,
        "density_score_variance": density_variance,
        "boundary_score_variance": boundary_variance,
        "density_boundary_covariance": covariance,
        "joint_score_variance": joint_variance,
        "boundary_l_loading": lower,
        "boundary_u_loading": upper,
    }


def calculate_limiting_variance() -> Dict[str, Any]:
    """Calculate the K-to-infinity variance in the spline-density theorem."""
    truth = population_truth()
    phi = truth["hard_phi_star"]
    curvature_squared = truth["hard_curvature"] ** 2

    utility_variance = _eta_expectation(
        lambda eta: (
            _alpha_minus_cost(eta) * norm.pdf(phi - eta)
        ) ** 2,
        L0,
        U0,
    )
    alpha = _alpha_riesz_variance(phi)

    representer_lo = phi - U0
    representer_hi = phi - L0

    def representer(t: float) -> float:
        if t < representer_lo or t > representer_hi:
            return 0.0
        eta = phi - t
        return _alpha_minus_cost(eta) * norm.pdf(eta)

    shared = _joint_density_boundary_variance(
        phi, representer, lambda value: norm.pdf(value)
    )
    full_score_variance = (
        utility_variance
        + alpha["score_variance"]
        + shared["joint_score_variance"]
    )

    quantile = float(norm.ppf(0.9))
    quantile_density = float(norm.pdf(quantile))
    empirical_quantile_variance = 0.1 * 0.9 / quantile_density ** 2
    generated_quantile_variance = 1.0 + quantile ** 2
    per_boundary_variance = (
        empirical_quantile_variance + generated_quantile_variance
    )
    independent_boundary_variance = (
        shared["boundary_l_loading"] ** 2
        / HONEST_FOLD_FRACTIONS["boundary_l"]
        + shared["boundary_u_loading"] ** 2
        / HONEST_FOLD_FRACTIONS["boundary_u"]
    ) * per_boundary_variance
    honest_components = {
        "utility": utility_variance / HONEST_FOLD_FRACTIONS["utility"],
        "alpha": alpha["score_variance"] / HONEST_FOLD_FRACTIONS["outcome"],
        "spline_density": (
            shared["density_score_variance"]
            / HONEST_FOLD_FRACTIONS["density"]
        ),
        "two_independent_boundaries": independent_boundary_variance,
        "shared_main_first_stage": 0.0,
    }
    honest_score_variance = float(sum(honest_components.values()))
    return {
        "description": (
            "K-to-infinity population variance for the hard-trimmed estimator "
            "with the manuscript's spline projection density"
        ),
        "truth": truth,
        "T_density_support": list(T_DENSITY_SUPPORT),
        "curvature_squared": curvature_squared,
        "oracle_evaluation_only": {
            "score_variance": utility_variance,
            "threshold_asymptotic_variance": utility_variance / curvature_squared,
        },
        "alpha_block": alpha,
        "spline_density_and_boundary": shared,
        "full_sample": {
            "score_variance": full_score_variance,
            "threshold_asymptotic_variance": (
                full_score_variance / curvature_squared
            ),
            "asymptotic_sd_times_sqrt_n": (
                np.sqrt(full_score_variance / curvature_squared)
            ),
        },
        "crossfit_5fold": {
            "threshold_asymptotic_variance": (
                full_score_variance / curvature_squared
            ),
            "note": "Fixed-fold cross-fitting has the same first-order limit.",
        },
        "honest_split": {
            "fold_fractions": HONEST_FOLD_FRACTIONS,
            "score_variance_components": honest_components,
            "score_variance": honest_score_variance,
            "threshold_asymptotic_variance": (
                honest_score_variance / curvature_squared
            ),
            "asymptotic_sd_times_sqrt_n": (
                np.sqrt(honest_score_variance / curvature_squared)
            ),
        },
    }


@lru_cache(maxsize=None)
def population_sieve_target(n_basis: int) -> Dict[str, float]:
    """Return the exact population target of a fixed-K density projection."""
    fit = project_known_density(norm.pdf, T_DENSITY_SUPPORT, int(n_basis))

    def score(phi: float) -> float:
        return -_eta_expectation(
            lambda eta: (
                _alpha_minus_cost(eta)
                * float(fit.density(np.asarray(phi - eta)))
            ),
            L0,
            U0,
        )

    phi = float(brentq(score, *POLICY_BOUNDS))
    curvature = -_eta_expectation(
        lambda eta: (
            _alpha_minus_cost(eta)
            * float(fit.density_derivative(np.asarray(phi - eta)))
        ),
        L0,
        U0,
    )
    true_target = population_truth()["hard_phi_star"]
    return {
        "n_basis": int(n_basis),
        "pseudo_target": phi,
        "projection_bias": phi - true_target,
        "pseudo_curvature": curvature,
        "density_support_mass": fit.support_fraction,
        "gram_condition_number": fit.gram_condition_number,
    }


def add_projection_diagnostics(payload: Dict[str, Any], ns: Sequence[int]) -> None:
    diagnostics: Dict[str, Any] = {}
    for n in sorted(set(int(value) for value in ns)):
        schemes = {
            "honest_split": int(round(HONEST_FOLD_FRACTIONS["density"] * n)),
            "crossfit_5fold": int(round(0.8 * n)),
            "full_sample": n,
        }
        diagnostics[str(n)] = {}
        for label, n_fit in schemes.items():
            n_basis = spline_basis_dimension(n_fit)
            target = dict(population_sieve_target(n_basis))
            target["density_training_size"] = n_fit
            target["sqrt_n_projection_bias"] = (
                np.sqrt(n) * target["projection_bias"]
            )
            diagnostics[str(n)][label] = target
    payload["finite_sieve_projection"] = diagnostics


def attach_monte_carlo_comparison(
    payload: Dict[str, Any], summary_path: Path,
) -> None:
    if not summary_path.exists():
        payload["monte_carlo_comparison"] = {
            "available": False,
            "path": str(summary_path),
        }
        return
    experiment = json.loads(summary_path.read_text())
    if experiment.get("density_method") != "spline":
        raise ValueError("Monte Carlo summary was not generated with spline density")
    summary = experiment["summary"]
    replications_path = summary_path.with_name("replications.csv")
    replication_rows = list(csv.DictReader(replications_path.open()))
    estimators = ("honest_split", "crossfit_5fold", "full_ridge_0")
    comparison: Dict[str, Any] = {}
    true_target = payload["truth"]["hard_phi_star"]
    for estimator in estimators:
        benchmark_name = (
            "honest_split" if estimator == "honest_split" else "full_sample"
        )
        benchmark = payload[benchmark_name]["threshold_asymptotic_variance"]
        constants = []
        rows = []
        for n_text, block in sorted(summary.items(), key=lambda item: int(item[0])):
            n = int(n_text)
            result = block["estimators"][estimator]
            constants.append(n * result["rmse"] ** 2)
            matching = [
                row for row in replication_rows if int(row["n"]) == n
            ]
            standardized = np.asarray([
                (float(row[f"{estimator}_phi"]) - true_target)
                * np.sqrt(n / benchmark)
                for row in matching
            ])
            n_basis = int(round(result["mean_density_basis"]))
            pseudo_target = population_sieve_target(n_basis)["pseudo_target"]
            pseudo_standardized = np.asarray([
                (float(row[f"{estimator}_phi"]) - pseudo_target)
                * np.sqrt(n / benchmark)
                for row in matching
            ])
            rows.append({
                "n": n,
                "replications": block["replications"],
                "density_basis": n_basis,
                "n_times_mse": n * result["rmse"] ** 2,
                "n_times_variance": n * result["sd"] ** 2,
                "sqrt_n_bias": np.sqrt(n) * result["bias"],
                "population_limit_standardization": {
                    "mean": float(np.mean(standardized)),
                    "sd": float(np.std(standardized, ddof=1)),
                    "coverage_95": float(np.mean(np.abs(standardized) <= 1.96)),
                },
                "finite_sieve_centering": {
                    "pseudo_target": pseudo_target,
                    "standardized_mean": float(np.mean(pseudo_standardized)),
                    "standardized_sd": float(
                        np.std(pseudo_standardized, ddof=1)
                    ),
                    "coverage_95": float(
                        np.mean(np.abs(pseudo_standardized) <= 1.96)
                    ),
                },
            })
        pooled = float(np.mean(constants))
        comparison[estimator] = {
            "rows": rows,
            "pooled_n_times_mse": pooled,
            "population_benchmark": benchmark,
            "ratio_to_population_benchmark": pooled / benchmark,
        }
    payload["monte_carlo_comparison"] = {
        "available": True,
        "path": str(summary_path),
        "estimators": comparison,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--monte-carlo-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--n", type=int, nargs="+", default=[20_000, 40_000, 80_000]
    )
    args = parser.parse_args(argv)
    payload = calculate_limiting_variance()
    add_projection_diagnostics(payload, args.n)
    attach_monte_carlo_comparison(payload, args.monte_carlo_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
