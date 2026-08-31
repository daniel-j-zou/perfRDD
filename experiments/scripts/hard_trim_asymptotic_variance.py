"""DGP-known asymptotic variance for the Gaussian hard-trim simulation.

The calculation matches the estimator implemented in
``hard_trim_crossfit_regularization.py``:

* alpha is estimated by the correctly specified partially linear model on a
  fixed nuisance interval;
* the distribution of T is estimated as Gaussian through its sample mean and
  standard deviation;
* hard-trim endpoints are empirical T quantiles; and
* the policy threshold is the maximizer of the exact hard-gated criterion.

This is not the manuscript's density-sieve variance.  It is the population
variance benchmark for the simulation estimator actually run.  The distinction
matters because the Gaussian mean/standard-deviation block has a different
influence function from the manuscript's spline density block.

Under the DGP's X-independent standard-normal eta and a shared first-stage
estimate, the first-stage perturbations cancel across the main plug-in blocks.
For the honest split they also cancel across the main blocks, while the two
separately estimated boundaries retain their generated-quantile variance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from experiments.scripts.hard_trim_gaussian_baseline import (
    COST,
    L0,
    NUISANCE_SUPPORT,
    SIGMA_Y,
    U0,
    population_truth,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "hard_trim_asymptotic_check" / "variance_benchmark.json"
DEFAULT_LARGE_N_SUMMARY = (
    ROOT / "runs" / "hard_trim_asymptotic_check" / "summary.json"
)

# Fractions produced by hard_trim_gaussian_baseline.make_folds().
HONEST_FOLD_FRACTIONS = {
    "first_stage": 0.15,
    "boundary_l": 0.10,
    "boundary_u": 0.10,
    "outcome": 0.30,
    "density": 0.10,
    "utility": 0.25,
}


def _eta_expectation(function, lo: float, hi: float) -> float:
    return float(quad(
        lambda eta: function(eta) * norm.pdf(eta),
        lo,
        hi,
        epsabs=2e-12,
        limit=300,
    )[0])


def _alpha_riesz_variance(phi_star: float) -> Dict[str, float]:
    """Compute Var(epsilon * r_alpha) from the continuous Riesz solution.

    Rotational symmetry reduces X to T=gamma'X.  On the fixed nuisance
    interval J, the Riesz representer has the form

        r_alpha = (D-e(eta)) A(eta) + k T,

    where e(eta)=Phi(eta).  The scalar k enforces orthogonality to T and A
    enforces the treatment-effect loading pointwise in eta.
    """
    lo, hi = NUISANCE_SUPPORT
    probability_J = float(norm.cdf(hi) - norm.cdf(lo))

    def propensity(eta: float) -> float:
        return float(norm.cdf(eta))

    def selection_moment(eta: float) -> float:
        return float(norm.pdf(eta))

    def loading(eta: float) -> float:
        if eta < L0 or eta > U0:
            return 0.0
        return float(norm.pdf(phi_star - eta))

    def propensity_variance(eta: float) -> float:
        value = propensity(eta)
        return value * (1.0 - value)

    B = _eta_expectation(
        lambda eta: selection_moment(eta) ** 2 / propensity_variance(eta),
        lo,
        hi,
    )
    C = _eta_expectation(
        lambda eta: selection_moment(eta) * loading(eta)
        / propensity_variance(eta),
        lo,
        hi,
    )
    k = C / (B - probability_J)

    def A(eta: float) -> float:
        return (
            loading(eta) - k * selection_moment(eta)
        ) / propensity_variance(eta)

    representer_second_moment = _eta_expectation(
        lambda eta: (
            propensity_variance(eta) * A(eta) ** 2
            + 2.0 * k * A(eta) * selection_moment(eta)
            + k ** 2
        ),
        lo,
        hi,
    )
    score_variance = SIGMA_Y ** 2 * representer_second_moment
    return {
        "riesz_linear_T_coefficient": float(k),
        "riesz_second_moment": representer_second_moment,
        "score_variance": score_variance,
    }


def _density_and_boundary_quantities(phi_star: float, curvature: float) -> Dict[str, float]:
    """Return Gaussian-density derivatives and hard-boundary loadings."""
    density_mu_derivative = -curvature
    density_sd_derivative = -_eta_expectation(
        lambda eta: (
            (eta + 2.0 - COST)
            * ((phi_star - eta) ** 2 - 1.0)
            * norm.pdf(phi_star - eta)
        ),
        L0,
        U0,
    )
    density_score_variance = (
        density_mu_derivative ** 2 + 0.5 * density_sd_derivative ** 2
    )
    boundary_l = (
        (L0 + 2.0 - COST) * norm.pdf(phi_star - L0) * norm.pdf(L0)
    )
    boundary_u = (
        (U0 + 2.0 - COST) * norm.pdf(phi_star - U0) * norm.pdf(U0)
    )
    return {
        "density_mu_derivative": density_mu_derivative,
        "density_sd_derivative": density_sd_derivative,
        "density_score_variance": density_score_variance,
        "boundary_l_loading": boundary_l,
        "boundary_u_loading": boundary_u,
    }


def _shared_density_boundary_variance(quantities: Dict[str, float]) -> Dict[str, float]:
    """Integrate the correlated T-block influence function exactly."""
    d_mu = quantities["density_mu_derivative"]
    d_sd = quantities["density_sd_derivative"]
    b_l = quantities["boundary_l_loading"]
    b_u = quantities["boundary_u_loading"]
    q10 = float(norm.ppf(0.1))
    q90 = float(norm.ppf(0.9))
    f_quantile = float(norm.pdf(q90))

    def density_if(t: float) -> float:
        return d_mu * t + 0.5 * d_sd * (t ** 2 - 1.0)

    def boundary_if(t: float) -> float:
        q90_if = (0.9 - float(t <= q90)) / f_quantile
        q10_if = (0.1 - float(t <= q10)) / f_quantile
        return -b_l * q90_if + b_u * q10_if

    intervals = ((-10.0, q10), (q10, q90), (q90, 10.0))
    density_variance = sum(quad(
        lambda t: density_if(t) ** 2 * norm.pdf(t), a, b, epsabs=2e-12
    )[0] for a, b in intervals)
    boundary_variance = sum(quad(
        lambda t: boundary_if(t) ** 2 * norm.pdf(t), a, b, epsabs=2e-12
    )[0] for a, b in intervals)
    joint_variance = sum(quad(
        lambda t: (density_if(t) + boundary_if(t)) ** 2 * norm.pdf(t),
        a,
        b,
        epsabs=2e-12,
    )[0] for a, b in intervals)
    covariance = 0.5 * (joint_variance - density_variance - boundary_variance)
    return {
        "density_score_variance": float(density_variance),
        "boundary_score_variance": float(boundary_variance),
        "density_boundary_covariance": float(covariance),
        "joint_score_variance": float(joint_variance),
    }


def calculate_variance_benchmarks() -> Dict[str, Any]:
    truth = population_truth()
    phi_star = truth["hard_phi_star"]
    curvature = truth["hard_curvature"]
    curvature_squared = curvature ** 2

    utility_score_variance = _eta_expectation(
        lambda eta: (
            (eta + 2.0 - COST) * norm.pdf(phi_star - eta)
        ) ** 2,
        L0,
        U0,
    )
    alpha = _alpha_riesz_variance(phi_star)
    quantities = _density_and_boundary_quantities(phi_star, curvature)
    shared_T = _shared_density_boundary_variance(quantities)

    full_score_variance = (
        utility_score_variance
        + alpha["score_variance"]
        + shared_T["joint_score_variance"]
    )
    full_threshold_variance = full_score_variance / curvature_squared

    quantile = float(norm.ppf(0.9))
    quantile_density = float(norm.pdf(quantile))
    empirical_quantile_variance = 0.1 * 0.9 / quantile_density ** 2
    generated_quantile_variance = 1.0 + quantile ** 2
    per_boundary_variance = (
        empirical_quantile_variance + generated_quantile_variance
    )
    honest_boundary_score_variance = (
        quantities["boundary_l_loading"] ** 2
        / HONEST_FOLD_FRACTIONS["boundary_l"]
        + quantities["boundary_u_loading"] ** 2
        / HONEST_FOLD_FRACTIONS["boundary_u"]
    ) * per_boundary_variance
    honest_components = {
        "utility": utility_score_variance / HONEST_FOLD_FRACTIONS["utility"],
        "alpha": alpha["score_variance"] / HONEST_FOLD_FRACTIONS["outcome"],
        "gaussian_density": quantities["density_score_variance"]
        / HONEST_FOLD_FRACTIONS["density"],
        "two_independent_boundaries": honest_boundary_score_variance,
        "shared_main_first_stage": 0.0,
    }
    honest_score_variance = float(sum(honest_components.values()))
    honest_threshold_variance = honest_score_variance / curvature_squared

    return {
        "description": (
            "Population asymptotic variance for the exact Gaussian hard-trim "
            "simulation estimator; Gaussian T nuisance, not manuscript density sieve"
        ),
        "truth": truth,
        "nuisance_support": list(NUISANCE_SUPPORT),
        "curvature_squared": curvature_squared,
        "oracle_evaluation_only": {
            "score_variance": utility_score_variance,
            "threshold_asymptotic_variance": (
                utility_score_variance / curvature_squared
            ),
        },
        "alpha_block": alpha,
        "gaussian_density_and_boundary": {
            **quantities,
            "shared_sample": shared_T,
            "empirical_quantile_variance": empirical_quantile_variance,
            "generated_quantile_variance": generated_quantile_variance,
        },
        "full_sample": {
            "score_variance": full_score_variance,
            "threshold_asymptotic_variance": full_threshold_variance,
            "asymptotic_sd_times_sqrt_n": float(np.sqrt(full_threshold_variance)),
        },
        "crossfit_5fold": {
            "threshold_asymptotic_variance": full_threshold_variance,
            "note": (
                "Ordinary fixed-fold cross-fitting is first-order equivalent "
                "to the full-sample estimator in this correctly specified DGP."
            ),
        },
        "honest_split": {
            "fold_fractions": HONEST_FOLD_FRACTIONS,
            "score_variance_components": honest_components,
            "score_variance": honest_score_variance,
            "threshold_asymptotic_variance": honest_threshold_variance,
            "asymptotic_sd_times_sqrt_n": float(np.sqrt(honest_threshold_variance)),
        },
    }


def attach_monte_carlo_comparison(
    payload: Dict[str, Any], summary_path: Path,
) -> None:
    if not summary_path.exists():
        payload["monte_carlo_comparison"] = {
            "available": False,
            "path": str(summary_path),
        }
        return
    summary = json.loads(summary_path.read_text())["summary"]
    replications_path = summary_path.with_name("replications.csv")
    replication_rows = (
        list(csv.DictReader(replications_path.open()))
        if replications_path.exists()
        else []
    )
    estimators = ("honest_split", "crossfit_5fold", "full_ridge_0")
    comparison: Dict[str, Any] = {}
    total_replications = 0
    for estimator in estimators:
        constants = []
        rows = []
        for n_text, block in sorted(summary.items(), key=lambda item: int(item[0])):
            n = int(n_text)
            result = block["estimators"][estimator]
            constant = n * result["rmse"] ** 2
            constants.append(constant)
            row: Dict[str, Any] = {
                "n": n,
                "replications": block["replications"],
                "n_times_mse": constant,
                "n_times_variance": n * result["sd"] ** 2,
                "sqrt_n_bias": np.sqrt(n) * result["bias"],
            }
            matching_replications = [
                item for item in replication_rows if int(item["n"]) == n
            ]
            if matching_replications:
                benchmark_name = (
                    "honest_split" if estimator == "honest_split" else "full_sample"
                )
                benchmark = payload[benchmark_name]["threshold_asymptotic_variance"]
                target = payload["truth"]["hard_phi_star"]
                standardized = np.asarray([
                    (float(item[f"{estimator}_phi"]) - target)
                    * np.sqrt(n / benchmark)
                    for item in matching_replications
                ])
                row["population_variance_standardization"] = {
                    "mean": float(np.mean(standardized)),
                    "sd": float(np.std(standardized, ddof=1)),
                    "coverage_95": float(np.mean(np.abs(standardized) <= 1.96)),
                    "q025": float(np.quantile(standardized, 0.025)),
                    "median": float(np.median(standardized)),
                    "q975": float(np.quantile(standardized, 0.975)),
                }
            rows.append(row)
            if estimator == estimators[0]:
                total_replications += int(block["replications"])
        benchmark_name = "honest_split" if estimator == "honest_split" else "full_sample"
        benchmark = payload[benchmark_name]["threshold_asymptotic_variance"]
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
        "total_replications_per_estimator": total_replications,
        "estimators": comparison,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--monte-carlo-summary", type=Path, default=DEFAULT_LARGE_N_SUMMARY
    )
    args = parser.parse_args(argv)
    payload = calculate_variance_benchmarks()
    attach_monte_carlo_comparison(payload, args.monte_carlo_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
