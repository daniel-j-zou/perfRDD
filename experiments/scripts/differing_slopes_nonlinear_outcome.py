"""Short nonlinear alpha/b stress test for the differing-slopes estimator.

The DGP replaces the linear alpha(eta) and b(eta) functions by quadratic
functions while retaining the full D * X block. The experiment compares the
current linear outcome fit with a correctly specified quadratic fit and a
cubic B-spline outcome fit. The index and hard-trim endpoints are estimated by
OLS, and the threshold maximizes the differing-slopes utility U_J with the
running-variable nuisances g and p_X estimated by Lebesgue-Gram spline
projection (``experiments.methods.weighted_tails``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import roots_hermitenorm
from scipy.stats import norm

from experiments.methods.spline_density import _basis_info, evaluate_basis_zero_outside
from experiments.methods.weighted_tails import fit_weighted_tails, uj_utility


POLICY_BOUNDS = (-3.0, 3.0)
SUPPORT = (-3.0, 3.0)
DENSITY_SUPPORT = (-3.0, 3.0)
TRIM_EPS = 0.10


@dataclass(frozen=True)
class DGP:
    a0: float = 0.35
    a1: float = 0.90
    a2: float = 0.0
    b0: float = 0.20
    b1: float = 0.60
    b2: float = 0.0
    beta1: tuple[float, float] = (0.30, -0.20)
    beta2: tuple[float, float] = (0.80, 0.25)
    gamma: tuple[float, float] = (1.0, 0.0)
    cost: float = 0.25
    sigma_eps: float = 0.50

    @property
    def gamma_array(self) -> np.ndarray:
        return np.asarray(self.gamma, dtype=float)

    @property
    def trim_bounds(self) -> tuple[float, float]:
        q = float(norm.ppf(1.0 - TRIM_EPS))
        return -q, q


SCENARIOS = {
    "linear": DGP(),
    "nonlinear": DGP(a2=0.35, b2=0.50),
}
VARIANTS = ("linear_ols", "quadratic_ols", "spline_ols")


def alpha(eta: np.ndarray, dgp: DGP) -> np.ndarray:
    return dgp.a0 + dgp.a1 * eta + dgp.a2 * (eta**2 - 1.0)


def baseline(eta: np.ndarray, dgp: DGP) -> np.ndarray:
    return dgp.b0 + dgp.b1 * eta + dgp.b2 * (eta**2 - 1.0)


def population_truth(dgp: DGP) -> float:
    nodes, weights = roots_hermitenorm(320)
    eta = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float) / np.sqrt(2.0 * np.pi)
    lo, hi = dgp.trim_bounds
    keep = (eta >= lo) & (eta <= hi)
    gamma = dgp.gamma_array
    sigma_t = float(np.linalg.norm(gamma))
    beta2 = np.asarray(dgp.beta2)

    def utility(phi: float) -> float:
        z = (float(phi) - eta) / sigma_t
        value = (alpha(eta, dgp) - dgp.cost) * norm.sf(z)
        value = value + float(beta2 @ gamma) / sigma_t * norm.pdf(z)
        return float(np.sum(weights * keep * value))

    result = minimize_scalar(
        lambda value: -utility(value),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def generate_sample(n: int, seed: int, dgp: DGP) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((int(n), 2))
    eta = rng.standard_normal(int(n))
    t = x @ dgp.gamma_array
    q = t + eta
    d = (q > 0.0).astype(float)
    effect = alpha(eta, dgp) + x @ np.asarray(dgp.beta2)
    y = baseline(eta, dgp) + x @ np.asarray(dgp.beta1) + d * effect
    y = y + rng.normal(0.0, dgp.sigma_eps, int(n))
    return {"X": x, "eta": eta, "T": t, "Q": q, "D": d, "Y": y}


def _fit_index(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = sample["X"]
    design = np.column_stack((np.ones(len(x)), x))
    gamma_hat, *_ = np.linalg.lstsq(design, sample["Q"], rcond=None)
    t_hat = design @ gamma_hat
    return gamma_hat, sample["Q"] - t_hat


def _linear_design(x: np.ndarray, eta: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.column_stack((
        np.ones(len(eta)), eta, x[:, 0], x[:, 1],
        d, d * eta, d * x[:, 0], d * x[:, 1],
    ))


def _quadratic_design(x: np.ndarray, eta: np.ndarray, d: np.ndarray) -> np.ndarray:
    quad = eta**2 - 1.0
    return np.column_stack((
        np.ones(len(eta)), eta, quad, x[:, 0], x[:, 1],
        d, d * eta, d * quad, d * x[:, 0], d * x[:, 1],
    ))


def _spline_design(
    x: np.ndarray, eta: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    info = _basis_info(10, SUPPORT)
    basis = evaluate_basis_zero_outside(eta, info)
    design = np.column_stack((basis, x, d[:, None] * basis, d[:, None] * x))
    return design, info


def estimate(sample: dict[str, np.ndarray], dgp: DGP, variant: str) -> float:
    gamma_hat, eta_hat = _fit_index(sample)
    t_hat = sample["X"] @ gamma_hat[1:] + gamma_hat[0]
    lower = -float(np.quantile(t_hat, 1.0 - TRIM_EPS))
    upper = -float(np.quantile(t_hat, TRIM_EPS))
    keep = (eta_hat >= lower) & (eta_hat <= upper)
    if int(keep.sum()) < 40:
        raise ValueError("too few observations survive estimated trimming")

    x = sample["X"]
    d = sample["D"]
    if variant == "linear_ols":
        design = _linear_design(x, eta_hat, d)
        coef, *_ = np.linalg.lstsq(design, sample["Y"], rcond=None)

        def effect_hat(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return coef[4] + coef[5] * values, coef[6:8]
    elif variant == "quadratic_ols":
        design = _quadratic_design(x, eta_hat, d)
        coef, *_ = np.linalg.lstsq(design, sample["Y"], rcond=None)

        def effect_hat(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (
                coef[5] + coef[6] * values + coef[7] * (values**2 - 1.0),
                coef[8:10],
            )
    elif variant == "spline_ols":
        design, info = _spline_design(x, eta_hat, d)
        coef, *_ = np.linalg.lstsq(design, sample["Y"], rcond=None)
        # The design is [B, X, D*B, D*X], hence the two spline blocks
        # together occupy ``design.shape[1] - 4`` columns.
        n_basis = (design.shape[1] - 4) // 2
        treatment_start = n_basis + 2
        alpha_coef = coef[treatment_start:treatment_start + n_basis]
        beta2_coef = coef[treatment_start + n_basis:treatment_start + n_basis + 2]

        def effect_hat(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            basis = evaluate_basis_zero_outside(values, info)
            return basis @ alpha_coef, beta2_coef
    else:
        raise ValueError(f"unknown variant {variant!r}")

    eta_eval = eta_hat[keep]
    alpha_eval, beta2_hat = effect_hat(eta_eval)
    # U_J with g_hat and p_X_hat of the estimated index.
    tails = fit_weighted_tails(t_hat, x, DENSITY_SUPPORT)
    weights = np.ones(len(eta_eval))

    def objective(phi: float) -> float:
        return uj_utility(
            phi, eta_eval, weights, alpha_eval - dgp.cost, np.asarray(beta2_hat), tails
        )

    result = minimize_scalar(
        lambda value: -objective(value),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-7},
    )
    return float(result.x)


def run(
    scenarios: Iterable[str],
    n_values: Iterable[int],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    scenarios = tuple(scenarios)
    n_values = tuple(int(n) for n in n_values)
    rows: list[dict[str, Any]] = []
    truths = {name: population_truth(SCENARIOS[name]) for name in scenarios}
    for scenario_index, name in enumerate(scenarios):
        dgp = SCENARIOS[name]
        for n in n_values:
            for rep in range(int(reps)):
                sample = generate_sample(
                    int(n),
                    int(seed + 1_000_003 * scenario_index + 10_007 * int(n) + rep),
                    dgp,
                )
                for variant in VARIANTS:
                    rows.append({
                        "scenario": name,
                        "n": int(n),
                        "rep": int(rep),
                        "variant": variant,
                        "phi_hat": estimate(sample, dgp, variant),
                    })
    summary: dict[str, Any] = {}
    for name in scenarios:
        summary[name] = {}
        target = truths[name]
        for n in n_values:
            summary[name][str(n)] = {}
            for variant in VARIANTS:
                values = np.asarray([
                    row["phi_hat"] for row in rows
                    if row["scenario"] == name
                    and row["n"] == n
                    and row["variant"] == variant
                ])
                summary[name][str(n)][variant] = {
                    "target_phi": target,
                    "mean_phi": float(np.mean(values)),
                    "bias": float(np.mean(values - target)),
                    "rmse": float(np.sqrt(np.mean((values - target) ** 2))),
                    "n_scaled_variance": float(n * np.var(values, ddof=1)),
                }
    return {
        "description": "Short nonlinear alpha/b differing-slopes outcome-flexibility diagnostic.",
        "scenarios": list(scenarios),
        "n_values": list(n_values),
        "reps": int(reps),
        "seed": int(seed),
        "truths": truths,
        "summary": summary,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        nargs="+",
        default=["linear", "nonlinear"],
    )
    parser.add_argument("--n", type=int, nargs="+", default=[800, 1600, 3200])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.scenario, args.n, args.reps, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
