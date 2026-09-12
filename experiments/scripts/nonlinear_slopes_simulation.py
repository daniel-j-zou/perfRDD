"""Known-target simulations for nonlinear treatment-effect heterogeneity.

This experiment asks whether the linear differing-slopes extension remains
valid when the treatment effect is nonlinear in the threshold score.  It uses
the same controlled index setup as ``differing_slopes_simulation.py`` but adds
the centered quadratic term ``T**2 - Var(T)`` to the treatment effect,
where ``T = gamma'X``.  Three outcome models are compared:

``alpha_only``
    The original reduction, which omits all dependence on ``X``.
``linear_slopes``
    The existing extension, with ``D*T`` and ``D*X2`` terms.
``quadratic_slopes``
    The correctly specified extension, which additionally includes
    ``D*(T**2 - Var(T))``.

The first-stage residual ``eta``, the normal law of ``T``, and the hard trim
interval are treated as known.  This isolates the nonlinear outcome-model
question; it is not a validation of generated-index or moving-boundary terms.

Example::

    python -m experiments.scripts.nonlinear_slopes_simulation \
        --scenario quadratic --n 500 1000 2000 4000 --reps 300 \
        --seed 20260914 --out experiments/runs/nonlinear_quadratic.json
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


@dataclass(frozen=True)
class DGP:
    """Parameters for the nonlinear treatment-effect DGP."""

    gamma: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    beta1: np.ndarray = field(default_factory=lambda: np.array([0.30, -0.20]))
    # Moderate curvature keeps the misspecified objectives interior while
    # making the asymptotic discrepancy easy to see.
    delta_t2: float = 0.40
    a0: float = 0.35
    a1: float = 0.90
    b0: float = 0.20
    b1: float = 0.60
    cost: float = 0.25
    sigma_eps: float = 0.50
    trim_eps: float = 0.10

    @property
    def sigma_T(self) -> float:
        return float(np.linalg.norm(self.gamma))

    @property
    def trim_bounds(self) -> tuple[float, float]:
        q = self.sigma_T * float(norm.ppf(1.0 - self.trim_eps))
        return -q, q


DEFAULT_DGP = DGP()
SCENARIOS = {
    "quadratic": DEFAULT_DGP,
    "null_quadratic": replace(DEFAULT_DGP, delta_t2=0.0),
    "strong_quadratic": replace(DEFAULT_DGP, delta_t2=0.80),
}
MODEL_LABELS = ("alpha_only", "linear_slopes", "quadratic_slopes")
POLICY_BOUNDS = (-3.0, 3.0)


def _quadrature(dgp: DGP, n_nodes: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss--Legendre nodes/weights on the hard-trim interval.

    Integrating directly over the retained interval avoids the boundary error
    that arises when a Gauss--Hermite rule is multiplied by a discontinuous
    trim indicator.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    lo, hi = dgp.trim_bounds
    eta = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    weights = 0.5 * (hi - lo) * weights * norm.pdf(eta)
    return np.asarray(eta, dtype=float), np.asarray(weights, dtype=float)


def _keep(eta: np.ndarray, dgp: DGP) -> np.ndarray:
    lo, hi = dgp.trim_bounds
    return ((eta >= lo) & (eta <= hi)).astype(float)


def _conditional_moments(phi: float, eta: np.ndarray, dgp: DGP) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return P(T > phi-eta), E[T 1{...}], E[(T^2-sigma^2)1{...}], and z."""
    sigma = dgp.sigma_T
    z = (float(phi) - eta) / sigma
    density = norm.pdf(z)
    probability = norm.sf(z)
    first = sigma * density
    centered_second = sigma**2 * z * density
    return probability, first, centered_second, z


def population_utility(phi: float, dgp: DGP = DEFAULT_DGP, *, include_quadratic: bool = True) -> float:
    """Evaluate the hard-trimmed population utility by Gaussian quadrature."""
    eta, weights = _quadrature(dgp)
    probability, first, centered_second, _ = _conditional_moments(phi, eta, dgp)
    alpha = dgp.a0 + dgp.a1 * eta - dgp.cost
    value = alpha * probability
    if include_quadratic:
        value = value + dgp.delta_t2 * centered_second
    return float(np.sum(weights * value))


def population_truth(dgp: DGP = DEFAULT_DGP, *, include_quadratic: bool = True) -> dict[str, float]:
    """Find the population maximizer and its curvature."""
    result = minimize_scalar(
        lambda value: -population_utility(value, dgp, include_quadratic=include_quadratic),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 300},
    )
    phi = float(result.x)
    h = 2e-4
    curvature = (
        population_utility(phi + h, dgp, include_quadratic=include_quadratic)
        - 2.0 * population_utility(phi, dgp, include_quadratic=include_quadratic)
        + population_utility(phi - h, dgp, include_quadratic=include_quadratic)
    ) / h**2
    return {"phi_star": phi, "utility": float(-result.fun), "curvature": float(curvature)}


def generate_sample(n: int, seed: int, dgp: DGP = DEFAULT_DGP) -> dict[str, np.ndarray]:
    """Generate one iid sample; eta is observed by construction."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((int(n), len(dgp.gamma)))
    eta = rng.standard_normal(int(n))
    t = x @ dgp.gamma
    q = t + eta
    d = (q > 0.0).astype(float)
    alpha = dgp.a0 + dgp.a1 * eta
    nonlinear = dgp.delta_t2 * (t**2 - dgp.sigma_T**2)
    baseline = dgp.b0 + dgp.b1 * eta + x @ dgp.beta1
    error = rng.normal(0.0, dgp.sigma_eps, int(n))
    y = baseline + d * (alpha + nonlinear) + error
    return {"X": x, "T": t, "eta": eta, "Q": q, "D": d, "Y": y}


def _basis(model: str) -> tuple[str, ...]:
    if model == "alpha_only":
        return ()
    if model == "linear_slopes":
        return ("t", "x2")
    if model == "quadratic_slopes":
        return ("t", "x2", "q2")
    raise ValueError(f"unknown model: {model!r}")


def _design(sample: dict[str, np.ndarray], model: str, dgp: DGP) -> np.ndarray:
    x, t, eta, d = sample["X"], sample["T"], sample["eta"], sample["D"]
    columns = [np.ones(len(eta)), eta, x[:, 0], x[:, 1], d, d * eta]
    for term in _basis(model):
        if term == "t":
            columns.append(d * t)
        elif term == "x2":
            columns.append(d * x[:, 1])
        elif term == "q2":
            columns.append(d * (t**2 - dgp.sigma_T**2))
    return np.column_stack(columns)


def _fit_outcome(sample: dict[str, np.ndarray], model: str, dgp: DGP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = _design(sample, model, dgp)
    coef, *_ = np.linalg.lstsq(design, sample["Y"], rcond=None)
    residual = sample["Y"] - design @ coef
    return coef, residual, design


def _term_coefficients(coef: np.ndarray, model: str) -> tuple[float, float]:
    """Extract coefficients on D*T and D*(T^2-sigma_T^2)."""
    names = _basis(model)
    linear = coef[6 + names.index("t")] if "t" in names else 0.0
    quadratic = coef[6 + names.index("q2")] if "q2" in names else 0.0
    return float(linear), float(quadratic)


def _sample_utility(phi: float, sample: dict[str, np.ndarray], coef: np.ndarray, model: str, dgp: DGP) -> float:
    eta = sample["eta"]
    probability, first, centered_second, _ = _conditional_moments(phi, eta, dgp)
    alpha = coef[4] + coef[5] * eta - dgp.cost
    linear, quadratic = _term_coefficients(coef, model)
    value = alpha * probability + linear * first + quadratic * centered_second
    return float(np.mean(_keep(eta, dgp) * value))


def _policy_terms(phi: float, sample: dict[str, np.ndarray], coef: np.ndarray, model: str, dgp: DGP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return score, curvature, and coefficient-gradient arrays."""
    eta = sample["eta"]
    sigma = dgp.sigma_T
    probability, _, _, z = _conditional_moments(phi, eta, dgp)
    density = norm.pdf(z)
    alpha = coef[4] + coef[5] * eta - dgp.cost
    linear, quadratic = _term_coefficients(coef, model)
    keep = _keep(eta, dgp)

    score = keep * (
        -alpha * density / sigma
        -linear * z * density
        +quadratic * sigma * (1.0 - z**2) * density
    )
    curvature = keep * (
        alpha * z * density / sigma**2
        +linear * (z**2 - 1.0) * density / sigma
        +quadratic * (z**3 - 3.0 * z) * density
    )

    gradient = np.zeros((len(eta), len(coef)))
    # This is the derivative of the policy first-order condition with
    # respect to the outcome-regression coefficients (not the derivative of
    # utility itself).  It is the cross-derivative needed in the argmax
    # influence function.
    gradient[:, 4] = keep * (-density / sigma)
    gradient[:, 5] = keep * (-eta * density / sigma)
    names = _basis(model)
    if "t" in names:
        gradient[:, 6 + names.index("t")] = keep * (-z * density)
    if "q2" in names:
        gradient[:, 6 + names.index("q2")] = keep * (sigma * (1.0 - z**2) * density)
    # The conditional mean of X2 given T is zero, so D*X2 has zero direct
    # policy-gradient contribution in this design.
    return score, curvature, gradient


def estimate_threshold(sample: dict[str, np.ndarray], model: str, dgp: DGP = DEFAULT_DGP) -> dict[str, float]:
    """Estimate the policy threshold and conditional plug-in variance."""
    coef, residual, design = _fit_outcome(sample, model, dgp)
    result = minimize_scalar(
        lambda value: -_sample_utility(value, sample, coef, model, dgp),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    phi = float(result.x)
    score, curvature_terms, gradient = _policy_terms(phi, sample, coef, model, dgp)
    bread = np.linalg.inv((design.T @ design) / len(design))
    theta_influence = (design @ bread.T) * residual[:, None]
    gradient_mean = np.mean(gradient, axis=0)
    score_influence = (score - np.mean(score)) + theta_influence @ gradient_mean
    score_variance = float(np.var(score_influence, ddof=1))
    curvature = float(np.mean(curvature_terms))
    variance_constant = score_variance / max(curvature**2, 1e-16)
    return {
        "phi_hat": phi,
        "curvature_hat": curvature,
        "score_variance_hat": score_variance,
        "variance_constant_hat": variance_constant,
        "variance_hat": variance_constant / len(design),
        "condition_number": float(np.linalg.cond((design.T @ design) / len(design))),
    }


def run_simulation(n_values: Iterable[int], reps: int, seed: int = 20260914, dgp: DGP = DEFAULT_DGP) -> dict[str, Any]:
    """Run the nonlinear-model comparison and return rows plus summaries."""
    n_values = tuple(int(value) for value in n_values)
    truth = {
        "full": population_truth(dgp, include_quadratic=True),
        "restricted": population_truth(dgp, include_quadratic=False),
    }
    rows: list[dict[str, Any]] = []
    for n_index, n in enumerate(n_values):
        for rep in range(int(reps)):
            sample_seed = int(seed + 1_000_003 * n_index + rep)
            sample = generate_sample(n, sample_seed, dgp)
            for model in MODEL_LABELS:
                estimate = estimate_threshold(sample, model, dgp)
                rows.append({
                    "n": n,
                    "rep": rep,
                    "model": model,
                    "target_phi": truth["full"]["phi_star"],
                    "restricted_target_phi": truth["restricted"]["phi_star"],
                    **estimate,
                })

    summaries: dict[str, Any] = {}
    for n in sorted({int(row["n"]) for row in rows}):
        summaries[str(n)] = {}
        for model in MODEL_LABELS:
            subset = [row for row in rows if int(row["n"]) == n and row["model"] == model]
            target = float(subset[0]["target_phi"])
            errors = np.asarray([float(row["phi_hat"]) - target for row in subset])
            variance_hats = np.asarray([float(row["variance_hat"]) for row in subset])
            intervals = [
                (float(row["phi_hat"]) - 1.96 * np.sqrt(max(float(row["variance_hat"]), 0.0)),
                 float(row["phi_hat"]) + 1.96 * np.sqrt(max(float(row["variance_hat"]), 0.0)))
                for row in subset
            ]
            mc_variance = float(np.var([float(row["phi_hat"]) for row in subset], ddof=1))
            summaries[str(n)][model] = {
                "replications": len(subset),
                "target_phi": target,
                "restricted_target_phi": float(subset[0]["restricted_target_phi"]),
                "mean_phi": float(np.mean([float(row["phi_hat"]) for row in subset])),
                "bias_to_full_target": float(np.mean(errors)),
                "rmse_to_full_target": float(np.sqrt(np.mean(errors**2))),
                "mc_variance": mc_variance,
                "mean_variance_hat": float(np.mean(variance_hats)),
                "variance_ratio_hat_to_mc": float(np.mean(variance_hats) / max(mc_variance, 1e-16)),
                "normal_coverage": float(np.mean([low <= target <= high for low, high in intervals])),
                "mean_condition_number": float(np.mean([float(row["condition_number"]) for row in subset])),
            }
    return {
        "description": "Known-eta hard-trim simulation with quadratic treatment-effect heterogeneity in T=gamma'X.",
        "dgp": {
            "gamma": dgp.gamma.tolist(),
            "beta1": dgp.beta1.tolist(),
            "delta_t2": dgp.delta_t2,
            "a0": dgp.a0,
            "a1": dgp.a1,
            "b0": dgp.b0,
            "b1": dgp.b1,
            "cost": dgp.cost,
            "sigma_eps": dgp.sigma_eps,
            "trim_eps": dgp.trim_eps,
            "trim_bounds": list(dgp.trim_bounds),
        },
        "truth": truth,
        "n_values": list(n_values),
        "reps": int(reps),
        "seed": int(seed),
        "models": list(MODEL_LABELS),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(result: dict[str, Any], out: Path) -> None:
    """Write JSON and a compact CSV for auditing and plotting."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = out.with_suffix(".csv")
    rows = result["rows"]
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[800, 1600, 3200])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="quadratic")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_simulation(args.n, args.reps, args.seed, dgp=SCENARIOS[args.scenario])
    write_outputs(result, args.out)
    print(json.dumps(result["summary"], indent=2))
    print(f"[wrote] {args.out}")
    print(f"[wrote] {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
