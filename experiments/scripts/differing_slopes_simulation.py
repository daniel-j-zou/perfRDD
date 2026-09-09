"""Known-target simulation for the differing-slopes extension.

This experiment isolates the extra ``D * X`` block proposed for applications in
which the treatment effect depends on the level component of the running score.
The data-generating process is

    Q = gamma'X + eta,
    Y = b0 + b1 eta + X'beta1
        + D * (a0 + a1 eta + X'beta2) + eps,

where ``D = 1{Q > 0}``, ``X`` and ``eta`` are independent, and ``eps`` is
independent noise.  The target policy utility integrates over the known normal
distribution of ``T = gamma'X``.  This keeps the policy criterion smooth and
lets us calculate a delta-method sandwich variance for the threshold estimator.

The script compares a misspecified alpha-only fit with the correctly specified
fit containing ``D * X``.  The variance check is intentionally conditional on
the true first-stage residual and the true trimming interval: it validates the
new interaction block and the plug-in variance calculation, not the full
generated-index/moving-boundary theorem.  A later experiment can add those
terms once the extension's theory is settled.

Example::

    python -m experiments.scripts.differing_slopes_simulation \
        --n 800 1600 3200 --reps 200 \
        --out experiments/runs/differing_slopes_short.json

Robustness scenarios are available with ``--scenario``: ``baseline``,
``null_interaction``, ``strong_interaction``, ``t5_errors``, ``skewed_errors``,
and ``heteroskedastic_errors``.
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
from scipy.special import roots_hermitenorm
from scipy.stats import norm


@dataclass(frozen=True)
class DGP:
    """Parameters for the level-dependent treatment-effect DGP."""

    gamma: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    beta1: np.ndarray = field(default_factory=lambda: np.array([0.30, -0.20]))
    beta2: np.ndarray = field(default_factory=lambda: np.array([0.80, 0.25]))
    a0: float = 0.35
    a1: float = 0.90
    b0: float = 0.20
    b1: float = 0.60
    cost: float = 0.25
    sigma_eps: float = 0.50
    error_law: str = "normal"
    trim_eps: float = 0.10

    @property
    def sigma_T(self) -> float:
        return float(np.linalg.norm(self.gamma))

    @property
    def trim_bounds(self) -> tuple[float, float]:
        q = self.sigma_T * float(norm.ppf(1.0 - self.trim_eps))
        return -q, q


DEFAULT_DGP = DGP()
POLICY_BOUNDS = (-3.0, 3.0)
MODEL_LABELS = ("alpha_only", "differing_slopes")


SCENARIOS = {
    "baseline": DEFAULT_DGP,
    # With no D*X effect, the restricted and full specifications target the
    # same policy.  This is a useful overfitting/variance sanity check.
    "null_interaction": replace(DEFAULT_DGP, beta2=np.zeros(2)),
    # A larger effect heterogeneity makes the omitted-interaction bias easier to
    # detect while retaining an interior optimum.
    "strong_interaction": replace(
        DEFAULT_DGP, beta2=np.array([1.60, 0.50]),
    ),
    # These error laws leave the conditional mean unchanged, so the population
    # target is unchanged while the robust variance calculation is stressed.
    "t5_errors": replace(DEFAULT_DGP, error_law="t5"),
    "skewed_errors": replace(DEFAULT_DGP, error_law="skewed"),
    "heteroskedastic_errors": replace(DEFAULT_DGP, error_law="heteroskedastic"),
}


def _quadrature(dgp: DGP, n_nodes: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes/weights for expectations under eta ~ N(0, 1)."""
    nodes, weights = roots_hermitenorm(n_nodes)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float) / np.sqrt(2.0 * np.pi)


def _utility_integrand(
    phi: float, eta: np.ndarray, dgp: DGP, *, include_beta2: bool,
) -> np.ndarray:
    """Known conditional utility contribution at eta values."""
    z = (float(phi) - eta) / dgp.sigma_T
    alpha = dgp.a0 + dgp.a1 * eta - dgp.cost
    out = alpha * norm.sf(z)
    if include_beta2:
        # E[(beta2'X) 1{gamma'X > phi-eta} | eta]
        # = (beta2'gamma / sigma_T) * standard-normal-density(z).
        out = out + (float(dgp.beta2 @ dgp.gamma) / dgp.sigma_T) * norm.pdf(z)
    return out


def population_utility(phi: float, dgp: DGP = DEFAULT_DGP, *, include_beta2: bool = True) -> float:
    """Evaluate the hard-trimmed population utility by Gaussian quadrature."""
    eta, weights = _quadrature(dgp)
    lo, hi = dgp.trim_bounds
    keep = (eta >= lo) & (eta <= hi)
    return float(np.sum(weights * keep * _utility_integrand(
        phi, eta, dgp, include_beta2=include_beta2,
    )))


def population_truth(dgp: DGP = DEFAULT_DGP, *, include_beta2: bool = True) -> dict[str, float]:
    """Find the unique maximizer and curvature of the known utility."""
    result = minimize_scalar(
        lambda value: -population_utility(value, dgp, include_beta2=include_beta2),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 300},
    )
    phi = float(result.x)
    h = 2e-4
    curvature = (
        population_utility(phi + h, dgp, include_beta2=include_beta2)
        - 2.0 * population_utility(phi, dgp, include_beta2=include_beta2)
        + population_utility(phi - h, dgp, include_beta2=include_beta2)
    ) / h**2
    return {
        "phi_star": phi,
        "utility": float(-result.fun),
        "curvature": float(curvature),
    }


def generate_sample(n: int, seed: int, dgp: DGP = DEFAULT_DGP) -> dict[str, np.ndarray]:
    """Generate one iid sample with the true residual eta observed by the fit."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((int(n), len(dgp.gamma)))
    eta = rng.standard_normal(int(n))
    q = x @ dgp.gamma + eta
    d = (q > 0.0).astype(float)
    alpha = dgp.a0 + dgp.a1 * eta
    baseline = dgp.b0 + dgp.b1 * eta + x @ dgp.beta1
    effect = alpha + x @ dgp.beta2
    if dgp.error_law == "normal":
        error = rng.normal(0.0, dgp.sigma_eps, int(n))
    elif dgp.error_law == "t5":
        # Standardize t_5 to unit variance before applying sigma_eps.
        error = dgp.sigma_eps * rng.standard_t(5, int(n)) / np.sqrt(5.0 / 3.0)
    elif dgp.error_law == "skewed":
        # Centered exponential errors are skewed but have unit variance.
        error = dgp.sigma_eps * (rng.exponential(1.0, int(n)) - 1.0)
    elif dgp.error_law == "heteroskedastic":
        scale = dgp.sigma_eps * (0.50 + 0.75 * np.abs(x[:, 0]))
        error = rng.normal(0.0, scale, int(n))
    else:
        raise ValueError(f"unknown error law: {dgp.error_law!r}")
    y = baseline + d * effect + error
    return {"X": x, "eta": eta, "Q": q, "D": d, "Y": y}


def _design(sample: dict[str, np.ndarray], *, include_beta2: bool) -> np.ndarray:
    """Build the correctly ordered outcome-regression design matrix."""
    x = sample["X"]
    eta = sample["eta"]
    d = sample["D"]
    columns = [np.ones(len(eta)), eta, x[:, 0], x[:, 1], d, d * eta]
    if include_beta2:
        columns.extend([d * x[:, 0], d * x[:, 1]])
    return np.column_stack(columns)


def _fit_outcome(sample: dict[str, np.ndarray], *, include_beta2: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the pooled or differing-slopes linear outcome model."""
    design = _design(sample, include_beta2=include_beta2)
    coef, *_ = np.linalg.lstsq(design, sample["Y"], rcond=None)
    residual = sample["Y"] - design @ coef
    return coef, residual, design


def _policy_terms(
    phi: float,
    eta: np.ndarray,
    coef: np.ndarray,
    dgp: DGP,
    *,
    include_beta2: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return utility score, curvature contribution, and theta score gradient."""
    lo, hi = dgp.trim_bounds
    keep = ((eta >= lo) & (eta <= hi)).astype(float)
    z = (float(phi) - eta) / dgp.sigma_T
    density = norm.pdf(z)
    alpha = coef[4] + coef[5] * eta - dgp.cost
    beta2_dot_gamma = 0.0
    if include_beta2:
        beta2_dot_gamma = float(coef[6:8] @ dgp.gamma)
    kappa = beta2_dot_gamma / dgp.sigma_T
    score = keep * (-(alpha * density / dgp.sigma_T) - kappa * z * density / dgp.sigma_T)
    curvature = keep * (
        alpha * z * density / dgp.sigma_T**2
        + kappa * (z**2 - 1.0) * density / dgp.sigma_T**2
    )
    gradient = np.zeros((len(eta), len(coef)))
    gradient[:, 4] = keep * (-density / dgp.sigma_T)
    gradient[:, 5] = keep * (-eta * density / dgp.sigma_T)
    if include_beta2:
        gradient[:, 6:8] = keep[:, None] * (
            -z[:, None] * density[:, None] * dgp.gamma[None, :] / dgp.sigma_T**2
        )
    return score, curvature, gradient


def _sample_utility(
    phi: float,
    eta: np.ndarray,
    coef: np.ndarray,
    dgp: DGP,
    *,
    include_beta2: bool,
) -> float:
    """Evaluate the smooth plug-in policy utility on the empirical eta distribution."""
    lo, hi = dgp.trim_bounds
    keep = (eta >= lo) & (eta <= hi)
    z = (float(phi) - eta) / dgp.sigma_T
    alpha = coef[4] + coef[5] * eta - dgp.cost
    value = alpha * norm.sf(z)
    if include_beta2:
        value = value + (float(coef[6:8] @ dgp.gamma) / dgp.sigma_T) * norm.pdf(z)
    return float(np.mean(np.where(keep, value, 0.0)))


def estimate_threshold(
    sample: dict[str, np.ndarray],
    dgp: DGP = DEFAULT_DGP,
    *,
    include_beta2: bool,
) -> dict[str, float]:
    """Estimate the threshold and a plug-in sandwich variance constant."""
    coef, residual, design = _fit_outcome(sample, include_beta2=include_beta2)
    result = minimize_scalar(
        lambda value: -_sample_utility(
            value, sample["eta"], coef, dgp, include_beta2=include_beta2,
        ),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    phi = float(result.x)
    score, curvature_terms, gradient = _policy_terms(
        phi, sample["eta"], coef, dgp, include_beta2=include_beta2,
    )
    bread = np.linalg.inv((design.T @ design) / len(design))
    # OLS influence: M^{-1} Z_i residual_i.  The score also has the direct
    # empirical-eta term; centering removes the numerical optimizer tolerance.
    # The regression contribution is the *population-average* policy gradient
    # multiplying each observation's OLS influence.  Using gradient_i times
    # influence_i would incorrectly estimate E[g_i Z_i e_i] rather than the
    # delta-method term E[g_i] E[Z_i e_i].
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


def run_simulation(
    n_values: Iterable[int],
    reps: int,
    seed: int = 20260908,
    dgp: DGP = DEFAULT_DGP,
) -> dict[str, Any]:
    """Run the Monte Carlo comparison and return rows plus cell summaries."""
    n_values = tuple(int(value) for value in n_values)
    rows: list[dict[str, Any]] = []
    truth = {
        "differing_slopes": population_truth(dgp, include_beta2=True),
        "alpha_only": population_truth(dgp, include_beta2=False),
    }
    for n_index, n in enumerate(n_values):
        for rep in range(int(reps)):
            sample_seed = int(seed + 1_000_003 * n_index + rep)
            sample = generate_sample(int(n), sample_seed, dgp)
            for label, include_beta2 in (("alpha_only", False), ("differing_slopes", True)):
                estimate = estimate_threshold(sample, dgp, include_beta2=include_beta2)
                rows.append({
                    "n": int(n),
                    "rep": int(rep),
                    "model": label,
                    "target_phi": truth["differing_slopes"]["phi_star"],
                    "model_target_phi": truth[label]["phi_star"],
                    **estimate,
                })
    summaries: dict[str, Any] = {}
    for n in sorted({int(row["n"]) for row in rows}):
        summaries[str(n)] = {}
        for label in MODEL_LABELS:
            subset = [row for row in rows if int(row["n"]) == n and row["model"] == label]
            target = float(subset[0]["target_phi"])
            model_target = float(subset[0]["model_target_phi"])
            errors = np.asarray([float(row["phi_hat"]) - target for row in subset])
            model_errors = np.asarray([float(row["phi_hat"]) - model_target for row in subset])
            variance_hats = np.asarray([float(row["variance_hat"]) for row in subset])
            intervals = [
                (float(row["phi_hat"]) - 1.96 * np.sqrt(max(float(row["variance_hat"]), 0.0)),
                 float(row["phi_hat"]) + 1.96 * np.sqrt(max(float(row["variance_hat"]), 0.0)))
                for row in subset
            ]
            summaries[str(n)][label] = {
                "replications": len(subset),
                "target_phi": target,
                "model_target_phi": model_target,
                "mean_phi": float(np.mean([float(row["phi_hat"]) for row in subset])),
                "bias_to_full_target": float(np.mean(errors)),
                "rmse_to_full_target": float(np.sqrt(np.mean(errors**2))),
                "bias_to_model_target": float(np.mean(model_errors)),
                "rmse_to_model_target": float(np.sqrt(np.mean(model_errors**2))),
                "mc_variance": float(np.var([float(row["phi_hat"]) for row in subset], ddof=1)),
                "mean_variance_hat": float(np.mean(variance_hats)),
                "variance_ratio_hat_to_mc": float(np.mean(variance_hats) / max(float(np.var([float(row["phi_hat"]) for row in subset], ddof=1)), 1e-16)),
                "normal_coverage": float(np.mean([
                    low <= target <= high for low, high in intervals
                ])),
                "mean_condition_number": float(np.mean([float(row["condition_number"]) for row in subset])),
            }
    return {
        "description": "Known-eta smooth-utility simulation for differing treated/control slopes; variance is conditional on true first-stage residual and fixed trim bounds.",
        "dgp": {
            "gamma": dgp.gamma.tolist(),
            "beta1": dgp.beta1.tolist(),
            "beta2": dgp.beta2.tolist(),
            "a0": dgp.a0,
            "a1": dgp.a1,
            "b0": dgp.b0,
            "b1": dgp.b1,
            "cost": dgp.cost,
            "sigma_eps": dgp.sigma_eps,
            "error_law": dgp.error_law,
            "trim_eps": dgp.trim_eps,
            "trim_bounds": list(dgp.trim_bounds),
        },
        "truth": truth,
        "n_values": [int(value) for value in n_values],
        "reps": int(reps),
        "seed": int(seed),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(result: dict[str, Any], out: Path) -> None:
    """Write JSON plus a compact CSV for downstream plotting/audits."""
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
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="baseline")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_simulation(args.n, args.reps, args.seed, dgp=SCENARIOS[args.scenario])
    write_outputs(result, args.out)
    print(json.dumps(result["summary"], indent=2))
    print(f"[wrote] {args.out}")
    print(f"[wrote] {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
