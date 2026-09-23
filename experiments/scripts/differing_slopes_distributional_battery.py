"""Distributional robustness battery for the differing-slopes model.

The outcome model in every scenario contains the treatment--covariate block
``D * X``.  The scenarios differ only in which part of the data-generating
process is changed:

``x_t5``, ``x_skewed``, ``x_mixture``
    Non-Gaussian covariate/index laws.  Because ``T = gamma' X``, these change
    the running-variable law and therefore test the weighted-tail nuisance.
``eta_t5``, ``eta_skewed``
    Non-Gaussian latent residual laws.  These change the heterogeneity
    distribution and the population target while leaving the index law fixed.
``error_t5``, ``error_skewed``, ``error_heteroskedastic``
    Outcome-error robustness checks.  The conditional mean remains unchanged,
    so the population target is unchanged and only finite-sample variability
    should move.

This is a known-index, known-trim diagnostic: ``eta`` and the population trim
window are supplied to the estimator.  It tests the differing-slopes outcome
and weighted-tail algebra, not the generated-index/moving-boundary theorem.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import lognorm, norm, t as student_t


POLICY_BOUNDS = (-3.0, 3.0)
TRIM_EPS = 0.10
MIXTURE_WEIGHTS = np.array([0.5, 0.5])
MIXTURE_MEANS = np.array([-1.0, 1.0])
MIXTURE_SDS = np.array([0.45, 0.45])
MIXTURE_MEAN = float(MIXTURE_WEIGHTS @ MIXTURE_MEANS)
MIXTURE_VARIANCE = float(
    MIXTURE_WEIGHTS @ (MIXTURE_SDS**2 + (MIXTURE_MEANS - MIXTURE_MEAN) ** 2)
)
MIXTURE_SCALE = float(np.sqrt(MIXTURE_VARIANCE))
T5_SCALE = float(np.sqrt(5.0 / 3.0))


@dataclass(frozen=True)
class DGP:
    """Differing-slopes DGP with one law selector per component."""

    x_law: str = "normal"
    eta_law: str = "normal"
    error_law: str = "normal"
    beta1: tuple[float, float] = (0.30, -0.20)
    beta2: tuple[float, float] = (0.80, 0.25)
    a0: float = 0.35
    a1: float = 0.90
    b0: float = 0.20
    b1: float = 0.60
    cost: float = 0.25
    sigma_eps: float = 0.50


SCENARIOS: dict[str, DGP] = {
    "baseline": DGP(),
    "x_t5": DGP(x_law="t5"),
    "x_skewed": DGP(x_law="skewed"),
    "x_mixture": DGP(x_law="mixture"),
    "eta_t5": DGP(eta_law="t5"),
    "eta_skewed": DGP(eta_law="skewed"),
    "error_t5": DGP(error_law="t5"),
    "error_skewed": DGP(error_law="skewed"),
    "error_heteroskedastic": DGP(error_law="heteroskedastic"),
}


def _draw_law(rng: np.random.Generator, n: int, law: str) -> np.ndarray:
    if law == "normal":
        return rng.standard_normal(int(n))
    if law == "t5":
        return rng.standard_t(5, int(n)) / T5_SCALE
    if law == "skewed":
        return rng.exponential(1.0, int(n)) - 1.0
    if law == "mixture":
        component = rng.choice(len(MIXTURE_WEIGHTS), size=int(n), p=MIXTURE_WEIGHTS)
        return (
            MIXTURE_MEANS[component]
            + MIXTURE_SDS[component] * rng.standard_normal(int(n))
        ) / MIXTURE_SCALE
    raise ValueError(f"unknown law: {law!r}")


def _law_cdf(value: np.ndarray | float, law: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if law == "normal":
        return norm.cdf(value)
    if law == "t5":
        return student_t.cdf(T5_SCALE * value, 5)
    if law == "skewed":
        return np.where(value < -1.0, 0.0, 1.0 - np.exp(-(value + 1.0)))
    if law == "mixture":
        raw = MIXTURE_MEAN + MIXTURE_SCALE * value
        return sum(
            weight * norm.cdf((raw - mean) / sd)
            for weight, mean, sd in zip(MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_SDS)
        )
    raise ValueError(f"unknown law: {law!r}")


def _law_pdf(value: np.ndarray | float, law: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if law == "normal":
        return norm.pdf(value)
    if law == "t5":
        return T5_SCALE * student_t.pdf(T5_SCALE * value, 5)
    if law == "skewed":
        return np.where(value < -1.0, 0.0, np.exp(-(value + 1.0)))
    if law == "mixture":
        raw = MIXTURE_MEAN + MIXTURE_SCALE * value
        return sum(
            weight * norm.pdf((raw - mean) / sd) / sd
            for weight, mean, sd in zip(MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_SDS)
        ) * MIXTURE_SCALE
    raise ValueError(f"unknown law: {law!r}")


def _law_survival(value: np.ndarray | float, law: str) -> np.ndarray:
    return 1.0 - _law_cdf(value, law)


def _law_weighted_tail(value: np.ndarray | float, law: str) -> np.ndarray:
    """Return E[X 1{X > value}] for a standardized index law."""
    value = np.asarray(value, dtype=float)
    if law == "normal":
        return norm.pdf(value)
    if law == "t5":
        raw = T5_SCALE * value
        constant = (
            student_t.pdf(0.0, 5) * 5.0 / 4.0 / T5_SCALE
        )
        return constant * (1.0 + raw**2 / 5.0) ** (-2.0)
    if law == "skewed":
        a = value + 1.0
        return np.where(value < -1.0, 0.0, a * np.exp(-a))
    if law == "mixture":
        raw = MIXTURE_MEAN + MIXTURE_SCALE * value
        output = np.zeros_like(value, dtype=float)
        for weight, mean, sd in zip(MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_SDS):
            z = (raw - mean) / sd
            prob = norm.sf(z)
            first_raw = mean * prob + sd * norm.pdf(z)
            output += weight * (first_raw - MIXTURE_MEAN * prob) / MIXTURE_SCALE
        return output
    raise ValueError(f"unknown law: {law!r}")


def _law_quantile(probability: float, law: str) -> float:
    if law == "normal":
        return float(norm.ppf(probability))
    if law == "t5":
        return float(student_t.ppf(probability, 5) / T5_SCALE)
    if law == "skewed":
        return float(-1.0 - np.log1p(-probability))
    if law == "mixture":
        return float(brentq(lambda x: float(_law_cdf(x, law)) - probability, -10.0, 10.0))
    raise ValueError(f"unknown law: {law!r}")


def _eta_integral(function, dgp: DGP) -> float:
    lo = -float("inf")
    hi = float("inf")
    # The hard trim window is finite and is applied by the caller.  Splitting
    # at the skewed-law support endpoint helps QUADPACK with its kink.
    points: list[float] = []
    if dgp.eta_law == "skewed":
        points = [-1.0]
    return float(quad(
        lambda value: float(function(value)) * float(_law_pdf(value, dgp.eta_law)),
        lo, hi, points=points, epsabs=2e-9, limit=250,
    )[0])


def trim_bounds(dgp: DGP) -> tuple[float, float]:
    lower = -_law_quantile(1.0 - TRIM_EPS, dgp.x_law)
    upper = -_law_quantile(TRIM_EPS, dgp.x_law)
    return lower, upper


def population_utility(phi: float, dgp: DGP, *, include_beta2: bool = True) -> float:
    lower, upper = trim_bounds(dgp)
    beta2 = float(dgp.beta2[0]) if include_beta2 else 0.0

    def integrand(eta: float) -> float:
        if eta < lower or eta > upper:
            return 0.0
        cutoff = float(phi) - eta
        alpha = dgp.a0 + dgp.a1 * eta - dgp.cost
        return alpha * float(_law_survival(cutoff, dgp.x_law)) + beta2 * float(
            _law_weighted_tail(cutoff, dgp.x_law)
        )

    # Restrict integration to the trimmed eta support; this also avoids
    # numerical work in tails that are deterministically dropped.
    return float(quad(
        lambda value: integrand(value) * float(_law_pdf(value, dgp.eta_law)),
        lower, upper,
        points=[-1.0] if dgp.eta_law == "skewed" and lower < -1.0 < upper else None,
        epsabs=2e-9,
        limit=250,
    )[0])


def population_truth(dgp: DGP, *, include_beta2: bool = True) -> dict[str, float]:
    result = minimize_scalar(
        lambda value: -population_utility(value, dgp, include_beta2=include_beta2),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-9, "maxiter": 300},
    )
    phi = float(result.x)
    return {
        "phi_star": phi,
        "utility": float(-result.fun),
        "trim_lower": float(trim_bounds(dgp)[0]),
        "trim_upper": float(trim_bounds(dgp)[1]),
    }


def generate_sample(n: int, seed: int, dgp: DGP) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack((_draw_law(rng, n, dgp.x_law), rng.standard_normal(int(n))))
    eta = _draw_law(rng, n, dgp.eta_law)
    q = x[:, 0] + eta
    d = (q > 0.0).astype(float)
    beta1 = np.asarray(dgp.beta1)
    beta2 = np.asarray(dgp.beta2)
    alpha = dgp.a0 + dgp.a1 * eta
    baseline = dgp.b0 + dgp.b1 * eta + x @ beta1
    effect = alpha + x @ beta2
    if dgp.error_law == "normal":
        error = rng.normal(0.0, dgp.sigma_eps, int(n))
    elif dgp.error_law == "t5":
        error = dgp.sigma_eps * rng.standard_t(5, int(n)) / T5_SCALE
    elif dgp.error_law == "skewed":
        error = dgp.sigma_eps * (rng.exponential(1.0, int(n)) - 1.0)
    elif dgp.error_law == "heteroskedastic":
        scale = dgp.sigma_eps * (0.50 + 0.75 * np.abs(x[:, 0]))
        error = rng.normal(0.0, scale, int(n))
    else:
        raise ValueError(f"unknown error law: {dgp.error_law!r}")
    return {
        "X": x,
        "eta": eta,
        "Q": q,
        "D": d,
        "Y": baseline + d * effect + error,
    }


def _design(sample: dict[str, np.ndarray]) -> np.ndarray:
    x, eta, d = sample["X"], sample["eta"], sample["D"]
    return np.column_stack((
        np.ones(len(eta)), eta, x[:, 0], x[:, 1], d, d * eta,
        d * x[:, 0], d * x[:, 1],
    ))


def estimate_threshold(sample: dict[str, np.ndarray], dgp: DGP) -> dict[str, float]:
    design = _design(sample)
    coef, *_ = np.linalg.lstsq(design, sample["Y"], rcond=None)
    eta = sample["eta"]
    lower, upper = trim_bounds(dgp)
    keep = (eta >= lower) & (eta <= upper)
    beta2 = float(coef[6])

    def utility(phi: float) -> float:
        cutoff = float(phi) - eta
        value = (
            (coef[4] + coef[5] * eta - dgp.cost)
            * _law_survival(cutoff, dgp.x_law)
            + beta2 * _law_weighted_tail(cutoff, dgp.x_law)
        )
        return float(np.mean(np.where(keep, value, 0.0)))

    result = minimize_scalar(
        lambda value: -utility(value),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    phi = float(result.x)
    boundary = bool(
        phi <= POLICY_BOUNDS[0] + 2e-4 or phi >= POLICY_BOUNDS[1] - 2e-4
    )
    return {"phi_hat": phi, "boundary": boundary}


def run_simulation(
    n_values: Sequence[int], reps: int, seed: int, dgp: DGP,
) -> dict[str, Any]:
    target = population_truth(dgp, include_beta2=True)
    rows: list[dict[str, Any]] = []
    for n_index, n in enumerate(n_values):
        for rep in range(int(reps)):
            sample_seed = int(seed + 1_000_003 * n_index + rep)
            estimate = estimate_threshold(generate_sample(int(n), sample_seed, dgp), dgp)
            rows.append({"n": int(n), "rep": int(rep), **estimate})
    summary: dict[str, Any] = {}
    for n in n_values:
        subset = [row for row in rows if row["n"] == int(n)]
        values = np.asarray([row["phi_hat"] for row in subset], dtype=float)
        errors = values - float(target["phi_star"])
        summary[str(n)] = {
            "replications": len(subset),
            "target_phi": float(target["phi_star"]),
            "mean_phi": float(np.mean(values)),
            "bias": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "empirical_sd": float(np.std(values, ddof=1)),
            "sqrt_n_scaled_bias": float(np.sqrt(n) * np.mean(errors)),
            "n_times_empirical_variance": float(n * np.var(values, ddof=1)),
            "boundary_rate": float(np.mean([row["boundary"] for row in subset])),
        }
    return {
        "description": (
            "Known-index differing-slopes distributional battery; every fit "
            "contains D*X, eta and trim bounds are supplied, and the target "
            "is recomputed for each X/eta law."
        ),
        "dgp": {
            "x_law": dgp.x_law,
            "eta_law": dgp.eta_law,
            "error_law": dgp.error_law,
            "beta1": list(dgp.beta1),
            "beta2": list(dgp.beta2),
            "a0": dgp.a0,
            "a1": dgp.a1,
            "cost": dgp.cost,
            "sigma_eps": dgp.sigma_eps,
            "trim_eps": TRIM_EPS,
        },
        "truth": target,
        "n_values": [int(n) for n in n_values],
        "reps": int(reps),
        "seed": int(seed),
        "summary": summary,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1200, 2400, 4800])
    parser.add_argument("--reps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="baseline")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_simulation(args.n, args.reps, args.seed, SCENARIOS[args.scenario])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps(result["summary"], indent=2))
    print(f"[wrote] {args.out}")
    print(f"[wrote] {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
