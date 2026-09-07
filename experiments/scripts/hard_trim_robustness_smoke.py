"""Short robustness checks before launching the long hard-trim simulations.

The existing Monte Carlo evidence uses a Gaussian running variable and a correctly
specified outcome model.  This module deliberately stays small: it generates a few
synthetic samples with a non-Gaussian running variable, alternative fixed nuisance
supports, and an omitted treatment--covariate interaction.  It then calls the public
hard-trim estimator and records whether the point estimate is finite and whether the
caller-supplied support is valid.

These checks are smoke tests, not evidence for the theorem.  In particular,
``perfrdd_hard_trim`` currently reports ``inference_available=False``; the bootstrap
helper below is only a numerical stability diagnostic until a feasible influence
function variance estimator is implemented.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import lognorm, norm, t as student_t

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim
from experiments.methods.spline_density import fit_spline_density


DEFAULT_N = 600
DEFAULT_REPS = 3
DEFAULT_SUPPORT = (-3.0, 3.0)
DEFAULT_PHI_GRID = np.linspace(-3.0, 3.0, 241)
POLICY_BOUNDS = (-3.0, 3.0)
EPS = 0.10
LOGNORMAL_SIGMA = 0.75
LOGNORMAL_MEAN = float(np.exp(LOGNORMAL_SIGMA ** 2 / 2.0))
LOGNORMAL_SD = float(np.sqrt(
    (np.exp(LOGNORMAL_SIGMA ** 2) - 1.0)
    * np.exp(LOGNORMAL_SIGMA ** 2)
))
MIXTURE_COMPONENTS = ((0.7, -0.4, 0.7), (0.3, 1.0, 0.8))
MIXTURE_MEAN = float(sum(weight * mean for weight, mean, _ in MIXTURE_COMPONENTS))
MIXTURE_VARIANCE = float(sum(
    weight * (sd ** 2 + (mean - MIXTURE_MEAN) ** 2)
    for weight, mean, sd in MIXTURE_COMPONENTS
))
MIXTURE_SD = float(np.sqrt(MIXTURE_VARIANCE))
GAMMA = np.array([1.0, 0.0, 0.0])
BETA = np.array([0.3, -0.2, 0.1])


def _running_variable(rng: np.random.Generator, n: int, law: str) -> np.ndarray:
    """Return a mean-zero, variance-one continuous running variable."""
    if law == "t5":
        # A t_5 variable has variance 5/(5-2)=5/3.
        return rng.standard_t(5, size=n) / np.sqrt(5.0 / 3.0)
    if law == "skewed":
        raw = rng.lognormal(mean=0.0, sigma=0.75, size=n)
        return (raw - np.exp(0.75 ** 2 / 2.0)) / np.sqrt(
            (np.exp(0.75 ** 2) - 1.0) * np.exp(0.75 ** 2)
        )
    if law == "mixture":
        component = rng.random(n) >= MIXTURE_COMPONENTS[0][0]
        raw = np.empty(n)
        for indicator, (_, mean, sd) in enumerate(MIXTURE_COMPONENTS):
            mask = component == bool(indicator)
            raw[mask] = rng.normal(mean, sd, size=int(mask.sum()))
        return (raw - MIXTURE_MEAN) / MIXTURE_SD
    raise ValueError(f"unknown running-variable law: {law!r}")


def running_density(value: float, law: str) -> float:
    """Evaluate the known density for one of the smoke-test running laws."""
    value = float(value)
    if law == "t5":
        scale = np.sqrt(5.0 / 3.0)
        return float(scale * student_t.pdf(scale * value, 5))
    if law == "skewed":
        raw = LOGNORMAL_MEAN + LOGNORMAL_SD * value
        if raw <= 0.0:
            return 0.0
        return float(LOGNORMAL_SD * lognorm.pdf(
            raw, s=LOGNORMAL_SIGMA, scale=1.0
        ))
    if law == "mixture":
        raw = MIXTURE_MEAN + MIXTURE_SD * value
        return float(MIXTURE_SD * sum(
            weight * norm.pdf(raw, mean, sd)
            for weight, mean, sd in MIXTURE_COMPONENTS
        ))
    raise ValueError(f"unknown running-variable law: {law!r}")


def running_survival(value: float, law: str) -> float:
    """Evaluate the known survival function for one smoke-test law."""
    value = float(value)
    if law == "t5":
        scale = np.sqrt(5.0 / 3.0)
        return float(student_t.sf(scale * value, 5))
    if law == "skewed":
        raw = LOGNORMAL_MEAN + LOGNORMAL_SD * value
        if raw <= 0.0:
            return 1.0
        return float(lognorm.sf(raw, s=LOGNORMAL_SIGMA, scale=1.0))
    if law == "mixture":
        raw = MIXTURE_MEAN + MIXTURE_SD * value
        return float(sum(
            weight * norm.sf(raw, mean, sd)
            for weight, mean, sd in MIXTURE_COMPONENTS
        ))
    raise ValueError(f"unknown running-variable law: {law!r}")


def running_quantile(probability: float, law: str) -> float:
    """Return a quantile of the known running-variable distribution."""
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    if law == "t5":
        return float(student_t.ppf(probability, 5) / np.sqrt(5.0 / 3.0))
    if law == "skewed":
        return float((
            lognorm.ppf(probability, s=LOGNORMAL_SIGMA, scale=1.0)
            - LOGNORMAL_MEAN
        ) / LOGNORMAL_SD)
    if law == "mixture":
        raw_quantile = brentq(
            lambda raw: sum(
                weight * norm.cdf(raw, mean, sd)
                for weight, mean, sd in MIXTURE_COMPONENTS
            ) - probability,
            -12.0,
            12.0,
        )
        return float((raw_quantile - MIXTURE_MEAN) / MIXTURE_SD)
    raise ValueError(f"unknown running-variable law: {law!r}")


@lru_cache(maxsize=None)
def trim_bounds(law: str) -> tuple[float, float]:
    """Return the population hard-trim interval induced by T's quantiles."""
    q_low = running_quantile(EPS, law)
    q_high = running_quantile(1.0 - EPS, law)
    return -q_high, -q_low


def population_utility(phi: float, law: str) -> float:
    """Known hard-trimmed population criterion for a running-variable law."""
    lower, upper = trim_bounds(law)
    return float(quad(
        lambda eta: (eta - 0.25)
        * running_survival(phi - eta, law)
        * norm.pdf(eta),
        lower,
        upper,
        epsabs=2e-10,
        limit=250,
    )[0])


def population_score(phi: float, law: str) -> float:
    """Derivative of the known population criterion."""
    lower, upper = trim_bounds(law)
    return float(-quad(
        lambda eta: (eta - 0.25)
        * running_density(phi - eta, law)
        * norm.pdf(eta),
        lower,
        upper,
        epsabs=2e-10,
        limit=250,
    )[0])


def population_truth(
    law: str,
    policy_bounds: tuple[float, float] = POLICY_BOUNDS,
) -> Dict[str, float]:
    """Find the global maximizer of the known criterion on policy_bounds."""
    lo, hi = map(float, policy_bounds)
    if not lo < hi:
        raise ValueError("policy_bounds must be increasing")
    grid = np.linspace(lo, hi, 301)
    scores = np.asarray([population_score(value, law) for value in grid])
    candidates = [lo, hi]
    for left, right, left_score, right_score in zip(
        grid[:-1], grid[1:], scores[:-1], scores[1:]
    ):
        if left_score == 0.0:
            candidates.append(float(left))
        elif left_score * right_score < 0.0:
            candidates.append(float(brentq(
                lambda value: population_score(value, law), left, right
            )))
    values = np.asarray([population_utility(value, law) for value in candidates])
    index = int(np.argmax(values))
    return {
        "phi_star": float(candidates[index]),
        "utility": float(values[index]),
        "score_at_phi_star": float(population_score(candidates[index], law)),
        "trim_lower": float(trim_bounds(law)[0]),
        "trim_upper": float(trim_bounds(law)[1]),
        "policy_lower": lo,
        "policy_upper": hi,
    }


def make_sample(
    n: int,
    seed: int,
    *,
    running_law: str = "t5",
    misspecified_outcome: bool = False,
) -> RDDSample:
    """Generate one compact smoke-test sample."""
    rng = np.random.default_rng(seed)
    X = np.column_stack((
        _running_variable(rng, n, running_law),
        rng.standard_normal((n, 2)),
    ))
    eta = rng.standard_normal(n)
    T = X @ GAMMA
    Q = T + eta
    D = (Q > 0.0).astype(float)
    treatment_effect = 2.0 + eta
    baseline = 0.5 * eta ** 2
    if misspecified_outcome:
        # The estimator permits alpha(eta) but not a treatment--X interaction.
        treatment_effect = treatment_effect + 0.75 * X[:, 1]
        baseline = baseline + 0.25 * X[:, 1] * eta
    Y = D * treatment_effect + baseline + X @ BETA
    Y += rng.normal(0.0, 0.5, size=n)
    return RDDSample(
        Q=Q,
        X=X,
        Y=Y,
        threshold=0.0,
        name=f"smoke_{running_law}_{'misspecified' if misspecified_outcome else 'correct'}",
        feature_names=["x1", "x2", "x3"],
        extras={"T": T, "eta": eta, "running_law": running_law},
    )


def estimate_once(
    sample: RDDSample,
    support: tuple[float, float] = DEFAULT_SUPPORT,
    *,
    crossfit_folds: int = 1,
) -> Dict[str, Any]:
    """Run one small point estimate without writing figures or CSV files."""
    with TemporaryDirectory(prefix="perfrdd_smoke_") as directory:
        result = perfrdd_hard_trim(
            sample,
            Path(directory),
            support,
            c_values=(2.25,),
            phi_grid=DEFAULT_PHI_GRID,
            max_n=None,
            crossfit_folds=crossfit_folds,
            write_outputs=False,
            return_curves=False,
        )
    phi = float(result["phi_star"]["2.25"])
    if not np.isfinite(phi):
        raise AssertionError("hard-trim smoke estimate is not finite")
    return {
        "phi": phi,
        "hard_retention": float(result["hard_retention"]),
        "inference_available": bool(result["inference_available"]),
        "grid_boundary": bool(result["phi_star_at_grid_boundary"]["2.25"]),
        "support": list(support),
        "crossfit_folds": int(crossfit_folds),
    }


def density_smoke(sample: RDDSample, support: tuple[float, float]) -> Dict[str, Any]:
    """Check that the spline density nuisance remains finite off Gaussian designs."""
    values = np.asarray(sample.extras["T"], dtype=float)
    fit = fit_spline_density(values, support)
    points = np.linspace(support[0] + 0.1, support[1] - 0.1, 11)
    density = fit.density(points)
    survival = fit.survival(points)
    if not np.isfinite(density).all() or not np.isfinite(survival).all():
        raise AssertionError("spline density produced non-finite values")
    return {
        "support_fraction": float(fit.support_fraction),
        "n_basis": int(fit.n_basis),
        "max_density": float(np.max(np.abs(density))),
        "max_survival": float(np.max(np.abs(survival))),
        "survival_outside_unit_interval": bool(
            np.any((survival < -0.01) | (survival > 1.01))
        ),
    }


def short_bootstrap(
    sample: RDDSample,
    reps: int = 5,
    seed: int = 9_173,
) -> Dict[str, Any]:
    """Run a tiny iid bootstrap as a stability check, not inferential output."""
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(reps)):
        indices = rng.integers(0, len(sample.Y), size=len(sample.Y))
        boot = RDDSample(
            Q=sample.Q[indices],
            X=sample.X[indices],
            Y=sample.Y[indices],
            threshold=sample.threshold,
            name=sample.name,
            feature_names=sample.feature_names,
        )
        estimates.append(estimate_once(boot)["phi"])
    values = np.asarray(estimates, dtype=float)
    return {
        "reps": int(reps),
        "finite": bool(np.isfinite(values).all()),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def run_smoke(
    n: int = DEFAULT_N,
    reps: int = DEFAULT_REPS,
    seed: int = 12_345,
) -> Dict[str, Any]:
    """Run the short robustness suite and return JSON-serializable diagnostics."""
    if n < 200:
        raise ValueError("smoke tests need at least 200 observations")
    if reps < 1:
        raise ValueError("reps must be positive")

    output: Dict[str, Any] = {"n": int(n), "reps": int(reps), "scenarios": {}}
    for law in ("t5", "skewed"):
        estimates = []
        for rep in range(reps):
            sample = make_sample(n, seed + rep, running_law=law)
            estimates.append({
                "full_sample": estimate_once(sample, crossfit_folds=1),
                "crossfit": estimate_once(sample, crossfit_folds=3),
                "spline_density": density_smoke(sample, (-8.0, 8.0)),
            })
        output["scenarios"][law] = estimates

    misspecified = make_sample(
        n, seed + 10_000, running_law="t5", misspecified_outcome=True
    )
    output["scenarios"]["misspecified_outcome"] = {
        "estimate": estimate_once(misspecified),
        "crossfit": estimate_once(misspecified, crossfit_folds=3),
    }

    support_sample = make_sample(n, seed + 20_000, running_law="t5")
    output["scenarios"]["support_sensitivity"] = {
        "narrow": estimate_once(support_sample, (-2.5, 2.5)),
        "wide": estimate_once(support_sample, (-3.5, 3.5)),
    }
    output["scenarios"]["bootstrap_diagnostic"] = short_bootstrap(
        support_sample, reps=min(5, reps + 2), seed=seed + 30_000
    )
    output["warnings"] = {
        "skewed_grid_boundary": any(
            item["full_sample"]["grid_boundary"]
            or item["crossfit"]["grid_boundary"]
            for item in output["scenarios"]["skewed"]
        ),
        "spline_survival_outside_unit_interval": any(
            item["spline_density"]["survival_outside_unit_interval"]
            for law in ("t5", "skewed")
            for item in output["scenarios"][law]
        ),
    }
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_smoke(args.n, args.reps, args.seed)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f"[wrote] {args.out}")
    print(payload, end="")


if __name__ == "__main__":
    main()
