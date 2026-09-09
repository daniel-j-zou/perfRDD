"""Full-pipeline robustness battery for the differing-slopes extension.

This module tests the pieces that are deliberately held fixed in
``differing_slopes_simulation.py``:

* estimated first-stage residuals and estimated hard-trim endpoints;
* Gaussian versus least-squares spline estimates of the running-variable law;
* full-sample fitting versus five-fold cross-fitting and ridge stabilization;
* a nonlinear treatment--covariate interaction omitted from the fitted model;
* weak curvature and boundary-valued policy optima; and
* clustered outcome errors, including an iid versus cluster-robust variance
  comparison for the oracle-index estimator.

The generated-index estimators here are intentionally simple linear outcome
fits.  They are a diagnostic for the differing ``D * X`` block and the
first-stage/trim mechanics, not a replacement for the theorem's spline/Riesz
construction.  The generated-index variants therefore report Monte Carlo
dispersion but do not claim an analytic variance estimator.  The clustered
scenario explicitly demonstrates that an iid variance calculation is not
appropriate when observations share outcome shocks.

Example::

    python -m experiments.scripts.differing_slopes_full_pipeline \
        --scenario baseline --n 800 1600 3200 --reps 100 \
        --out experiments/runs/differing_slopes_full_baseline.json

Run ``--list-scenarios`` to see the available data-generating processes.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from scipy.special import roots_hermitenorm
from scipy.stats import norm

from experiments.methods.spline_density import (
    SplineDensityFit,
    _basis_info,
    evaluate_basis_zero_outside,
    lebesgue_gram,
    spline_basis_dimension,
)


POLICY_BOUNDS = (-3.0, 3.0)
POLICY_THRESHOLD = 0.0
DENSITY_SUPPORT = (-3.0, 3.0)
N_FOLDS = 5


@dataclass(frozen=True)
class DGP:
    """Parameters for the generated-index differing-slopes benchmark."""

    gamma: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    beta1: np.ndarray = field(default_factory=lambda: np.array([0.30, -0.20]))
    beta2: np.ndarray = field(default_factory=lambda: np.array([0.80, 0.25]))
    a0: float = 0.35
    a1: float = 0.90
    b0: float = 0.20
    b1: float = 0.60
    cost: float = 0.25
    sigma_eps: float = 0.50
    trim_eps: float = 0.10
    error_law: str = "normal"
    quadratic_effect: float = 0.0
    cluster_size: int = 1
    cluster_sigma: float = 0.0
    cluster_treatment_shock: bool = False

    @property
    def sigma_T(self) -> float:
        return float(np.linalg.norm(self.gamma))

    @property
    def trim_bounds(self) -> tuple[float, float]:
        q = self.sigma_T * float(norm.ppf(1.0 - self.trim_eps))
        return -q, q


DEFAULT_DGP = DGP()
SCENARIOS: dict[str, DGP] = {
    "baseline": DEFAULT_DGP,
    "quadratic_misspecification": replace(DEFAULT_DGP, quadratic_effect=0.90),
    "weak_curvature": replace(
        DEFAULT_DGP,
        a1=0.20,
        beta2=np.array([0.20, 0.05]),
    ),
    # The upper policy bound is the population optimum; the interior CLT is
    # intentionally not expected to hold in this diagnostic.
    "boundary_upper": replace(DEFAULT_DGP, cost=3.0),
    "clustered": replace(
        DEFAULT_DGP,
        cluster_size=20,
        cluster_sigma=0.35,
        cluster_treatment_shock=True,
    ),
}


@dataclass(frozen=True)
class Sample:
    X: np.ndarray
    eta: np.ndarray
    T: np.ndarray
    Q: np.ndarray
    D: np.ndarray
    Y: np.ndarray
    cluster_id: np.ndarray


def _quadrature(dgp: DGP, n_nodes: int = 240) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_hermitenorm(n_nodes)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float) / np.sqrt(
        2.0 * np.pi
    )


def _true_integrand(phi: float, eta: np.ndarray, dgp: DGP) -> np.ndarray:
    """Conditional expected utility, including the optional quadratic term.

    The quadratic misspecification is ``q2 * (X1**2 - 1)``.  The supplied
    scenarios use ``gamma=(1,0)``, for which
    ``E[(X1**2-1) 1{X1 > t}] = t * normal_pdf(t)``.
    """
    if dgp.quadratic_effect and not np.allclose(dgp.gamma, [1.0, 0.0]):
        raise ValueError("quadratic_effect currently requires gamma=(1, 0)")
    z = (float(phi) - eta) / dgp.sigma_T
    alpha = dgp.a0 + dgp.a1 * eta - dgp.cost
    value = alpha * norm.sf(z)
    value = value + float(dgp.beta2 @ dgp.gamma) / dgp.sigma_T * norm.pdf(z)
    if dgp.quadratic_effect:
        value = value + dgp.quadratic_effect * z * norm.pdf(z)
    return value


def population_utility(phi: float, dgp: DGP = DEFAULT_DGP) -> float:
    eta, weights = _quadrature(dgp)
    lo, hi = dgp.trim_bounds
    keep = (eta >= lo) & (eta <= hi)
    return float(np.sum(weights * keep * _true_integrand(phi, eta, dgp)))


def population_truth(dgp: DGP = DEFAULT_DGP) -> dict[str, float]:
    result = minimize_scalar(
        lambda value: -population_utility(value, dgp),
        bounds=POLICY_BOUNDS,
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 300},
    )
    phi = float(result.x)
    h = 2e-4
    curvature = (
        population_utility(phi + h, dgp)
        - 2.0 * population_utility(phi, dgp)
        + population_utility(phi - h, dgp)
    ) / h**2
    return {
        "phi_star": phi,
        "utility": float(-result.fun),
        "curvature": float(curvature),
        "boundary": float(
            phi <= POLICY_BOUNDS[0] + 2e-4 or phi >= POLICY_BOUNDS[1] - 2e-4
        ),
    }


def _draw_error(
    rng: np.random.Generator,
    x: np.ndarray,
    d: np.ndarray,
    dgp: DGP,
    cluster_id: np.ndarray,
) -> np.ndarray:
    n = len(x)
    if dgp.cluster_size > 1 or dgp.cluster_sigma > 0.0:
        if not 0.0 <= dgp.cluster_sigma < dgp.sigma_eps:
            raise ValueError("cluster_sigma must lie in [0, sigma_eps)")
        n_clusters = int(cluster_id.max()) + 1
        cluster_shock = rng.normal(0.0, dgp.cluster_sigma, n_clusters)
        idio_sd = float(np.sqrt(dgp.sigma_eps**2 - dgp.cluster_sigma**2))
        if dgp.cluster_treatment_shock:
            # A treated-outcome cluster shock loads directly on the policy
            # effect block, so iid standard errors should visibly fail.
            cluster_part = cluster_shock[cluster_id] * d
        else:
            cluster_part = cluster_shock[cluster_id]
        return cluster_part + rng.normal(0.0, idio_sd, n)
    if dgp.error_law == "normal":
        return rng.normal(0.0, dgp.sigma_eps, n)
    if dgp.error_law == "t5":
        return dgp.sigma_eps * rng.standard_t(5, n) / np.sqrt(5.0 / 3.0)
    if dgp.error_law == "skewed":
        return dgp.sigma_eps * (rng.exponential(1.0, n) - 1.0)
    if dgp.error_law == "heteroskedastic":
        scale = dgp.sigma_eps * (0.50 + 0.75 * np.abs(x[:, 0]))
        return rng.normal(0.0, scale, n)
    raise ValueError(f"unknown error law: {dgp.error_law!r}")


def generate_sample(n: int, seed: int, dgp: DGP = DEFAULT_DGP) -> Sample:
    rng = np.random.default_rng(seed)
    n = int(n)
    x = rng.standard_normal((n, len(dgp.gamma)))
    eta = rng.standard_normal(n)
    t = x @ dgp.gamma
    q = t + eta
    d = (q > POLICY_THRESHOLD).astype(float)
    alpha = dgp.a0 + dgp.a1 * eta
    baseline = dgp.b0 + dgp.b1 * eta + x @ dgp.beta1
    effect = alpha + x @ dgp.beta2
    if dgp.quadratic_effect:
        effect = effect + dgp.quadratic_effect * (x[:, 0] ** 2 - 1.0)
    cluster_id = np.arange(n, dtype=int) // max(int(dgp.cluster_size), 1)
    y = baseline + d * effect + _draw_error(rng, x, d, dgp, cluster_id)
    return Sample(x, eta, t, q, d, y, cluster_id)


def _fit_first_stage(sample: Sample, idx: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(idx)), sample.X[idx]))
    coef, *_ = np.linalg.lstsq(design, sample.Q[idx], rcond=None)
    return coef


def _predict_t(x: np.ndarray, gamma_hat: np.ndarray) -> np.ndarray:
    return gamma_hat[0] + x @ gamma_hat[1:]


def _design(eta: np.ndarray, x: np.ndarray, d: np.ndarray, include_beta2: bool) -> np.ndarray:
    columns = [np.ones(len(eta)), eta, x[:, 0], x[:, 1], d, d * eta]
    if include_beta2:
        columns.extend([d * x[:, 0], d * x[:, 1]])
    return np.column_stack(columns)


@dataclass(frozen=True)
class OutcomeFit:
    coef: np.ndarray
    residual: np.ndarray
    design: np.ndarray
    include_beta2: bool


def _fit_outcome(
    sample: Sample,
    eta_hat: np.ndarray,
    idx: np.ndarray,
    *,
    include_beta2: bool,
    ridge: float,
) -> OutcomeFit:
    design = _design(eta_hat[idx], sample.X[idx], sample.D[idx], include_beta2)
    if ridge == 0.0:
        coef, *_ = np.linalg.lstsq(design, sample.Y[idx], rcond=None)
    else:
        penalty = np.zeros(design.shape[1])
        # Keep the exogenous level controls unpenalized; stabilize the
        # eta/treatment blocks that drive the policy utility.
        penalty[4:] = float(ridge) / np.sqrt(len(idx))
        lhs = design.T @ design + len(idx) * np.diag(penalty)
        rhs = design.T @ sample.Y[idx]
        coef = np.linalg.solve(lhs, rhs)
    residual = sample.Y[idx] - design @ coef
    return OutcomeFit(coef, residual, design, include_beta2)


class TailEstimator(Protocol):
    """Scalar survival and vector weighted-tail interface."""

    def survival(self, values: np.ndarray | float) -> np.ndarray:
        ...

    def weighted_tail(self, values: np.ndarray | float) -> np.ndarray:
        ...


@dataclass(frozen=True)
class GaussianTail:
    mean: float
    sd: float
    gamma: np.ndarray

    def survival(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(norm.sf((np.asarray(values) - self.mean) / self.sd))

    def weighted_tail(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        z = (values - self.mean) / self.sd
        out = norm.pdf(z)[..., None] * (self.gamma / self.sd)
        return out if values.ndim else out.reshape(-1)


@dataclass(frozen=True)
class SplineTail:
    scalar_fit: SplineDensityFit
    scalar_antiderivative: Any
    vector_antiderivatives: tuple[Any, ...]
    degree: int
    support: tuple[float, float]

    def survival(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        flat = values.reshape(-1)
        lo, hi = self.support
        clipped = np.clip(flat, lo, hi)
        output = np.asarray(
            self.scalar_antiderivative(hi) - self.scalar_antiderivative(clipped),
            dtype=float,
        )
        output[flat >= hi] = 0.0
        return output.reshape(values.shape)

    def weighted_tail(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        flat = values.reshape(-1)
        lo, hi = self.support
        clipped = np.clip(flat, lo, hi)
        out = np.column_stack([
            antiderivative(hi) - antiderivative(clipped)
            for antiderivative in self.vector_antiderivatives
        ])
        if out.ndim == 1:
            out = out.reshape(1, -1)
        for column in range(out.shape[1]):
            out[flat >= hi, column] = 0.0
        return out.reshape(values.shape + (len(self.vector_antiderivatives),))


@lru_cache(maxsize=None)
def _spline_metadata(n_basis: int) -> tuple[dict[str, Any], np.ndarray]:
    """Cache deterministic spline knots and Lebesgue Gram matrices."""
    info = _basis_info(int(n_basis), DENSITY_SUPPORT)
    return info, lebesgue_gram(info)


def _fit_spline_tail(t_values: np.ndarray, x_values: np.ndarray) -> SplineTail:
    n_basis = spline_basis_dimension(len(t_values))
    info, gram = _spline_metadata(n_basis)
    basis = evaluate_basis_zero_outside(t_values, info)
    scalar_coefficients = np.linalg.solve(gram, np.mean(basis, axis=0))
    vector_coefficients = np.linalg.solve(gram, basis.T @ x_values / len(t_values))
    scalar_fit = SplineDensityFit(
        knots=np.asarray(info["t"], dtype=float),
        degree=int(info["degree"]),
        support=DENSITY_SUPPORT,
        coefficients=np.asarray(scalar_coefficients, dtype=float),
        gram_condition_number=float(np.linalg.cond(gram)),
        n_fit=len(t_values),
        n_basis=int(n_basis),
        support_fraction=float(np.mean(
            (t_values >= DENSITY_SUPPORT[0]) & (t_values <= DENSITY_SUPPORT[1])
        )),
    )
    scalar_spline = BSpline(
        scalar_fit.knots,
        scalar_fit.coefficients,
        scalar_fit.degree,
        extrapolate=False,
    )
    vector_splines = tuple(
        BSpline(
            np.asarray(info["t"], dtype=float),
            vector_coefficients[:, column],
            int(info["degree"]),
            extrapolate=False,
        ).antiderivative()
        for column in range(vector_coefficients.shape[1])
    )
    return SplineTail(
        scalar_fit=scalar_fit,
        scalar_antiderivative=scalar_spline.antiderivative(),
        vector_antiderivatives=vector_splines,
        degree=int(info["degree"]),
        support=DENSITY_SUPPORT,
    )


def _fit_tail(
    t_values: np.ndarray,
    x_values: np.ndarray,
    gamma_hat: np.ndarray,
    method: str,
) -> TailEstimator:
    if method == "gaussian":
        sd = float(np.std(t_values, ddof=1))
        if not np.isfinite(sd) or sd <= 1e-8:
            raise ValueError("estimated T standard deviation is degenerate")
        return GaussianTail(float(np.mean(t_values)), sd, gamma_hat)
    if method == "spline":
        return _fit_spline_tail(t_values, x_values)
    raise ValueError(f"unknown density method: {method!r}")


@dataclass(frozen=True)
class Component:
    eta: np.ndarray
    hard_weights: np.ndarray
    outcome: OutcomeFit
    tail: TailEstimator
    include_beta2: bool
    l_hat: float
    u_hat: float


def _trim_bounds_from_t(t_values: np.ndarray, eps: float) -> tuple[float, float]:
    lower = POLICY_THRESHOLD - float(np.quantile(t_values, 1.0 - eps))
    upper = POLICY_THRESHOLD - float(np.quantile(t_values, eps))
    if not lower < upper:
        raise ValueError(f"invalid estimated trim interval: [{lower}, {upper}]")
    return lower, upper


def _component(
    sample: Sample,
    dgp: DGP,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    density_method: str,
    ridge: float,
    include_beta2: bool,
) -> Component:
    gamma_hat = _fit_first_stage(sample, train_idx)
    t_train = _predict_t(sample.X[train_idx], gamma_hat)
    eta_hat = sample.Q - _predict_t(sample.X, gamma_hat)
    l_hat, u_hat = _trim_bounds_from_t(t_train, dgp.trim_eps)
    outcome = _fit_outcome(
        sample,
        eta_hat,
        train_idx,
        include_beta2=include_beta2,
        ridge=ridge,
    )
    tail = _fit_tail(t_train, sample.X[train_idx], gamma_hat[1:], density_method)
    eta_eval = eta_hat[eval_idx]
    weights = ((eta_eval >= l_hat) & (eta_eval <= u_hat)).astype(float)
    return Component(
        eta=eta_eval,
        hard_weights=weights,
        outcome=outcome,
        tail=tail,
        include_beta2=include_beta2,
        l_hat=l_hat,
        u_hat=u_hat,
    )


def _maximize(components: Sequence[Component], dgp: DGP) -> tuple[float, bool, float]:
    denominator = float(sum(np.sum(part.hard_weights) for part in components))
    if denominator < 20.0:
        raise ValueError("too few observations survive hard trimming")

    def objective(phi: float) -> float:
        numerator = 0.0
        for part in components:
            coef = part.outcome.coef
            eta = part.eta
            alpha_hat = coef[4] + coef[5] * eta
            values = (alpha_hat - dgp.cost) * part.tail.survival(phi - eta)
            if part.include_beta2:
                weighted = part.tail.weighted_tail(phi - eta)
                values = values + weighted @ coef[6:8]
            numerator += float(np.sum(part.hard_weights * values))
        return numerator / denominator

    # Vectorize the coarse search over candidate thresholds.  This is important
    # for the spline/cross-fit battery: evaluating a scalar spline 241 times
    # per fold is needlessly expensive and does not improve the argmax audit.
    grid = np.linspace(POLICY_BOUNDS[0], POLICY_BOUNDS[1], 81)
    numerator = np.zeros(len(grid))
    for part in components:
        coef = part.outcome.coef
        eta = part.eta
        alpha_hat = coef[4] + coef[5] * eta
        arguments = grid[:, None] - eta[None, :]
        survival = part.tail.survival(arguments)
        values_grid = (alpha_hat[None, :] - dgp.cost) * survival
        if part.include_beta2:
            weighted = part.tail.weighted_tail(arguments)
            values_grid = values_grid + weighted @ coef[6:8]
        numerator += np.sum(part.hard_weights[None, :] * values_grid, axis=1)
    values = numerator / denominator
    index = int(np.argmax(values))
    if index == 0 or index == len(grid) - 1:
        phi_hat = float(grid[index])
    else:
        local = minimize_scalar(
            lambda value: -objective(float(value)),
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1e-8, "maxiter": 200},
        )
        phi_hat = float(local.x)
    boundary = bool(
        phi_hat <= POLICY_BOUNDS[0] + 2e-4
        or phi_hat >= POLICY_BOUNDS[1] - 2e-4
    )
    return phi_hat, boundary, denominator


def _folds(n: int, seed: int, n_folds: int = N_FOLDS) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 91_401_221)
    return [np.asarray(part, dtype=int) for part in np.array_split(rng.permutation(n), n_folds)]


def _fit_variant(
    sample: Sample,
    dgp: DGP,
    seed: int,
    variant: str,
) -> tuple[float, bool, float, Component | None]:
    if variant == "oracle":
        idx = np.arange(len(sample.Y))
        outcome = _fit_outcome(
            sample,
            sample.eta,
            idx,
            include_beta2=True,
            ridge=0.0,
        )
        tail = GaussianTail(0.0, dgp.sigma_T, dgp.gamma)
        lo, hi = dgp.trim_bounds
        component = Component(
            eta=sample.eta,
            hard_weights=((sample.eta >= lo) & (sample.eta <= hi)).astype(float),
            outcome=outcome,
            tail=tail,
            include_beta2=True,
            l_hat=lo,
            u_hat=hi,
        )
        phi, boundary, retained = _maximize([component], dgp)
        return phi, boundary, retained, component

    include_beta2 = "alpha_only" not in variant
    ridge = 0.0
    if "ridge" in variant:
        ridge = 0.50
    density = "spline" if "spline" in variant else "gaussian"
    if "crossfit5" in variant:
        all_idx = np.arange(len(sample.Y))
        parts: list[Component] = []
        for eval_idx in _folds(len(sample.Y), seed):
            train_mask = np.ones(len(sample.Y), dtype=bool)
            train_mask[eval_idx] = False
            parts.append(
                _component(
                    sample,
                    dgp,
                    all_idx[train_mask],
                    eval_idx,
                    density_method=density,
                    ridge=ridge,
                    include_beta2=include_beta2,
                )
            )
        phi, boundary, retained = _maximize(parts, dgp)
        return phi, boundary, retained, None

    idx = np.arange(len(sample.Y))
    component = _component(
        sample,
        dgp,
        idx,
        idx,
        density_method=density,
        ridge=ridge,
        include_beta2=include_beta2,
    )
    phi, boundary, retained = _maximize([component], dgp)
    return phi, boundary, retained, component


def _oracle_variance(
    sample: Sample,
    dgp: DGP,
    component: Component,
    phi_hat: float,
) -> tuple[float, float]:
    """Return iid and cluster-robust variance estimates for oracle scoring.

    This is the same delta-method calculation used in the known-target script,
    now with a cluster-sum alternative.  It intentionally excludes generated
    index and endpoint terms; those terms are precisely what the full theorem
    must add.
    """
    coef = component.outcome.coef
    design = component.outcome.design
    residual = component.outcome.residual
    eta = component.eta
    keep = component.hard_weights
    sig = dgp.sigma_T
    z = (phi_hat - eta) / sig
    density = norm.pdf(z)
    alpha_hat = coef[4] + coef[5] * eta - dgp.cost
    beta2_dot_gamma = float(coef[6:8] @ dgp.gamma)
    score = keep * (
        -alpha_hat * density / sig
        - beta2_dot_gamma * z * density / sig**2
    )
    curvature_terms = keep * (
        alpha_hat * z * density / sig**2
        + beta2_dot_gamma * (z**2 - 1.0) * density / sig**3
    )
    gradient = np.zeros((len(eta), len(coef)))
    gradient[:, 4] = keep * (-density / sig)
    gradient[:, 5] = keep * (-eta * density / sig)
    gradient[:, 6:8] = keep[:, None] * (
        -z[:, None] * density[:, None] * dgp.gamma[None, :] / sig**2
    )
    bread = np.linalg.inv((design.T @ design) / len(design))
    theta_influence = (design @ bread.T) * residual[:, None]
    psi = (score - np.mean(score)) + theta_influence @ np.mean(gradient, axis=0)
    curvature = float(np.mean(curvature_terms))
    iid = float(np.var(psi, ddof=1) / max(curvature**2, 1e-16) / len(psi))
    cluster_sums = np.asarray([
        np.sum(psi[sample.cluster_id == group])
        for group in np.unique(sample.cluster_id)
    ])
    cluster = float(
        np.var(cluster_sums, ddof=1)
        / max(len(cluster_sums), 1)
        / max(curvature**2, 1e-16)
    )
    # The preceding expression is Var(mean psi) / curvature^2: the sample
    # mean is the cluster-sum mean divided by n, so restore that denominator.
    cluster = cluster * (len(cluster_sums) / len(psi)) ** 2
    return iid, cluster


VARIANTS = (
    "oracle",
    "full_gaussian_alpha_only",
    "full_gaussian_ols",
    "full_gaussian_ridge",
    "full_spline_ols",
    "crossfit5_gaussian",
    "crossfit5_spline",
    "crossfit5_gaussian_alpha_only",
)


def run_experiment(
    scenarios: Sequence[str],
    n_values: Iterable[int],
    reps: int,
    seed: int,
    variants: Sequence[str] = VARIANTS,
) -> dict[str, Any]:
    scenarios = tuple(str(name) for name in scenarios)
    n_values = tuple(int(value) for value in n_values)
    variants = tuple(str(value) for value in variants)
    rows: list[dict[str, Any]] = []
    truths = {name: population_truth(SCENARIOS[name]) for name in scenarios}
    for scenario_index, scenario_name in enumerate(scenarios):
        dgp = SCENARIOS[scenario_name]
        for n in n_values:
            for rep in range(int(reps)):
                sample_seed = int(seed + 1_000_003 * scenario_index + 10_007 * int(n) + rep)
                sample = generate_sample(int(n), sample_seed, dgp)
                for variant in variants:
                    phi, boundary, retained, component = _fit_variant(
                        sample, dgp, sample_seed, variant
                    )
                    row: dict[str, Any] = {
                        "scenario": scenario_name,
                        "n": int(n),
                        "rep": int(rep),
                        "variant": variant,
                        "phi_hat": phi,
                        "boundary": int(boundary),
                        "retained": retained,
                        "variance_iid": None,
                        "variance_cluster": None,
                    }
                    if variant == "oracle" and component is not None:
                        iid, cluster = _oracle_variance(sample, dgp, component, phi)
                        row["variance_iid"] = iid
                        row["variance_cluster"] = cluster
                    rows.append(row)

    summaries: dict[str, Any] = {}
    for scenario_name in scenarios:
        target = truths[scenario_name]["phi_star"]
        summaries[scenario_name] = {}
        for n in sorted({int(row["n"]) for row in rows if row["scenario"] == scenario_name}):
            summaries[scenario_name][str(n)] = {}
            for variant in variants:
                subset = [
                    row for row in rows
                    if row["scenario"] == scenario_name
                    and int(row["n"]) == n
                    and row["variant"] == variant
                ]
                estimates = np.asarray([float(row["phi_hat"]) for row in subset])
                mc_variance = float(np.var(estimates, ddof=1))
                item: dict[str, Any] = {
                    "replications": len(subset),
                    "target_phi": target,
                    "mean_phi": float(np.mean(estimates)),
                    "bias": float(np.mean(estimates - target)),
                    "rmse": float(np.sqrt(np.mean((estimates - target) ** 2))),
                    "mc_variance": mc_variance,
                    "n_scaled_mc_variance": float(n * mc_variance),
                    "boundary_rate": float(np.mean([row["boundary"] for row in subset])),
                    "mean_retained_fraction": float(np.mean([
                        float(row["retained"]) / n for row in subset
                    ])),
                }
                iid_values = [row["variance_iid"] for row in subset if row["variance_iid"] is not None]
                cluster_values = [row["variance_cluster"] for row in subset if row["variance_cluster"] is not None]
                if iid_values:
                    iid_mean = float(np.mean(iid_values))
                    cluster_mean = float(np.mean(cluster_values))
                    iid_ratio = None if mc_variance <= 1e-12 else iid_mean / mc_variance
                    cluster_ratio = None if mc_variance <= 1e-12 else cluster_mean / mc_variance
                    item.update({
                        "mean_variance_iid": iid_mean,
                        "iid_variance_ratio": iid_ratio,
                        "mean_variance_cluster": cluster_mean,
                        "cluster_variance_ratio": cluster_ratio,
                        "iid_coverage": float(np.mean([
                            low <= target <= high
                            for low, high in (
                                (float(row["phi_hat"]) - 1.96 * np.sqrt(float(row["variance_iid"])),
                                 float(row["phi_hat"]) + 1.96 * np.sqrt(float(row["variance_iid"])))
                                for row in subset
                            )
                        ])),
                        "cluster_coverage": float(np.mean([
                            low <= target <= high
                            for low, high in (
                                (float(row["phi_hat"]) - 1.96 * np.sqrt(float(row["variance_cluster"])),
                                 float(row["phi_hat"]) + 1.96 * np.sqrt(float(row["variance_cluster"])))
                                for row in subset
                            )
                        ])),
                    })
                summaries[scenario_name][str(n)][variant] = item

    return {
        "description": "Full generated-index differing-slopes robustness battery; generated variants report Monte Carlo dispersion, while oracle variance is conditional on known eta and fixed support.",
        "scenarios": {
            name: {
                "dgp": {
                    "gamma": SCENARIOS[name].gamma.tolist(),
                    "beta1": SCENARIOS[name].beta1.tolist(),
                    "beta2": SCENARIOS[name].beta2.tolist(),
                    "a0": SCENARIOS[name].a0,
                    "a1": SCENARIOS[name].a1,
                    "cost": SCENARIOS[name].cost,
                    "sigma_eps": SCENARIOS[name].sigma_eps,
                    "trim_eps": SCENARIOS[name].trim_eps,
                    "error_law": SCENARIOS[name].error_law,
                    "quadratic_effect": SCENARIOS[name].quadratic_effect,
                    "cluster_size": SCENARIOS[name].cluster_size,
                    "cluster_sigma": SCENARIOS[name].cluster_sigma,
                    "cluster_treatment_shock": SCENARIOS[name].cluster_treatment_shock,
                    "trim_bounds": list(SCENARIOS[name].trim_bounds),
                },
                "truth": truths[name],
            }
            for name in scenarios
        },
        "n_values": [int(value) for value in n_values],
        "reps": int(reps),
        "seed": int(seed),
        "variants": list(variants),
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(result: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    rows = result["rows"]
    if rows:
        with out.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), nargs="+", default=["baseline"])
    parser.add_argument("--n", type=int, nargs="+", default=[800, 1600, 3200])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--variant", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args(argv)
    if args.list_scenarios:
        print("\n".join(sorted(SCENARIOS)))
        return
    if args.reps <= 1:
        parser.error("--reps must exceed one")
    result = run_experiment(args.scenario, args.n, args.reps, args.seed, args.variant)
    write_outputs(result, args.out)
    print(json.dumps(result["summary"], indent=2))
    print(f"[wrote] {args.out}")
    print(f"[wrote] {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
