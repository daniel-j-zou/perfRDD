"""Least-squares B-spline projection density estimator.

This module implements the density nuisance used in ``prefRDD.tex``.  For a
fixed B-spline basis ``B_K`` on a deterministic support, the estimator is

    g_hat(t) = B_K(t)' G_K^{-1} P_n B_K(T),
    G_K = integral B_K(t) B_K(t)' dt.

The Lebesgue Gram matrix and knot sequence are deterministic.  Observations
outside the fixed support contribute a zero basis vector, rather than being
clipped to a boundary knot.  The fitted density is intentionally not forced
to be nonnegative or to integrate to one; this matches the series estimator
in the manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline

from experiments.methods.perfrdd import _basis_params


@dataclass(frozen=True)
class SplineDensityFit:
    """Fitted projection density and its deterministic basis metadata."""

    knots: np.ndarray
    degree: int
    support: Tuple[float, float]
    coefficients: np.ndarray
    gram_condition_number: float
    n_fit: int
    n_basis: int
    support_fraction: float

    def density(self, points: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted density, taking it to be zero off support."""
        values = np.asarray(points, dtype=float)
        flat = values.reshape(-1)
        output = np.zeros_like(flat)
        inside = (flat >= self.support[0]) & (flat <= self.support[1])
        if np.any(inside):
            spline = BSpline(
                self.knots, self.coefficients, self.degree, extrapolate=False
            )
            output[inside] = spline(flat[inside])
        return output.reshape(values.shape)

    def density_derivative(self, points: np.ndarray | float) -> np.ndarray:
        """Evaluate the ordinary derivative in the support interior."""
        values = np.asarray(points, dtype=float)
        flat = values.reshape(-1)
        output = np.zeros_like(flat)
        inside = (flat > self.support[0]) & (flat < self.support[1])
        if np.any(inside):
            spline = BSpline(
                self.knots, self.coefficients, self.degree, extrapolate=False
            ).derivative()
            output[inside] = spline(flat[inside])
        return output.reshape(values.shape)

    def survival(self, points: np.ndarray | float) -> np.ndarray:
        """Return ``integral_t^support_hi g_hat(s) ds``.

        Values below the support receive the fitted total mass and values
        above the support receive zero.  The relevant policy arguments should
        lie strictly inside the support in theorem-style uses.
        """
        values = np.asarray(points, dtype=float)
        flat = values.reshape(-1)
        lo, hi = self.support
        clipped = np.clip(flat, lo, hi)
        spline = BSpline(
            self.knots, self.coefficients, self.degree, extrapolate=False
        ).antiderivative()
        output = np.asarray(spline(hi) - spline(clipped), dtype=float)
        output[flat >= hi] = 0.0
        return output.reshape(values.shape)


def spline_basis_dimension(
    n_fit: int,
    exponent: float = 11.0 / 60.0,
    multiplier: float = 2.0,
    minimum: int = 8,
) -> int:
    """Choose the total number of cubic-spline basis functions.

    The exponent lies strictly between the manuscript's lower and upper rate
    exponents, 1/6 and 1/5.  ``multiplier`` is a fixed finite-sample constant;
    unlike a data-selected bandwidth it does not change the rate calculation.
    """
    if n_fit <= 0:
        raise ValueError("n_fit must be positive")
    if not (1.0 / 6.0 < exponent < 1.0 / 5.0):
        raise ValueError("exponent must lie strictly between 1/6 and 1/5")
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("multiplier must be finite and positive")
    return max(int(minimum), int(round(multiplier * n_fit ** exponent)))


def _basis_info(n_basis: int, support: Tuple[float, float]) -> Dict[str, Any]:
    degree = 3
    if n_basis < degree + 1:
        raise ValueError("a cubic spline needs at least four basis functions")
    lo, hi = map(float, support)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("support must contain two increasing finite endpoints")
    return _basis_params(n_basis - degree - 1, (lo, hi))


def evaluate_basis_zero_outside(
    points: np.ndarray,
    info: Dict[str, Any],
) -> np.ndarray:
    """Evaluate a B-spline design with zero extension off its support."""
    values = np.asarray(points, dtype=float).reshape(-1)
    n_basis = len(info["t"]) - int(info["degree"]) - 1
    design = np.zeros((len(values), n_basis))
    inside = (values >= info["lo"]) & (values <= info["hi"])
    if np.any(inside):
        design[inside] = BSpline.design_matrix(
            values[inside], info["t"], info["degree"], extrapolate=False
        ).toarray()
    return design


def lebesgue_gram(info: Dict[str, Any], quadrature_order: int = 5) -> np.ndarray:
    """Compute the deterministic Lebesgue Gram exactly up to roundoff.

    A product of two cubic B-splines is degree six on each knot interval, so
    five-point Gauss--Legendre quadrature integrates it exactly interval by
    interval.
    """
    if quadrature_order < 4:
        raise ValueError("quadrature_order must be at least four")
    nodes, weights = leggauss(quadrature_order)
    breaks = np.unique(np.asarray(info["t"], dtype=float))
    gram = None
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right <= left:
            continue
        points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
        interval_weights = 0.5 * (right - left) * weights
        basis = evaluate_basis_zero_outside(points, info)
        contribution = basis.T @ (interval_weights[:, None] * basis)
        gram = contribution if gram is None else gram + contribution
    if gram is None:
        raise ValueError("basis has no positive-length knot intervals")
    return gram


def fit_spline_density(
    values: np.ndarray,
    support: Tuple[float, float],
    *,
    n_basis: int | None = None,
    exponent: float = 11.0 / 60.0,
    multiplier: float = 2.0,
) -> SplineDensityFit:
    """Fit the manuscript's least-squares projection density estimator."""
    sample = np.asarray(values, dtype=float).reshape(-1)
    if len(sample) < 20 or not np.isfinite(sample).all():
        raise ValueError("density sample must contain at least 20 finite values")
    if n_basis is None:
        n_basis = spline_basis_dimension(len(sample), exponent, multiplier)
    info = _basis_info(int(n_basis), support)
    basis = evaluate_basis_zero_outside(sample, info)
    gram = lebesgue_gram(info)
    moments = np.mean(basis, axis=0)
    try:
        coefficients = np.linalg.solve(gram, moments)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Lebesgue spline Gram matrix is singular") from exc
    lo, hi = support
    return SplineDensityFit(
        knots=np.asarray(info["t"], dtype=float),
        degree=int(info["degree"]),
        support=(float(lo), float(hi)),
        coefficients=np.asarray(coefficients, dtype=float),
        gram_condition_number=float(np.linalg.cond(gram)),
        n_fit=len(sample),
        n_basis=int(n_basis),
        support_fraction=float(np.mean((sample >= lo) & (sample <= hi))),
    )


def project_known_density(
    density: Callable[[np.ndarray], np.ndarray],
    support: Tuple[float, float],
    n_basis: int,
    *,
    quadrature_order: int = 20,
) -> SplineDensityFit:
    """Return the population spline projection of a known density.

    This companion to :func:`fit_spline_density` is useful for DGP-known
    simulation benchmarks.  It replaces empirical basis moments with accurate
    numerical integrals while retaining the identical deterministic Gram.
    """
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least eight")
    info = _basis_info(int(n_basis), support)
    gram = lebesgue_gram(info)
    nodes, weights = leggauss(quadrature_order)
    breaks = np.unique(np.asarray(info["t"], dtype=float))
    moments = np.zeros(int(n_basis))
    total_mass = 0.0
    for left, right in zip(breaks[:-1], breaks[1:]):
        if right <= left:
            continue
        points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
        interval_weights = 0.5 * (right - left) * weights
        density_values = np.asarray(density(points), dtype=float)
        basis = evaluate_basis_zero_outside(points, info)
        moments += basis.T @ (interval_weights * density_values)
        total_mass += float(np.sum(interval_weights * density_values))
    coefficients = np.linalg.solve(gram, moments)
    lo, hi = support
    return SplineDensityFit(
        knots=np.asarray(info["t"], dtype=float),
        degree=int(info["degree"]),
        support=(float(lo), float(hi)),
        coefficients=np.asarray(coefficients, dtype=float),
        gram_condition_number=float(np.linalg.cond(gram)),
        n_fit=0,
        n_basis=int(n_basis),
        support_fraction=total_mass,
    )
