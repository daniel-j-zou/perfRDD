"""Series estimates of g and p_X for the differing-slopes utility U_J.

The differing-slopes utility and its derivative are

    U_J(phi)  = E[ I(eta) {(alpha(eta)-c) Gbar(phi-eta) + beta2' H_X(phi-eta)} ],
    U_J'(phi) = -E[ I(eta) {(alpha(eta)-c) g(phi-eta)    + beta2' p_X(phi-eta)} ],

with g = f_T, Gbar(s) = P(T > s), p_X(t) = E(X | T=t) f_T(t) and
H_X(s) = E[X 1{T > s}] (``this_week.tex``, Sections 2--3).  Both nuisances
are estimated by the same Lebesgue-Gram orthogonal-series projection on a
fixed interval ``T``:

    omega_g = G_L^{-1} P_n N_L(T),        g_hat(t)   = N_L(t)' omega_g,
    Omega_X = G_L^{-1} P_n N_L(T) X',     p_X_hat(t) = Omega_X' N_L(t),

and the tails are their integrals from ``s`` to the upper support endpoint,
with zero extension outside the support.  The columns of ``x_values`` may be
any effect regressors R (for example X, or functions of T), in which case the
projection estimates p_R(t) = E(R | T=t) f_T(t).

``uj_score_terms`` and ``density_influence`` give the per-observation pieces
needed by the delta-method variance: the evaluation score, its curvature,
and the finite-sieve influence of the density sample on the score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipy.interpolate import BSpline

from experiments.methods.spline_density import (
    _basis_info,
    evaluate_basis_zero_outside,
    lebesgue_gram,
    spline_basis_dimension,
)


@dataclass(frozen=True)
class WeightedTails:
    """Projection estimates of g and p_X (or p_R) on one fixed interval."""

    info: Dict[str, Any]
    support: Tuple[float, float]
    gram: np.ndarray
    omega_g: np.ndarray
    omega_x: np.ndarray
    n_fit: int
    support_fraction: float

    @property
    def n_basis(self) -> int:
        return int(len(self.omega_g))

    def basis(self, values: np.ndarray | float) -> np.ndarray:
        """Spline design N_L(values), zero outside the support."""
        return evaluate_basis_zero_outside(np.asarray(values, dtype=float), self.info)

    def _spline(self, coefficients: np.ndarray) -> BSpline:
        return BSpline(
            np.asarray(self.info["t"], dtype=float),
            np.asarray(coefficients, dtype=float),
            int(self.info["degree"]),
            extrapolate=False,
        )

    def _evaluate(self, coefficients: np.ndarray, values, nu: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        flat = values.reshape(-1)
        coefficients = np.asarray(coefficients, dtype=float)
        shape = flat.shape + coefficients.shape[1:]
        out = np.zeros(shape)
        lo, hi = self.support
        inside = (flat > lo) & (flat < hi) if nu else (flat >= lo) & (flat <= hi)
        if np.any(inside):
            spline = self._spline(coefficients)
            if nu:
                spline = spline.derivative(nu)
            out[inside] = spline(flat[inside])
        return out.reshape(values.shape + coefficients.shape[1:])

    def _tail(self, coefficients: np.ndarray, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        flat = values.reshape(-1)
        lo, hi = self.support
        antiderivative = self._spline(coefficients).antiderivative()
        out = np.asarray(
            antiderivative(hi) - antiderivative(np.clip(flat, lo, hi)), dtype=float
        )
        out[flat >= hi] = 0.0
        coefficients = np.asarray(coefficients)
        return out.reshape(values.shape + coefficients.shape[1:])

    # -- scalar density g and survival Gbar ---------------------------------
    def density(self, values) -> np.ndarray:
        return self._evaluate(self.omega_g, values, 0)

    def density_derivative(self, values) -> np.ndarray:
        return self._evaluate(self.omega_g, values, 1)

    def survival(self, values) -> np.ndarray:
        return self._tail(self.omega_g, values)

    # -- weighted density p_X and weighted tail H_X ---------------------------
    def weighted_density(self, values, beta: np.ndarray | None = None) -> np.ndarray:
        """p_X_hat(values), or the scalar beta' p_X_hat(values) if beta is given."""
        return self._evaluate(self._coefficients(beta), values, 0)

    def weighted_density_derivative(self, values, beta: np.ndarray | None = None) -> np.ndarray:
        return self._evaluate(self._coefficients(beta), values, 1)

    def weighted_tail(self, values, beta: np.ndarray | None = None) -> np.ndarray:
        """H_X_hat(values), or the scalar beta' H_X_hat(values) if beta is given."""
        return self._tail(self._coefficients(beta), values)

    def _coefficients(self, beta: np.ndarray | None) -> np.ndarray:
        if beta is None:
            return self.omega_x
        return self.omega_x @ np.asarray(beta, dtype=float)


def fit_weighted_tails(
    t_values: np.ndarray,
    x_values: np.ndarray,
    support: Tuple[float, float],
    *,
    n_basis: int | None = None,
) -> WeightedTails:
    """Fit g_hat and p_X_hat by the Lebesgue-Gram projection on ``support``."""
    t = np.asarray(t_values, dtype=float).reshape(-1)
    x = np.asarray(x_values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(t) != len(x):
        raise ValueError("t_values and x_values must have the same length")
    if len(t) < 20 or not (np.isfinite(t).all() and np.isfinite(x).all()):
        raise ValueError("need at least 20 finite observations")
    if n_basis is None:
        n_basis = spline_basis_dimension(len(t))
    info = _basis_info(int(n_basis), support)
    gram = lebesgue_gram(info)
    basis = evaluate_basis_zero_outside(t, info)
    omega_g = np.linalg.solve(gram, basis.mean(axis=0))
    omega_x = np.linalg.solve(gram, basis.T @ x / len(t))
    lo, hi = map(float, support)
    return WeightedTails(
        info=info,
        support=(lo, hi),
        gram=gram,
        omega_g=omega_g,
        omega_x=omega_x,
        n_fit=len(t),
        support_fraction=float(np.mean((t >= lo) & (t <= hi))),
    )


def uj_utility(
    phi: float,
    eta: np.ndarray,
    keep: np.ndarray,
    alpha_minus_c: np.ndarray,
    beta2: np.ndarray,
    tails: WeightedTails,
) -> float:
    """Sample U_J(phi): mean of I_i[(alpha_i-c) Gbar(phi-eta_i) + beta2' H_X(phi-eta_i)]."""
    s = float(phi) - np.asarray(eta, dtype=float)
    value = alpha_minus_c * tails.survival(s) + tails.weighted_tail(s, beta2)
    return float(np.mean(np.asarray(keep, dtype=float) * value))


@dataclass(frozen=True)
class UJScoreTerms:
    """Per-observation pieces of U_J'(phi) on the evaluation sample."""

    score: np.ndarray            # -I_i[(alpha_i-c) g(s_i) + beta2' p_X(s_i)]
    curvature: np.ndarray        # -I_i[(alpha_i-c) g'(s_i) + beta2' p_X'(s_i)]
    density: np.ndarray          # g_hat(s_i)
    weighted_density: np.ndarray # p_X_hat(s_i), one column per regressor


def uj_score_terms(
    phi: float,
    eta: np.ndarray,
    keep: np.ndarray,
    alpha_minus_c: np.ndarray,
    beta2: np.ndarray,
    tails: WeightedTails,
) -> UJScoreTerms:
    s = float(phi) - np.asarray(eta, dtype=float)
    keep = np.asarray(keep, dtype=float)
    density = tails.density(s)
    weighted = tails.weighted_density(s)
    score = -keep * (alpha_minus_c * density + weighted @ np.asarray(beta2))
    curvature = -keep * (
        alpha_minus_c * tails.density_derivative(s)
        + tails.weighted_density_derivative(s, beta2)
    )
    return UJScoreTerms(score, curvature, density, weighted)


def density_influence(
    phi: float,
    eta: np.ndarray,
    keep: np.ndarray,
    alpha_minus_c: np.ndarray,
    beta2: np.ndarray,
    tails: WeightedTails,
    t_fold: np.ndarray,
    x_fold: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    """Finite-sieve influence of the density sample on the U_J' score.

    The score is linear in the projection coefficients:
    S = -(a' P_n N_L(T) + b' P_n N_L(T) X'beta2) with
    a = G_L^{-1} mean_i I_i (alpha_i-c) N_L(s_i) and b = G_L^{-1} mean_i I_i N_L(s_i).
    Observation j of the density sample therefore contributes the centered
    value of -(N_L(T_j)'a + (X_j'beta2) N_L(T_j)'b).  This is the sieve
    version of the zeta_rho terms in ``this_week.tex``.
    """
    s = float(phi) - np.asarray(eta, dtype=float)
    keep = np.asarray(keep, dtype=float)
    basis_eval = tails.basis(s)
    a = np.linalg.solve(tails.gram, basis_eval.T @ (keep * alpha_minus_c) / len(s))
    b = np.linalg.solve(tails.gram, basis_eval.T @ keep / len(s))
    x = np.asarray(x_fold, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    basis_fold = tails.basis(t_fold)
    values = -(basis_fold @ a + (x @ np.asarray(beta2)) * (basis_fold @ b))
    return values - values.mean() if center else values
