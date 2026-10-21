"""Cross-validated spline sizes, as a robustness check on the fixed n^{11/60} rule.

The paper uses a fixed rule for the spline sizes (author decision, weekly scope review
2026-10-20); cross-validation over a fixed finite grid inside the (DS5) range,
n^{1/7} << K << n^{1/4} and n^{1/8} << L << n^{1/4}, is a robustness check only.
Appendix 1 of ``differing_slopes.tex`` shows that any selection from such a grid,
using the same data, leaves the limit unchanged.

* Density block.  Least-squares cross-validation of the projection estimator of
  h = E(X~ | T) f_T = (g, rho_X')' on the index fold:
  LSCV(L) = int ||h_hat||^2 - (2/n) sum_i h_hat_{-i}(T_i)' X~_i,
  an unbiased estimate of int ||h_hat - h||^2 - int ||h||^2.  For a projection
  estimator the leave-one-out fit has the closed form
  h_hat_{-i}(T_i)' X~_i = {n s_i - l_i} / (n - 1), with s_i = h_hat(T_i)' X~_i and
  l_i = N(T_i)' G^{-1} N(T_i) ||X~_i||^2.
* Outcome fit.  V-fold cross-validated squared error of the arm-wise stacked
  spline regression on J, over a grid of interior-knot counts.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np

from experiments.methods.spline_density import (
    _basis_info,
    evaluate_basis_zero_outside,
    lebesgue_gram,
)

DENSITY_EXPONENTS = (0.13, 0.155, 0.18, 0.205, 0.23)   # inside (1/8, 1/4)
KNOT_EXPONENTS = (0.15, 0.17, 0.19, 0.21, 0.23)         # inside (1/7, 1/4)


def density_size_grid(n_fit: int, multiplier: float = 2.0, minimum: int = 8,
                      exponents: Sequence[float] = DENSITY_EXPONENTS) -> list:
    """Basis sizes L = max(minimum, round(multiplier n^a)) for a on a grid inside (1/8, 1/4)."""
    for a in exponents:
        if not (1.0 / 8.0 < a < 1.0 / 4.0):
            raise ValueError("density exponents must lie strictly between 1/8 and 1/4")
    return sorted({max(int(minimum), int(round(multiplier * n_fit ** a))) for a in exponents})


def knot_grid(n_eff: int, minimum: int = 3, exponents: Sequence[float] = KNOT_EXPONENTS) -> list:
    """Interior-knot counts max(minimum, round(n^a)) for a on a grid inside (1/7, 1/4)."""
    for a in exponents:
        if not (1.0 / 7.0 < a < 1.0 / 4.0):
            raise ValueError("knot exponents must lie strictly between 1/7 and 1/4")
    return sorted({max(int(minimum), int(round(n_eff ** a))) for a in exponents})


def density_lscv(t_values: np.ndarray, x_values: np.ndarray, support: Tuple[float, float],
                 sizes: Iterable[int]) -> Dict[int, float]:
    """LSCV score of the density-block projection for each basis size (smaller is better)."""
    t = np.asarray(t_values, dtype=float).reshape(-1)
    x = np.asarray(x_values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    xt = np.column_stack((np.ones(len(t)), x))
    n = len(t)
    norm2 = np.sum(xt ** 2, axis=1)
    out = {}
    for L in sizes:
        info = _basis_info(int(L), support)
        gram = lebesgue_gram(info)
        B = evaluate_basis_zero_outside(t, info)
        omega = np.linalg.solve(gram, B.T @ xt / n)            # L x (1 + d)
        int_sq = float(np.trace(omega.T @ gram @ omega))
        s = np.sum((B @ omega) * xt, axis=1)
        lev = np.sum(B * np.linalg.solve(gram, B.T).T, axis=1) * norm2
        loo = (n * s - lev) / (n - 1)
        out[int(L)] = int_sq - 2.0 * float(np.mean(loo))
    return out


def outcome_cv(n_rows: int, design: Callable[[np.ndarray, int], np.ndarray], y: np.ndarray,
               knots: Iterable[int], rng: np.random.Generator, folds: int = 5) -> Dict[int, float]:
    """V-fold cross-validated mean squared error for each interior-knot count.

    ``design(rows, k)`` returns the regressor matrix of the given rows for k interior
    knots (the knot sequence itself must not depend on the rows)."""
    y = np.asarray(y, dtype=float)
    split = np.array_split(rng.permutation(n_rows), folds)
    out = {}
    for k in knots:
        A = design(np.arange(n_rows), int(k))
        sse = 0.0
        for v in range(folds):
            test = split[v]
            train = np.concatenate([split[w] for w in range(folds) if w != v])
            coef = np.linalg.lstsq(A[train], y[train], rcond=None)[0]
            sse += float(np.sum((y[test] - A[test] @ coef) ** 2))
        out[int(k)] = sse / n_rows
    return out


def argmin(scores: Dict[int, float]) -> int:
    return int(min(scores, key=scores.get))
