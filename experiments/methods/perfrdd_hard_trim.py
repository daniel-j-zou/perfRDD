"""Exact hard-support-trimmed PerfRDD point estimator.

This module implements the empirical criterion

    sum_i (alpha_hat(eta_hat_i) - c)
          Gbar_hat(phi - eta_hat_i)
          1{l_hat <= eta_hat_i <= u_hat}

without smoothing the support indicator.  Unlike the legacy
``perfrdd_trim`` implementation, the nuisance spline is fit on a compact
interval supplied by the caller.  Its support and knots therefore do not move
with the estimated trim endpoints.

Two point-estimation modes are available:

* ``crossfit_folds=1`` fits and evaluates on the full sample;
* ``crossfit_folds>=2`` evaluates every observation with nuisances estimated
  without that observation.

The cross-fitted mode is useful as a robustness check, but it is not by itself
the manuscript's complete inference construction: that construction also
uses separate lower- and upper-boundary folds and an explicit influence-
function variance estimator.  This module reports point estimates only.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd import (
    DEFAULT_MAX_N,
    _basis_params,
    _detect_direction,
    _eval_basis,
    _reduce_to_primary_axis,
    _subsample,
)


DEFAULT_KNOT_EXPONENT = 11.0 / 60.0


@dataclass(frozen=True)
class HardTrimFold:
    """Held-out policy inputs and diagnostics from one nuisance fit."""

    eval_idx: np.ndarray
    eta: np.ndarray
    hard_weights: np.ndarray
    treatment_effect: np.ndarray
    T_sorted: np.ndarray
    l_hat: float
    u_hat: float
    n_train: int
    n_fit: int
    n_fit_treated: int
    n_fit_control: int
    n_basis: int
    design_rank: int
    design_condition_number: float
    first_stage_R2: float
    alpha_grid: np.ndarray


def _validate_inputs(
    eps: float,
    nuisance_support: tuple[float, float],
    ridge_scale: float,
    knot_const: float,
    knot_exponent: float,
    crossfit_folds: int,
) -> tuple[float, float]:
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must lie strictly between zero and one half")
    lo, hi = (float(nuisance_support[0]), float(nuisance_support[1]))
    if not np.isfinite([lo, hi]).all() or not lo < hi:
        raise ValueError("nuisance_support must contain two finite increasing values")
    if not np.isfinite(ridge_scale) or ridge_scale < 0.0:
        raise ValueError("ridge_scale must be finite and nonnegative")
    if not np.isfinite(knot_const) or knot_const <= 0.0:
        raise ValueError("knot_const must be finite and positive")
    if not 0.0 < knot_exponent < 1.0:
        raise ValueError("knot_exponent must lie strictly between zero and one")
    if crossfit_folds < 1:
        raise ValueError("crossfit_folds must be a positive integer")
    return lo, hi


def _make_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return evaluation folds that partition the analysis sample exactly."""
    if n_folds == 1:
        return [np.arange(n)]
    if n_folds > n:
        raise ValueError("crossfit_folds cannot exceed the analysis sample size")
    rng = np.random.default_rng(seed)
    return [
        np.asarray(part, dtype=int)
        for part in np.array_split(rng.permutation(n), n_folds)
    ]


def _fit_first_stage(
    Q: np.ndarray, X: np.ndarray, train_idx: np.ndarray,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(train_idx)), X[train_idx]))
    gamma, *_ = np.linalg.lstsq(design, Q[train_idx], rcond=None)
    return gamma


def _fit_fold(
    Q: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    threshold: float,
    eps: float,
    support: tuple[float, float],
    ridge_scale: float,
    knot_const: float,
    knot_exponent: float,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    alpha_grid: np.ndarray,
) -> HardTrimFold:
    """Estimate all nuisances on one training set and score one evaluation set."""
    gamma = _fit_first_stage(Q, X, train_idx)
    eta_all = Q - gamma[0] - X @ gamma[1:]
    T_train = gamma[0] + X[train_idx] @ gamma[1:]
    l_hat = threshold - float(np.quantile(T_train, 1.0 - eps))
    u_hat = threshold - float(np.quantile(T_train, eps))
    if not l_hat < u_hat:
        raise ValueError(f"estimated hard-trim interval is invalid: [{l_hat}, {u_hat}]")

    lo, hi = support
    if l_hat <= lo or u_hat >= hi:
        raise ValueError(
            "estimated trim interval is not strictly inside nuisance_support: "
            f"trim=[{l_hat:.6g}, {u_hat:.6g}], support=[{lo:.6g}, {hi:.6g}]"
        )

    in_support = (eta_all[train_idx] >= lo) & (eta_all[train_idx] <= hi)
    fit_idx = train_idx[in_support]
    D_fit = D[fit_idx]
    n_treated = int(D_fit.sum())
    n_control = int(len(fit_idx) - n_treated)
    if len(fit_idx) < 50 or min(n_treated, n_control) < 10:
        raise ValueError(
            "nuisance-support training sample is too small: "
            f"n={len(fit_idx)}, treated={n_treated}, control={n_control}"
        )

    n_eff = min(n_treated, n_control)
    n_interior_knots = max(
        3, int(round(float(knot_const) * n_eff ** float(knot_exponent)))
    )
    basis_info = _basis_params(n_interior_knots, support)
    Phi = _eval_basis(eta_all[fit_idx], basis_info)
    design = np.column_stack((
        X[fit_idx],
        Phi,
        D_fit[:, None] * Phi,
    ))
    n_features = X.shape[1]
    if ridge_scale == 0.0:
        coefficients, *_ = np.linalg.lstsq(design, Y[fit_idx], rcond=None)
    else:
        penalty = np.zeros(design.shape[1])
        penalty[n_features:] = ridge_scale / np.sqrt(len(fit_idx))
        lhs = design.T @ design + len(fit_idx) * np.diag(penalty)
        rhs = design.T @ Y[fit_idx]
        try:
            coefficients = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coefficients, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    n_basis = Phi.shape[1]
    design_rank = int(np.linalg.matrix_rank(design))
    design_condition_number = float(np.linalg.cond(design))
    omega_treat = coefficients[
        n_features + n_basis:n_features + 2 * n_basis
    ]
    eta_eval = eta_all[eval_idx]
    hard_weights = (
        (eta_eval >= l_hat) & (eta_eval <= u_hat)
    ).astype(float)
    treatment_effect = _eval_basis(eta_eval, basis_info) @ omega_treat
    return HardTrimFold(
        eval_idx=eval_idx,
        eta=eta_eval,
        hard_weights=hard_weights,
        treatment_effect=treatment_effect,
        T_sorted=np.sort(T_train),
        l_hat=l_hat,
        u_hat=u_hat,
        n_train=len(train_idx),
        n_fit=len(fit_idx),
        n_fit_treated=n_treated,
        n_fit_control=n_control,
        n_basis=n_basis,
        design_rank=design_rank,
        design_condition_number=design_condition_number,
        first_stage_R2=float(
            1.0 - np.var(eta_all[train_idx]) / np.var(Q[train_idx])
        ),
        alpha_grid=_eval_basis(alpha_grid, basis_info) @ omega_treat,
    )


def _utility_curves(
    folds: Sequence[HardTrimFold],
    direction: str,
    phi_grid: np.ndarray,
    c_values: Sequence[float],
) -> Dict[float, np.ndarray]:
    """Evaluate the exact hard-gated criterion using training-fold empirical CDFs."""
    denominator = float(sum(np.sum(fold.hard_weights) for fold in folds))
    if denominator < 20.0:
        raise ValueError("fewer than 20 evaluation observations survive hard trimming")

    alpha_part = np.zeros(len(phi_grid))
    treatment_part = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        for fold in folds:
            cdf = np.searchsorted(
                fold.T_sorted, phi - fold.eta
            ) / len(fold.T_sorted)
            probability = cdf if direction == "below" else 1.0 - cdf
            weighted_probability = fold.hard_weights * probability
            alpha_part[j] += float(np.sum(
                weighted_probability * fold.treatment_effect
            ))
            treatment_part[j] += float(np.sum(weighted_probability))
    return {
        float(cost): (alpha_part - float(cost) * treatment_part) / denominator
        for cost in c_values
    }


def _write_curve_csv(
    path: Path,
    x_name: str,
    x: np.ndarray,
    columns: Dict[str, np.ndarray],
) -> None:
    names = [x_name, *columns]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row in zip(x, *(columns[name] for name in columns)):
            writer.writerow([float(value) for value in row])


def _plot_diagnostics(
    out_dir: Path,
    sample_name: str,
    threshold: float,
    support: tuple[float, float],
    trim_interval: tuple[float, float],
    alpha_grid: np.ndarray,
    alpha_mean: np.ndarray,
    phi_grid: np.ndarray,
    utilities: Dict[float, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(alpha_grid, alpha_mean, color="C0", lw=2.0)
    axes[0].axvline(support[0], color="black", ls=":", lw=0.9)
    axes[0].axvline(support[1], color="black", ls=":", lw=0.9)
    axes[0].axvline(
        trim_interval[0], color="darkgreen", ls="--", lw=1.0,
        label="mean hard-trim endpoints",
    )
    axes[0].axvline(trim_interval[1], color="darkgreen", ls="--", lw=1.0)
    axes[0].axhline(0.0, color="black", lw=0.5)
    axes[0].set_title(f"{sample_name}: treatment-effect nuisance")
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel(r"$\hat\alpha(\eta)$")
    axes[0].legend(fontsize=8)

    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(utilities)))
    for (cost, utility), color in zip(utilities.items(), colors):
        best = int(np.argmax(utility))
        axes[1].plot(
            phi_grid,
            utility,
            color=color,
            lw=1.7,
            label=f"c={cost:.3g}, phi*={phi_grid[best]:.3g}",
        )
        axes[1].axvline(phi_grid[best], color=color, ls="--", lw=0.7)
    axes[1].axvline(threshold, color="black", ls=":", lw=1.0)
    axes[1].axhline(0.0, color="black", lw=0.5)
    axes[1].set_title(f"{sample_name}: exact hard-trim utility")
    axes[1].set_xlabel(r"policy threshold $\phi$")
    axes[1].set_ylabel(r"$\hat U_\epsilon(\phi)$")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def perfrdd_hard_trim(
    sample: RDDSample,
    out_dir: Path,
    nuisance_support: tuple[float, float],
    *,
    eps: float = 0.1,
    c_values: Tuple[float, ...] | None = None,
    c_ratios: Tuple[float, ...] | None = None,
    phi_grid: np.ndarray | None = None,
    max_n: int | None = DEFAULT_MAX_N,
    ridge_scale: float = 0.0,
    knot_const: float = 1.0,
    knot_exponent: float = DEFAULT_KNOT_EXPONENT,
    crossfit_folds: int = 1,
    fold_seed: int = 72_931,
    write_outputs: bool = True,
    return_curves: bool = False,
) -> Dict[str, Any]:
    """Estimate the hard-trimmed policy target and write reproducible curves.

    ``nuisance_support`` is mandatory because its scientific choice cannot be
    inferred from the analysis sample while retaining the fixed-support
    interpretation of the manuscript.  A full-sample point estimate is
    requested with ``crossfit_folds=1``.  Values of two or more construct
    ordinary K-fold out-of-fold point estimates.
    """
    support = _validate_inputs(
        eps,
        nuisance_support,
        ridge_scale,
        knot_const,
        knot_exponent,
        crossfit_folds,
    )
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y = np.asarray(sample.Y, dtype=float)
    D = np.asarray(sample.D, dtype=float)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if max_n is not None and len(Y) > max_n:
        (Q, X, Y, D), n_used, _ = _subsample([Q, X, Y, D], max_n)
    else:
        n_used = len(Y)
    if n_used < 100:
        raise ValueError("hard-trim estimation requires at least 100 observations")

    direction = _detect_direction(D, Q)
    if phi_grid is None:
        phi_grid = np.linspace(
            threshold - 3.0 * float(np.std(Q)),
            threshold + 3.0 * float(np.std(Q)),
            601,
        )
    phi_grid = np.asarray(phi_grid, dtype=float)
    if phi_grid.ndim != 1 or len(phi_grid) < 3 or not np.all(np.diff(phi_grid) > 0):
        raise ValueError("phi_grid must be a strictly increasing one-dimensional array")

    alpha_grid = np.linspace(support[0], support[1], 501)
    evaluation_folds = _make_folds(n_used, crossfit_folds, fold_seed)
    all_idx = np.arange(n_used)
    fitted_folds: list[HardTrimFold] = []
    for eval_idx in evaluation_folds:
        if crossfit_folds == 1:
            train_idx = all_idx
        else:
            train_mask = np.ones(n_used, dtype=bool)
            train_mask[eval_idx] = False
            train_idx = all_idx[train_mask]
        fitted_folds.append(_fit_fold(
            Q,
            X,
            Y,
            D,
            threshold,
            eps,
            support,
            ridge_scale,
            knot_const,
            knot_exponent,
            train_idx,
            eval_idx,
            alpha_grid,
        ))

    total_weight = float(sum(np.sum(fold.hard_weights) for fold in fitted_folds))
    if total_weight < 20.0:
        raise ValueError("fewer than 20 evaluation observations survive hard trimming")
    weighted_alpha = float(sum(np.sum(
        fold.hard_weights * fold.treatment_effect
    ) for fold in fitted_folds) / total_weight)
    if c_values is None:
        if c_ratios is None:
            c_ratios = (0.0, 0.5, 1.0, 1.5)
        scale = abs(weighted_alpha) if abs(weighted_alpha) > 1e-12 else 1.0
        c_values = tuple(float(ratio) * scale for ratio in c_ratios)
    else:
        c_values = tuple(float(cost) for cost in c_values)

    utilities = _utility_curves(fitted_folds, direction, phi_grid, c_values)
    phi_star = {
        str(cost): float(phi_grid[int(np.argmax(utility))])
        for cost, utility in utilities.items()
    }
    boundary_flags = {
        str(cost): bool(np.argmax(utility) in (0, len(phi_grid) - 1))
        for cost, utility in utilities.items()
    }
    boundary_band = max(1, int(np.ceil(0.01 * (len(phi_grid) - 1))))
    near_boundary_flags = {
        str(cost): bool(
            int(np.argmax(utility)) <= boundary_band
            or int(np.argmax(utility)) >= len(phi_grid) - 1 - boundary_band
        )
        for cost, utility in utilities.items()
    }
    alpha_mean = np.mean(
        np.vstack([fold.alpha_grid for fold in fitted_folds]), axis=0
    )
    if write_outputs:
        _write_curve_csv(
            out_dir / "alpha_curve.csv",
            "eta",
            alpha_grid,
            {"alpha": alpha_mean},
        )
        _write_curve_csv(
            out_dir / "utility_curve.csv",
            "phi",
            phi_grid,
            {f"cost_{cost:g}": utility for cost, utility in utilities.items()},
        )
        _plot_diagnostics(
            out_dir,
            sample.name,
            threshold,
            support,
            (
                float(np.mean([fold.l_hat for fold in fitted_folds])),
                float(np.mean([fold.u_hat for fold in fitted_folds])),
            ),
            alpha_grid,
            alpha_mean,
            phi_grid,
            utilities,
        )

    fold_diagnostics = [{
        "l_hat": fold.l_hat,
        "u_hat": fold.u_hat,
        "lower_support_margin": fold.l_hat - support[0],
        "upper_support_margin": support[1] - fold.u_hat,
        "n_train": fold.n_train,
        "n_nuisance_fit": fold.n_fit,
        "n_nuisance_fit_treated": fold.n_fit_treated,
        "n_nuisance_fit_control": fold.n_fit_control,
        "n_eval": len(fold.eval_idx),
        "n_hard_window_eval": int(np.sum(fold.hard_weights)),
        "n_basis_treat": fold.n_basis,
        "design_rank": fold.design_rank,
        "design_columns": int(X.shape[1] + 2 * fold.n_basis),
        "design_condition_number": fold.design_condition_number,
        "first_stage_R2": fold.first_stage_R2,
    } for fold in fitted_folds]
    result: Dict[str, Any] = {
        "name": sample.name,
        "method": "perfrdd_hard_trim",
        "estimand": "exact_hard_support_trimmed",
        "point_estimation": (
            "full_sample" if crossfit_folds == 1 else "cross_fitted"
        ),
        "inference_available": False,
        "eps": float(eps),
        "n_used": int(n_used),
        "n_treated": int(D.sum()),
        "direction": direction,
        "threshold_actual": float(threshold),
        "nuisance_support": [support[0], support[1]],
        "support_is_caller_supplied": True,
        "ridge_scale": float(ridge_scale),
        "ridge_lambda_definition": "ridge_scale / sqrt(n_nuisance_fit)",
        "knot_const": float(knot_const),
        "knot_exponent": float(knot_exponent),
        "crossfit_folds": int(crossfit_folds),
        "fold_seed": int(fold_seed),
        "hard_retention": total_weight / n_used,
        "avg_alpha_hard_weighted": weighted_alpha,
        "c_values": [float(cost) for cost in c_values],
        "c_ratios": [float(ratio) for ratio in c_ratios] if c_ratios is not None else None,
        "phi_grid": [float(phi_grid[0]), float(phi_grid[-1]), int(len(phi_grid))],
        "phi_star": phi_star,
        "phi_star_at_grid_boundary": boundary_flags,
        "phi_star_near_grid_boundary": near_boundary_flags,
        "grid_boundary_band_fraction": 0.01,
        "fold_diagnostics": fold_diagnostics,
        "out_dir": str(out_dir),
    }
    if return_curves:
        result["returned_phi_grid"] = [float(value) for value in phi_grid]
        result["returned_utility_curves"] = {
            str(cost): [float(value) for value in utility]
            for cost, utility in utilities.items()
        }
    return result


def run(sample: RDDSample) -> Dict[str, Any]:
    raise ValueError(
        "perfrdd_hard_trim requires a caller-supplied nuisance_support; "
        "call perfrdd_hard_trim(sample, out_dir, nuisance_support=(lo, hi))"
    )
