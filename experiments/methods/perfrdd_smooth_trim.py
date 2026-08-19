"""Smooth implementation of the support-trimmed PerfRDD estimator.

The population target is still the hard-support criterion

    E[(alpha(eta) - c) * G_bar(phi - eta) * 1{l_0 <= eta <= u_0}].

Only its sample implementation is smoothed.  The compact, symmetric gate

    W_delta(eta; l, u)
      = H((eta-l)/delta) H((u-eta)/delta)

equals one on [l+delta, u-delta] and zero outside
[l-delta, u+delta].  Symmetry makes the leading smoothing bias cancel, so
the criterion error is O(delta^2), rather than O(delta).  This is a support
gate; it never weights observations by their propensity score.

The default rates mirror the manuscript audit:

    K_n       proportional to n^(11/60),
    delta_n   proportional to (u-l) n^(-1/3).

Thus K_n and delta_n are deliberately different tuning objects.  In
particular, delta_n is much smaller than a spline knot spacing; the gate is
evaluated analytically and is not represented by the spline basis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd import (
    DEFAULT_MAX_N,
    FitResult,
    _detect_direction,
    _eval_basis,
    _reduce_to_primary_axis,
    _subsample,
)
from experiments.methods.perfrdd_trim import (
    _basis_params_quantile,
    _compute_overlap_window,
)


DEFAULT_KNOT_EXPONENT = 11.0 / 60.0
DEFAULT_DELTA_EXPONENT = 1.0 / 3.0


def _symmetric_smooth_step(v: np.ndarray | float) -> np.ndarray:
    """C-infinity step on [-1, 1], with H(-v) = 1 - H(v).

    The logit form avoids the underflow that arises from directly evaluating
    exp(-1/z) at the two edges.
    """
    values = np.asarray(v, dtype=float)
    out = np.empty_like(values)
    out[values <= -1.0] = 0.0
    out[values >= 1.0] = 1.0
    middle = (values > -1.0) & (values < 1.0)
    if np.any(middle):
        z = (values[middle] + 1.0) / 2.0
        logit = -1.0 / z + 1.0 / (1.0 - z)
        out[middle] = expit(logit)
    return out


def _smooth_trim_weights(
    eta: np.ndarray, l_hat: float, u_hat: float, delta: float,
) -> np.ndarray:
    """Evaluate the symmetric flat-top support gate."""
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be a finite positive number")
    if not l_hat < u_hat:
        raise ValueError("overlap window must satisfy l_hat < u_hat")
    eta = np.asarray(eta, dtype=float)
    lower = _symmetric_smooth_step((eta - l_hat) / delta)
    upper = _symmetric_smooth_step((u_hat - eta) / delta)
    return lower * upper


def _select_delta(
    l_hat: float,
    u_hat: float,
    n: int,
    delta: float | None,
    delta_const: float,
    delta_exponent: float,
) -> float:
    width = float(u_hat - l_hat)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("overlap window must have positive finite width")
    if delta is not None:
        selected = float(delta)
    else:
        selected = float(delta_const * width * n ** (-delta_exponent))
    if not np.isfinite(selected) or selected <= 0.0:
        raise ValueError("selected delta must be finite and positive")
    if selected >= width / 2.0:
        raise ValueError("delta must be smaller than half the overlap-window width")
    return selected


def _fit_pooled_plm_smooth_trimmed(
    Q: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    direction: str,
    l_hat: float,
    u_hat: float,
    eta_hat: np.ndarray,
    delta: float,
    ridge_scale: float = 0.01,
    knot_const: float = 1.0,
    knot_exponent: float = DEFAULT_KNOT_EXPONENT,
    outer_buffer_ratio: float = 0.10,
) -> FitResult:
    """Fit nuisances on an open neighborhood of the target window.

    Unlike the hard-trim implementation, the spline support does not end at
    l_hat and u_hat.  This separation prevents the nuisance basis boundary
    from being confused with the target-support boundary.  The hard outer
    localization is only a numerical nuisance-fit choice; the final estimand
    is defined by the smooth gate below.
    """
    width = float(u_hat - l_hat)
    buffer = max(float(outer_buffer_ratio) * width, 2.0 * delta)
    fit_lo = float(l_hat - buffer)
    fit_hi = float(u_hat + buffer)
    in_fit = (eta_hat >= fit_lo) & (eta_hat <= fit_hi)

    X_s = X[in_fit]
    Y_s = Y[in_fit]
    D_s = D[in_fit]
    eta_s = eta_hat[in_fit]
    n_s = len(Y_s)
    n_tr_s = int(D_s.sum())
    n_co_s = n_s - n_tr_s
    if n_s < 50 or n_tr_s < 10 or n_co_s < 10:
        raise ValueError(
            "smooth-trim nuisance sample too small "
            f"(n={n_s}, treated={n_tr_s}, control={n_co_s})"
        )

    n_eff = min(n_tr_s, n_co_s)
    n_interior_knots = max(
        3, int(round(float(knot_const) * n_eff ** float(knot_exponent)))
    )
    info = _basis_params_quantile(n_interior_knots, eta_s, (fit_lo, fit_hi))
    Phi = _eval_basis(eta_s, info)
    n_basis = Phi.shape[1]

    X_aug = np.column_stack((np.ones(n_s), X_s))
    H = np.column_stack((X_aug, Phi, D_s[:, None] * Phi))
    p = X_aug.shape[1]
    penalty = np.zeros((H.shape[1], H.shape[1]))
    np.fill_diagonal(penalty[p:, p:], ridge_scale / np.sqrt(n_s))
    lhs = H.T @ H + n_s * penalty
    rhs = H.T @ Y_s
    try:
        coefs = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    return FitResult(
        gamma=np.zeros(X.shape[1] + 1),
        eta=eta_hat,
        omega_base=coefs[p:p + n_basis],
        omega_treat=coefs[p + n_basis:],
        beta=coefs[1:p],
        info=info,
        eta_eval=(float(l_hat), float(u_hat)),
        direction=direction,
        n=len(Y),
        n_treated=int(D.sum()),
        intercept=float(coefs[0]),
    )


def _utility_curve_smooth_trimmed(
    fit: FitResult,
    Q: np.ndarray,
    phi_grid: np.ndarray,
    c_values: Tuple[float, ...],
    delta: float,
) -> Dict[float, np.ndarray]:
    """Compute the smoothly implemented hard-support utility.

    Dividing by sum(W_delta) only puts utilities on a conditional-mean scale.
    The denominator does not depend on phi, so it cannot change the estimated
    optimal policy.
    """
    eta = fit.eta
    l_hat, u_hat = fit.eta_eval
    weights = _smooth_trim_weights(eta, l_hat, u_hat, delta)
    active = weights > 0.0
    if not np.any(active):
        raise ValueError("smooth support gate has no positive-weight observations")

    eta_a = eta[active]
    weights_a = weights[active]
    alpha_a = _eval_basis(eta_a, fit.info) @ fit.omega_treat
    normalizer = float(weights_a.sum())
    t_sorted = np.sort(Q - eta)
    n = len(t_sorted)

    out: Dict[float, np.ndarray] = {}
    for c in c_values:
        utility = np.empty(len(phi_grid))
        value = weights_a * (alpha_a - c)
        for j, phi in enumerate(phi_grid):
            cdf = np.searchsorted(t_sorted, phi - eta_a) / n
            probs = cdf if fit.direction == "below" else (1.0 - cdf)
            utility[j] = float(np.sum(value * probs) / normalizer)
        out[float(c)] = utility
    return out


def _plot_support_gate(
    eta: np.ndarray,
    D: np.ndarray,
    l_hat: float,
    u_hat: float,
    delta: float,
    out: Path,
    name: str,
    eps: float,
) -> None:
    grid = np.linspace(l_hat - 1.5 * delta, u_hat + 1.5 * delta, 800)
    gate = _smooth_trim_weights(grid, l_hat, u_hat, delta)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(eta[D == 0], bins=80, density=True, alpha=0.30, color="steelblue", label="control eta")
    ax.hist(eta[D == 1], bins=80, density=True, alpha=0.30, color="firebrick", label="treated eta")
    ax.set_xlabel(r"$\hat\eta$")
    ax.set_ylabel("eta density")
    ax2 = ax.twinx()
    ax2.plot(grid, gate, color="darkgreen", lw=2.0, label=r"$W_{\delta}(\eta;\hat l,\hat u)$")
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_ylabel("support weight")
    ax.axvline(l_hat, color="darkgreen", ls="--", lw=1.0)
    ax.axvline(u_hat, color="darkgreen", ls="--", lw=1.0)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    ax.set_title(f"{name}: symmetric smooth support trim (epsilon={eps:g}, delta={delta:.3g})")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_alpha(
    fit: FitResult, out: Path, name: str,
) -> float:
    l_hat, u_hat = fit.eta_eval
    grid = np.linspace(l_hat, u_hat, 500)
    alpha = _eval_basis(grid, fit.info) @ fit.omega_treat
    avg_alpha = float(alpha.mean())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, alpha, color="C0", lw=2.0, label=r"$\hat\alpha(\eta)$")
    ax.axvline(l_hat, color="darkgreen", ls="--", lw=1.0)
    ax.axvline(u_hat, color="darkgreen", ls="--", lw=1.0)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\alpha(\eta)$")
    ax.set_title(f"{name}: nuisance fit on an open neighborhood of the target support")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return avg_alpha


def _plot_utility(
    phi_grid: np.ndarray,
    utilities: Dict[float, np.ndarray],
    threshold: float,
    out: Path,
    name: str,
) -> Dict[float, float]:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(utilities)))
    phi_stars: Dict[float, float] = {}
    for (cost, utility), color in zip(utilities.items(), colors):
        idx = int(np.argmax(utility))
        phi_star = float(phi_grid[idx])
        phi_stars[cost] = phi_star
        ax.plot(phi_grid, utility, color=color, lw=1.8, label=f"c={cost:.3g}, phi*={phi_star:.3g}")
        ax.axvline(phi_star, color=color, ls="--", lw=0.8, alpha=0.55)
    ax.axvline(threshold, color="black", ls=":", lw=1.2, label=f"current phi={threshold:.3g}")
    ax.axhline(0.0, color="black", lw=0.4)
    ax.set_xlabel(r"threshold $\phi$")
    ax.set_ylabel(r"$\hat U_{\epsilon,\delta}(\phi)$")
    ax.set_title(f"{name}: smoothly implemented support-trimmed utility")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return phi_stars


def perfrdd_smooth_trim(
    sample: RDDSample,
    out_dir: Path,
    eps: float = 0.1,
    c_values: Tuple[float, ...] | None = None,
    c_ratios: Tuple[float, ...] | None = None,
    phi_grid: np.ndarray | None = None,
    max_n: int | None = DEFAULT_MAX_N,
    delta: float | None = None,
    delta_const: float = 1.0,
    delta_exponent: float = DEFAULT_DELTA_EXPONENT,
    knot_const: float = 1.0,
    knot_exponent: float = DEFAULT_KNOT_EXPONENT,
    outer_buffer_ratio: float = 0.10,
) -> Dict[str, Any]:
    """Estimate the hard-support policy using a vanishing smooth gate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y = sample.Y
    D = sample.D
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if max_n is not None and len(Y) > max_n:
        (Q, X, Y, D), n_used, _ = _subsample([Q, X, Y, D], max_n)
    else:
        n_used = len(Y)

    direction = _detect_direction(D, Q)
    l_hat, u_hat, eta_hat, _ = _compute_overlap_window(Q, X, threshold, eps)
    selected_delta = _select_delta(
        l_hat, u_hat, n_used, delta, delta_const, delta_exponent
    )
    weights = _smooth_trim_weights(eta_hat, l_hat, u_hat, selected_delta)
    hard = (eta_hat >= l_hat) & (eta_hat <= u_hat)
    core = (eta_hat >= l_hat + selected_delta) & (eta_hat <= u_hat - selected_delta)

    fit = _fit_pooled_plm_smooth_trimmed(
        Q, X, Y, D, direction, l_hat, u_hat, eta_hat, selected_delta,
        knot_const=knot_const,
        knot_exponent=knot_exponent,
        outer_buffer_ratio=outer_buffer_ratio,
    )
    alpha_all = _eval_basis(eta_hat, fit.info) @ fit.omega_treat
    weight_sum = float(weights.sum())
    avg_alpha_for_c = float(np.sum(weights * alpha_all) / weight_sum)
    if c_values is None:
        if c_ratios is None:
            c_ratios = (0.0, 0.5, 1.0, 1.5)
        scale = abs(avg_alpha_for_c) if abs(avg_alpha_for_c) > 1e-12 else 1.0
        c_values = tuple(float(r) * scale for r in c_ratios)
    else:
        c_values = tuple(float(c) for c in c_values)

    if phi_grid is None:
        q_scale = float(Q.std())
        phi_grid = np.linspace(threshold - 3.0 * q_scale, threshold + 3.0 * q_scale, 400)
    utilities = _utility_curve_smooth_trimmed(
        fit, Q, np.asarray(phi_grid), c_values, selected_delta
    )

    _plot_support_gate(
        eta_hat, D, l_hat, u_hat, selected_delta,
        out_dir / "smooth_support_gate.png", sample.name, eps,
    )
    avg_alpha_grid = _plot_alpha(fit, out_dir / "alpha.png", sample.name)
    phi_stars = _plot_utility(
        np.asarray(phi_grid), utilities, threshold,
        out_dir / "utility.png", sample.name,
    )
    phi_star_at_grid_boundary = {
        str(cost): bool(np.argmax(utility) in (0, len(phi_grid) - 1))
        for cost, utility in utilities.items()
    }

    return {
        "name": sample.name,
        "method": "perfrdd_smooth_trim",
        "estimand": "hard_support_trimmed",
        "implementation": "symmetric_C_infinity_support_gate",
        "propensity_weighted": False,
        "eps": float(eps),
        "n_used": int(n_used),
        "n_treated": int(D.sum()),
        "n_hard_window": int(hard.sum()),
        "n_core": int(core.sum()),
        "n_positive_weight": int((weights > 0.0).sum()),
        "effective_weight_sum": weight_sum,
        "effective_minus_hard_mass": float(weight_sum - hard.sum()),
        "relative_effective_minus_hard_mass": float(
            (weight_sum - hard.sum()) / max(int(hard.sum()), 1)
        ),
        "effective_treated_weight_sum": float(weights[D == 1].sum()),
        "effective_control_weight_sum": float(weights[D == 0].sum()),
        "direction": direction,
        "threshold_actual": float(threshold),
        "l_hat": float(l_hat),
        "u_hat": float(u_hat),
        "window_width": float(u_hat - l_hat),
        "delta": float(selected_delta),
        "delta_const": float(delta_const),
        "delta_exponent": float(delta_exponent),
        "sqrt_n_delta_squared": float(np.sqrt(n_used) * selected_delta ** 2 / max((u_hat - l_hat) ** 2, 1e-30)),
        "sqrt_n_delta": float(np.sqrt(n_used) * selected_delta / max(u_hat - l_hat, 1e-30)),
        "knot_const": float(knot_const),
        "knot_exponent": float(knot_exponent),
        "n_basis_treat": int(fit.omega_treat.shape[0]),
        "K_delta_scaled": float(fit.omega_treat.shape[0] * selected_delta / (u_hat - l_hat)),
        "outer_support": [float(fit.info["lo"]), float(fit.info["hi"])],
        "avg_alpha_smooth_weighted": avg_alpha_for_c,
        "avg_alpha_grid": avg_alpha_grid,
        "c_values": [float(c) for c in c_values],
        "c_ratios": [float(r) for r in c_ratios] if c_ratios is not None else None,
        "phi_star": {str(c): phi_stars[c] for c in phi_stars},
        "phi_star_at_grid_boundary": phi_star_at_grid_boundary,
        "first_stage_R2": float(1.0 - eta_hat.var() / Q.var()),
        "out_dir": str(out_dir),
    }


def run(sample: RDDSample) -> Dict[str, Any]:
    out = Path(__file__).resolve().parent.parent / "runs" / "perfrdd_smooth_trim" / sample.name
    return perfrdd_smooth_trim(sample, out)
