"""Trimmed (overlap-restricted) Performative-RDD estimator.

Implements Theorem 2 of `prefRDD.tex` Section 3: the trimmed estimand
    phi*_eps := argmax_phi E[(alpha(eta) - c) * G_bar(phi - eta) * 1{eta in [l_0, u_0]}]
where [l_0, u_0] = { eta : G_bar(phi_0 - eta) in [eps, 1-eps] } is the overlap
window determined by quantiles of T = gamma^T X.

Differences from the standard `perfrdd.py` estimator:
  - The overlap window [l_hat, u_hat] is computed from the eps and (1-eps)
    quantiles of X @ gamma_hat, then the PLM regression, the spline basis,
    and the utility average are all restricted to observations with
    eta_hat in [l_hat, u_hat].
  - The "eta_eval" interval used by the standard estimator (5th/95th
    percentile of TREATED eta) is replaced by the overlap-driven
    [l_hat, u_hat], which is a more principled trimming.

For point-estimation comparison purposes, the boundary (l_hat, u_hat) is
computed from the FULL sample. The auxiliary boundary fold described in
Section 3.1 is only needed for the boundary-CLT variance term (Theorem 2's
(1/c)(B_l, B_u) Sigma_bdry (...) piece), which doesn't affect the point
estimate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd import (
    _basis_params,
    _eval_basis,
    _reduce_to_primary_axis,
    _detect_direction,
    _subsample,
    _auto_c_values,
    FitResult,
    DEFAULT_MAX_N,
)


# ---------------------------------------------------------------- knot placement

def _basis_params_quantile(kn: int, eta_data: np.ndarray, support) -> Dict[str, Any]:
    """Cubic B-spline knot vector with interior knots at QUANTILES of eta_data
    (rather than uniform on the support). Concentrating knots where the data is
    dense avoids spending degrees of freedom on near-empty sub-regions — which,
    on a trimmed window whose edges have a sparse minority group, is what
    produced the boundary wiggle under uniform placement.
    """
    degree = 3
    lo, hi = support
    if kn <= 0:
        interior = np.array([])
    else:
        qs = np.linspace(0.0, 1.0, kn + 2)[1:-1]
        interior = np.quantile(eta_data, qs)
        # keep knots strictly interior and strictly increasing
        interior = np.clip(interior, lo + 1e-9, hi - 1e-9)
        interior = np.unique(interior)
    t = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
    return {"t": t, "degree": degree, "lo": lo, "hi": hi}


# ---------------------------------------------------------------- trimming

def _compute_overlap_window(
    Q: np.ndarray, X: np.ndarray, threshold: float, eps: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute (l_hat, u_hat, eta_hat, T_hat) using the full sample.

    l_hat = phi_0 - Q_{1-eps}(T_hat), u_hat = phi_0 - Q_eps(T_hat).
    Both formulas are direction-independent: the overlap region
    {eta : e_phi(eta) in [eps, 1-eps]} is the same interval whether
    "treated" means Q > phi or Q < phi.
    """
    n = len(Q)
    X_design = np.column_stack((np.ones(n), X))
    gamma, *_ = np.linalg.lstsq(X_design, Q, rcond=None)
    T_hat = X_design @ gamma
    eta_hat = Q - T_hat

    q_lo = float(np.quantile(T_hat, eps))
    q_hi = float(np.quantile(T_hat, 1.0 - eps))
    l_hat = threshold - q_hi
    u_hat = threshold - q_lo
    return l_hat, u_hat, eta_hat, T_hat


# ---------------------------------------------------------------- core fit

def _fit_pooled_plm_trimmed(
    Q: np.ndarray, X: np.ndarray, Y: np.ndarray, D: np.ndarray,
    direction: str, l_hat: float, u_hat: float, eta_hat: np.ndarray,
    ridge_scale: float = 0.1,
    knot_const: float = 0.5,
) -> FitResult:
    """Same shape as `_fit_pooled_plm` in perfrdd.py, but operating on the
    overlap-restricted subsample. The spline basis is supported on
    [l_hat, u_hat] (not on a quantile-based interval of all eta), and the
    OLS uses only observations with eta in [l_hat, u_hat].

    Knot count uses the undersmoothing rate from (A5): kn = knot_const * n^{1/5},
    floored at 3. The base sample n is the MINORITY
    group within the window, min(n_treated, n_control): alpha(eta) is the
    treated-vs-control contrast, so its resolution is limited by whichever group
    is scarcer (near the window edges the minority fraction is ~eps by
    construction). Interior knots are placed at QUANTILES of the in-window eta
    (see _basis_params_quantile), not uniformly, so they track the data density.
    The asymptotic rate fixes only the exponent; knot_const sets the constant.
    """
    n = len(Y)
    n_design = X.shape[1] + 1

    in_window = (eta_hat >= l_hat) & (eta_hat <= u_hat)
    Q_s, X_s, Y_s, D_s, eta_s = (
        Q[in_window], X[in_window], Y[in_window], D[in_window], eta_hat[in_window]
    )
    n_s = len(Y_s)
    n_tr_s = int(D_s.sum())
    n_co_s = n_s - n_tr_s
    if n_s < 50 or n_tr_s < 10 or n_co_s < 10:
        raise ValueError(
            f"trimmed sample too small (n={n_s}, treated={n_tr_s}, control={n_co_s}) — "
            "consider a smaller eps or a larger dataset"
        )

    # Spline basis directly on [l_hat, u_hat]. Count scales with the minority
    # group (the binding constraint for the treated-vs-control contrast); knots
    # are placed at in-window eta quantiles to track density.
    support = (l_hat, u_hat)
    n_eff = min(n_tr_s, n_co_s)
    kn = max(2, int(round(knot_const * n_eff ** (1.0 / 5.0))))
    info = _basis_params_quantile(kn, eta_s, support)
    Phi = _eval_basis(eta_s, info)
    n_basis = Phi.shape[1]

    DPhi = D_s[:, None] * Phi
    X_aug = np.column_stack((np.ones(n_s), X_s))
    H = np.column_stack((X_aug, Phi, DPhi))
    p = X_aug.shape[1]
    total = H.shape[1]

    lam = ridge_scale / np.sqrt(n_s)
    P = np.zeros((total, total))
    np.fill_diagonal(P[p:, p:], lam)
    coefs = np.linalg.solve(H.T @ H + n_s * P, H.T @ Y_s)

    intercept = float(coefs[0])
    beta = coefs[1:p]
    omega_base = coefs[p:p + n_basis]
    omega_treat = coefs[p + n_basis:]

    # We expose eta_hat for the FULL sample (so utility evaluation can use it),
    # but the regression was fit only on the in-window subsample.
    gamma_full = np.zeros(n_design)  # not used downstream — placeholder
    return FitResult(
        gamma=gamma_full, eta=eta_hat,
        omega_base=omega_base, omega_treat=omega_treat,
        beta=beta, info=info,
        eta_eval=(l_hat, u_hat),
        direction=direction,
        n=n, n_treated=int(D.sum()),
        intercept=intercept,
    )


def _utility_curve_trimmed(
    fit: FitResult, Q: np.ndarray,
    phi_grid: np.ndarray, c_values: Tuple[float, ...],
) -> Dict[float, np.ndarray]:
    """U_eps(phi) = (1/n_in) sum_{i: eta_i in [l_hat, u_hat]} (alpha(eta_i) - c) * P(D | eta_i),
    where the sample is restricted to the overlap window and probabilities
    P(Q on treated side of phi | eta) use the empirical CDF of T_hat = Q - eta.
    """
    eta = fit.eta
    info = fit.info
    l_hat, u_hat = fit.eta_eval

    in_window = (eta >= l_hat) & (eta <= u_hat)
    eta_in = eta[in_window]
    Q_in = Q[in_window]
    T_in = Q_in - eta_in   # = X_i @ gamma_hat, restricted to window

    # alpha(eta) at the in-window observations.
    Phi_in = _eval_basis(eta_in, info)
    alpha_in = Phi_in @ fit.omega_treat

    # CDF of T_hat estimated on the FULL sample (more accurate than the
    # restricted one, and doesn't depend on the indicator).
    gX_sorted = np.sort(Q - eta)
    n_full = len(gX_sorted)

    n_in = len(eta_in)
    out: Dict[float, np.ndarray] = {}
    for c in c_values:
        u = np.empty(len(phi_grid))
        for j, phi in enumerate(phi_grid):
            thresh = phi - eta_in
            cdf = np.searchsorted(gX_sorted, thresh) / n_full
            probs = cdf if fit.direction == "below" else (1.0 - cdf)
            # Average over the in-window sample only; normalize by n_in.
            u[j] = float(np.mean(alpha_in * probs) - c * np.mean(probs))
        out[c] = u
    return out


# ---------------------------------------------------------------- plots

def _plot_eta_window(eta: np.ndarray, D: np.ndarray, l_hat: float, u_hat: float,
                    out: Path, name: str, eps: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(eta[D == 0], bins=80, density=True, alpha=0.45, color="steelblue",
            label=f"Control (n={int((D==0).sum()):,})")
    ax.hist(eta[D == 1], bins=80, density=True, alpha=0.45, color="firebrick",
            label=f"Treated (n={int((D==1).sum()):,})")
    ax.axvline(l_hat, color="darkgreen", lw=1.6, label=fr"$\hat l$, $\hat u$ ($\epsilon$={eps})")
    ax.axvline(u_hat, color="darkgreen", lw=1.6)
    ax.set_xlabel(r"$\hat\eta = Q - \hat\gamma^\top X$")
    ax.set_ylabel("density")
    ax.set_title(f"{name}: overlap window for trimmed estimator")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_alpha_trimmed(fit: FitResult, out: Path, name: str) -> float:
    eta_lo, eta_hi = fit.eta_eval
    grid = np.linspace(eta_lo, eta_hi, 500)
    Phi = _eval_basis(grid, fit.info)
    alpha = Phi @ fit.omega_treat
    avg_alpha = float(alpha.mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, alpha, "C0-", lw=2, label=r"$\hat\alpha_\epsilon(\eta)$")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(avg_alpha, color="red", ls="--", lw=1, label=f"avg = {avg_alpha:.3g}")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\alpha(\eta)$")
    ax.set_title(f"{name}: trimmed treatment effect "
                 fr"$\hat\alpha_\epsilon(\eta)$ on $[\hat l, \hat u]$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return avg_alpha


def _plot_utility_trimmed(
    phi_grid: np.ndarray, utils: Dict[float, np.ndarray],
    threshold_actual: float, out: Path, name: str,
) -> Dict[float, float]:
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.85, len(utils)))
    phi_stars: Dict[float, float] = {}
    for (c, u), col in zip(utils.items(), colors):
        idx = int(np.argmax(u))
        phi_star = float(phi_grid[idx])
        phi_stars[c] = phi_star
        ax.plot(phi_grid, u, color=col, lw=1.8,
                label=f"c={c:.3g}, $\\phi^*_\\epsilon$={phi_star:.3g}")
        ax.axvline(phi_star, color=col, ls="--", lw=0.8, alpha=0.5)
    ax.axvline(threshold_actual, color="black", ls=":", lw=1.2,
               label=f"current $\\phi$={threshold_actual:.3g}")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel(r"threshold $\phi$")
    ax.set_ylabel(r"$\hat U_\epsilon(\phi)$")
    ax.set_title(f"{name}: trimmed estimated utility")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return phi_stars


# ---------------------------------------------------------------- public API

def perfrdd_trim(
    sample: RDDSample,
    out_dir: Path,
    eps: float = 0.1,
    c_values: Tuple[float, ...] | None = None,
    c_ratios: Tuple[float, ...] | None = None,
    phi_grid: np.ndarray | None = None,
    max_n: int | None = DEFAULT_MAX_N,
    knot_const: float = 0.5,
) -> Dict[str, Any]:
    """Trimmed-estimator analog of `perfrdd`. Returns a JSON-serializable summary.

    Cost-grid calibration:
      - `c_values` given        -> use those absolute costs verbatim.
      - else `c_ratios` given   -> cost grid = c_ratios * |in-window avg alpha|.
                                   This is the recommended choice: it scales the
                                   cost to the TRIMMED treatment-effect magnitude,
                                   so phi*(c) shows the estimator's true cost
                                   sensitivity rather than being flattened by a
                                   grid calibrated to a (possibly overlap-biased)
                                   standard avg alpha.
      - else                    -> default ratios (0, 0.5, 1, 1.5) * |trim avg alpha|.

    `eps` is the propensity-trimming parameter from Section 3 of the paper.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y = sample.Y
    D = sample.D
    direction = _detect_direction(D, Q)

    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]

    if max_n is not None and len(Y) > max_n:
        (Q, X, Y, D), n_used, _ = _subsample([Q, X, Y, D], max_n)
    else:
        n_used = len(Y)

    # Step 1: compute the overlap window from the full sample.
    l_hat, u_hat, eta_hat, T_hat = _compute_overlap_window(Q, X, threshold, eps)
    in_window = (eta_hat >= l_hat) & (eta_hat <= u_hat)
    n_in = int(in_window.sum())
    n_tr_in = int(((D == 1) & in_window).sum())
    n_co_in = int(((D == 0) & in_window).sum())
    propensity_in = n_tr_in / n_in if n_in else float("nan")

    # Step 2: fit the PLM restricted to the overlap window.
    fit = _fit_pooled_plm_trimmed(Q, X, Y, D, direction, l_hat, u_hat, eta_hat,
                                  knot_const=knot_const)
    n_basis_trt = int(fit.omega_treat.shape[0])

    # Build phi grid centered on the actual threshold, span ±3 std(Q).
    if phi_grid is None:
        s = float(Q.std())
        phi_grid = np.linspace(threshold - 3 * s, threshold + 3 * s, 400)

    # In-window average alpha used to scale the cost grid.
    Phi_in = _eval_basis(eta_hat[in_window], fit.info)
    alpha_in = Phi_in @ fit.omega_treat
    avg_alpha_for_c = float(alpha_in.mean()) if len(alpha_in) else 0.0
    if c_values is None:
        if c_ratios is None:
            c_ratios = (0.0, 0.5, 1.0, 1.5)
        s = abs(avg_alpha_for_c) if abs(avg_alpha_for_c) > 1e-12 else 1.0
        c_values = tuple(float(r) * s for r in c_ratios)

    utils = _utility_curve_trimmed(fit, Q, phi_grid, c_values)

    _plot_eta_window(eta_hat, D, l_hat, u_hat,
                     out_dir / "eta_window.png", sample.name, eps)
    avg_alpha = _plot_alpha_trimmed(fit, out_dir / "alpha.png", sample.name)
    phi_stars = _plot_utility_trimmed(
        phi_grid, utils, threshold, out_dir / "utility.png", sample.name
    )

    return {
        "name": sample.name,
        "method": "perfrdd_trim",
        "eps": eps,
        "n_used": int(n_used),
        "n_treated": int(D.sum()),
        "n_in_window": n_in,
        "n_treated_in_window": n_tr_in,
        "n_control_in_window": n_co_in,
        "propensity_in_window": propensity_in,
        "direction": direction,
        "threshold_actual": threshold,
        "l_hat": float(l_hat),
        "u_hat": float(u_hat),
        "knot_const": float(knot_const),
        "n_basis_treat": n_basis_trt,
        "avg_alpha_trimmed": avg_alpha,
        "avg_alpha_for_c": avg_alpha_for_c,
        "c_values": [float(c) for c in c_values],
        "c_ratios": [float(r) for r in c_ratios] if c_ratios is not None else None,
        "phi_star": {str(c): phi_stars[c] for c in phi_stars},
        "first_stage_R2": float(1.0 - eta_hat.var() / Q.var()),
        "out_dir": str(out_dir),
    }


def run(sample: RDDSample) -> Dict[str, Any]:
    out = Path(__file__).resolve().parent.parent / "runs" / "perfrdd_trim" / sample.name
    return perfrdd_trim(sample, out)
