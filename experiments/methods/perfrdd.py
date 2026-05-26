"""Performative-RDD plug-in estimator, generalized over RDDSample.

Steps (matches `prefRDD.tex` Section 1):
  1. First stage:   OLS of Q on [1, X] -> gamma_hat, eta_hat = Q - X @ gamma_hat.
  2. Spline basis:  cubic B-spline on eta_hat, knots ~ n_treated^(1/3).
  3. Pooled PLM:    Y = beta0 + X @ beta + Phi(eta) @ omega_base + D * Phi(eta) @ omega_treat,
                    with a small ridge penalty on the spline coefficients only;
                    beta0 (intercept) and beta (X coefs) are unpenalised so the
                    population mean of Y stays out of alpha and ω_base.
                    alpha(eta) = Phi(eta) @ omega_treat is the treatment-effect curve.
  4. Utility:       U(phi) = E[ alpha_bounded(eta) * P(Q on treated side of phi | eta) ]
                              - c * E[ P(Q on treated side of phi | eta) ]
                    using the empirical CDF of X @ gamma_hat.
  5. Output:        alpha(eta) curve, utility curve, eta histogram per dataset.

Direction handling: the "treated side" is detected from sample.D — if treated
units have lower mean Q than controls, we use Q < phi; otherwise Q > phi.

Multi-threshold (k > 1): the running variable is reduced to its first axis
(typically the primary economic threshold). Other Q columns are folded into
X. The threshold used is sample.threshold[0]. This is a simplification; a
proper k-D treatment requires a multi-dimensional density estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

from experiments._core.sample import RDDSample


# ---------------------------------------------------------------- B-spline

def _basis_params(kn: int, support: Tuple[float, float]) -> Dict[str, Any]:
    degree = 3
    lo, hi = support
    interior = np.linspace(lo, hi, kn + 2)[1:-1]
    t = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
    return {"t": t, "degree": degree, "lo": lo, "hi": hi}


def _eval_basis(pts, info) -> np.ndarray:
    pts_c = np.clip(np.asarray(pts, dtype=float), info["lo"], info["hi"])
    return BSpline.design_matrix(pts_c, info["t"], info["degree"]).toarray()


# ---------------------------------------------------------------- helpers

def _reduce_to_primary_axis(sample: RDDSample) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (Q_primary 1-D, X_with_extra_q_cols_appended, threshold_primary)."""
    if sample.Q.ndim == 1:
        return sample.Q, sample.X, float(np.atleast_1d(sample.threshold)[0])
    Q_primary = sample.Q[:, 0]
    extra_q = sample.Q[:, 1:]
    X_aug = np.hstack([sample.X, extra_q]) if extra_q.size else sample.X
    return Q_primary, X_aug, float(np.atleast_1d(sample.threshold)[0])


def _detect_direction(D: np.ndarray, Q: np.ndarray) -> str:
    """'below' means treated have Q < threshold; 'above' means Q > threshold."""
    treated = D == 1
    control = D == 0
    if not treated.any() or not control.any():
        return "above"
    return "below" if Q[treated].mean() < Q[control].mean() else "above"


def _subsample(arrays, n_max, seed=0):
    n = len(arrays[0])
    if n <= n_max:
        return arrays, n, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_max, replace=False)
    return [a[idx] for a in arrays], n_max, idx


# ---------------------------------------------------------------- core fit

@dataclass
class FitResult:
    gamma: np.ndarray
    eta: np.ndarray
    omega_base: np.ndarray
    omega_treat: np.ndarray
    beta: np.ndarray            # coefficients on X (length X.shape[1]); excludes intercept
    info: Dict[str, Any]
    eta_eval: Tuple[float, float]
    direction: str
    n: int
    n_treated: int
    intercept: float = 0.0      # PLM intercept; unpenalised, lets β/ω_base shed the constant offset of Y


def _fit_pooled_plm(
    Q: np.ndarray, X: np.ndarray, Y: np.ndarray, D: np.ndarray,
    direction: str, ridge_scale: float = 0.1,
    eta: np.ndarray | None = None,
    use_x_in_plm: bool = True,
) -> FitResult:
    """Fit the pooled-PLM after `eta` is determined.

    If `eta` is None, do a linear OLS first stage (Q on [1,X]).
    `use_x_in_plm=False` removes the X term from the PLM design — used after Y has
    been residualized nonparametrically against X.
    """
    n = len(Y)
    n_tr = int(D.sum())

    if eta is None:
        X_design = np.column_stack((np.ones(n), X))
        gamma, *_ = np.linalg.lstsq(X_design, Q, rcond=None)
        eta = Q - X_design @ gamma
    else:
        gamma = np.zeros(X.shape[1] + 1)

    # Spline support: trim 0.5%/99.5% of eta so extreme tails don't drive knots.
    support = (float(np.percentile(eta, 0.5)), float(np.percentile(eta, 99.5)))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _basis_params(kn, support)
    Phi = _eval_basis(eta, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    if use_x_in_plm:
        # Prepend an intercept so the constant offset of Y has an unpenalised home;
        # otherwise the constant leaks into the L2-penalised ω_base/ω_treat blocks
        # and inflates α̂ in treated/control-imbalanced samples.
        X_aug = np.column_stack((np.ones(n), X))
        H = np.column_stack((X_aug, Phi, DPhi))
        p = X_aug.shape[1]
    else:
        H = np.column_stack((Phi, DPhi))
        p = 0
    total = H.shape[1]

    lam = ridge_scale / np.sqrt(n)
    P = np.zeros((total, total))
    np.fill_diagonal(P[p:, p:], lam)
    coefs = np.linalg.solve(H.T @ H + n * P, H.T @ Y)

    if use_x_in_plm:
        intercept = float(coefs[0])
        beta = coefs[1:p]
    else:
        intercept = 0.0
        beta = np.zeros(X.shape[1])
    omega_base = coefs[p:p + n_basis]
    omega_treat = coefs[p + n_basis:]

    eta_treated = eta[D == 1]
    if len(eta_treated) >= 20:
        eval_lo = max(support[0], float(np.percentile(eta_treated, 5)))
        eval_hi = min(support[1], float(np.percentile(eta_treated, 95)))
    else:
        eval_lo, eval_hi = support

    return FitResult(
        gamma=gamma, eta=eta,
        omega_base=omega_base, omega_treat=omega_treat,
        beta=beta, info=info,
        eta_eval=(eval_lo, eval_hi),
        direction=direction,
        n=n, n_treated=n_tr,
        intercept=intercept,
    )


def _utility_curve(
    fit: FitResult, Q: np.ndarray, X: np.ndarray,
    phi_grid: np.ndarray, c_values: Tuple[float, ...],
) -> Dict[float, np.ndarray]:
    """U(phi) at each cost using empirical CDF of X @ gamma_hat[1:] + gamma_hat[0]."""
    eta = fit.eta
    info = fit.info

    # alpha(eta) bounded to [5th, 95th] of treated eta to avoid extrapolation.
    Phi = _eval_basis(eta, info)
    alpha_all = Phi @ fit.omega_treat
    in_supp = (eta >= fit.eta_eval[0]) & (eta <= fit.eta_eval[1])
    alpha_b = np.where(in_supp, alpha_all, 0.0)

    gX = Q - eta              # = X_design @ gamma_hat
    gX_sorted = np.sort(gX)
    n = len(gX_sorted)

    out: Dict[float, np.ndarray] = {}
    for c in c_values:
        u = np.empty(len(phi_grid))
        for j, phi in enumerate(phi_grid):
            thresh = phi - eta
            cdf = np.searchsorted(gX_sorted, thresh) / n
            # P(Q_i on treated side of phi | eta_i) — direction matters.
            probs = cdf if fit.direction == "below" else (1.0 - cdf)
            u[j] = float(np.mean(alpha_b * probs) - c * np.mean(probs))
        out[c] = u
    return out


# ---------------------------------------------------------------- plots

def _plot_eta_distribution(fit: FitResult, D: np.ndarray, out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    eta_tr = fit.eta[D == 1]
    eta_co = fit.eta[D == 0]
    ax.hist(eta_co, bins=80, density=True, alpha=0.45, color="steelblue", label=f"Control (n={len(eta_co):,})")
    ax.hist(eta_tr, bins=80, density=True, alpha=0.45, color="firebrick", label=f"Treated (n={len(eta_tr):,})")
    for x in fit.eta_eval:
        ax.axvline(x, color="green", linestyle="--", lw=1, alpha=0.7)
    ax.set_xlabel(r"$\hat\eta = Q - \hat\gamma^\top X$")
    ax.set_ylabel("density")
    ax.set_title(f"{name}: distribution of $\\hat\\eta$ by treatment status")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_alpha(fit: FitResult, out: Path, name: str) -> Tuple[float, float]:
    eta_lo, eta_hi = fit.eta_eval
    grid = np.linspace(eta_lo, eta_hi, 500)
    Phi = _eval_basis(grid, fit.info)
    alpha = Phi @ fit.omega_treat

    Phi_full = _eval_basis(np.linspace(fit.info["lo"], fit.info["hi"], 500), fit.info)
    h_base_full = Phi_full @ fit.omega_base
    h_treat_eval = Phi @ (fit.omega_base + fit.omega_treat)

    avg_alpha = float(alpha.mean())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(grid, alpha, "C0-", lw=2, label=r"$\hat\alpha(\eta)$")
    ax[0].axhline(0, color="black", lw=0.5)
    ax[0].axhline(avg_alpha, color="red", ls="--", lw=1, label=f"avg = {avg_alpha:.3g}")
    ax[0].set_xlabel(r"$\eta$")
    ax[0].set_ylabel(r"$\alpha(\eta)$")
    ax[0].set_title(f"{name}: treatment effect $\\alpha(\\eta)$")
    ax[0].legend()

    ax[1].plot(np.linspace(fit.info["lo"], fit.info["hi"], 500), h_base_full,
               "C0-", lw=1.8, label="control component")
    ax[1].plot(grid, h_treat_eval, "C3-", lw=1.8, label="treated component")
    ax[1].axvline(eta_lo, color="green", ls=":", alpha=0.5)
    ax[1].axvline(eta_hi, color="green", ls=":", alpha=0.5, label="eval region")
    ax[1].set_xlabel(r"$\eta$")
    ax[1].set_ylabel("h")
    ax[1].set_title("nonparametric components")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return avg_alpha, (eta_lo, eta_hi)


def _plot_utility(
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
                label=f"c={c:.3g}, $\\phi^*$={phi_star:.3g}")
        ax.axvline(phi_star, color=col, ls="--", lw=0.8, alpha=0.5)
    ax.axvline(threshold_actual, color="black", ls=":", lw=1.2, label=f"current $\\phi$={threshold_actual:.3g}")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel(r"threshold $\phi$")
    ax.set_ylabel(r"$\hat U(\phi)$")
    ax.set_title(f"{name}: estimated utility")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return phi_stars


# ---------------------------------------------------------------- public API

DEFAULT_MAX_N = 30_000


def _auto_c_values(avg_alpha: float) -> Tuple[float, ...]:
    """Cost grid scaled to the estimated alpha magnitude."""
    s = abs(avg_alpha) if abs(avg_alpha) > 1e-12 else 1.0
    return (0.0, 0.5 * s, 1.0 * s, 1.5 * s)


def perfrdd(
    sample: RDDSample,
    out_dir: Path,
    c_values: Tuple[float, ...] | None = None,
    phi_grid: np.ndarray | None = None,
    max_n: int | None = DEFAULT_MAX_N,
    first_stage: str = "linear",
) -> Dict[str, Any]:
    """Run the full pipeline on `sample` and save plots into `out_dir`.

    `first_stage`:
      - "linear"        : eta = Q - X @ gamma_lin (the original setting).
      - "q_nonlinear"   : eta = Q - f_hat(X), f_hat cross-fitted nonparametric.
                          Y is still treated linearly in X inside the PLM.
      - "all_nonlinear" : same eta as above; additionally Y is residualized
                          against X nonparametrically (cross-fitted), so the PLM
                          has no X term.

    Returns a JSON-serializable summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y = sample.Y
    D = sample.D
    direction = _detect_direction(D, Q)

    # Drop NaN/inf rows (some datasets have post-load issues).
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]

    # Subsample for tractable PLM fitting on very large datasets.
    if max_n is not None and len(Y) > max_n:
        (Q, X, Y, D), n_used, _ = _subsample([Q, X, Y, D], max_n)
    else:
        n_used = len(Y)

    first_stage_info: Dict[str, Any] = {"mode": first_stage}
    if first_stage == "linear":
        fit = _fit_pooled_plm(Q, X, Y, D, direction)
    else:
        from experiments.methods._nonpar import residualize
        eta_resid, _q_pred, q_info = residualize(Q, X, label="Q")
        first_stage_info["Q"] = q_info
        if first_stage == "q_nonlinear":
            fit = _fit_pooled_plm(Q, X, Y, D, direction,
                                  eta=eta_resid, use_x_in_plm=True)
        elif first_stage == "all_nonlinear":
            y_resid, _y_pred, y_info = residualize(Y, X, label="Y", seed=1)
            first_stage_info["Y"] = y_info
            fit = _fit_pooled_plm(Q, X, y_resid, D, direction,
                                  eta=eta_resid, use_x_in_plm=False)
        else:
            raise ValueError(f"unknown first_stage={first_stage!r}")

    if phi_grid is None:
        # Build a phi grid centered on the actual threshold, span ±3*std(Q).
        s = float(Q.std())
        phi_grid = np.linspace(threshold - 3 * s, threshold + 3 * s, 400)

    # Provisional alpha mean to scale costs.
    Phi_eta = _eval_basis(fit.eta, fit.info)
    alpha_all = Phi_eta @ fit.omega_treat
    in_supp = (fit.eta >= fit.eta_eval[0]) & (fit.eta <= fit.eta_eval[1])
    avg_alpha_for_c = float(np.mean(np.where(in_supp, alpha_all, 0.0)))
    if c_values is None:
        c_values = _auto_c_values(avg_alpha_for_c)

    utils = _utility_curve(fit, Q, X, phi_grid, c_values)

    _plot_eta_distribution(fit, D, out_dir / "eta_distribution.png", sample.name)
    avg_alpha, eta_eval = _plot_alpha(fit, out_dir / "alpha.png", sample.name)
    phi_stars = _plot_utility(phi_grid, utils, threshold, out_dir / "utility.png", sample.name)

    return {
        "name": sample.name,
        "n_used": int(n_used),
        "n_treated": int(D.sum()),
        "direction": direction,
        "threshold_actual": threshold,
        "eta_eval": list(eta_eval),
        "avg_alpha": avg_alpha,
        "phi_star": {str(c): phi_stars[c] for c in phi_stars},
        "first_stage_R2": float(1.0 - fit.eta.var() / Q.var()),
        "first_stage": first_stage_info,
        "out_dir": str(out_dir),
    }


# CLI-friendly alias the runner picks up automatically:
def run(sample: RDDSample) -> Dict[str, Any]:
    """Default CLI entry: writes into experiments/runs/perfrdd/<name>/."""
    out = Path(__file__).resolve().parent.parent / "runs" / "perfrdd" / sample.name
    return perfrdd(sample, out)
