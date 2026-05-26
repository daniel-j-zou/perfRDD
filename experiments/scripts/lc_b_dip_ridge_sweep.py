"""Sweep the ridge penalty on ω_base for lending_club and see whether the
η ≈ 15–20 dip in b̂(η) flattens.

Per fit:
    lam_base  = ridge_base  / sqrt(n)          # applied to ω_base only
    lam_treat = ridge_treat / sqrt(n)           # applied to ω_treat only
    coefs = (HᵀH + n · diag([0…0, lam_base…, lam_treat…]))⁻¹ Hᵀ Y

Reports per-fit RMSE between b̂(η) and the empirical bin-mean of Y − X·β̂ for
controls (using the same fit's β̂), restricted to bins with ≥5 controls and
η ∈ [-15, 15] where controls are dense.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._core.registry import load
from experiments.methods.perfrdd import (
    DEFAULT_MAX_N,
    _basis_params,
    _detect_direction,
    _eval_basis,
    _reduce_to_primary_axis,
    _subsample,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "lc_b_dip_ridge_sweep.png"

RIDGE_BASE_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]   # default is 0.1
RIDGE_TREAT = 0.1


def fit_split_ridge(Q, X, Y, D, ridge_base, ridge_treat):
    n = len(Y)
    n_tr = int(D.sum())
    X_design = np.column_stack((np.ones(n), X))
    gamma, *_ = np.linalg.lstsq(X_design, Q, rcond=None)
    eta = Q - X_design @ gamma

    support = (float(np.percentile(eta, 0.5)), float(np.percentile(eta, 99.5)))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _basis_params(kn, support)
    Phi = _eval_basis(eta, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]
    total = H.shape[1]

    lam_base = ridge_base / np.sqrt(n)
    lam_treat = ridge_treat / np.sqrt(n)
    P = np.zeros((total, total))
    np.fill_diagonal(P[p:p + n_basis, p:p + n_basis], lam_base)
    np.fill_diagonal(P[p + n_basis:, p + n_basis:], lam_treat)

    coefs = np.linalg.solve(H.T @ H + n * P, H.T @ Y)
    beta = coefs[:p]
    omega_base = coefs[p:p + n_basis]
    omega_treat = coefs[p + n_basis:]

    eta_treated = eta[D == 1]
    if len(eta_treated) >= 20:
        eval_lo = max(support[0], float(np.percentile(eta_treated, 5)))
        eval_hi = min(support[1], float(np.percentile(eta_treated, 95)))
    else:
        eval_lo, eval_hi = support

    return {
        "eta": eta, "info": info, "beta": beta,
        "omega_base": omega_base, "omega_treat": omega_treat,
        "eta_eval": (eval_lo, eval_hi),
    }


def bin_means(eta, Y_resid, edges, min_per_bin=5):
    n_bins = len(edges) - 1
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (eta >= edges[i]) & (eta < edges[i + 1])
        if sel.sum() >= min_per_bin:
            means[i] = Y_resid[sel].mean()
    return means


def main() -> None:
    sample = load("lending_club")
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y, D = sample.Y, sample.D
    direction = _detect_direction(D, Q)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if len(Y) > DEFAULT_MAX_N:
        (Q, X, Y, D), _, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)

    fits = {}
    for rb in RIDGE_BASE_VALUES:
        fits[rb] = fit_split_ridge(Q, X, Y, D, rb, RIDGE_TREAT)

    # Common η grid for plotting and bin edges for diagnostics (use the lowest
    # ridge fit's spline support; all fits share the same info since support
    # is data-driven and identical).
    info = fits[RIDGE_BASE_VALUES[0]]["info"]
    eta_baseline = fits[RIDGE_BASE_VALUES[0]]["eta"]
    grid = np.linspace(info["lo"], info["hi"], 500)

    bin_edges = np.linspace(float(eta_baseline.min()),
                            float(eta_baseline.max()), 61)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Eval ridge-base fits: per-fit Y_resid and per-fit bin means for controls.
    co_mask = D == 0
    eta = eta_baseline
    eta_co = eta[co_mask]

    # Compute RMSE between b̂ on bin centers vs control bin means, restricted
    # to η ∈ [-15, 15] where controls are dense and ≥5 obs/bin.
    rmse_window = (-15.0, 15.0)

    report_rows = []
    fig, ax = plt.subplots(2, 1, figsize=(12, 8),
                           gridspec_kw={"height_ratios": [3.0, 1.0]},
                           sharex=True)

    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(RIDGE_BASE_VALUES)))

    # bottom: η density
    bw = bin_edges[1] - bin_edges[0]
    counts_co, _ = np.histogram(eta_co, bins=bin_edges)
    counts_tr, _ = np.histogram(eta[D == 1], bins=bin_edges)
    ax[1].bar(bin_centers, counts_co, width=bw, color="steelblue",
              alpha=0.65, label=f"control (n={counts_co.sum():,})")
    ax[1].bar(bin_centers, counts_tr, width=bw, bottom=counts_co,
              color="firebrick", alpha=0.65,
              label=f"treated (n={counts_tr.sum():,})")
    ax[1].axvspan(13.5, 19.5, color="orange", alpha=0.12)
    ax[1].set_xlabel(r"$\eta$")
    ax[1].set_ylabel("count")
    ax[1].legend(loc="upper left", fontsize=9)

    for col, rb in zip(colors, RIDGE_BASE_VALUES):
        f = fits[rb]
        b_grid = _eval_basis(grid, f["info"]) @ f["omega_base"]
        Y_resid = Y - X @ f["beta"]
        means_co = bin_means(eta_co, Y_resid[co_mask], bin_edges, min_per_bin=5)

        # RMSE in dense window.
        in_window = ((bin_centers >= rmse_window[0])
                     & (bin_centers <= rmse_window[1]))
        valid = in_window & ~np.isnan(means_co)
        b_at_centers = _eval_basis(bin_centers, f["info"]) @ f["omega_base"]
        rmse = float(np.sqrt(np.mean((b_at_centers[valid] - means_co[valid]) ** 2)))

        # Also report b̂ at the dip mid-point and at η=10 to characterize shape.
        b_at_10 = float(_eval_basis(np.array([10.0]), f["info"]) @ f["omega_base"])
        b_at_18 = float(_eval_basis(np.array([18.0]), f["info"]) @ f["omega_base"])
        report_rows.append((rb, rmse, b_at_10, b_at_18, b_at_10 - b_at_18))

        label = (rf"$\rho_b={rb:g}$   "
                 rf"RMSE={rmse:.3f}   "
                 rf"$\hat b(10)-\hat b(18)$={b_at_10 - b_at_18:+.3f}")
        ax[0].plot(grid, b_grid, color=col, lw=2.0, label=label)
        # Only plot empirical points once (they're tiny shifts across fits).
        if rb == RIDGE_BASE_VALUES[0]:
            ax[0].errorbar(bin_centers, means_co, fmt="o", color="black",
                           ms=3.5, alpha=0.7,
                           label=r"control bin-mean $Y-X\hat\beta$  ($\rho_b=0.1$ fit)")

    ax[0].axvspan(13.5, 19.5, color="orange", alpha=0.12, label="dip region")
    ax[0].set_ylabel(r"$\hat b(\eta)$  /  $Y - X\hat\beta$")
    ax[0].set_title(r"Lending Club: sweep of ridge on $\omega_{\mathrm{base}}$ "
                    r"(default $\rho_b=0.1$, fixed $\rho_{\mathrm{treat}}=0.1$)")
    ax[0].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'ridge_base':>10}  {'RMSE (η∈[-15,15])':>18}  "
          f"{'b̂(10)':>9}  {'b̂(18)':>9}  {'b̂(10)-b̂(18)':>15}")
    for rb, rmse, b10, b18, gap in report_rows:
        print(f"{rb:>10.2g}  {rmse:>18.3f}  {b10:>9.3f}  {b18:>9.3f}  {gap:>+15.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
