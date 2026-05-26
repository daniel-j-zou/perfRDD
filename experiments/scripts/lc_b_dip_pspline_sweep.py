"""P-spline (second-difference) penalty on ω_base for lending_club.

Replaces the L2 penalty on ω_base with λ · ‖D₂ ω_base‖², where D₂ is the
discrete second-difference operator. The L2 penalty on ω_treat is kept at
its original scale (ridge_treat = 0.1, applied as ridge_treat·√n · I).

Solve:
    (HᵀH + Penalty) coef = Hᵀ Y
    Penalty = block_diag(0_X,  λ·D₂ᵀD₂ + ε·I,  ridge_treat·√n · I)

ε is a tiny stabilising ridge on ω_base so the system stays well-posed when λ
is very small.

Reports per-fit RMSE between b̂(η) and the control bin-mean of Y − X·β̂ over
the dense region η ∈ [-15, 15], same as the L2 sweep, so the two sweeps are
directly comparable.
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
OUT = ROOT / "runs" / "lc_b_dip_pspline_sweep.png"

LAMBDAS = [0.0, 1.0, 1e2, 1e4, 1e6, 1e8]
RIDGE_TREAT = 0.1
EPS_RIDGE_BASE = 1e-6   # tiny L2 floor on ω_base for numerical stability


def second_diff_matrix(m: int) -> np.ndarray:
    """(m-2) × m discrete second-difference operator."""
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def fit_pspline_base(Q, X, Y, D, lam_smooth_base, ridge_treat=RIDGE_TREAT):
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

    D2 = second_diff_matrix(n_basis)
    DtD = D2.T @ D2

    P = np.zeros((total, total))
    # Smoothness + tiny stabilising ridge on ω_base
    P[p:p + n_basis, p:p + n_basis] = (
        lam_smooth_base * DtD + EPS_RIDGE_BASE * np.sqrt(n) * np.eye(n_basis)
    )
    # Original L2 scaling on ω_treat:  ridge_treat·√n · I
    P[p + n_basis:, p + n_basis:] = ridge_treat * np.sqrt(n) * np.eye(n_basis)

    coefs = np.linalg.solve(H.T @ H + P, H.T @ Y)
    beta = coefs[:p]
    omega_base = coefs[p:p + n_basis]
    omega_treat = coefs[p + n_basis:]

    return {
        "eta": eta, "info": info, "beta": beta,
        "omega_base": omega_base, "omega_treat": omega_treat,
    }


def bin_means(eta_vec, y_vec, edges, min_per_bin=5):
    n_bins = len(edges) - 1
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (eta_vec >= edges[i]) & (eta_vec < edges[i + 1])
        if sel.sum() >= min_per_bin:
            means[i] = y_vec[sel].mean()
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

    fits = {lam: fit_pspline_base(Q, X, Y, D, lam) for lam in LAMBDAS}

    info0 = fits[LAMBDAS[0]]["info"]
    eta0 = fits[LAMBDAS[0]]["eta"]
    grid = np.linspace(info0["lo"], info0["hi"], 600)

    bin_edges = np.linspace(float(eta0.min()), float(eta0.max()), 61)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    co_mask = D == 0
    eta_co = eta0[co_mask]
    counts_co, _ = np.histogram(eta_co, bins=bin_edges)
    counts_tr, _ = np.histogram(eta0[D == 1], bins=bin_edges)

    rmse_window = (-15.0, 15.0)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8),
                           gridspec_kw={"height_ratios": [3.0, 1.0]},
                           sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(LAMBDAS)))

    rows = []
    for col, lam in zip(colors, LAMBDAS):
        f = fits[lam]
        b_grid = _eval_basis(grid, f["info"]) @ f["omega_base"]
        Y_resid = Y - X @ f["beta"]
        means_co = bin_means(eta_co, Y_resid[co_mask], bin_edges)

        b_at_centers = _eval_basis(bin_centers, f["info"]) @ f["omega_base"]
        in_window = ((bin_centers >= rmse_window[0])
                     & (bin_centers <= rmse_window[1]))
        valid = in_window & ~np.isnan(means_co)
        rmse = float(np.sqrt(np.mean((b_at_centers[valid] - means_co[valid]) ** 2)))

        b10 = float(_eval_basis(np.array([10.0]), f["info"]) @ f["omega_base"])
        b18 = float(_eval_basis(np.array([18.0]), f["info"]) @ f["omega_base"])

        # Quick shape metric: derivative discontinuity around the dip,
        # measured as the change in slope across η = 14.
        b_near_14 = _eval_basis(np.array([12.0, 14.0, 16.0]), f["info"]) @ f["omega_base"]
        slope_before = (b_near_14[1] - b_near_14[0]) / 2.0
        slope_after = (b_near_14[2] - b_near_14[1]) / 2.0

        rows.append((lam, rmse, b10, b18, b10 - b18, slope_before, slope_after))
        label = (rf"$\lambda={lam:g}$   RMSE={rmse:.3f}   "
                 rf"$\hat b(10)-\hat b(18)$={b10 - b18:+.3f}")
        ax[0].plot(grid, b_grid, color=col, lw=2.0, label=label)
        if lam == LAMBDAS[0]:
            ax[0].errorbar(bin_centers, means_co, fmt="o", color="black",
                           ms=3.5, alpha=0.7,
                           label=r"control bin-mean $Y-X\hat\beta$ (λ=0 fit)")

    ax[0].axvspan(13.5, 19.5, color="orange", alpha=0.12, label="dip region")
    ax[0].set_ylabel(r"$\hat b(\eta)$  /  $Y - X\hat\beta$")
    ax[0].set_title(r"Lending Club: P-spline second-difference penalty $\lambda \, \|D_2 \omega_{\mathrm{base}}\|^2$"
                    r"  (fixed $\rho_{\mathrm{treat}}=0.1$)")
    ax[0].legend(loc="lower right", fontsize=8)

    bw = bin_edges[1] - bin_edges[0]
    ax[1].bar(bin_centers, counts_co, width=bw, color="steelblue",
              alpha=0.65, label=f"control (n={counts_co.sum():,})")
    ax[1].bar(bin_centers, counts_tr, width=bw, bottom=counts_co,
              color="firebrick", alpha=0.65,
              label=f"treated (n={counts_tr.sum():,})")
    ax[1].axvspan(13.5, 19.5, color="orange", alpha=0.12)
    ax[1].set_xlabel(r"$\eta$")
    ax[1].set_ylabel("count")
    ax[1].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'lambda':>10}  {'RMSE':>7}  {'b̂(10)':>8}  {'b̂(18)':>8}  "
          f"{'b̂(10)-b̂(18)':>13}  {'slope[12-14]':>13}  {'slope[14-16]':>13}")
    for lam, rmse, b10, b18, gap, s_before, s_after in rows:
        print(f"{lam:>10.3g}  {rmse:>7.3f}  {b10:>8.3f}  {b18:>8.3f}  "
              f"{gap:>+13.3f}  {s_before:>+13.3f}  {s_after:>+13.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
