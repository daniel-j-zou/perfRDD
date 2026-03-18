"""
Bootstrap visualization of the treatment effect function alpha(eta).

Resamples the GPA probation data B times, re-runs the full estimation
pipeline each time, and evaluates alpha on a fixed eta grid.

Produces:
  boot_fig1_alpha_spaghetti.png  — all bootstrap curves + point estimate
  boot_fig2_alpha_band.png       — pointwise 95% envelope + point estimate
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

from gpa_analysis import (
    _basis_params, _eval_basis,
    X_COLS, Q_COL, Y_COL, DATA_PATH, OUTDIR,
)

B = 200  # number of bootstrap resamples
RNG_SEED = 42


def _estimate_alpha_on_grid(df, eta_grid, spline_info):
    """
    Full pipeline on a (possibly resampled) dataframe.
    Returns alpha evaluated on the fixed eta_grid, or None on failure.
    """
    n = len(df)
    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    if n_tr < 10:
        return None

    # First stage: Q = [1, X] @ gamma + eta
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat

    # B-spline support and knots (data-driven per bootstrap)
    support = (np.percentile(eta_hat, 2), np.percentile(eta_hat, 98))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _basis_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    # Pooled PLM
    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]

    try:
        be, *_ = np.linalg.lstsq(H, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    omega_treat = be[p + n_basis:]

    # Evaluate on the FIXED eta_grid using THIS bootstrap's spline info
    Phi_grid = _eval_basis(eta_grid, info)
    return Phi_grid @ omega_treat


def main():
    rng = np.random.default_rng(RNG_SEED)

    # ---- Load data (same as gpa_analysis.py) ----
    df = pd.read_csv(DATA_PATH)
    df = df[df[Y_COL].notna()].copy()
    n = len(df)
    print(f"Loaded {n} complete cases")

    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    # ---- Original point estimate (trimmed mode) ----
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat

    eta_Tr = eta_hat[D == 1]

    support = (np.percentile(eta_hat, 2), np.percentile(eta_hat, 98))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info_orig = _basis_params(kn, support)
    Phi = _eval_basis(eta_hat, info_orig)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]
    be, *_ = np.linalg.lstsq(H, y, rcond=None)
    omega_treat_orig = be[p + n_basis:]

    # Trimmed eval region (fixed across all bootstraps)
    eval_lo = max(support[0], np.percentile(eta_Tr, 5))
    eval_hi = min(support[1], np.percentile(eta_Tr, 95))
    eta_grid = np.linspace(eval_lo, eval_hi, 500)

    Phi_grid_orig = _eval_basis(eta_grid, info_orig)
    alpha_orig = Phi_grid_orig @ omega_treat_orig

    # Plug-in average
    in_eval = (eta_Tr >= eval_lo) & (eta_Tr <= eval_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr[in_eval], info_orig)
    alpha_hat = np.mean(Phi_Tr_eval @ omega_treat_orig)
    print(f"Original plug-in avg: {alpha_hat:.4f}")
    print(f"Eval region: [{eval_lo:.3f}, {eval_hi:.3f}]")

    # ---- Bootstrap loop ----
    print(f"\nRunning {B} bootstrap resamples...")
    alpha_boots = []
    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        alpha_b = _estimate_alpha_on_grid(df_b, eta_grid, info_orig)
        if alpha_b is not None:
            alpha_boots.append(alpha_b)
        if (b + 1) % 50 == 0:
            print(f"  {b + 1}/{B} done ({len(alpha_boots)} successful)")

    alpha_boots = np.array(alpha_boots)  # shape (B_success, 500)
    print(f"\n{len(alpha_boots)} successful bootstrap resamples")

    # ---- Figure 1: Spaghetti plot ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for i in range(len(alpha_boots)):
        ax1.plot(eta_grid, alpha_boots[i], color="steelblue",
                 alpha=0.08, linewidth=0.5)
    ax1.plot(eta_grid, alpha_orig, color="darkblue", linewidth=2.5,
             label=r"Point estimate $\hat\alpha(\eta)$")
    ax1.axhline(alpha_hat, color="red", linestyle="--", linewidth=1.5,
                label=f"Plug-in avg = {alpha_hat:.3f}")
    ax1.axhline(0, color="black", linewidth=0.5)

    ax1_hist = ax1.twinx()
    ax1_hist.hist(eta_Tr[in_eval], bins=50, alpha=0.12, color="gray", density=True)
    ax1_hist.set_ylabel("Density (treated)", color="gray")
    ax1_hist.tick_params(axis="y", labelcolor="gray")

    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\alpha(\eta)$")
    ax1.set_title(f"Bootstrap spaghetti: treatment effect on next-year GPA (B={len(alpha_boots)})")
    ax1.legend(loc="best")
    ax1.set_zorder(ax1_hist.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "boot_fig1_alpha_spaghetti.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved boot_fig1_alpha_spaghetti.png")

    # ---- Figure 2: Percentile band ----
    alpha_lo = np.percentile(alpha_boots, 2.5, axis=0)
    alpha_hi = np.percentile(alpha_boots, 97.5, axis=0)
    alpha_med = np.median(alpha_boots, axis=0)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.fill_between(eta_grid, alpha_lo, alpha_hi,
                     color="steelblue", alpha=0.25, label="95% bootstrap band")
    ax2.plot(eta_grid, alpha_med, color="steelblue", linestyle="--",
             linewidth=1.5, label="Bootstrap median")
    ax2.plot(eta_grid, alpha_orig, color="darkblue", linewidth=2.5,
             label=r"Point estimate $\hat\alpha(\eta)$")
    ax2.axhline(alpha_hat, color="red", linestyle="--", linewidth=1.5,
                label=f"Plug-in avg = {alpha_hat:.3f}")
    ax2.axhline(0, color="black", linewidth=0.5)

    ax2_hist = ax2.twinx()
    ax2_hist.hist(eta_Tr[in_eval], bins=50, alpha=0.12, color="gray", density=True)
    ax2_hist.set_ylabel("Density (treated)", color="gray")
    ax2_hist.tick_params(axis="y", labelcolor="gray")

    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel(r"$\alpha(\eta)$")
    ax2.set_title(f"Bootstrap 95% band: treatment effect on next-year GPA (B={len(alpha_boots)})")
    ax2.legend(loc="best")
    ax2.set_zorder(ax2_hist.get_zorder() + 1)
    ax2.patch.set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "boot_fig2_alpha_band.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved boot_fig2_alpha_band.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
