"""
Bootstrap visualization of the treatment effect function alpha(eta).

Resamples the GPA probation data B times, re-runs the full estimation
pipeline each time, and evaluates alpha on a fixed eta grid.

Produces (trimmed mode — 5th-95th pct of treated eta):
  boot_fig1_alpha_spaghetti.png  — all bootstrap curves + point estimate
  boot_fig2_alpha_band.png       — pointwise 95% envelope + point estimate

Produces (full mode — all treated eta, clipped to support):
  full_boot_fig1_alpha_spaghetti.png
  full_boot_fig2_alpha_band.png
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


def _estimate_alpha_on_grid(df, eta_grid):
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


def _run_bootstrap(df, eta_grid, eta_Tr, alpha_orig, alpha_hat, tag, title_tag):
    """
    Run B bootstrap resamples and produce spaghetti + band figures.

    tag: filename prefix ("" for trimmed, "full_" for full)
    title_tag: string appended to figure titles for context
    """
    rng = np.random.default_rng(RNG_SEED)
    n = len(df)

    eval_lo, eval_hi = eta_grid[0], eta_grid[-1]
    in_eval = (eta_Tr >= eval_lo) & (eta_Tr <= eval_hi)

    print(f"\nRunning {B} bootstrap resamples ({title_tag})...")
    alpha_boots = []
    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        alpha_b = _estimate_alpha_on_grid(df_b, eta_grid)
        if alpha_b is not None:
            alpha_boots.append(alpha_b)
        if (b + 1) % 50 == 0:
            print(f"  {b + 1}/{B} done ({len(alpha_boots)} successful)")

    alpha_boots = np.array(alpha_boots)
    print(f"{len(alpha_boots)} successful bootstrap resamples")

    alpha_lo = np.percentile(alpha_boots, 2.5, axis=0)
    alpha_hi = np.percentile(alpha_boots, 97.5, axis=0)
    alpha_med = np.median(alpha_boots, axis=0)

    # Y-limits: based on point estimate range with generous padding so
    # interior structure is visible; boundary blow-up shows as clipped curves.
    pt_floor = alpha_orig.min()
    pt_ceil = alpha_orig.max()
    pt_range = max(pt_ceil - pt_floor, 0.5)  # at least 0.5
    ylim = (pt_floor - 1.5 * pt_range, pt_ceil + 1.5 * pt_range)

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
    ax1.set_ylim(ylim)

    ax1_hist = ax1.twinx()
    ax1_hist.hist(eta_Tr[in_eval], bins=50, alpha=0.12, color="gray", density=True)
    ax1_hist.set_ylabel("Density (treated)", color="gray")
    ax1_hist.tick_params(axis="y", labelcolor="gray")

    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\alpha(\eta)$")
    ax1.set_title(f"Bootstrap spaghetti ({title_tag}): treatment effect (B={len(alpha_boots)})")
    ax1.legend(loc="best")
    ax1.set_zorder(ax1_hist.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig1.tight_layout()
    fname1 = f"{tag}boot_fig1_alpha_spaghetti.png"
    fig1.savefig(os.path.join(OUTDIR, fname1), dpi=150, bbox_inches="tight")
    print(f"Saved {fname1}")

    # ---- Figure 2: Percentile band ----
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
    ax2.set_ylim(ylim)

    ax2_hist = ax2.twinx()
    ax2_hist.hist(eta_Tr[in_eval], bins=50, alpha=0.12, color="gray", density=True)
    ax2_hist.set_ylabel("Density (treated)", color="gray")
    ax2_hist.tick_params(axis="y", labelcolor="gray")

    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel(r"$\alpha(\eta)$")
    ax2.set_title(f"Bootstrap 95% band ({title_tag}): treatment effect (B={len(alpha_boots)})")
    ax2.legend(loc="best")
    ax2.set_zorder(ax2_hist.get_zorder() + 1)
    ax2.patch.set_visible(False)
    fig2.tight_layout()
    fname2 = f"{tag}boot_fig2_alpha_band.png"
    fig2.savefig(os.path.join(OUTDIR, fname2), dpi=150, bbox_inches="tight")
    print(f"Saved {fname2}")

    plt.close("all")


def main():
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

    # ---- Original point estimate ----
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

    # ---- Trimmed eval region ----
    trim_lo = max(support[0], np.percentile(eta_Tr, 5))
    trim_hi = min(support[1], np.percentile(eta_Tr, 95))
    eta_grid_trim = np.linspace(trim_lo, trim_hi, 500)

    Phi_grid_trim = _eval_basis(eta_grid_trim, info_orig)
    alpha_orig_trim = Phi_grid_trim @ omega_treat_orig

    in_trim = (eta_Tr >= trim_lo) & (eta_Tr <= trim_hi)
    alpha_hat_trim = np.mean(_eval_basis(eta_Tr[in_trim], info_orig) @ omega_treat_orig)
    print(f"Trimmed eval region: [{trim_lo:.3f}, {trim_hi:.3f}]")
    print(f"Trimmed plug-in avg: {alpha_hat_trim:.4f}")

    # ---- Full eval region (all treated eta, clipped to support) ----
    full_lo = max(support[0], eta_Tr.min())
    full_hi = min(support[1], eta_Tr.max())
    eta_grid_full = np.linspace(full_lo, full_hi, 500)

    Phi_grid_full = _eval_basis(eta_grid_full, info_orig)
    alpha_orig_full = Phi_grid_full @ omega_treat_orig

    alpha_hat_full = np.mean(_eval_basis(eta_Tr, info_orig) @ omega_treat_orig)
    print(f"Full eval region: [{full_lo:.3f}, {full_hi:.3f}]")
    print(f"Full plug-in avg: {alpha_hat_full:.4f}")

    # ---- Run bootstraps ----
    print("\n" + "=" * 60)
    print("TRIMMED (5th-95th pct)")
    print("=" * 60)
    _run_bootstrap(df, eta_grid_trim, eta_Tr, alpha_orig_trim,
                   alpha_hat_trim, tag="", title_tag="trimmed")

    print("\n" + "=" * 60)
    print("FULL (all treated eta)")
    print("=" * 60)
    _run_bootstrap(df, eta_grid_full, eta_Tr, alpha_orig_full,
                   alpha_hat_full, tag="full_", title_tag="full")

    print("\nDone.")


if __name__ == "__main__":
    main()
