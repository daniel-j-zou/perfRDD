"""
GPA probation analysis with treatment cost C.

Reruns the B-spline PLM estimation (same as gpa_analysis.py trimmed mode)
and computes the utility function U(phi) = E[(alpha(eta) - C) * P(Q < phi | eta)].

Figures saved with c{C}_prefix.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from gpa_analysis import (
    _basis_params, _eval_basis,
    X_COLS, Q_COL, Y_COL, DATA_PATH, OUTDIR,
)

C = 0.275


def main():
    ctag = f"{C:.3f}".replace(".", "")  # e.g. "0275"

    # ---- Load data ----
    df = pd.read_csv(DATA_PATH)
    df = df[df[Y_COL].notna()].copy()
    n = len(df)
    print(f"Loaded {n} complete cases")

    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())
    print(f"Treated: {n_tr}, Control: {n - n_tr}")

    # ---- First stage ----
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat
    eta_Tr = eta_hat[D == 1]

    # ---- B-spline PLM ----
    support = (np.percentile(eta_hat, 2), np.percentile(eta_hat, 98))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _basis_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]
    be, *_ = np.linalg.lstsq(H, y, rcond=None)
    omega_treat = be[p + n_basis:]

    # ---- Trimmed eval region ----
    eval_lo = max(support[0], np.percentile(eta_Tr, 5))
    eval_hi = min(support[1], np.percentile(eta_Tr, 95))
    eta_grid = np.linspace(eval_lo, eval_hi, 500)
    Phi_grid = _eval_basis(eta_grid, info)
    alpha_vals = Phi_grid @ omega_treat

    in_eval = (eta_Tr >= eval_lo) & (eta_Tr <= eval_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr[in_eval], info)
    alpha_Tr = Phi_Tr_eval @ omega_treat
    alpha_hat = np.mean(alpha_Tr)
    print(f"\nEval region: [{eval_lo:.3f}, {eval_hi:.3f}]")
    print(f"Plug-in avg alpha: {alpha_hat:.4f}")
    print(f"Cost C = {C}")
    print(f"Plug-in avg (alpha - C): {alpha_hat - C:.4f}")
    print(f"Fraction with alpha > C: {np.mean(alpha_Tr > C):.3f}")

    # ---- Utility with C = 0.25 ----
    gX_vals = q - eta_hat
    gX_sorted = np.sort(gX_vals)

    in_eval_all = (eta_hat >= eval_lo) & (eta_hat <= eval_hi)
    eta_eval_obs = eta_hat[in_eval_all]
    Phi_eval_obs = _eval_basis(eta_eval_obs, info)
    alpha_eval_obs = Phi_eval_obs @ omega_treat
    n_eval = in_eval_all.sum()
    print(f"Utility: using {n_eval} observations in eval region")

    phi_grid = np.linspace(-2.0, 2.0, 200)
    util = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_eval_obs
        probs = np.searchsorted(gX_sorted, thresh) / len(gX_sorted)
        util[j] = np.mean((alpha_eval_obs - C) * probs)

    opt_idx = np.argmax(util)
    opt_phi = phi_grid[opt_idx]

    # Also compute C=0 optimal for comparison
    util_c0 = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_eval_obs
        probs = np.searchsorted(gX_sorted, thresh) / len(gX_sorted)
        util_c0[j] = np.mean(alpha_eval_obs * probs)
    opt_phi_c0 = phi_grid[np.argmax(util_c0)]

    print(f"\nC=0.25: optimal phi* = {opt_phi:.3f}")
    print(f"C=0.00: optimal phi* = {opt_phi_c0:.3f}")
    print(f"Current cutoff: phi = 0.0")
    print(f"Utility at current cutoff (C={C}): {np.interp(0.0, phi_grid, util):.4f}")
    print(f"Utility at optimal (C={C}): {util[opt_idx]:.4f}")

    # ---- Figure 1: alpha(eta) with cost threshold ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(eta_grid, alpha_vals, "b-", linewidth=2,
             label=r"$\hat\alpha(\eta)$")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(C, color="orange", linestyle="-", linewidth=1.5,
                label=f"Cost C = {C}")
    ax1.axhline(alpha_hat, color="red", linestyle="--", linewidth=1,
                label=f"Plug-in avg = {alpha_hat:.3f}")

    # Shade regions where alpha > C vs alpha < C
    ax1.fill_between(eta_grid, C, alpha_vals,
                     where=(alpha_vals > C), color="green", alpha=0.15,
                     label=r"$\alpha(\eta) > C$ (net benefit)")
    ax1.fill_between(eta_grid, C, alpha_vals,
                     where=(alpha_vals < C), color="red", alpha=0.15,
                     label=r"$\alpha(\eta) < C$ (net cost)")

    ax1_hist = ax1.twinx()
    ax1_hist.hist(eta_Tr[in_eval], bins=50, alpha=0.12, color="gray", density=True)
    ax1_hist.set_ylabel("Density (treated in eval region)", color="gray")
    ax1_hist.tick_params(axis="y", labelcolor="gray")

    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\alpha(\eta)$")
    ax1.set_title(f"Treatment effect vs cost (C = {C})")
    ax1.legend(loc="best")
    ax1.set_zorder(ax1_hist.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, f"c{ctag}_fig1_alpha.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved c{ctag}_fig1_alpha.png")

    # ---- Figure 3: Utility function (both C values) ----
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(phi_grid, util_c0, color="steelblue", linewidth=1.5,
             linestyle="--", alpha=0.7, label=r"$U(\phi)$, C = 0")
    ax3.plot(phi_grid, util, "b-", linewidth=2,
             label=rf"$U(\phi)$, C = {C}")
    ax3.axvline(0, color="gray", linestyle=":", linewidth=1,
                label="Current cutoff (0)")
    ax3.axvline(opt_phi_c0, color="steelblue", linestyle="--", linewidth=1,
                alpha=0.7, label=f"Optimal (C=0) = {opt_phi_c0:.3f}")
    ax3.axvline(opt_phi, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal (C={C}) = {opt_phi:.3f}")
    ax3.set_xlabel(r"Threshold $\phi$")
    ax3.set_ylabel("Utility")
    ax3.set_title(f"Utility function: C = {C} vs C = 0")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, f"c{ctag}_fig3_utility.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved c{ctag}_fig3_utility.png")

    # ---- Figure 4: Q distribution with thresholds ----
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(q, bins=100, density=True, alpha=0.5, color="steelblue",
             label="Q = dist_from_cut")
    ax4.axvline(0, color="gray", linestyle=":", linewidth=1.5,
                label="Current cutoff (0)")
    ax4.axvline(opt_phi_c0, color="steelblue", linestyle="--", linewidth=1.5,
                alpha=0.7, label=f"Optimal (C=0) = {opt_phi_c0:.3f}")
    ax4.axvline(opt_phi, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal (C={C}) = {opt_phi:.3f}")
    ax4.set_xlabel("Q (dist_from_cut)")
    ax4.set_ylabel("Density")
    ax4.set_title(f"Distribution of Q with cutoff thresholds (C = {C})")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTDIR, f"c{ctag}_fig4_Q_distribution.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved c{ctag}_fig4_Q_distribution.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
