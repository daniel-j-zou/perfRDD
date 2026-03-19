"""
Analyze whether dropping out (left_school) is correlated with eta_hat.

Uses the FULL sample (including missing nextGPA) for the first stage,
then examines how left_school relates to eta_hat, especially among treated.
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


def main():
    # ---- Load FULL data (don't drop missing nextGPA) ----
    df = pd.read_csv(DATA_PATH)
    n = len(df)
    print(f"Full sample: {n}")

    X = df[X_COLS].values
    q = df[Q_COL].values
    D = (q < 0).astype(float)
    left = df["left_school"].values.astype(float)
    has_y = df[Y_COL].notna().values

    n_tr = int(D.sum())
    n_con = n - n_tr
    print(f"Treated: {n_tr}, Control: {n_con}")

    # ---- First stage on full sample ----
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat

    print(f"\nFirst stage (full sample):")
    print(f"  eta_hat range: [{eta_hat.min():.3f}, {eta_hat.max():.3f}]")
    print(f"  R^2: {1 - eta_hat.var() / q.var():.4f}")

    # ---- Dropout rates by treatment status ----
    eta_Tr = eta_hat[D == 1]
    eta_Con = eta_hat[D == 0]
    left_Tr = left[D == 1]
    left_Con = left[D == 0]

    print(f"\n=== Dropout rates ===")
    print(f"Treated:  {left_Tr.mean():.3f} ({int(left_Tr.sum())}/{len(left_Tr)})")
    print(f"Control:  {left_Con.mean():.3f} ({int(left_Con.sum())}/{len(left_Con)})")

    # ---- Dropout rate by eta_hat decile (treated only) ----
    print(f"\n=== Treated: dropout rate by eta_hat decile ===")
    pcts = np.percentile(eta_Tr, np.arange(0, 101, 10))
    for i in range(10):
        mask = (eta_Tr >= pcts[i]) & (eta_Tr < pcts[i + 1])
        if i == 9:
            mask = (eta_Tr >= pcts[i]) & (eta_Tr <= pcts[i + 1])
        rate = left_Tr[mask].mean() if mask.any() else np.nan
        print(f"  D{i+1:2d} [{pcts[i]:6.3f}, {pcts[i+1]:6.3f}]: "
              f"dropout={rate:.3f}  (n={mask.sum()})")

    # ---- PLM: left_school = X @ beta + Phi(eta) @ omega_base + D*Phi(eta) @ omega_treat ----
    support = (np.percentile(eta_hat, 2), np.percentile(eta_hat, 98))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _basis_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]

    be, *_ = np.linalg.lstsq(H, left, rcond=None)
    omega_treat = be[p + n_basis:]
    omega_base = be[p:p + n_basis]
    beta = be[:p]

    resid = left - H @ be
    sigma2 = np.sum(resid**2) / (n - H.shape[1])
    print(f"\nPLM for left_school:")
    print(f"  Residual std: {np.sqrt(sigma2):.4f}")
    print(f"  beta: {np.array2string(beta, precision=4, separator=', ')}")

    # ---- Eval on trimmed grid ----
    eval_lo = max(support[0], np.percentile(eta_Tr, 5))
    eval_hi = min(support[1], np.percentile(eta_Tr, 95))
    eta_grid = np.linspace(eval_lo, eval_hi, 500)
    Phi_grid = _eval_basis(eta_grid, info)

    dropout_effect = Phi_grid @ omega_treat  # treatment effect on dropout
    dropout_base = Phi_grid @ omega_base     # baseline dropout

    in_eval = (eta_Tr >= eval_lo) & (eta_Tr <= eval_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr[in_eval], info)
    avg_dropout_effect = np.mean(Phi_Tr_eval @ omega_treat)
    print(f"\nPlug-in avg treatment effect on dropout: {avg_dropout_effect:.4f}")
    print(f"  (positive = probation increases dropout)")

    # ---- Figure 1: Dropout effect alpha_dropout(eta) ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(eta_grid, dropout_effect, "b-", linewidth=2,
             label=r"$\hat\alpha_{\mathrm{dropout}}(\eta)$")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(avg_dropout_effect, color="red", linestyle="--", linewidth=1,
                label=f"Plug-in avg = {avg_dropout_effect:.3f}")

    ax1_hist = ax1.twinx()
    ax1_hist.hist(eta_Tr[in_eval], bins=50, alpha=0.12, color="gray", density=True)
    ax1_hist.set_ylabel("Density (treated)", color="gray")
    ax1_hist.tick_params(axis="y", labelcolor="gray")

    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\alpha_{\mathrm{dropout}}(\eta)$")
    ax1.set_title("Treatment effect of probation on dropping out")
    ax1.legend(loc="best")
    ax1.set_zorder(ax1_hist.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "dropout_fig1_alpha.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved dropout_fig1_alpha.png")

    # ---- Figure 2: Dropout rate by eta_hat (nonparametric, binned) ----
    n_bins = 30
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for label, eta_g, left_g, color in [
        ("Treated", eta_Tr, left_Tr, "red"),
        ("Control", eta_Con, left_Con, "blue"),
    ]:
        # Bin and compute mean dropout rate
        lo, hi = np.percentile(eta_g, [2, 98])
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        rates = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        for j in range(n_bins):
            mask = (eta_g >= edges[j]) & (eta_g < edges[j + 1])
            if j == n_bins - 1:
                mask = (eta_g >= edges[j]) & (eta_g <= edges[j + 1])
            if mask.sum() > 5:
                rates[j] = left_g[mask].mean()
                counts[j] = mask.sum()
            else:
                rates[j] = np.nan
        valid = ~np.isnan(rates)
        ax2.plot(centers[valid], rates[valid], "o-", color=color,
                 markersize=4, linewidth=1.5, alpha=0.8, label=label)

    ax2.set_xlabel(r"$\hat\eta$")
    ax2.set_ylabel("Dropout rate (left_school)")
    ax2.set_title("Dropout rate by eta_hat and treatment status")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "dropout_fig2_binned_rates.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved dropout_fig2_binned_rates.png")

    # ---- Figure 3: Dropout + h functions (base and treated) ----
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(eta_grid, dropout_base, "b-", linewidth=2,
             label=r"$\hat h_{\mathrm{base}}(\eta)$ (control)")
    ax3.plot(eta_grid, dropout_base + dropout_effect, "r-", linewidth=2,
             label=r"$\hat h_{\mathrm{base}} + \hat\alpha_{\mathrm{dropout}}$ (treated)")
    ax3.axvline(eval_lo, color="green", linestyle=":", alpha=0.5)
    ax3.axvline(eval_hi, color="green", linestyle=":", alpha=0.5,
                label="Eval region")
    ax3.set_xlabel(r"$\eta$")
    ax3.set_ylabel("P(left_school)")
    ax3.set_title("Nonparametric dropout probability (pooled beta)")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "dropout_fig3_h_functions.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved dropout_fig3_h_functions.png")

    # ---- Figure 4: Compare eta distributions (stayers vs leavers, treated) ----
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    stayers = eta_Tr[left_Tr == 0]
    leavers = eta_Tr[left_Tr == 1]
    ax4.hist(stayers, bins=60, density=True, alpha=0.5, color="blue",
             label=f"Stayed (n={len(stayers)})")
    ax4.hist(leavers, bins=40, density=True, alpha=0.5, color="red",
             label=f"Left school (n={len(leavers)})")
    ax4.set_xlabel(r"$\hat\eta$")
    ax4.set_ylabel("Density")
    ax4.set_title(r"Distribution of $\hat\eta$ among treated: stayers vs dropouts")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTDIR, "dropout_fig4_eta_distributions.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved dropout_fig4_eta_distributions.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
