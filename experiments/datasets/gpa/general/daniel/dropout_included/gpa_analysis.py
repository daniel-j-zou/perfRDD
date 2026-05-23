"""
Plug-in semiparametric estimator for the GPA probation dataset.
DROPOUT-INCLUSIVE version: students who left school have nextGPA
imputed as (0 - GPA_year1), reflecting a GPA drop to zero.

Setting:
  X = [hsgrade_pct, totcredits_year1, loc_campus1, loc_campus2,
       male, bpl_north_america, age_at_entry, english]
  Q = dist_from_cut  (distance from probation GPA cutoff)
  Y = nextGPA  (imputed for dropouts)
  Treatment: s = 1{Q < 0}  (on probation)

Method: cubic B-spline PLM with POOLED beta.
  1. First stage: OLS of Q on [1, X] -> gamma_hat, eta_hat = Q - X @ gamma_hat
  2. Pooled PLM: Y = X @ beta + Phi(eta_hat) @ omega_base + D * Phi(eta_hat) @ omega_treat
     where D = treatment indicator.
     This constrains beta to be common, so alpha(eta) = Phi(eta) @ omega_treat
     directly gives the treatment effect function.
  3. Utility: U(phi) = E[alpha(eta) * P(Q < phi | eta)]  (C = 0, cost internalized)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(OUTDIR), "..", "..", "Dep_Data", "final_processed_data.csv"
)

X_COLS = [
    "hsgrade_pct", "totcredits_year1", "loc_campus1", "loc_campus2",
    "male", "bpl_north_america", "age_at_entry", "english",
]
Q_COL = "dist_from_cut"
Y_COL = "nextGPA"


# ---- B-spline basis ----

def _basis_params(kn, support):
    degree = 3
    lo, hi = support
    interior = np.linspace(lo, hi, kn + 2)[1:-1]
    t = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
    return {"t": t, "degree": degree, "lo": lo, "hi": hi}


def _eval_basis(pts, info):
    pts_c = np.clip(np.asarray(pts, dtype=float), info["lo"], info["hi"])
    return BSpline.design_matrix(pts_c, info["t"], info["degree"]).toarray()


def load_data():
    """Load data with dropout imputation: nextGPA = 0 - GPA_year1 for dropouts."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} observations")

    # Impute nextGPA for students who left school and have missing nextGPA
    missing_y = df[Y_COL].isna()
    left = df["left_school"] == 1
    impute_mask = missing_y & left
    n_imputed = impute_mask.sum()
    df.loc[impute_mask, Y_COL] = 0.0 - df.loc[impute_mask, "GPA_year1"]

    print(f"Imputed nextGPA = -GPA_year1 for {n_imputed} dropouts with missing nextGPA")
    print(f"  Their GPA_year1: mean={df.loc[impute_mask, 'GPA_year1'].mean():.3f}, "
          f"range=[{df.loc[impute_mask, 'GPA_year1'].min():.3f}, "
          f"{df.loc[impute_mask, 'GPA_year1'].max():.3f}]")
    print(f"  Imputed nextGPA: mean={df.loc[impute_mask, Y_COL].mean():.3f}")

    # Still drop rows with missing nextGPA (non-dropout missing)
    still_missing = df[Y_COL].isna().sum()
    print(f"Still missing nextGPA after imputation: {still_missing} "
          f"(non-dropout, dropped)")
    df = df[df[Y_COL].notna()].copy()
    return df


def run_analysis(eval_mode="trimmed"):
    """
    eval_mode: "trimmed" uses 5th-95th pct of treated eta (original),
               "full" uses all treated eta values (clipped to support).
    """
    df = load_data()
    n = len(df)
    print(f"Using {n} complete cases")

    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values

    # Treatment: probation = Q < 0
    D = (q < 0).astype(float)
    n_tr = int(D.sum())
    n_con = n - n_tr
    print(f"Treated (on probation): {n_tr}, Control: {n_con}")

    # ---- First stage: Q = [1, X] @ gamma + eta ----
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat

    print(f"\nFirst stage:")
    print(f"  eta_hat range: [{eta_hat.min():.3f}, {eta_hat.max():.3f}]")
    print(f"  eta_hat std: {eta_hat.std():.3f}")
    print(f"  R^2: {1 - eta_hat.var() / q.var():.4f}")

    eta_Tr = eta_hat[D == 1]
    eta_Con = eta_hat[D == 0]
    print(f"\nTreated eta_hat:  mean={eta_Tr.mean():.3f}, "
          f"5th={np.percentile(eta_Tr, 5):.3f}, 95th={np.percentile(eta_Tr, 95):.3f}")
    print(f"Control eta_hat:  mean={eta_Con.mean():.3f}, "
          f"5th={np.percentile(eta_Con, 5):.3f}, 95th={np.percentile(eta_Con, 95):.3f}")

    # ---- Sanity check: raw RD estimate near cutoff ----
    near = np.abs(q) < 0.1
    if near.any():
        y_near_tr = y[(q < 0) & (q > -0.1)]
        y_near_con = y[(q >= 0) & (q < 0.1)]
        if len(y_near_tr) > 0 and len(y_near_con) > 0:
            rd_raw = y_near_tr.mean() - y_near_con.mean()
            print(f"\nRaw RD (|Q|<0.1): {rd_raw:.4f} "
                  f"(Tr mean={y_near_tr.mean():.3f}, n={len(y_near_tr)}; "
                  f"Con mean={y_near_con.mean():.3f}, n={len(y_near_con)})")

    # ---- B-spline support and knots ----
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    print(f"\nB-spline support: [{support[0]:.3f}, {support[1]:.3f}]")
    print(f"Knots (based on n_tr={n_tr}): {kn}")

    info = _basis_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]
    print(f"Basis functions: {n_basis}")

    # ---- Pooled PLM: Y = X @ beta + Phi @ omega_base + D*Phi @ omega_treat ----
    # Ridge regularization on spline coefficients only (not X)
    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    # Penalty: ridge on spline coefficients, zero on X coefficients
    lam = 0.1 / np.sqrt(n)
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:, p:], lam)

    print(f"\nDesign matrix: {H.shape} (p_X={p}, p_base={n_basis}, p_treat={n_basis})")
    print(f"Ridge lambda: {lam:.6f}")
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)

    beta = be[:p]
    omega_base = be[p:p + n_basis]
    omega_treat = be[p + n_basis:]

    print(f"\nbeta: {np.array2string(beta, precision=4, separator=', ')}")

    resid = y - H @ be
    sigma2 = np.sum(resid**2) / (n - H.shape[1])
    print(f"Residual std: {np.sqrt(sigma2):.4f}")

    # ---- Evaluation region ----
    if eval_mode == "trimmed":
        eval_lo = max(support[0], np.percentile(eta_Tr, 5))
        eval_hi = min(support[1], np.percentile(eta_Tr, 95))
        tag = ""  # default figures
    else:
        eval_lo = eta_Tr.min()
        eval_hi = eta_Tr.max()
        tag = "full_"
    print(f"Evaluation region ({eval_mode}): [{eval_lo:.3f}, {eval_hi:.3f}]")

    eta_grid = np.linspace(eval_lo, eval_hi, 500)
    Phi_grid = _eval_basis(eta_grid, info)
    alpha_vals = Phi_grid @ omega_treat

    # Plug-in
    in_eval = (eta_Tr >= eval_lo) & (eta_Tr <= eval_hi)
    if in_eval.any():
        Phi_Tr_eval = _eval_basis(eta_Tr[in_eval], info)
        alpha_Tr = Phi_Tr_eval @ omega_treat
        alpha_hat = np.mean(alpha_Tr)
        print(f"\nPlug-in E[alpha | treated, in eval] ({in_eval.sum()} obs): {alpha_hat:.4f}")
    else:
        alpha_hat = np.nan
        print("\nNo treated observations in evaluation region!")

    Phi_Tr_all = _eval_basis(eta_Tr, info)
    alpha_all_tr = Phi_Tr_all @ omega_treat
    print(f"Mean alpha over ALL treated ({n_tr} obs): {np.mean(alpha_all_tr):.4f}")

    # ---- Figure 0: eta distributions ----
    fig0, ax0 = plt.subplots(figsize=(10, 5))
    ax0.hist(eta_Tr, bins=80, density=True, alpha=0.5, color="red", label="Treated")
    ax0.hist(eta_Con, bins=80, density=True, alpha=0.5, color="blue", label="Control")
    ax0.axvline(eval_lo, color="green", linestyle="--", linewidth=1.5,
                label=f"Eval region [{eval_lo:.2f}, {eval_hi:.2f}]")
    ax0.axvline(eval_hi, color="green", linestyle="--", linewidth=1.5)
    ax0.set_xlabel(r"$\hat\eta$")
    ax0.set_ylabel("Density")
    ax0.set_title(r"Distribution of $\hat\eta$ by treatment status (dropout-inclusive)")
    ax0.legend()
    fig0.tight_layout()
    fig0.savefig(os.path.join(OUTDIR, f"{tag}fig0_eta_distributions.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved {tag}fig0_eta_distributions.png")

    # ---- Figure 1: alpha(eta) ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(eta_grid, alpha_vals, "b-", linewidth=2,
             label=r"$\hat\alpha(\eta)$")
    ax1.axhline(0, color="black", linewidth=0.5)
    if not np.isnan(alpha_hat):
        ax1.axhline(alpha_hat, color="red", linestyle="--", linewidth=1,
                     label=f"Plug-in avg = {alpha_hat:.3f}")

    ax1_hist = ax1.twinx()
    ax1_hist.hist(eta_Tr[in_eval], bins=50, alpha=0.15, color="gray", density=True)
    ax1_hist.set_ylabel("Density (treated in eval region)", color="gray")
    ax1_hist.tick_params(axis="y", labelcolor="gray")

    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\alpha(\eta)$")
    ax1.set_title("Treatment effect of probation on GPA change (dropout-inclusive)")
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, f"{tag}fig1_alpha.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {tag}fig1_alpha.png")

    # ---- Figure 2: h_base and h_treat ----
    eta_grid_full = np.linspace(support[0], support[1], 500)
    Phi_full = _eval_basis(eta_grid_full, info)
    h_base_full = Phi_full @ omega_base

    h_treat_eval = Phi_grid @ (omega_base + omega_treat)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(eta_grid_full, h_base_full, "b-", linewidth=2,
             label=r"$\hat h_{\mathrm{base}}(\eta)$ (control)")
    ax2.plot(eta_grid, h_treat_eval, "r-", linewidth=2,
             label=r"$\hat h_{\mathrm{base}}(\eta) + \hat\alpha(\eta)$ (treated)")
    ax2.axvline(eval_lo, color="green", linestyle=":", alpha=0.5)
    ax2.axvline(eval_hi, color="green", linestyle=":", alpha=0.5,
                label="Eval region")
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel("h")
    ax2.set_title("Nonparametric components (pooled beta, dropout-inclusive)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, f"{tag}fig2_h_functions.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {tag}fig2_h_functions.png")

    # ---- Utility function: bounded alpha extrapolation ----
    # alpha is set to 0 for eta outside the treated [5th, 95th] support.
    # This prevents extrapolating treatment effects to students who were
    # never on probation (high eta) or extreme-tail treated (low eta).
    # A small per-capita cost c captures non-GPA costs of probation.
    gX_vals = q - eta_hat
    gX_sorted = np.sort(gX_vals)

    # Compute alpha for ALL observations, then clip
    Phi_all = _eval_basis(eta_hat, info)
    alpha_all = Phi_all @ omega_treat

    # Bound: zero alpha outside treated support
    eta_Tr_lo = np.percentile(eta_Tr, 5)
    eta_Tr_hi = np.percentile(eta_Tr, 95)
    in_support = (eta_hat >= eta_Tr_lo) & (eta_hat <= eta_Tr_hi)
    alpha_bounded = np.where(in_support, alpha_all, 0.0)

    n_in = in_support.sum()
    n_out = n - n_in
    print(f"\nBounded alpha extrapolation:")
    print(f"  Treated support: [{eta_Tr_lo:.3f}, {eta_Tr_hi:.3f}]")
    print(f"  In support: {n_in}, outside (alpha=0): {n_out}")
    print(f"  Avg alpha (in support): {alpha_all[in_support].mean():.4f}")
    print(f"  Avg alpha (bounded, all): {alpha_bounded.mean():.4f}")

    phi_grid = np.linspace(-3.0, 3.0, 600)

    def compute_util_bounded(c_val):
        """U(phi) = E[alpha_bounded(eta) * P(Q<phi|eta)] - c * E[P(Q<phi|eta)]"""
        u = np.zeros(len(phi_grid))
        for j, phi in enumerate(phi_grid):
            thresh = phi - eta_hat
            probs = np.searchsorted(gX_sorted, thresh) / len(gX_sorted)
            benefit = np.mean(alpha_bounded * probs)
            cost = c_val * np.mean(probs)
            u[j] = benefit - cost
        return u

    # Also compute unbounded (original) for comparison
    def compute_util_unbounded(c_val):
        u = np.zeros(len(phi_grid))
        for j, phi in enumerate(phi_grid):
            thresh = phi - eta_hat
            probs = np.searchsorted(gX_sorted, thresh) / len(gX_sorted)
            u[j] = np.mean((alpha_all - c_val) * probs)
        return u

    # C = 0, bounded
    util_b0 = compute_util_bounded(0.0)
    opt_b0 = phi_grid[np.argmax(util_b0)]

    # C = 0, unbounded
    util_u0 = compute_util_unbounded(0.0)
    opt_u0 = phi_grid[np.argmax(util_u0)]

    print(f"\nUnbounded, c=0: phi* = {opt_u0:.3f}, U(phi*)={util_u0.max():.4f}")
    print(f"Bounded,   c=0: phi* = {opt_b0:.3f}, U(phi*)={util_b0.max():.4f}")

    # Sweep cost values with bounded alpha
    print(f"\nBounded alpha + per-capita cost sweep:")
    print(f"{'c':>8s}  {'phi*':>8s}  {'U(phi*)':>10s}  {'U(0)':>10s}")
    print("-" * 42)
    c_results = []
    for c_val in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]:
        u = compute_util_bounded(c_val)
        oi = np.argmax(u)
        op = phi_grid[oi]
        u0 = np.interp(0.0, phi_grid, u)
        print(f"{c_val:8.3f}  {op:8.3f}  {u[oi]:10.4f}  {u0:10.4f}")
        c_results.append((c_val, op, u, u[oi]))

    # Default cost for figures: c=0.06 (non-GPA cost per treated student)
    C_DEFAULT = 0.06
    util_default = compute_util_bounded(C_DEFAULT)
    opt_idx_default = np.argmax(util_default)
    opt_phi_default = phi_grid[opt_idx_default]
    print(f"\nDefault cost c={C_DEFAULT}: phi* = {opt_phi_default:.3f}, "
          f"U(phi*)={util_default[opt_idx_default]:.4f}")

    # ---- Figure 3: Bounded utility + phi*(c) curve ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

    # Left: bounded utility for several costs
    c_plot = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    colors = ["steelblue", "orange", "green", "red", "purple", "brown"]
    for c_val, col in zip(c_plot, colors):
        u = compute_util_bounded(c_val)
        oi = np.argmax(u)
        op = phi_grid[oi]
        axes3[0].plot(phi_grid, u, color=col, linewidth=2,
                      label=rf"c={c_val:.2f}, $\phi^*$={op:.2f}")
    axes3[0].axhline(0, color="black", linewidth=0.5)
    axes3[0].axvline(0, color="gray", linestyle=":", linewidth=1)
    axes3[0].set_xlabel(r"Threshold $\phi$")
    axes3[0].set_ylabel("Utility")
    axes3[0].set_title("Bounded alpha + per-capita cost")
    axes3[0].legend(fontsize=9)

    # Right: phi*(c) curve
    c_fine = np.linspace(0.0, 0.20, 100)
    phi_star = np.zeros(len(c_fine))
    for i, c_val in enumerate(c_fine):
        u = compute_util_bounded(c_val)
        phi_star[i] = phi_grid[np.argmax(u)]
    axes3[1].plot(c_fine, phi_star, "b-", linewidth=2)
    axes3[1].axhline(0, color="gray", linestyle=":", linewidth=1,
                     label="Current cutoff")
    axes3[1].set_xlabel("Per-capita cost c")
    axes3[1].set_ylabel(r"Optimal threshold $\phi^*$")
    axes3[1].set_title(r"$\phi^*(c)$ with bounded alpha")
    axes3[1].legend()

    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, f"{tag}fig3_utility.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved {tag}fig3_utility.png")

    # ---- Figure 4: Distribution of Q overlaid with optimal thresholds ----
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(q, bins=100, density=True, alpha=0.5, color="steelblue", label="Q = dist_from_cut")
    ax4.axvline(0, color="gray", linestyle=":", linewidth=1.5, label="Current cutoff (0)")
    ax4.axvline(opt_b0, color="steelblue", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Bounded, c=0: phi*={opt_b0:.2f}")
    ax4.axvline(opt_phi_default, color="red", linestyle="--", linewidth=1.5,
                label=f"Bounded, c={C_DEFAULT}: phi*={opt_phi_default:.2f}")
    ax4.set_xlabel("Q (dist_from_cut)")
    ax4.set_ylabel("Density")
    ax4.set_title("Distribution of Q with optimal thresholds (bounded alpha)")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTDIR, f"{tag}fig4_Q_distribution.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {tag}fig4_Q_distribution.png")

    plt.close("all")


def main():
    print("=" * 60)
    print("TRIMMED (5th-95th percentile of treated eta)")
    print("=" * 60)
    run_analysis("trimmed")

    print("\n" + "=" * 60)
    print("FULL (all treated eta values, clipped to support)")
    print("=" * 60)
    run_analysis("full")


if __name__ == "__main__":
    main()
