"""
Plug-in semiparametric estimator for the GPA probation dataset — GAUSSIAN BASIS.

Same analysis as gpa_analysis.py but using Gaussian (normal PDF) basis functions
instead of cubic B-splines. This matches the original plm.py approach.

Setting:
  X = [hsgrade_pct, totcredits_year1, loc_campus1, loc_campus2,
       male, bpl_north_america, age_at_entry, english]
  Q = dist_from_cut  (distance from probation GPA cutoff)
  Y = nextGPA
  Treatment: s = 1{Q < 0}  (on probation)

Method: Gaussian-basis PLM with POOLED beta.
  1. First stage: OLS of Q on [1, X] -> gamma_hat, eta_hat = Q - X @ gamma_hat
  2. Pooled PLM: Y = X @ beta + Phi(eta_hat) @ omega_base + D * Phi(eta_hat) @ omega_treat
     where D = treatment indicator, Phi uses Gaussian basis.
     alpha(eta) = Phi(eta) @ omega_treat gives the treatment effect function.
  3. Utility: U(phi) = E[(alpha(eta) - C) * P(Q < phi | eta)]
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(OUTDIR), "..", "Dep_Data", "final_processed_data.csv"
)

X_COLS = [
    "hsgrade_pct", "totcredits_year1", "loc_campus1", "loc_campus2",
    "male", "bpl_north_america", "age_at_entry", "english",
]
Q_COL = "dist_from_cut"
Y_COL = "nextGPA"


# ---- Gaussian basis ----

def _basis_params(kn, support, bwf=1.0):
    lo, hi = support
    qr = hi - lo
    if qr == 0:
        qr = 1.0
    idx = np.arange(1, kn + 1) - 0.5
    mu = lo + idx * qr / kn
    bw = bwf * qr / kn
    return {"mu": mu, "bw": bw, "lo": lo, "hi": hi}


def _eval_basis(pts, info):
    pts = np.asarray(pts, dtype=float)
    return norm.pdf((pts[:, None] - info["mu"][None, :]), scale=info["bw"])


def run_analysis(eval_mode="trimmed"):
    """
    eval_mode: "trimmed" uses 5th-95th pct of treated eta (original),
               "full" uses all treated eta values.
    """
    # ---- Load data ----
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} observations, {df[Y_COL].notna().sum()} with non-missing {Y_COL}")

    df = df[df[Y_COL].notna()].copy()
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

    # ---- Gaussian basis support and knots ----
    support = (np.percentile(eta_hat, 2), np.percentile(eta_hat, 98))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    print(f"\nGaussian basis support: [{support[0]:.3f}, {support[1]:.3f}]")
    print(f"Centers (based on n_tr={n_tr}): {kn}")

    info = _basis_params(kn, support, bwf=1.0)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]
    print(f"Basis functions: {n_basis}, bandwidth: {info['bw']:.4f}")

    # ---- Pooled PLM: Y = X @ beta + Phi @ omega_base + D*Phi @ omega_treat ----
    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]

    print(f"\nDesign matrix: {H.shape} (p_X={p}, p_base={n_basis}, p_treat={n_basis})")
    be, *_ = np.linalg.lstsq(H, y, rcond=None)

    beta = be[:p]
    omega_base = be[p:p + n_basis]
    omega_treat = be[p + n_basis:]

    print(f"\nbeta: {np.array2string(beta, precision=4, separator=', ')}")

    # ---- OLS covariance for confidence intervals ----
    resid = y - H @ be
    sigma2 = np.sum(resid**2) / (n - H.shape[1])
    HtH_inv = np.linalg.pinv(H.T @ H)
    cov_treat = sigma2 * HtH_inv[p + n_basis:, p + n_basis:]
    cov_base = sigma2 * HtH_inv[p:p + n_basis, p:p + n_basis]
    cov_sum = cov_base + cov_treat + \
              sigma2 * (HtH_inv[p:p + n_basis, p + n_basis:] +
                        HtH_inv[p + n_basis:, p:p + n_basis])
    print(f"Residual std: {np.sqrt(sigma2):.4f}")

    # ---- Evaluation region ----
    if eval_mode == "trimmed":
        eval_lo = max(support[0], np.percentile(eta_Tr, 5))
        eval_hi = min(support[1], np.percentile(eta_Tr, 95))
        tag = "gauss_"
    else:
        eval_lo = eta_Tr.min()
        eval_hi = eta_Tr.max()
        tag = "gauss_full_"
    print(f"Evaluation region ({eval_mode}): [{eval_lo:.3f}, {eval_hi:.3f}]")

    eta_grid = np.linspace(eval_lo, eval_hi, 500)
    Phi_grid = _eval_basis(eta_grid, info)
    alpha_vals = Phi_grid @ omega_treat

    # 95% CI for alpha
    alpha_se = np.sqrt(np.sum((Phi_grid @ cov_treat) * Phi_grid, axis=1))
    alpha_lo95 = alpha_vals - 1.96 * alpha_se
    alpha_hi95 = alpha_vals + 1.96 * alpha_se

    # 95% CI for h functions
    hbase_se = np.sqrt(np.sum((Phi_grid @ cov_base) * Phi_grid, axis=1))
    htreat_se = np.sqrt(np.sum((Phi_grid @ cov_sum) * Phi_grid, axis=1))

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
    ax0.set_title(r"Distribution of $\hat\eta$ by treatment status (Gaussian basis)")
    ax0.legend()
    fig0.tight_layout()
    fig0.savefig(os.path.join(OUTDIR, f"{tag}fig0_eta_distributions.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved {tag}fig0_eta_distributions.png")

    # ---- Figure 1: alpha(eta) with 95% CI ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.fill_between(eta_grid, alpha_lo95, alpha_hi95, color="blue", alpha=0.15,
                      label="95% CI")
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
    ax1.set_title("Estimated treatment effect (Gaussian basis)")
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, f"{tag}fig1_alpha.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {tag}fig1_alpha.png")

    # ---- Figure 2: h_base and h_treat with 95% CI ----
    eta_grid_full = np.linspace(support[0], support[1], 500)
    Phi_full = _eval_basis(eta_grid_full, info)
    h_base_full = Phi_full @ omega_base
    hbase_se_full = np.sqrt(np.sum((Phi_full @ cov_base) * Phi_full, axis=1))

    h_treat_eval = Phi_grid @ (omega_base + omega_treat)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.fill_between(eta_grid_full,
                     h_base_full - 1.96 * hbase_se_full,
                     h_base_full + 1.96 * hbase_se_full,
                     color="blue", alpha=0.1)
    ax2.plot(eta_grid_full, h_base_full, "b-", linewidth=2,
             label=r"$\hat h_{\mathrm{base}}(\eta)$ (control)")
    ax2.fill_between(eta_grid,
                     h_treat_eval - 1.96 * htreat_se,
                     h_treat_eval + 1.96 * htreat_se,
                     color="red", alpha=0.1)
    ax2.plot(eta_grid, h_treat_eval, "r-", linewidth=2,
             label=r"$\hat h_{\mathrm{base}}(\eta) + \hat\alpha(\eta)$ (treated)")
    ax2.axvline(eval_lo, color="green", linestyle=":", alpha=0.5)
    ax2.axvline(eval_hi, color="green", linestyle=":", alpha=0.5,
                label="Eval region")
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel("h")
    ax2.set_title("Nonparametric components with 95% CI — Gaussian basis (pooled beta)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, f"{tag}fig2_h_functions.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {tag}fig2_h_functions.png")

    # ---- Utility function and optimal threshold ----
    gX_vals = q - eta_hat
    gX_sorted = np.sort(gX_vals)

    in_eval_all = (eta_hat >= eval_lo) & (eta_hat <= eval_hi)
    eta_eval_obs = eta_hat[in_eval_all]
    Phi_eval_obs = _eval_basis(eta_eval_obs, info)
    alpha_eval_obs = Phi_eval_obs @ omega_treat
    n_eval = in_eval_all.sum()
    print(f"\nUtility: using {n_eval} observations in eval region")

    C = 0.0
    phi_grid = np.linspace(-2.0, 2.0, 200)
    util = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_eval_obs
        probs = np.searchsorted(gX_sorted, thresh) / len(gX_sorted)
        util[j] = np.mean((alpha_eval_obs - C) * probs)

    opt_idx = np.argmax(util)
    opt_phi = phi_grid[opt_idx]

    print(f"\nUtility maximizer (C={C}): phi* = {opt_phi:.3f}")
    print(f"Current cutoff: phi = 0.0")
    print(f"Utility at current cutoff: {np.interp(0.0, phi_grid, util):.4f}")
    print(f"Utility at optimal: {util[opt_idx]:.4f}")

    # ---- Figure 3: Utility function ----
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(phi_grid, util, "b-", linewidth=2, label=r"Utility $U(\phi)$")
    ax3.axvline(0, color="gray", linestyle=":", linewidth=1, label="Current cutoff (0)")
    ax3.axvline(opt_phi, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal cutoff = {opt_phi:.3f}")
    ax3.set_xlabel(r"Threshold $\phi$")
    ax3.set_ylabel("Utility")
    ax3.set_title(f"Utility function — Gaussian basis (C = {C})")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, f"{tag}fig3_utility.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {tag}fig3_utility.png")

    # ---- Figure 4: Distribution of Q overlaid with optimal threshold ----
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(q, bins=100, density=True, alpha=0.5, color="steelblue", label="Q = dist_from_cut")
    ax4.axvline(0, color="gray", linestyle=":", linewidth=1.5, label="Current cutoff (0)")
    ax4.axvline(opt_phi, color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal cutoff = {opt_phi:.3f}")
    ax4.set_xlabel("Q (dist_from_cut)")
    ax4.set_ylabel("Density")
    ax4.set_title("Distribution of Q with cutoff thresholds (Gaussian basis)")
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
    print("FULL (all treated eta values)")
    print("=" * 60)
    run_analysis("full")


if __name__ == "__main__":
    main()
