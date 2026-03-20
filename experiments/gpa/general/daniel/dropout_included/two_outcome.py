"""
Two-outcome decomposition: GPA benefit vs. induced dropouts.

Estimates:
  1. alpha_GPA(eta): treatment effect on nextGPA (stayers only)
  2. alpha_dropout(eta): treatment effect on P(dropout)

Then computes the Pareto frontier: as phi varies, how much GPA benefit
do we get and how many additional dropouts do we induce?

The planner's utility is:
  U(phi; lambda) = GPA_benefit(phi) - lambda * induced_dropouts(phi)

We find the lambda where phi* = 0 (current policy is optimal), giving the
"revealed" cost of a dropout in GPA units.

Uses Gaussian treatment basis throughout.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(OUTDIR), "..", "..", "Dep_Data", "final_processed_data.csv",
)

X_COLS = [
    "hsgrade_pct", "totcredits_year1", "loc_campus1", "loc_campus2",
    "male", "bpl_north_america", "age_at_entry", "english",
]
Q_COL = "dist_from_cut"
Y_COL = "nextGPA"


# ---- Basis functions ----

def _bspline_params(kn, support):
    degree = 3
    lo, hi = support
    interior = np.linspace(lo, hi, kn + 2)[1:-1]
    t = np.concatenate([np.repeat(lo, degree + 1), interior,
                        np.repeat(hi, degree + 1)])
    return {"type": "bspline", "t": t, "degree": degree, "lo": lo, "hi": hi}


def _eval_bspline(pts, info):
    pts_c = np.clip(np.asarray(pts, dtype=float), info["lo"], info["hi"])
    return BSpline.design_matrix(pts_c, info["t"], info["degree"]).toarray()


def _gaussian_params(centers, bwf=1.5):
    spacing = np.mean(np.diff(centers)) if len(centers) > 1 else 1.0
    bw = spacing * bwf
    return {"type": "gaussian", "mu": centers, "bw": bw}


def _eval_gaussian(pts, info):
    pts = np.asarray(pts, dtype=float)
    return norm.pdf((pts[:, None] - info["mu"][None, :]) / info["bw"])


def _eval_basis(pts, info):
    if info["type"] == "bspline":
        return _eval_bspline(pts, info)
    else:
        return _eval_gaussian(pts, info)


# ---- PLM estimation ----

def run_plm(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    """Gaussian-treat B-spline-base PLM. Returns treatment basis info + omega."""
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn_base = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info_base = _bspline_params(kn_base, support)
    Phi_base = _eval_basis(eta_hat, info_base)
    n_base = Phi_base.shape[1]

    kn_treat = max(4, int(round(n_tr ** (1.0 / 3.0))))
    tr_lo = np.percentile(eta_Tr, 5)
    tr_hi = np.percentile(eta_Tr, 95)
    centers = np.linspace(tr_lo, tr_hi, kn_treat)
    info_treat = _gaussian_params(centers, bwf=1.5)
    Phi_treat = _eval_basis(eta_hat, info_treat)

    DPhi = D[:, None] * Phi_treat
    H = np.column_stack((X, Phi_base, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam = 0.1 / np.sqrt(n)
    P_mat = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P_mat[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P_mat, H.T @ y)
    omega_treat = be[p + n_base:]

    return info_treat, omega_treat


def compute_expected(alpha_all, eta_hat, gX_sorted, phi_grid):
    """E[alpha(eta) * P(Q < phi | eta)] on a grid."""
    n_total = len(gX_sorted)
    vals = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_hat
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        vals[j] = np.mean(alpha_all * probs)
    return vals


def compute_treat_prob(eta_hat, gX_sorted, phi_grid):
    """E[P(Q < phi | eta)] = P(Q < phi) on a grid."""
    n_total = len(gX_sorted)
    vals = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_hat
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        vals[j] = np.mean(probs)
    return vals


# ---- Main ----

def main():
    df_full = pd.read_csv(DATA_PATH)
    n_full = len(df_full)

    left = df_full["left_school"].values.astype(float)
    has_y = df_full[Y_COL].notna().values

    print(f"Full sample: {n_full}")
    print(f"Has nextGPA: {has_y.sum()}")
    print(f"Left school: {int(left.sum())}")
    print(f"Left & missing nextGPA: {int((left == 1).sum() - (left[has_y] == 1).sum())}")

    # ---- Common first stage on FULL sample ----
    X_full = df_full[X_COLS].values
    q_full = df_full[Q_COL].values
    D_full = (q_full < 0).astype(float)
    n_tr_full = int(D_full.sum())

    X_design = np.column_stack((np.ones(n_full), X_full))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q_full, rcond=None)
    eta_full = q_full - X_design @ gamma_hat
    eta_Tr_full = eta_full[D_full == 1]

    gX_sorted = np.sort(q_full - eta_full)

    print(f"\nFirst stage (full sample):")
    print(f"  N = {n_full}, N_treated = {n_tr_full}")
    print(f"  Treated eta: 5th={np.percentile(eta_Tr_full, 5):.3f}, "
          f"95th={np.percentile(eta_Tr_full, 95):.3f}")

    # ---- Outcome 1: nextGPA (stayers only) ----
    stayer_mask = has_y  # stayers have observed nextGPA
    X_s = X_full[stayer_mask]
    q_s = q_full[stayer_mask]
    y_s = df_full.loc[stayer_mask, Y_COL].values
    D_s = D_full[stayer_mask]
    eta_s = eta_full[stayer_mask]
    eta_Tr_s = eta_s[D_s == 1]
    n_s = len(y_s)
    n_tr_s = int(D_s.sum())

    info_gpa, omega_gpa = run_plm(X_s, q_s, y_s, D_s, eta_s, eta_Tr_s, n_s, n_tr_s)

    # Evaluate alpha_GPA for ALL full-sample observations
    Phi_gpa_all = _eval_basis(eta_full, info_gpa)
    alpha_gpa_all = Phi_gpa_all @ omega_gpa

    # Trimmed average
    tr_lo = np.percentile(eta_Tr_full, 5)
    tr_hi = np.percentile(eta_Tr_full, 95)
    tr_mask = (eta_Tr_full >= tr_lo) & (eta_Tr_full <= tr_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr_full[tr_mask], info_gpa)
    alpha_gpa_hat = np.mean(Phi_Tr_eval @ omega_gpa)
    print(f"\nalpha_GPA (stayers): avg = {alpha_gpa_hat:.4f}")

    # ---- Outcome 2: left_school (full sample) ----
    info_drop, omega_drop = run_plm(
        X_full, q_full, left, D_full, eta_full, eta_Tr_full, n_full, n_tr_full
    )

    Phi_drop_all = _eval_basis(eta_full, info_drop)
    alpha_drop_all = Phi_drop_all @ omega_drop

    Phi_Tr_drop = _eval_basis(eta_Tr_full[tr_mask], info_drop)
    alpha_drop_hat = np.mean(Phi_Tr_drop @ omega_drop)
    print(f"alpha_dropout: avg = {alpha_drop_hat:.4f}")
    print(f"  (positive = probation increases dropout)")

    # ---- Compute components on phi grid ----
    phi_grid = np.linspace(-3.0, 3.0, 600)

    gpa_benefit = compute_expected(alpha_gpa_all, eta_full, gX_sorted, phi_grid)
    dropout_cost = compute_expected(alpha_drop_all, eta_full, gX_sorted, phi_grid)
    treat_prob = compute_treat_prob(eta_full, gX_sorted, phi_grid)

    # ---- Find lambda where phi* = 0 ----
    # U(phi; lam) = gpa_benefit(phi) - lam * dropout_cost(phi)
    # At phi = 0, dU/dphi = 0  =>  dGPA/dphi = lam * dDropout/dphi
    # Approximate derivatives at phi = 0
    idx0 = np.argmin(np.abs(phi_grid))
    dGPA = np.gradient(gpa_benefit, phi_grid)[idx0]
    dDrop = np.gradient(dropout_cost, phi_grid)[idx0]

    if dDrop > 0:
        lam_implied = dGPA / dDrop
    else:
        lam_implied = np.inf

    print(f"\n=== Revealed preference ===")
    print(f"At phi=0: dGPA/dphi = {dGPA:.6f}")
    print(f"At phi=0: dDropout/dphi = {dDrop:.6f}")
    print(f"Implied lambda (cost per induced dropout in GPA units): {lam_implied:.4f}")

    # ---- Sweep lambda values ----
    lam_vals = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    print(f"\n{'lambda':>8s}  {'phi*':>8s}  {'GPA benefit':>12s}  "
          f"{'Dropouts':>10s}  {'Utility':>10s}")
    print("-" * 60)

    for lam in lam_vals:
        util = gpa_benefit - lam * dropout_cost
        oi = np.argmax(util)
        op = phi_grid[oi]
        print(f"{lam:8.1f}  {op:8.3f}  {gpa_benefit[oi]:12.4f}  "
              f"{dropout_cost[oi]:10.4f}  {util[oi]:10.4f}")

    # ---- Figure 1: Alpha functions for both outcomes ----
    eta_grid = np.linspace(tr_lo, tr_hi, 500)
    Phi_grid_gpa = _eval_basis(eta_grid, info_gpa)
    Phi_grid_drop = _eval_basis(eta_grid, info_drop)
    alpha_gpa_grid = Phi_grid_gpa @ omega_gpa
    alpha_drop_grid = Phi_grid_drop @ omega_drop

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))

    ax1a.plot(eta_grid, alpha_gpa_grid, "b-", linewidth=2,
              label=f"$\\alpha_{{GPA}}(\\eta)$ (avg={alpha_gpa_hat:.3f})")
    ax1a.axhline(0, color="black", linewidth=0.5)
    ax1a.axhline(alpha_gpa_hat, color="red", linestyle="--", linewidth=1,
                 label=f"avg = {alpha_gpa_hat:.3f}")
    ax1a.set_xlabel(r"$\eta$")
    ax1a.set_ylabel(r"$\alpha_{GPA}(\eta)$")
    ax1a.set_title("Treatment effect on nextGPA (stayers)")
    ax1a.legend(fontsize=9)

    ax1b.plot(eta_grid, alpha_drop_grid, "r-", linewidth=2,
              label=f"$\\alpha_{{dropout}}(\\eta)$ (avg={alpha_drop_hat:.3f})")
    ax1b.axhline(0, color="black", linewidth=0.5)
    ax1b.axhline(alpha_drop_hat, color="orange", linestyle="--", linewidth=1,
                 label=f"avg = {alpha_drop_hat:.3f}")
    ax1b.set_xlabel(r"$\eta$")
    ax1b.set_ylabel(r"$\alpha_{dropout}(\eta)$")
    ax1b.set_title("Treatment effect on dropout")
    ax1b.legend(fontsize=9)

    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "two_outcome_fig1_alphas.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved two_outcome_fig1_alphas.png")

    # ---- Figure 2: Pareto frontier (GPA benefit vs induced dropouts) ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(dropout_cost, gpa_benefit, "b-", linewidth=2)

    # Mark specific phi values
    for phi_mark, marker, label in [
        (0.0, "ro", r"$\phi=0$ (current)"),
        (-0.5, "gs", r"$\phi=-0.5$"),
        (0.5, "m^", r"$\phi=0.5$"),
        (1.0, "cv", r"$\phi=1.0$"),
    ]:
        idx = np.argmin(np.abs(phi_grid - phi_mark))
        ax2.plot(dropout_cost[idx], gpa_benefit[idx], marker,
                 markersize=10, label=label)

    # Implied lambda slope at phi=0
    idx0 = np.argmin(np.abs(phi_grid))
    if np.isfinite(lam_implied) and lam_implied > 0:
        x0, y0 = dropout_cost[idx0], gpa_benefit[idx0]
        dx = 0.01
        ax2.plot([x0 - dx, x0 + dx],
                 [y0 - lam_implied * dx, y0 + lam_implied * dx],
                 "k--", linewidth=1.5,
                 label=f"slope = $\\lambda^*$ = {lam_implied:.2f}")

    ax2.set_xlabel("Induced dropouts (treatment effect on dropout)")
    ax2.set_ylabel("GPA benefit (treatment effect on nextGPA)")
    ax2.set_title("Pareto frontier: GPA benefit vs. induced dropouts")
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "two_outcome_fig2_pareto.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved two_outcome_fig2_pareto.png")

    # ---- Figure 3: Utility for different lambda ----
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lam_vals)))
    for lam, col in zip(lam_vals, colors):
        util = gpa_benefit - lam * dropout_cost
        oi = np.argmax(util)
        op = phi_grid[oi]
        ax3.plot(phi_grid, util, color=col, linewidth=1.5,
                 label=f"$\\lambda$={lam:.0f}, $\\phi^*$={op:.2f}")

    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax3.set_xlabel(r"Threshold $\phi$")
    ax3.set_ylabel("Utility")
    ax3.set_title(r"$U(\phi) = \mathrm{GPA\ benefit} - \lambda \times \mathrm{induced\ dropouts}$")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "two_outcome_fig3_utility.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved two_outcome_fig3_utility.png")

    # ---- Figure 4: phi*(lambda) ----
    lam_fine = np.linspace(0, 15, 100)
    phi_stars = np.zeros(len(lam_fine))
    for i, lam in enumerate(lam_fine):
        util = gpa_benefit - lam * dropout_cost
        phi_stars[i] = phi_grid[np.argmax(util)]

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(lam_fine, phi_stars, "b.-", markersize=3)
    ax4.axhline(0, color="gray", linestyle=":", linewidth=1, label="Current cutoff")
    if np.isfinite(lam_implied):
        ax4.axvline(lam_implied, color="red", linestyle="--", linewidth=1.5,
                    label=f"$\\lambda^*$ = {lam_implied:.2f} (current policy optimal)")
    ax4.set_xlabel(r"$\lambda$ (GPA-equivalent cost per induced dropout)")
    ax4.set_ylabel(r"Optimal threshold $\phi^*$")
    ax4.set_title(r"$\phi^*(\lambda)$: optimal threshold vs. dropout cost")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTDIR, "two_outcome_fig4_phi_vs_lambda.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved two_outcome_fig4_phi_vs_lambda.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
