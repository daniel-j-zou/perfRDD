"""
GPA-dependent cost of probation, stayers only (no dropout imputation).

U(phi) = E[alpha(eta) * P(Q<phi|eta)] - c * E[GPA_year1 * P(Q<phi|eta)]

The cost of treating student i is c * GPA_year1_i: students with higher
first-year GPA have more to lose from probation (opportunity cost).

Since GPA_year1 is positively correlated with g(X), the marginal cost
of raising phi increases — creating interior optima.

Uses Gaussian treatment basis.
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


def main():
    # ---- Load stayers only ----
    df = pd.read_csv(DATA_PATH)
    df = df[df[Y_COL].notna()].copy()
    n = len(df)

    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    gpa1 = df["GPA_year1"].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    print(f"Stayers only: N = {n}, N_treated = {n_tr}")
    print(f"GPA_year1: mean={gpa1.mean():.3f}, std={gpa1.std():.3f}")
    print(f"GPA_year1 among treated: mean={gpa1[D==1].mean():.3f}")
    print(f"GPA_year1 among control: mean={gpa1[D==0].mean():.3f}")

    # ---- First stage ----
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat
    eta_Tr = eta_hat[D == 1]

    gX_sorted = np.sort(q - eta_hat)

    # ---- Gaussian-treat PLM ----
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

    lam_reg = 0.1 / np.sqrt(n)
    P_mat = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P_mat[p:, p:], lam_reg)
    be = np.linalg.solve(H.T @ H + n * P_mat, H.T @ y)
    omega_treat = be[p + n_base:]

    # Alpha for all observations
    Phi_all = _eval_basis(eta_hat, info_treat)
    alpha_all = Phi_all @ omega_treat

    # Trimmed average
    tr_mask = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr[tr_mask], info_treat)
    alpha_hat = np.mean(Phi_Tr_eval @ omega_treat)
    print(f"\nalpha_hat (trimmed treated avg) = {alpha_hat:.4f}")

    # ---- Correlation check ----
    corr_gpa_q = np.corrcoef(gpa1, q)[0, 1]
    gX = q - eta_hat
    corr_gpa_gx = np.corrcoef(gpa1, gX)[0, 1]
    print(f"\nCorrelations:")
    print(f"  corr(GPA_year1, Q) = {corr_gpa_q:.4f}")
    print(f"  corr(GPA_year1, g(X)) = {corr_gpa_gx:.4f}")

    # ---- Utility computation ----
    phi_grid = np.linspace(-3.0, 3.0, 600)
    n_total = len(gX_sorted)

    def compute_util(c_val):
        """U = E[alpha * P(Q<phi|eta)] - c * E[GPA1 * P(Q<phi|eta)]"""
        util = np.zeros(len(phi_grid))
        for j, phi in enumerate(phi_grid):
            thresh = phi - eta_hat
            probs = np.searchsorted(gX_sorted, thresh) / n_total
            benefit = np.mean(alpha_all * probs)
            cost = c_val * np.mean(gpa1 * probs)
            util[j] = benefit - cost
        return util

    def compute_util_fixed(c_val):
        """U = E[alpha * P(Q<phi|eta)] - c * E[P(Q<phi|eta)]  (fixed cost)"""
        util = np.zeros(len(phi_grid))
        for j, phi in enumerate(phi_grid):
            thresh = phi - eta_hat
            probs = np.searchsorted(gX_sorted, thresh) / n_total
            benefit = np.mean(alpha_all * probs)
            cost = c_val * np.mean(probs)
            util[j] = benefit - cost
        return util

    # ---- Sweep GPA-dependent cost ----
    c_vals = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]

    print(f"\n{'c':>6s}  {'phi*':>8s}  {'U(phi*)':>10s}  {'U(0)':>10s}")
    print("-" * 40)
    print("--- GPA-dependent cost: c * GPA_year1 ---")
    for c in c_vals:
        util = compute_util(c)
        oi = np.argmax(util)
        op = phi_grid[oi]
        u0 = np.interp(0.0, phi_grid, util)
        print(f"{c:6.3f}  {op:8.3f}  {util[oi]:10.4f}  {u0:10.4f}")

    print("\n--- Fixed cost: c ---")
    for c in c_vals:
        util = compute_util_fixed(c)
        oi = np.argmax(util)
        op = phi_grid[oi]
        u0 = np.interp(0.0, phi_grid, util)
        print(f"{c:6.3f}  {op:8.3f}  {util[oi]:10.4f}  {u0:10.4f}")

    # ---- Figure 1: Utility curves, GPA-dependent cost ----
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.coolwarm

    select_c = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
    for i, c in enumerate(select_c):
        color = cmap(i / (len(select_c) - 1))
        util = compute_util(c)
        oi = np.argmax(util)
        op = phi_grid[oi]
        ax1a.plot(phi_grid, util, color=color, linewidth=1.5,
                  label=f"c={c:.2f}, $\\phi^*$={op:.2f}")

    ax1a.axhline(0, color="black", linewidth=0.5)
    ax1a.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax1a.set_xlabel(r"Threshold $\phi$")
    ax1a.set_ylabel("Utility")
    ax1a.set_title(r"GPA-dependent cost: $c \times GPA_1$")
    ax1a.legend(fontsize=8)

    for i, c in enumerate(select_c):
        color = cmap(i / (len(select_c) - 1))
        util = compute_util_fixed(c)
        oi = np.argmax(util)
        op = phi_grid[oi]
        ax1b.plot(phi_grid, util, color=color, linewidth=1.5,
                  label=f"c={c:.2f}, $\\phi^*$={op:.2f}")

    ax1b.axhline(0, color="black", linewidth=0.5)
    ax1b.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax1b.set_xlabel(r"Threshold $\phi$")
    ax1b.set_ylabel("Utility")
    ax1b.set_title("Fixed cost: $c$")
    ax1b.legend(fontsize=8)

    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "gpa_cost_fig1_utility.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved gpa_cost_fig1_utility.png")

    # ---- Figure 2: phi*(c) for both cost types ----
    c_fine = np.linspace(0.0, 0.25, 100)
    phi_star_gpa = np.zeros(len(c_fine))
    phi_star_fix = np.zeros(len(c_fine))

    for i, c in enumerate(c_fine):
        util_g = compute_util(c)
        util_f = compute_util_fixed(c)
        phi_star_gpa[i] = phi_grid[np.argmax(util_g)]
        phi_star_fix[i] = phi_grid[np.argmax(util_f)]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(c_fine, phi_star_gpa, "b-", linewidth=2,
             label=r"GPA-dependent: $c \times GPA_1$")
    ax2.plot(c_fine, phi_star_fix, "r--", linewidth=2,
             label="Fixed: $c$")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1,
                label="Current cutoff")
    ax2.set_xlabel("Cost parameter c")
    ax2.set_ylabel(r"Optimal threshold $\phi^*$")
    ax2.set_title(r"$\phi^*(c)$: GPA-dependent vs. fixed cost")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "gpa_cost_fig2_phi_star.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved gpa_cost_fig2_phi_star.png")

    # ---- Figure 3: Marginal benefit vs marginal cost ----
    # dU/dphi = E[alpha(eta) * f_gX(phi-eta)] - c * E[GPA1 * f_gX(phi-eta)]
    c_show = 0.06
    util_g = compute_util(c_show)

    # Numerical marginal benefit and cost
    dphi = phi_grid[1] - phi_grid[0]
    mb = np.zeros(len(phi_grid))
    mc = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_hat
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        if j > 0:
            thresh_prev = phi_grid[j-1] - eta_hat
            probs_prev = np.searchsorted(gX_sorted, thresh_prev) / n_total
            dprobs = (probs - probs_prev) / dphi
            mb[j] = np.mean(alpha_all * dprobs)
            mc[j] = c_show * np.mean(gpa1 * dprobs)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(phi_grid[1:], mb[1:], "b-", linewidth=2, label="Marginal benefit")
    ax3.plot(phi_grid[1:], mc[1:], "r-", linewidth=2,
             label=f"Marginal cost (c={c_show})")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axvline(0, color="gray", linestyle=":", linewidth=1)

    # Mark crossing point
    diff = mb - mc
    crossings = np.where(np.diff(np.sign(diff[1:])))[0] + 1
    for cx in crossings:
        ax3.axvline(phi_grid[cx], color="green", linestyle="--", alpha=0.5)
        ax3.annotate(f"$\\phi^*$ = {phi_grid[cx]:.2f}",
                     xy=(phi_grid[cx], mb[cx]),
                     xytext=(phi_grid[cx] + 0.3, mb[cx] + 0.001),
                     fontsize=10)

    ax3.set_xlabel(r"Threshold $\phi$")
    ax3.set_ylabel("Marginal utility")
    ax3.set_title(f"Marginal benefit vs. marginal GPA-dependent cost (c={c_show})")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "gpa_cost_fig3_marginal.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved gpa_cost_fig3_marginal.png")

    # ---- Figure 4: Alpha function + average GPA1 at each phi ----
    eta_grid = np.linspace(tr_lo, tr_hi, 500)
    Phi_grid_t = _eval_basis(eta_grid, info_treat)
    alpha_grid = Phi_grid_t @ omega_treat

    # Average GPA_year1 of marginal students at each phi
    avg_gpa_at_phi = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        # Students with Q near phi (within a small window)
        window = 0.1
        near = np.abs(q - phi) < window
        if near.sum() > 10:
            avg_gpa_at_phi[j] = gpa1[near].mean()
        else:
            avg_gpa_at_phi[j] = np.nan

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

    ax4a.plot(eta_grid, alpha_grid, "b-", linewidth=2)
    ax4a.axhline(0, color="black", linewidth=0.5)
    ax4a.axhline(alpha_hat, color="red", linestyle="--", linewidth=1,
                 label=f"avg = {alpha_hat:.3f}")
    ax4a.set_xlabel(r"$\eta$")
    ax4a.set_ylabel(r"$\alpha(\eta)$")
    ax4a.set_title("Treatment effect (stayers only)")
    ax4a.legend()

    valid = ~np.isnan(avg_gpa_at_phi)
    ax4b.plot(phi_grid[valid], avg_gpa_at_phi[valid], "r-", linewidth=2)
    ax4b.set_xlabel(r"Threshold $\phi$")
    ax4b.set_ylabel("Mean GPA_year1")
    ax4b.set_title("Average GPA of marginal students at threshold $\\phi$")

    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTDIR, "gpa_cost_fig4_context.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved gpa_cost_fig4_context.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
