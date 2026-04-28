"""
GPA-dependent cost with dropout imputation.

Dropouts get Y = -GPA_year1 (negative outcome), then PLM runs on full sample.
Cost of probation is c * GPA_year1 per student.

U(phi) = E[alpha(eta) * P(Q<phi|eta)] - c * E[GPA_year1 * P(Q<phi|eta)]

where alpha already internalizes dropout harm (via imputation).

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
    # ---- Load full sample ----
    df = pd.read_csv(DATA_PATH)
    n_full = len(df)

    X_full = df[X_COLS].values
    q_full = df[Q_COL].values
    gpa1_full = df["GPA_year1"].values
    D_full = (q_full < 0).astype(float)
    left_full = df["left_school"].values.astype(float)
    has_y = df[Y_COL].notna().values

    n_tr_full = int(D_full.sum())
    n_dropouts_tr = int((D_full * left_full).sum())
    n_dropouts_con = int(((1 - D_full) * left_full).sum())

    print(f"Full sample: {n_full}")
    print(f"Treated: {n_tr_full}, Control: {n_full - n_tr_full}")
    print(f"Dropouts among treated: {n_dropouts_tr} ({n_dropouts_tr/n_tr_full:.1%})")
    print(f"Dropouts among control: {n_dropouts_con} ({n_dropouts_con/(n_full-n_tr_full):.1%})")

    # ---- First stage on FULL sample ----
    X_design = np.column_stack((np.ones(n_full), X_full))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q_full, rcond=None)
    eta_full = q_full - X_design @ gamma_hat
    eta_Tr_full = eta_full[D_full == 1]

    gX_sorted = np.sort(q_full - eta_full)

    # ---- Impute Y = -GPA_year1 for dropouts, then PLM on full sample ----
    y_full = df[Y_COL].values.copy()
    impute_mask = (~has_y) & (left_full == 1)
    y_full[impute_mask] = -gpa1_full[impute_mask]
    # Drop any remaining missing (non-dropout missing Y)
    valid = ~np.isnan(y_full)
    n_still_missing = n_full - valid.sum() - impute_mask.sum()

    n_imputed = impute_mask.sum()
    avg_imputed = y_full[impute_mask].mean()
    print(f"\nImputed Y = -GPA_year1 for {n_imputed} dropouts "
          f"(avg imputed Y = {avg_imputed:.3f})")
    print(f"Dropped {n_still_missing} with missing Y and not left_school")

    X_v = X_full[valid]
    q_v = q_full[valid]
    y_v = y_full[valid]
    D_v = D_full[valid]
    eta_v = eta_full[valid]
    gpa1_v = gpa1_full[valid]
    eta_Tr_v = eta_v[D_v == 1]
    n_v = len(y_v)
    n_tr_v = int(D_v.sum())

    print(f"PLM sample: {n_v}, Treated: {n_tr_v}")

    support = (np.percentile(eta_v, 0.5), np.percentile(eta_v, 99.5))
    kn_base = max(4, int(round(n_tr_v ** (1.0 / 3.0))))
    info_base = _bspline_params(kn_base, support)
    Phi_base = _eval_basis(eta_v, info_base)
    n_base = Phi_base.shape[1]

    tr_lo = np.percentile(eta_Tr_v, 5)
    tr_hi = np.percentile(eta_Tr_v, 95)
    kn_treat = max(4, int(round(n_tr_v ** (1.0 / 3.0))))
    centers = np.linspace(tr_lo, tr_hi, kn_treat)
    info_treat = _gaussian_params(centers, bwf=1.5)
    Phi_treat = _eval_basis(eta_v, info_treat)

    DPhi = D_v[:, None] * Phi_treat
    H = np.column_stack((X_v, Phi_base, DPhi))
    p = X_v.shape[1]
    total_cols = H.shape[1]

    lam_reg = 0.1 / np.sqrt(n_v)
    P_mat = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P_mat[p:, p:], lam_reg)
    be = np.linalg.solve(H.T @ H + n_v * P_mat, H.T @ y_v)
    omega_treat = be[p + n_base:]

    # Alpha for valid sample
    alpha_all = Phi_treat @ omega_treat

    # Trimmed average on treated
    tr_mask = (eta_Tr_v >= tr_lo) & (eta_Tr_v <= tr_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr_v[tr_mask], info_treat)
    alpha_hat = np.mean(Phi_Tr_eval @ omega_treat)
    print(f"alpha (with dropout imputation, trimmed avg) = {alpha_hat:.4f}")

    # Use valid-sample arrays for utility
    eta_full = eta_v
    gpa1_full = gpa1_v
    alpha_effective = alpha_all
    q_full = q_v
    D_full = D_v
    left_full = left_full[valid]
    gX_sorted = np.sort(q_v - eta_v)
    n_full = n_v
    eta_Tr_full = eta_Tr_v

    # ---- Utility computation ----
    phi_grid = np.linspace(-3.0, 3.0, 600)
    n_total = len(gX_sorted)

    def compute_util(c_val):
        """U = E[alpha_eff * P(Q<phi|eta)] - c * E[GPA1 * P(Q<phi|eta)]"""
        util = np.zeros(len(phi_grid))
        for j, phi in enumerate(phi_grid):
            thresh = phi - eta_full
            probs = np.searchsorted(gX_sorted, thresh) / n_total
            benefit = np.mean(alpha_effective * probs)
            cost = c_val * np.mean(gpa1_full * probs)
            util[j] = benefit - cost
        return util

    # ---- Sweep cost values ----
    c_vals = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

    print(f"\n{'c':>6s}  {'phi*':>8s}  {'U(phi*)':>10s}  {'U(0)':>10s}  "
          f"{'treat_frac':>11s}  {'GPA1_marginal':>14s}")
    print("-" * 75)

    results = []
    for c in c_vals:
        util = compute_util(c)
        oi = np.argmax(util)
        op = phi_grid[oi]
        u0 = np.interp(0.0, phi_grid, util)

        # Fraction treated and avg GPA of marginal students at optimal phi
        window = 0.15
        near_opt = np.abs(q_full - op) < window
        gpa_marginal = gpa1_full[near_opt].mean() if near_opt.sum() > 10 else np.nan

        treat_frac = np.mean(q_full < op)

        print(f"{c:6.3f}  {op:8.3f}  {util[oi]:10.4f}  {u0:10.4f}  "
              f"{treat_frac:11.3f}  {gpa_marginal:14.3f}")
        results.append({"c": c, "phi": op, "util": util, "treat_frac": treat_frac})

    # ---- Figure 1: Utility curves ----
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.coolwarm
    for i, res in enumerate(results):
        if res["c"] > 0.15:
            continue
        color = cmap(i / (len(results) - 2))
        ax1.plot(phi_grid, res["util"], color=color, linewidth=1.5,
                 label=f'c={res["c"]:.2f}, $\\phi^*$={res["phi"]:.2f} '
                       f'({res["treat_frac"]:.0%} treated)')

    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1,
                label="Current cutoff")
    ax1.set_xlabel(r"Threshold $\phi$")
    ax1.set_ylabel("Utility")
    ax1.set_title("Utility with GPA-dependent cost (dropouts imputed as -GPA_year1)")
    ax1.legend(fontsize=8, loc="best")
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "gpa_cost_v2_fig1_utility.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved gpa_cost_v2_fig1_utility.png")

    # ---- Figure 2: phi*(c) ----
    c_fine = np.linspace(0.0, 0.20, 100)
    phi_stars = np.zeros(len(c_fine))
    treat_fracs = np.zeros(len(c_fine))
    for i, c in enumerate(c_fine):
        util = compute_util(c)
        idx = np.argmax(util)
        phi_stars[i] = phi_grid[idx]
        treat_fracs[i] = np.mean(q_full < phi_grid[idx])

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    ax2a.plot(c_fine, phi_stars, "b-", linewidth=2)
    ax2a.axhline(0, color="gray", linestyle=":", linewidth=1,
                 label="Current cutoff")
    ax2a.set_xlabel("Cost parameter c")
    ax2a.set_ylabel(r"Optimal threshold $\phi^*$")
    ax2a.set_title(r"$\phi^*(c)$")
    ax2a.legend()

    ax2b.plot(c_fine, treat_fracs * 100, "r-", linewidth=2)
    ax2b.axhline(np.mean(q_full < 0) * 100, color="gray", linestyle=":",
                 linewidth=1, label="Current (16.1%)")
    ax2b.set_xlabel("Cost parameter c")
    ax2b.set_ylabel("% students on probation")
    ax2b.set_title("Fraction treated at optimal threshold")
    ax2b.legend()

    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "gpa_cost_v2_fig2_phi_star.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved gpa_cost_v2_fig2_phi_star.png")

    # ---- Figure 3: What c means in dollar terms ----
    # Cost per student = c * GPA_year1. Show distribution at different phi* values.
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    c_examples = [0.04, 0.06, 0.08, 0.10]
    for c in c_examples:
        util = compute_util(c)
        op = phi_grid[np.argmax(util)]
        treated_at_opt = q_full < op
        costs = c * gpa1_full[treated_at_opt]
        ax3.hist(costs, bins=50, alpha=0.4, density=True,
                 label=f"c={c:.2f}: avg cost={costs.mean():.3f}, "
                       f"$\\phi^*$={op:.2f}")

    ax3.set_xlabel("Per-student cost (c * GPA_year1)")
    ax3.set_ylabel("Density")
    ax3.set_title("Distribution of treatment cost among optimally treated students")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "gpa_cost_v2_fig3_cost_dist.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved gpa_cost_v2_fig3_cost_dist.png")

    # ---- Figure 4: Alpha function + context ----
    eta_grid = np.linspace(tr_lo, tr_hi, 500)
    Phi_grid_t = _eval_basis(eta_grid, info_treat)
    alpha_grid = Phi_grid_t @ omega_treat

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

    ax4a.plot(eta_grid, alpha_grid, "b-", linewidth=2)
    ax4a.axhline(0, color="black", linewidth=0.5)
    ax4a.axhline(alpha_hat, color="red", linestyle="--", linewidth=1,
                 label=f"avg = {alpha_hat:.3f}")
    ax4a.set_xlabel(r"$\eta$")
    ax4a.set_ylabel(r"$\alpha_{GPA}(\eta)$")
    ax4a.set_title("Treatment effect (dropout-inclusive)")
    ax4a.legend()

    # Dropout rate among treated by eta decile
    n_dec = 10
    pcts = np.percentile(eta_Tr_full, np.linspace(0, 100, n_dec + 1))
    centers_dec = 0.5 * (pcts[:-1] + pcts[1:])
    drop_rates = np.zeros(n_dec)
    for j in range(n_dec):
        mask = (eta_Tr_full >= pcts[j]) & (eta_Tr_full < pcts[j + 1])
        if j == n_dec - 1:
            mask = (eta_Tr_full >= pcts[j]) & (eta_Tr_full <= pcts[j + 1])
        drop_rates[j] = left_full[D_full == 1][mask].mean()

    ax4b.bar(centers_dec, drop_rates * 100, width=0.12, alpha=0.7, color="red")
    ax4b.set_xlabel(r"$\eta$ (decile centers)")
    ax4b.set_ylabel("Dropout rate (%)")
    ax4b.set_title("Dropout rate by eta decile (treated)")

    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTDIR, "gpa_cost_v2_fig4_context.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved gpa_cost_v2_fig4_context.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
