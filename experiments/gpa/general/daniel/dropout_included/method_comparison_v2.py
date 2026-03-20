"""
Compare nonparametric methods under:
  - Dropout imputation: Y = -GPA_year1 for dropouts
  - GPA-dependent cost: c * GPA_year1

Methods:
  1. Current: B-spline + ridge (uniform lambda)
  2. Gaussian treatment basis (B-spline baseline)
  3. Heavy ridge on treatment coefficients (50x)
  4. Variance thresholding on treatment basis
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


# ---- Estimation methods ----

def estimate_current(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _bspline_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam = 0.1 / np.sqrt(n)
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)
    omega_treat = be[p + n_basis:]
    return info, omega_treat, "B-spline + ridge"


def estimate_gaussian(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
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
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)
    omega_treat = be[p + n_base:]
    return info_treat, omega_treat, "Gaussian treat"


def estimate_heavy_ridge(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _bspline_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    H = np.column_stack((X, Phi, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam_base = 0.1 / np.sqrt(n)
    lam_treat = 5.0 / np.sqrt(n)
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:p + n_basis, p:p + n_basis], lam_base)
    np.fill_diagonal(P[p + n_basis:, p + n_basis:], lam_treat)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)
    omega_treat = be[p + n_basis:]
    return info, omega_treat, "Heavy ridge (50x)"


def estimate_var_threshold(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _bspline_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi
    v_treat = DPhi.var(axis=0, ddof=1)
    thresh = 0.03 * v_treat.max()
    keep = v_treat > thresh
    n_dropped = n_basis - keep.sum()

    DPhi_active = DPhi[:, keep]
    H = np.column_stack((X, Phi, DPhi_active))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam = 0.1 / np.sqrt(n)
    P_mat = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P_mat[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P_mat, H.T @ y)

    omega_treat = np.zeros(n_basis)
    omega_treat[keep] = be[p + n_basis:]
    return info, omega_treat, f"Var threshold ({n_dropped} dropped)"


# ---- Utility ----

def compute_utility(alpha_all, gpa1, eta_hat, gX_sorted, phi_grid, c):
    n_total = len(gX_sorted)
    util = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_hat
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        benefit = np.mean(alpha_all * probs)
        cost = c * np.mean(gpa1 * probs)
        util[j] = benefit - cost
    return util


# ---- Main ----

def main():
    df = pd.read_csv(DATA_PATH)
    n_full = len(df)

    # Impute Y = -GPA_year1 for dropouts
    has_y = df[Y_COL].notna().values
    left = df["left_school"].values.astype(float)
    impute_mask = (~has_y) & (left == 1)
    df.loc[impute_mask, Y_COL] = -df.loc[impute_mask, "GPA_year1"]
    df = df[df[Y_COL].notna()].copy()
    n = len(df)

    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    gpa1 = df["GPA_year1"].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    print(f"N = {n}, N_treated = {n_tr}, Imputed dropouts = {impute_mask.sum()}")

    # First stage
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat
    eta_Tr = eta_hat[D == 1]
    gX_sorted = np.sort(q - eta_hat)

    tr_lo = np.percentile(eta_Tr, 5)
    tr_hi = np.percentile(eta_Tr, 95)

    # Eval grid for alpha plots
    eta_grid = np.linspace(tr_lo, tr_hi, 500)
    phi_grid = np.linspace(-3.0, 3.0, 600)

    # Run all methods
    methods = [estimate_current, estimate_gaussian,
               estimate_heavy_ridge, estimate_var_threshold]

    results = []
    for method_fn in methods:
        info_t, omega_t, label = method_fn(X, q, y, D, eta_hat, eta_Tr, n, n_tr)

        Phi_grid = _eval_basis(eta_grid, info_t)
        alpha_grid = Phi_grid @ omega_t

        Phi_all = _eval_basis(eta_hat, info_t)
        alpha_all = Phi_all @ omega_t

        tr_mask = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
        Phi_Tr = _eval_basis(eta_Tr[tr_mask], info_t)
        alpha_hat = np.mean(Phi_Tr @ omega_t)

        results.append({
            "label": label, "alpha_grid": alpha_grid,
            "alpha_all": alpha_all, "alpha_hat": alpha_hat,
        })
        print(f"\n{label}:")
        print(f"  alpha_hat = {alpha_hat:.4f}")
        print(f"  alpha range: [{alpha_grid.min():.4f}, {alpha_grid.max():.4f}]")

    # ---- Figure 1: Alpha functions ----
    colors = ["steelblue", "red", "green", "purple"]
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for res, col in zip(results, colors):
        ax1.plot(eta_grid, res["alpha_grid"], color=col, linewidth=2,
                 label=f'{res["label"]} (avg={res["alpha_hat"]:.3f})')
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(tr_lo, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(tr_hi, color="gray", linestyle=":", alpha=0.5,
                label="Treated [5th, 95th]")

    ax1h = ax1.twinx()
    in_trim = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
    ax1h.hist(eta_Tr[in_trim], bins=50, alpha=0.1, color="gray", density=True)
    ax1h.set_ylabel("Density (treated)", color="gray")
    ax1h.tick_params(axis="y", labelcolor="gray")
    ax1.set_zorder(ax1h.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\alpha(\eta)$")
    ax1.set_title("Alpha comparison (dropout imputation)")
    ax1.legend(loc="upper left", fontsize=9)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "meth_v2_fig1_alpha.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved meth_v2_fig1_alpha.png")

    # ---- Figure 2: phi*(c) with GPA-dependent cost ----
    c_fine = np.linspace(0.0, 0.15, 80)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for res, col in zip(results, colors):
        phi_stars = np.zeros(len(c_fine))
        for i, c in enumerate(c_fine):
            util = compute_utility(res["alpha_all"], gpa1, eta_hat,
                                   gX_sorted, phi_grid, c)
            phi_stars[i] = phi_grid[np.argmax(util)]
        ax2.plot(c_fine, phi_stars, color=col, linewidth=2, label=res["label"])

    ax2.axhline(0, color="gray", linestyle=":", linewidth=1, label="Current cutoff")
    ax2.set_xlabel("Cost parameter c")
    ax2.set_ylabel(r"Optimal threshold $\phi^*$")
    ax2.set_title(r"$\phi^*(c)$ with GPA-dependent cost + dropout imputation")
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "meth_v2_fig2_phi_star.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved meth_v2_fig2_phi_star.png")

    # ---- Figure 3: Utility at c=0.04 ----
    c_show = 0.04
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for res, col in zip(results, colors):
        util = compute_utility(res["alpha_all"], gpa1, eta_hat,
                               gX_sorted, phi_grid, c_show)
        oi = np.argmax(util)
        op = phi_grid[oi]
        ax3.plot(phi_grid, util, color=col, linewidth=2,
                 label=f'{res["label"]}: $\\phi^*$={op:.2f}')

    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax3.set_xlabel(r"Threshold $\phi$")
    ax3.set_ylabel("Utility")
    ax3.set_title(f"Utility comparison at c = {c_show}")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "meth_v2_fig3_utility.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved meth_v2_fig3_utility.png")

    # ---- Summary table ----
    print(f"\n{'Method':<28s}  {'alpha_hat':>10s}", end="")
    for c in [0.0, 0.02, 0.04, 0.06]:
        print(f"  {'phi*(c='+f'{c:.2f}'+')':>14s}", end="")
    print()
    print("-" * 90)
    for res in results:
        print(f'{res["label"]:<28s}  {res["alpha_hat"]:10.4f}', end="")
        for c in [0.0, 0.02, 0.04, 0.06]:
            util = compute_utility(res["alpha_all"], gpa1, eta_hat,
                                   gX_sorted, phi_grid, c)
            op = phi_grid[np.argmax(util)]
            print(f"  {op:14.3f}", end="")
        print()

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
