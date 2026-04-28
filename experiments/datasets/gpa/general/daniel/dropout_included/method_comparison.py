"""
Compare three methods for naturally bounding alpha extrapolation:
  1. Gaussian basis centered on treated eta distribution
  2. Heavier ridge penalty on treatment spline coefficients
  3. Variance thresholding on treatment basis functions

All methods use the same first stage and dropout imputation.
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
    os.path.dirname(OUTDIR), "..", "..", "Dep_Data", "final_processed_data.csv"
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
    """Gaussian basis with given centers and bandwidth factor."""
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


# ---- Load data ----

def load_data():
    df = pd.read_csv(DATA_PATH)
    missing_y = df[Y_COL].isna()
    left = df["left_school"] == 1
    impute_mask = missing_y & left
    df.loc[impute_mask, Y_COL] = 0.0 - df.loc[impute_mask, "GPA_year1"]
    df = df[df[Y_COL].notna()].copy()
    return df


# ---- Estimation methods ----

def estimate_current(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    """Current method: B-spline + ridge (baseline)."""
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

    return info, omega_treat, "Current (B-spline + ridge)"


def estimate_gaussian(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    """Method 1: Gaussian basis centered on treated eta for treatment effect."""
    # Baseline: B-spline on full support (unchanged)
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn_base = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info_base = _bspline_params(kn_base, support)
    Phi_base = _eval_basis(eta_hat, info_base)
    n_base = Phi_base.shape[1]

    # Treatment: Gaussian basis centered on treated eta quantiles
    kn_treat = max(4, int(round(n_tr ** (1.0 / 3.0))))
    tr_lo = np.percentile(eta_Tr, 5)
    tr_hi = np.percentile(eta_Tr, 95)
    centers = np.linspace(tr_lo, tr_hi, kn_treat)
    info_treat = _gaussian_params(centers, bwf=1.5)
    Phi_treat = _eval_basis(eta_hat, info_treat)
    n_treat = Phi_treat.shape[1]

    DPhi = D[:, None] * Phi_treat
    H = np.column_stack((X, Phi_base, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam = 0.1 / np.sqrt(n)
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)
    omega_treat = be[p + n_base:]

    return info_treat, omega_treat, "Gaussian (treat-centered)"


def estimate_heavy_ridge(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    """Method 2: Heavier ridge on treatment coefficients."""
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
    lam_treat = 5.0 / np.sqrt(n)  # 50x heavier on treatment
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:p + n_basis, p:p + n_basis], lam_base)
    np.fill_diagonal(P[p + n_basis:, p + n_basis:], lam_treat)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)
    omega_treat = be[p + n_basis:]

    return info, omega_treat, "Heavy ridge (50x on treat)"


def estimate_var_threshold(X, q, y, D, eta_hat, eta_Tr, n, n_tr):
    """Method 3: Variance thresholding on treatment basis."""
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info = _bspline_params(kn, support)
    Phi = _eval_basis(eta_hat, info)
    n_basis = Phi.shape[1]

    DPhi = D[:, None] * Phi

    # Threshold: drop treatment basis functions with low variance
    v_treat = DPhi.var(axis=0, ddof=1)
    thresh = 0.03 * v_treat.max()
    keep = v_treat > thresh
    n_kept = keep.sum()
    n_dropped = n_basis - n_kept

    DPhi_active = DPhi[:, keep]
    H = np.column_stack((X, Phi, DPhi_active))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam = 0.1 / np.sqrt(n)
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)

    # Map back to full omega_treat
    omega_treat = np.zeros(n_basis)
    omega_treat[keep] = be[p + n_basis:]

    return info, omega_treat, f"Var threshold ({n_dropped} dropped)"


# ---- Utility computation ----

def compute_utility(alpha_vals_obs, eta_hat, gX_sorted, phi_grid, c):
    """U(phi) = E[alpha(eta)*P(Q<phi|eta)] - c*E[P(Q<phi|eta)]"""
    util = np.zeros(len(phi_grid))
    n_total = len(gX_sorted)
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_hat
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        benefit = np.mean(alpha_vals_obs * probs)
        cost = c * np.mean(probs)
        util[j] = benefit - cost
    return util


# ---- Main ----

def main():
    df = load_data()
    n = len(df)
    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    # First stage (common)
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat
    eta_Tr = eta_hat[D == 1]

    gX_vals = q - eta_hat
    gX_sorted = np.sort(gX_vals)

    print(f"N = {n}, N_treated = {n_tr}")
    print(f"Treated eta: [{eta_Tr.min():.3f}, {eta_Tr.max():.3f}], "
          f"5th={np.percentile(eta_Tr, 5):.3f}, 95th={np.percentile(eta_Tr, 95):.3f}")

    # Eval grid
    eval_lo = np.percentile(eta_hat, 0.5)
    eval_hi = np.percentile(eta_hat, 99.5)
    eta_grid = np.linspace(eval_lo, eval_hi, 500)

    # Run all methods
    methods = [
        estimate_current,
        estimate_gaussian,
        estimate_heavy_ridge,
        estimate_var_threshold,
    ]

    results = []
    for method_fn in methods:
        info_treat, omega_treat, label = method_fn(
            X, q, y, D, eta_hat, eta_Tr, n, n_tr
        )
        # Evaluate alpha on grid
        Phi_grid = _eval_basis(eta_grid, info_treat)
        alpha_grid = Phi_grid @ omega_treat

        # Evaluate alpha for all observations (for utility)
        Phi_all = _eval_basis(eta_hat, info_treat)
        alpha_all = Phi_all @ omega_treat

        # Plug-in average (trimmed treated)
        tr_mask = (eta_Tr >= np.percentile(eta_Tr, 5)) & \
                  (eta_Tr <= np.percentile(eta_Tr, 95))
        Phi_Tr_eval = _eval_basis(eta_Tr[tr_mask], info_treat)
        alpha_hat = np.mean(Phi_Tr_eval @ omega_treat)

        results.append({
            "label": label,
            "info": info_treat,
            "omega": omega_treat,
            "alpha_grid": alpha_grid,
            "alpha_all": alpha_all,
            "alpha_hat": alpha_hat,
        })
        print(f"\n{label}:")
        print(f"  n_basis_treat = {len(omega_treat)}")
        print(f"  alpha_hat (trimmed treated avg) = {alpha_hat:.4f}")
        print(f"  alpha range on grid: [{alpha_grid.min():.4f}, {alpha_grid.max():.4f}]")

    # ---- Figure 1: Alpha functions comparison ----
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    colors = ["steelblue", "red", "green", "purple"]
    for res, col in zip(results, colors):
        ax1.plot(eta_grid, res["alpha_grid"], color=col, linewidth=2,
                 label=f'{res["label"]} (avg={res["alpha_hat"]:.3f})')
    ax1.axhline(0, color="black", linewidth=0.5)

    # Treated eta density
    ax1h = ax1.twinx()
    tr_lo = np.percentile(eta_Tr, 5)
    tr_hi = np.percentile(eta_Tr, 95)
    in_trim = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
    ax1h.hist(eta_Tr[in_trim], bins=50, alpha=0.1, color="gray", density=True)
    ax1h.set_ylabel("Density (treated)", color="gray")
    ax1h.tick_params(axis="y", labelcolor="gray")

    ax1.axvline(tr_lo, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(tr_hi, color="gray", linestyle=":", alpha=0.5,
                label="Treated [5th, 95th]")

    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$\alpha(\eta)$")
    ax1.set_title("Treatment effect function: method comparison")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_zorder(ax1h.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "comparison_fig1_alpha.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved comparison_fig1_alpha.png")

    # ---- Figure 2: Utility curves at c=0.06 ----
    phi_grid = np.linspace(-3.0, 3.0, 600)
    c_default = 0.06

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for res, col in zip(results, colors):
        util = compute_utility(res["alpha_all"], eta_hat, gX_sorted,
                               phi_grid, c_default)
        oi = np.argmax(util)
        op = phi_grid[oi]
        ax2.plot(phi_grid, util, color=col, linewidth=2,
                 label=f'{res["label"]}: phi*={op:.2f}')

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel(r"Threshold $\phi$")
    ax2.set_ylabel("Utility")
    ax2.set_title(f"Utility comparison (c = {c_default})")
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "comparison_fig2_utility.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved comparison_fig2_utility.png")

    # ---- Figure 3: phi*(c) curves ----
    c_range = np.linspace(0.0, 0.20, 100)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for res, col in zip(results, colors):
        phi_stars = np.zeros(len(c_range))
        for i, c_val in enumerate(c_range):
            util = compute_utility(res["alpha_all"], eta_hat, gX_sorted,
                                   phi_grid, c_val)
            phi_stars[i] = phi_grid[np.argmax(util)]
        ax3.plot(c_range, phi_stars, color=col, linewidth=2,
                 label=res["label"])

    ax3.axhline(0, color="gray", linestyle=":", linewidth=1,
                label="Current cutoff")
    ax3.set_xlabel("Per-capita cost c")
    ax3.set_ylabel(r"Optimal threshold $\phi^*$")
    ax3.set_title(r"$\phi^*(c)$ comparison across methods")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "comparison_fig3_phi_star.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved comparison_fig3_phi_star.png")

    # ---- Summary table ----
    print(f"\n{'Method':<30s}  {'alpha_hat':>10s}  {'phi*(c=0)':>10s}  "
          f"{'phi*(c=.06)':>12s}  {'phi*(c=.10)':>12s}")
    print("-" * 80)
    for res in results:
        phis = {}
        for c_val in [0.0, 0.06, 0.10]:
            util = compute_utility(res["alpha_all"], eta_hat, gX_sorted,
                                   phi_grid, c_val)
            phis[c_val] = phi_grid[np.argmax(util)]
        print(f'{res["label"]:<30s}  {res["alpha_hat"]:10.4f}  '
              f'{phis[0.0]:10.3f}  {phis[0.06]:12.3f}  {phis[0.10]:12.3f}')

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
