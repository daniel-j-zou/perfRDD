"""
Sweep over dropout penalty levels to find one that produces an interior
optimal threshold WITHOUT needing per-capita cost c.

Imputation: nextGPA = 0 - GPA_year1 - p  for dropouts (left_school=1, nextGPA missing).
When p=0 this is the current approach. Larger p penalizes dropout more.

Uses Gaussian basis for treatment effect (best method from comparison).
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


# ---- Core estimation (Gaussian treatment basis) ----

def estimate(df):
    """Run Gaussian-treat PLM. Returns estimation objects for utility."""
    n = len(df)
    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    # First stage
    X_design = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    eta_hat = q - X_design @ gamma_hat
    eta_Tr = eta_hat[D == 1]

    # Baseline: B-spline on full support
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

    DPhi = D[:, None] * Phi_treat
    H = np.column_stack((X, Phi_base, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam = 0.1 / np.sqrt(n)
    P = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P[p:, p:], lam)
    be = np.linalg.solve(H.T @ H + n * P, H.T @ y)
    omega_treat = be[p + n_base:]

    # Alpha for all observations
    Phi_all = _eval_basis(eta_hat, info_treat)
    alpha_all = Phi_all @ omega_treat

    # Plug-in average (trimmed treated)
    tr_mask = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr[tr_mask], info_treat)
    alpha_hat = np.mean(Phi_Tr_eval @ omega_treat)

    # For utility
    gX_sorted = np.sort(q - eta_hat)

    # For alpha grid plot
    eta_grid = np.linspace(tr_lo, tr_hi, 500)
    Phi_grid = _eval_basis(eta_grid, info_treat)
    alpha_grid = Phi_grid @ omega_treat

    return {
        "alpha_hat": alpha_hat,
        "alpha_all": alpha_all,
        "alpha_grid": alpha_grid,
        "eta_grid": eta_grid,
        "eta_hat": eta_hat,
        "eta_Tr": eta_Tr,
        "gX_sorted": gX_sorted,
        "n": n, "n_tr": n_tr,
        "tr_lo": tr_lo, "tr_hi": tr_hi,
    }


def compute_utility(est, phi_grid):
    """U(phi) = E[alpha(eta) * P(Q < phi | eta)], no cost term."""
    alpha_all = est["alpha_all"]
    eta_hat = est["eta_hat"]
    gX_sorted = est["gX_sorted"]
    n_total = len(gX_sorted)

    util = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_hat
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        util[j] = np.mean(alpha_all * probs)
    return util


# ---- Main ----

def main():
    df_raw = pd.read_csv(DATA_PATH)
    missing_y = df_raw[Y_COL].isna()
    left = df_raw["left_school"] == 1
    impute_mask = missing_y & left

    n_dropouts = impute_mask.sum()
    gpa1_dropouts = df_raw.loc[impute_mask, "GPA_year1"]
    print(f"Dropouts to impute: {n_dropouts}")
    print(f"Their GPA_year1: mean={gpa1_dropouts.mean():.3f}, "
          f"median={gpa1_dropouts.median():.3f}")

    phi_grid = np.linspace(-3.0, 3.0, 600)

    # Sweep over penalty levels
    p_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    print(f"\n{'p':>5s}  {'alpha_hat':>10s}  {'phi*(c=0)':>10s}  "
          f"{'U(phi*)':>10s}  {'U(0)':>10s}  {'avg imputed Y':>14s}")
    print("-" * 70)

    all_results = []
    for p in p_vals:
        df = df_raw.copy()
        df.loc[impute_mask, Y_COL] = 0.0 - df.loc[impute_mask, "GPA_year1"] - p
        df = df[df[Y_COL].notna()].copy()

        avg_imputed_y = (0.0 - gpa1_dropouts - p).mean()

        est = estimate(df)
        util = compute_utility(est, phi_grid)
        oi = np.argmax(util)
        op = phi_grid[oi]
        u0 = np.interp(0.0, phi_grid, util)

        print(f"{p:5.1f}  {est['alpha_hat']:10.4f}  {op:10.3f}  "
              f"{util[oi]:10.4f}  {u0:10.4f}  {avg_imputed_y:14.3f}")

        all_results.append({
            "p": p, "est": est, "util": util, "phi_star": op,
        })

    # ---- Figure 1: Utility curves for different p ----
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.coolwarm
    for i, res in enumerate(all_results):
        color = cmap(i / (len(all_results) - 1))
        ax1.plot(phi_grid, res["util"], color=color, linewidth=1.5,
                 label=f'p={res["p"]:.1f}, phi*={res["phi_star"]:.2f}')
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax1.set_xlabel(r"Threshold $\phi$")
    ax1.set_ylabel("Utility (c = 0)")
    ax1.set_title("Utility at c=0 for different dropout penalties p")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "penalty_sweep_fig1_utility.png"),
                 dpi=150, bbox_inches="tight")
    print("\nSaved penalty_sweep_fig1_utility.png")

    # ---- Figure 2: Alpha functions for select p values ----
    select_p = [0.0, 1.0, 2.0, 3.0, 5.0]
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    colors = ["steelblue", "green", "orange", "red", "purple"]
    for res, col in zip(all_results, colors):
        if res["p"] in select_p:
            est = res["est"]
            ax2.plot(est["eta_grid"], est["alpha_grid"], color=col, linewidth=2,
                     label=f'p={res["p"]:.0f} (avg={est["alpha_hat"]:.3f})')

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel(r"$\alpha(\eta)$")
    ax2.set_title("Treatment effect function for different dropout penalties")
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "penalty_sweep_fig2_alpha.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved penalty_sweep_fig2_alpha.png")

    # ---- Figure 3: phi*(c=0) as a function of p ----
    p_fine = np.linspace(0.0, 6.0, 50)
    phi_stars_fine = np.zeros(len(p_fine))
    for i, p in enumerate(p_fine):
        df = df_raw.copy()
        df.loc[impute_mask, Y_COL] = 0.0 - df.loc[impute_mask, "GPA_year1"] - p
        df = df[df[Y_COL].notna()].copy()
        est = estimate(df)
        util = compute_utility(est, phi_grid)
        phi_stars_fine[i] = phi_grid[np.argmax(util)]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(p_fine, phi_stars_fine, "b.-", markersize=4)
    ax3.axhline(0, color="gray", linestyle=":", linewidth=1,
                label="Current cutoff")
    ax3.set_xlabel("Dropout penalty p")
    ax3.set_ylabel(r"Optimal threshold $\phi^*$ (c = 0)")
    ax3.set_title(r"$\phi^*(p)$ at c = 0: how much dropout penalty is needed?")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, "penalty_sweep_fig3_phi_vs_p.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved penalty_sweep_fig3_phi_vs_p.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
