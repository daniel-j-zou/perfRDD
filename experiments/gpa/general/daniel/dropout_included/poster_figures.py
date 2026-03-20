"""
Poster-ready figures for GPA probation analysis.

Method: Gaussian treatment basis + B-spline baseline, dropout imputation
        (Y = -GPA_year1), GPA-dependent cost c * GPA_year1 with c = 0.05.

Usage:
  python poster_figures.py            # compute if needed, then plot
  python poster_figures.py --recompute # force recompute, then plot
  python poster_figures.py --plot-only # plot from cached data (no estimation)

Figures:
  1. poster_fig1_alpha.{png,pdf}   — alpha(eta) with 95% bootstrap band
  2. poster_fig2_utility.{png,pdf} — utility curve at c=0.05
  3. poster_fig3_phi_star.{png,pdf} — phi*(c) + fraction treated
  4. poster_fig4_Q_dist.{png,pdf}  — Q distribution with thresholds
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

# ---- Paths ----
OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(OUTDIR), "..", "..", "Dep_Data", "final_processed_data.csv",
)
CACHE_PATH = os.path.join(OUTDIR, "poster_data.npz")

X_COLS = [
    "hsgrade_pct", "totcredits_year1", "loc_campus1", "loc_campus2",
    "male", "bpl_north_america", "age_at_entry", "english",
]
Q_COL = "dist_from_cut"
Y_COL = "nextGPA"

C_DEFAULT = 0.04
B_BOOT = 200
RNG_SEED = 42


# ---- Poster style ----
def set_poster_style():
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


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


# ---- Full estimation pipeline ----

def estimate_on_data(df, eta_grid=None):
    """
    Run the full Gaussian-treat PLM with dropout imputation.
    If eta_grid is provided, evaluate alpha on it.
    Returns dict with all estimation objects.
    """
    n = len(df)
    X = df[X_COLS].values
    q = df[Q_COL].values
    y = df[Y_COL].values
    gpa1 = df["GPA_year1"].values
    D = (q < 0).astype(float)
    n_tr = int(D.sum())

    if n_tr < 10:
        return None

    # First stage
    X_design = np.column_stack((np.ones(n), X))
    try:
        gamma_hat, *_ = np.linalg.lstsq(X_design, q, rcond=None)
    except np.linalg.LinAlgError:
        return None
    eta_hat = q - X_design @ gamma_hat
    eta_Tr = eta_hat[D == 1]

    # Basis construction
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn_base = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info_base = _bspline_params(kn_base, support)
    Phi_base = _eval_basis(eta_hat, info_base)
    n_base = Phi_base.shape[1]

    tr_lo = np.percentile(eta_Tr, 5)
    tr_hi = np.percentile(eta_Tr, 95)
    kn_treat = max(4, int(round(n_tr ** (1.0 / 3.0))))
    centers = np.linspace(tr_lo, tr_hi, kn_treat)
    info_treat = _gaussian_params(centers, bwf=1.5)
    Phi_treat = _eval_basis(eta_hat, info_treat)

    # PLM
    DPhi = D[:, None] * Phi_treat
    H = np.column_stack((X, Phi_base, DPhi))
    p = X.shape[1]
    total_cols = H.shape[1]

    lam_reg = 0.1 / np.sqrt(n)
    P_mat = np.zeros((total_cols, total_cols))
    np.fill_diagonal(P_mat[p:, p:], lam_reg)

    try:
        be = np.linalg.solve(H.T @ H + n * P_mat, H.T @ y)
    except np.linalg.LinAlgError:
        return None

    omega_treat = be[p + n_base:]

    # Alpha for all observations
    alpha_all = Phi_treat @ omega_treat

    # Trimmed average
    tr_mask = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
    Phi_Tr_eval = _eval_basis(eta_Tr[tr_mask], info_treat)
    alpha_hat = np.mean(Phi_Tr_eval @ omega_treat)

    result = {
        "alpha_all": alpha_all,
        "alpha_hat": alpha_hat,
        "eta_hat": eta_hat,
        "eta_Tr": eta_Tr,
        "gpa1": gpa1,
        "q": q,
        "gX_sorted": np.sort(q - eta_hat),
        "info_treat": info_treat,
        "omega_treat": omega_treat,
        "tr_lo": tr_lo,
        "tr_hi": tr_hi,
        "n": n,
        "n_tr": n_tr,
    }

    if eta_grid is not None:
        Phi_grid = _eval_basis(eta_grid, info_treat)
        result["alpha_grid"] = Phi_grid @ omega_treat

    return result


def compute_utility(alpha_all, gpa1, eta_hat, gX_sorted, phi_grid, c):
    """U(phi) = E[alpha * P(Q<phi|eta)] - c * E[GPA1 * P(Q<phi|eta)]"""
    n_total = len(gX_sorted)
    util = np.zeros(len(phi_grid))
    for j, phi in enumerate(phi_grid):
        thresh = phi - eta_hat
        probs = np.searchsorted(gX_sorted, thresh) / n_total
        benefit = np.mean(alpha_all * probs)
        cost = c * np.mean(gpa1 * probs)
        util[j] = benefit - cost
    return util


# ---- Data loading ----

def load_data():
    df = pd.read_csv(DATA_PATH)
    has_y = df[Y_COL].notna().values
    left = df["left_school"].values.astype(float)
    impute_mask = (~has_y) & (left == 1)
    # Dropout imputation: GPA_year2 = 0 → nextGPA = 0 - cutoff = -cutoff
    df.loc[impute_mask, Y_COL] = -df.loc[impute_mask, "gpacutoff"]
    df = df[df[Y_COL].notna()].copy()
    return df


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    print(f"Saved {name}.png and {name}.pdf")


# ---- Computation (expensive) ----

def compute_and_save():
    """Run estimation + bootstrap + utility sweeps, save to .npz cache."""
    df = load_data()
    n = len(df)
    print(f"Loaded {n} observations")

    # Point estimate
    est = estimate_on_data(df)
    eta_Tr = est["eta_Tr"]
    tr_lo, tr_hi = est["tr_lo"], est["tr_hi"]
    eta_grid = np.linspace(tr_lo, tr_hi, 500)

    est = estimate_on_data(df, eta_grid=eta_grid)
    alpha_grid = est["alpha_grid"]
    alpha_hat = est["alpha_hat"]
    print(f"alpha_hat = {alpha_hat:.4f}")

    # Bootstrap
    print(f"\nRunning {B_BOOT} bootstrap resamples...")
    rng = np.random.default_rng(RNG_SEED)
    alpha_boots = []
    for b in range(B_BOOT):
        if (b + 1) % 50 == 0:
            print(f"  {b+1}/{B_BOOT}")
        idx = rng.choice(n, size=n, replace=True)
        df_boot = df.iloc[idx].reset_index(drop=True)
        res_b = estimate_on_data(df_boot, eta_grid=eta_grid)
        if res_b is not None:
            alpha_boots.append(res_b["alpha_grid"])

    alpha_boots = np.array(alpha_boots)
    print(f"Successful: {len(alpha_boots)}/{B_BOOT}")

    alpha_lo = np.percentile(alpha_boots, 2.5, axis=0)
    alpha_hi = np.percentile(alpha_boots, 97.5, axis=0)

    # Utility at c=C_DEFAULT
    phi_grid = np.linspace(-3.0, 3.0, 600)
    util = compute_utility(est["alpha_all"], est["gpa1"], est["eta_hat"],
                           est["gX_sorted"], phi_grid, C_DEFAULT)
    opt_idx = np.argmax(util)
    opt_phi = phi_grid[opt_idx]
    frac_opt = np.mean(est["q"] < opt_phi)
    frac_current = np.mean(est["q"] < 0)

    print(f"\nc = {C_DEFAULT}")
    print(f"phi* = {opt_phi:.3f}")
    print(f"Fraction treated at phi*: {frac_opt:.1%}")
    print(f"Fraction treated currently: {frac_current:.1%}")

    # phi*(c) sweep
    phi_grid_fine = np.linspace(-3.0, 3.0, 3000)
    c_fine = np.linspace(0.0, 0.10, 80)
    phi_stars = np.zeros(len(c_fine))
    treat_fracs = np.zeros(len(c_fine))
    for i, c in enumerate(c_fine):
        u = compute_utility(est["alpha_all"], est["gpa1"], est["eta_hat"],
                            est["gX_sorted"], phi_grid_fine, c)
        idx = np.argmax(u)
        phi_stars[i] = phi_grid_fine[idx]
        treat_fracs[i] = np.mean(est["q"] < phi_grid_fine[idx])

    # Save everything needed for plotting
    np.savez(
        CACHE_PATH,
        # Fig 1: alpha
        eta_grid=eta_grid,
        alpha_grid=alpha_grid,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
        alpha_hat=alpha_hat,
        eta_Tr=eta_Tr,
        tr_lo=tr_lo,
        tr_hi=tr_hi,
        # Fig 2: utility
        phi_grid=phi_grid,
        util=util,
        opt_phi=opt_phi,
        opt_idx=opt_idx,
        frac_opt=frac_opt,
        frac_current=frac_current,
        # Fig 3: phi*(c)
        c_fine=c_fine,
        phi_stars=phi_stars,
        treat_fracs=treat_fracs,
        # Fig 4: Q distribution
        q=est["q"],
    )
    print(f"\nCached data to {CACHE_PATH}")


def load_cache():
    """Load precomputed data from .npz cache."""
    d = np.load(CACHE_PATH)
    print(f"Loaded cached data from {CACHE_PATH}")
    return d


# ---- Plotting (fast) ----

def plot_figures(d):
    """Produce all poster figures from cached data dict."""
    set_poster_style()

    eta_grid = d["eta_grid"]
    alpha_grid = d["alpha_grid"]
    alpha_lo = d["alpha_lo"]
    alpha_hi = d["alpha_hi"]
    alpha_hat = float(d["alpha_hat"])
    eta_Tr = d["eta_Tr"]
    tr_lo = float(d["tr_lo"])
    tr_hi = float(d["tr_hi"])
    phi_grid = d["phi_grid"]
    util = d["util"]
    opt_phi = float(d["opt_phi"])
    opt_idx = int(d["opt_idx"])
    frac_opt = float(d["frac_opt"])
    frac_current = float(d["frac_current"])
    c_fine = d["c_fine"]
    phi_stars = d["phi_stars"]
    treat_fracs = d["treat_fracs"]
    q = d["q"]

    # ================================================================
    # FIGURE 1: Alpha with bootstrap confidence band
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.fill_between(eta_grid, alpha_lo, alpha_hi,
                     color="steelblue", alpha=0.2, label="95% bootstrap CI")
    ax1.plot(eta_grid, alpha_grid, color="steelblue", linewidth=2.5,
             label=r"$\hat{\alpha}(\eta)$")
    ax1.axhline(0, color="black", linewidth=0.8)

    ax1h = ax1.twinx()
    in_trim = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
    ax1h.hist(eta_Tr[in_trim], bins=40, alpha=0.08, color="gray",
              density=True, zorder=0)
    ax1h.set_ylabel("Density (treated)", color="gray", fontsize=18)
    ax1h.tick_params(axis="y", labelcolor="gray", labelsize=14)
    ax1h.spines["top"].set_visible(False)

    ax1.set_zorder(ax1h.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax1.set_xlabel(r"$\eta$ (natural ability)")
    ax1.set_ylabel(r"$\alpha(\eta)$")
    ax1.set_title("Treatment Effect of Academic Probation")
    ax1.legend(loc="lower right", framealpha=0.9)

    fig1.tight_layout()
    save_fig(fig1, "poster_fig1_alpha")

    # ================================================================
    # FIGURE 2: Utility curve at c = C_DEFAULT
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.plot(phi_grid, util, color="steelblue", linewidth=2.5)
    ax2.axhline(0, color="black", linewidth=0.8)

    ax2.axvline(0, color="gray", linestyle=":", linewidth=1.5,
                label=f"Current cutoff ($\\phi=0$)")

    ax2.axvline(opt_phi, color="crimson", linestyle="--", linewidth=2,
                label=f"Optimal $\\phi^*={opt_phi:.2f}$")
    ax2.plot(opt_phi, util[opt_idx], "o", color="crimson", markersize=8,
             zorder=5)

    ax2.annotate(f"$\\phi^*={opt_phi:.2f}$\n{frac_opt:.0%} treated",
                 xy=(opt_phi, util[opt_idx]),
                 xytext=(opt_phi + 0.8, util[opt_idx] * 0.5),
                 fontsize=16, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))

    ax2.set_xlabel(r"Probation threshold $\phi$")
    ax2.set_ylabel("Utility")
    ax2.set_title("Utility Function")
    ax2.legend(loc="lower left", framealpha=0.9)

    fig2.tight_layout()
    save_fig(fig2, "poster_fig2_utility")

    # ================================================================
    # FIGURE 3: phi*(c) and fraction treated
    # ================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    ax3a.plot(c_fine, phi_stars, color="steelblue", linewidth=2.5)
    ax3a.axhline(0, color="gray", linestyle=":", linewidth=1,
                 label="Current cutoff")

    ax3a.set_xlabel("Cost parameter $c$")
    ax3a.set_ylabel(r"Optimal threshold $\phi^*$")
    ax3a.set_title(r"Optimal Threshold $\phi^*(c)$")
    ax3a.legend(framealpha=0.9)

    ax3b.plot(c_fine, treat_fracs * 100, color="steelblue", linewidth=2.5)
    ax3b.axhline(frac_current * 100, color="gray", linestyle=":",
                 linewidth=1, label=f"Current ({frac_current:.0%})")

    ax3b.set_xlabel("Cost parameter $c$")
    ax3b.set_ylabel("Students on probation (%)")
    ax3b.set_title("Fraction Treated at Optimal Threshold")
    ax3b.legend(framealpha=0.9)

    fig3.tight_layout()
    save_fig(fig3, "poster_fig3_phi_star")

    # ================================================================
    # FIGURE 4: Q distribution with policy thresholds
    # ================================================================
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    bins = np.linspace(q.min(), q.max(), 120)
    counts, edges = np.histogram(q, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    for j in range(len(centers)):
        if centers[j] < 0:
            color = "steelblue"
            alpha_val = 0.6
        elif centers[j] < opt_phi:
            color = "orange"
            alpha_val = 0.5
        else:
            color = "lightgray"
            alpha_val = 0.5
        ax4.bar(centers[j], counts[j], width=width, color=color,
                alpha=alpha_val, edgecolor="none")

    ax4.axvline(0, color="black", linewidth=2, linestyle="-",
                label="Current cutoff")
    ax4.axvline(opt_phi, color="crimson", linewidth=2, linestyle="--",
                label=f"Optimal cutoff ($\\phi^*={opt_phi:.2f}$)")

    ax4.set_xlabel("$Q$ (distance from probation cutoff)")
    ax4.set_ylabel("Count")
    ax4.set_title("Policy Comparison: Current vs. Optimal")
    ax4.legend(loc="upper right", framealpha=0.9)

    ax4.set_xlim(-3, 4)

    fig4.tight_layout()
    save_fig(fig4, "poster_fig4_Q_dist")

    plt.close("all")
    print("\nAll poster figures saved.")


# ---- Main ----

def main():
    recompute = "--recompute" in sys.argv
    plot_only = "--plot-only" in sys.argv

    if plot_only:
        d = load_cache()
    elif recompute or not os.path.exists(CACHE_PATH):
        compute_and_save()
        d = load_cache()
    else:
        print(f"Using cached data ({CACHE_PATH})")
        d = load_cache()

    plot_figures(d)


if __name__ == "__main__":
    main()
