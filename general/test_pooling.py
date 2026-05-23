"""
test_pooling.py

Tests whether pooling beta (constraining X coefficients to be common
across treatment/control) matters for estimating alpha(eta).

Pooled:   H = [X,     Phi_base(eta_hat), D * Phi_treat(eta_hat)]
Unpooled: H = [X, DX, Phi_base(eta_hat), D * Phi_treat(eta_hat)]

where DX = D[:,None] * X.

Experiments
-----------
  Sim A - pGen DGP with common beta (pooling is the correct model)
  Sim B - same DGP + beta_diff * D * X in Y (pooling is misspecified)
  GPA   - real data comparison

Output files (written to general/)
-----------------------------------
  pooling_simA.png / pooling_simB.png  -- bias / RMSE curves over N
  pooling_gpa.png                      -- alpha(eta) for pooled vs unpooled
  pooling_tables.txt                   -- summary statistics
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths ----------------------------------------------------------------
OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    OUTDIR, "..", "experiments", "gpa", "Dep_Data", "final_processed_data.csv",
)
GPA_X_COLS = [
    "hsgrade_pct", "totcredits_year1", "loc_campus1", "loc_campus2",
    "male", "bpl_north_america", "age_at_entry", "english",
]
GPA_Q_COL = "dist_from_cut"
GPA_Y_COL = "nextGPA"


# ===========================================================================
# Basis functions  (same spec as poster_figures.py)
# ===========================================================================

def _bspline_params(kn, support):
    degree = 3
    lo, hi = support
    interior = np.linspace(lo, hi, kn + 2)[1:-1]
    t = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
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


def eval_basis(pts, info):
    if info["type"] == "bspline":
        return _eval_bspline(pts, info)
    return _eval_gaussian(pts, info)


# ===========================================================================
# Core estimator: pooled vs unpooled
# ===========================================================================

def estimate_alpha(X, q, y, D, pooled=True, eta_grid=None,
                   use_ridge=True, treat_basis="gaussian"):
    """
    Semiparametric PLM estimator of the treatment effect function alpha(eta).

    Model (pooled):
        Y = X beta  +  Phi_base(eta_hat) omega_base  +  D * Phi_treat(eta_hat) omega_treat  +  eps

    Model (unpooled):
        Y = X beta_con  +  D*X beta_diff  +  Phi_base(eta_hat) omega_base
              +  D * Phi_treat(eta_hat) omega_treat  +  eps

    Parameters
    ----------
    X     : (n,) or (n, p) array of covariates
    q     : (n,) running variable
    y     : (n,) outcome
    D     : (n,) treatment indicator (float 0/1)
    pooled: if True, use pooled beta; if False, add D*X to design matrix
    eta_grid : optional 1-D array; if given, return alpha_grid evaluated there
    use_ridge : if True, apply ridge (lambda=0.1/sqrt(n)) to basis columns only
    treat_basis : "gaussian" (default) or "bspline" for the treatment basis

    Returns dict or None (if too few treated observations).
    """
    n = len(y)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    p = X.shape[1]

    # --- First stage: Q = [1, X] gamma + eta ---
    Xd = np.column_stack((np.ones(n), X))
    gamma_hat, *_ = np.linalg.lstsq(Xd, q, rcond=None)
    eta_hat = q - Xd @ gamma_hat

    eta_Tr = eta_hat[D == 1]
    n_tr = int(D.sum())
    if n_tr < 10:
        return None

    # --- Baseline basis (B-spline over full eta support) ---
    support = (np.percentile(eta_hat, 0.5), np.percentile(eta_hat, 99.5))
    kn_base = max(4, int(round(n_tr ** (1.0 / 3.0))))
    info_base = _bspline_params(kn_base, support)
    Phi_base = eval_basis(eta_hat, info_base)

    # --- Treatment basis ---
    tr_lo = np.percentile(eta_Tr, 5)
    tr_hi = np.percentile(eta_Tr, 95)
    kn_treat = max(4, int(round(n_tr ** (1.0 / 3.0))))
    if treat_basis == "bspline":
        info_treat = _bspline_params(kn_treat, (tr_lo, tr_hi))
    else:
        centers = np.linspace(tr_lo, tr_hi, kn_treat)
        info_treat = _gaussian_params(centers, bwf=1.5)
    Phi_treat = eval_basis(eta_hat, info_treat)
    n_treat = Phi_treat.shape[1]

    DPhi = D[:, None] * Phi_treat

    # --- Design matrix ---
    if pooled:
        H = np.column_stack((X, Phi_base, DPhi))
        p_param = p                # columns not penalised
    else:
        DX = D[:, None] * X
        H = np.column_stack((X, DX, Phi_base, DPhi))
        p_param = 2 * p            # X and DX both unpenalised

    total_cols = H.shape[1]
    if use_ridge:
        lam_reg = 0.1 / np.sqrt(n)
        P_mat = np.zeros((total_cols, total_cols))
        np.fill_diagonal(P_mat[p_param:, p_param:], lam_reg)
        A = H.T @ H + n * P_mat
    else:
        A = H.T @ H

    try:
        be = np.linalg.solve(A, H.T @ y)
    except np.linalg.LinAlgError:
        return None

    # omega_treat: last n_treat coefficients
    omega_treat = be[-n_treat:]

    # Trimmed average treatment effect
    tr_mask = (eta_Tr >= tr_lo) & (eta_Tr <= tr_hi)
    Phi_Tr_eval = eval_basis(eta_Tr[tr_mask], info_treat)
    alpha_hat = float(np.mean(Phi_Tr_eval @ omega_treat))

    result = {
        "alpha_hat": alpha_hat,
        "eta_hat": eta_hat,
        "eta_Tr": eta_Tr,
        "info_treat": info_treat,
        "omega_treat": omega_treat,
        "tr_lo": tr_lo,
        "tr_hi": tr_hi,
    }
    if eta_grid is not None:
        Phi_grid = eval_basis(eta_grid, info_treat)
        result["alpha_grid"] = Phi_grid @ omega_treat

    return result


# ===========================================================================
# DGP for simulation  (pGen from splines.py, extended with beta_diff)
# ===========================================================================

def pGen(n, beta, gamma, beta_diff=0.0, rng=None):
    """
    DGP:
        x   ~ N(0, 1)
        a   ~ +/- Exp(1)
        eta | a ~ N(a, 1)
        nu_mean = 3 exp(a)/(1+exp(a))
        nu  ~ N(nu_mean, 1)
        w_mean = 2 exp(eta)/(1+exp(eta))
        w   ~ N(w_mean, 0.1)
        q   = gamma * x + eta
        D   = 1{q > 0}
        y   = D*w + beta*x + beta_diff*D*x + nu

    True alpha(eta) = E[W|eta] = 2 exp(eta)/(1+exp(eta))
    (independent of beta_diff — that only affects the X effect for treated).
    """
    if rng is None:
        rng = np.random.default_rng()
    x = rng.normal(0.0, 1.0, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    a = signs * rng.exponential(scale=1.0, size=n)
    eta = rng.normal(loc=a, scale=1.0, size=n)
    nu_mean = 3 * np.exp(a) / (1 + np.exp(a))
    nu = rng.normal(loc=nu_mean, scale=1.0, size=n)
    w_mean = 2 * np.exp(eta) / (1 + np.exp(eta))
    w = rng.normal(loc=w_mean, scale=0.1, size=n)
    q = gamma * x + eta
    D = (q > 0).astype(float)
    y = D * w + beta * x + beta_diff * D * x + nu
    return x, y, eta, q, D


def true_alpha(eta):
    """True pointwise treatment effect in pGen DGP."""
    return 2 * np.exp(eta) / (1 + np.exp(eta))


# ===========================================================================
# Monte Carlo runner
# ===========================================================================

# Fixed evaluation grid for simulation (covers bulk of treated eta support).
SIM_ETA_GRID = np.linspace(-2.0, 2.0, 200)
TRUE_ALPHA_GRID = true_alpha(SIM_ETA_GRID)


def run_sim(beta_diff, N_values, R=200, gamma=1.0, beta=1.0, seed=42):
    """
    Returns a long-form DataFrame with one row per (N, rep, method):
      mae_grid  : mean |alpha_hat(eta) - true_alpha(eta)| over SIM_ETA_GRID
      alpha_hat : trimmed average treatment effect
      alpha_grid: (stored separately in a nested dict for bias plots)

    Also returns a dict  grid_data[N][method] = array (R, len(SIM_ETA_GRID))
    of per-rep alpha curves for computing bias and std pointwise.
    """
    rng_master = np.random.default_rng(seed)
    rows = []
    # grid_data[N][method] accumulates (R, G) arrays
    grid_data = {N: {"pooled": [], "unpooled": []} for N in N_values}

    for N in N_values:
        print(f"  N = {N:>6d} ...", end=" ", flush=True)
        for r in range(R):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            x, y, eta, q, D = pGen(N, beta, gamma, beta_diff=beta_diff, rng=rng)
            for pooled, name in [(True, "pooled"), (False, "unpooled")]:
                res = estimate_alpha(x, q, y, D, pooled=pooled, eta_grid=SIM_ETA_GRID)
                if res is None:
                    continue
                ag = res["alpha_grid"]
                mae = float(np.mean(np.abs(ag - TRUE_ALPHA_GRID)))
                rows.append({
                    "N": N, "rep": r, "method": name,
                    "alpha_hat": res["alpha_hat"],
                    "mae_grid": mae,
                })
                grid_data[N][name].append(ag)
        print("done")

    df = pd.DataFrame(rows)
    # convert lists to arrays
    for N in N_values:
        for name in ("pooled", "unpooled"):
            grid_data[N][name] = np.array(grid_data[N][name])

    return df, grid_data


# ===========================================================================
# Plotting helpers
# ===========================================================================

def plot_sim_results(df, grid_data, N_values, beta_diff, tag, title_suffix):
    """
    Two panels:
      Left  : Mean absolute error (MAE) vs N for pooled and unpooled.
      Right : Bias curves alpha_hat(eta) - alpha(eta) at two sample sizes.
    """
    colors = {"pooled": "steelblue", "unpooled": "crimson"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Pooled vs Unpooled  —  {title_suffix}", fontsize=13)

    # ---- Left: MAE vs N ----
    ax = axes[0]
    for method in ("pooled", "unpooled"):
        sub = df[df["method"] == method].groupby("N")["mae_grid"]
        means = sub.mean()
        stds = sub.std()
        ax.plot(means.index, means.values, marker="o",
                color=colors[method], label=method, linewidth=2)
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        color=colors[method], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("N (log scale)")
    ax.set_ylabel("Mean |α̂(η) − α(η)|  (MAE over grid)")
    ax.set_title("MAE of α̂(η)")
    ax.legend()

    # ---- Right: pointwise bias at smallest and largest N ----
    ax2 = axes[1]
    N_lo, N_hi = min(N_values), max(N_values)
    linestyles = {N_lo: "--", N_hi: "-"}
    for N, ls in [(N_lo, "--"), (N_hi, "-")]:
        for method in ("pooled", "unpooled"):
            arr = grid_data[N][method]  # (R, G)
            if len(arr) == 0:
                continue
            bias = arr.mean(axis=0) - TRUE_ALPHA_GRID
            ax2.plot(SIM_ETA_GRID, bias,
                     color=colors[method], linestyle=ls, linewidth=1.5,
                     label=f"{method} N={N:,}")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel(r"Bias of $\hat\alpha(\eta)$")
    ax2.set_title("Pointwise Bias (-- small N, — large N)")
    ax2.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, f"pooling_{tag}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


# ===========================================================================
# GPA data comparison
# ===========================================================================

def run_gpa_comparison():
    df = pd.read_csv(DATA_PATH)
    has_y = df[GPA_Y_COL].notna().values
    left = df["left_school"].values.astype(float)
    impute_mask = (~has_y) & (left == 1)
    df.loc[impute_mask, GPA_Y_COL] = -df.loc[impute_mask, "gpacutoff"]
    df = df[df[GPA_Y_COL].notna()].copy()
    print(f"GPA: {len(df)} observations after imputation")

    X = df[GPA_X_COLS].values
    q = df[GPA_Q_COL].values
    y = df[GPA_Y_COL].values
    D = (q < 0).astype(float)          # probation = below cutoff

    # First run to get treated eta range, then use as evaluation grid
    res0 = estimate_alpha(X, q, y, D, pooled=True)
    eta_grid = np.linspace(res0["tr_lo"], res0["tr_hi"], 400)

    results = {}
    for pooled, name in [(True, "pooled"), (False, "unpooled")]:
        res = estimate_alpha(X, q, y, D, pooled=pooled, eta_grid=eta_grid)
        results[name] = res
        print(f"  {name:10s}: alpha_hat = {res['alpha_hat']:.4f}")

    diff = results["pooled"]["alpha_grid"] - results["unpooled"]["alpha_grid"]
    print(f"  max |pooled - unpooled| on grid: {np.max(np.abs(diff)):.4f}")
    print(f"  mean|pooled - unpooled| on grid: {np.mean(np.abs(diff)):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GPA Probation: Pooled vs Unpooled β", fontsize=13)
    colors = {"pooled": "steelblue", "unpooled": "crimson"}

    ax = axes[0]
    for name, res in results.items():
        ax.plot(eta_grid, res["alpha_grid"],
                color=colors[name], linewidth=2, label=name)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\eta$ (natural ability)")
    ax.set_ylabel(r"$\hat\alpha(\eta)$")
    ax.set_title(r"$\hat\alpha(\eta)$: pooled vs unpooled")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(eta_grid, diff, color="darkgreen", linewidth=2)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel("pooled − unpooled")
    ax2.set_title("Difference in α̂(η)")

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "pooling_gpa.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

    return results, eta_grid


# ===========================================================================
# Summary tables
# ===========================================================================

def print_and_save_tables(dfA, dfB, results_gpa, outfile):
    lines = []

    for label, df in [("Sim A  (beta_diff=0, pooling correct)", dfA),
                      ("Sim B  (beta_diff=0.5, pooling wrong)", dfB)]:
        lines.append(f"\n{'='*60}")
        lines.append(label)
        lines.append(f"{'='*60}")
        tbl = (df.groupby(["N", "method"])["mae_grid"]
                 .agg(mean_MAE="mean", std_MAE="std")
                 .reset_index())
        pivot = tbl.pivot(index="N", columns="method", values=["mean_MAE", "std_MAE"])
        lines.append(pivot.to_string())

        lines.append("")
        tbl2 = (df.groupby(["N", "method"])["alpha_hat"]
                  .agg(mean_ahat="mean", std_ahat="std")
                  .reset_index())
        pivot2 = tbl2.pivot(index="N", columns="method", values=["mean_ahat", "std_ahat"])
        lines.append(pivot2.to_string())

    lines.append(f"\n{'='*60}")
    lines.append("GPA Data")
    lines.append(f"{'='*60}")
    for name, res in results_gpa.items():
        lines.append(f"  {name:10s}  alpha_hat = {res['alpha_hat']:.5f}")

    text = "\n".join(lines)
    print(text)
    with open(outfile, "w") as f:
        f.write(text + "\n")
    print(f"\nTables saved to {outfile}")


# ===========================================================================
# Main
# ===========================================================================

N_VALUES = [2000, 5000, 10000, 50000]
R = 200


def main():
    print("=" * 60)
    print("Simulation A: pooling correct (beta_diff = 0)")
    print("=" * 60)
    dfA, grid_dataA = run_sim(beta_diff=0.0, N_values=N_VALUES, R=R, seed=42)
    plot_sim_results(dfA, grid_dataA, N_VALUES, beta_diff=0.0,
                     tag="simA", title_suffix="beta_diff=0 (pooling correct)")

    print()
    print("=" * 60)
    print("Simulation B: pooling misspecified (beta_diff = 0.5)")
    print("=" * 60)
    dfB, grid_dataB = run_sim(beta_diff=0.5, N_values=N_VALUES, R=R, seed=42)
    plot_sim_results(dfB, grid_dataB, N_VALUES, beta_diff=0.5,
                     tag="simB", title_suffix="beta_diff=0.5 (pooling wrong)")

    print()
    print("=" * 60)
    print("GPA Data Comparison")
    print("=" * 60)
    results_gpa, _ = run_gpa_comparison()

    print_and_save_tables(dfA, dfB, results_gpa,
                          outfile=os.path.join(OUTDIR, "pooling_tables.txt"))


if __name__ == "__main__":
    main()
