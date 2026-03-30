"""
test_pooling_gamma.py

Why pooling beats unpooling even when beta_diff = 0.

The mechanism: D = 1{Q < 0} = 1{gamma*X + eta < 0} induces negative
correlation between X and eta within the treated group.  This makes D*X
correlated with D*Phi_treat(eta), so adding D*X to the design matrix creates
collinearity that inflates the variance (and finite-sample bias) of omega_treat.
Larger gamma => stronger selection => stronger within-treated X-eta correlation
=> bigger problem for the unpooled estimator.

Experiments
-----------
  1. Gamma sweep (N=5000, N=50000, R=200): MAE of alpha-hat vs gamma
  2. Collinearity diagnostic: Corr(X,eta|D=1) and Max|Corr(D*X, D*Phi)| vs gamma
  3. Pointwise bias curves at gamma=1 (weak) and gamma=2 (strong)

Outputs written to general/
  pooling_gamma_mae.png     -- MAE vs gamma at two sample sizes
  pooling_gamma_corr.png    -- collinearity diagnostics vs gamma
  pooling_gamma_bias.png    -- pointwise bias at gamma=1 vs gamma=2
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pooling import (
    pGen, true_alpha, estimate_alpha,
    eval_basis, _bspline_params, _gaussian_params,
    SIM_ETA_GRID, TRUE_ALPHA_GRID,
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
COLORS = {"pooled": "steelblue", "unpooled": "crimson"}


# ---------------------------------------------------------------------------
# Collinearity diagnostics for one dataset
# ---------------------------------------------------------------------------

def collinearity_diagnostics(x, q, D, eta_hat):
    """Return within-treated Corr(X,eta) and max|Corr(D*X, D*Phi)|."""
    eta_Tr = eta_hat[D == 1]
    n_tr = int(D.sum())

    corr_x_eta = float(np.corrcoef(x[D == 1], eta_Tr)[0, 1])

    kn = max(4, int(round(n_tr ** (1.0 / 3.0))))
    tr_lo = np.percentile(eta_Tr, 5)
    tr_hi = np.percentile(eta_Tr, 95)
    centers = np.linspace(tr_lo, tr_hi, kn)
    bw = np.mean(np.diff(centers)) * 1.5 if kn > 1 else 1.0
    info_treat = {"type": "gaussian", "mu": centers, "bw": bw}
    Phi_treat = eval_basis(eta_hat, info_treat)
    DPhi = D[:, None] * Phi_treat
    DX = D * x

    max_corr = float(np.max(np.abs(
        [np.corrcoef(DX, DPhi[:, k])[0, 1] for k in range(DPhi.shape[1])]
    )))
    return corr_x_eta, max_corr


# ---------------------------------------------------------------------------
# Gamma sweep
# ---------------------------------------------------------------------------

def run_gamma_sweep(gamma_values, N, R=200, beta=1.0, seed=42):
    """
    For each gamma, run R replications at sample size N.
    Returns a DataFrame with columns: gamma, method, mae_grid, alpha_hat,
    and a separate DataFrame with collinearity diagnostics.
    """
    rng_master = np.random.default_rng(seed)
    rows = []
    diag_rows = []

    for gamma in gamma_values:
        print(f"  gamma={gamma:.1f} N={N} ...", end=" ", flush=True)
        corrs_xe, corrs_dp = [], []

        for r in range(R):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            x, y, eta, q, D = pGen(N, beta=beta, gamma=gamma,
                                   beta_diff=0.0, rng=rng)

            # First-stage eta_hat
            Xd = np.column_stack((np.ones(N), x))
            ghat, *_ = np.linalg.lstsq(Xd, q, rcond=None)
            eta_hat = q - Xd @ ghat

            c_xe, c_dp = collinearity_diagnostics(x, q, D, eta_hat)
            corrs_xe.append(c_xe)
            corrs_dp.append(c_dp)

            for pooled, name in [(True, "pooled"), (False, "unpooled")]:
                res = estimate_alpha(x, q, y, D, pooled=pooled,
                                     eta_grid=SIM_ETA_GRID)
                if res is None:
                    continue
                ag = res["alpha_grid"]
                mae = float(np.mean(np.abs(ag - TRUE_ALPHA_GRID)))
                rows.append({
                    "gamma": gamma, "N": N, "rep": r,
                    "method": name,
                    "alpha_hat": res["alpha_hat"],
                    "mae_grid": mae,
                })

        diag_rows.append({
            "gamma": gamma, "N": N,
            "mean_corr_x_eta": float(np.mean(corrs_xe)),
            "mean_max_corr_dp": float(np.mean(corrs_dp)),
        })
        print("done")

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


# ---------------------------------------------------------------------------
# Pointwise bias at a fixed gamma and N (large, so bias dominates variance)
# ---------------------------------------------------------------------------

def run_bias_curves(gamma_values, N=50000, R=200, beta=1.0, seed=99):
    """
    Returns dict: gamma -> {"pooled": (R, G) array, "unpooled": (R, G) array}
    """
    rng_master = np.random.default_rng(seed)
    results = {}

    for gamma in gamma_values:
        print(f"  bias curves gamma={gamma:.1f} N={N} ...", end=" ", flush=True)
        arrs = {"pooled": [], "unpooled": []}
        for r in range(R):
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            x, y, eta, q, D = pGen(N, beta=beta, gamma=gamma,
                                   beta_diff=0.0, rng=rng)
            for pooled, name in [(True, "pooled"), (False, "unpooled")]:
                res = estimate_alpha(x, q, y, D, pooled=pooled,
                                     eta_grid=SIM_ETA_GRID)
                if res is not None:
                    arrs[name].append(res["alpha_grid"])
        results[gamma] = {k: np.array(v) for k, v in arrs.items()}
        print("done")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mae_vs_gamma(df_small, df_large, gamma_values):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"MAE of $\hat\alpha(\eta)$ vs $\gamma$  (beta_diff = 0)",
                 fontsize=13)

    for ax, (df, N) in zip(axes, [(df_small, df_small["N"].iloc[0]),
                                   (df_large, df_large["N"].iloc[0])]):
        for method in ("pooled", "unpooled"):
            sub = df[df["method"] == method].groupby("gamma")["mae_grid"]
            means = sub.mean()
            stds  = sub.std()
            ax.plot(means.index, means.values, marker="o",
                    color=COLORS[method], linewidth=2, label=method)
            ax.fill_between(means.index,
                            means - stds, means + stds,
                            color=COLORS[method], alpha=0.15)
        ax.set_xlabel(r"$\gamma$  (strength of X in first stage)")
        ax.set_ylabel(r"Mean $|\hat\alpha(\eta) - \alpha(\eta)|$")
        ax.set_title(f"N = {N:,}")
        ax.legend()

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "pooling_gamma_mae.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_collinearity(diag_small, diag_large, gamma_values):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Selection-induced collinearity vs γ", fontsize=13)

    ax = axes[0]
    for df, N, ls in [(diag_small, diag_small["N"].iloc[0], "--"),
                       (diag_large, diag_large["N"].iloc[0], "-")]:
        ax.plot(df["gamma"], df["mean_corr_x_eta"],
                color="navy", linestyle=ls, marker="o", linewidth=2,
                label=f"N={N:,}")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Mean Corr$(X, \hat\eta \mid D=1)$")
    ax.set_title(r"Within-treated $X$–$\hat\eta$ correlation")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()

    ax2 = axes[1]
    for df, N, ls in [(diag_small, diag_small["N"].iloc[0], "--"),
                       (diag_large, diag_large["N"].iloc[0], "-")]:
        ax2.plot(df["gamma"], df["mean_max_corr_dp"],
                 color="darkred", linestyle=ls, marker="o", linewidth=2,
                 label=f"N={N:,}")
    ax2.set_xlabel(r"$\gamma$")
    ax2.set_ylabel(r"Mean max$_k|\text{Corr}(D \cdot X,\; D \cdot \Phi_k(\hat\eta))|$")
    ax2.set_title(r"$D \cdot X$ vs $D \cdot \Phi_{\rm treat}$ collinearity")
    ax2.legend()

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "pooling_gamma_corr.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_bias_curves(bias_results, gamma_values):
    """One row per gamma, two panels: pointwise bias and std."""
    n_g = len(gamma_values)
    fig, axes = plt.subplots(n_g, 2, figsize=(13, 4 * n_g), squeeze=False)
    fig.suptitle(r"Pointwise bias/std of $\hat\alpha(\eta)$  (N=50,000, R=200)",
                 fontsize=13)

    for row, gamma in enumerate(gamma_values):
        arrs = bias_results[gamma]
        ax_b, ax_s = axes[row, 0], axes[row, 1]

        for method in ("pooled", "unpooled"):
            arr = arrs[method]
            if len(arr) == 0:
                continue
            bias = arr.mean(axis=0) - TRUE_ALPHA_GRID
            std  = arr.std(axis=0)
            ax_b.plot(SIM_ETA_GRID, bias, color=COLORS[method],
                      linewidth=2, label=method)
            ax_s.plot(SIM_ETA_GRID, std,  color=COLORS[method],
                      linewidth=2, label=method)

        for ax in (ax_b, ax_s):
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel(r"$\eta$")
            ax.legend(fontsize=9)

        ax_b.set_ylabel("Bias")
        ax_s.set_ylabel("Std dev")
        ax_b.set_title(rf"$\gamma = {gamma}$  — Bias")
        ax_s.set_title(rf"$\gamma = {gamma}$  — Std")

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "pooling_gamma_bias.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GAMMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]
R = 200
N_SMALL = 5_000
N_LARGE = 50_000


def main():
    print("=== Gamma sweep, N_small ===")
    df_small, diag_small = run_gamma_sweep(GAMMA_VALUES, N=N_SMALL, R=R, seed=42)

    print("\n=== Gamma sweep, N_large ===")
    df_large, diag_large = run_gamma_sweep(GAMMA_VALUES, N=N_LARGE, R=R, seed=42)

    print("\n=== Pointwise bias curves ===")
    bias_results = run_bias_curves([1.0, 2.0], N=N_LARGE, R=R, seed=99)

    # ---- Tables ----
    print("\n--- MAE summary (N_small) ---")
    tbl = (df_small.groupby(["gamma", "method"])["mae_grid"]
           .agg(mean="mean", std="std").reset_index()
           .pivot(index="gamma", columns="method", values=["mean", "std"]))
    print(tbl.to_string())

    print("\n--- MAE summary (N_large) ---")
    tbl2 = (df_large.groupby(["gamma", "method"])["mae_grid"]
            .agg(mean="mean", std="std").reset_index()
            .pivot(index="gamma", columns="method", values=["mean", "std"]))
    print(tbl2.to_string())

    print("\n--- Collinearity diagnostics (N_large) ---")
    print(diag_large.to_string(index=False))

    # ---- Figures ----
    plot_mae_vs_gamma(df_small, df_large, GAMMA_VALUES)
    plot_collinearity(diag_small, diag_large, GAMMA_VALUES)
    plot_bias_curves(bias_results, [1.0, 2.0])


if __name__ == "__main__":
    main()
