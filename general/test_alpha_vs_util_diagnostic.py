"""
test_alpha_vs_util_diagnostic.py

Diagnoses whether the high RMSE of phi_hat_star at phi_0=0 comes from:
  (A) poor estimation of alpha(eta), or
  (B) the utility maximization step given alpha_hat

Method
------
For each dataset, compute phi_hat_star three ways:
  1. "estimated": plug in alpha_hat from PLM  (what we normally do)
  2. "oracle_alpha": plug in TRUE alpha(eta) at eta_hat  (bypasses alpha estimation)
  3. "oracle_both": plug in TRUE alpha(eta) at TRUE eta  (perfect first stage too)

If RMSE(estimated) >> RMSE(oracle_alpha):  error comes from alpha estimation.
If RMSE(oracle_alpha) >> RMSE(oracle_both): error comes from eta_hat (first stage).
If all three are similar:                  error comes from utility maximization.

Uses the steeper alpha DGP (k=5) at phi_0 in {0.00, 0.28, 0.56, 1.11}.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUTDIR)
from test_pooling import estimate_alpha, eval_basis
from test_threshold_distance import compute_utility, compute_true_phi_star

# ── Parameters ────────────────────────────────────────────────────────────────
GAMMA    = 1.0
C        = 1.0
K_ALPHA  = 5
BETA     = 1.0
N        = 5_000
M        = 200
PHI_GRID = np.linspace(-3.0, 3.0, 500)
PHI0_VALUES = [0.00, 0.28, 0.56, 1.11]


def steep_alpha(eta):
    return 2.0 * np.exp(K_ALPHA * eta) / (1.0 + np.exp(K_ALPHA * eta))


def pGen_steep(n, phi0=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x     = rng.normal(0.0, 1.0, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    a     = signs * rng.exponential(1.0, size=n)
    eta   = rng.normal(a, 1.0, size=n)
    nu    = rng.normal(3 * np.exp(a) / (1 + np.exp(a)), 1.0, size=n)
    w     = rng.normal(steep_alpha(eta), 0.1, size=n)
    q     = GAMMA * x + eta
    D     = (q > phi0).astype(float)
    y     = D * w + BETA * x + nu
    return x, y, q, D, eta   # also return true eta


def compute_true_phi_star_steep(n_mc=200_000, seed=0):
    rng   = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=n_mc)
    a     = signs * rng.exponential(1.0, size=n_mc)
    eta   = rng.normal(a, 1.0, size=n_mc)
    alpha_t = steep_alpha(eta)
    util = np.array([
        np.mean((alpha_t - C) * (1.0 - sp_norm.cdf((phi - eta) / GAMMA)))
        for phi in PHI_GRID
    ])
    return float(PHI_GRID[np.argmax(util)])


def run_phi0(phi0, true_phi_star, seed=42):
    rng_master = np.random.default_rng(seed)
    rows = []

    for _ in range(M):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        x, y, q, D, eta_true = pGen_steep(N, phi0=phi0, rng=rng)

        # ── 1. Estimated alpha ────────────────────────────────────────────────
        res = estimate_alpha(x, q, y, D, pooled=True, eta_grid=None,
                             use_ridge=False, treat_basis="bspline")
        if res is None:
            continue
        eta_hat   = res["eta_hat"]
        alpha_hat = eval_basis(eta_hat, res["info_treat"]) @ res["omega_treat"]
        util_est  = compute_utility(alpha_hat, eta_hat, q, C, PHI_GRID)
        phi_est   = float(PHI_GRID[np.argmax(util_est)])

        # MAE of alpha_hat vs true alpha over treated eta range
        tr_mask   = D == 1
        mae_alpha = float(np.mean(np.abs(
            alpha_hat[tr_mask] - steep_alpha(eta_hat[tr_mask])
        )))

        # ── 2. Oracle alpha (true alpha at eta_hat) ───────────────────────────
        alpha_oracle = steep_alpha(eta_hat)
        util_orac    = compute_utility(alpha_oracle, eta_hat, q, C, PHI_GRID)
        phi_orac     = float(PHI_GRID[np.argmax(util_orac)])

        # ── 3. Oracle both (true alpha at true eta) ───────────────────────────
        alpha_both = steep_alpha(eta_true)
        util_both  = compute_utility(alpha_both, eta_true, q, C, PHI_GRID)
        phi_both   = float(PHI_GRID[np.argmax(util_both)])

        rows.append({
            "phi_est":   phi_est,
            "phi_orac":  phi_orac,
            "phi_both":  phi_both,
            "err_est":   phi_est  - true_phi_star,
            "err_orac":  phi_orac - true_phi_star,
            "err_both":  phi_both - true_phi_star,
            "mae_alpha": mae_alpha,
        })

    return pd.DataFrame(rows)


def main():
    print("Computing true phi_star (k=5 alpha) ...")
    true_phi_star = compute_true_phi_star_steep()
    print(f"  phi_star = {true_phi_star:.4f}\n")

    records = []
    for phi0 in PHI0_VALUES:
        p_ident = float(1.0 - sp_norm.cdf(phi0 / GAMMA))
        print(f"phi_0 = {phi0:.2f}  P(D=1|phi*) = {p_ident:.3f} ...", flush=True)
        df = run_phi0(phi0, true_phi_star, seed=42)

        rmse_est  = float(np.sqrt((df["err_est"]  ** 2).mean()))
        rmse_orac = float(np.sqrt((df["err_orac"] ** 2).mean()))
        rmse_both = float(np.sqrt((df["err_both"] ** 2).mean()))
        mae_alpha = float(df["mae_alpha"].mean())

        print(f"  MAE(alpha_hat):          {mae_alpha:.4f}")
        print(f"  RMSE phi_hat (estimated): {rmse_est:.4f}")
        print(f"  RMSE phi_hat (oracle-α):  {rmse_orac:.4f}  [true alpha, est eta]")
        print(f"  RMSE phi_hat (oracle-all):{rmse_both:.4f}  [true alpha, true eta]")

        records.append(dict(
            phi0=phi0, p_ident=p_ident,
            mae_alpha=mae_alpha,
            rmse_est=rmse_est, rmse_orac=rmse_orac, rmse_both=rmse_both,
        ))

    df_sum = pd.DataFrame(records)

    # ── Table ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'phi_0':>6} {'P(D=1|φ*)':>10} {'MAE(α̂)':>10} "
          f"{'RMSE(est)':>12} {'RMSE(orc-α)':>12} {'RMSE(orc-all)':>14}")
    print("-" * 80)
    for _, r in df_sum.iterrows():
        print(f"{r.phi0:>6.2f} {r.p_ident:>10.3f} {r.mae_alpha:>10.4f} "
              f"{r.rmse_est:>12.4f} {r.rmse_orac:>12.4f} {r.rmse_both:>14.4f}")

    outfile = os.path.join(OUTDIR, "alpha_vs_util_diagnostic.txt")
    df_sum.to_csv(outfile.replace(".txt", ".csv"), index=False)
    with open(outfile, "w") as f:
        f.write(df_sum.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nTable saved to {outfile}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"Source of error in $\hat\phi^*$: $\alpha$ estimation vs utility maximisation"
        "\n"
        rf"($k={K_ALPHA}$, $\gamma=1$, $N={N}$, $M={M}$ reps)",
        fontsize=12,
    )

    phi0s = df_sum["phi0"].values
    x_pos = np.arange(len(phi0s))
    labels = [f"φ₀={v:.2f}\n(P={p:.2f})"
              for v, p in zip(df_sum["phi0"], df_sum["p_ident"])]

    ax = axes[0]
    ax.bar(x_pos, df_sum["mae_alpha"], color="steelblue")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE"); ax.set_title(r"MAE of $\hat\alpha(\eta)$ on treated units")

    ax = axes[1]
    w = 0.25
    ax.bar(x_pos - w, df_sum["rmse_est"],  width=w, label="estimated α̂",      color="steelblue")
    ax.bar(x_pos,     df_sum["rmse_orac"], width=w, label="oracle α (est η̂)", color="darkorange")
    ax.bar(x_pos + w, df_sum["rmse_both"], width=w, label="oracle α & η",      color="crimson")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE"); ax.set_title(r"RMSE of $\hat\phi^*$ by oracle level")
    ax.legend(fontsize=8)

    ax = axes[2]
    reduction_alpha = df_sum["rmse_est"] - df_sum["rmse_orac"]
    reduction_eta   = df_sum["rmse_orac"] - df_sum["rmse_both"]
    ax.bar(x_pos - w/2, reduction_alpha, width=w,
           label="reduction from perfect α", color="steelblue")
    ax.bar(x_pos + w/2, reduction_eta,   width=w,
           label="reduction from perfect η", color="darkorange")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE reduction")
    ax.set_title("How much each oracle helps")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "alpha_vs_util_diagnostic.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
