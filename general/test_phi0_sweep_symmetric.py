"""
test_phi0_sweep_symmetric.py

Extends the phi_0 sweep to negative values: phi_0 in linspace(-2.5, 2.5, 19).

phi_0 < 0: more than 50% treated, P(D=1|eta=phi*) > 0.5
phi_0 = 0: 50% treated, P(D=1|eta=phi*) = 0.50  (= phi*)
phi_0 > 0: less than 50% treated, P(D=1|eta=phi*) < 0.50

Uses the steeper alpha DGP (k=5) and no-ridge B-spline estimator.
No bootstrap — just bias/RMSE over M=200 reps for speed.
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
from test_threshold_distance import compute_utility

GAMMA       = 1.0
C           = 1.0
K_ALPHA     = 5
BETA        = 1.0
N           = 50_000
M           = 200
PHI_GRID    = np.linspace(-3.0, 3.0, 500)
PHI0_VALUES = np.linspace(-2.5, 2.5, 19)   # symmetric around 0


def steep_alpha(eta):
    return 2.0 * np.exp(K_ALPHA * eta) / (1.0 + np.exp(K_ALPHA * eta))


def pGen_steep(n, phi0, rng=None):
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
    return x, y, q, D


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
    errors, mae_alphas = [], []
    for _ in range(M):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        x, y, q, D = pGen_steep(N, phi0=phi0, rng=rng)
        res = estimate_alpha(x, q, y, D, pooled=True, eta_grid=None,
                             use_ridge=False, treat_basis="bspline")
        if res is None:
            continue
        alpha_hat = eval_basis(res["eta_hat"], res["info_treat"]) @ res["omega_treat"]
        util      = compute_utility(alpha_hat, res["eta_hat"], q, C, PHI_GRID)
        phi_hat   = float(PHI_GRID[np.argmax(util)])
        errors.append(phi_hat - true_phi_star)

        tr_mask = D == 1
        mae_alphas.append(float(np.mean(np.abs(
            alpha_hat[tr_mask] - steep_alpha(res["eta_hat"][tr_mask])
        ))))

    errors = np.array(errors)
    return {
        "bias":      float(errors.mean()),
        "rmse":      float(np.sqrt((errors**2).mean())),
        "mc_std":    float(errors.std()),
        "mae_alpha": float(np.mean(mae_alphas)),
        "p_treat":   float(D.mean()),   # last dataset's treatment rate
        "n_reps":    len(errors),
    }


def main():
    print("Computing true phi_star (k=5 alpha) ...")
    true_phi_star = compute_true_phi_star_steep()
    print(f"  phi_star = {true_phi_star:.4f}\n")

    records = []
    for phi0 in PHI0_VALUES:
        p_ident = float(1.0 - sp_norm.cdf(phi0 / GAMMA))
        print(f"phi_0 = {phi0:+.2f}  P(D=1|phi*) = {p_ident:.3f} ...",
              end=" ", flush=True)
        stats = run_phi0(phi0, true_phi_star, seed=42)
        print(f"bias={stats['bias']:+.3f}  rmse={stats['rmse']:.3f}  "
              f"mae_alpha={stats['mae_alpha']:.3f}  p_treat={stats['p_treat']:.2f}")
        records.append(dict(phi0=phi0, p_ident=p_ident,
                            true_phi_star=true_phi_star, **stats))

    df = pd.DataFrame(records)

    # ── Table ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'phi_0':>6} {'P(D=1|phi*)':>12} {'P(treat)':>9} "
          f"{'Bias':>8} {'RMSE':>8} {'MAE(alpha)':>11}")
    print("-" * 75)
    for _, r in df.iterrows():
        print(f"{r.phi0:>+6.2f} {r.p_ident:>12.3f} {r.p_treat:>9.3f} "
              f"{r.bias:>+8.3f} {r.rmse:>8.3f} {r.mae_alpha:>11.4f}")

    outfile = os.path.join(OUTDIR, "phi0_sweep_symmetric_N50k_table.txt")
    df.to_csv(outfile.replace(".txt", ".csv"), index=False)
    with open(outfile, "w") as f:
        f.write(df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nTable saved to {outfile}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        rf"Symmetric $\phi_0$ sweep  ($\alpha(\eta)=2\sigma(5\eta)$, "
        rf"no ridge, B-spline; N={N}, M={M})"
        "\n"
        r"$\phi^*\approx 0$;  left of centre = more treated,  right = fewer treated",
        fontsize=11,
    )

    phi0s = df["phi0"].values

    ax = axes[0]
    ax.plot(phi0s, df["rmse"].values, "o-", color="steelblue", linewidth=2)
    ax.axvline(true_phi_star, color="gray", linestyle="--", linewidth=1,
               label=rf"$\phi^*={true_phi_star:.2f}$")
    ax.set_xlabel(r"$\phi_0$")
    ax.set_ylabel("RMSE")
    ax.set_title(r"RMSE of $\hat\phi^*$")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(phi0s, df["bias"].values, "o-", color="steelblue", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(true_phi_star, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\phi_0$")
    ax.set_ylabel("Bias")
    ax.set_title(r"Bias of $\hat\phi^*$")

    ax = axes[2]
    ax.plot(phi0s, df["mae_alpha"].values, "o-", color="crimson", linewidth=2)
    ax.axvline(true_phi_star, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\phi_0$")
    ax.set_ylabel("MAE")
    ax.set_title(r"MAE of $\hat\alpha(\eta)$ on treated")

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "phi0_sweep_symmetric_N50k_results.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
