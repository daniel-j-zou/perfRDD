"""
test_n_sweep.py

Does the bias from phi_0 != phi_star shrink as N grows?
Version 2: no ridge regularisation, B-spline treatment basis.

Design
------
Fix c = 1.0  =>  phi_star ≈ -0.006  (true optimum is near phi_0=0)
Fix three phi_0 values: 0.0 (P~0.50), 0.83 (P~0.20), 1.67 (P~0.05)
Sweep N in {1000, 2000, 5000, 10000, 50000}
M = 100 MC reps per cell (no bootstrap — just bias/RMSE)

Key question: does bias ~ 1/sqrt(N) (standard) or does it shrink slower?

Output (general/)
-----------------
  n_sweep_v2_results.png   -- bias and RMSE vs N, one line per phi_0
  n_sweep_v2_table.txt     -- numeric summary
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
from test_threshold_distance import pGen, compute_utility, compute_true_phi_star

# ── Parameters ───────────────────────────────────────────────────────────────
GAMMA    = 1.0
C        = 1.0
M        = 100
PHI_GRID = np.linspace(-3.0, 3.0, 500)

PHI0_VALUES = [0.0, 0.83, 1.67]   # P(D=1|eta=phi*) ≈ 0.50, 0.20, 0.05
N_VALUES    = [1_000, 2_000, 5_000, 10_000, 50_000]

COLORS = {0.0: "steelblue", 0.83: "darkorange", 1.67: "crimson"}
LABELS = {
    0.0:  r"$\phi_0=0.00$, $P\approx 0.50$",
    0.83: r"$\phi_0=0.83$, $P\approx 0.20$",
    1.67: r"$\phi_0=1.67$, $P\approx 0.05$",
}


# ===========================================================================
# Estimation
# ===========================================================================

def estimate_phi_star(x, q, y, D):
    res = estimate_alpha(x, q, y, D, pooled=True, eta_grid=None,
                         use_ridge=False, treat_basis="bspline")
    if res is None:
        return None
    alpha_all = eval_basis(res["eta_hat"], res["info_treat"]) @ res["omega_treat"]
    util      = compute_utility(alpha_all, res["eta_hat"], q, C, PHI_GRID)
    return float(PHI_GRID[np.argmax(util)])


def run_cell(phi0, N, true_phi_star, seed=42):
    rng_master = np.random.default_rng(seed)
    errors = []
    for _ in range(M):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        x, y, q, D = pGen(N, phi0=phi0, rng=rng)
        phi_hat = estimate_phi_star(x, q, y, D)
        if phi_hat is not None:
            errors.append(phi_hat - true_phi_star)
    errors = np.array(errors)
    return {
        "bias":   float(errors.mean()),
        "rmse":   float(np.sqrt((errors**2).mean())),
        "mc_std": float(errors.std()),
        "n_reps": len(errors),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Computing true phi_star for c=1.0 ...")
    true_phi_star, _, _ = compute_true_phi_star(C)
    print(f"  phi_star = {true_phi_star:.4f}\n")

    records = []
    for phi0 in PHI0_VALUES:
        p_ident = float(1.0 - sp_norm.cdf(phi0 / GAMMA))
        for N in N_VALUES:
            print(f"phi_0={phi0:.2f}  N={N:6,}  P(D=1|phi*)={p_ident:.3f} ...",
                  end=" ", flush=True)
            stats = run_cell(phi0, N, true_phi_star, seed=42)
            print(f"bias={stats['bias']:+.3f}  rmse={stats['rmse']:.3f}")
            records.append(dict(phi0=phi0, N=N, p_ident=p_ident,
                                true_phi_star=true_phi_star, **stats))

    df = pd.DataFrame(records)

    # ── Table ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'phi_0':>6} {'N':>7} {'P(D=1|phi*)':>12} "
          f"{'Bias':>8} {'RMSE':>8} {'Std':>8}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r.phi0:>6.2f} {int(r.N):>7,} {r.p_ident:>12.3f} "
              f"{r.bias:>+8.3f} {r.rmse:>8.3f} {r.mc_std:>8.3f}")

    outfile = os.path.join(OUTDIR, "n_sweep_v2_table.txt")
    df.to_csv(outfile.replace(".txt", ".csv"), index=False)
    with open(outfile, "w") as f:
        f.write(df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nTable saved to {outfile}")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        r"Does bias shrink with $N$? Three $\phi_0$ scenarios"
        "\n"
        rf"(no ridge, B-spline basis; fixed $\phi^*\approx 0$, $c=1.0$; M={M} MC reps)",
        fontsize=12,
    )

    for phi0 in PHI0_VALUES:
        sub = df[df["phi0"] == phi0].sort_values("N")
        col = COLORS[phi0]
        lbl = LABELS[phi0]

        axes[0].plot(sub["N"], sub["bias"].abs(), "o-", color=col,
                     linewidth=2, markersize=6, label=lbl)
        axes[1].plot(sub["N"], sub["rmse"], "o-", color=col,
                     linewidth=2, markersize=6, label=lbl)

    # Reference 1/sqrt(N) line anchored at phi0=0, N=1000
    ref_row = df[(df["phi0"] == 0.0) & (df["N"] == 1_000)].iloc[0]
    ns = np.array(N_VALUES, dtype=float)
    ref_bias = ref_row["bias"] * np.sqrt(1_000 / ns)
    for ax in axes:
        ax.plot(ns, np.abs(ref_bias), "k--", linewidth=1,
                alpha=0.5, label=r"$\propto 1/\sqrt{N}$ reference")

    for ax, ylabel, title in zip(
        axes,
        ["|Bias|", "RMSE"],
        [r"|Bias| of $\hat\phi^*$ vs $N$", r"RMSE of $\hat\phi^*$ vs $N$"],
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$N$  (log scale)")
        ax.set_ylabel(ylabel + "  (log scale)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([f"{n:,}" for n in N_VALUES], fontsize=8)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "n_sweep_v2_results.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
