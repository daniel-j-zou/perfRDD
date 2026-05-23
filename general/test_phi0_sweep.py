"""
test_phi0_sweep.py

Does estimation quality degrade smoothly as phi_0 moves away from phi_star?

Design
------
Fix c = 1.0  =>  phi_star ≈ 0  (true optimum never changes)
Sweep phi_0 in linspace(0, 2.5, 10)
  - phi_0 = 0  : current threshold = optimum, P(D=1|eta=phi*) = 50%
  - phi_0 = 2.5: far from optimum,          P(D=1|eta=phi*) ≈ 6%

At each phi_0:
  Outer: M = 100 independent datasets  ->  sampling distribution of phi_hat_star
  Inner: B =  50 bootstrap resamples   ->  bootstrap SE and CI per dataset

Key x-axis: P(D=1|eta=phi_star) = 1 - Phi(phi_0 / gamma)
  = identification quality at the optimal threshold.

Output (general/)
-----------------
  phi0_sweep_results.png   -- RMSE, bias, boot SE, coverage vs identification quality
  phi0_sweep_table.txt     -- numeric summary
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
N        = 5_000
M        = 100
B        = 50
PHI_GRID = np.linspace(-3.0, 3.0, 500)
PHI0_VALUES = np.linspace(0.0, 2.5, 10)   # 0.0, 0.28, 0.56, ..., 2.5


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


# ===========================================================================
# Run one phi_0 value
# ===========================================================================

def run_phi0(phi0, true_phi_star, seed=42):
    rng_master = np.random.default_rng(seed)
    rows = []

    for m in range(M):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        x, y, q, D = pGen(N, phi0=phi0, rng=rng)

        phi_hat = estimate_phi_star(x, q, y, D)
        if phi_hat is None:
            continue

        boot_stars = []
        for _ in range(B):
            rng_b = np.random.default_rng(rng_master.integers(0, 2**31))
            idx   = rng_b.choice(N, size=N, replace=True)
            phi_b = estimate_phi_star(x[idx], q[idx], y[idx], D[idx])
            if phi_b is not None:
                boot_stars.append(phi_b)

        if len(boot_stars) < B // 2:
            continue

        boot    = np.array(boot_stars)
        boot_lo = float(np.percentile(boot, 2.5))
        boot_hi = float(np.percentile(boot, 97.5))

        rows.append({
            "phi_hat":  phi_hat,
            "error":    phi_hat - true_phi_star,
            "sq_err":   (phi_hat - true_phi_star) ** 2,
            "boot_se":  float(np.std(boot)),
            "boot_lo":  boot_lo,
            "boot_hi":  boot_hi,
            "ci_width": boot_hi - boot_lo,
            "covers":   int(boot_lo <= true_phi_star <= boot_hi),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Plot
# ===========================================================================

def plot_results(summary):
    # x-axis: P(D=1|eta=phi_star) = identification quality
    p_ident = summary["p_ident"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        rf"Estimation quality vs identification at $\phi^*$"
        "\n"
        rf"(fixed $\phi^*\approx 0$, $c=1.0$; $\phi_0$ swept from 0 to 2.5; "
        rf"N={N}, M={M}, B={B})",
        fontsize=12,
    )

    def _ax(ax, y, ylabel, title, hline=None, y2=None, y2label=None):
        ax.plot(p_ident, y, "o-", color="steelblue", linewidth=2, markersize=6)
        if y2 is not None:
            ax.plot(p_ident, y2, "s--", color="crimson", linewidth=1.5,
                    markersize=5, label=y2label)
            ax.legend(fontsize=9)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel(r"$P(D=1\mid\eta=\phi^*)$  [identification quality]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.invert_xaxis()   # left = best identification, right = worst

    _ax(axes[0, 0],
        summary["rmse"].values, "RMSE", "RMSE of $\\hat\\phi^*$")

    _ax(axes[0, 1],
        summary["bias"].values, "Bias", "Bias of $\\hat\\phi^*$",
        hline=0)

    _ax(axes[1, 0],
        summary["mc_std"].values, "Std dev",
        "MC std vs mean bootstrap SE",
        y2=summary["boot_se"].values, y2label="Mean bootstrap SE")

    _ax(axes[1, 1],
        summary["coverage"].values * 100, "Coverage (%)",
        "Bootstrap 95\\% CI coverage",
        hline=95)

    # Annotate phi_0 values on the top axis of RMSE panel
    ax2 = axes[0, 0].twiny()
    ax2.set_xlim(axes[0, 0].get_xlim())
    tick_p  = summary["p_ident"].values[::2]
    tick_ph = summary["phi0"].values[::2]
    ax2.set_xticks(tick_p)
    ax2.set_xticklabels([f"{v:.1f}" for v in tick_ph], fontsize=8)
    ax2.set_xlabel(r"$\phi_0$", fontsize=9)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "phi0_sweep_v2_results.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"Computing true phi_star for c={C} ...")
    true_phi_star, _, _ = compute_true_phi_star(C)
    print(f"  phi_star = {true_phi_star:.4f}")
    print(f"  phi_0 values: {PHI0_VALUES.round(2)}")
    print(f"  M={M}, B={B}, N={N}\n")

    records = []
    for phi0 in PHI0_VALUES:
        p_ident   = float(1.0 - sp_norm.cdf(phi0 / GAMMA))
        p_overall = float(1.0 - sp_norm.cdf(
            phi0 / np.sqrt(1 + 2.0)   # rough: Var(eta) ≈ 2 for pGen
        ))
        print(f"phi_0 = {phi0:.2f}  "
              f"P(D=1|eta=phi*) = {p_ident:.3f}  "
              f"P(D=1) ≈ {p_overall:.3f} ...",
              flush=True)

        df = run_phi0(phi0, true_phi_star, seed=42)

        rmse     = float(np.sqrt(df["sq_err"].mean()))
        bias     = float(df["error"].mean())
        mc_std   = float(df["error"].std())
        boot_se  = float(df["boot_se"].mean())
        coverage = float(df["covers"].mean())
        ci_width = float(df["ci_width"].mean())

        print(f"  RMSE={rmse:.3f}  bias={bias:+.3f}  "
              f"boot_se={boot_se:.3f}  coverage={coverage:.1%}")

        records.append(dict(
            phi0=phi0,
            p_ident=p_ident,
            true_phi_star=true_phi_star,
            rmse=rmse, bias=bias, mc_std=mc_std,
            boot_se=boot_se, coverage=coverage, ci_width=ci_width,
            n_reps=len(df),
        ))

    summary = pd.DataFrame(records)

    # ── Table ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'phi_0':>6} {'P(D=1|phi*)':>12} {'Bias':>8} {'RMSE':>8} "
          f"{'BootSE':>8} {'Coverage':>10} {'CI width':>10}")
    print("-" * 75)
    for _, r in summary.iterrows():
        print(f"{r.phi0:>6.2f} {r.p_ident:>12.3f} {r.bias:>+8.3f} {r.rmse:>8.3f} "
              f"{r.boot_se:>8.3f} {r.coverage * 100:>9.1f}% {r.ci_width:>10.3f}")

    outfile = os.path.join(OUTDIR, "phi0_sweep_v2_table.txt")
    summary.to_csv(outfile.replace(".txt", ".csv"), index=False)
    with open(outfile, "w") as f:
        f.write(summary.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nTable saved to {outfile}")

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_results(summary)


if __name__ == "__main__":
    main()
