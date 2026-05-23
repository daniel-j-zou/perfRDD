"""
test_phi0_sweep_gamma05.py

Same as test_phi0_sweep.py but with gamma=0.5 (sharper utility around phi*).

With smaller gamma, P(Q>phi|eta) = 1 - Phi((phi-eta)/gamma) is more step-like,
giving U(phi) a more pronounced peak at phi* and less of the flat-landscape
problem that made phi_0=0 the worst case in the gamma=1 sweep.

phi_0 range reduced to [0, 1.5] since P(D=1|eta=phi*) = 1-Phi(phi_0/0.5)
hits ~0.001 by phi_0=1.5.
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
GAMMA       = 0.5
C           = 1.0
N           = 5_000
M           = 100
B           = 50
PHI_GRID    = np.linspace(-3.0, 3.0, 500)
PHI0_VALUES = np.linspace(0.0, 1.5, 10)   # P(D=1|phi*) from 0.50 to ~0.001


def estimate_phi_star(x, q, y, D):
    res = estimate_alpha(x, q, y, D, pooled=True, eta_grid=None,
                         use_ridge=False, treat_basis="bspline")
    if res is None:
        return None
    alpha_all = eval_basis(res["eta_hat"], res["info_treat"]) @ res["omega_treat"]
    util      = compute_utility(alpha_all, res["eta_hat"], q, C, PHI_GRID)
    return float(PHI_GRID[np.argmax(util)])


def run_phi0(phi0, true_phi_star, seed=42):
    rng_master = np.random.default_rng(seed)
    rows = []

    for m in range(M):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        x, y, q, D = pGen(N, phi0=phi0, gamma=GAMMA, rng=rng)

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


def plot_results(summary):
    p_ident = summary["p_ident"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        rf"Estimation quality vs identification at $\phi^*$  ($\gamma=0.5$, sharper utility)"
        "\n"
        rf"(fixed $\phi^*\approx 0$, $c=1.0$; $\phi_0$ swept 0 to 1.5; "
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
        ax.invert_xaxis()

    _ax(axes[0, 0], summary["rmse"].values, "RMSE", r"RMSE of $\hat\phi^*$")
    _ax(axes[0, 1], summary["bias"].values, "Bias", r"Bias of $\hat\phi^*$", hline=0)
    _ax(axes[1, 0], summary["mc_std"].values, "Std dev",
        "MC std vs mean bootstrap SE",
        y2=summary["boot_se"].values, y2label="Mean bootstrap SE")
    _ax(axes[1, 1], summary["coverage"].values * 100, "Coverage (%)",
        r"Bootstrap 95\% CI coverage", hline=95)

    ax2 = axes[0, 0].twiny()
    ax2.set_xlim(axes[0, 0].get_xlim())
    tick_p  = summary["p_ident"].values[::2]
    tick_ph = summary["phi0"].values[::2]
    ax2.set_xticks(tick_p)
    ax2.set_xticklabels([f"{v:.2f}" for v in tick_ph], fontsize=8)
    ax2.set_xlabel(r"$\phi_0$", fontsize=9)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "phi0_sweep_gamma05_results.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def main():
    print(f"Computing true phi_star for c={C}, gamma={GAMMA} ...")
    true_phi_star, _, _ = compute_true_phi_star(C, gamma=GAMMA)
    print(f"  phi_star = {true_phi_star:.4f}")
    print(f"  phi_0 values: {PHI0_VALUES.round(2)}")
    print(f"  M={M}, B={B}, N={N}, gamma={GAMMA}\n")

    records = []
    for phi0 in PHI0_VALUES:
        p_ident = float(1.0 - sp_norm.cdf(phi0 / GAMMA))
        print(f"phi_0 = {phi0:.2f}  P(D=1|eta=phi*) = {p_ident:.3f} ...",
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
            phi0=phi0, p_ident=p_ident, true_phi_star=true_phi_star,
            rmse=rmse, bias=bias, mc_std=mc_std,
            boot_se=boot_se, coverage=coverage, ci_width=ci_width,
            n_reps=len(df),
        ))

    summary = pd.DataFrame(records)

    print("\n" + "=" * 75)
    print(f"{'phi_0':>6} {'P(D=1|phi*)':>12} {'Bias':>8} {'RMSE':>8} "
          f"{'BootSE':>8} {'Coverage':>10} {'CI width':>10}")
    print("-" * 75)
    for _, r in summary.iterrows():
        print(f"{r.phi0:>6.2f} {r.p_ident:>12.3f} {r.bias:>+8.3f} {r.rmse:>8.3f} "
              f"{r.boot_se:>8.3f} {r.coverage * 100:>9.1f}% {r.ci_width:>10.3f}")

    outfile = os.path.join(OUTDIR, "phi0_sweep_gamma05_table.txt")
    summary.to_csv(outfile.replace(".txt", ".csv"), index=False)
    with open(outfile, "w") as f:
        f.write(summary.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nTable saved to {outfile}")

    plot_results(summary)


if __name__ == "__main__":
    main()
