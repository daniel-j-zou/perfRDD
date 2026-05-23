"""
test_threshold_distance.py

Does estimation of phi_star get worse when phi_star is far from phi_0?

Intuition: identification of alpha(eta) relies on having both treated and
control units at each eta level.  P(D=1 | eta) = P(Q > phi_0 | eta) is close
to 0.5 only near eta ≈ phi_0.  When phi_star is far from phi_0, alpha(eta)
is poorly identified at eta ≈ phi_star, distorting both the point estimate
phi_hat_star and its bootstrap uncertainty.

Three scenarios (gamma=1, N=5000)
----------------------------------
  Close      phi_0 = 0.0, c = 1.0  =>  phi_star ≈ 0     P(D=1|eta=phi_star) ≈ 0.50
  Far-raise  phi_0 = 0.0, c = 1.5  =>  phi_star ≈ 1.1   P(D=1|eta=phi_star) ≈ 0.86
  Far-lower  phi_0 = 1.5, c = 1.0  =>  phi_star ≈ 0     P(D=1|eta=phi_star) ≈ 0.07

For "far-lower": data collected at an imbalanced current threshold (phi_0=1.5,
only ~7% treated).  The optimal policy is at the balanced point phi_star≈0,
but identification of alpha(0) is poor because very few treated units have
eta near 0.

Method
------
  Outer: M independent datasets  →  sampling distribution of phi_hat_star
  Inner: B bootstrap resamples   →  bootstrap SE and 95% CI per dataset

Metrics
-------
  Bias, RMSE of phi_hat_star
  Mean bootstrap SE
  Bootstrap CI coverage (% of CIs containing true phi_star)
  Mean CI width

Output  (general/)
------------------
  threshold_dist_sampling.png   -- distribution of phi_hat_star errors
  threshold_dist_bootstrap.png  -- bootstrap SE and coverage
  threshold_dist_ident.png      -- P(D=1|eta) identification diagnostic
  threshold_dist_table.txt      -- summary table
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
from test_pooling import estimate_alpha, eval_basis, true_alpha

# ── Experiment parameters ────────────────────────────────────────────────────
GAMMA   = 1.0
BETA    = 1.0
N       = 5_000
M       = 200   # outer MC replications
B       = 200   # bootstrap replications per dataset
PHI_GRID = np.linspace(-3.0, 3.0, 500)

SCENARIOS = [
    dict(name="close",     label="Close",      phi0=0.0, c=1.0),
    dict(name="far_raise", label="Far-raise",  phi0=0.0, c=1.5),
    dict(name="far_lower", label="Far-lower",  phi0=1.5, c=1.0),
]
COLORS = {"close": "steelblue", "far_raise": "crimson", "far_lower": "darkorange"}


# ===========================================================================
# DGP
# ===========================================================================

def pGen(n, phi0=0.0, gamma=GAMMA, beta=BETA, rng=None):
    """
    pGen DGP with configurable initial threshold phi0.
    Treatment: D = 1{Q > phi0}.
    True treatment effect: alpha(eta) = 2 exp(eta) / (1 + exp(eta)).
    """
    if rng is None:
        rng = np.random.default_rng()
    x     = rng.normal(0.0, 1.0, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    a     = signs * rng.exponential(1.0, size=n)
    eta   = rng.normal(a, 1.0, size=n)
    nu    = rng.normal(3 * np.exp(a) / (1 + np.exp(a)), 1.0, size=n)
    w     = rng.normal(2 * np.exp(eta) / (1 + np.exp(eta)), 0.1, size=n)
    q     = gamma * x + eta
    D     = (q > phi0).astype(float)
    y     = D * w + beta * x + nu
    return x, y, q, D


# ===========================================================================
# Utility
# ===========================================================================

def compute_utility(alpha_all, eta_hat, q, c, phi_grid):
    """
    U(phi) = mean_i [ (alpha_hat_i - c) * P_hat(Q > phi | eta_hat_i) ]

    Uses the empirical distribution of gX = Q - eta_hat to estimate
    P(Q > phi | eta_i) = P(gX > phi - eta_i).
    Vectorised over all phi simultaneously to avoid a Python loop.
    """
    gX_sorted = np.sort(q - eta_hat)
    n = len(eta_hat)
    G = len(phi_grid)

    # thresh[j, i] = phi_grid[j] - eta_hat[i]
    thresh      = phi_grid[:, None] - eta_hat[None, :]          # (G, n)
    frac_below  = np.searchsorted(gX_sorted,
                                  thresh.ravel()).reshape(G, n) / n
    probs       = 1.0 - frac_below                              # P(Q>phi|eta_i)

    return ((alpha_all[None, :] - c) * probs).mean(axis=1)      # (G,)


# ===========================================================================
# True phi_star
# ===========================================================================

def compute_true_phi_star(c, gamma=GAMMA, n_mc=200_000, seed=0):
    """
    Approximate true phi_star = argmax_phi E[(alpha(eta)-c)*P(Q>phi|eta)]
    using a large Monte Carlo draw from the pGen distribution.

    P(Q>phi|eta) = P(gamma*X + eta > phi) = 1 - Phi((phi-eta)/gamma)
    is computed analytically given eta.
    """
    rng   = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=n_mc)
    a     = signs * rng.exponential(1.0, size=n_mc)
    eta   = rng.normal(a, 1.0, size=n_mc)
    alpha_t = true_alpha(eta)

    util = np.array([
        np.mean((alpha_t - c) * (1.0 - sp_norm.cdf((phi - eta) / gamma)))
        for phi in PHI_GRID
    ])
    return float(PHI_GRID[np.argmax(util)]), PHI_GRID, util


# ===========================================================================
# Single-dataset estimation of phi_star
# ===========================================================================

def estimate_phi_star(x, q, y, D, c):
    """Run PLM and return phi_hat_star. Returns None on failure."""
    res = estimate_alpha(x, q, y, D, pooled=True, eta_grid=None)
    if res is None:
        return None
    alpha_all = eval_basis(res["eta_hat"], res["info_treat"]) @ res["omega_treat"]
    util      = compute_utility(alpha_all, res["eta_hat"], q, c, PHI_GRID)
    return float(PHI_GRID[np.argmax(util)])


# ===========================================================================
# MC + bootstrap experiment for one scenario
# ===========================================================================

def run_scenario(phi0, c, true_phi_star, seed=42):
    rng_master = np.random.default_rng(seed)
    rows = []

    for m in range(M):
        if (m + 1) % 40 == 0:
            frac_tr = "(~{:.0%} treated)".format(
                1.0 - sp_norm.cdf((phi0) / GAMMA)  # rough P(Q>phi0)
            )
            print(f"    rep {m+1}/{M} {frac_tr}", flush=True)

        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        x, y, q, D = pGen(N, phi0=phi0, rng=rng)

        phi_hat = estimate_phi_star(x, q, y, D, c)
        if phi_hat is None:
            continue

        # Bootstrap
        boot_stars = []
        for _ in range(B):
            rng_b = np.random.default_rng(rng_master.integers(0, 2**31))
            idx   = rng_b.choice(N, size=N, replace=True)
            phi_b = estimate_phi_star(x[idx], q[idx], y[idx], D[idx], c)
            if phi_b is not None:
                boot_stars.append(phi_b)

        if len(boot_stars) < B // 2:
            continue

        boot     = np.array(boot_stars)
        boot_se  = float(np.std(boot))
        boot_lo  = float(np.percentile(boot, 2.5))
        boot_hi  = float(np.percentile(boot, 97.5))

        rows.append({
            "phi_hat":  phi_hat,
            "error":    phi_hat - true_phi_star,
            "abs_err":  abs(phi_hat - true_phi_star),
            "sq_err":   (phi_hat - true_phi_star) ** 2,
            "boot_se":  boot_se,
            "boot_lo":  boot_lo,
            "boot_hi":  boot_hi,
            "ci_width": boot_hi - boot_lo,
            "covers":   int(boot_lo <= true_phi_star <= boot_hi),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Identification diagnostic: P(D=1 | eta) near phi_star for each scenario
# ===========================================================================

def identification_profile(phi0, gamma=GAMMA):
    """P(D=1|eta) = P(Q > phi_0|eta) = 1 - Phi((phi_0-eta)/gamma)."""
    eta_grid = np.linspace(-3.0, 3.0, 400)
    p_treat  = 1.0 - sp_norm.cdf((phi0 - eta_grid) / gamma)
    return eta_grid, p_treat


# ===========================================================================
# Plotting
# ===========================================================================

def plot_sampling_distribution(results, true_phi_stars):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle(r"Sampling distribution of $\hat\phi^* - \phi^*$  (N=5,000, M=200)",
                 fontsize=13)

    for ax, sc in zip(axes, SCENARIOS):
        df  = results[sc["name"]]
        col = COLORS[sc["name"]]
        tps = true_phi_stars[sc["name"]]
        errors = df["error"].values

        ax.hist(errors, bins=30, color=col, alpha=0.7, density=True, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1.2)
        ax.axvline(np.mean(errors), color=col, linewidth=2, linestyle="--",
                   label=f"Mean bias = {np.mean(errors):+.3f}")

        rmse = float(np.sqrt(df["sq_err"].mean()))
        ax.set_title(f"{sc['label']}\n"
                     rf"$\phi_0={sc['phi0']}$, $c={sc['c']}$, "
                     rf"$\phi^*={tps:.2f}$"
                     f"\nRMSE = {rmse:.3f}")
        ax.set_xlabel(r"$\hat\phi^* - \phi^*$")
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "threshold_dist_sampling.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_bootstrap(results, true_phi_stars):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Bootstrap performance by scenario", fontsize=13)

    labels   = [sc["label"] for sc in SCENARIOS]
    colors   = [COLORS[sc["name"]] for sc in SCENARIOS]

    # Left: MC std vs mean bootstrap SE
    ax = axes[0]
    mc_stds  = [results[sc["name"]]["error"].std()   for sc in SCENARIOS]
    boot_ses = [results[sc["name"]]["boot_se"].mean() for sc in SCENARIOS]
    x_pos = np.arange(len(SCENARIOS))
    w = 0.35
    ax.bar(x_pos - w/2, mc_stds,  width=w, label="MC std of error", color=colors, alpha=0.7)
    ax.bar(x_pos + w/2, boot_ses, width=w, label="Mean bootstrap SE",
           color=colors, alpha=0.4, edgecolor="black", linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Standard deviation of $\\hat\\phi^*$")
    ax.set_title("MC std (solid) vs Bootstrap SE (hatched)")
    ax.legend()

    # Right: coverage and CI width
    ax2 = axes[1]
    coverages  = [results[sc["name"]]["covers"].mean() * 100  for sc in SCENARIOS]
    ci_widths  = [results[sc["name"]]["ci_width"].mean()       for sc in SCENARIOS]
    ax2_twin   = ax2.twinx()

    bars = ax2.bar(x_pos, coverages, color=colors, alpha=0.7, width=0.4)
    ax2_twin.plot(x_pos, ci_widths, "ks--", markersize=8, linewidth=1.5,
                  label="Mean CI width")

    ax2.axhline(95, color="black", linestyle=":", linewidth=1, label="Nominal 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Bootstrap CI coverage (%)")
    ax2_twin.set_ylabel("Mean CI width")
    ax2.set_title("Coverage and CI width")
    ax2.legend(loc="lower left")
    ax2_twin.legend(loc="lower right")

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "threshold_dist_bootstrap.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_identification(true_phi_stars):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(r"Identification: $P(D=1\mid\eta) = P(Q > \phi_0 \mid \eta)$"
                 "\n(balanced = good identification; extreme = poor)",
                 fontsize=12)

    for sc in SCENARIOS:
        eta_grid, p_treat = identification_profile(sc["phi0"])
        tps = true_phi_stars[sc["name"]]
        ax.plot(eta_grid, p_treat, color=COLORS[sc["name"]],
                linewidth=2.5, label=sc["label"])
        # Mark phi_star on the x-axis
        p_at_star = float(1.0 - sp_norm.cdf((sc["phi0"] - tps) / GAMMA))
        ax.plot(tps, p_at_star, "o", color=COLORS[sc["name"]], markersize=9,
                zorder=5)
        ax.annotate(
            f"$\\phi^*={tps:.2f}$\n$P={p_at_star:.2f}$",
            xy=(tps, p_at_star),
            xytext=(tps + 0.2, p_at_star + 0.06 * (1 if p_at_star < 0.7 else -1)),
            fontsize=9, color=COLORS[sc["name"]],
            arrowprops=dict(arrowstyle="->", color=COLORS[sc["name"]], lw=1),
        )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="50/50 balance")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$P(D=1\mid\eta)$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    fig.tight_layout()
    fname = os.path.join(OUTDIR, "threshold_dist_ident.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


# ===========================================================================
# Summary table
# ===========================================================================

def print_table(results, true_phi_stars, outfile):
    lines = [
        f"N={N}, M={M} MC reps, B={B} bootstrap reps, gamma={GAMMA}",
        "",
        f"{'Scenario':<14} {'phi_0':>6} {'c':>5} {'phi_star':>9} "
        f"{'P(D=1|phi*)':>12} {'Bias':>8} {'RMSE':>8} "
        f"{'BootSE':>8} {'Coverage':>10} {'CI width':>10}",
        "-" * 90,
    ]
    for sc in SCENARIOS:
        df   = results[sc["name"]]
        tps  = true_phi_stars[sc["name"]]
        p_id = 1.0 - sp_norm.cdf((sc["phi0"] - tps) / GAMMA)
        lines.append(
            f"{sc['label']:<14} {sc['phi0']:>6.1f} {sc['c']:>5.1f} {tps:>9.3f} "
            f"{p_id:>12.3f} "
            f"{df['error'].mean():>+8.3f} "
            f"{np.sqrt(df['sq_err'].mean()):>8.3f} "
            f"{df['boot_se'].mean():>8.3f} "
            f"{df['covers'].mean() * 100:>9.1f}% "
            f"{df['ci_width'].mean():>10.3f}"
        )

    text = "\n".join(lines)
    print(text)
    with open(outfile, "w") as f:
        f.write(text + "\n")
    print(f"\nTable saved to {outfile}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    # ── True phi_star ─────────────────────────────────────────────────────
    print("Computing true phi_star values...")
    true_phi_stars = {}
    for c_val in sorted({sc["c"] for sc in SCENARIOS}):
        tps, _, _ = compute_true_phi_star(c_val)
        print(f"  c={c_val}: phi_star = {tps:.4f}")
        for sc in SCENARIOS:
            if sc["c"] == c_val:
                true_phi_stars[sc["name"]] = tps

    # ── Identification diagnostic plot (no simulation needed) ─────────────
    plot_identification(true_phi_stars)

    # ── MC + bootstrap for each scenario ──────────────────────────────────
    results = {}
    for sc in SCENARIOS:
        print(f"\n{'='*55}")
        print(f"Scenario: {sc['label']}  "
              f"(phi_0={sc['phi0']}, c={sc['c']}, "
              f"phi_star={true_phi_stars[sc['name']]:.3f})")
        print(f"  P(D=1) overall ≈ "
              f"{1.0 - sp_norm.cdf(sc['phi0'] / GAMMA):.2%}")
        print(f"  P(D=1|eta=phi_star) ≈ "
              f"{1.0 - sp_norm.cdf((sc['phi0'] - true_phi_stars[sc['name']]) / GAMMA):.2%}")
        print(f"{'='*55}")

        df = run_scenario(sc["phi0"], sc["c"],
                          true_phi_stars[sc["name"]], seed=42)
        results[sc["name"]] = df
        print(f"  Completed: {len(df)} valid reps")
        print(f"  RMSE      = {np.sqrt(df['sq_err'].mean()):.4f}")
        print(f"  Bias      = {df['error'].mean():+.4f}")
        print(f"  Boot SE   = {df['boot_se'].mean():.4f}")
        print(f"  Coverage  = {df['covers'].mean():.1%}")

    # ── Figures and table ─────────────────────────────────────────────────
    plot_sampling_distribution(results, true_phi_stars)
    plot_bootstrap(results, true_phi_stars)
    print_table(results, true_phi_stars,
                outfile=os.path.join(OUTDIR, "threshold_dist_table.txt"))


if __name__ == "__main__":
    main()
