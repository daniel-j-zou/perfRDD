"""Synthetic-DGP study of the trimmed estimator under controlled overlap.

Two scenarios, identical except for the η distribution:
  * "good":  η ~ Uniform(-1, 1)  → e_phi(η) bounded in [≈0.16, ≈0.84] globally.
  * "bad":   η ~ N(0, 4)         → e_phi(η) → 0 at η<<0 and → 1 at η>>0
                                    (OULAD-style structural near-determinism).

In both, X ~ N(0, I_5), γ = (1, …, 1)/√5 so γ^T X ~ N(0, 1). Treatment is
D = 1{γ^T X + η > φ_0 = 0}. The outcome model is
    Y = α(η) D + b(η) + β^T X + ε
with α(η) = η (centered, linear), b(η) = 0.5, β = 0.2·1, ε ~ N(0, 0.5).
Centered α ensures all three cost levels c ∈ {0, 0.5, 1.0} give an interior
optimum φ* (instead of hitting the boundary "treat everyone" / "treat no one").

For each scenario:
  * Repeat over `n_seeds` Monte Carlo replicates.
  * Run perfrdd_trim at a grid of ε values plus the standard perfrdd as
    reference.
  * Plot the resulting φ*(ε) distribution (median + IQR band) on a single
    figure with the two scenarios side-by-side.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.sample import RDDSample
from experiments.methods.perfrdd import perfrdd
from experiments.methods.perfrdd_trim import perfrdd_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "synthetic_trim_overlap"
OUT_FIG = ROOT / "runs" / "synthetic_trim_overlap.png"
OUT_JSON = ROOT / "runs" / "synthetic_trim_overlap.json"


# -------- DGP ---------------------------------------------------------------

def _gen_sample(seed: int, scenario: str, n: int = 5000, d_X: int = 5) -> RDDSample:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_X))
    gamma = np.ones(d_X) / np.sqrt(d_X)         # so γ^T X ~ N(0, 1)
    T = X @ gamma
    if scenario == "good":
        eta = rng.uniform(-1.0, 1.0, size=n)
    elif scenario == "bad":
        eta = rng.normal(0.0, 4.0, size=n)
    else:
        raise ValueError(scenario)
    Q = T + eta
    phi_0 = 0.0
    D = (Q > phi_0).astype(int)
    alpha = eta.astype(float)                 # centered, slope 1
    b = np.full_like(eta, 0.5)
    beta = np.full(d_X, 0.2)
    eps_noise = rng.normal(0.0, 0.5, size=n)
    Y = alpha * D + b + X @ beta + eps_noise
    return RDDSample(
        Q=Q.astype(float), X=X, Y=Y.astype(float),
        threshold=phi_0,
        name=f"synth_{scenario}_seed{seed}",
        feature_names=[f"x{i}" for i in range(d_X)],
        description=f"synthetic ({scenario} overlap)",
    )


def _true_phi_star_grid(scenario: str, n_big: int = 200_000,
                         cost_grid: tuple[float, ...] = (0.0, 0.5, 1.0)) -> Dict[float, float]:
    """Approximate the true φ*(c) by MC on a much larger sample using the
    population integrand E[(α(η)-c)·G_bar(φ-η)]."""
    sample = _gen_sample(seed=999_999, scenario=scenario, n=n_big)
    Q, _, threshold = sample.Q, sample.X, sample.threshold
    # eta = Q - X@gamma actually unknown; we know the true gamma here.
    rng = np.random.default_rng(0)
    # Recompute eta directly from DGP.
    if scenario == "good":
        eta = rng.uniform(-1, 1, n_big)
    else:
        eta = rng.normal(0, 4, n_big)
    # T independent of eta with T ~ N(0, 1). Bar-G(t) = P(T > t).
    from scipy.stats import norm
    # Wider grid for the bad scenario where η has heavier tails.
    span = 4.0 if scenario == "good" else 8.0
    phi_grid = np.linspace(threshold - span, threshold + span, 800)
    out: Dict[float, float] = {}
    for c in cost_grid:
        u = np.empty_like(phi_grid)
        for j, phi in enumerate(phi_grid):
            bar_G = 1.0 - norm.cdf(phi - eta)
            alpha = eta.astype(float)
            u[j] = float(np.mean((alpha - c) * bar_G))
        out[c] = float(phi_grid[int(np.argmax(u))])
    return out


# -------- run experiment ----------------------------------------------------

EPS_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
N_SEEDS = 25
N = 10000
PHI_GRID_GOOD = np.linspace(-4.0, 4.0, 400)
PHI_GRID_BAD = np.linspace(-8.0, 8.0, 600)
COSTS = (0.0, 0.5, 1.0)


def _run_one(scenario: str) -> Dict[str, Any]:
    print(f"  scenario={scenario}: {N_SEEDS} seeds × ({len(EPS_GRID)} ε + 1 std) per seed")
    phi_grid = PHI_GRID_GOOD if scenario == "good" else PHI_GRID_BAD
    std_phi: Dict[float, List[float]] = {c: [] for c in COSTS}
    trim_phi: Dict[float, Dict[float, List[float]]] = {
        eps: {c: [] for c in COSTS} for eps in EPS_GRID
    }
    retention: Dict[float, List[float]] = {eps: [] for eps in EPS_GRID}

    for seed in range(N_SEEDS):
        sample = _gen_sample(seed, scenario, n=N)
        # Standard.
        out = OUT_DIR / scenario / "_runs"
        try:
            r_std = perfrdd(sample, out, c_values=COSTS, phi_grid=phi_grid, max_n=None)
            for c in COSTS:
                std_phi[c].append(float(r_std["phi_star"][str(c)]))
        except Exception as e:
            print(f"    seed={seed} std FAILED: {e!r}")

        for eps in EPS_GRID:
            try:
                r = perfrdd_trim(sample, out, eps=eps, c_values=COSTS,
                                  phi_grid=phi_grid, max_n=None)
                for c in COSTS:
                    trim_phi[eps][c].append(float(r["phi_star"][str(c)]))
                retention[eps].append(r["n_in_window"] / r["n_used"])
            except Exception as e:
                print(f"    seed={seed} ε={eps}: FAILED {e!r}")

    return {
        "scenario": scenario,
        "n_seeds": N_SEEDS,
        "n_per_seed": N,
        "eps_grid": EPS_GRID,
        "costs": list(COSTS),
        "standard_phi_star": {str(c): std_phi[c] for c in COSTS},
        "trimmed_phi_star": {
            str(eps): {str(c): trim_phi[eps][c] for c in COSTS}
            for eps in EPS_GRID
        },
        "retention": {str(eps): retention[eps] for eps in EPS_GRID},
    }


def _plot_scenario(ax, ax_sec, r: Dict[str, Any], truth: Dict[float, float],
                   title: str) -> None:
    eps_grid = np.array(r["eps_grid"])
    costs = r["costs"]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.85, len(costs)))

    for c, col in zip(costs, colors):
        median = np.array([np.median(r["trimmed_phi_star"][str(eps)][str(c)]) for eps in eps_grid])
        q25 = np.array([np.percentile(r["trimmed_phi_star"][str(eps)][str(c)], 25) for eps in eps_grid])
        q75 = np.array([np.percentile(r["trimmed_phi_star"][str(eps)][str(c)], 75) for eps in eps_grid])
        ax.plot(eps_grid, median, "o-", color=col, lw=2.0, ms=6,
                label=fr"trim, $c={c}$")
        ax.fill_between(eps_grid, q25, q75, color=col, alpha=0.18)
        # Standard horizontal reference (median of MC standard).
        std_med = float(np.median(r["standard_phi_star"][str(c)]))
        ax.axhline(std_med, color=col, ls="--", lw=1.2, alpha=0.7)
        # True φ*(c).
        ax.axhline(truth[c], color=col, ls=":", lw=1.0, alpha=0.55)

    ax.axhline(0.0, color="black", ls="-", lw=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylabel(r"$\phi^*$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25)
    plt.setp(ax.get_xticklabels(), visible=False)

    median_ret = np.array([np.median(r["retention"][str(eps)]) for eps in eps_grid])
    ax_sec.plot(eps_grid, median_ret, "k^-", lw=1.5, ms=5)
    ax_sec.set_xscale("log")
    ax_sec.set_xlabel(r"$\epsilon$", fontsize=11)
    ax_sec.set_ylabel("retained", fontsize=9)
    ax_sec.set_ylim(-0.02, 1.02)
    ax_sec.tick_params(labelsize=9)
    ax_sec.grid(alpha=0.25)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    truths: Dict[str, Dict[float, float]] = {}
    for scenario in ("good", "bad"):
        print(f"[synth] scenario={scenario}")
        truths[scenario] = _true_phi_star_grid(scenario, n_big=200_000, cost_grid=COSTS)
        print(f"  true φ*: {truths[scenario]}")
        results[scenario] = _run_one(scenario)

    OUT_JSON.write_text(json.dumps({
        "results": results,
        "true_phi_star": {sc: {str(c): v for c, v in t.items()} for sc, t in truths.items()},
    }, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    fig = plt.figure(figsize=(16, 7.5))
    gs_outer = fig.add_gridspec(1, 2, wspace=0.22, left=0.05, right=0.99,
                                top=0.86, bottom=0.10)
    fig.suptitle(
        r"Synthetic DGP: trimmed estimator under controlled overlap (n=5000, "
        f"{N_SEEDS} seeds)",
        fontsize=15, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.91,
        r"solid markers: median trim $\phi^*(\epsilon)$ over MC seeds with IQR shaded;"
        r"  dashed: median std $\phi^*$;  dotted: TRUE $\phi^*$",
        ha="center", fontsize=10,
    )

    titles = {
        "good": r"good overlap:  $\eta\sim U(-1,1)$, propensity $\in[0.16,0.84]$",
        "bad":  r"bad overlap:  $\eta\sim N(0,4)$, propensity hits $0$/$1$ in tails",
    }
    for k, scenario in enumerate(("good", "bad")):
        inner = gs_outer[0, k].subgridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.08)
        ax_main = fig.add_subplot(inner[0])
        ax_sec = fig.add_subplot(inner[1], sharex=ax_main)
        _plot_scenario(ax_main, ax_sec, results[scenario],
                       truths[scenario], titles[scenario])

    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
