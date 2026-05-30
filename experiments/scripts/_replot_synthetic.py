"""Replot the synthetic study with two TRUE phi* curves per panel:

  - the STANDARD estimand (unconditional optimum; constant in eps),
  - the TRIMMED estimand phi*_eps (varies with eps as the indicator support
    shrinks).

This makes the comparison honest: each estimator should track ITS OWN true
target, and any gap between estimator-median and truth-line is the bias of
that estimator relative to its own estimand.

Reads the per-seed MC results from synthetic_trim_overlap.json and writes
synthetic_trim_overlap_2x2.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent.parent
IN_JSON = ROOT / "runs" / "synthetic_trim_overlap.json"
OUT_FIG = ROOT / "runs" / "synthetic_trim_overlap_2x2.png"

COSTS_SHOWN = (0.0, 0.5)
PHI_0 = 0.0


def _true_phi_star(scenario: str, eps: float | None, c: float,
                   n_big: int = 200_000, seed: int = 42) -> float:
    """Compute true phi* by 200k-MC integration.

    eps=None   -> standard estimand (no trimming, indicator = 1 always)
    eps>0      -> trimmed estimand phi*_eps with window
                  [phi_0 - Q_{1-eps}(T), phi_0 - Q_eps(T)],  T ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    if scenario == "good":
        eta = rng.uniform(-1.0, 1.0, n_big)
        span = 4.0
    else:
        eta = rng.normal(0.0, 4.0, n_big)
        span = 8.0
    alpha = eta.astype(float)               # DGP: alpha(eta) = eta
    if eps is None:
        weight = np.ones_like(eta)
    else:
        l0 = PHI_0 - norm.ppf(1.0 - eps)
        u0 = PHI_0 - norm.ppf(eps)
        weight = ((eta >= l0) & (eta <= u0)).astype(float)
    phi_grid = np.linspace(PHI_0 - span, PHI_0 + span, 800)
    u_curve = np.empty_like(phi_grid)
    for j, phi in enumerate(phi_grid):
        bar_G = 1.0 - norm.cdf(phi - eta)
        u_curve[j] = float(np.mean((alpha - c) * bar_G * weight))
    return float(phi_grid[int(np.argmax(u_curve))])


def main() -> None:
    j = json.loads(IN_JSON.read_text())
    scenarios = [
        ("good", r"good overlap:  $\eta\sim U(-1,1)$, propensity $\in[0.16,0.84]$"),
        ("bad",  r"bad overlap:  $\eta\sim N(0,4)$, propensity hits $0$/$1$ in tails"),
    ]

    fig = plt.figure(figsize=(15, 10))
    gs_outer = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.20,
                                left=0.06, right=0.99, top=0.86, bottom=0.06)
    fig.suptitle(
        r"Synthetic DGP: trim estimator vs its OWN true estimand (n=10000, 25 seeds)",
        fontsize=15, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.925,
        r"blue solid = trim estimator median over MC seeds, 95% band shaded.  "
        r"green dotted (constant) = TRUE std estimand $\phi^*$;  "
        r"green solid markers (varies with $\epsilon$) = TRUE trim estimand $\phi^*_\epsilon$.",
        ha="center", fontsize=9.5,
    )

    for irow, (sc, sc_title) in enumerate(scenarios):
        r = j["results"][sc]
        eps_grid = np.array(r["eps_grid"])
        for icol, c in enumerate(COSTS_SHOWN):
            outer = gs_outer[irow, icol]
            inner = outer.subgridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.08)
            ax = fig.add_subplot(inner[0])
            ax_sec = fig.add_subplot(inner[1], sharex=ax)

            meds = np.array([np.median(r["trimmed_phi_star"][str(e)][str(c)]) for e in eps_grid])
            q025 = np.array([np.percentile(r["trimmed_phi_star"][str(e)][str(c)], 2.5) for e in eps_grid])
            q975 = np.array([np.percentile(r["trimmed_phi_star"][str(e)][str(c)], 97.5) for e in eps_grid])
            ax.plot(eps_grid, meds, "o-", color="C0", lw=2.0, ms=7,
                    label=r"trim est. median")
            ax.fill_between(eps_grid, q025, q975, color="C0", alpha=0.22, label="trim est. 95%")

            # Truth lines: two of them, both green.
            true_std = _true_phi_star(sc, eps=None, c=c)
            ax.axhline(true_std, color="C2", ls=":", lw=2.0,
                       label=fr"TRUE std est. $\phi^*$ = {true_std:.3f}")

            true_trim = np.array([_true_phi_star(sc, eps=float(e), c=c) for e in eps_grid])
            ax.plot(eps_grid, true_trim, color="C2", ls="-", lw=2.0, marker="s", ms=4,
                    label=r"TRUE trim est. $\phi^*_\epsilon$")

            ax.set_xscale("log")
            ax.set_ylabel(r"$\phi^*$", fontsize=11)
            ax.set_title(f"{sc_title}    |    $c={c}$", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
            ax.tick_params(labelsize=9)
            ax.grid(alpha=0.25)
            plt.setp(ax.get_xticklabels(), visible=False)

            median_ret = np.array([np.median(r["retention"][str(e)]) for e in eps_grid])
            ax_sec.plot(eps_grid, median_ret, "k^-", lw=1.5, ms=5)
            ax_sec.set_xscale("log")
            ax_sec.set_xlabel(r"$\epsilon$", fontsize=11)
            ax_sec.set_ylabel("retained", fontsize=9)
            ax_sec.set_ylim(-0.02, 1.02)
            ax_sec.tick_params(labelsize=9)
            ax_sec.grid(alpha=0.25)

    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
