"""Figure for this_week.tex: variance of the threshold relative to the paper's estimator.

Left: homoskedastic arms, varying the outcome noise s.d. in the paper design; the
stacked first stage attains the bound there.  Right: heteroskedastic arms,
sigma_d^2(eta) = 0.25 exp(kappa eta), for the unweighted stacked first stage, a
weighted outcome fit with the OLS first stage, and both weighted (the bound).
All values are rotated N-scaled variances divided by the paper estimator's.

Run:
    python -m experiments.scripts.ds_local_efficiency_figure --out <png>
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.scripts.ds_local_efficiency import DESIGNS, Population
from experiments.scripts.ds_local_efficiency_hetero import hetero_variances

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # reference palette slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    base = DESIGNS["paper_baseline"]
    sig = np.round(np.arange(0.15, 1.001, 0.05), 3)
    left = []
    for s in sig:
        r = Population(replace(base, sigma_arm=(s, s))).decomposition()
        left.append(100 * r["V_rotated_eff"] / r["V_rotated_ols"])
    pop = Population(base)
    kap = np.round(np.arange(0.0, 1.501, 0.1), 3)
    rows = [hetero_variances(pop, k) for k in kap]
    st = [100 * r["stacked"] / r["paper"] for r in rows]
    wo = [100 * r["weighted_ols"] / r["paper"] for r in rows]
    bd = [100 * r["bound"] / r["paper"] for r in rows]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax in axes:
        ax.axhline(100, color=INK2, lw=1.2, ls=(0, (4, 3)))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax = axes[0]
    ax.plot(sig, left, color=BLUE, lw=2, marker="o", ms=4)
    ax.text(0.62, 84, "stacked first stage\n(= bound under H)", color=INK, fontsize=8)
    ax.annotate("paper's estimator", (sig[0], 100), xytext=(2, 3), textcoords="offset points",
                color=INK2, fontsize=8)
    ax.set_xlabel("outcome noise s.d., both arms")
    ax.set_ylabel("rotated variance, % of paper's")
    ax.set_title("Homoskedastic arms", fontsize=9, color=INK, loc="left")
    ax = axes[1]
    ax.plot(kap, st, color=BLUE, lw=2, marker="o", ms=4, label="stacked, unweighted")
    ax.plot(kap, wo, color=ORANGE, lw=2, ls=(0, (6, 2)), marker="s", ms=4,
            label="weighted outcome, OLS first stage")
    ax.plot(kap, bd, color=AQUA, lw=2, ls=(0, (2, 1.5)), marker="^", ms=4,
            label="both weighted (= bound)")
    for y, txt, dy in ((st[-1], "stacked, unweighted", -4), (wo[-1], "weighted outcome,\nOLS first stage", -6),
                       (bd[-1], "both weighted\n(= bound)", -10)):
        ax.annotate(txt, (kap[-1], y), xytext=(4, dy), textcoords="offset points", color=INK, fontsize=8,
                    va="center")
    ax.set_xlim(-0.05, 2.05)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel(r"heteroskedasticity $\kappa$, $\sigma_d^2(\eta)=0.25\,e^{\kappa\eta}$")
    ax.set_title("Heteroskedastic arms", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"[wrote] {args.out}")
    print("left:", dict(zip(sig.tolist(), np.round(left, 1).tolist())))
    print("right stacked:", np.round(st, 1).tolist())
    print("right w_ols:", np.round(wo, 1).tolist())
    print("right bound:", np.round(bd, 1).tolist())


if __name__ == "__main__":
    main()
