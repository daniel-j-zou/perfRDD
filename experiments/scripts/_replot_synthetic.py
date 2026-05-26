"""Replot the synthetic study with a cleaner 2x2 layout (rows = scenarios,
cols = cost levels) from the existing synthetic_trim_overlap.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_JSON = ROOT / "runs" / "synthetic_trim_overlap.json"
OUT_FIG = ROOT / "runs" / "synthetic_trim_overlap_2x2.png"

COSTS_SHOWN = (0.0, 0.5)  # c=1.0 hits boundary on "good" -> not informative

j = json.loads(IN_JSON.read_text())

fig = plt.figure(figsize=(15, 10))
gs_outer = fig.add_gridspec(
    2, 2, hspace=0.45, wspace=0.20,
    left=0.06, right=0.99, top=0.88, bottom=0.06,
)
fig.suptitle(
    r"Synthetic DGP: trimmed estimator under controlled overlap (n=5000, 25 seeds)",
    fontsize=15, fontweight="bold", y=0.97,
)
fig.text(
    0.5, 0.925,
    r"solid markers: median trim $\phi^*(\epsilon)$ over MC seeds with IQR shaded;"
    r"  red dashed: median std $\phi^*$;  green dotted: TRUE $\phi^*$",
    ha="center", fontsize=10,
)

scenarios = [("good", r"good overlap:  $\eta\sim U(-1,1)$, propensity $\in[0.16,0.84]$"),
             ("bad",  r"bad overlap:  $\eta\sim N(0,4)$, propensity hits $0$/$1$ in tails")]

for irow, (sc, sc_title) in enumerate(scenarios):
    r = j["results"][sc]
    truth = j["true_phi_star"][sc]
    eps_grid = np.array(r["eps_grid"])
    for icol, c in enumerate(COSTS_SHOWN):
        outer_cell = gs_outer[irow, icol]
        inner = outer_cell.subgridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        ax_sec = fig.add_subplot(inner[1], sharex=ax)

        meds = np.array([np.median(r["trimmed_phi_star"][str(eps)][str(c)]) for eps in eps_grid])
        q25 = np.array([np.percentile(r["trimmed_phi_star"][str(eps)][str(c)], 25) for eps in eps_grid])
        q75 = np.array([np.percentile(r["trimmed_phi_star"][str(eps)][str(c)], 75) for eps in eps_grid])
        ax.plot(eps_grid, meds, "o-", color="C0", lw=2.0, ms=7,
                label=fr"trimmed median $\phi^*(\epsilon)$")
        ax.fill_between(eps_grid, q25, q75, color="C0", alpha=0.22, label="trimmed IQR")

        std_med = float(np.median(r["standard_phi_star"][str(c)]))
        ax.axhline(std_med, color="C3", ls="--", lw=2.0,
                   label=fr"std median = {std_med:.3f}")

        ax.axhline(float(truth[str(c)]), color="C2", ls=":", lw=2.0,
                   label=fr"TRUE $\phi^*$ = {float(truth[str(c)]):.3f}")

        ax.set_xscale("log")
        ax.set_ylabel(r"$\phi^*$", fontsize=11)
        ax.set_title(f"{sc_title}    |    $c={c}$", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
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

fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_FIG}")
