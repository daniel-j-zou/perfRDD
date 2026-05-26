"""Finer ε sweep starting from ε = 0.2 and going DOWN, to nail down the
convergence rate of the trimmed estimator to the standard estimator as
ε → 0 on the three datasets where the convergence is visible (GPA,
NHANES, Lending Club). OULAD is omitted because its overlap window is
structurally tiny — see the synthetic study for that case.

Output:
  experiments/runs/three_datasets_trim_eps_fine.png
  experiments/runs/three_datasets_trim_eps_fine.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.registry import load
from experiments.methods.perfrdd import (
    perfrdd, _reduce_to_primary_axis, _subsample, DEFAULT_MAX_N,
)
from experiments.methods.perfrdd_trim import perfrdd_trim


ROOT = Path(__file__).resolve().parent.parent
RUNS_STD = ROOT / "runs" / "perfrdd"
OUT_FIG = ROOT / "runs" / "three_datasets_trim_eps_fine.png"
OUT_JSON = ROOT / "runs" / "three_datasets_trim_eps_fine.json"

DATASETS = [
    ("gpa", "GPA — academic probation"),
    ("nhanes", "NHANES — HbA1c diabetic cutoff"),
    ("lending_club", "Lending Club — DTI trigger"),
]

# ε grid: from 0.2 down to 0.005, weighted toward small values.
EPS_GRID = [0.20, 0.15, 0.10, 0.075, 0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005]


def _sweep(name: str) -> Dict[str, Any]:
    sample = load(name)
    std_out = RUNS_STD / name
    res_std = perfrdd(sample, std_out)
    cost_grid = tuple(float(c) for c in res_std["phi_star"].keys())

    Q, _, threshold = _reduce_to_primary_axis(sample)
    Y = sample.Y
    keep = np.isfinite(Q) & np.isfinite(Y)
    Q = Q[keep]
    if len(Q) > DEFAULT_MAX_N:
        (Q,), _, _ = _subsample([Q], DEFAULT_MAX_N)
    phi_span = 3.0 * float(Q.std())
    phi_grid = np.linspace(threshold - phi_span, threshold + phi_span, 600)

    results: List[Dict[str, Any]] = []
    for eps in EPS_GRID:
        out_dir = ROOT / "runs" / "perfrdd_trim_fine" / name / f"eps_{eps:.4f}"
        try:
            r = perfrdd_trim(sample, out_dir, eps=eps,
                             c_values=cost_grid, phi_grid=phi_grid)
        except Exception as e:
            print(f"  ε={eps}: FAILED {e!r}")
            r = {"eps": eps, "error": repr(e)}
        results.append(r)
    return {
        "name": name,
        "threshold": float(res_std["threshold_actual"]),
        "cost_grid": list(cost_grid),
        "standard": res_std,
        "eps_grid": list(EPS_GRID),
        "sweep": results,
    }


def _plot_panel(ax_main, ax_sec, r: Dict[str, Any], title: str) -> None:
    eps_grid = np.array(r["eps_grid"])
    costs = r["cost_grid"]
    phi0 = r["threshold"]
    phi_std_by_c = {float(c): v for c, v in r["standard"]["phi_star"].items()}
    valid = [(eps_grid[i], s) for i, s in enumerate(r["sweep"]) if "error" not in s]
    if not valid:
        ax_main.text(0.5, 0.5, "all failed", ha="center", va="center")
        return

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.85, len(costs)))

    for k, (c, col) in enumerate(zip(costs, colors)):
        phi_trim = np.array([s["phi_star"][str(c)] for _, s in valid])
        eps_x = np.array([e for e, _ in valid])
        ax_main.plot(eps_x, phi_trim, "o-", color=col, lw=2.0, ms=6,
                     label=fr"$c_{k}={c:.3g}$")
        ax_main.axhline(phi_std_by_c[c], color=col, ls="--", lw=1.4, alpha=0.65)

    ax_main.axhline(phi0, color="black", ls=":", lw=1.2, label=fr"$\phi_0={phi0:.3g}$")
    ax_main.axvline(0.10, color="grey", ls="-", lw=0.7, alpha=0.45)
    ax_main.set_xscale("log")
    ax_main.set_ylabel(r"$\phi^*$", fontsize=11)
    ax_main.set_title(title, fontsize=12, fontweight="bold")
    ax_main.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    ax_main.tick_params(labelsize=9)
    ax_main.grid(alpha=0.25)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Annotate the difference at the smallest ε (the "convergence quality").
    smallest_eps, smallest_s = valid[-1]  # last entry (sorted desc -> asc by index)
    c0 = costs[0]
    diff = smallest_s["phi_star"][str(c0)] - phi_std_by_c[c0]
    ax_main.annotate(
        fr"at $\epsilon={smallest_eps:g}$:  $\phi^*_\epsilon - \phi^*_{{\rm std}}$ = {diff:+.3f}",
        xy=(smallest_eps, smallest_s["phi_star"][str(c0)]),
        xytext=(0.05, 0.05), textcoords="axes fraction",
        fontsize=9, color="darkred",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkred", alpha=0.85),
    )

    # Retention strip.
    eps_x = np.array([e for e, _ in valid])
    retention = np.array([s["n_in_window"] / s["n_used"] for _, s in valid])
    ax_sec.plot(eps_x, retention, "k^-", lw=1.5, ms=5)
    ax_sec.axvline(0.10, color="grey", ls="-", lw=0.7, alpha=0.45)
    ax_sec.set_xscale("log")
    ax_sec.set_xlabel(r"$\epsilon$", fontsize=11)
    ax_sec.set_ylabel("retained", fontsize=9)
    ax_sec.set_ylim(-0.02, 1.02)
    ax_sec.tick_params(labelsize=9)
    ax_sec.grid(alpha=0.25)


def main() -> None:
    results: Dict[str, Any] = {}
    for name, _ in DATASETS:
        print(f"[fine] {name}")
        results[name] = _sweep(name)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    fig = plt.figure(figsize=(20, 6.5))
    gs_outer = fig.add_gridspec(1, 3, hspace=0.32, wspace=0.20,
                                left=0.05, right=0.99, top=0.88, bottom=0.10)
    fig.suptitle(
        r"Convergence of trimmed estimator as $\epsilon \to 0$ — three linear-Q datasets",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.93,
        r"solid markers: trimmed $\phi^*(\epsilon)$; dashed: standard $\phi^*$ at matching cost;"
        r"  grey vertical: default $\epsilon=0.10$;  black dotted: operating $\phi_0$",
        ha="center", fontsize=10,
    )

    for k, (name, title) in enumerate(DATASETS):
        inner = gs_outer[0, k].subgridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.08)
        ax_main = fig.add_subplot(inner[0])
        ax_sec = fig.add_subplot(inner[1], sharex=ax_main)
        _plot_panel(ax_main, ax_sec, results[name], title)

    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
