"""Sensitivity sweep over the trimming parameter ε for the trimmed perfrdd
estimator on the four linear-Q datasets.

For each dataset:
  1. Run the standard `perfrdd` estimator once (reference; ε-independent).
  2. Run `perfrdd_trim` at each ε in a grid.
  3. Force both estimators to use the SAME cost grid (taken from the standard
     fit) so φ*(c) is directly comparable across ε and against standard.

Outputs:
  experiments/runs/four_datasets_trim_eps_sweep.json    — full sweep summary
  experiments/runs/four_datasets_trim_eps_sweep.png     — 2×2 sensitivity figure

The figure shows, per dataset:
  - φ*(ε) for each cost c, with markers/colors
  - standard estimator's φ*(c) as horizontal dashed reference lines (same colors)
  - the operating threshold φ_0 as a thin black dotted line
  - the default ε = 0.1 marked with a vertical grey line
A small secondary panel shows the fraction of data retained vs ε.
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
from experiments.methods.perfrdd import perfrdd, _auto_c_values, _reduce_to_primary_axis, _subsample, DEFAULT_MAX_N
from experiments.methods.perfrdd_trim import perfrdd_trim


ROOT = Path(__file__).resolve().parent.parent
RUNS_STD = ROOT / "runs" / "perfrdd"
RUNS_TRIM_SWEEP = ROOT / "runs" / "perfrdd_trim_sweep"
OUT_FIG = ROOT / "runs" / "four_datasets_trim_eps_sweep.png"
OUT_JSON = ROOT / "runs" / "four_datasets_trim_eps_sweep.json"

DATASETS = [
    ("gpa", "GPA — academic probation"),
    ("nhanes", "NHANES — HbA1c diabetic cutoff"),
    ("oulad", "OULAD — first-TMA pass mark"),
    ("lending_club", "Lending Club — DTI trigger"),
]

# ε grid: log-spaced from very loose trimming (0.02) to aggressive (0.4).
EPS_GRID = [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]
DEFAULT_EPS = 0.10


def _run_sweep_for(name: str) -> Dict[str, Any]:
    """Run standard once + trimmed at each ε, sharing a cost grid."""
    sample = load(name)

    # Run standard estimator first to fix the cost grid for both methods.
    std_out = RUNS_STD / name
    res_std = perfrdd(sample, std_out)
    cost_grid = tuple(float(c) for c in res_std["phi_star"].keys())
    # Use a shared phi_grid spanning ±3*std(Q) so the argmax search is identical.
    Q, _, threshold = _reduce_to_primary_axis(sample)
    Y = sample.Y
    keep = np.isfinite(Q) & np.isfinite(Y)
    Q = Q[keep]
    if len(Q) > DEFAULT_MAX_N:
        (Q,), _, _ = _subsample([Q], DEFAULT_MAX_N)
    phi_span = 3.0 * float(Q.std())
    phi_grid = np.linspace(threshold - phi_span, threshold + phi_span, 400)

    eps_results: List[Dict[str, Any]] = []
    for eps in EPS_GRID:
        out_dir = RUNS_TRIM_SWEEP / name / f"eps_{eps:.3f}"
        try:
            res = perfrdd_trim(
                sample, out_dir, eps=eps,
                c_values=cost_grid, phi_grid=phi_grid,
            )
        except Exception as e:  # too-small windows on small datasets
            print(f"  ε={eps:.3f}: FAILED ({e!r})")
            res = {"eps": eps, "error": repr(e)}
        eps_results.append(res)

    return {
        "name": name,
        "threshold": float(res_std["threshold_actual"]),
        "cost_grid": list(cost_grid),
        "standard": res_std,
        "eps_grid": list(EPS_GRID),
        "sweep": eps_results,
    }


def _plot_panel(ax_main, ax_sec, r: Dict[str, Any], title: str) -> None:
    """One dataset panel: main = φ*(ε), secondary thin strip = retention %.

    Solid lines with markers: trimmed φ*(ε), color-coded by cost index c_k.
    Dashed horizontal lines: standard φ* at the same costs (same colors).
    Black dotted: operating threshold φ_0.
    Vertical grey: ε = 0.1 (default).
    """
    eps_grid = np.array(r["eps_grid"])
    costs = r["cost_grid"]
    phi0 = r["threshold"]
    phi_std_by_c = {float(c): v for c, v in r["standard"]["phi_star"].items()}

    sweep = r["sweep"]
    valid = [(eps_grid[i], s) for i, s in enumerate(sweep) if "error" not in s]
    if not valid:
        ax_main.text(0.5, 0.5, "all ε values failed", ha="center", va="center")
        ax_main.set_title(title, fontsize=10)
        return

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.85, len(costs)))

    handles = []
    for k, (c, col) in enumerate(zip(costs, colors)):
        phi_trim = np.array([s["phi_star"][str(c)] for _, s in valid])
        eps_x = np.array([e for e, _ in valid])
        line, = ax_main.plot(eps_x, phi_trim, "o-", color=col, lw=2.0, ms=6,
                             label=fr"$c_{k}={c:.3g}$")
        ax_main.axhline(phi_std_by_c[c], color=col, ls="--", lw=1.4, alpha=0.65)
        handles.append(line)

    ax_main.axhline(phi0, color="black", ls=":", lw=1.2,
                    label=fr"$\phi_0={phi0:.3g}$")
    ax_main.axvline(DEFAULT_EPS, color="grey", ls="-", lw=0.7, alpha=0.45)
    ax_main.set_xscale("log")
    ax_main.set_ylabel(r"$\phi^*$", fontsize=11)
    ax_main.set_title(title, fontsize=12, fontweight="bold")
    ax_main.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    ax_main.tick_params(labelsize=9)
    ax_main.grid(alpha=0.25)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Thin retention strip.
    eps_x = np.array([e for e, _ in valid])
    retention = np.array([s["n_in_window"] / s["n_used"] for _, s in valid])
    ax_sec.plot(eps_x, retention, "k^-", lw=1.5, ms=5)
    ax_sec.axvline(DEFAULT_EPS, color="grey", ls="-", lw=0.7, alpha=0.45)
    ax_sec.set_xscale("log")
    ax_sec.set_xlabel(r"$\epsilon$", fontsize=11)
    ax_sec.set_ylabel("retained", fontsize=9)
    ax_sec.set_ylim(-0.02, 1.02)
    ax_sec.tick_params(labelsize=9)
    ax_sec.grid(alpha=0.25)


def main() -> None:
    RUNS_TRIM_SWEEP.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for name, title in DATASETS:
        print(f"[sweep] {name}")
        results[name] = _run_sweep_for(name)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    # 2×2 grid for the four datasets; each "cell" is two stacked axes
    # (main φ*(ε) + retention strip).
    fig = plt.figure(figsize=(17, 12))
    gs_outer = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22,
                                left=0.06, right=0.98, top=0.91, bottom=0.05)
    fig.suptitle(
        "Sensitivity to trimming parameter ε — four linear-Q datasets",
        fontsize=15, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.945,
        r"solid markers: trimmed $\phi^*(\epsilon)$ at increasing cost $c_0<c_1<c_2<c_3$;  "
        r"dashed: standard $\phi^*$ at matching cost;  "
        r"vertical grey: default $\epsilon=0.10$;  black dotted: operating $\phi_0$",
        ha="center", fontsize=10,
    )

    for k, (name, title) in enumerate(DATASETS):
        i, j = k // 2, k % 2
        inner = gs_outer[i, j].subgridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.08)
        ax_main = fig.add_subplot(inner[0])
        ax_sec = fig.add_subplot(inner[1], sharex=ax_main)
        _plot_panel(ax_main, ax_sec, results[name], title)

    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
