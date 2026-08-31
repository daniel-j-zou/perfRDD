"""Evaluate a prespecified GPA welfare menu with exact hard trimming.

This script reports all 16 welfare outcomes from
``experiments.datasets.gpa.welfare``. It never selects a valuation because it
produces a desired threshold. Each outcome receives:

* four full-sample ridge specifications on the primary policy grid;
* a five-fold unregularized cross-fit robustness estimate; and
* an expanded-grid audit at four illustrative direct probation costs.

The direct costs are sensitivity values in GPA-equivalent units, not estimated
administrative or student burdens. Outputs are written to the ignored directory
``experiments/runs/gpa_welfare_hard_trim``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.datasets.gpa.welfare import load_welfare_menu
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "gpa_welfare_hard_trim"
OUT_JSON = OUT_ROOT / "summary.json"
EPS = 0.1
NUISANCE_SUPPORT = (-2.0, 0.0)
PRIMARY_PHI_GRID = np.linspace(-0.6, 0.6, 241)
EXPANDED_PHI_GRID = np.linspace(-1.2, 1.2, 481)
RIDGE_GRID = (0.0, 0.0001, 0.001, 0.01)
DIRECT_COST_GRID = (0.0, 0.025, 0.05, 0.10)
PRIMARY_SPECIFICATION = "full_ridge_0p001"


def _ridge_label(value: float) -> str:
    return f"ridge_{value:g}".replace(".", "p")


def _run_primary_specifications(sample, key: str) -> Dict[str, Dict[str, Any]]:
    specifications: Dict[str, Dict[str, Any]] = {}
    for ridge in RIDGE_GRID:
        label = f"full_{_ridge_label(ridge)}"
        print(f"[run] {key}: {label}")
        specifications[label] = perfrdd_hard_trim(
            sample,
            OUT_ROOT / key / label,
            NUISANCE_SUPPORT,
            eps=EPS,
            c_values=(0.0,),
            phi_grid=PRIMARY_PHI_GRID,
            max_n=None,
            ridge_scale=ridge,
            crossfit_folds=1,
        )
    label = "crossfit_5fold_ridge_0"
    print(f"[run] {key}: {label}")
    specifications[label] = perfrdd_hard_trim(
        sample,
        OUT_ROOT / key / label,
        NUISANCE_SUPPORT,
        eps=EPS,
        c_values=(0.0,),
        phi_grid=PRIMARY_PHI_GRID,
        max_n=None,
        ridge_scale=0.0,
        crossfit_folds=5,
    )
    return specifications


def _run_expanded_cost_audit(sample, key: str) -> Dict[str, Any]:
    print(f"[audit] {key}: expanded grid and direct-cost sensitivity")
    return perfrdd_hard_trim(
        sample,
        OUT_ROOT / key / "expanded_cost_audit",
        NUISANCE_SUPPORT,
        eps=EPS,
        c_values=DIRECT_COST_GRID,
        phi_grid=EXPANDED_PHI_GRID,
        max_n=None,
        ridge_scale=0.001,
        crossfit_folds=1,
    )


def _plot_summary(results: Dict[str, Dict[str, Any]]) -> None:
    labels = [item["metadata"]["welfare_label"] for item in results.values()]
    categories = [item["metadata"]["welfare_category"] for item in results.values()]
    full = np.array([
        item["specifications"][PRIMARY_SPECIFICATION]["avg_alpha_hard_weighted"]
        for item in results.values()
    ])
    crossfit = np.array([
        item["specifications"]["crossfit_5fold_ridge_0"]["avg_alpha_hard_weighted"]
        for item in results.values()
    ])
    phi = np.array([
        item["specifications"][PRIMARY_SPECIFICATION]["phi_star"]["0.0"]
        for item in results.values()
    ])
    colors = {
        "direct": "C0",
        "missing_gpa_sensitivity": "C1",
        "status_adjusted_stress": "C2",
    }
    point_colors = [colors[category] for category in categories]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].axvline(0.0, color="black", lw=0.7)
    axes[0].scatter(full, y, c=point_colors, marker="o", label="Full, ridge 0.001")
    axes[0].scatter(crossfit, y, c=point_colors, marker="x", label="5-fold cross-fit")
    for j in range(len(y)):
        axes[0].plot([full[j], crossfit[j]], [j, j], color="0.7", lw=0.8, zorder=0)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"Hard-window average treatment effect $\hat\alpha$")
    axes[0].set_title("Welfare effects: full sample vs cross-fit")
    axes[0].legend(loc="lower right")

    axes[1].axvline(0.0, color="black", lw=0.7)
    axes[1].scatter(phi, y, c=point_colors)
    axes[1].axvline(PRIMARY_PHI_GRID[0], color="0.4", ls="--", lw=0.8)
    axes[1].axvline(PRIMARY_PHI_GRID[-1], color="0.4", ls="--", lw=0.8)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"No-cost policy threshold $\hat\phi^*$")
    axes[1].set_title("All no-cost optima are policy-grid boundaries")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "welfare_menu_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> Dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    samples = load_welfare_menu()
    results: Dict[str, Dict[str, Any]] = {}
    for key, sample in samples.items():
        print(f"[sample] {key}: n={sample.n:,}")
        results[key] = {
            "metadata": sample.extras,
            "specifications": _run_primary_specifications(sample, key),
            "expanded_cost_audit": _run_expanded_cost_audit(sample, key),
        }

    _plot_summary(results)
    primary_boundary_count = sum(
        item["specifications"][PRIMARY_SPECIFICATION][
            "phi_star_at_grid_boundary"
        ]["0.0"]
        for item in results.values()
    )
    expanded_interior_count = sum(
        not boundary
        for item in results.values()
        for boundary in item["expanded_cost_audit"]["phi_star_at_grid_boundary"].values()
    )
    payload: Dict[str, Any] = {
        "description": "Prespecified 16-outcome GPA welfare menu under exact hard trimming",
        "confirmatory": False,
        "eps": EPS,
        "nuisance_support": list(NUISANCE_SUPPORT),
        "primary_phi_grid": [
            float(PRIMARY_PHI_GRID[0]),
            float(PRIMARY_PHI_GRID[-1]),
            int(len(PRIMARY_PHI_GRID)),
        ],
        "expanded_phi_grid": [
            float(EXPANDED_PHI_GRID[0]),
            float(EXPANDED_PHI_GRID[-1]),
            int(len(EXPANDED_PHI_GRID)),
        ],
        "ridge_grid": list(RIDGE_GRID),
        "direct_cost_grid": list(DIRECT_COST_GRID),
        "direct_cost_note": (
            "Illustrative GPA-equivalent stress values, not estimated probation costs."
        ),
        "primary_specification": PRIMARY_SPECIFICATION,
        "n_welfare_outcomes": len(results),
        "primary_boundary_count": int(primary_boundary_count),
        "expanded_cost_interior_count": int(expanded_interior_count),
        "inference_available": False,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[wrote] {OUT_JSON}")
    return payload


if __name__ == "__main__":
    main()
