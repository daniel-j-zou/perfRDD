"""Exploratory exact-hard-trim runs on every available one-dimensional dataset.

The nuisance intervals below were rounded outward from the support diagnostics
in the earlier smooth-gate exploration.  They are therefore pilot-derived,
not confirmatory choices.  This script records that provenance and uses the
same locked interval for every ridge and cross-fitting specification.

For each dataset the script runs:

* full-sample point estimation over a ridge sensitivity grid; and
* a five-fold, unregularized robustness estimate.

The cost grid is held fixed within a dataset.  It is calibrated once from the
unregularized full-sample hard-window average treatment effect, so ridge
comparisons cannot change merely because their costs were rescaled.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from experiments._core.registry import list_datasets, load
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "hard_trim_existing"
OUT_JSON = OUT_ROOT / "summary.json"
RIDGE_GRID = (0.0, 0.0001, 0.001, 0.01)
COST_RATIOS = (0.0, 0.5, 1.0, 1.5)

# Pilot-derived from the August 2026 smooth-support diagnostic and rounded
# outward in each dataset's native eta units.  These values must be revisited
# on scientific grounds before a confirmatory application is declared.
PILOT_FIXED_SUPPORTS = {
    "gpa": (-2.0, 0.0),
    "lending_club": (6.0, 18.0),
    "nhanes": (0.0, 1.6),
    "oulad": (-37.0, -25.0),
    "taxi": (-6.0, 11.0),
}


def _ridge_label(value: float) -> str:
    return f"ridge_{value:g}".replace(".", "p")


def main() -> Dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}
    for name in list_datasets():
        if name not in PILOT_FIXED_SUPPORTS:
            skipped[name] = "no pilot-fixed nuisance support is registered"
            continue
        try:
            sample = load(name)
        except FileNotFoundError as exc:
            skipped[name] = str(exc)
            print(f"[skip] {name}: data not present")
            continue
        if sample.k != 1:
            skipped[name] = "hard-trim implementation currently requires one Q axis"
            print(f"[skip] {name}: running variable has {sample.k} axes")
            continue

        support = PILOT_FIXED_SUPPORTS[name]
        print(f"[probe] {name}: n={sample.n:,}, support={support}")
        probe = perfrdd_hard_trim(
            sample,
            OUT_ROOT / name / "probe_ridge_0",
            support,
            eps=0.1,
            c_values=(0.0,),
            ridge_scale=0.0,
        )
        effect_scale = abs(float(probe["avg_alpha_hard_weighted"]))
        if effect_scale <= 1e-12:
            effect_scale = 1.0
        costs = tuple(ratio * effect_scale for ratio in COST_RATIOS)

        specifications: Dict[str, Dict[str, Any]] = {}
        for ridge in RIDGE_GRID:
            label = f"full_{_ridge_label(ridge)}"
            print(f"[run] {name}: {label}")
            specifications[label] = perfrdd_hard_trim(
                sample,
                OUT_ROOT / name / label,
                support,
                eps=0.1,
                c_values=costs,
                ridge_scale=ridge,
                crossfit_folds=1,
            )
        print(f"[run] {name}: crossfit_5fold_ridge_0")
        specifications["crossfit_5fold_ridge_0"] = perfrdd_hard_trim(
            sample,
            OUT_ROOT / name / "crossfit_5fold_ridge_0",
            support,
            eps=0.1,
            c_values=costs,
            ridge_scale=0.0,
            crossfit_folds=5,
        )
        results[name] = {
            "sample_description": sample.description,
            "sample_citation": sample.citation,
            "sample_extras": sample.extras,
            "nuisance_support": list(support),
            "support_provenance": "pilot-derived and rounded outward",
            "cost_ratios": list(COST_RATIOS),
            "cost_values": list(costs),
            "specifications": specifications,
        }

    payload: Dict[str, Any] = {
        "description": (
            "Exploratory exact-hard-trim application comparison with fixed "
            "nuisance supports, full-sample ridge sensitivity, and five-fold "
            "unregularized robustness estimates"
        ),
        "confirmatory": False,
        "support_provenance": (
            "Intervals were rounded outward from the prior smooth-support pilot; "
            "they are locked within this comparison but require scientific review."
        ),
        "eps": 0.1,
        "ridge_grid": list(RIDGE_GRID),
        "cost_ratios": list(COST_RATIOS),
        "results": results,
        "skipped": skipped,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[wrote] {OUT_JSON}")
    return payload


if __name__ == "__main__":
    main()
