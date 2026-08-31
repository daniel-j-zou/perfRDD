"""Run exact hard-trimmed PerfRDD on the redesigned GPA outcomes.

The outcome redesign separates full-population persistence outcomes from the
post-treatment-selected subsequent-GPA diagnostic and from explicitly valued
full-population composite outcomes.  Every outcome is estimated with the same
exact hard support indicator and the same fixed nuisance support.

The application reports two implementation strategies without selecting among
them after looking at the answers:

* full-sample point estimates over a prespecified ridge-sensitivity grid; and
* a five-fold, unregularized cross-fit robustness estimate.

The full-sample ridge-0.001 specification is used only to organize the summary
plots.  All specifications are retained in ``summary.json``.  The fixed
``(-2, 0)`` nuisance support was rounded outward from the August 2026 pilot
support diagnostic, so these runs are exploratory rather than confirmatory.

The script reports no-cost educational-value criteria only.  Adding a direct
administrative or student cost of probation is a separate author judgment.

Outputs are ignored by git and written to
``experiments/runs/gpa_redesign_hard_trim``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.sample import RDDSample
from experiments.datasets.gpa.redesign import (
    DEFAULT_NO_GRADE_ABSOLUTE_GPAS,
    DEFAULT_NO_GRADE_PENALTIES,
    load_frame,
    load_redesign_bundle,
)
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "gpa_redesign_hard_trim"
OUT_JSON = OUT_ROOT / "summary.json"
PHI_GRID = np.linspace(-0.6, 0.6, 241)
EPS = 0.1
NUISANCE_SUPPORT = (-2.0, 0.0)
RIDGE_GRID = (0.0, 0.0001, 0.001, 0.01)
PRIMARY_SPECIFICATION = "full_ridge_0p001"


def _ridge_label(value: float) -> str:
    return f"ridge_{value:g}".replace(".", "p")


def _local_linear_rd(sample: RDDSample, bandwidth: float = 0.6) -> Dict[str, float]:
    """Uncontrolled local-linear RD with Q-clustered standard error.

    This mirrors the published paper's central specification and provides a
    scale and outcome-definition check before the policy estimator is run.
    """
    q = np.asarray(sample.Q, dtype=float)
    y = np.asarray(sample.Y, dtype=float)
    d = np.asarray(sample.D, dtype=float)
    keep = np.isfinite(q) & np.isfinite(y) & (np.abs(q) <= bandwidth)
    q, y, d = q[keep], y[keep], d[keep]
    design = np.column_stack((np.ones(len(y)), d, q, d * q))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta

    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]))
    clusters = np.unique(q)
    for cluster in clusters:
        in_cluster = q == cluster
        score = design[in_cluster].T @ resid[in_cluster]
        meat += np.outer(score, score)
    n, k, g = len(y), design.shape[1], len(clusters)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k)) if g > 1 and n > k else 1.0
    covariance = correction * bread @ meat @ bread
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    return {
        "bandwidth": float(bandwidth),
        "n": int(n),
        "n_score_clusters": int(g),
        "discontinuity": float(beta[1]),
        "clustered_standard_error": standard_error,
    }


def _data_audit() -> Dict[str, Any]:
    df = load_frame()
    observed = df["nextGPA"].notna()
    below = df["dist_from_cut"] < 0.0
    near = df["dist_from_cut"].abs() <= 0.6

    cross = (
        df.assign(next_gpa_recorded=observed.astype(int))
        .groupby(["left_school", "next_gpa_recorded"])
        .size()
    )
    cross_dict = {
        f"left_school={int(left)},recorded={int(recorded)}": int(count)
        for (left, recorded), count in cross.items()
    }
    return {
        "n_full": int(len(df)),
        "n_next_gpa_recorded": int(observed.sum()),
        "n_next_gpa_missing": int((~observed).sum()),
        "n_left_school": int(df["left_school"].sum()),
        "missingness_by_left_school": cross_dict,
        "n_within_0p6": int(near.sum()),
        "n_next_gpa_recorded_within_0p6": int((near & observed).sum()),
        "recorded_rate_below_cutoff_within_0p6": float(observed[near & below].mean()),
        "recorded_rate_above_cutoff_within_0p6": float(observed[near & ~below].mean()),
        "fall_return_rate_below_cutoff_within_0p6": float(
            df.loc[near & below, "fallreg_year2"].mean()
        ),
        "fall_return_rate_above_cutoff_within_0p6": float(
            df.loc[near & ~below, "fallreg_year2"].mean()
        ),
        "next_gpa_units": "subsequent absolute GPA minus campus-specific cutoff",
    }


def _run_specifications(sample: RDDSample, key: str) -> Dict[str, Dict[str, Any]]:
    """Run the locked full-sample ridge grid and cross-fit robustness check."""
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
            phi_grid=PHI_GRID,
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
        phi_grid=PHI_GRID,
        max_n=None,
        ridge_scale=0.0,
        crossfit_folds=5,
    )
    return specifications


def _result_row(
    result: Dict[str, Any], sensitivity_value: float,
) -> Tuple[float, float, float, bool]:
    cost_key = str(result["c_values"][0])
    return (
        float(sensitivity_value),
        float(result["avg_alpha_hard_weighted"]),
        float(result["phi_star"][cost_key]),
        bool(result["phi_star_at_grid_boundary"][cost_key]),
    )


def _primary_result(results: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    return results[key]["specifications"][PRIMARY_SPECIFICATION]


def _plot_sensitivity(results: Dict[str, Dict[str, Any]]) -> None:
    gpa_rows: list[Tuple[float, float, float, bool]] = []
    for assumed in DEFAULT_NO_GRADE_ABSOLUTE_GPAS:
        key = f"composite_no_grade_{assumed:.2f}"
        gpa_rows.append(_result_row(_primary_result(results, key), assumed))

    penalty_rows = [
        _result_row(_primary_result(results, "composite_no_grade_0.00"), 0.0)
    ]
    for penalty in DEFAULT_NO_GRADE_PENALTIES:
        key = f"composite_zero_gpa_penalty_{penalty:.2f}"
        penalty_rows.append(_result_row(_primary_result(results, key), penalty))

    gpa_values = np.asarray(gpa_rows, dtype=float)
    penalty_values = np.asarray(penalty_rows, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()
    ax1.plot(gpa_values[:, 0], gpa_values[:, 1], "o-", color="C0")
    ax1.axhline(0.0, color="black", linewidth=0.6)
    ax1.set_xlabel("Assigned absolute GPA for no subsequent record")
    ax1.set_ylabel(r"Hard-trimmed average $\hat\alpha$")
    ax1.set_title("Physical-GPA sensitivity: effect")

    ax2.plot(gpa_values[:, 0], gpa_values[:, 2], "o-", color="C1")
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Assigned absolute GPA for no subsequent record")
    ax2.set_ylabel(r"No-cost policy threshold $\hat\phi^*$")
    ax2.set_title("Physical-GPA sensitivity: policy")

    ax3.plot(penalty_values[:, 0], penalty_values[:, 1], "o-", color="C2")
    ax3.axhline(0.0, color="black", linewidth=0.6)
    ax3.set_xlabel("GPA-equivalent penalty for no subsequent record")
    ax3.set_ylabel(r"Hard-trimmed average $\hat\alpha$")
    ax3.set_title("No-record valuation: effect")

    ax4.plot(penalty_values[:, 0], penalty_values[:, 2], "o-", color="C3")
    ax4.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax4.set_xlabel("GPA-equivalent penalty for no subsequent record")
    ax4.set_ylabel(r"No-cost policy threshold $\hat\phi^*$")
    ax4.set_title("No-record valuation: policy")
    fig.suptitle("Primary display: full sample, ridge scale 0.001", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "composite_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> Dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    samples = load_redesign_bundle()
    audit = _data_audit()
    local_rd = {key: _local_linear_rd(sample) for key, sample in samples.items()}
    results: Dict[str, Dict[str, Any]] = {}

    print(json.dumps(audit, indent=2))
    for key, sample in samples.items():
        print(f"[sample] {key}: n={sample.n:,}")
        results[key] = {
            "sample_description": sample.description,
            "sample_extras": sample.extras,
            "specifications": _run_specifications(sample, key),
        }

    _plot_sensitivity(results)
    payload: Dict[str, Any] = {
        "description": (
            "GPA outcome redesign estimated with exact hard trimming, a locked "
            "full-sample ridge grid, and five-fold unregularized robustness"
        ),
        "confirmatory": False,
        "data_audit": audit,
        "eps": EPS,
        "nuisance_support": list(NUISANCE_SUPPORT),
        "support_provenance": (
            "Rounded outward from the August 2026 pilot support diagnostic; "
            "not selected separately by outcome and not yet confirmatory."
        ),
        "policy_grid": [float(PHI_GRID[0]), float(PHI_GRID[-1]), int(len(PHI_GRID))],
        "policy_grid_note": (
            "Provisional local policy range matching the original paper's 0.6-GPA "
            "central RD bandwidth; not a theoretically selected policy set."
        ),
        "cost_values": [0.0],
        "cost_note": (
            "Direct costs of probation are intentionally omitted. Any welfare-optimal "
            "threshold requires an author-specified cost."
        ),
        "ridge_grid": list(RIDGE_GRID),
        "primary_display_specification": PRIMARY_SPECIFICATION,
        "primary_display_note": (
            "Used to organize plots only; every full-sample ridge and cross-fit result "
            "is retained for transparent sensitivity analysis."
        ),
        "inference_available": False,
        "inference_note": (
            "These are point estimates. The application does not yet implement the "
            "manuscript's boundary-aware hard-trim influence-function variance."
        ),
        "local_linear_validation": local_rd,
        "hard_trim_results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[wrote] {OUT_JSON}")
    return payload


if __name__ == "__main__":
    main()
