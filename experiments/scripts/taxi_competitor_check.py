"""Competitor-only placebo check for the taxi hard-trim outcome decomposition.

The Vendor (VTS) records identify a local jump because the suggested-tip menu
changes at a $15 fare.  Competitor (CMT) records do *not* have that jump: their
percentage suggestions apply on both sides of $15.  We therefore fit the same
hard-trim model to CMT after assigning an artificial ``D = 1{Fare_Amt >= 15}``
placebo treatment.  A small placebo ``alpha`` is reassuring about the nuisance
decomposition; it is not a second causal treatment estimate.

Both vendors use the published paper restrictions, the same VTS-based control
standardization, the same deterministic 30,000-trip subsample size, hard-trim
window, ridge penalty, and policy grid.  The script writes each fitted model's
component exports plus a comparison figure and machine-readable summary.

Run from the ``code`` repository root::

    python -m experiments.scripts.taxi_competitor_check
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.sample import RDDSample
from experiments.datasets.taxi.adapter import load_haggag_paci_vendor
from experiments.methods.perfrdd import _subsample
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "taxi_competitor_check"
ANALYSIS_N = 30_000
SUBSAMPLE_SEED = 0
EPS = 0.1
RIDGE_SCALE = 0.001
NUISANCE_SUPPORT = (-6.0, 11.0)
THRESHOLD = 15.0
POLICY_GRID = np.linspace(2.5, 25.0, 451)


def _placebo_treatment(q: np.ndarray) -> np.ndarray:
    """Artificial threshold split used only for the CMT placebo."""
    return (np.asarray(q) >= THRESHOLD).astype(int)


def _lock_sample(
    source: RDDSample,
    *,
    name: str,
    description: str,
    treatment_assignment: str,
) -> RDDSample:
    """Take the common deterministic pilot-size subsample."""
    arrays, _, _ = _subsample(
        [source.Q, source.X, source.Y], ANALYSIS_N, seed=SUBSAMPLE_SEED
    )
    q, x, y = arrays
    return RDDSample(
        Q=q,
        X=x,
        Y=y,
        threshold=THRESHOLD,
        treatment_rule=_placebo_treatment,
        name=name,
        feature_names=list(source.feature_names),
        description=description,
        citation=source.citation,
        extras={
            **source.extras,
            "analysis_n": ANALYSIS_N,
            "subsample_seed": SUBSAMPLE_SEED,
            "treatment_assignment": treatment_assignment,
        },
    )


def _fit(sample: RDDSample, output_dir: Path) -> dict[str, Any]:
    """Fit one vendor's point model with the locked specification."""
    result = perfrdd_hard_trim(
        sample,
        output_dir,
        NUISANCE_SUPPORT,
        eps=EPS,
        c_values=(0.0,),
        phi_grid=POLICY_GRID,
        max_n=None,
        ridge_scale=RIDGE_SCALE,
        crossfit_folds=1,
        write_outputs=True,
        return_curves=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _local_fare_cells(source: RDDSample) -> dict[str, Any]:
    """Return adjacent standard-meter fare cells around the artificial split."""
    cells: dict[str, Any] = {}
    for fare in (14.9, 15.3):
        mask = np.isclose(source.Q, fare, atol=1e-6)
        if not mask.any():
            cells[f"{fare:.1f}"] = {"n": 0}
            continue
        cells[f"{fare:.1f}"] = {
            "n": int(mask.sum()),
            "mean_tip_dollars": float(np.mean(source.Y[mask])),
            "mean_tip_rate": float(np.mean(source.Y[mask] / source.Q[mask])),
        }
    lower = cells["14.9"]
    upper = cells["15.3"]
    if lower["n"] and upper["n"]:
        cells["adjacent_tip_jump_15p3_minus_14p9"] = float(
            upper["mean_tip_dollars"] - lower["mean_tip_dollars"]
        )
    return cells


def _summary(result: dict[str, Any], source: RDDSample) -> dict[str, Any]:
    """Extract comparable alpha/beta/b diagnostics from one fitted result."""
    eta_grid = np.linspace(NUISANCE_SUPPORT[0], NUISANCE_SUPPORT[1], 501)
    alpha = np.asarray(result["returned_alpha_curve"], dtype=float)
    baseline = np.asarray(result["returned_baseline_curve"], dtype=float)
    trim_lo, trim_hi = result["fold_diagnostics"][0]["l_hat"], result[
        "fold_diagnostics"
    ][0]["u_hat"]
    in_trim = (eta_grid >= trim_lo) & (eta_grid <= trim_hi)
    if not in_trim.any():
        raise RuntimeError("fitted hard-trim window has no grid points")
    return {
        "name": result["name"],
        "n_used": int(result["n_used"]),
        "n_treated": int(result["n_treated"]),
        "hard_retention": float(result["hard_retention"]),
        "hard_trim_interval": [float(trim_lo), float(trim_hi)],
        "avg_alpha_hard_weighted": float(result["avg_alpha_hard_weighted"]),
        "alpha_min_on_hard_trim_grid": float(np.min(alpha[in_trim])),
        "alpha_max_on_hard_trim_grid": float(np.max(alpha[in_trim])),
        "alpha_mean_abs_on_hard_trim_grid": float(np.mean(np.abs(alpha[in_trim]))),
        "alpha_max_abs_on_hard_trim_grid": float(np.max(np.abs(alpha[in_trim]))),
        "beta_coefficients": result["beta_coefficients"],
        "design_condition_number": float(
            result["fold_diagnostics"][0]["design_condition_number"]
        ),
        "first_stage_R2": float(result["fold_diagnostics"][0]["first_stage_R2"]),
        "adjacent_fare_cells": _local_fare_cells(source),
        "eta_grid": eta_grid,
        "alpha_curve": alpha,
        "baseline_curve": baseline,
    }


def _comparison_figure(
    vts: dict[str, Any], cmt: dict[str, Any], output_path: Path
) -> None:
    """Plot alpha and baseline curves on the common VTS-standardized index."""
    eta = np.asarray(vts["eta_grid"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for label, fit, color in (
        ("VTS: actual $15 menu jump", vts, "#176D9C"),
        ("CMT: placebo split at $15", cmt, "#C0504D"),
    ):
        axes[0].plot(eta, fit["alpha_curve"], color=color, lw=2.1, label=label)
        axes[1].plot(eta, fit["baseline_curve"], color=color, lw=2.1, label=label)
    for fit in (vts, cmt):
        lo, hi = fit["hard_trim_interval"]
        for axis in axes:
            axis.axvspan(lo, hi, color="#74C476", alpha=0.08)
    axes[0].axhline(0.0, color="black", lw=0.7)
    axes[0].set_title(r"Estimated $\hat\alpha(\eta)$")
    axes[0].set_ylabel("Tip-dollar change from the artificial split")
    axes[1].set_title(r"Estimated baseline $\hat b(\eta)$")
    axes[1].set_ylabel("Predicted tip dollars")
    for axis in axes:
        axis.set_xlabel(r"VTS-standardized fare residual $\hat\eta$")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Taxi hard-trim outcome decomposition: VTS benchmark vs CMT placebo",
        y=1.02,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def run(output_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Run the VTS benchmark and CMT-only placebo comparison."""
    vts_source = load_haggag_paci_vendor("VTS")
    means = np.asarray(vts_source.extras["control_standardization_means"])
    scales = np.asarray(vts_source.extras["control_standardization_scales"])
    cmt_source = load_haggag_paci_vendor(
        "CMT", standardization_means=means, standardization_scales=scales
    )
    vts = _lock_sample(
        vts_source,
        name="taxi_haggag_paci_vts_30k",
        description="VTS benchmark with the actual $15 menu discontinuity.",
        treatment_assignment="1{Fare_Amt >= 15}; actual VTS menu assignment",
    )
    cmt = _lock_sample(
        cmt_source,
        name="taxi_haggag_paci_cmt_placebo_30k",
        description=(
            "CMT placebo: CMT used percentage suggestions on both sides of $15; "
            "D=1{Fare_Amt >= 15} is not an actual treatment assignment."
        ),
        treatment_assignment="1{Fare_Amt >= 15}; artificial CMT placebo split",
    )
    vts_result = _fit(vts, output_root / "vts")
    cmt_result = _fit(cmt, output_root / "cmt_placebo")
    vts_summary = _summary(vts_result, vts_source)
    cmt_summary = _summary(cmt_result, cmt_source)

    common = (
        np.asarray(vts_summary["eta_grid"]) >= max(
            vts_summary["hard_trim_interval"][0], cmt_summary["hard_trim_interval"][0]
        )
    ) & (
        np.asarray(vts_summary["eta_grid"]) <= min(
            vts_summary["hard_trim_interval"][1], cmt_summary["hard_trim_interval"][1]
        )
    )
    if common.sum() < 10:
        raise RuntimeError("VTS and CMT hard-trim windows barely overlap")
    b_vts = np.asarray(vts_summary["baseline_curve"])[common]
    b_cmt = np.asarray(cmt_summary["baseline_curve"])[common]
    b_difference = b_vts - b_cmt
    alpha_vts = np.asarray(vts_summary["alpha_curve"])[common]
    alpha_cmt = np.asarray(cmt_summary["alpha_curve"])[common]
    figure_path = output_root / "vts_cmt_components.png"
    _comparison_figure(vts_summary, cmt_summary, figure_path)

    # Arrays are useful to the figure helper but should not be serialized into
    # the headline JSON, which is intended to be easy to inspect and diff.
    for summary in (vts_summary, cmt_summary):
        summary.pop("eta_grid")
        summary.pop("alpha_curve")
        summary.pop("baseline_curve")
    payload: dict[str, Any] = {
        "description": (
            "Competitor-only placebo validation of the taxi hard-trim outcome "
            "decomposition on the paper-restricted January 2009 sample"
        ),
        "identification_status": (
            "CMT has percentage suggestions below and above $15; its alpha is a "
            "placebo threshold split, not a causal menu effect."
        ),
        "common_specification": {
            "paper_restrictions": True,
            "analysis_n_each": ANALYSIS_N,
            "subsample_seed": SUBSAMPLE_SEED,
            "eps": EPS,
            "ridge_scale": RIDGE_SCALE,
            "nuisance_support": list(NUISANCE_SUPPORT),
            "threshold_split": THRESHOLD,
            "control_standardization": "VTS full restricted source moments",
        },
        "source_rows_after_paper_restrictions": {
            "VTS": int(vts_source.extras["source_rows_after_paper_restrictions"]),
            "CMT": int(cmt_source.extras["source_rows_after_paper_restrictions"]),
        },
        "vts_actual_menu_jump": vts_summary,
        "cmt_placebo": cmt_summary,
        "cross_vendor_baseline_comparison_on_common_trim": {
            "n_eta_grid_points": int(common.sum()),
            "baseline_difference_vts_minus_cmt_mean": float(np.mean(b_difference)),
            "baseline_difference_vts_minus_cmt_rmse": float(
                np.sqrt(np.mean(b_difference**2))
            ),
            "baseline_curve_correlation": float(np.corrcoef(b_vts, b_cmt)[0, 1]),
            "alpha_difference_vts_minus_cmt_mean": float(np.mean(alpha_vts - alpha_cmt)),
            "cmt_placebo_alpha_mean_abs_common_trim": float(np.mean(np.abs(alpha_cmt))),
            "cmt_placebo_alpha_max_abs_common_trim": float(np.max(np.abs(alpha_cmt))),
        },
        "interpretation": [
            "A near-zero CMT placebo alpha supports the model's smooth outcome decomposition at the artificial split.",
            "A nonzero CMT placebo alpha flags residual fare/menu or vendor-composition structure; it does not estimate the VTS treatment effect.",
            "The CMT baseline curve is a shape/scale diagnostic only because CMT's actual percentage menu differs from VTS's fixed-dollar low-fare menu.",
            "Neither vendor-only comparison identifies the VTS counterfactual effect below $15 without a menu-aware outcome model or additional overlap assumptions.",
        ],
        "outputs": {
            "vts_summary": str(output_root / "vts" / "summary.json"),
            "cmt_summary": str(output_root / "cmt_placebo" / "summary.json"),
            "comparison_figure": str(figure_path),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[wrote] {output_root / 'summary.json'}")
    print(
        "CMT placebo alpha on common trim: "
        f"mean abs={payload['cross_vendor_baseline_comparison_on_common_trim']['cmt_placebo_alpha_mean_abs_common_trim']:.4f}; "
        f"max abs={payload['cross_vendor_baseline_comparison_on_common_trim']['cmt_placebo_alpha_max_abs_common_trim']:.4f}"
    )
    return payload


if __name__ == "__main__":
    run()
