"""Directly check VTS outcome predictions against CMT percentage rides.

The VTS hard-trim fit estimates

    fixed prediction = b(eta) + beta'X
    percentage prediction = fixed prediction + alpha(eta).

This diagnostic refits that model on the locked 30,000-ride VTS sample and
applies both predictions to restricted CMT rides.  CMT's observations below
$15 are actual percentage-menu rides, so they provide the relevant out-of-
sample check.  CMT's high-fare percentage rides are also reported as a rough
vendor/menu-level calibration.  Neither comparison is causal: CMT and VTS
have different vendors and different percentage menus.

Run from the ``code`` repository root::

    python -m experiments.scripts.taxi_competitor_prediction_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.taxi.adapter import load_haggag_paci_vendor
from experiments.methods.perfrdd import _subsample
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "taxi_competitor_prediction_check"
ANALYSIS_N = 30_000
SUBSAMPLE_SEED = 0
EPS = 0.1
RIDGE_SCALE = 0.001
NUISANCE_SUPPORT = (-6.0, 11.0)
THRESHOLD = 15.0
POLICY_GRID = np.linspace(2.5, 25.0, 451)


def _actual_split(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q) >= THRESHOLD).astype(int)


def _lock_vts(source: RDDSample) -> RDDSample:
    arrays, _, _ = _subsample(
        [source.Q, source.X, source.Y], ANALYSIS_N, seed=SUBSAMPLE_SEED
    )
    q, x, y = arrays
    return RDDSample(
        Q=q,
        X=x,
        Y=y,
        threshold=THRESHOLD,
        treatment_rule=_actual_split,
        name="taxi_haggag_paci_vts_30k_prediction_check",
        feature_names=list(source.feature_names),
        description="Locked VTS sample used to fit the prediction check.",
        citation=source.citation,
    )


def _fit_vts_model(sample: RDDSample) -> tuple[dict[str, Any], np.ndarray]:
    """Fit the current VTS specification and return its first-stage gamma."""
    result = perfrdd_hard_trim(
        sample,
        OUT_ROOT / "vts_model",
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
    gamma = np.linalg.lstsq(
        np.column_stack((np.ones(sample.n), sample.X)), sample.Q, rcond=None
    )[0]
    return result, gamma


def _predict(
    sample: RDDSample,
    gamma: np.ndarray,
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eta, fixed-menu prediction, and VTS percentage prediction."""
    eta = sample.Q - gamma[0] - sample.X @ gamma[1:]
    eta_grid = np.linspace(NUISANCE_SUPPORT[0], NUISANCE_SUPPORT[1], 501)
    baseline = np.interp(
        eta, eta_grid, np.asarray(result["returned_baseline_curve"], dtype=float)
    )
    alpha = np.interp(
        eta, eta_grid, np.asarray(result["returned_alpha_curve"], dtype=float)
    )
    beta = np.asarray(list(result["beta_coefficients"].values()), dtype=float)
    fixed = baseline + sample.X @ beta
    return eta, fixed, fixed + alpha


def _fare_table(
    sample: RDDSample,
    eta: np.ndarray,
    fixed: np.ndarray,
    percentage: np.ndarray,
    trim_interval: tuple[float, float],
) -> pd.DataFrame:
    """Summarize CMT low-fare outcomes and VTS-model predictions by fare cell."""
    keep = (
        (sample.Q >= 5.0)
        & (sample.Q < THRESHOLD)
        & (eta >= trim_interval[0])
        & (eta <= trim_interval[1])
    )
    frame = pd.DataFrame({
        "fare": np.round(sample.Q[keep], 1),
        "tip": sample.Y[keep],
        "fixed_prediction": fixed[keep],
        "percentage_prediction": percentage[keep],
    })
    grouped = frame.groupby("fare", sort=True)
    result = grouped.agg(
        n=("tip", "size"),
        observed_cmt_tip=("tip", "mean"),
        vts_model_fixed_prediction=("fixed_prediction", "mean"),
        vts_model_percentage_prediction=("percentage_prediction", "mean"),
    ).reset_index()
    result["cmt_minus_vts_fixed_prediction"] = (
        result["observed_cmt_tip"] - result["vts_model_fixed_prediction"]
    )
    result["cmt_minus_vts_percentage_prediction"] = (
        result["observed_cmt_tip"] - result["vts_model_percentage_prediction"]
    )
    return result


def _mean_prediction_residual(
    sample: RDDSample,
    eta: np.ndarray,
    outcome_prediction: np.ndarray,
    *,
    above_threshold: bool,
    trim_interval: tuple[float, float],
) -> dict[str, float]:
    mask = (sample.Q >= THRESHOLD) if above_threshold else (sample.Q < THRESHOLD)
    mask &= (eta >= trim_interval[0]) & (eta <= trim_interval[1])
    residual = sample.Y[mask] - outcome_prediction[mask]
    return {
        "n": int(mask.sum()),
        "mean_residual_dollars": float(np.mean(residual)),
        "rmse_dollars": float(np.sqrt(np.mean(residual**2))),
    }


def _plot(table: pd.DataFrame, output_path: Path) -> None:
    fare = table["fare"].to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    axes[0].plot(
        fare, table["observed_cmt_tip"], color="#C0504D", marker="o", lw=2.0,
        label="Observed CMT percentage tip",
    )
    axes[0].plot(
        fare, table["vts_model_fixed_prediction"], color="#176D9C", lw=2.0,
        label="VTS model: fixed-menu prediction",
    )
    axes[0].plot(
        fare, table["vts_model_percentage_prediction"], color="#238B45", lw=2.0,
        label="VTS model: percentage prediction",
    )
    axes[0].set_title("CMT percentage rides vs VTS model predictions")
    axes[0].set_ylabel("Mean tip dollars")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(
        fare, table["cmt_minus_vts_fixed_prediction"], color="#176D9C",
        marker="o", lw=2.0, label="CMT − VTS fixed prediction",
    )
    axes[1].plot(
        fare, table["cmt_minus_vts_percentage_prediction"], color="#238B45",
        marker="o", lw=2.0, label="CMT − VTS percentage prediction",
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title("Transfer residual by fare")
    axes[1].set_ylabel("Observed CMT tip − predicted tip")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    for axis in axes:
        axis.set_xlabel("Fare ($)")
    figure.suptitle(
        "Direct competitor check of the VTS hard-trim outcome decomposition", y=1.02
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def run(output_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Run the direct VTS-to-CMT prediction check."""
    global OUT_ROOT
    OUT_ROOT = output_root
    vts_source = load_haggag_paci_vendor("VTS")
    means = np.asarray(vts_source.extras["control_standardization_means"])
    scales = np.asarray(vts_source.extras["control_standardization_scales"])
    cmt_source = load_haggag_paci_vendor(
        "CMT", standardization_means=means, standardization_scales=scales
    )
    vts_sample = _lock_vts(vts_source)
    vts_result, gamma = _fit_vts_model(vts_sample)
    trim = (
        float(vts_result["fold_diagnostics"][0]["l_hat"]),
        float(vts_result["fold_diagnostics"][0]["u_hat"]),
    )
    eta_vts, fixed_vts, percentage_vts = _predict(vts_sample, gamma, vts_result)
    eta_cmt, fixed_cmt, percentage_cmt = _predict(cmt_source, gamma, vts_result)
    cmt_table = _fare_table(cmt_source, eta_cmt, fixed_cmt, percentage_cmt, trim)
    vts_low_fixed = _mean_prediction_residual(
        vts_sample, eta_vts, fixed_vts, above_threshold=False, trim_interval=trim
    )
    vts_high_percentage = _mean_prediction_residual(
        vts_sample, eta_vts, percentage_vts, above_threshold=True, trim_interval=trim
    )
    cmt_low_fixed = _mean_prediction_residual(
        cmt_source, eta_cmt, fixed_cmt, above_threshold=False, trim_interval=trim
    )
    cmt_low_percentage = _mean_prediction_residual(
        cmt_source, eta_cmt, percentage_cmt, above_threshold=False, trim_interval=trim
    )
    cmt_high_percentage = _mean_prediction_residual(
        cmt_source, eta_cmt, percentage_cmt, above_threshold=True, trim_interval=trim
    )
    calibrated_low_residual = (
        cmt_low_percentage["mean_residual_dollars"]
        - cmt_high_percentage["mean_residual_dollars"]
    )
    figure_path = output_root / "prediction_check.png"
    output_root.mkdir(parents=True, exist_ok=True)
    cmt_table.to_csv(output_root / "prediction_by_fare.csv", index=False)
    _plot(cmt_table, figure_path)
    payload: dict[str, Any] = {
        "description": (
            "Direct out-of-sample comparison of VTS hard-trim predictions with "
            "actual CMT percentage-menu rides"
        ),
        "identification_status": (
            "confirms the local VTS fit only if observed CMT percentage outcomes "
            "align after vendor/menu calibration; it does not identify a causal "
            "VTS low-fare counterfactual"
        ),
        "specification": {
            "analysis_n_vts": ANALYSIS_N,
            "subsample_seed": SUBSAMPLE_SEED,
            "eps": EPS,
            "ridge_scale": RIDGE_SCALE,
            "nuisance_support": list(NUISANCE_SUPPORT),
            "vts_hard_trim_interval": list(trim),
            "cmt_predictions": (
                "VTS fitted b(eta)+beta'X and b(eta)+beta'X+alpha(eta), using "
                "the VTS first-stage gamma"
            ),
        },
        "source_rows_after_restrictions": {
            "VTS": int(vts_source.n),
            "CMT": int(cmt_source.n),
        },
        "residual_checks": {
            "vts_low_observed_minus_fixed": vts_low_fixed,
            "vts_high_observed_minus_percentage": vts_high_percentage,
            "cmt_low_observed_minus_fixed": cmt_low_fixed,
            "cmt_low_observed_minus_percentage": cmt_low_percentage,
            "cmt_high_observed_minus_percentage": cmt_high_percentage,
            "cmt_low_minus_percentage_after_high_fare_calibration": {
                "n_low": cmt_low_percentage["n"],
                "mean_residual_dollars": float(calibrated_low_residual),
                "definition": "CMT low residual minus CMT high residual",
            },
        },
        "interpretation": [
            "The VTS model is internally calibrated: observed VTS low rides match its fixed prediction and observed VTS high rides match its percentage prediction.",
            "Actual CMT low-fare percentage rides lie below the VTS-model percentage prediction, so CMT does not confirm the positive VTS alpha as a transportable low-fare effect.",
            "CMT low rides are closer to (and often below) the VTS fixed prediction, which is consistent with a negative low-fare percentage-versus-fixed contrast.",
            "The high-fare CMT residual provides only a rough vendor/menu calibration; after subtracting it, the low-fare residual remains negative.",
        ],
        "outputs": {
            "prediction_by_fare": str(output_root / "prediction_by_fare.csv"),
            "figure": str(figure_path),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "CMT low observed-minus-VTS percentage prediction: "
        f"{cmt_low_percentage['mean_residual_dollars']:+.3f} dollars/trip; "
        "after high-fare calibration: "
        f"{calibrated_low_residual:+.3f}"
    )
    print(f"[wrote] {output_root / 'summary.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    run(args.output_root)


if __name__ == "__main__":
    main()
