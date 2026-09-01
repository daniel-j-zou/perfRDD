"""Estimate a CMT-based proxy for the low-fare VTS menu effect.

The hard-trim RDD identifies the VTS menu jump locally at $15.  It does not,
by itself, identify what would happen if the VTS percentage menu were applied
to fares below $15.  This diagnostic uses the institutional comparison in the
paper as an auxiliary proxy:

* CMT used percentage suggestions below $15;
* VTS used fixed-dollar suggestions below $15.

On the pooled low-fare sample, we estimate fare-cell fixed effects, common
control slopes, CMT-by-control slopes, and a CMT-by-fare-cell contrast.  The
reported contrast is CMT minus VTS, evaluated at the mean VTS controls in each
fare cell.  If vendors are exchangeable conditional on the observed controls,
this is a proxy for the percentage-menu-minus-fixed-menu effect at that fare.
It is not the PerfRDD causal ``alpha(eta)``: vendor selection, unobserved
composition, and the different menus can all violate that transport condition.

Run from the ``code`` repository root::

    python -m experiments.scripts.taxi_low_fare_proxy
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

from experiments.datasets.taxi.adapter import load_haggag_paci_vendor


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "taxi_low_fare_proxy"
MIN_FARE = 5.0
MAX_FARE_EXCLUSIVE = 15.0


def _load_common_standardized_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load restricted VTS/CMT data on the same VTS control scale."""
    vts = load_haggag_paci_vendor("VTS")
    means = np.asarray(vts.extras["control_standardization_means"], dtype=float)
    scales = np.asarray(vts.extras["control_standardization_scales"], dtype=float)
    cmt = load_haggag_paci_vendor(
        "CMT", standardization_means=means, standardization_scales=scales
    )
    frame = pd.concat(
        [
            pd.DataFrame({
                "fare": vts.Q,
                "tip": vts.Y,
                "vendor_cmt": np.zeros(vts.n, dtype=float),
                **{f"x{j}": vts.X[:, j] for j in range(vts.p)},
            }),
            pd.DataFrame({
                "fare": cmt.Q,
                "tip": cmt.Y,
                "vendor_cmt": np.ones(cmt.n, dtype=float),
                **{f"x{j}": cmt.X[:, j] for j in range(cmt.p)},
            }),
        ],
        ignore_index=True,
    )
    metadata = {
        "source_rows_after_paper_restrictions": {
            "VTS": int(vts.n),
            "CMT": int(cmt.n),
        },
        "control_standardization": "VTS full restricted source moments",
        "feature_names": list(vts.feature_names),
        "control_standardization_means": means.tolist(),
        "control_standardization_scales": scales.tolist(),
    }
    return frame, metadata


def _fare_cells(fare: np.ndarray) -> np.ndarray:
    """Return the observed standard-meter cells in the low-fare region."""
    cells = np.unique(np.round(np.asarray(fare, dtype=float), 1))
    cells = cells[(cells >= MIN_FARE) & (cells < MAX_FARE_EXCLUSIVE)]
    if len(cells) < 2:
        raise ValueError("too few low-fare cells for the proxy diagnostic")
    return cells


def _robust_covariance(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Return the HC0 sandwich covariance without a large temporary matrix."""
    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    block_size = 100_000
    for start in range(0, len(design), block_size):
        stop = min(start + block_size, len(design))
        block = design[start:stop]
        meat += block.T @ (block * (residual[start:stop] ** 2)[:, None])
    return bread @ meat @ bread


def fit_low_fare_proxy(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit fare-cell/vendor contrasts and return a tidy proxy table."""
    controls = [column for column in frame.columns if column.startswith("x")]
    low = frame["fare"].between(MIN_FARE, MAX_FARE_EXCLUSIVE, inclusive="left")
    data = frame.loc[low].copy()
    data["fare_cell"] = np.round(data["fare"].to_numpy(dtype=float), 1)
    cells = _fare_cells(data["fare_cell"].to_numpy(dtype=float))
    data = data[data["fare_cell"].isin(cells)].copy()
    fare_index = pd.Categorical(data["fare_cell"], categories=cells).codes
    if (fare_index < 0).any():
        raise ValueError("failed to encode one or more fare cells")
    n_cells = len(cells)
    fare_fe = np.zeros((len(data), n_cells), dtype=float)
    fare_fe[np.arange(len(data)), fare_index] = 1.0
    x = data[controls].to_numpy(dtype=float)
    cmt = data["vendor_cmt"].to_numpy(dtype=float)
    # VTS fare-cell means are the reference curve.  The CMT-by-cell block is
    # the requested percentage-menu-minus-fixed-menu contrast at X=0.
    design = np.column_stack((fare_fe, x, cmt[:, None] * x, cmt[:, None] * fare_fe))
    y = data["tip"].to_numpy(dtype=float)
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficient
    covariance = _robust_covariance(design, residual)
    n_controls = len(controls)
    cmt_x = coefficient[n_cells + n_controls:n_cells + 2 * n_controls]
    cmt_fare = coefficient[n_cells + 2 * n_controls:]

    rows: list[dict[str, Any]] = []
    for index, fare in enumerate(cells):
        vts_cell = data[
            (data["fare_cell"] == fare) & (data["vendor_cmt"] == 0.0)
        ]
        cmt_cell = data[
            (data["fare_cell"] == fare) & (data["vendor_cmt"] == 1.0)
        ]
        mean_x_vts = vts_cell[controls].to_numpy(dtype=float).mean(axis=0)
        contrast_at_vts_x = float(cmt_fare[index] + mean_x_vts @ cmt_x)
        linear_functional = np.zeros(len(coefficient), dtype=float)
        linear_functional[n_cells + n_controls:n_cells + 2 * n_controls] = mean_x_vts
        linear_functional[n_cells + 2 * n_controls + index] = 1.0
        standard_error = float(
            np.sqrt(max(0.0, linear_functional @ covariance @ linear_functional))
        )
        raw_difference = float(cmt_cell["tip"].mean() - vts_cell["tip"].mean())
        rows.append({
            "fare": float(fare),
            "n_vts": int(len(vts_cell)),
            "n_cmt": int(len(cmt_cell)),
            "mean_tip_vts": float(vts_cell["tip"].mean()),
            "mean_tip_cmt": float(cmt_cell["tip"].mean()),
            "raw_cmt_minus_vts": raw_difference,
            "alpha_proxy_cmt_minus_vts": contrast_at_vts_x,
            "alpha_proxy_hc0_se": standard_error,
            "alpha_proxy_ci95_lower": contrast_at_vts_x - 1.96 * standard_error,
            "alpha_proxy_ci95_upper": contrast_at_vts_x + 1.96 * standard_error,
        })
    result = pd.DataFrame(rows)
    vts_weights = result["n_vts"].to_numpy(dtype=float)
    vts_weights /= vts_weights.sum()
    proxy = result["alpha_proxy_cmt_minus_vts"].to_numpy(dtype=float)
    raw = result["raw_cmt_minus_vts"].to_numpy(dtype=float)
    negative_weight = float(vts_weights[proxy < 0.0].sum())
    crossing = None
    for left, right in zip(result.itertuples(), result.iloc[1:].itertuples()):
        if left.alpha_proxy_cmt_minus_vts < 0.0 <= right.alpha_proxy_cmt_minus_vts:
            slope = (
                right.alpha_proxy_cmt_minus_vts - left.alpha_proxy_cmt_minus_vts
            ) / (right.fare - left.fare)
            crossing = float(left.fare - left.alpha_proxy_cmt_minus_vts / slope)
            break
    return {
        "table": result,
        "summary": {
            "low_fare_analysis_rows": int(len(data)),
            "fare_cells": [float(value) for value in cells],
            "fare_cell_count": int(len(cells)),
            "model": (
                "low-fare fare-cell fixed effects + common X slopes + CMT-by-X "
                "slopes + CMT-by-fare-cell effects"
            ),
            "proxy_definition": (
                "CMT minus VTS predicted tip at the mean VTS controls in each "
                "fare cell"
            ),
            "vts_weighted_proxy_dollars_per_trip": float(np.dot(proxy, vts_weights)),
            "vts_weighted_raw_difference_dollars_per_trip": float(
                np.dot(raw, vts_weights)
            ),
            "vts_weighted_share_with_negative_proxy": negative_weight,
            "negative_proxy_fare_min": float(result.loc[proxy < 0.0, "fare"].min())
            if (proxy < 0.0).any() else None,
            "negative_proxy_fare_max": float(result.loc[proxy < 0.0, "fare"].max())
            if (proxy < 0.0).any() else None,
            "point_crossing_fare": crossing,
            "design_rank": int(np.linalg.matrix_rank(design)),
            "design_columns": int(design.shape[1]),
            "design_condition_number": float(np.linalg.cond(design)),
        },
    }


def _plot_proxy(table: pd.DataFrame, output_path: Path) -> None:
    """Plot the proxy alpha curve and the underlying fare-cell means."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fare = table["fare"].to_numpy(dtype=float)
    proxy = table["alpha_proxy_cmt_minus_vts"].to_numpy(dtype=float)
    lower = table["alpha_proxy_ci95_lower"].to_numpy(dtype=float)
    upper = table["alpha_proxy_ci95_upper"].to_numpy(dtype=float)
    axes[0].plot(fare, proxy, color="#176D9C", marker="o", lw=2.1,
                 label="CMT − VTS adjusted proxy")
    axes[0].fill_between(fare, lower, upper, color="#9ecae1", alpha=0.4,
                         label="95% HC0 interval")
    axes[0].plot(
        fare, table["raw_cmt_minus_vts"], color="#777777", ls="--", lw=1.3,
        label="Raw CMT − VTS difference",
    )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_title("Low-fare menu-effect proxy")
    axes[0].set_ylabel("Tip-dollar difference (CMT − VTS)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(fare, table["mean_tip_vts"], color="#176D9C", marker="o",
                 lw=1.8, label="VTS fixed-dollar menu")
    axes[1].plot(fare, table["mean_tip_cmt"], color="#C0504D", marker="o",
                 lw=1.8, label="CMT percentage menu")
    axes[1].set_title("Observed low-fare tip means")
    axes[1].set_ylabel("Mean tip dollars")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    for axis in axes:
        axis.set_xlabel("Fare ($)")
    figure.suptitle(
        "Taxi low-fare CMT proxy for percentage versus fixed-dollar tips", y=1.02
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def run(output_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Run the proxy, write exports, and return the summary."""
    frame, metadata = _load_common_standardized_data()
    fitted = fit_low_fare_proxy(frame)
    table: pd.DataFrame = fitted["table"]
    summary = fitted["summary"]
    # Populate source counts after applying the low-fare window.  Keeping this
    # separate from the fit helper makes that helper easy to test on fixtures.
    low = frame["fare"].between(MIN_FARE, MAX_FARE_EXCLUSIVE, inclusive="left")
    summary["low_fare_source_rows"] = {
        "VTS": int((low & (frame["vendor_cmt"] == 0.0)).sum()),
        "CMT": int((low & (frame["vendor_cmt"] == 1.0)).sum()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "fare_proxy.csv"
    figure_path = output_root / "fare_proxy.png"
    table.to_csv(table_path, index=False)
    _plot_proxy(table, figure_path)
    payload: dict[str, Any] = {
        "description": (
            "CMT-based low-fare proxy for the VTS percentage-menu-minus-fixed-"
            "menu effect"
        ),
        "identification_status": (
            "auxiliary vendor-contrast proxy; not the causal PerfRDD alpha(eta)"
        ),
        "sample": metadata,
        "specification": {
            "minimum_fare": MIN_FARE,
            "maximum_fare_exclusive": MAX_FARE_EXCLUSIVE,
            "controls": metadata["feature_names"],
            "standardization": metadata["control_standardization"],
        },
        "results": summary,
        "limitations": [
            "CMT and VTS are different vendors; conditional exchangeability is unverified.",
            "Public January data lack the driver identifiers needed for within-driver vendor comparisons.",
            "Fare-cell contrasts are not indexed by the PerfRDD residual eta and do not identify the VTS counterfactual alone.",
            "HC0 intervals treat trips as independent and are exploratory, not application inference.",
            "CMT's percentage menu and VTS's fixed-dollar menu may change behavior through channels beyond suggested amount differences.",
        ],
        "outputs": {
            "fare_proxy_csv": str(table_path),
            "figure": str(figure_path),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "VTS-weighted low-fare proxy: "
        f"{summary['vts_weighted_proxy_dollars_per_trip']:+.3f} dollars/trip; "
        f"negative share={summary['vts_weighted_share_with_negative_proxy']:.1%}; "
        f"crossing={summary['point_crossing_fare']}"
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
