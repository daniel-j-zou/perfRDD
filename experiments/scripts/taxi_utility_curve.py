"""Plot an economically interpretable hard-trim utility curve for taxi tips.

The generic hard-trim application runner writes utility at several treatment
costs.  Because the criterion is affine in the per-treated-trip cost, those
outputs identify two components at every candidate fare threshold:

    utility(phi; cost) = tip_benefit(phi) - cost * exposure(phi).

This script reconstructs those components and evaluates the curve at an
explicit dollar cost.  The plotted vertical axis is relative to the utility at
the observed $15 policy, which makes the magnitude readable in cents per
hard-trimmed trip.  It compares a mildly regularized full-sample estimate with
the unregularized five-fold cross-fitted robustness estimate.

Run after ``python -m experiments.scripts.hard_trim_existing_applications``::

    python -m experiments.scripts.taxi_utility_curve --cost 0.20
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


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = ROOT / "runs" / "hard_trim_existing" / "taxi"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "taxi_utility_curve"
CURRENT_THRESHOLD = 15.0
OBSERVED_MINIMUM_FARE = 2.5
DEFAULT_POLICY_MAXIMUM = 30.0

SPECIFICATIONS = {
    "Regularized full sample": "full_ridge_0p001",
    "Five-fold cross-fit": "crossfit_5fold_ridge_0",
}


def reconstruct_components(path: Path) -> pd.DataFrame:
    """Recover gross tip benefit and treatment exposure from a curve CSV."""
    frame = pd.read_csv(path)
    if "phi" not in frame or "cost_0" not in frame:
        raise ValueError(f"{path} must contain phi and cost_0 columns")

    positive_cost_columns: list[tuple[float, str]] = []
    for column in frame.columns:
        if not column.startswith("cost_") or column == "cost_0":
            continue
        try:
            cost = float(column[len("cost_"):])
        except ValueError:
            continue
        if cost > 0.0:
            positive_cost_columns.append((cost, column))
    if not positive_cost_columns:
        raise ValueError(f"{path} has no positive-cost utility column")

    reference_cost, reference_column = min(positive_cost_columns)
    gross_benefit = frame["cost_0"].to_numpy(dtype=float)
    exposure = (
        gross_benefit - frame[reference_column].to_numpy(dtype=float)
    ) / reference_cost
    if np.any(exposure < -1e-8) or np.any(exposure > 1.0 + 1e-8):
        raise ValueError("reconstructed treatment exposure falls outside [0, 1]")

    return pd.DataFrame({
        "phi": frame["phi"].to_numpy(dtype=float),
        "gross_tip_benefit": gross_benefit,
        "treatment_exposure": np.clip(exposure, 0.0, 1.0),
    })


def evaluate_curve(components: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Evaluate net utility at ``cost`` dollars per treated transaction."""
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("cost must be finite and nonnegative")
    evaluated = components.copy()
    evaluated["utility"] = (
        evaluated["gross_tip_benefit"]
        - cost * evaluated["treatment_exposure"]
    )
    return evaluated


def summarize_curve(
    curve: pd.DataFrame,
    *,
    current_threshold: float = CURRENT_THRESHOLD,
    policy_minimum: float = OBSERVED_MINIMUM_FARE,
    policy_maximum: float = DEFAULT_POLICY_MAXIMUM,
) -> dict[str, float]:
    """Return the grid optimum and its gain relative to the current policy."""
    domain = curve["phi"].between(policy_minimum, policy_maximum)
    if not domain.any():
        raise ValueError("the requested policy domain does not intersect the curve")
    optimum_index = curve.loc[domain, "utility"].idxmax()
    current_index = (curve["phi"] - current_threshold).abs().idxmin()
    optimum = curve.loc[optimum_index]
    current = curve.loc[current_index]
    return {
        "phi_star": float(optimum["phi"]),
        "utility_at_phi_star_dollars_per_trimmed_trip": float(optimum["utility"]),
        "utility_at_current_dollars_per_trimmed_trip": float(current["utility"]),
        "gain_over_current_cents_per_trimmed_trip": float(
            100.0 * (optimum["utility"] - current["utility"])
        ),
        "treatment_exposure_at_phi_star": float(optimum["treatment_exposure"]),
        "treatment_exposure_at_current": float(current["treatment_exposure"]),
    }


def make_figure(
    curves: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, float]],
    *,
    cost: float,
    output_path: Path,
    current_threshold: float = CURRENT_THRESHOLD,
    policy_minimum: float = OBSERVED_MINIMUM_FARE,
    policy_maximum: float = DEFAULT_POLICY_MAXIMUM,
) -> None:
    """Plot utility changes relative to the observed threshold."""
    figure, axis = plt.subplots(figsize=(9.5, 5.7))
    colors = ("#176D9C", "#C0504D")
    for (label, curve), color in zip(curves.items(), colors):
        current_index = (curve["phi"] - current_threshold).abs().idxmin()
        current_utility = float(curve.loc[current_index, "utility"])
        domain = curve["phi"].between(policy_minimum, policy_maximum)
        x = curve.loc[domain, "phi"]
        relative_cents = 100.0 * (
            curve.loc[domain, "utility"] - current_utility
        )
        axis.plot(x, relative_cents, color=color, linewidth=2.3, label=label)
        result = summaries[label]
        axis.scatter(
            result["phi_star"],
            result["gain_over_current_cents_per_trimmed_trip"],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=62,
            zorder=4,
        )

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(
        current_threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Current $15 threshold",
    )
    axis.set_title(
        f"NYC taxi default tips: estimated utility at ${cost:.2f} policy cost"
    )
    axis.set_xlabel("Fare threshold for percentage tip suggestions ($)")
    axis.set_ylabel("Utility change from current policy\n(cents per hard-trimmed trip)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def run(
    *,
    cost: float = 0.20,
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Reconstruct, summarize, and plot both taxi point-estimation modes."""
    curves: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, float]] = {}
    for label, directory in SPECIFICATIONS.items():
        path = input_root / directory / "utility_curve.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run hard_trim_existing_applications first"
            )
        curve = evaluate_curve(reconstruct_components(path), cost)
        curves[label] = curve
        summaries[label] = summarize_curve(curve)

    cost_label = f"{cost:.2f}".replace(".", "p")
    figure_path = output_root / f"estimated_utility_curve_cost_{cost_label}.png"
    make_figure(curves, summaries, cost=cost, output_path=figure_path)
    payload: dict[str, Any] = {
        "application": "NYC TLC January 2009 VTS credit-card taxi trips",
        "outcome": "tip amount in dollars",
        "treatment": "percentage rather than fixed-dollar tip suggestions",
        "current_fare_threshold": CURRENT_THRESHOLD,
        "policy_cost_dollars_per_treated_trip": float(cost),
        "policy_domain": [OBSERVED_MINIMUM_FARE, DEFAULT_POLICY_MAXIMUM],
        "estimand_units": "dollars per hard-trimmed trip",
        "specifications": summaries,
        "figure": str(figure_path),
        "limitations": [
            "January 2009 only",
            "deterministic 30,000-trip pilot subsample",
            "pilot-derived nuisance support [-6, 11]",
            "point estimates only; no application confidence band",
            "the $0.20 policy cost is illustrative, not measured",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost",
        type=float,
        default=0.20,
        help="Policy cost in dollars per trip assigned percentage suggestions",
    )
    args = parser.parse_args()
    payload = run(cost=args.cost)
    for label, result in payload["specifications"].items():
        print(
            f"{label}: phi*=${result['phi_star']:.2f}; "
            f"gain={result['gain_over_current_cents_per_trimmed_trip']:.3f} "
            "cents per hard-trimmed trip"
        )
    print(f"[wrote] {payload['figure']}")


if __name__ == "__main__":
    main()
