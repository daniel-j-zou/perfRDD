"""Empirical rank diagnostics for the restricted VTS differing-slopes design.

This is a finite-sample diagnostic for the primitive uniform-rank condition in
Appendix 4 of the prelim.  It cannot prove a population condition.  It reports
within-eta-bin treatment shares and conditional second-moment eigenvalues, the
condition number of the column-normalized theorem design, and the eigenvalues
of the D*X block after residualizing it on the spline nuisance columns.

Run from the ``code`` repository root::

    python -m experiments.scripts.taxi_differing_slopes_rank
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline

from experiments.datasets.taxi.adapter import load_haggag_paci_vendor


EPS = 0.10
THRESHOLD = 15.0
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "runs"
    / "taxi_differing_slopes_rank"
    / "summary.json"
)


def _clean(
    q: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same outcome-error screen as the headline taxi diagnostic."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(q > 0.0, y / q, np.nan)
    weird = np.isfinite(ratio) & (ratio > 1.0) & (y >= 10.0)
    keep = np.isfinite(ratio) & (q > 0.0) & (~weird)
    return q[keep], x[keep], y[keep]


def _spline_basis(
    values: np.ndarray, knots: int, support: tuple[float, float]
) -> np.ndarray:
    degree = 3
    lo, hi = support
    interior = np.linspace(lo, hi, knots + 2)[1:-1]
    knot_vector = np.concatenate(
        [np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)]
    )
    clipped = np.clip(values, lo, hi)
    return BSpline.design_matrix(clipped, knot_vector, degree).toarray()


def compute_diagnostics(bins: int = 10) -> dict[str, Any]:
    sample = load_haggag_paci_vendor("VTS")
    q, x, y = _clean(
        np.asarray(sample.Q, dtype=float),
        np.asarray(sample.X, dtype=float),
        np.asarray(sample.Y, dtype=float),
    )
    n = len(q)
    x_first = np.column_stack((np.ones(n), x))
    gamma = np.linalg.lstsq(x_first, q, rcond=None)[0]
    eta = q - x_first @ gamma
    t_index = x_first @ gamma
    treatment = (q >= THRESHOLD).astype(float)

    endpoint_a = THRESHOLD - np.quantile(t_index, 1.0 - EPS)
    endpoint_b = THRESHOLD - np.quantile(t_index, EPS)
    support = tuple(float(v) for v in np.percentile(eta, [0.5, 99.5]))
    lower = max(min(endpoint_a, endpoint_b), support[0])
    upper = min(max(endpoint_a, endpoint_b), support[1])
    trimmed = (eta >= lower) & (eta <= upper)

    edges = np.quantile(eta[trimmed], np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1e-10
    edges[-1] += 1e-10
    bin_results: list[dict[str, Any]] = []
    for index in range(bins):
        in_bin = trimmed & (eta > edges[index]) & (eta <= edges[index + 1])
        result: dict[str, Any] = {
            "bin": index + 1,
            "eta_lower": float(edges[index]),
            "eta_upper": float(edges[index + 1]),
            "treatment_share": float(np.mean(treatment[in_bin])),
        }
        for d, label in ((0.0, "control"), (1.0, "treated")):
            cell = in_bin & (treatment == d)
            design = np.column_stack((np.ones(int(cell.sum())), x[cell]))
            moment = design.T @ design / int(cell.sum())
            eigenvalues = np.linalg.eigvalsh(moment)
            result[f"n_{label}"] = int(cell.sum())
            result[f"minimum_eigenvalue_{label}"] = float(eigenvalues[0])
            result[f"condition_number_{label}"] = float(
                eigenvalues[-1] / eigenvalues[0]
            )
        bin_results.append(result)

    knot_count = max(4, int(round(int(treatment.sum()) ** (1.0 / 5.0)))) + 1
    spline = _spline_basis(eta, knot_count, support)
    augmented = np.column_stack(
        (treatment[:, None] * spline, spline, x, treatment[:, None] * x)
    )
    rms = np.sqrt(np.mean(augmented**2, axis=0))
    singular_values = np.linalg.svd(augmented / rms, compute_uv=False)

    nuisance = np.column_stack((treatment[:, None] * spline, spline, x))
    interaction = treatment[:, None] * x
    projection = np.linalg.lstsq(nuisance, interaction, rcond=None)[0]
    residualized_interaction = interaction - nuisance @ projection
    residualized_gram = residualized_interaction.T @ residualized_interaction / n
    residualized_eigenvalues = np.linalg.eigvalsh(residualized_gram)

    # The current ridge diagnostic includes a separate intercept even though
    # the baseline B-spline basis sums to one.  This is harmless only because
    # the spline coefficients are penalized; it must not be copied into the
    # theorem-facing unregularized implementation.
    ridge_script_design = np.column_stack(
        (np.ones(n), x, interaction, spline, treatment[:, None] * spline)
    )
    ridge_script_singular_values = np.linalg.svd(
        ridge_script_design, compute_uv=False
    )

    return {
        "sample": "paper-restricted VTS",
        "n": n,
        "p": int(x.shape[1]),
        "n_treated": int(treatment.sum()),
        "n_trimmed": int(trimmed.sum()),
        "trim_interval": [float(lower), float(upper)],
        "eta_bins": bin_results,
        "minimum_treatment_share_across_bins": float(
            min(row["treatment_share"] for row in bin_results)
        ),
        "minimum_cell_count_across_bins": int(
            min(
                min(row["n_control"], row["n_treated"])
                for row in bin_results
            )
        ),
        "minimum_conditional_moment_eigenvalue_across_bins": float(
            min(
                min(
                    row["minimum_eigenvalue_control"],
                    row["minimum_eigenvalue_treated"],
                )
                for row in bin_results
            )
        ),
        "spline_knot_count": knot_count,
        "spline_basis_dimension": int(spline.shape[1]),
        "column_normalized_augmented_design_condition_number": float(
            singular_values[0] / singular_values[-1]
        ),
        "residualized_interaction_gram_eigenvalues": [
            float(value) for value in residualized_eigenvalues
        ],
        "residualized_interaction_condition_number": float(
            residualized_eigenvalues[-1] / residualized_eigenvalues[0]
        ),
        "ridge_script_unregularized_condition_number": float(
            ridge_script_singular_values[0] / ridge_script_singular_values[-1]
        ),
        "spline_partition_of_unity_error": float(
            np.max(np.abs(np.sum(spline, axis=1) - 1.0))
        ),
        "interpretation": (
            "Finite-bin diagnostics support local augmented rank in this sample, "
            "but do not prove the uniform population condition. The theorem-facing "
            "unregularized design must omit the separate intercept because the "
            "baseline spline basis spans constants."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    result = compute_diagnostics(args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
