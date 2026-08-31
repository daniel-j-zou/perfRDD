"""Diagnostic full-reestimation bootstrap for the 30,000-trip taxi application.

The bootstrap targets expected driver tip revenue, so its primary policy cost is
zero.  Every replication resamples trips and re-estimates the first-stage fare
residual, hard-trim endpoints, treatment-effect spline, empirical policy
probabilities, utility curve, and its argmax.  This is deliberately an i.i.d.
trip bootstrap: the public TLC file lacks the driver identifiers used in the
original paper, and January alone contains only 31 calendar-day clusters.

The analysis sample applies the published Haggag--Paci main-RDD restrictions
before taking the same deterministic 30,000-observation subsample used by the
existing pilot.  Outputs are diagnostic rather than publication-ready.

Example::

    python -m experiments.scripts.taxi_bootstrap_30k --replications 199
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Avoid oversubscribing BLAS inside multiprocessing workers.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.taxi.adapter import load_haggag_paci
from experiments.methods.perfrdd import _subsample
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "runs" / "taxi_bootstrap_30k"
ANALYSIS_N = 30_000
SUBSAMPLE_SEED = 0
BOOTSTRAP_SEED = 91_731
EPS = 0.1
RIDGE_SCALE = 0.001
NUISANCE_SUPPORT = (-6.0, 11.0)
CURRENT_THRESHOLD = 15.0
POLICY_GRID = np.linspace(2.5, 25.0, 451)
COST = 0.0

_WORKER_SAMPLE: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _percentage_regime(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q) >= CURRENT_THRESHOLD).astype(int)


def make_analysis_sample() -> RDDSample:
    """Return the locked deterministic 30,000-trip restricted sample."""
    source = load_haggag_paci()
    arrays, _, _ = _subsample(
        [source.Q, source.X, source.Y], ANALYSIS_N, seed=SUBSAMPLE_SEED
    )
    Q, X, Y = arrays
    return RDDSample(
        Q=Q,
        X=X,
        Y=Y,
        threshold=CURRENT_THRESHOLD,
        treatment_rule=_percentage_regime,
        name="taxi_haggag_paci_30k",
        feature_names=list(source.feature_names),
        description=source.description,
        citation=source.citation,
        extras={
            **source.extras,
            "analysis_n": ANALYSIS_N,
            "subsample_seed": SUBSAMPLE_SEED,
        },
    )


def centered_interval(
    estimate: float, bootstrap_values: np.ndarray, level: float = 0.95
) -> Tuple[float, float]:
    """Return a basic centered-bootstrap confidence interval."""
    alpha = 1.0 - level
    deviations = np.asarray(bootstrap_values, dtype=float) - float(estimate)
    lower_deviation, upper_deviation = np.quantile(
        deviations, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    return float(estimate - upper_deviation), float(estimate - lower_deviation)


def simultaneous_relative_band(
    estimate: np.ndarray,
    bootstrap_curves: np.ndarray,
    current_index: int,
    level: float = 0.95,
) -> Dict[str, np.ndarray | float]:
    """Construct pointwise and max-standardized bands relative to current policy."""
    estimate_relative = estimate - estimate[current_index]
    bootstrap_relative = (
        bootstrap_curves - bootstrap_curves[:, [current_index]]
    )
    deviations = bootstrap_relative - estimate_relative[None, :]
    standard_error = deviations.std(axis=0, ddof=1)
    positive = standard_error > 1e-12
    max_statistics = np.zeros(len(bootstrap_curves))
    if positive.any():
        max_statistics = np.max(
            np.abs(deviations[:, positive] / standard_error[None, positive]),
            axis=1,
        )
    critical_value = float(np.quantile(max_statistics, level))
    alpha = 1.0 - level
    pointwise_lower, pointwise_upper = np.quantile(
        bootstrap_relative, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0
    )
    return {
        "estimate": estimate_relative,
        "pointwise_lower": pointwise_lower,
        "pointwise_upper": pointwise_upper,
        "simultaneous_lower": estimate_relative - critical_value * standard_error,
        "simultaneous_upper": estimate_relative + critical_value * standard_error,
        "standard_error": standard_error,
        "critical_value": critical_value,
    }


def _init_worker(Q: np.ndarray, X: np.ndarray, Y: np.ndarray) -> None:
    global _WORKER_SAMPLE
    _WORKER_SAMPLE = (Q, X, Y)


def _bootstrap_replication(seed: int) -> Dict[str, Any]:
    if _WORKER_SAMPLE is None:
        raise RuntimeError("bootstrap worker was not initialized")
    Q, X, Y = _WORKER_SAMPLE
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(Y), size=len(Y))
    sample = RDDSample(
        Q=Q[index],
        X=X[index],
        Y=Y[index],
        threshold=CURRENT_THRESHOLD,
        treatment_rule=_percentage_regime,
        name="taxi_bootstrap_replication",
        feature_names=[f"x{j}" for j in range(X.shape[1])],
    )
    try:
        result = perfrdd_hard_trim(
            sample,
            OUT_ROOT / "unused_worker_output",
            NUISANCE_SUPPORT,
            eps=EPS,
            c_values=(COST,),
            phi_grid=POLICY_GRID,
            max_n=None,
            ridge_scale=RIDGE_SCALE,
            crossfit_folds=1,
            write_outputs=False,
            return_curves=True,
        )
        curve = result["returned_utility_curves"][str(COST)]
        return {
            "seed": int(seed),
            "curve": curve,
            "phi_star": float(result["phi_star"][str(COST)]),
            "avg_alpha_hard_weighted": float(
                result["avg_alpha_hard_weighted"]
            ),
            "hard_retention": float(result["hard_retention"]),
            "design_condition_number": float(
                result["fold_diagnostics"][0]["design_condition_number"]
            ),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - recorded for long runs
        return {"seed": int(seed), "error": f"{type(exc).__name__}: {exc}"}


def _plot_results(
    bands: Dict[str, np.ndarray | float],
    gains_cents: np.ndarray,
    gain_estimate_cents: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = POLICY_GRID
    estimate = 100.0 * np.asarray(bands["estimate"])
    pointwise_lower = 100.0 * np.asarray(bands["pointwise_lower"])
    pointwise_upper = 100.0 * np.asarray(bands["pointwise_upper"])
    simultaneous_lower = 100.0 * np.asarray(bands["simultaneous_lower"])
    simultaneous_upper = 100.0 * np.asarray(bands["simultaneous_upper"])

    axes[0].fill_between(
        x,
        simultaneous_lower,
        simultaneous_upper,
        color="#9ecae1",
        alpha=0.35,
        label="95% simultaneous band",
    )
    axes[0].fill_between(
        x,
        pointwise_lower,
        pointwise_upper,
        color="#3182bd",
        alpha=0.25,
        label="95% pointwise interval",
    )
    axes[0].plot(x, estimate, color="#08519c", linewidth=2.1)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].axvline(
        CURRENT_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Current $15 threshold",
    )
    axes[0].set_title("Driver-tip utility relative to current policy")
    axes[0].set_xlabel("Fare threshold for percentage suggestions ($)")
    axes[0].set_ylabel("Change in expected tips\n(cents per hard-trimmed trip)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(gains_cents, bins=22, color="#6baed6", edgecolor="white")
    axes[1].axvline(
        gain_estimate_cents,
        color="#cb181d",
        linewidth=2.0,
        label=f"Estimate: {gain_estimate_cents:.2f} cents",
    )
    axes[1].set_title("Bootstrap gain from percentage suggestions on all fares")
    axes[1].set_xlabel("Gain over current $15 policy (cents per trimmed trip)")
    axes[1].set_ylabel("Bootstrap replications")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def run(replications: int = 199, workers: int | None = None) -> Dict[str, Any]:
    """Run the diagnostic bootstrap and write machine-readable inference."""
    if replications < 39:
        raise ValueError("use at least 39 bootstrap replications")
    if workers is None:
        workers = max(1, min(6, (os.cpu_count() or 2) - 1))
    if workers < 1:
        raise ValueError("workers must be positive")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sample = make_analysis_sample()
    baseline = perfrdd_hard_trim(
        sample,
        OUT_ROOT / "baseline",
        NUISANCE_SUPPORT,
        eps=EPS,
        c_values=(COST,),
        phi_grid=POLICY_GRID,
        max_n=None,
        ridge_scale=RIDGE_SCALE,
        crossfit_folds=1,
        return_curves=True,
    )
    estimate = np.asarray(
        baseline["returned_utility_curves"][str(COST)], dtype=float
    )
    seeds = [BOOTSTRAP_SEED + index for index in range(replications)]
    context = mp.get_context("spawn")
    records: List[Dict[str, Any]] = []
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(sample.Q, sample.X, sample.Y),
    ) as pool:
        for completed, record in enumerate(
            pool.imap_unordered(_bootstrap_replication, seeds), start=1
        ):
            records.append(record)
            if completed % 25 == 0 or completed == replications:
                print(f"[bootstrap] completed {completed}/{replications}")

    failures = [record for record in records if record["error"] is not None]
    successful = [record for record in records if record["error"] is None]
    if len(successful) < max(39, int(0.9 * replications)):
        raise RuntimeError(
            f"only {len(successful)} of {replications} replications succeeded"
        )
    successful.sort(key=lambda record: record["seed"])
    curves = np.asarray([record["curve"] for record in successful], dtype=float)
    phi_star = np.asarray([record["phi_star"] for record in successful])
    current_index = int(np.argmin(np.abs(POLICY_GRID - CURRENT_THRESHOLD)))
    all_fares_index = 0
    bands = simultaneous_relative_band(estimate, curves, current_index)

    gain_estimate = float(estimate[all_fares_index] - estimate[current_index])
    bootstrap_gains = curves[:, all_fares_index] - curves[:, current_index]
    gain_interval = centered_interval(gain_estimate, bootstrap_gains)
    gain_estimate_cents = 100.0 * gain_estimate
    gains_cents = 100.0 * bootstrap_gains

    bands_frame = pd.DataFrame({
        "phi": POLICY_GRID,
        "relative_utility": bands["estimate"],
        "pointwise_lower": bands["pointwise_lower"],
        "pointwise_upper": bands["pointwise_upper"],
        "simultaneous_lower": bands["simultaneous_lower"],
        "simultaneous_upper": bands["simultaneous_upper"],
        "bootstrap_standard_error": bands["standard_error"],
    })
    bands_frame.to_csv(OUT_ROOT / "utility_bands.csv", index=False)
    records_frame = pd.DataFrame([{
        key: value for key, value in record.items() if key not in {"curve", "error"}
    } for record in successful])
    records_frame["gain_all_fares_over_current"] = bootstrap_gains
    records_frame.to_csv(OUT_ROOT / "bootstrap_estimates.csv", index=False)
    np.savez_compressed(
        OUT_ROOT / "bootstrap_curves.npz",
        phi=POLICY_GRID,
        estimate=estimate,
        bootstrap_curves=curves,
    )
    figure_path = OUT_ROOT / "utility_curve_bootstrap.png"
    _plot_results(bands, gains_cents, gain_estimate_cents, figure_path)

    payload: Dict[str, Any] = {
        "description": (
            "Diagnostic iid full-reestimation bootstrap for expected driver tip "
            "revenue on the restricted January 2009 30,000-trip taxi sample"
        ),
        "confirmatory": False,
        "bootstrap_scope": (
            "iid trips from the locked 30,000-trip analysis sample; not clustered "
            "by driver or calendar block"
        ),
        "sample_description": sample.description,
        "source_rows_after_paper_restrictions": sample.extras[
            "source_rows_after_paper_restrictions"
        ],
        "analysis_n": sample.n,
        "subsample_seed": SUBSAMPLE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "replications_requested": replications,
        "replications_successful": len(successful),
        "replications_failed": len(failures),
        "failure_messages": [record["error"] for record in failures],
        "workers": workers,
        "eps": EPS,
        "ridge_scale": RIDGE_SCALE,
        "nuisance_support": list(NUISANCE_SUPPORT),
        "policy_cost": COST,
        "policy_grid": [
            float(POLICY_GRID[0]), float(POLICY_GRID[-1]), len(POLICY_GRID)
        ],
        "current_threshold": CURRENT_THRESHOLD,
        "baseline": {
            "phi_star": float(baseline["phi_star"][str(COST)]),
            "phi_star_at_boundary": bool(
                baseline["phi_star_at_grid_boundary"][str(COST)]
            ),
            "avg_alpha_hard_weighted": float(
                baseline["avg_alpha_hard_weighted"]
            ),
            "hard_retention": float(baseline["hard_retention"]),
            "design_condition_number": float(
                baseline["fold_diagnostics"][0]["design_condition_number"]
            ),
            "gain_all_fares_over_current_cents_per_trimmed_trip": (
                gain_estimate_cents
            ),
        },
        "bootstrap_inference": {
            "phi_star_percentile_95": [
                float(value) for value in np.quantile(phi_star, [0.025, 0.975])
            ],
            "phi_star_boundary_share": float(
                np.mean(np.isclose(phi_star, POLICY_GRID[0]))
            ),
            "gain_all_fares_over_current_centered_95_cents": [
                100.0 * gain_interval[0], 100.0 * gain_interval[1]
            ],
            "gain_bootstrap_mean_cents": float(np.mean(gains_cents)),
            "gain_bootstrap_standard_error_cents": float(
                np.std(gains_cents, ddof=1)
            ),
            "simultaneous_band_critical_value": float(bands["critical_value"]),
        },
        "limitations": [
            "January 2009 only",
            "deterministic 30,000-trip subsample",
            "iid trip bootstrap rather than driver-cluster bootstrap",
            "public TLC file lacks the paper's anonymized driver and car identifiers",
            "pilot-derived nuisance support [-6, 11]",
            "high nuisance-design condition number remains a numerical warning",
            "conventional local-RD replication has not yet been reconciled",
            "boundary argmax makes an ordinary symmetric threshold interval unsuitable",
        ],
        "outputs": {
            "figure": str(figure_path),
            "utility_bands": str(OUT_ROOT / "utility_bands.csv"),
            "bootstrap_estimates": str(OUT_ROOT / "bootstrap_estimates.csv"),
            "bootstrap_curves": str(OUT_ROOT / "bootstrap_curves.npz"),
        },
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[wrote] {OUT_ROOT / 'summary.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=199)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    payload = run(args.replications, args.workers)
    baseline = payload["baseline"]
    inference = payload["bootstrap_inference"]
    print(
        f"phi*=${baseline['phi_star']:.2f}; boundary share="
        f"{inference['phi_star_boundary_share']:.3f}; all-fares gain="
        f"{baseline['gain_all_fares_over_current_cents_per_trimmed_trip']:.3f} cents"
    )


if __name__ == "__main__":
    main()
