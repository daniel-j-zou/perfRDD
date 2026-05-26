"""Bootstrap confidence intervals for the trimmed estimator across ε on each
of the four linear-Q datasets.

For each (dataset, ε): draw B bootstrap resamples with replacement from the
working sample (after the subsampling cap and NaN drop), run perfrdd_trim on
each, record φ*. Plot median + IQR (and 2.5%/97.5%) bands as a function of ε,
alongside the bootstrap CI for the standard estimator's φ*.

Designed to clarify whether the wild OULAD oscillations in
four_datasets_trim_eps_sweep are consistent with sampling noise.
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
from experiments.methods.perfrdd import (
    perfrdd, _reduce_to_primary_axis, _subsample, DEFAULT_MAX_N,
)
from experiments.methods.perfrdd_trim import perfrdd_trim
from experiments._core.sample import RDDSample


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_RUNS = ROOT / "runs" / "perfrdd_trim_bootstrap"
OUT_FIG = ROOT / "runs" / "four_datasets_trim_bootstrap.png"
OUT_JSON = ROOT / "runs" / "four_datasets_trim_bootstrap.json"

DATASETS = [
    ("gpa", "GPA — academic probation"),
    ("nhanes", "NHANES — HbA1c diabetic cutoff"),
    ("oulad", "OULAD — first-TMA pass mark"),
    ("lending_club", "Lending Club — DTI trigger"),
]

EPS_GRID = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
B = 30  # bootstrap reps per (dataset, ε)


def _sample_from(sample: RDDSample, idx: np.ndarray) -> RDDSample:
    """Build a new RDDSample from the given row indices."""
    return RDDSample(
        Q=sample.Q[idx],
        X=sample.X[idx],
        Y=sample.Y[idx],
        threshold=sample.threshold,
        name=sample.name,
        feature_names=sample.feature_names,
        description=sample.description,
        treatment_rule=sample.treatment_rule,
    )


def _prepare(sample: RDDSample):
    """Apply NaN-drop and cap-subsample once, then return a fixed-size sample
    for bootstrapping. Also return cost grid + phi grid from the standard fit."""
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y, D = sample.Y, sample.D
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if len(Y) > DEFAULT_MAX_N:
        (Q, X, Y, D), n_used, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)
    else:
        n_used = len(Y)
    base_sample = RDDSample(
        Q=Q, X=X, Y=Y, threshold=threshold, name=sample.name,
        feature_names=sample.feature_names, description=sample.description,
        treatment_rule=sample.treatment_rule,
    )
    # Phi grid from std(Q).
    phi_span = 3.0 * float(Q.std())
    phi_grid = np.linspace(threshold - phi_span, threshold + phi_span, 400)

    # Standard fit on the prepared sample to fix cost grid.
    out_std = OUT_DIR_RUNS / sample.name / "_std_main"
    res_std_main = perfrdd(base_sample, out_std, phi_grid=phi_grid)
    cost_grid = tuple(float(c) for c in res_std_main["phi_star"].keys())
    return base_sample, cost_grid, phi_grid, res_std_main


def _bootstrap_for(name: str) -> Dict[str, Any]:
    sample = load(name)
    base, costs, phi_grid, res_std_main = _prepare(sample)
    n = len(base.Y)
    rng = np.random.default_rng(0)

    # Standard bootstrap (B reps).
    std_boot: Dict[float, List[float]] = {c: [] for c in costs}
    print(f"  [{name}] standard bootstrap, B={B}")
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sb = _sample_from(base, idx)
        try:
            r = perfrdd(sb, OUT_DIR_RUNS / name / "_std_boot",
                        c_values=costs, phi_grid=phi_grid, max_n=None)
            for c in costs:
                std_boot[c].append(float(r["phi_star"][str(c)]))
        except Exception:
            pass

    # Trimmed bootstrap per ε.
    trim_boot: Dict[float, Dict[float, List[float]]] = {
        eps: {c: [] for c in costs} for eps in EPS_GRID
    }
    retention_boot: Dict[float, List[float]] = {eps: [] for eps in EPS_GRID}
    for eps in EPS_GRID:
        print(f"  [{name}] trimmed bootstrap ε={eps}, B={B}")
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            sb = _sample_from(base, idx)
            try:
                r = perfrdd_trim(sb, OUT_DIR_RUNS / name / f"_eps_{eps:.3f}_boot",
                                 eps=eps, c_values=costs, phi_grid=phi_grid,
                                 max_n=None)
                for c in costs:
                    trim_boot[eps][c].append(float(r["phi_star"][str(c)]))
                retention_boot[eps].append(r["n_in_window"] / r["n_used"])
            except Exception:
                pass

    return {
        "name": name,
        "n": int(n),
        "B": B,
        "cost_grid": list(costs),
        "eps_grid": EPS_GRID,
        "phi_star_std_full": res_std_main["phi_star"],
        "standard_boot": {str(c): std_boot[c] for c in costs},
        "trimmed_boot": {
            str(eps): {str(c): trim_boot[eps][c] for c in costs}
            for eps in EPS_GRID
        },
        "retention_boot": {str(eps): retention_boot[eps] for eps in EPS_GRID},
    }


def _summary_band(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": float("nan"), "q025": float("nan"), "q975": float("nan"),
                "q25": float("nan"), "q75": float("nan")}
    a = np.array(values)
    return {
        "median": float(np.median(a)),
        "q025": float(np.percentile(a, 2.5)),
        "q975": float(np.percentile(a, 97.5)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
    }


def _plot_panel(ax_main, ax_sec, r: Dict[str, Any], title: str) -> None:
    eps_grid = np.array(r["eps_grid"])
    costs = r["cost_grid"]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.85, len(costs)))

    for c, col in zip(costs, colors):
        # Trimmed bands.
        meds = []
        q025 = []
        q975 = []
        for eps in eps_grid:
            s = _summary_band(r["trimmed_boot"][str(eps)][str(c)])
            meds.append(s["median"])
            q025.append(s["q025"])
            q975.append(s["q975"])
        meds = np.array(meds); q025 = np.array(q025); q975 = np.array(q975)
        ax_main.plot(eps_grid, meds, "o-", color=col, lw=2.0, ms=6,
                     label=fr"trim, $c={c:.3g}$")
        ax_main.fill_between(eps_grid, q025, q975, color=col, alpha=0.18)
        # Standard 95% CI band as a wide horizontal grey rectangle.
        s_std = _summary_band(r["standard_boot"][str(c)])
        ax_main.axhline(s_std["median"], color=col, ls="--", lw=1.4, alpha=0.65)

    ax_main.set_xscale("log")
    ax_main.set_ylabel(r"$\phi^*$", fontsize=11)
    ax_main.set_title(title, fontsize=12, fontweight="bold")
    ax_main.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    ax_main.tick_params(labelsize=9)
    ax_main.grid(alpha=0.25)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Retention strip (median + IQR).
    med = np.array([np.median(r["retention_boot"][str(eps)]) for eps in eps_grid])
    q25 = np.array([np.percentile(r["retention_boot"][str(eps)], 25) for eps in eps_grid])
    q75 = np.array([np.percentile(r["retention_boot"][str(eps)], 75) for eps in eps_grid])
    ax_sec.plot(eps_grid, med, "k^-", lw=1.5, ms=5)
    ax_sec.fill_between(eps_grid, q25, q75, color="black", alpha=0.15)
    ax_sec.set_xscale("log")
    ax_sec.set_xlabel(r"$\epsilon$", fontsize=11)
    ax_sec.set_ylabel("retained", fontsize=9)
    ax_sec.set_ylim(-0.02, 1.02)
    ax_sec.tick_params(labelsize=9)
    ax_sec.grid(alpha=0.25)


def main() -> None:
    OUT_DIR_RUNS.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for name, _ in DATASETS:
        print(f"[bootstrap] {name}")
        results[name] = _bootstrap_for(name)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    fig = plt.figure(figsize=(17, 12))
    gs_outer = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22,
                                left=0.06, right=0.98, top=0.91, bottom=0.05)
    fig.suptitle(
        rf"Bootstrap CIs for the trimmed estimator (B={B})",
        fontsize=15, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.945,
        r"solid markers: median trim $\phi^*(\epsilon)$ over B resamples with 95% band;"
        r"  dashed: median std $\phi^*$",
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
