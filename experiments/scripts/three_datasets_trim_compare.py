"""Compare the standard `perfrdd` estimator with the trimmed `perfrdd_trim`
estimator on the three linear-Q datasets (GPA, NHANES, OULAD, Lending Club).

Each estimator is run separately and its own per-dataset summary and figures
are saved under experiments/runs/perfrdd/<name>/ and
experiments/runs/perfrdd_trim/<name>/ respectively. This script then
produces a single combined comparison figure
experiments/runs/three_datasets_trim_compare.png plus a side-by-side
summary JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from experiments._core.registry import load
from experiments.methods.perfrdd import perfrdd, _eval_basis as _eval_basis_std
from experiments.methods.perfrdd_trim import perfrdd_trim


ROOT = Path(__file__).resolve().parent.parent
RUNS_STD = ROOT / "runs" / "perfrdd"
RUNS_TRIM = ROOT / "runs" / "perfrdd_trim"
OUT_DIR = ROOT / "runs" / "trim_compare"
OUT_FIG = ROOT / "runs" / "three_datasets_trim_compare.png"
OUT_JSON = ROOT / "runs" / "three_datasets_trim_compare.json"

DATASETS = [
    ("gpa", "GPA — academic probation"),
    ("nhanes", "NHANES — HbA1c diabetic cutoff"),
    ("lending_club", "Lending Club — DTI trigger"),
]

EPS = 0.1


def _alpha_curve(summary_json: Path, kind: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Re-fit and evaluate alpha(eta) curve. We don't persist the fit object, so we
    re-run the estimator quickly from a saved sample. Instead, since the standard
    estimator's alpha is plotted in runs/<name>/alpha.png and the trimmed
    estimator's in runs/perfrdd_trim/<name>/alpha.png, we just stash a numerical
    dump from each run for plotting overlays.
    """
    raise NotImplementedError  # see below; we instead extract during the run


def _run_both(name: str) -> Dict[str, Any]:
    """Run both estimators on the dataset; return dict with both summaries
    plus the alpha-grid evaluations we will overlay.

    Both estimators are evaluated on the SAME absolute cost grid, calibrated
    to the trimmed estimator's in-window avg alpha (the trimmed value is the
    more reliable scale; the standard avg alpha is often biased toward 0 by
    the no-overlap region, which would give a misleadingly small grid).
    """
    sample = load(name)

    # Run trim first to get the in-window avg alpha, then build a shared
    # cost grid {0, 0.5, 1.0, 1.5} * |trim avg alpha| and pass it to both.
    trim_out = RUNS_TRIM / name
    res_trim_probe = perfrdd_trim(sample, trim_out, eps=EPS)
    a = abs(res_trim_probe["avg_alpha_for_c"])
    if a < 1e-12:
        a = abs(res_trim_probe["avg_alpha_trimmed"]) or 1.0
    shared_costs = (0.0, round(0.5 * a, 4), round(1.0 * a, 4), round(1.5 * a, 4))

    std_out = RUNS_STD / name
    res_std = perfrdd(sample, std_out, c_values=shared_costs)
    std_curve = _alpha_grid_from_perfrdd(sample, std_out)

    res_trim = perfrdd_trim(sample, trim_out, eps=EPS, c_values=shared_costs)
    trim_curve = _alpha_grid_from_perfrdd_trim(sample, trim_out)

    return {
        "name": name,
        "standard": res_std,
        "trimmed": res_trim,
        "alpha_std": std_curve,
        "alpha_trim": trim_curve,
    }


def _alpha_grid_from_perfrdd(sample, out_dir):
    """Re-evaluate the standard estimator's alpha on a dense grid. We re-run
    the fit because alpha curve points aren't persisted."""
    from experiments.methods.perfrdd import (
        _reduce_to_primary_axis, _detect_direction, _subsample, _fit_pooled_plm,
        _eval_basis, DEFAULT_MAX_N
    )
    Q, X, _ = _reduce_to_primary_axis(sample)
    Y, D = sample.Y, sample.D
    direction = _detect_direction(D, Q)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if len(Y) > DEFAULT_MAX_N:
        (Q, X, Y, D), _, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)
    fit = _fit_pooled_plm(Q, X, Y, D, direction)
    lo, hi = fit.eta_eval
    grid = np.linspace(lo, hi, 400)
    Phi = _eval_basis(grid, fit.info)
    alpha = Phi @ fit.omega_treat
    return {"grid": grid.tolist(), "alpha": alpha.tolist(),
            "lo": float(lo), "hi": float(hi)}


def _alpha_grid_from_perfrdd_trim(sample, out_dir):
    """Re-evaluate the trimmed estimator's alpha on a dense grid."""
    from experiments.methods.perfrdd_trim import (
        _compute_overlap_window, _fit_pooled_plm_trimmed,
    )
    from experiments.methods.perfrdd import (
        _reduce_to_primary_axis, _detect_direction, _subsample, _eval_basis,
        DEFAULT_MAX_N,
    )
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y, D = sample.Y, sample.D
    direction = _detect_direction(D, Q)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if len(Y) > DEFAULT_MAX_N:
        (Q, X, Y, D), _, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)
    l_hat, u_hat, eta_hat, _ = _compute_overlap_window(Q, X, threshold, EPS)
    fit = _fit_pooled_plm_trimmed(Q, X, Y, D, direction, l_hat, u_hat, eta_hat)
    grid = np.linspace(l_hat, u_hat, 400)
    Phi = _eval_basis(grid, fit.info)
    alpha = Phi @ fit.omega_treat
    return {"grid": grid.tolist(), "alpha": alpha.tolist(),
            "lo": float(l_hat), "hi": float(u_hat)}


def _plot_alpha_overlay(ax, std_curve: dict, trim_curve: dict, title: str) -> None:
    g_s = np.array(std_curve["grid"])
    a_s = np.array(std_curve["alpha"])
    g_t = np.array(trim_curve["grid"])
    a_t = np.array(trim_curve["alpha"])
    ax.plot(g_s, a_s, color="C0", lw=1.8, label="standard", alpha=0.85)
    ax.plot(g_t, a_t, color="C3", lw=2.2, label="trimmed (ε=0.1)")
    ax.axvspan(trim_curve["lo"], trim_curve["hi"], color="C3", alpha=0.06,
               label=fr"$[\hat l, \hat u]$")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\alpha(\eta)$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")


def _plot_utility_overlay(ax, summary_std: dict, summary_trim: dict,
                          threshold: float, title: str) -> None:
    """Plot utility curves at c=0 for both estimators. We don't have the raw
    utility arrays in the summary, so we re-evaluate using the persisted PNG
    paths — actually, easier: just label the phi* points on the x-axis."""
    # The summary stores phi_star at multiple cost values. Use the smallest
    # cost (typically 0) and a couple of higher ones.
    cs_s = sorted(summary_std["phi_star"].keys(), key=float)
    cs_t = sorted(summary_trim["phi_star"].keys(), key=float)
    # Plot phi* vs cost for both methods as scatter / line.
    xs_s = [float(c) for c in cs_s]
    ys_s = [summary_std["phi_star"][c] for c in cs_s]
    xs_t = [float(c) for c in cs_t]
    ys_t = [summary_trim["phi_star"][c] for c in cs_t]
    ax.plot(xs_s, ys_s, "o-", color="C0", lw=1.6, ms=6, label="standard")
    ax.plot(xs_t, ys_t, "s-", color="C3", lw=1.6, ms=6, label="trimmed (ε=0.1)")
    ax.axhline(threshold, color="black", ls=":", lw=1, label=f"current φ={threshold:.3g}")
    ax.set_xlabel("cost c")
    ax.set_ylabel(r"$\phi^*$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")


def _stats_block(s_std: dict, s_trim: dict) -> str:
    n_used = s_std["n_used"]
    n_tr = s_std["n_treated"]
    n_in = s_trim["n_in_window"]
    n_tr_in = s_trim["n_treated_in_window"]
    prop_in = s_trim["propensity_in_window"]
    return (
        f"n = {n_used:,}, treated = {n_tr:,}\n"
        f"first-stage R²  = {s_std['first_stage_R2']:.3f}\n"
        f"\n"
        f"standard:\n"
        f"  eta_eval = [{s_std['eta_eval'][0]:.3g}, {s_std['eta_eval'][1]:.3g}]\n"
        f"  avg α    = {s_std['avg_alpha']:.3g}\n"
        f"\n"
        f"trimmed (ε=0.1):\n"
        f"  [l̂, û] = [{s_trim['l_hat']:.3g}, {s_trim['u_hat']:.3g}]\n"
        f"  n_in   = {n_in:,} ({n_in/n_used:.1%})\n"
        f"  treated_in = {n_tr_in:,} (prop. {prop_in:.3f})\n"
        f"  avg α  = {s_trim['avg_alpha_trimmed']:.3g}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for name, _title in DATASETS:
        print(f"[run] {name} ...")
        results[name] = _run_both(name)
        print(f"      done")

    # Save JSON (drop the alpha curves from the JSON to keep it small;
    # those are persisted as numpy npz alongside the figure).
    json_safe = {
        name: {
            "standard": r["standard"],
            "trimmed": r["trimmed"],
        }
        for name, r in results.items()
    }
    OUT_JSON.write_text(json.dumps(json_safe, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    # Combined figure.
    n_rows = len(DATASETS)
    fig = plt.figure(figsize=(18, 3.8 * n_rows))
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=3,
        width_ratios=[1.0, 2.5, 1.8],
        hspace=0.45, wspace=0.22,
        left=0.04, right=0.99, top=0.94, bottom=0.04,
    )
    fig.suptitle(
        "Standard vs trimmed estimator — three linear-Q datasets (ε = 0.1)",
        fontsize=15, fontweight="bold", y=0.985,
    )

    for i, (name, title) in enumerate(DATASETS):
        r = results[name]
        s_std = r["standard"]
        s_trim = r["trimmed"]

        # Stats panel.
        ax_txt = fig.add_subplot(gs[i, 0])
        ax_txt.axis("off")
        ax_txt.text(0.02, 0.96, title, transform=ax_txt.transAxes,
                    fontsize=11, fontweight="bold", va="top")
        ax_txt.text(0.02, 0.84, _stats_block(s_std, s_trim),
                    transform=ax_txt.transAxes, fontsize=9.0,
                    family="monospace", va="top")

        # Alpha overlay.
        ax_a = fig.add_subplot(gs[i, 1])
        _plot_alpha_overlay(ax_a, r["alpha_std"], r["alpha_trim"],
                            rf"{name}: $\alpha(\eta)$ — standard vs trimmed")

        # phi* vs cost overlay.
        ax_p = fig.add_subplot(gs[i, 2])
        _plot_utility_overlay(ax_p, s_std, s_trim, s_std["threshold_actual"],
                              rf"{name}: $\phi^*$ vs cost")

    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
