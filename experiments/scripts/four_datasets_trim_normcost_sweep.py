"""Finer ε sweep with the cost grid calibrated to each estimator's OWN average
α (via relative cost ratios ρ = c / |avg α|), rather than forcing the standard
estimator's absolute cost grid onto the trimmed estimator.

Motivation: the auto cost grid is c ∈ {0, 0.5, 1, 1.5}·|avg α|. If you calibrate
to the STANDARD avg α and the standard α is biased toward zero by the
no-overlap region (e.g. NHANES, where std avg α ≈ −0.08 but trimmed avg α ≈ 2),
the resulting costs are negligible relative to the trimmed benefit, and the
trimmed φ*(c) collapses onto a single curve. Calibrating each estimator to its
own α restores the trimmed estimator's true cost sensitivity while keeping the
two comparable through the shared *relative* cost ρ.

Per panel (one per dataset):
  - solid markers: trimmed φ*(ε) at relative cost ρ_k = c/|trim avg α(ε)|.
  - dashed: standard φ* at the SAME ρ_k = c/|std avg α| (ε-independent).
  - legend labels the ratio ρ_k.

Output:
  experiments/runs/four_datasets_trim_normcost_sweep.png
  experiments/runs/four_datasets_trim_normcost_sweep.json
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


ROOT = Path(__file__).resolve().parent.parent
RUNS_STD = ROOT / "runs" / "perfrdd"
OUT_FIG = ROOT / "runs" / "four_datasets_trim_normcost_sweep.png"
OUT_JSON = ROOT / "runs" / "four_datasets_trim_normcost_sweep.json"

DATASETS = [
    ("gpa", "GPA — academic probation"),
    ("nhanes", "NHANES — HbA1c diabetic cutoff"),
    ("oulad", "OULAD — first-TMA pass mark"),
    ("lending_club", "Lending Club — DTI trigger"),
]

EPS_GRID = [0.20, 0.15, 0.10, 0.075, 0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005]
RATIOS = (0.0, 0.5, 1.0, 1.5)   # relative cost ρ = c / |avg α|, shared by both methods


def _sweep(name: str) -> Dict[str, Any]:
    sample = load(name)
    # Standard auto-calibrates its own cost grid to |std avg α| with the same
    # ratios (perfrdd._auto_c_values uses 0, 0.5, 1, 1.5).
    res_std = perfrdd(sample, RUNS_STD / name)
    std_costs_sorted = sorted((float(c) for c in res_std["phi_star"].keys()))
    # Map each ratio index k -> standard φ* (costs already in ratio order).
    std_phi_by_ratio = {
        RATIOS[k]: res_std["phi_star"][_match_key(res_std["phi_star"], std_costs_sorted[k])]
        for k in range(len(RATIOS))
    }

    Q, _, threshold = _reduce_to_primary_axis(sample)
    Y = sample.Y
    keep = np.isfinite(Q) & np.isfinite(Y)
    Q = Q[keep]
    if len(Q) > DEFAULT_MAX_N:
        (Q,), _, _ = _subsample([Q], DEFAULT_MAX_N)
    phi_span = 3.0 * float(Q.std())
    phi_grid = np.linspace(threshold - phi_span, threshold + phi_span, 600)

    sweep: List[Dict[str, Any]] = []
    for eps in EPS_GRID:
        out_dir = ROOT / "runs" / "perfrdd_trim_normcost" / name / f"eps_{eps:.4f}"
        try:
            r = perfrdd_trim(sample, out_dir, eps=eps,
                             c_ratios=RATIOS, phi_grid=phi_grid)
            # r["c_values"] is ordered to match RATIOS.
            phi_by_ratio = {
                RATIOS[k]: r["phi_star"][str(r["c_values"][k])]
                for k in range(len(RATIOS))
            }
            sweep.append({
                "eps": eps,
                "phi_by_ratio": {str(rt): phi_by_ratio[rt] for rt in RATIOS},
                "avg_alpha_trimmed": r["avg_alpha_trimmed"],
                "avg_alpha_for_c": r["avg_alpha_for_c"],
                "c_values": r["c_values"],
                "n_in_window": r["n_in_window"],
                "n_used": r["n_used"],
            })
        except Exception as e:
            print(f"  ε={eps}: FAILED {e!r}")
            sweep.append({"eps": eps, "error": repr(e)})

    return {
        "name": name,
        "threshold": float(res_std["threshold_actual"]),
        "ratios": list(RATIOS),
        "std_avg_alpha": res_std["avg_alpha"],
        "std_phi_by_ratio": {str(rt): std_phi_by_ratio[rt] for rt in RATIOS},
        "eps_grid": list(EPS_GRID),
        "sweep": sweep,
    }


def _match_key(d: Dict[str, float], val: float) -> str:
    """Find the dict string-key whose float value == val (within tol)."""
    for k in d:
        if abs(float(k) - val) < 1e-12:
            return k
    # Fallback: nearest.
    return min(d, key=lambda k: abs(float(k) - val))


def _plot_panel(ax_main, ax_sec, r: Dict[str, Any], title: str) -> None:
    eps_grid = np.array(r["eps_grid"])
    ratios = r["ratios"]
    phi0 = r["threshold"]
    std_phi = {float(rt): v for rt, v in r["std_phi_by_ratio"].items()}
    valid = [(eps_grid[i], s) for i, s in enumerate(r["sweep"]) if "error" not in s]
    if not valid:
        ax_main.text(0.5, 0.5, "all failed", ha="center", va="center")
        return

    cmap = plt.get_cmap("plasma")
    colors = cmap(np.linspace(0.1, 0.8, len(ratios)))

    for rt, col in zip(ratios, colors):
        phi_trim = np.array([s["phi_by_ratio"][str(rt)] for _, s in valid])
        eps_x = np.array([e for e, _ in valid])
        ax_main.plot(eps_x, phi_trim, "o-", color=col, lw=2.0, ms=6,
                     label=fr"$\rho={rt:g}$")
        ax_main.axhline(std_phi[rt], color=col, ls="--", lw=1.4, alpha=0.65)

    ax_main.axhline(phi0, color="black", ls=":", lw=1.2, label=fr"$\phi_0={phi0:.3g}$")
    ax_main.axvline(0.10, color="grey", ls="-", lw=0.7, alpha=0.45)
    ax_main.set_xscale("log")
    ax_main.set_ylabel(r"$\phi^*$", fontsize=11)
    ax_main.set_title(title, fontsize=12, fontweight="bold")
    ax_main.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2,
                   title=r"relative cost $\rho=c/|\bar\alpha|$", title_fontsize=8)
    ax_main.tick_params(labelsize=9)
    ax_main.grid(alpha=0.25)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Annotate the trimmed vs standard avg-α scales (the crux).
    smallest_eps, smallest_s = valid[-1]
    ax_main.annotate(
        fr"$\bar\alpha_{{\rm trim}}$={smallest_s['avg_alpha_for_c']:.3g},  "
        fr"$\bar\alpha_{{\rm std}}$={r['std_avg_alpha']:.3g}",
        xy=(0.03, 0.04), xycoords="axes fraction",
        fontsize=8.5, color="darkred",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkred", alpha=0.85),
    )

    eps_x = np.array([e for e, _ in valid])
    retention = np.array([s["n_in_window"] / s["n_used"] for _, s in valid])
    ax_sec.plot(eps_x, retention, "k^-", lw=1.5, ms=5)
    ax_sec.axvline(0.10, color="grey", ls="-", lw=0.7, alpha=0.45)
    ax_sec.set_xscale("log")
    ax_sec.set_xlabel(r"$\epsilon$", fontsize=11)
    ax_sec.set_ylabel("retained", fontsize=9)
    ax_sec.set_ylim(-0.02, 1.02)
    ax_sec.tick_params(labelsize=9)
    ax_sec.grid(alpha=0.25)


def main() -> None:
    results: Dict[str, Any] = {}
    for name, _ in DATASETS:
        print(f"[normcost] {name}")
        results[name] = _sweep(name)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    fig = plt.figure(figsize=(17, 12))
    gs_outer = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22,
                                left=0.06, right=0.98, top=0.91, bottom=0.05)
    fig.suptitle(
        r"Trimmed estimator with cost calibrated to its OWN $\bar\alpha$ "
        r"(relative cost $\rho=c/|\bar\alpha|$) — four datasets",
        fontsize=15, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.945,
        r"solid markers: trimmed $\phi^*(\epsilon)$ at relative cost $\rho$;  "
        r"dashed: standard $\phi^*$ at the same $\rho$;  "
        r"grey vertical: $\epsilon=0.10$;  black dotted: $\phi_0$",
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
