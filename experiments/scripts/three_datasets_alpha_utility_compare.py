"""Compare α(η) and the utility U(φ) between the standard and trimmed
estimators, across several ε values for the trimmed estimator and several
ABSOLUTE cost values c.

Layout: one row per dataset, four columns:
    [ α(η) ]  [ U(φ; c_1) ]  [ U(φ; c_2) ]  [ U(φ; c_3) ]

In every panel:
    - black solid  : standard estimator
    - colored solid: trimmed estimator at ε ∈ {0.05, 0.10, 0.20}
In the α column the trimmed overlap windows [l̂, û] are shaded (one per ε).
In the U columns the argmax φ* of each curve is marked with a vertical line.

The absolute cost values c_1<c_2<c_3 are fixed per dataset (computed once from
the trimmed avg α at ε=0.10 so they sit on a meaningful scale) and printed in
each U-panel title, so the axis is genuinely absolute.

Output:
  experiments/runs/three_datasets_alpha_utility_compare.png
  experiments/runs/three_datasets_alpha_utility_compare.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.registry import load
from experiments.methods.perfrdd import (
    _reduce_to_primary_axis, _detect_direction, _subsample, _fit_pooled_plm,
    _utility_curve, _eval_basis, DEFAULT_MAX_N,
)
from experiments.methods.perfrdd_trim import (
    _compute_overlap_window, _fit_pooled_plm_trimmed, _utility_curve_trimmed,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "runs" / "three_datasets_alpha_utility_compare.png"
OUT_JSON = ROOT / "runs" / "three_datasets_alpha_utility_compare.json"

DATASETS = [
    ("gpa", "GPA — academic probation"),
    ("nhanes", "NHANES — HbA1c diabetic cutoff"),
    ("lending_club", "Lending Club — DTI trigger"),
]

EPS_SHOWN = [0.05, 0.10, 0.20]
N_GRID_ALPHA = 400
N_GRID_PHI = 500
EPS_COLORS = {0.05: "#1f4e79", 0.10: "#2e8b57", 0.20: "#d1701a"}  # dark->light
STD_COLOR = "black"


def _prep(sample) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, np.ndarray]:
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y, D = sample.Y, sample.D
    direction = _detect_direction(D, Q)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if len(Y) > DEFAULT_MAX_N:
        (Q, X, Y, D), _, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)
    s = float(Q.std())
    phi_grid = np.linspace(threshold - 3 * s, threshold + 3 * s, N_GRID_PHI)
    return Q, X, Y, D, threshold, direction, phi_grid


def _compute(name: str) -> Dict[str, Any]:
    sample = load(name)
    Q, X, Y, D, threshold, direction, phi_grid = _prep(sample)

    # ---- standard fit ----
    std_fit = _fit_pooled_plm(Q, X, Y, D, direction)
    std_lo, std_hi = std_fit.eta_eval
    std_alpha_grid = np.linspace(std_lo, std_hi, N_GRID_ALPHA)
    std_alpha = _eval_basis(std_alpha_grid, std_fit.info) @ std_fit.omega_treat

    # ---- trimmed fits at each ε ----
    trim: Dict[float, Dict[str, Any]] = {}
    for eps in EPS_SHOWN:
        l_hat, u_hat, eta_hat, _ = _compute_overlap_window(Q, X, threshold, eps)
        fit = _fit_pooled_plm_trimmed(Q, X, Y, D, direction, l_hat, u_hat, eta_hat)
        a_grid = np.linspace(l_hat, u_hat, N_GRID_ALPHA)
        a_vals = _eval_basis(a_grid, fit.info) @ fit.omega_treat
        # in-window observation-weighted avg alpha (for cost calibration).
        in_w = (eta_hat >= l_hat) & (eta_hat <= u_hat)
        avg_a = float((_eval_basis(eta_hat[in_w], fit.info) @ fit.omega_treat).mean())
        trim[eps] = {
            "fit": fit, "l_hat": l_hat, "u_hat": u_hat, "eta_hat": eta_hat,
            "alpha_grid": a_grid, "alpha": a_vals, "avg_alpha": avg_a,
        }

    # ---- absolute cost grid (from trimmed avg α at ε=0.10) ----
    a_ref = abs(trim[0.10]["avg_alpha"]) if abs(trim[0.10]["avg_alpha"]) > 1e-9 else 1.0
    c_grid = (0.0, round(0.5 * a_ref, 4), round(1.0 * a_ref, 4))

    # ---- utility curves at the absolute costs ----
    std_U = _utility_curve(std_fit, Q, X, phi_grid, c_grid)
    for eps in EPS_SHOWN:
        trim[eps]["U"] = _utility_curve_trimmed(trim[eps]["fit"], Q, phi_grid, c_grid)

    def argmax_phi(u):
        return float(phi_grid[int(np.argmax(u))])

    return {
        "name": name,
        "threshold": float(threshold),
        "direction": direction,
        "phi_grid": phi_grid,
        "c_grid": c_grid,
        "std": {
            "alpha_grid": std_alpha_grid, "alpha": std_alpha,
            "U": std_U, "phi_star": {c: argmax_phi(std_U[c]) for c in c_grid},
            "eta_eval": (std_lo, std_hi),
        },
        "trim": {
            eps: {
                "alpha_grid": trim[eps]["alpha_grid"], "alpha": trim[eps]["alpha"],
                "l_hat": trim[eps]["l_hat"], "u_hat": trim[eps]["u_hat"],
                "avg_alpha": trim[eps]["avg_alpha"],
                "U": trim[eps]["U"],
                "phi_star": {c: argmax_phi(trim[eps]["U"][c]) for c in c_grid},
            }
            for eps in EPS_SHOWN
        },
    }


def _plot_row(fig, gs_row, r: Dict[str, Any], title: str) -> None:
    phi_grid = r["phi_grid"]
    c_grid = r["c_grid"]
    phi0 = r["threshold"]

    # --- α column ---
    ax_a = fig.add_subplot(gs_row[0])
    ax_a.plot(r["std"]["alpha_grid"], r["std"]["alpha"], color=STD_COLOR, lw=2.0,
              label="standard")
    for eps in EPS_SHOWN:
        t = r["trim"][eps]
        col = EPS_COLORS[eps]
        ax_a.plot(t["alpha_grid"], t["alpha"], color=col, lw=1.8,
                  label=fr"trim $\epsilon$={eps}")
        ax_a.axvspan(t["l_hat"], t["u_hat"], color=col, alpha=0.05)
    ax_a.axhline(0, color="grey", lw=0.5)
    ax_a.set_xlabel(r"$\eta$", fontsize=9)
    ax_a.set_ylabel(r"$\hat\alpha(\eta)$", fontsize=10)
    ax_a.set_title(f"{title}\n$\\hat\\alpha(\\eta)$", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7, loc="best")
    ax_a.tick_params(labelsize=8)
    ax_a.grid(alpha=0.2)

    # --- U columns ---
    for jc, c in enumerate(c_grid):
        ax_u = fig.add_subplot(gs_row[jc + 1])
        # standard
        u_std = r["std"]["U"][c]
        ax_u.plot(phi_grid, u_std, color=STD_COLOR, lw=2.0, label="standard")
        ax_u.axvline(r["std"]["phi_star"][c], color=STD_COLOR, ls="--", lw=1.0, alpha=0.6)
        # trimmed per ε
        for eps in EPS_SHOWN:
            col = EPS_COLORS[eps]
            u = r["trim"][eps]["U"][c]
            ax_u.plot(phi_grid, u, color=col, lw=1.6, label=fr"trim $\epsilon$={eps}")
            ax_u.axvline(r["trim"][eps]["phi_star"][c], color=col, ls="--", lw=0.9, alpha=0.6)
        ax_u.axvline(phi0, color="grey", ls=":", lw=1.0)
        ax_u.set_xlabel(r"$\phi$", fontsize=9)
        if jc == 0:
            ax_u.set_ylabel(r"$\hat U(\phi)$", fontsize=10)
        ax_u.set_title(fr"$\hat U(\phi)$ at $c={c:g}$", fontsize=10, fontweight="bold")
        ax_u.tick_params(labelsize=8)
        ax_u.grid(alpha=0.2)
        if jc == 0:
            ax_u.legend(fontsize=7, loc="best")


def main() -> None:
    results: Dict[str, Any] = {}
    for name, _ in DATASETS:
        print(f"[alpha-util] {name}")
        results[name] = _compute(name)

    # JSON-safe dump (drop big arrays; keep φ*, c_grid, avg α, windows).
    safe = {}
    for name, r in results.items():
        safe[name] = {
            "threshold": r["threshold"], "direction": r["direction"],
            "c_grid": list(r["c_grid"]),
            "std_phi_star": {str(c): r["std"]["phi_star"][c] for c in r["c_grid"]},
            "std_eta_eval": list(r["std"]["eta_eval"]),
            "trim": {
                str(eps): {
                    "l_hat": r["trim"][eps]["l_hat"], "u_hat": r["trim"][eps]["u_hat"],
                    "avg_alpha": r["trim"][eps]["avg_alpha"],
                    "phi_star": {str(c): r["trim"][eps]["phi_star"][c] for c in r["c_grid"]},
                }
                for eps in EPS_SHOWN
            },
        }
    OUT_JSON.write_text(json.dumps(safe, indent=2, default=float))
    print(f"wrote {OUT_JSON}")

    n_rows = len(DATASETS)
    fig = plt.figure(figsize=(22, 5.2 * n_rows))
    gs = fig.add_gridspec(n_rows, 4, hspace=0.42, wspace=0.22,
                          left=0.04, right=0.99, top=0.945, bottom=0.04)
    fig.suptitle(
        "Standard vs trimmed: α(η) and utility U(φ) across ε and absolute cost c",
        fontsize=16, fontweight="bold", y=0.992,
    )
    fig.text(
        0.5, 0.972,
        "black = standard;  colored = trimmed at ε∈{0.05,0.10,0.20}.  "
        "α-panel shading marks each trimmed window [l̂,û].  "
        "Dashed verticals = φ* (argmax); grey dotted = operating φ₀.  "
        "Trimmed U is normalized per in-window individual, standard per full sample "
        "— compare φ* locations and curve shapes, not absolute heights.",
        ha="center", fontsize=9.5,
    )
    for i, (name, title) in enumerate(DATASETS):
        gs_row = [gs[i, j] for j in range(4)]
        _plot_row(fig, gs_row, results[name], title)

    fig.savefig(OUT_FIG, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
