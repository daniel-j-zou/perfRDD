"""Diagnose the dip in b̂(η) around η ≈ 15–20 for lending_club.

Strategy:
  1. Fit the linear perfrdd as usual; recover β, ω_base, ω_treat.
  2. Form Y_resid = Y − X·β̂.  For controls this satisfies
        Y_resid ≈ b(η) + ε
     so binning Y_resid by η over controls gives a model-free check on b̂.
  3. Plot b̂(η) against bin-mean Y_resid for controls, with the η density
     stacked underneath, so we can read off whether the dip is data-driven or
     a spline artifact in a sparse region.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments._core.registry import load
from experiments.methods.perfrdd import (
    DEFAULT_MAX_N,
    _detect_direction,
    _eval_basis,
    _fit_pooled_plm,
    _reduce_to_primary_axis,
    _subsample,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "lc_b_dip_diagnostic.png"


def main() -> None:
    sample = load("lending_club")
    Q, X, threshold = _reduce_to_primary_axis(sample)
    Y, D = sample.Y, sample.D
    direction = _detect_direction(D, Q)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y, D = Q[keep], X[keep], Y[keep], D[keep]
    if len(Y) > DEFAULT_MAX_N:
        (Q, X, Y, D), _, _ = _subsample([Q, X, Y, D], DEFAULT_MAX_N)

    fit = _fit_pooled_plm(Q, X, Y, D, direction)

    # Residualized Y: only the nonparametric (and, for treated, α) piece remains.
    Y_resid = Y - fit.intercept - X @ fit.beta

    # Fitted b̂ on the full spline support.
    grid = np.linspace(fit.info["lo"], fit.info["hi"], 500)
    b_fitted = _eval_basis(grid, fit.info) @ fit.omega_base

    # Bin Y_resid by η on the data range.
    eta = fit.eta
    eta_lo, eta_hi = float(eta.min()), float(eta.max())
    n_bins = 60
    bin_edges = np.linspace(eta_lo, eta_hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    co_mask = D == 0
    tr_mask = D == 1
    eta_co = eta[co_mask]
    Y_resid_co = Y_resid[co_mask]
    Y_resid_tr = Y_resid[tr_mask]
    eta_tr = eta[tr_mask]

    bin_idx_co = np.digitize(eta_co, bin_edges) - 1
    bin_idx_co = np.clip(bin_idx_co, 0, n_bins - 1)
    bin_idx_tr = np.digitize(eta_tr, bin_edges) - 1
    bin_idx_tr = np.clip(bin_idx_tr, 0, n_bins - 1)

    counts_co = np.bincount(bin_idx_co, minlength=n_bins).astype(int)
    counts_tr = np.bincount(bin_idx_tr, minlength=n_bins).astype(int)

    # Mean Y_resid per bin for controls (skip bins with fewer than 5 obs).
    means_co = np.full(n_bins, np.nan)
    sems_co = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = bin_idx_co == i
        if sel.sum() >= 5:
            vals = Y_resid_co[sel]
            means_co[i] = vals.mean()
            sems_co[i] = vals.std(ddof=1) / np.sqrt(len(vals))

    means_tr = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = bin_idx_tr == i
        if sel.sum() >= 5:
            means_tr[i] = Y_resid_tr[sel].mean()

    # ------- numerical readout for the dip region -------
    print("η range of the dip (η ∈ [13, 20]):")
    print(f"{'bin':>14}  {'n_co':>6}  {'n_tr':>6}  {'mean(Y-Xβ|co)':>14}  {'fitted b̂':>10}")
    for i in range(n_bins):
        if bin_edges[i] < 13 or bin_edges[i] > 20:
            continue
        b_here = _eval_basis(np.array([bin_centers[i]]), fit.info) @ fit.omega_base
        m = means_co[i]
        mstr = f"{m:>14.3f}" if not np.isnan(m) else f"{'(n<5)':>14}"
        print(f"[{bin_edges[i]:6.2f},{bin_edges[i+1]:6.2f})  "
              f"{counts_co[i]:6d}  {counts_tr[i]:6d}  {mstr}  {b_here[0]:10.3f}")

    # ------- figure -------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        sharex=True,
    )

    ax_top.plot(grid, b_fitted, "C0-", lw=2.2,
                label=r"$\hat b(\eta)$  (spline fit)")
    ax_top.errorbar(bin_centers, means_co, yerr=sems_co,
                    fmt="o", color="steelblue", ms=4, lw=0.8, alpha=0.85,
                    label=r"bin-mean $Y - X\hat\beta$  (controls, ≥5 / bin)")
    ax_top.scatter(bin_centers, means_tr, marker="x", s=22, color="firebrick",
                   alpha=0.7,
                   label=r"bin-mean $Y - X\hat\beta$  (treated, for context)")
    ax_top.axvspan(13.5, 19.5, color="orange", alpha=0.12, label="dip region")
    ax_top.set_ylabel(r"$Y - X\hat\beta$  /  $\hat b(\eta)$")
    ax_top.set_title("Lending Club: does the dip in b̂(η) at η ≈ 15–20 reflect the data?")
    ax_top.legend(loc="upper left", fontsize=9)

    width = bin_edges[1] - bin_edges[0]
    ax_bot.bar(bin_centers, counts_co, width=width, color="steelblue",
               alpha=0.65, label=f"control  (n={counts_co.sum():,})")
    ax_bot.bar(bin_centers, counts_tr, width=width, bottom=counts_co,
               color="firebrick", alpha=0.65,
               label=f"treated  (n={counts_tr.sum():,})")
    ax_bot.axvspan(13.5, 19.5, color="orange", alpha=0.12)
    ax_bot.set_xlabel(r"$\eta$")
    ax_bot.set_ylabel("count")
    ax_bot.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
