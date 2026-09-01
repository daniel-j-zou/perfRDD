"""Reproduce the taxi performative-RDD application (paper: taxi_application.tex).

NYC TLC 2009 VTS credit-card rides. Q = fare, phi_0 = $15, D = 1{fare >= 15}
(percentage-tip menu), Y = tip, X = trip covariates. The FULL (unrestricted)
sample is used on purpose: the performative threshold question needs variation
across the whole fare range, which the Haggag-Paci $5-25 local-RD restriction
removes.

Produces:
  * point estimate via the inference-grade hard-trim estimator (perfrdd_hard_trim),
  * a bootstrap distribution of the optimal threshold phi_hat,
  * the alpha / b / utility figures (via screen_candidate) and a bootstrap histogram.

Run:
    python -m experiments.scripts.taxi_perfrdd_application
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.registry import load
from experiments.methods.perfrdd import (
    _detect_direction, _eval_basis, _fit_pooled_plm, _reduce_to_primary_axis,
)
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim
from experiments.scripts.screen_candidate import _utility_profiles, screen

EPS = 0.10
# Prespecified, data-dense nuisance support + light ridge -> well-conditioned design
# (cond# ~5.6e3; the naive wide/zero-ridge fit is singular, cond# ~1e18).
NUISANCE_SUPPORT = (-5.0, 10.0)
RIDGE = 1.0
BOOT_B = 120
BOOT_M = 120_000
OUT = Path(__file__).resolve().parent.parent / "runs" / "taxi_application"


def _phistar(Q, X, Y, thr, n_grid=241):
    D = (Q > thr).astype(float)
    fit = _fit_pooled_plm(Q, X, Y, D, _detect_direction(D, Q))
    lo, hi = fit.eta_eval
    T = Q - fit.eta
    l0 = thr - np.quantile(T, 1 - EPS); u0 = thr - np.quantile(T, EPS)
    l0, u0 = max(min(l0, u0), lo), min(max(l0, u0), hi)
    pg = np.linspace(thr - 3 * np.std(Q), thr + 3 * np.std(Q), n_grid)
    A, _ = _utility_profiles(fit, Q, pg, window=(l0, u0))
    j = int(np.argmax(A))
    return float(pg[j]), (j not in (0, len(pg) - 1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = load("taxi")
    Q, X, thr = _reduce_to_primary_axis(sample)
    Y = np.asarray(sample.Y, float)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(1)
    Q, X, Y = Q[keep], X[keep], Y[keep]
    N = len(Y)
    print(f"n={N:,}  fare median={np.median(Q):.1f}")

    # (1) alpha / b / utility figures via the screen (ridge-regularized pooled PLM).
    screen(sample, out_root=OUT)

    # (2) inference-grade point estimate.
    r = perfrdd_hard_trim(sample, OUT / "hard_trim", nuisance_support=NUISANCE_SUPPORT,
                          eps=EPS, c_values=(0.0,), crossfit_folds=1, max_n=None,
                          ridge_scale=RIDGE)
    fd = r["fold_diagnostics"][0]
    print(f"hard-trim phi*={float(r['phi_star']['0.0']):.2f} "
          f"interior={not r['phi_star_at_grid_boundary']['0.0']} cond#={fd['design_condition_number']:.1e}")

    # (3) bootstrap distribution of phi_hat (iid trips).
    rng = np.random.default_rng(11)
    phis, nint = [], 0
    for _ in range(BOOT_B):
        idx = rng.integers(0, N, BOOT_M)
        ph, it = _phistar(Q[idx], X[idx], Y[idx], thr)
        phis.append(ph); nint += it
    phis = np.array(phis)
    print(f"bootstrap B={BOOT_B} m={BOOT_M}: mean={phis.mean():.2f} sd={phis.std():.2f} "
          f"median={np.median(phis):.2f} 95%CI=[{np.percentile(phis,2.5):.2f},{np.percentile(phis,97.5):.2f}] "
          f"interior={nint/BOOT_B:.0%}")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hist(phis, bins=25, color="#3b6", alpha=.85, edgecolor="white")
    ax.axvline(phis.mean(), color="C3", lw=2, label=f"mean ${phis.mean():.2f}")
    ax.axvline(thr, color="black", ls=":", label=f"deployed cutoff ${thr:.0f}")
    ax.set_xlabel(r"$\hat\phi$ (optimal fare threshold, \$)"); ax.set_ylabel("count")
    ax.set_title(f"Bootstrap distribution of $\\hat\\phi$ (B={BOOT_B}, m={BOOT_M:,})")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(OUT / "taxi_bootstrap.png", dpi=150, bbox_inches="tight")
    np.save(OUT / "phis.npy", phis)


if __name__ == "__main__":
    main()
