"""Taxi performative-RDD application with a logit tip-share outcome.

Identical design and estimator to ``taxi_perfrdd_application.py`` (NYC TLC 2009
VTS credit-card rides; Q = fare, phi_0 = $15, D = 1{fare >= 15} = percentage-tip
menu; X = trip covariates; full unrestricted sample), except the outcome is the
*logit tip share* instead of the tip in dollars:

    p_i = Tip_i / Fare_i,   Y_i = logit(clip(p_i, e, 1 - e)),   e = 0.01.

The tip share is a bounded proportion; the logit maps it to the whole line so it
is a natural regression outcome. About 3.6% of rides tip exactly zero (p = 0) and
0.28% record a ratio >= 1 (including data-entry outliers up to 40x); the clip to
[0.01, 0.99] keeps the logit finite for these boundary cases.

Produces (mirrors the dollar-outcome application, with a ``taxi_logit`` name):
  * alpha / b / utility figures via ``screen``;
  * the inference-grade hard-trim point estimate (``perfrdd_hard_trim``);
  * a bootstrap distribution of the optimal threshold phi_hat + histogram.

Run:
    python -m experiments.scripts.taxi_perfrdd_logit_tip
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._core.registry import load
from experiments.methods.perfrdd import _reduce_to_primary_axis
from experiments.methods.perfrdd_hard_trim import perfrdd_hard_trim
from experiments.scripts.screen_candidate import screen
from experiments.scripts.taxi_perfrdd_application import (
    EPS, NUISANCE_SUPPORT, RIDGE, BOOT_B, BOOT_M, _phistar,
)

CLIP_P = 0.01  # keep logit(tip share) finite at the 0 / 1 boundaries
OUT = Path(__file__).resolve().parent.parent / "runs" / "taxi_logit"


def _logit_tip_sample(sample):
    """Return a copy of the taxi sample whose outcome is the logit tip share."""
    Q = np.asarray(sample.Q, float)
    tip = np.asarray(sample.Y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(Q > 0, tip / Q, np.nan)
    n_zero = int(np.nansum(p <= 0.0))
    n_high = int(np.nansum(p >= 1.0))
    p = np.clip(p, CLIP_P, 1.0 - CLIP_P)
    y = np.log(p / (1.0 - p))
    print(f"logit tip share: clipped {n_zero:,} zero-tip and {n_high:,} ratio>=1 rows "
          f"to [{CLIP_P},{1-CLIP_P}]")
    return dataclasses.replace(
        sample,
        Y=y,
        name="taxi_logit",
        description=(
            "NYC TLC 2009 VTS credit-card rides. Q = fare; treatment = 1{Q>=15} "
            "(percentage-tip menu); Y = logit(Tip/Fare), the logit tip share "
            f"(ratio clipped to [{CLIP_P},{1-CLIP_P}])."
        ),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = _logit_tip_sample(load("taxi"))
    Q, X, thr = _reduce_to_primary_axis(sample)
    Y = np.asarray(sample.Y, float)
    keep = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(1)
    Q, X, Y = Q[keep], X[keep], Y[keep]
    N = len(Y)
    print(f"n={N:,}  fare median={np.median(Q):.1f}  Y(logit) mean={Y.mean():.3f} sd={Y.std():.3f}")

    # (1) alpha / b / utility figures via the ridge-regularized pooled PLM screen.
    screen(sample, out_root=OUT)

    # (2) inference-grade point estimate.
    r = perfrdd_hard_trim(sample, OUT / "hard_trim", nuisance_support=NUISANCE_SUPPORT,
                          eps=EPS, c_values=(0.0,), crossfit_folds=1, max_n=None,
                          ridge_scale=RIDGE)
    fd = r["fold_diagnostics"][0]
    print(f"hard-trim phi*={float(r['phi_star']['0.0']):.2f} "
          f"interior={not r['phi_star_at_grid_boundary']['0.0']} "
          f"cond#={fd['design_condition_number']:.1e}")

    # (3) bootstrap distribution of phi_hat (iid trips).
    rng = np.random.default_rng(11)
    phis, nint = [], 0
    for _ in range(BOOT_B):
        idx = rng.integers(0, N, BOOT_M)
        ph, it = _phistar(Q[idx], X[idx], Y[idx], thr)
        phis.append(ph); nint += it
    phis = np.array(phis)
    print(f"bootstrap B={BOOT_B} m={BOOT_M}: mean={phis.mean():.2f} sd={phis.std():.2f} "
          f"median={np.median(phis):.2f} "
          f"IQR=[{np.percentile(phis,25):.2f},{np.percentile(phis,75):.2f}] "
          f"95%CI=[{np.percentile(phis,2.5):.2f},{np.percentile(phis,97.5):.2f}] "
          f"interior={nint/BOOT_B:.0%}")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hist(phis, bins=25, color="#3b6", alpha=.85, edgecolor="white")
    ax.axvline(phis.mean(), color="C3", lw=2, label=f"mean ${phis.mean():.2f}")
    ax.axvline(thr, color="black", ls=":", label=f"deployed cutoff ${thr:.0f}")
    ax.set_xlabel(r"$\hat\phi$ (optimal fare threshold, \$)"); ax.set_ylabel("count")
    ax.set_title(f"Logit tip share: bootstrap of $\\hat\\phi$ (B={BOOT_B}, m={BOOT_M:,})")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(OUT / "taxi_logit_bootstrap.png", dpi=150, bbox_inches="tight")
    np.save(OUT / "phis.npy", phis)


if __name__ == "__main__":
    main()
