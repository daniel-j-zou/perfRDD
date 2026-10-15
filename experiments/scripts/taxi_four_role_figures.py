"""Figures for the taxi four-role application (reads ``taxi_four_role`` summaries).

Left panel: the estimated marginal net effect tau(fare) = E(W - c | fare, window),
the tip change from showing percentage suggestions to the rides at the margin of a
cutoff at that fare, with a 95% bootstrap band.  The optimum is where it crosses
zero.  The CMT-minus-VTS low-fare contrast (``taxi_low_fare_proxy``) is overlaid as
an external benchmark only: it is a different vendor, a different percentage menu
(15/20/25% against 20/25/30%) and the $5-$25 population, and it is not used by the
estimator.

Right panel: the estimated utility relative to the deployed $15 cutoff, in cents
per ride, with a 95% bootstrap band.

With ``--compare`` a second summary (for example the model with every score
covariate in the effect block) is overlaid on both panels.

Run (from code/):
    python -m experiments.scripts.taxi_four_role_figures \
        --summary experiments/runs/taxi_four_role/summary_full_lam0.5_eps0.05.json
    python -m experiments.scripts.taxi_four_role_figures \
        --summary experiments/runs/taxi_four_role/summary_full_lam0.5_eps0.05.json \
        --compare experiments/runs/taxi_four_role/summary_full_lam0.5_eps0.05_distance+Z.json \
        --labels "effect in distance" "effect in distance and Z" --no-benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROXY = ROOT / "runs" / "taxi_low_fare_proxy" / "fare_proxy.csv"


def _draw(ax_tau, ax_u, s, color, label, show_phi=True):
    fare = np.array(s["curves"]["fare"])
    tau = np.array(s["curves"]["tau"])
    U = 100 * np.array(s["curves"]["U_minus_U15"])
    boot = s.get("bootstrap")
    phi = s["estimate"]["phi_avg_dollars"]
    if boot:
        lo, hi = np.array(boot["tau_band"])
        ax_tau.fill_between(fare, lo, hi, color=color, alpha=0.22, lw=0,
                            label="95% bootstrap band" if not show_phi else None)
        lo, hi = 100 * np.array(boot["U_band"])
        ax_u.fill_between(fare, lo, hi, color=color, alpha=0.22, lw=0)
    ax_tau.plot(fare, tau, color=color, lw=2, label=label)
    ax_u.plot(fare, U, color=color, lw=2, label=f"{label}: \\${phi:.2f}" if show_phi else None)
    if show_phi:
        for ax in (ax_tau, ax_u):
            ax.axvline(phi, color=color, ls="--", lw=1.1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    s = json.loads(a.summary.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    if a.compare is None:
        _draw(ax1, ax2, s, "C0", r"estimated $\hat\tau_\epsilon$ (VTS)", show_phi=False)
        phi = s["estimate"]["phi_avg_dollars"]
        for ax in (ax1, ax2):
            ax.axvline(phi, color="C3", ls="--", lw=1.2)
        ax2.plot([], [], color="C3", ls="--", label=f"estimate \\${phi:.2f}")
    else:
        labels = a.labels or ["main", "comparison"]
        _draw(ax1, ax2, s, "C0", labels[0])
        _draw(ax1, ax2, json.loads(a.compare.read_text()), "C1", labels[1])
    if not a.no_benchmark and PROXY.exists():
        p = pd.read_csv(PROXY)
        ax1.errorbar(p["fare"], p["alpha_proxy_cmt_minus_vts"],
                     yerr=1.96 * p["alpha_proxy_hc0_se"], fmt="o", ms=3, color="0.35",
                     elinewidth=0.8, label="CMT $-$ VTS, fixed-menu fares (benchmark)")
    for ax in (ax1, ax2):
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(15, color="0.5", ls=":", lw=1.2)
        ax.set_xlabel("fare cutoff (\\$)")
    ax2.plot([], [], color="0.5", ls=":", label="deployed \\$15")
    ax1.set_ylabel("tip change at the margin (\\$ per ride)")
    ax1.set_title("Marginal effect of the percentage menu")
    ax1.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax2.set_ylabel("tips relative to \\$15 (cents per ride)")
    ax2.set_title("Estimated utility")
    ax2.legend(fontsize=8, loc="lower center", frameon=False)
    fig.tight_layout()
    stem = a.summary.stem.replace("summary", "figure") + ("_compare" if a.compare else "")
    out = a.out or a.summary.with_name(stem + ".png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
