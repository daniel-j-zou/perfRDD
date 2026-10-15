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

Run (from code/):
    python -m experiments.scripts.taxi_four_role_figures \
        --summary experiments/runs/taxi_four_role/summary_full_lam0.5_eps0.05.json
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    s = json.loads(a.summary.read_text())
    fare = np.array(s["curves"]["fare"])
    tau = np.array(s["curves"]["tau"])
    U = 100 * np.array(s["curves"]["U_minus_U15"])
    boot = s.get("bootstrap")
    phi = s["estimate"]["phi_avg_dollars"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    if boot:
        lo, hi = np.array(boot["tau_band"])
        ax1.fill_between(fare, lo, hi, color="C0", alpha=0.25, lw=0, label="95% bootstrap band")
    ax1.plot(fare, tau, color="C0", lw=2, label=r"estimated $\hat\tau_\epsilon$ (VTS)")
    if PROXY.exists():
        p = pd.read_csv(PROXY)
        ax1.errorbar(p["fare"], p["alpha_proxy_cmt_minus_vts"],
                     yerr=1.96 * p["alpha_proxy_hc0_se"], fmt="o", ms=3, color="0.35",
                     elinewidth=0.8, label="CMT $-$ VTS, fixed-menu fares (benchmark)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.axvline(phi, color="C3", ls="--", lw=1.2)
    ax1.axvline(15, color="0.5", ls=":", lw=1.2)
    ax1.set_xlabel("fare cutoff (\\$)")
    ax1.set_ylabel("tip change at the margin (\\$ per ride)")
    ax1.set_title("Marginal effect of the percentage menu")
    ax1.legend(fontsize=7.5, loc="lower right", frameon=False)

    if boot:
        lo, hi = 100 * np.array(boot["U_band"])
        ax2.fill_between(fare, lo, hi, color="C0", alpha=0.25, lw=0)
    ax2.plot(fare, U, color="C0", lw=2)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axvline(phi, color="C3", ls="--", lw=1.2, label=f"estimate \\${phi:.2f}")
    ax2.axvline(15, color="0.5", ls=":", lw=1.2, label="deployed \\$15")
    ax2.set_xlabel("fare cutoff (\\$)")
    ax2.set_ylabel("tips relative to \\$15 (cents per ride)")
    ax2.set_title("Estimated utility")
    ax2.legend(fontsize=8, loc="lower center", frameon=False)
    fig.tight_layout()
    out = a.out or a.summary.with_name(a.summary.stem.replace("summary", "figure") + ".png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
