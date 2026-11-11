"""Bootstrap the VLBW optimum of the binary routes: resample births, re-split the four
roles, refit everything, and report percentile intervals of the rotated average threshold.

    python -m experiments.scripts.binary_outcome_vlbw_boot --variant pooled --costs 0.02 --draws 20

Writes ``experiments/runs/binary_outcome_routes_vlbw_<variant>/bootstrap.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.scripts import binary_outcome_routes as R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="low", help="low (2002), pooled (1995-2002)")
    ap.add_argument("--route", default="logit_emp")
    ap.add_argument("--costs", type=float, nargs="+", default=[0.02, 0.01])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.1)
    a = ap.parse_args(argv)
    data, design, meta = R.build_vlbw(a.variant, eps=a.eps)
    curves = np.linspace(*design.policy, 31)
    out_dir = Path("experiments/runs") / f"binary_outcome_routes_vlbw_{a.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for cost in a.costs:
        design.cost = cost
        draws, t0 = [], time.time()
        for b in range(a.draws):
            idx = np.random.default_rng(1_000 + b).integers(0, len(data.Q), len(data.Q))
            r = R.four_role(data.subset(idx), design, a.route, seed=50_000 + b, curves=curves, grid_points=61)
            draws.append(-r["phi_avg"])
            print(f"c={cost} draw {b}: {-r['phi_avg']:.1f} g ({time.time() - t0:.0f}s)", flush=True)
        d = np.array(draws)
        out[str(cost)] = {"route": a.route, "n": meta["n"], "draws": d.tolist(), "mean": float(d.mean()),
                          "sd": float(d.std(ddof=1)),
                          "pct": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        print(f"c={cost}: mean {d.mean():.1f}, sd {d.std(ddof=1):.1f}, "
              f"95% [{np.percentile(d, 2.5):.0f}, {np.percentile(d, 97.5):.0f}]", flush=True)
        with open(out_dir / "bootstrap.json", "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
