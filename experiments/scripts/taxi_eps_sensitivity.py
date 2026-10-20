"""Taxi application: how the four-role estimate moves with the trim level eps.

Same setup as ``taxi_four_role_grid`` (fare in dollars; outcome fit on the
central 99.5% of residuals; trim window capped at the central 99%), with the
effect linear in distance and the three score specifications: distance (S1),
distance and log distance (S2) and a spline in log distance (S3). For each eps
the script reports the estimate, its fare cell, the four rotations, the pooled
maximizer, the supported share of rides and the gain over the deployed $15.
S2 also gets a full re-estimation bootstrap (default 20 draws, 90% interval
estimate +- 1.645 sd, as in the nine-way comparison).

With eps <= 0.05 both trim endpoints fall outside the observed residuals, so
the cap at the central 99% sets the window.

Run (from code/):
    python -m experiments.scripts.taxi_eps_sensitivity --boot 20 --workers 5
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.scripts import taxi_four_role as M

OUT = M.ROOT / "runs" / "taxi_eps_sensitivity"
LAM = 1.0
EPS_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
SCORES = ("linear", "distlog", "spline")
BOOT_SCORE = "distlog"
GRID_POINTS = 61

_B: dict = {}


def _init(data, design, seed):
    _B.update(data=data, design=design, seed=seed)


def _draw(b):
    data, design = _B["data"], _B["design"]
    idx = np.random.default_rng(_B["seed"] + 1_000 + b).integers(0, len(data.Q), len(data.Q))
    r = M.four_role(data.subset(idx), design, seed=_B["seed"] + 50_000 + b, grid_points=GRID_POINTS)
    return r["phi_avg"], r["at_boundary"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = M.load_frame("full")
    rows = []
    for score in SCORES:
        data = M.make_data(df, LAM, "distance", score=score)
        fares = data.fare
        d = lambda p: M.dollars(p, LAM)
        for eps in EPS_GRID:
            design = M.make_design(data, eps, effect=True, tail=0.0025, window_tail=0.005)
            r = M.four_role(data, design, a.seed, grid_points=GRID_POINTS)
            row = {"score": score, "eps": eps, "phi": d(r["phi_avg"]),
                   "cell": M.fare_cell(d(r["phi_avg"]), fares),
                   "phi_pooled": d(r["phi_pooled"]),
                   "rotations": [d(p) for p in r["phi_rotations"]],
                   "window": r["window"], "window_share": r["window_share"],
                   "gain_cents": 100 * r["gain_vs_deployed"], "beta2": r["beta2"],
                   "at_boundary": r["at_boundary"], "local_maxima": r["local_maxima"]}
            if score == BOOT_SCORE and a.boot:
                with ProcessPoolExecutor(a.workers, initializer=_init,
                                         initargs=(data, design, a.seed)) as ex:
                    draws = list(ex.map(_draw, range(a.boot)))
                phis = np.array([d(x[0]) for x in draws])
                sd = float(phis.std(ddof=1))
                row["boot"] = {"B": a.boot, "sd": sd,
                               "ci90": [row["phi"] - 1.645 * sd, row["phi"] + 1.645 * sd],
                               "draws": phis.round(3).tolist(),
                               "share_boundary": float(np.mean([x[1] for x in draws]))}
            rows.append(row)
            b = row.get("boot", {})
            print(f"{score:8s} eps {eps:5.3f} phi {row['phi']:6.2f} cell {row['cell']:5.2f} "
                  f"pooled {row['phi_pooled']:6.2f} rot {min(row['rotations']):5.2f}-{max(row['rotations']):5.2f} "
                  f"share {row['window_share']:.3f} gain {row['gain_cents']:.2f}c "
                  f"sd {b.get('sd', float('nan')):.2f} [{time.time() - t0:.0f}s]", flush=True)
            with open(a.out / "summary.json", "w") as fh:
                json.dump({"lambda": LAM, "eps_grid": EPS_GRID, "rows": rows}, fh, indent=1)
    print(f"wrote {a.out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
