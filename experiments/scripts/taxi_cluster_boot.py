"""Taxi main specification: bootstrap that resamples days (or day-hours) instead of rides.

Section 6 treats rides as independent; tips are likely to be dependent within drivers
(no driver identifiers in the 2009 records) and within days (weather, traffic, events).
This script reruns the full re-estimation bootstrap of the main specification (R2-E1,
daytime sample, ``taxi_section6.py``) with clusters resampled whole:

* ``day``: the calendar day of pickup (January 2009; 31 clusters);
* ``dayhour``: the calendar day and the hour of pickup (about 340 clusters).

Draw b draws the clusters with replacement (as many clusters as the sample has), keeps
every ride of each drawn cluster, splits the resample into four new random blocks
(seed + 50_000 + b, as in the ride bootstrap) and re-runs every fit with the main
design (J, eps, window cap, Phi) held fixed, as ``taxi_section6._draw`` does. The
resample size varies from draw to draw.

``--spec ls_poisson`` runs the location-scale row of Section 6.4
(``taxi_locscale_row.py``) on the same cluster draws and splits as the main
specification's draw b, so the two can be paired (the paired difference of Section 6.4).

Draws go to ``<out>/<cluster>_draws.jsonl`` (``<cluster>_ls_poisson_draws.jsonl``) as they
finish and a rerun resumes;
``--boot-start a --boot b`` runs draws a..b-1 (one Slurm array task).

Run (from code/):
    python -m experiments.scripts.taxi_cluster_boot --cluster day --boot 999 --workers 8
    python -m experiments.scripts.taxi_cluster_boot --cluster day --summarize
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_locscale_row as LS
from experiments.scripts import taxi_section6 as S6

OUT = M.ROOT / "runs" / "taxi_cluster_boot"
SPEC = S6.MAIN
CLUSTERS = ("day", "dayhour")
SPECS = ("main", "ls_poisson")

_B: dict = {}


def cluster_ids(df: pd.DataFrame, kind: str) -> np.ndarray:
    pickup = pd.to_datetime(df["Trip_Pickup_DateTime"])
    day = pickup.dt.day.to_numpy()
    if kind == "day":
        return day
    if kind == "dayhour":
        return day * 100 + pickup.dt.hour.to_numpy()
    raise ValueError(kind)


def setup(kind: str, spec: str = "main"):
    df = M.load_frame(S6.SAMPLES[S6.SAMPLE]["frame"])
    extra = {}
    if spec == "main":
        data, design = S6.build(df, SPEC)
    else:
        fit, zs, both = LS.VARIANTS[spec]
        data = M.make_data(df, LS.LAM, LS.CFG["effect"], score=LS.SCORE, hour_set=LS.CFG["hours"])
        Zs = LS.scale_covariates(df, data, zs)
        design, _ = LS.make_design(data, Zs, LS.STAGE[fit], both_arms=both)
        extra = {"Zs": Zs, "stage": LS.STAGE[fit], "both": both}
    ids = cluster_ids(df, kind)
    if len(ids) != len(data.Q):
        raise RuntimeError("cluster ids are not aligned with the data")
    labels, inv = np.unique(ids, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(labels))]
    if spec != "main":
        return data, design, labels, members, extra
    return data, design, labels, members


def _init(kind, seed, spec="main"):
    out = setup(kind, spec)
    data, design, labels, members = out[:4]
    _B.update(data=data, design=design, members=members, seed=seed, spec=spec,
              extra=out[4] if len(out) > 4 else {})


def _draw(b):
    data, design, members, seed = _B["data"], _B["design"], _B["members"], _B["seed"]
    G = len(members)
    pick = np.random.default_rng(seed + 3_000_000 + b).integers(0, G, G)
    idx = np.concatenate([members[g] for g in pick])
    if _B["spec"] == "main":
        r = M.four_role(data.subset(idx), design, seed=seed + 50_000 + b,
                        grid_points=S6.GRID_POINTS, tau_and_own=False)
    else:
        x = _B["extra"]
        r = LS.four_role(data.subset(idx), design, seed + 50_000 + b, x["Zs"][idx], x["stage"],
                         own=False, both_arms=x["both"])
    d = lambda p: M.dollars(p, data.lam)
    return {"b": b, "n": int(len(idx)), "phi": d(r["phi_avg"]), "phi_pooled": d(r["phi_pooled"]),
            "at_boundary": bool(r["at_boundary"]),
            "beta2": np.round(np.asarray(r["beta2"], float), 7).tolist(),
            "gain_cents": float(100 * r["gain_vs_deployed"])}


def _name(kind, spec):
    return kind if spec == "main" else f"{kind}_{spec}"


def run(kind, boot, boot_start, workers, seed, out: Path, spec="main"):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_name(kind, spec)}_draws.jsonl"
    rows = S6._load(path)
    done = {r["b"] for r in rows}
    todo = [b for b in range(boot_start, boot) if b not in done]
    if todo:
        t0 = time.time()
        print(f"[{kind}] {len(done)} done, running {len(todo)}", flush=True)
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(kind, seed, spec)) as ex, \
                path.open("a") as fh:
            futures = [ex.submit(_draw, b) for b in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} draws, {time.time() - t0:.0f}s", flush=True)


def summarize(kind, boot, seed, out: Path, ride_summary: Path, spec="main") -> dict:
    rows = sorted(S6._load(out / f"{_name(kind, spec)}_draws.jsonl"), key=lambda r: r["b"])[:boot]
    data, design, labels, members = setup(kind, spec)[:4]
    phis = np.array([r["phi"] for r in rows])
    gains = np.array([r["gain_cents"] for r in rows])
    sizes = np.array([len(m) for m in members])
    res = {"cluster": kind, "spec": SPEC, "clusters": int(len(labels)),
           "cluster_size": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                            "max": int(sizes.max())},
           "B": len(rows), "sd": float(phis.std(ddof=1)),
           "pct95": np.quantile(phis, [0.025, 0.975]).tolist(),
           "pct90": np.quantile(phis, [0.05, 0.95]).tolist(),
           "pooled_pct95": np.quantile([r["phi_pooled"] for r in rows], [0.025, 0.975]).tolist(),
           "gain_cents_pct95": np.quantile(gains, [0.025, 0.975]).tolist(),
           "share_gain_positive": float(np.mean(gains > 0)),
           "share_boundary": float(np.mean([r["at_boundary"] for r in rows])),
           "share_below_15": float(np.mean(phis < M.FARE0)),
           "cells": {str(k): int(v) for k, v in zip(*np.unique(
               [M.fare_cell(p, data.fare) for p in phis], return_counts=True))},
           "resample_n": {"min": int(min(r["n"] for r in rows)),
                          "max": int(max(r["n"] for r in rows))}}
    if spec != "main":
        main = {r["b"]: r["phi"] for r in S6._load(out / f"{kind}_draws.jsonl")}
        common = [r for r in rows if r["b"] in main]
        if common:
            diff = np.array([r["phi"] - main[r["b"]] for r in common])
            res["paired_with_main"] = {
                "n": len(common), "mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1)),
                "pct95_diff": np.quantile(diff, [0.025, 0.975]).tolist(),
                "share_negative": float(np.mean(diff < 0))}
        (out / f"{_name(kind, spec)}_summary.json").write_text(json.dumps(res, indent=1) + "\n")
        return res
    if ride_summary.exists():
        main = json.loads(ride_summary.read_text())["specs"][SPEC]
        res["ride_bootstrap"] = {"phi": main["phi"], "sd": main["boot"]["sd"],
                                 "pct95": main["boot"]["pct95"],
                                 "gain_cents_pct95": main["boot"].get("gain_cents_pct95")}
        res["sd_ratio_to_rides"] = res["sd"] / main["boot"]["sd"]
    (out / f"{kind}_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cluster", choices=CLUSTERS, required=True)
    ap.add_argument("--spec", choices=SPECS, default="main")
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--boot-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--ride-summary", type=Path, default=S6.OUT / "summary.json")
    args = ap.parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize(args.cluster, args.boot, args.seed, args.out,
                                   args.ride_summary, args.spec), indent=1))
        return
    run(args.cluster, args.boot, args.boot_start, args.workers, args.seed, args.out, args.spec)


if __name__ == "__main__":
    main()
