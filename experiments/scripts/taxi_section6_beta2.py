"""Bootstrap interval for beta2-hat (and the exact gain) in the Section 6 taxi application.

The saved 999-draw bootstrap of the main specification R2-E1
(``runs/taxi_section6/R2-E1_draws.jsonl``, written by ``taxi_section6.py``) stores the
threshold, the pooled maximizer, U-hat on the fare grid and alpha-hat, but not beta2-hat.
This script replays the same bootstrap: draw b resamples rides with
``default_rng(seed + 1000 + b)``, splits them with fold seed ``seed + 50000 + b`` and calls
the same ``four_role`` on the same data and design as ``taxi_section6._draw``. For each
draw it records

* beta2-hat as the paper reports it: the average of the four rotated D x distance
  coefficients. The effect covariate is the standardized distance, re-standardized in
  every resample, so the coefficient is per standard deviation of distance in that
  resample; ``beta2_per_mile`` divides it by that standard deviation (np.std, ddof 0, as
  in ``taxi_four_role._standardize``);
* the exact gain over the deployed $15 threshold, as in the point estimate:
  pooled U-hat at the pooled maximizer minus pooled U-hat at $15, in cents per ride;
* U-hat at the fixed fares $12.75, $12.90 and $13.00, relative to $15, in cents per ride;
* phi-hat and the pooled maximizer, which are checked against the saved draws.

Draws are appended to ``<out>/R2-E1_beta2_draws.jsonl`` as they finish, and a rerun
resumes from them. ``--summarize`` writes ``<out>/summary.json``.

Run (from code/):
    python -m experiments.scripts.taxi_section6_beta2 --boot 3 --workers 1      # replay check
    nohup python -m experiments.scripts.taxi_section6_beta2 --boot 999 --workers 2 \
        > experiments/runs/taxi_section6_beta2/run.log 2>&1 &
    python -m experiments.scripts.taxi_section6_beta2 --summarize
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

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_section6 as S

SPEC = S.MAIN
OUT = M.ROOT / "runs" / "taxi_section6_beta2"
SAVED = S.OUT / f"{SPEC}_draws.jsonl"
FIXED_FARES = (12.75, 12.90, 13.00)

_B: dict = {}


def _init(data, design, seed, dist):
    _B.update(data=data, design=design, seed=seed, dist=dist)


def _record(r, lam, sd_dist):
    d = lambda p: M.dollars(p, lam)
    b2 = float(np.asarray(r["beta2"])[0])
    return {"phi": d(r["phi_avg"]), "phi_pooled": d(r["phi_pooled"]),
            "at_boundary": bool(r["at_boundary"]),
            "beta2": b2, "sd_dist": sd_dist, "beta2_per_mile": b2 / sd_dist,
            "gain_cents": 100 * r["gain_vs_deployed"],
            "U_fixed_cents": (100 * np.asarray(r["U"])).tolist(),
            "window_share": r["window_share"]}


def _draw(b):
    data, design, seed, dist = _B["data"], _B["design"], _B["seed"], _B["dist"]
    t0 = time.time()
    idx = np.random.default_rng(seed + 1_000 + b).integers(0, len(data.Q), len(data.Q))
    r = M.four_role(data.subset(idx), design, seed=seed + 50_000 + b,
                    grid_points=S.GRID_POINTS, curves=M.boxcox(np.array(FIXED_FARES), data.lam),
                    tau_and_own=False)
    return {"b": b, **_record(r, data.lam, float(dist[idx].std())),
            "seconds": round(time.time() - t0, 1)}


def _saved():
    return {r["b"]: r for r in S._load(SAVED)}


def run(boot, workers, seed, out):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = M.load_frame("full")
    data, design = S.build(df, SPEC)
    dist = df["Trip_Distance"].to_numpy(float)
    point_path = out / "point.json"
    if not point_path.exists():
        r = M.four_role(data, design, seed, grid_points=S.GRID_POINTS,
                        curves=M.boxcox(np.array(FIXED_FARES), data.lam), tau_and_own=False)
        point = {"spec": SPEC, "seed": seed, "fixed_fares": list(FIXED_FARES),
                 **_record(r, data.lam, float(dist.std()))}
        point_path.write_text(json.dumps(point, indent=1) + "\n")
        print(f"point: phi {point['phi']:.4f} beta2 {point['beta2']:.4f} per sd, "
              f"{point['beta2_per_mile']:.4f} per mile, gain {point['gain_cents']:.4f} cents "
              f"[{time.time() - t0:.0f}s]", flush=True)
    path = out / f"{SPEC}_beta2_draws.jsonl"
    done = {r["b"] for r in S._load(path)}
    todo = [b for b in range(boot) if b not in done]
    saved = _saved()
    if not todo:
        return
    print(f"bootstrap: {len(done)} done, running {len(todo)} with {workers} workers", flush=True)
    t1 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(data, design, seed, dist)) as ex, path.open("a") as fh:
        futures = [ex.submit(_draw, b) for b in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            s = saved.get(row["b"])
            if s is not None:
                row["saved_diff"] = max(abs(row["phi"] - s["phi"]),
                                        abs(row["phi_pooled"] - s["phi_pooled"]))
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i <= 5 or i % 25 == 0 or i == len(todo):
                el = time.time() - t1
                print(f"  {i}/{len(todo)} draws, {el:.0f}s, b={row['b']} phi {row['phi']:.4f} "
                      f"saved diff {row.get('saved_diff', float('nan')):.1e} "
                      f"beta2/mile {row['beta2_per_mile']:.4f}; "
                      f"eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
    print(f"done [{time.time() - t0:.0f}s]", flush=True)


def _pct(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def saved_gain():
    """The gain over $15 from the saved draws alone (no refit). The saved U-hat is on a
    $0.25 fare grid, so each curve is interpolated with a cubic spline: the gain is the
    spline at the draw's pooled maximizer (the paper's 0.65 cents is the pooled U-hat at
    the pooled maximizer), and U-hat($12.90) - U-hat($15) is the spline at $12.90."""
    from scipy.interpolate import CubicSpline
    summ = json.loads((S.OUT / "summary.json").read_text())
    est, grid = summ["specs"][SPEC], np.array(summ["fare_grid"])
    rows = sorted(_saved().values(), key=lambda r: r["b"])
    U = 100 * np.array([r["U"] for r in rows])
    pp = np.array([r["phi_pooled"] for r in rows])
    U0 = 100 * np.array(est["U"])
    cs0 = CubicSpline(grid, U0)
    k1275, k13 = int(np.argmin(abs(grid - 12.75))), int(np.argmin(abs(grid - 13.0)))
    stats = {
        "gain_at_pooled_max": (float(cs0(est["phi_pooled"])),
                               np.array([CubicSpline(grid, u)(p) for u, p in zip(U, pp)])),
        "gain_max_on_grid": (float(U0.max()), U.max(axis=1)),
        "U_12.90": (float(cs0(12.90)), np.array([CubicSpline(grid, u)(12.90) for u in U])),
        "U_12.75": (float(U0[k1275]), U[:, k1275]),
        "U_13.00": (float(U0[k13]), U[:, k13]),
    }
    res = {"B": len(rows), "paper_gain_cents": est["gain_cents"]}
    for k, (pt, x) in stats.items():
        res[k] = {"point": pt, "mean": float(x.mean()), "pct95": _pct(x),
                  "share_positive": float(np.mean(x > 0)), "min": float(x.min())}
    draws = {k: dict(zip((r["b"] for r in rows), x)) for k, (_, x) in stats.items()}
    return res, draws


def summarize(out, boot):
    rows = sorted(S._load(out / f"{SPEC}_beta2_draws.jsonl"), key=lambda r: r["b"])[:boot]
    point = json.loads((out / "point.json").read_text())
    pct = _pct
    saved, sdraws = saved_gain()
    gap = lambda key, f: float(max(abs(f(r) - sdraws[key][r["b"]]) for r in rows))
    saved["interpolation_check_cents"] = {
        "gain_vs_exact_replay": gap("gain_at_pooled_max", lambda r: r["gain_cents"]),
        "U_12.90_vs_exact_replay": gap("U_12.90", lambda r: r["U_fixed_cents"][1])}
    res = {"spec": SPEC, "B": len(rows), "point": point, "saved_draws_gain_cents": saved,
           "replay": {"max_abs_diff_vs_saved": float(max(r.get("saved_diff", np.nan) for r in rows)),
                      "n_compared": int(sum("saved_diff" in r for r in rows))}}
    for key in ("beta2", "beta2_per_mile", "gain_cents"):
        x = np.array([r[key] for r in rows])
        res[key] = {"point": point[key], "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                    "pct95": pct(x), "share_positive": float(np.mean(x > 0))}
    Uf = np.array([r["U_fixed_cents"] for r in rows])
    res["U_fixed_cents"] = {
        f"{f:.2f}": {"point": point["U_fixed_cents"][k], "pct95": pct(Uf[:, k]),
                     "share_positive": float(np.mean(Uf[:, k] > 0))}
        for k, f in enumerate(FIXED_FARES)}
    res["seconds_per_draw"] = float(np.mean([r["seconds"] for r in rows]))
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    if a.summarize:
        summarize(a.out, a.boot)
        return
    run(a.boot, a.workers, a.seed, a.out)


if __name__ == "__main__":
    main()
