"""Freddie Mac credit-score band edges as thresholds, binary outcome routes (Claude, 2026-11-12).

Exploratory. Score Q = credit score at origination; treatment = being at or above a GSE pricing
band edge (680, 720, 740, 660); Y = 1 if the loan never reaches 90+ days delinquent (status >= 3
or RA) nor a default termination (zero-balance code 02, 03, 09) within 36 months of origination.
X = standardized loan features (the freddie_mac folder has no adapter.py; this script builds the
frame from the sample files via screen.ORIG). Runs routes of binary_outcome_routes.

Run (from code/):
    python -m experiments.scripts.binary_freddie_lines --edges 680 720 740 660
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._core.sample import RDDSample
from experiments.datasets.freddie_mac.screen import ORIG, RAW
from experiments.scripts import binary_outcome_routes as B

CACHE = Path(RAW).parent / "cache" / "freddie_default36_2010_2019.parquet"
OUT = B.ROOT / "runs" / "binary_freddie_lines"
YEARS = range(2010, 2020)
HORIZON = 36
DEFAULT_ZB = {"02", "03", "09"}


def _perf_year(y: int) -> pd.DataFrame:
    """Per loan: default within HORIZON months (any 90+ day status or default termination)."""
    f = RAW / f"sample_perf_{y}.txt"
    parts = []
    for ch in pd.read_csv(f, sep="|", header=None, usecols=[0, 3, 4, 8], names=["seq", "st", "age", "zb"] if False else None,
                          dtype=str, chunksize=2_000_000):
        ch.columns = ["seq", "st", "age", "zb"]
        age = pd.to_numeric(ch.age, errors="coerce")
        ch = ch[age <= HORIZON]
        st = pd.to_numeric(ch.st, errors="coerce")
        d90 = (st >= 3) | (ch.st == "RA")
        dzb = ch.zb.isin(DEFAULT_ZB)
        parts.append(pd.DataFrame({"seq": ch.seq, "d90": d90, "dzb": dzb}).groupby("seq").agg("max"))
    p = pd.concat(parts).groupby(level=0).max()
    return p.reset_index()


def load_frame(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    orig = []
    for y in YEARS:
        o = pd.read_csv(RAW / f"sample_orig_{y}.txt", sep="|", header=None, names=ORIG, low_memory=False)
        orig.append(o)
    orig = pd.concat(orig, ignore_index=True)
    with ProcessPoolExecutor(5) as ex:
        perf = pd.concat(list(ex.map(_perf_year, YEARS)), ignore_index=True)
    df = orig.merge(perf, on="seq", how="left")
    df["has_perf"] = df.d90.notna()
    df[["d90", "dzb"]] = df[["d90", "dzb"]].fillna(False).astype(bool)
    df["oy"] = df.first_pay // 100
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.fico.between(300, 850) & df.ltv.between(1, 105) & df.dti.between(0, 65)
           & df.has_perf & df.upb.notna()].copy()
    d["default"] = (d.d90 | d.dzb).astype(float)
    return d


def features(d: pd.DataFrame):
    num = pd.DataFrame({"ltv": d.ltv, "dti": d.dti, "log_upb": np.log(d.upb), "nborr2": (d.nborr == 2).astype(float),
                        "fthb": (d.fthb == "Y").astype(float)}, index=d.index)
    cat = pd.get_dummies(d[["purpose", "ptype", "occ", "channel", "state", "oy"]].astype(str), drop_first=True).astype(float)
    X = pd.concat([num, cat], axis=1)
    X = X.loc[:, X.std() > 0]
    return X


def r2_score_on_X(d: pd.DataFrame) -> float:
    X = features(d)
    A = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
    y = d.fico.to_numpy(float)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(1 - (y - A @ b).var() / y.var())


def make_sample(d: pd.DataFrame, edge: int) -> RDDSample:
    X = features(d)
    return RDDSample(Q=d.fico.to_numpy(float), X=X.to_numpy(float), Y=d.default.to_numpy(float),
                     threshold=float(edge), name=f"freddie_{edge}", feature_names=list(X.columns))


def local_jump(d: pd.DataFrame, edge: int, bw: int):
    """Uniform-kernel local linear jump in the default rate at the edge, HC1 robust s.e."""
    g = d[(d.fico >= edge - bw) & (d.fico < edge + bw)]
    q = g.fico.to_numpy(float) - (edge - 0.5)
    D = (g.fico >= edge).to_numpy(float)
    A = np.column_stack([np.ones(len(g)), D, q, D * q])
    y = g.default.to_numpy(float)
    XtXi = np.linalg.inv(A.T @ A)
    b = XtXi @ A.T @ y
    u = y - A @ b
    V = XtXi @ (A.T * u ** 2) @ A @ XtXi * len(g) / (len(g) - 4)
    return dict(bw=bw, n=len(g), jump=float(b[1]), se=float(np.sqrt(V[1, 1])), mean_below=float(y[D == 0].mean()),
                mean_above=float(y[D == 1].mean()))


def bunching(d: pd.DataFrame, edge: int, k: int = 5) -> dict:
    c = d.fico.value_counts()
    return {int(s): int(c.get(s, 0)) for s in range(edge - k, edge + k)}


def run_edge(d: pd.DataFrame, edge: int, costs, eps: float, routes, effect, seed: int, grid: int):
    sub = d[d.ltv.between(75, 80)]
    sample = make_sample(sub, edge)
    out = []
    for cost in costs:
        data, design, meta = B.build_sample(sample, (edge - 30.0, edge + 30.0), effect, eps=eps, mirror=False, survival=True)
        design.cost = cost
        curves = np.linspace(edge - 30.0, edge + 30.0, 61)
        for route in routes:
            t0 = time.time()
            try:
                r = B.four_role(data, design, route, seed, curves, grid)
            except Exception as e:  # keep going, record the failure
                print(f"FAIL edge={edge} route={route} cost={cost}: {type(e).__name__}: {e}", flush=True)
                out.append(dict(edge=edge, route=route, cost=cost, error=f"{type(e).__name__}: {e}"))
                continue
            row = dict(edge=edge, route=route, cost=cost, n=meta["n"], treated=float(data.D.mean()),
                       window_share=r["window_share"], phi=r["phi_pooled"], phi_avg=r["phi_avg"],
                       boundary=bool(r["at_boundary"]), maxima=r["local_maxima"], gain=r["gain_vs_deployed"],
                       eff_cut=r["effect_at_cutoff"], eff_win=r["effect_in_window"],
                       tau={int(c - edge): t for c, t in zip(curves, r["tau"]) if (c - edge) % 15 == 0},
                       mean_Y=float(data.Y.mean()), seconds=round(time.time() - t0, 1))
            out.append(row)
            print(f"edge={edge} {route:9s} c={cost}: n={row['n']} treated={row['treated']:.3f} win={row['window_share']:.2f} "
                  f"phi={row['phi']:.1f} bound={row['boundary']} tau={ {k: round(v, 4) for k, v in row['tau'].items()} } "
                  f"[{row['seconds']}s]", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", nargs="+", type=int, default=[680, 720, 740, 660])
    ap.add_argument("--routes", nargs="+", default=["logit_emp", "lpm"])
    ap.add_argument("--costs", nargs="+", type=float, default=[0.0, 0.005])
    ap.add_argument("--effect", nargs="+", default=["dti"])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    raw = load_frame(a.refresh)
    d = clean(raw)
    summ = dict(n_orig=len(raw), n_perf=int(raw.has_perf.sum()), n=len(d), default_rate=float(d.default.mean()),
                d90_rate=float(d.d90.mean()), r2=r2_score_on_X(d), n_ltv75_80=int(d.ltv.between(75, 80).sum()),
                default_rate_ltv75_80=float(d[d.ltv.between(75, 80)].default.mean()),
                mean_fico=float(d.fico.mean()), sd_fico=float(d.fico.std()))
    print(json.dumps(summ), flush=True)
    stats = {}
    for e in a.edges:
        sub = d[d.ltv.between(75, 80)]
        stats[e] = dict(jumps=[local_jump(sub, e, bw) for bw in (10, 20)], bunch=bunching(sub, e),
                        bunch_all_ltv=bunching(d, e))
        print(e, json.dumps(stats[e]), flush=True)
    json.dump(dict(summary=summ, stats=stats), open(a.out / "stats.json", "w"), indent=1)
    if a.stats_only:
        return
    rows = []
    for e in a.edges:
        rows += run_edge(d, e, a.costs, a.eps, a.routes, a.effect, a.seed, a.grid)
        json.dump(rows, open(a.out / "routes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
