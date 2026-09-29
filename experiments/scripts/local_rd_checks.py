"""Generic design-based checks at a deployed cutoff, for any RDDSample loader.

Local linear RD (uniform kernel, separate slopes, heteroskedasticity-robust SEs) of
the outcome and of every covariate on the side of the cutoff that is treated.
Covariate jumps are a sorting / balance check. The outcome jump is the ITT effect at
the cutoff, to compare with the global models' fitted effect near the cutoff
(``screen_flatness.py --near``). Optional subgroup estimates split at the terciles of
one covariate among units within the bandwidth. With a discrete running variable the
standard errors are approximate.

    PYTHONPATH=. python experiments/scripts/local_rd_checks.py MODULE:FUNC OUT.json \
        --bandwidth H [--below] [--donut V] [--subgroup COVARIATE]
"""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np


def rd(q, y, cut, h, below, donut=None):
    keep = (np.abs(q - cut) < h) & np.isfinite(y)
    if donut is not None:
        keep &= ~np.isclose(q, donut)
    x = q[keep] - cut
    d = (x < 0) if below else (x >= 0)
    d = d.astype(float)
    Z = np.column_stack((np.ones(len(x)), d, x, d * x))
    yy = y[keep]
    A = np.linalg.inv(Z.T @ Z)
    b = A @ Z.T @ yy
    e = yy - Z @ b
    V = A @ (Z.T * e ** 2) @ Z @ A
    return {"est": float(b[1]), "se": float(np.sqrt(V[1, 1])), "n": int(keep.sum())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("loader")
    p.add_argument("out")
    p.add_argument("--bandwidth", type=float, required=True)
    p.add_argument("--below", action="store_true")
    p.add_argument("--donut", type=float, default=None)
    p.add_argument("--subgroup", default=None)
    a = p.parse_args(argv)
    mod, func = a.loader.split(":")
    s = getattr(importlib.import_module(mod), func)()
    q, y, X = np.asarray(s.Q, float), np.asarray(s.Y, float), np.asarray(s.X, float)
    cut, h = float(s.threshold), a.bandwidth
    res = {"loader": a.loader, "cutoff": cut, "bandwidth": h, "below": a.below,
           "outcome": rd(q, y, cut, h, a.below),
           "balance": {n: rd(q, X[:, j], cut, h, a.below) for j, n in enumerate(s.feature_names)}}
    if a.donut is not None:
        res["donut"] = a.donut
        res["outcome_donut"] = rd(q, y, cut, h, a.below, a.donut)
    if a.subgroup:
        j = s.feature_names.index(a.subgroup)
        near = np.abs(q - cut) < h
        t1, t2 = np.quantile(X[near, j], [1 / 3, 2 / 3])
        groups = {"low": X[:, j] <= t1, "mid": (X[:, j] > t1) & (X[:, j] <= t2), "high": X[:, j] > t2}
        res["subgroups"] = {f"{a.subgroup}={k}": rd(q[m], y[m], cut, h, a.below) for k, m in groups.items()}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    fmt = lambda r: f"{r['est']:+.4f} ({r['se']:.4f})"
    print(f"{a.loader}: cutoff {cut:g}, bandwidth {h:g}, n in band {res['outcome']['n']:,}")
    print(f"  outcome ITT {fmt(res['outcome'])}" +
          (f" | donut {fmt(res['outcome_donut'])}" if a.donut is not None else ""))
    print("  balance: " + ", ".join(f"{n} {fmt(v)}" for n, v in res["balance"].items()))
    for k, v in res.get("subgroups", {}).items():
        print(f"  {k}: {fmt(v)} n={v['n']:,}")


if __name__ == "__main__":
    main()
