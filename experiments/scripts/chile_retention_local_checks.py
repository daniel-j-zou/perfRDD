"""Model-free checks at the Chilean retention cutoff (average below 4.5).

Local linear RD on the 0.1 grid with bandwidth 0.5 on each side of 4.45 (Q in
4.0-4.4 vs 4.5-4.9; the second cutoff at 5.0 is outside), heteroskedasticity-robust
standard errors. Reported:
  * first stage (retained) and intent-to-treat effects on each outcome, with and
    without the heaped value 4.5 (donut), and the Wald ratio ITT / first stage;
  * covariate jumps at the cutoff (sorting check);
  * ITT and first stage by subgroup (prior-average tercile, grade band, school type),
    which is a design-based check of the heterogeneity the differing-slopes fit claims.
The discrete running variable makes the standard errors approximate.

    PYTHONPATH=. python experiments/scripts/chile_retention_local_checks.py YEAR OUT.json OUTCOME [OUTCOME ...] [--long-run]

--long-run uses adapter.build_long_run (primary 6-8 and youth secondary 1-3, with the
completed_secondary outcome) instead of adapter.build_cohort.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

CUT, H = 4.45, 0.5


def rd(df: pd.DataFrame, col: str, donut: bool = False) -> dict:
    d = df[(df.Q > CUT - H) & (df.Q < CUT + H) & df[col].notna()]
    if donut:
        d = d[d.Q.round(1) != 4.5]
    x = d.Q.to_numpy() - CUT
    D = (x < 0).astype(float)
    Z = np.column_stack((np.ones(len(x)), D, x, D * x))
    y = d[col].to_numpy(float)
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    b = ZtZ_inv @ Z.T @ y
    e = y - Z @ b
    V = ZtZ_inv @ (Z.T * e ** 2) @ Z @ ZtZ_inv
    return {"est": float(b[1]), "se": float(np.sqrt(V[1, 1])), "n": int(len(y))}


def main(year: int, out: str, outcomes: list, long_run: bool = False) -> None:
    from experiments.datasets.chile_retention.adapter import build_cohort, build_long_run
    df = build_long_run(year) if long_run else build_cohort(year)
    res = {"year": year, "n": int(len(df)), "bandwidth": H}
    for donut in (False, True):
        key = "donut" if donut else "all"
        r = {"first_stage": rd(df, "retained", donut)}
        for y in outcomes:
            itt = rd(df, y, donut)
            r[y] = dict(itt, wald=itt["est"] / r["first_stage"]["est"])
        res[key] = r
    covs = ("prior_gpa", "prior_att", "prior_retained", "overage", "male",
            "grade_level", "municipal", "private", "rural")
    res["balance"] = {c: rd(df, c) for c in covs}
    res["balance_donut"] = {c: rd(df, c, donut=True) for c in covs}
    near = df[(df.Q > CUT - H) & (df.Q < CUT + H)]
    terc = pd.qcut(near.prior_gpa, 3, labels=["low", "mid", "high"])
    df = df.assign(prior_tercile=pd.Series(terc, index=near.index))
    groups = {
        "prior_tercile": df.prior_tercile,
        "grade_band": pd.cut(df.grade_level, [1, 4, 8, 11], labels=["2-4", "5-8", "sec1-3"]),
        "school": np.select([df.municipal == 1, df.private == 1], ["municipal", "private"], "subsidized"),
    }
    res["subgroups"] = {}
    for gname, g in groups.items():
        for level in pd.Series(g).dropna().unique():
            sub = df[np.asarray(g == level)]
            entry = {"first_stage": rd(sub, "retained")}
            for y in outcomes:
                entry[y] = rd(sub, y)
            res["subgroups"][f"{gname}={level}"] = entry
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    def fmt(r):
        return f"{r['est']:+.4f} ({r['se']:.4f})"
    print(f"year {year}: n = {res['n']:,}")
    for key in ("all", "donut"):
        r = res[key]
        line = f"  {key:6s} first stage {fmt(r['first_stage'])} n={r['first_stage']['n']:,}"
        for y in outcomes:
            line += f" | {y}: ITT {fmt(r[y])} Wald {r[y]['wald']:+.3f}"
        print(line)
    print("  balance:       " + ", ".join(f"{c} {fmt(v)}" for c, v in res["balance"].items()))
    print("  balance donut: " + ", ".join(f"{c} {fmt(v)}" for c, v in res["balance_donut"].items()))
    for k, v in res["subgroups"].items():
        print(f"  {k:24s} FS {fmt(v['first_stage'])} " +
              " ".join(f"{y} {fmt(v[y])}" for y in outcomes) + f" n={v['first_stage']['n']:,}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--long-run"]
    main(int(args[0]), args[1], args[2:], long_run="--long-run" in sys.argv)
