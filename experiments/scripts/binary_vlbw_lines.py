"""Other birth-weight lines in the pooled NCHS 1995-2002 sample (Claude, 2026-11-12).

For each line L (1000, 1250, 1500, 2000, 2500 g) run routes logit_emp and lpm of
binary_outcome_routes at costs 0 and 0.01, plus local linear jumps at L in one-year death
and assisted ventilation. Does not modify the routes script.

Run (from code/):
    python -m experiments.scripts.binary_vlbw_lines --lines 1000 --out experiments/runs/binary_vlbw_lines
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments.scripts import binary_outcome_routes as B
from experiments.scripts import gpa_four_role as G

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "binary_vlbw_lines"
LINES = (1000, 1250, 1500, 2000, 2500)


def window(L: int) -> tuple:
    lo = max(L - 1000, 500)
    hi = 2500 if L <= 1500 else 3500
    return float(lo), float(hi)


def local_linear_jump(w, y, L, h):
    """Local linear, triangular kernel, jump below minus above at L (weight < L treated);
    HC1 robust se. Returns (estimate, se, n_eff)."""
    x = w - (L - 0.5)
    m = np.abs(x) < h
    x, y = x[m], y[m]
    k = 1 - np.abs(x) / h
    t = (x < 0).astype(float)
    A = np.column_stack([np.ones_like(x), t, x, x * t])
    W = k
    AtWA = A.T @ (A * W[:, None])
    beta = np.linalg.solve(AtWA, A.T @ (W * y))
    r = y - A @ beta
    meat = (A * (W * r)[:, None]).T @ (A * (W * r)[:, None])
    Ainv = np.linalg.inv(AtWA)
    V = Ainv @ meat @ Ainv * len(x) / (len(x) - 4)
    return float(beta[1]), float(np.sqrt(V[1, 1])), int(m.sum())


def build_line(L: int, half: int, eps: float = 0.1):
    from experiments.datasets.vlbw import adapter as V
    d = V.sample_frame(window(L), V.YEARS)
    Q = -d.weight.to_numpy(float)
    Y = 1.0 - d.death.to_numpy(float)
    Z = G._standardize(d[V.X_COLS].to_numpy(float))
    Xe = G._standardize(d[["gestat"]].to_numpy(float))
    phi0 = -(L - 0.5)
    data = B.Data(Q, Z, Xe, Y, phi0)
    policy = (-float(L + half), -float(L - half))
    design = B.make_design(data, eps, policy)
    return data, design, d


def jumps(d, L):
    w = d.weight.to_numpy(float)
    death = d.death.to_numpy(float)
    vent = (d.venl30.eq(1) | d.ven30m.eq(1)).to_numpy(float)
    out = {"share_within_50g": float(np.mean(np.abs(w - (L - 0.5)) < 50)),
           "n_within_50g": int(np.sum(np.abs(w - (L - 0.5)) < 50))}
    for name, y in (("death", death), ("ventilation", vent)):
        for h in (50, 100):
            est, se, n = local_linear_jump(w, y, L, h)
            out[f"{name}_h{h}"] = {"jump": est, "se": se, "n": n}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="+", type=int, default=list(LINES))
    ap.add_argument("--routes", nargs="+", default=["logit_emp", "lpm"])
    ap.add_argument("--costs", nargs="+", type=float, default=[0.0, 0.01])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for L in a.lines:
        t0 = time.time()
        res = {"line": L, "window": window(L), "runs": [], "errors": []}
        data = design = d = None
        half_used = None
        for half in (300, 200):
            try:
                data, design, d = build_line(L, half, a.eps)
                half_used = half
                break
            except Exception as e:  # report and try narrower once
                res["errors"].append({"half": half, "error": repr(e), "trace": traceback.format_exc()})
                print(f"L={L} make_design failed with half={half}: {e!r}", flush=True)
        if data is None:
            json.dump(res, open(a.out / f"line_{L}.json", "w"), indent=1)
            continue
        res.update(half=half_used, n=int(len(data.Q)), treated_share=float(data.D.mean()),
                   mean_survival=float(data.Y.mean()), J=list(design.J), policy=list(design.policy),
                   jumps=jumps(d, L))
        print(f"\n== L={L}: n={res['n']}, treated={res['treated_share']:.3f}, half={half_used}, "
              f"jumps={json.dumps(res['jumps'])}", flush=True)
        grid = np.array([L - 300, L - 150, L, L + 150, L + 300], float)
        curves = -grid
        for cost in a.costs:
            for route in a.routes:
                des = copy.copy(design)
                des.cost = cost
                t1 = time.time()
                try:
                    r = B.four_role(data, des, route, a.seed, curves, 121, 12)
                except Exception as e:
                    res["errors"].append({"route": route, "cost": cost, "error": repr(e),
                                          "trace": traceback.format_exc()})
                    print(f"  {route} c={cost} FAILED {e!r}", flush=True)
                    continue
                r.update(cost=cost, n=res["n"], optimum_grams=-r["phi_pooled"],
                         tau_grid_grams=grid.tolist(), tau_at_grid=r["tau"],
                         seconds=round(time.time() - t1, 1))
                res["runs"].append(r)
                print(f"  {route:9s} c={cost}: opt={r['optimum_grams']:.1f} bound={r['at_boundary']} "
                      f"win={r['window_share']:.2f} tau={np.round(r['tau'], 4).tolist()} "
                      f"[{r['seconds']}s]", flush=True)
                res["seconds_total"] = round(time.time() - t0, 1)
                json.dump(res, open(a.out / f"line_{L}.json", "w"), indent=1)
        res["seconds_total"] = round(time.time() - t0, 1)
        json.dump(res, open(a.out / f"line_{L}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
