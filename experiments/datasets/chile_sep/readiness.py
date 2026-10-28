"""Readiness numbers for the SEP dose design (README "Readiness" section).

    python -m experiments.datasets.chile_sep.readiness            # load_dose()
    python -m experiments.datasets.chile_sep.readiness --json OUT.json

Reports, for the sample from ``adapter.load_dose()``:
* n, schools, years; R^2 of Q on [1, X] (OLS) and sd(eta), eta = Q - fit;
* the share at the reference dose (0, share below 15%);
* per dose level (and table period): n, Q range, and the 10/50/90% quantiles of eta;
* bunching: units in [c, c + w) over units in [c - w, c) at each tier cutoff c, against
  the same ratio at placebo points, for w = 1 and 2 percentage points.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

PLACEBO = (10.0, 20.0, 25.0, 35.0, 40.0, 50.0, 55.0, 65.0, 70.0, 75.0, 80.0)


def _ols(X, q):
    Z = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(Z, q, rcond=None)[0]
    fit = Z @ b
    return fit, q - fit


def bunching(q, points, w):
    out = {}
    for c in points:
        above = int(np.sum((q >= c) & (q < c + w)))
        below = int(np.sum((q >= c - w) & (q < c)))
        out[f"{c:g}"] = {"above": above, "below": below,
                         "ratio": round(above / below, 3) if below else None}
    return out


def readiness(s) -> dict:
    q, X, A = s.Q.astype(float), s.X.astype(float), np.asarray(s.extras["A"], float)
    fit, eta = _ols(X, q)
    cuts = s.extras.get("cuts", (15.0, 30.0, 45.0, 60.0))
    groups = np.asarray(s.extras["group"]) if s.extras.get("group") is not None else None
    r = {
        "name": s.name, "n": int(s.n), "p": int(s.p),
        "schools": int(np.unique(s.extras["rbd"]).size) if "rbd" in s.extras else None,
        "R2_Q_on_X": round(float(1 - eta.var() / q.var()), 3),
        "sd_Q": round(float(q.std()), 2), "sd_eta": round(float(eta.std()), 2),
        "share_at_dose0": round(float(np.mean(A == s.extras.get("a_ref", 0.0))), 3),
        "levels": [],
        "bunching_w1": bunching(q, cuts, 1.0), "bunching_w2": bunching(q, cuts, 2.0),
        "placebo_w1": bunching(q, PLACEBO, 1.0), "placebo_w2": bunching(q, PLACEBO, 2.0),
    }
    keys = sorted(set(zip(groups if groups is not None else ["all"] * len(A), A)))
    for g, a in keys:
        m = (A == a) & ((groups == g) if groups is not None else True)
        e = eta[m]
        r["levels"].append({
            "group": str(g), "dose": float(a), "n": int(m.sum()),
            "Q_min": round(float(q[m].min()), 2), "Q_max": round(float(q[m].max()), 2),
            "eta_q10": round(float(np.quantile(e, 0.1)), 2),
            "eta_q50": round(float(np.quantile(e, 0.5)), 2),
            "eta_q90": round(float(np.quantile(e, 0.9)), 2),
            "sd_eta": round(float(e.std()), 2),
        })
    rat = [v["ratio"] for v in r["placebo_w1"].values() if v["ratio"]]
    r["placebo_w1_range"] = [min(rat), max(rat)] if rat else None
    return r


if __name__ == "__main__":
    from experiments.datasets.chile_sep.adapter import load_dose

    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    res = readiness(load_dose())
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.json:
        open(a.json, "w").write(txt)
