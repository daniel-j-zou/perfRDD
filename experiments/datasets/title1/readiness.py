"""Readiness numbers for the Title I Targeted Grant dose design (README "Readiness").

    python -m experiments.datasets.title1.readiness            # load_dose() and lag_q=True
    python -m experiments.datasets.title1.readiness --json OUT.json

Reports, for the samples from ``adapter.load_dose()`` (X without and with last year's
SAIPE share):
* n, districts, years; R^2 of Q on [1, X] (OLS), sd(Q), sd(eta), eta = Q - fit;
* the share at dose 0 (q < 5%) and the share where the number scale binds;
* per dose decile, the 10/50/90% quantiles of eta (``dose_schedule.readiness``);
* bunching: district-years in [c, c + w) over [c - w, c) at the eligibility cutoff and each
  kink c, against the same ratio at placebo points, for w = 0.5 and 1 percentage point;
* ``adapter.rule_check`` and ``adapter.timing_check``: how well the formula dose predicts
  actual Title I revenue (F-33).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.datasets.title1 import adapter
from experiments.methods.dose_schedule import readiness as dose_readiness

PLACEBO = (0.075, 0.10, 0.125, 0.19, 0.26, 0.34, 0.42, 0.46, 0.50)


def bunching(q, points, w):
    out = {}
    for c in points:
        above = int(np.sum((q >= c) & (q < c + w)))
        below = int(np.sum((q >= c - w) & (q < c)))
        out[f"{100 * c:.2f}"] = {"below": below, "above": above,
                                 "ratio": round(above / below, 3) if below else None}
    return out


def report(s) -> dict:
    q, X, A = s.Q, s.X, np.asarray(s.extras["A"], float)
    r = dose_readiness(q, X, A, 0.0)
    out = {
        "n": int(s.n), "p": int(s.p),
        "districts": int(np.unique(s.extras["leaid"]).size),
        "years": sorted(int(t) for t in np.unique(s.extras["year"])),
        "R2_Q_on_X": round(r["R2"], 3), "sd_Q": round(float(q.std()), 4),
        "sd_eta": round(r["sd_eta"], 4), "sd_T": round(r["sd_T"], 4),
        "share_dose0": round(r["share_ref"], 4),
        "share_num_scale_binds": round(float(np.mean(s.extras["num_scale_binds"])), 4),
        "by_dose": [{"bin": b, "n": n, "eta10": round(e1, 4), "eta50": round(e5, 4), "eta90": round(e9, 4)}
                    for b, n, e1, e5, e9 in r["by_dose"]],
        "by_year": {},
    }
    yr = s.extras["year"]
    for t in np.unique(yr):
        m = yr == t
        keep = X[m].std(axis=0) > 0
        rt = dose_readiness(q[m], X[m][:, keep], A[m], 0.0)
        out["by_year"][int(t)] = {"n": int(m.sum()), "R2": round(rt["R2"], 3),
                                  "sd_eta": round(rt["sd_eta"], 4), "share_dose0": round(rt["share_ref"], 4)}
    for w in (0.005, 0.01):
        out[f"bunching_w{w}"] = bunching(q, adapter.KINKS, w)
        out[f"placebo_w{w}"] = bunching(q, PLACEBO, w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = {}
    for label, kw in (("base", {}), ("lag_q", {"lag_q": True})):
        s = adapter.load_dose(**kw)
        res[label] = report(s)
    rc = adapter.rule_check()
    res["rule_check"] = rc.to_dict("records")
    res["rule_check_corr"] = {k: round(v, 3) for k, v in rc.attrs.items()}
    res["timing_check"] = adapter.timing_check().to_dict("records")
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
