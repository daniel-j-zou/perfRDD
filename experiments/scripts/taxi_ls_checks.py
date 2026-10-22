"""Taxi checks behind this_week.tex, "A location-scale first stage".

1. Share treated against eta-hat for the three dollar scores of the nine-way
   comparison (distance; distance + log distance; spline in log distance): under X
   independent of eta it must rise with eta, so a U shape is a failure of independence.
2. Folds of the location-scale fit (dollars, distance + log distance, Harvey scale):
   rides where x -> T(x) + e s(x) decreases in distance, for e at the ends of the trim
   window.

Run (from code/):
    python -m experiments.scripts.taxi_ls_checks
Writes experiments/runs/taxi_score_scales/ls_checks.json.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json

import numpy as np
from scipy.interpolate import BSpline

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_score_scales as S

OUT = M.ROOT / "runs" / "taxi_score_scales" / "ls_checks.json"


def share_curve(df, terms, bins=40):
    data = S.make_data(df, 1.0, terms)
    A = M._xt(data.Z)
    eta = data.Q - A @ np.linalg.lstsq(A, data.Q, rcond=None)[0]
    D = data.D.astype(bool)
    lo, hi = np.quantile(eta, [0.005, 0.995])
    edges = np.linspace(lo, hi, bins + 1)
    b = np.digitize(eta, edges) - 1
    share = np.array([D[b == k].mean() for k in range(bins)])
    mid = 0.5 * (edges[1:] + edges[:-1])
    k = int(np.argmin(share))
    return {"eta_mid": mid.tolist(), "share": share.tolist(), "min_share": float(share[k]),
            "eta_at_min": float(mid[k]), "share_left_end": float(share[0]),
            "share_right_end": float(share[-1])}


def folds(df):
    d = df["Trip_Distance"].to_numpy(float)
    data = S.make_data(df, 1.0, "d+logd")
    Zs = M.score_covariates(df, "spline")
    A = M._xt(data.Z)
    g = np.linalg.lstsq(A, data.Q, rcond=None)[0]
    eta = data.Q - A @ g
    As = M._xt(Zs)
    idx = np.arange(len(eta))
    s = S.fit_scale(Zs, eta, idx)
    delta = np.linalg.lstsq(As, np.log(eta ** 2 + 1e-6 * np.var(eta)), rcond=None)[0]
    zeta = eta / s
    # d T / d distance: the score's distance columns are standardized d and log d
    raw = np.column_stack((d, np.log(d)))
    dT = g[1] / raw[:, 0].std() + g[2] / (raw[:, 1].std() * d)
    # d log s^2 / d distance through the spline in log distance (first columns of Zs)
    x = np.log(d)
    lo, hi = x.min(), x.max()
    t = np.concatenate([np.repeat(lo, 4), np.quantile(x, [0.2, 0.4, 0.6, 0.8]), np.repeat(hi, 4)])
    B = BSpline.design_matrix(np.clip(x, lo, hi), t, 3).toarray()[:, 1:]
    nb = B.shape[1]
    dB = np.column_stack([BSpline(t, np.eye(nb + 1)[k + 1], 3).derivative()(x) for k in range(nb)])
    ds = s * 0.5 * ((dB / B.std(0)) @ delta[1:1 + nb]) / d
    D = data.D.astype(bool)
    q = lambda v, p: float(np.quantile(v, p))
    window = (max(q(zeta[D], 0.005), q(zeta[~D], 0.005)), min(q(zeta[D], 0.995), q(zeta[~D], 0.995)))
    out = {"window": list(window)}
    for name, e in (("lower", window[0]), ("upper", window[1])):
        bad = dT + e * ds <= 0
        out[name] = {"e": e, "share_folding": float(bad.mean())}
        if bad.any():
            out[name].update(distance_range=[float(d[bad].min()), float(d[bad].max())],
                             median_fare=float(np.median(data.fare[bad])),
                             share_rides_this_short=float(np.mean(d <= d[bad].max())))
    return out


def main():
    df = M.load_frame("full")
    res = {"share_treated": {k: share_curve(df, k) for k in ("d", "d+logd", "spline")},
           "folds": folds(df)}
    for k, v in res["share_treated"].items():
        print(f"{k:7s} share treated: left end {v['share_left_end']:.3f}, minimum {v['min_share']:.3f} "
              f"at eta {v['eta_at_min']:+.2f}, right end {v['share_right_end']:.3f}")
    print("folds:", json.dumps(res["folds"]))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
