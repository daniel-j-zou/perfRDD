"""Shape of the fitted taxi score in distance, and identification of the effect slope.

For the §6 theory extension (weekly review and plan of 2026-11-04). On the square-root
fare scale, with the full-sample first stage of ``taxi_four_role.make_design``:

* R2-E1 (distance + log distance): the derivative of the distance part of the fitted
  index, at every recorded distance. A positive minimum means the score is strictly
  increasing in distance, as the several-columns version of (Q1m) asks.
* R3-E1 (cubic B-spline in log distance): the same derivative; where the spline turns,
  the index value at the turn in each indicator cell, against the argument set
  V = {phi - eta: phi in the policy set, eta in the window} on which the criterion
  evaluates the index density; and the share of the D x distance column left after
  partialling out the rest of the outcome design (``effect_diagnostics``).

Run (from code/):
    python -m experiments.scripts.taxi_score_shape_check
Writes ``runs/taxi_score_shape/summary.json``.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import BSpline

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_section6 as S6

OUT = M.ROOT / "runs" / "taxi_score_shape"
SEED = 20261027


def _spline_basis(x):
    """The basis of ``score_covariates(kind='spline')`` and its knots."""
    lo, hi = float(x.min()), float(x.max())
    t = np.concatenate([np.repeat(lo, 4), np.quantile(x, [0.2, 0.4, 0.6, 0.8]),
                        np.repeat(hi, 4)])
    return t, lo, hi


def main() -> dict:
    df = M.load_frame("full")
    d = df["Trip_Distance"].to_numpy(float)
    x = np.log(d)
    vals, counts = np.unique(d, return_counts=True)
    out = {"n": int(len(d)), "distinct_distances": int(len(vals))}
    for name, score in (("R2-E1", "distlog"), ("R3-E1", "spline")):
        data = M.make_data(df, 0.5, effect="distance", score=score)
        Xt = M._xt(data.Z)
        g = np.linalg.lstsq(Xt, data.Q, rcond=None)[0]
        if score == "distlog":
            raw = np.column_stack((d, x))
            gam = g[1:3] / raw.std(0)
            deriv = gam[0] + gam[1] / vals
            res = {"per_mile": float(gam[0]), "per_log_mile": float(gam[1])}
        else:
            t, lo, hi = _spline_basis(x)
            raw = BSpline.design_matrix(np.clip(x, lo, hi), t, 3).toarray()[:, 1:]
            k = raw.shape[1]
            gam = g[1:1 + k] / raw.std(0)
            spl = BSpline(t, np.concatenate([[0.0], gam]), 3)
            deriv = spl.derivative()(np.clip(np.log(vals), lo, hi)) / vals
            res = {}
        neg = deriv <= 0
        res.update({"min_derivative": float(deriv.min()),
                    "argmin_distance": float(vals[np.argmin(deriv)]),
                    "nonincreasing_distance_range": ([float(vals[neg].min()), float(vals[neg].max())]
                                                     if neg.any() else None),
                    "rides_where_nonincreasing": int(counts[neg].sum()),
                    "share_rides_where_nonincreasing": float(counts[neg].sum() / len(d))})
        design = M.make_design(data, S6.EPS, effect=True, tail=S6.TAIL,
                               window_tail=S6.WINDOW_TAIL, window_estimated=True)
        V = (design.policy[0] - design.window[1], design.policy[1] - design.window[0])
        res["argument_set_V"] = [float(V[0]), float(V[1])]
        if neg.any():
            # index value at the turn, per indicator cell
            grid = np.exp(np.linspace(np.log(vals.min()), np.log(1.0), 4001))
            Bg = BSpline.design_matrix(np.clip(np.log(grid), lo, hi), t, 3).toarray()[:, 1:]
            fg = ((Bg - raw.mean(0)) / raw.std(0)) @ g[1:1 + k]
            i = int(np.argmin(fg))
            rest = Xt @ g - data.Z[:, :k] @ g[1:1 + k]
            Tc = fg[i] + np.unique(np.round(rest, 10))
            res.update({"turning_distance": float(grid[i]),
                        "index_at_turn_range": [float(Tc.min()), float(Tc.max())],
                        "indicator_cells": int(len(Tc)),
                        "cells_with_turn_inside_V": int(np.sum((Tc >= V[0]) & (Tc <= V[1])))})
        res["effect_block"] = M.effect_diagnostics(data, design, SEED)
        out[name] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
