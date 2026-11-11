"""Taxi main specification (R2-E1) without the residual-quantile window cap.

Question (author, 2026-11-10): is the cap at the central 99% of the residuals needed,
and what happens without it?  Prints the eps-window [l0, u0] against the residual
quantiles, then re-estimates (point estimates, fold seeds 1-3) with
  * the main rule: J excludes 0.25% per tail, window capped at 0.5% per tail;
  * J stretched to exclude 0.01% per tail, cap kept at 0.5% (isolates J);
  * J at 0.01% and cap at 0.02%: the cap no longer binds, so the window is the
    eps-window itself ("no cap");
  * J and cap at 0 (J = all residuals), which fails (singular spline Gram).

Run from code/:  python -m experiments.scripts.taxi_window_cap_check
"""
from __future__ import annotations

import numpy as np

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_section6 as R6


def main():
    df = M.load_frame("full")
    data, design = R6.build(df, "R2-E1")
    T = M._xt(data.Z) @ np.linalg.lstsq(M._xt(data.Z), data.Q, rcond=None)[0]
    eta = data.Q - T
    l0, u0 = design.phi0 - np.quantile(T, 0.95), design.phi0 - np.quantile(T, 0.05)
    print(f"eps-window [l0, u0] = [{l0:.2f}, {u0:.2f}]; sd(eta) = {eta.std():.3f}; "
          f"share of rides inside {np.mean((eta >= l0) & (eta <= u0)):.4f}")
    print("eta quantiles (0, .05, .25, .5, 99.5, 99.75, 99.95, 100 %):",
          np.round(np.quantile(eta, [0, .0005, .0025, .005, .995, .9975, .9995, 1]), 2))
    D = data.D.astype(bool)
    for lo, hi in [(l0, design.window[0]), (design.window[1], u0)]:
        m = (eta >= lo) & (eta <= hi)
        print(f"rides with eta in [{lo:.2f}, {hi:.2f}]: {m.sum()} ({m.mean():.2%}), "
              f"treated share {D[m].mean():.2f}")
    for tail, wt in [(0.0025, 0.005), (0.0001, 0.005), (0.0001, 0.0002), (0.0, 0.0)]:
        try:
            d2, g2 = R6.build(df, "R2-E1", tail=tail, window_tail=wt)
        except Exception as e:                       # noqa: BLE001
            print(f"J tail {tail}, cap {wt}: {type(e).__name__}: {e}")
            continue
        for seed in (1, 2, 3):
            try:
                est = R6.estimate(d2, g2, seed)
            except Exception as e:                   # noqa: BLE001
                print(f"J tail {tail}, cap {wt}, seed {seed}: {type(e).__name__}: {e}")
                break
            print(f"J tail {tail}, cap {wt}, seed {seed}: ${est['phi']:.2f} (cell "
                  f"${est['cell']}), rotations {np.round(est['rotations'], 2)}, "
                  f"J {np.round(g2.J, 2)}, window share {est['window_share']:.4f}", flush=True)


if __name__ == "__main__":
    main()
