"""Readiness numbers for the ui_upjohn dose design (README, "Readiness").

    python -m experiments.datasets.ui_upjohn.readiness            # pooled WA + IL, ref=cap
    python -m experiments.datasets.ui_upjohn.readiness --states WA --ref floor
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.datasets.ui_upjohn.adapter import GROUPS, load_dose


def ols_eta(Q: np.ndarray, X: np.ndarray):
    Z = np.column_stack([np.ones(len(Q)), X])
    beta, *_ = np.linalg.lstsq(Z, Q, rcond=None)
    eta = Q - Z @ beta
    r2 = 1.0 - eta.var() / Q.var()
    return eta, r2


def bunching(s: np.ndarray, kink: float, h: float = 0.02):
    """Counts in [k(1-h), k) and [k, k(1+h)), and their ratio above/below."""
    lo = np.sum((s >= kink * (1 - h)) & (s < kink))
    hi = np.sum((s >= kink) & (s < kink * (1 + h)))
    return int(lo), int(hi), (hi / lo if lo else np.nan)


def report(states=("WA", "IL"), ref="cap") -> None:
    smp = load_dose(states=states, ref=ref)
    ex = smp.extras
    Q, A, grp = smp.Q, ex["A"], ex["group"]
    eta, r2 = ols_eta(Q, smp.X)
    at_ref = np.isclose(A, ex["a_ref"])
    print(f"states={states} ref={ref}: n={smp.n}, p={smp.p}")
    print(f"  R^2(Q on X) = {r2:.3f}; sd(Q) = {Q.std():.3f}; sd(eta) = {eta.std():.3f}")
    _, r2log = ols_eta(np.log(np.maximum(Q, 1e-3)), smp.X)
    print(f"  R^2(log Q on X) = {r2log:.3f}")
    print(f"  share at reference dose = {at_ref.mean():.3f}; formula = published for "
          f"{ex['dose_match'].mean():.3f}")
    for g in sorted(set(grp)):
        m = grp == g
        e_g, r2_g = ols_eta(Q[m], smp.X[m][:, smp.X[m].std(0) > 0])
        print(f"  {g:9s} n={m.sum():6d} R^2={r2_g:.3f} sd(eta)={e_g.std():.3f} "
              f"at ref={at_ref[m].mean():.3f}")
    # eta by dose decile (the reference level is its own row)
    rows = [{"bin": "reference", "n": int(at_ref.sum()),
             "A_range": f"{A[at_ref].min():.3f}" if at_ref.any() else "",
             **{f"eta_q{int(100 * p)}": np.quantile(eta[at_ref], p) for p in (.1, .5, .9)}}]
    off = ~at_ref
    edges = np.quantile(A[off], np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(edges, A[off], side="right") - 1, 0, 9)
    for k in range(10):
        m = b == k
        if not m.any():
            continue
        rows.append({"bin": f"decile {k + 1}", "n": int(m.sum()),
                     "A_range": f"{A[off][m].min():.3f}-{A[off][m].max():.3f}",
                     **{f"eta_q{int(100 * p)}": np.quantile(eta[off][m], p) for p in (.1, .5, .9)}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # bunching at kinks vs placebo points, in dollars of the score
    s = ex["score_dollars"]
    out = []
    for g in sorted(set(grp)):
        m = grp == g
        par = GROUPS[g]
        for name in ("s_floor", "s_cap"):
            k = par[name]
            if k is None:
                continue
            lo, hi, r = bunching(s[m], k)
            plac = [bunching(s[m], k * f)[2] for f in (0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2)]
            out.append({"group": g, "kink": name, "at": k, "below": lo, "above": hi,
                        "ratio": r, "placebo_min": np.nanmin(plac), "placebo_max": np.nanmax(plac)})
    print("bunching (+-2% windows of the score; placebo kinks at 0.80-1.20 x the kink):")
    print(pd.DataFrame(out).round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="*", default=["WA", "IL"])
    ap.add_argument("--ref", default="cap", choices=["cap", "floor"])
    args = ap.parse_args()
    pd.set_option("display.width", 160)
    report(tuple(args.states), args.ref)


if __name__ == "__main__":
    main()
