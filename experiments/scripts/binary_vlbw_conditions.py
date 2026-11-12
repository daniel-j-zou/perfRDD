"""Check the single-crossing conditions of the logit routes on the fitted VLBW model.

One rotation of the four-role split (seed 1) on the pooled 1995-2002 sample: fit the score
regression, the window and the logit on both arms, then report on the window

  (i)   c0(e) nonincreasing and c1(e) - c0(e) nondecreasing in the residual e;
  (iii) the unit-level gap Delta = a1 - a0 >= 0 and the mean logit a0 + Delta/2 >= 0;
  (V2)  Lambda(e, t) = E{W(X, e) | T = t} nondecreasing in e and in t, with the margins
        min d_e Lambda and min d_t Lambda for (V3);
  (ii)  the conditional means of X'beta1 and X_e'beta2 by bin of the linear predictor
        (a necessary check only).

The score is mirrored (Q = -weight), so a larger residual is a lighter infant than its
covariates predict and a larger T is a lower predicted weight. Section 6.5 of
manuscript/binary_outcome_vlbw.tex states the conditions.

    python -m experiments.scripts.binary_vlbw_conditions --variant wide
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.scripts import binary_outcome_routes as R

M = R.M


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="wide", help="pooled (1200-1800 g) or wide (1200-2200 g)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--tbins", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/binary_vlbw_conditions"))
    a = ap.parse_args(argv)
    data, design, meta = R.build_vlbw(a.variant, eps=0.1)
    blocks = np.array_split(np.random.default_rng(a.seed).permutation(len(data.Q)), 4)
    folds = {r: blocks[j] for j, r in enumerate(M.ROLES)}
    g = np.linalg.lstsq(M._xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
    eta = data.Q - M._xt(data.Z) @ g
    T = M._xt(data.Z) @ g
    T_idx = T[folds["index"]]
    l_hat, u_hat = R._window(data, design, eta, folds, T_idx)
    fit = R.fit_logit(data, eta, folds["outcome"], design)
    phiL, phiU = design.policy
    vlo, vhi = phiL - u_hat, phiU - l_hat              # the argument set V_Phi
    out = {"variant": a.variant, "n": int(len(data.Q)), "window": [l_hat, u_hat],
           "V_Phi": [vlo, vhi], "policy_grams": [-phiU, -phiL]}

    # (i) arm intercepts on the window
    eg = np.linspace(l_hat, u_hat, a.grid)
    c0, c1 = fit.c(eg, 0), fit.c(eg, 1)
    gap = c1 - c0
    out["i_c0_nonincreasing_share"] = float(np.mean(np.diff(c0) <= 1e-9))
    out["i_gap_nondecreasing_share"] = float(np.mean(np.diff(gap) >= -1e-9))
    out["i_c0_range"] = [float(c0.min()), float(c0.max())]
    out["i_gap_range"] = [float(gap.min()), float(gap.max())]
    out["i_c0_first_last"] = [float(c0[0]), float(c0[-1])]
    out["i_gap_first_last"] = [float(gap[0]), float(gap[-1])]

    # (iii) unit level, on the evaluation fold, units in the window with T in V_Phi
    U = folds["utility"]
    sel = U[(eta[U] >= l_hat) & (eta[U] <= u_hat) & (T[U] >= vlo) & (T[U] <= vhi)]
    zb, x, e = data.baseline[sel], data.X[sel], eta[sel]
    a0 = fit.c(e, 0) + zb @ fit.beta1
    a1 = fit.c(e, 1) + zb @ fit.beta1 + x @ fit.beta2
    delta = a1 - a0
    out["iii_n"] = int(len(sel))
    out["iii_share_delta_nonneg"] = float(np.mean(delta >= 0))
    out["iii_share_meanlogit_nonneg"] = float(np.mean(a0 + delta / 2 >= 0))
    out["iii_share_both"] = float(np.mean((delta >= 0) & (a0 + delta / 2 >= 0)))
    out["iii_share_a0_nonneg"] = float(np.mean(a0 >= 0))
    out["iii_delta_quantiles"] = [float(q) for q in np.quantile(delta, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95])]

    # (V2) Lambda(e, t) = E{W(X, e) | T = t}: average W over density-fold units in bins of T
    I = folds["index"]
    TI = T[I]
    inV = (TI >= vlo) & (TI <= vhi)
    I, TI = I[inV], TI[inV]
    edges = np.quantile(TI, np.linspace(0, 1, a.tbins + 1))
    tb = np.clip(np.searchsorted(edges, TI, side="right") - 1, 0, a.tbins - 1)
    tmid = np.array([TI[tb == b].mean() for b in range(a.tbins)])
    zbI, xI = data.baseline[I], data.X[I]
    Lam = np.empty((a.grid, a.tbins))
    for i, ei in enumerate(eg):
        W = fit.effect(np.full(len(I), ei), zbI, xI)
        Lam[i] = np.bincount(tb, weights=W, minlength=a.tbins) / np.bincount(tb, minlength=a.tbins)
    de = np.diff(Lam, axis=0) / np.diff(eg)[:, None]
    dt = np.diff(Lam, axis=1) / np.diff(tmid)[None, :]
    out["V2_share_nondecreasing_in_e"] = float(np.mean(de >= -1e-12))
    out["V2_share_nondecreasing_in_t"] = float(np.mean(dt >= -1e-12))
    out["V3_min_de"] = float(de.min())
    out["V3_min_dt"] = float(dt.min())
    out["V3_median_de"] = float(np.median(de))
    out["V3_median_dt"] = float(np.median(dt))
    out["Lambda_range"] = [float(Lam.min()), float(Lam.max())]
    out["Lambda_corners"] = {"e_lo_t_lo": float(Lam[0, 0]), "e_lo_t_hi": float(Lam[0, -1]),
                             "e_hi_t_lo": float(Lam[-1, 0]), "e_hi_t_hi": float(Lam[-1, -1])}
    out["Lambda_share_negative"] = float(np.mean(Lam < 0))

    # (ii) conditional means by bin of T (necessary check)
    v1 = zbI @ fit.beta1
    ve = xI @ fit.beta2
    m1 = np.bincount(tb, weights=v1, minlength=a.tbins) / np.bincount(tb, minlength=a.tbins)
    me = np.bincount(tb, weights=ve, minlength=a.tbins) / np.bincount(tb, minlength=a.tbins)
    out["ii_mean_xbeta1_by_tbin_first_last"] = [float(m1[0]), float(m1[-1])]
    out["ii_mean_xebeta2_by_tbin_first_last"] = [float(me[0]), float(me[-1])]
    out["ii_xbeta1_nonincreasing_share"] = float(np.mean(np.diff(m1) <= 1e-12))
    out["ii_xebeta2_nondecreasing_share"] = float(np.mean(np.diff(me) >= -1e-12))
    out["beta2"] = [float(b) for b in np.atleast_1d(fit.beta2)]

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"conditions_{a.variant}_seed{a.seed}.json").write_text(json.dumps(out, indent=1))
    np.savez(a.out / f"lambda_{a.variant}_seed{a.seed}.npz", e=eg, t=tmid, Lam=Lam, c0=c0, c1=c1)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
