"""Does the ridge penalty of fit_logit bias the VLBW arm fit? (Claude, 2026-11-17)

The routes fit the logit on both arms with a ridge lam = ridge * n / p on the summed
log-likelihood (ridge = 1e-3). Per observation that penalty is 1e-3 * |theta|^2 / (2p),
which does not vanish as n grows, so the fit converges to a shrunk value. This refits one
rotation (seed 1, the folds of binary_vlbw_conditions.py) with ridge 1e-3 and with a
negligible ridge and compares the arm intercepts on the window, the effect slope, and the
mean effect near the 1500 g line on both scales.

    python -m experiments.scripts.binary_vlbw_ridge_check --variant pooled
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from experiments.scripts import binary_outcome_routes as R

M = R.M


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pooled")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ridges", type=float, nargs="+", default=[1e-3, 1e-9])
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/binary_vlbw_ridge_check"))
    a = ap.parse_args(argv)
    data, design, meta = R.build_vlbw(a.variant, eps=0.1)
    blocks = np.array_split(np.random.default_rng(a.seed).permutation(len(data.Q)), 4)
    folds = {r: blocks[j] for j, r in enumerate(M.ROLES)}
    g = np.linalg.lstsq(M._xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
    eta = data.Q - M._xt(data.Z) @ g
    T = M._xt(data.Z) @ g
    l_hat, u_hat = R._window(data, design, eta, folds, T[folds["index"]])
    eg = np.linspace(l_hat, u_hat, 41)
    U = folds["utility"]
    near = U[np.abs(data.Q[U] - design.phi0) <= 0.05 * float(np.std(data.Q))]
    out = {"variant": a.variant, "window": [l_hat, u_hat], "n_near": int(len(near)), "fits": {}}
    for ridge in a.ridges:
        fit = R.fit_logit(data, eta, folds["outcome"], design, ridge=ridge)
        c0, c1 = fit.c(eg, 0), fit.c(eg, 1)
        e_n, zb, x = eta[near], data.baseline[near], data.X[near]
        a0 = fit.c(e_n, 0) + zb @ fit.beta1
        a1 = fit.c(e_n, 1) + zb @ fit.beta1 + x @ fit.beta2
        rec = {"c0": c0.tolist(), "c1": c1.tolist(), "beta2": np.atleast_1d(fit.beta2).tolist(),
               "beta1_norm": float(np.linalg.norm(fit.beta1)),
               "effect_prob_near_line": float(np.mean(expit(a1) - expit(a0))),
               "effect_logodds_near_line": float(np.mean(a1 - a0)),
               "mean_fitted_survival_near_line": float(np.mean(np.where(data.D[near] == 1, expit(a1), expit(a0)))),
               "observed_survival_near_line": float(np.mean(data.Y[near]))}
        out["fits"][str(ridge)] = rec
        print(f"ridge {ridge:g}: beta2 {rec['beta2']}, effect near line prob {rec['effect_prob_near_line']:+.4f} "
              f"log-odds {rec['effect_logodds_near_line']:+.4f}; c0 {c0[0]:.3f}..{c0[-1]:.3f}, "
              f"c1 {c1[0]:.3f}..{c1[-1]:.3f}; fitted/observed survival near line "
              f"{rec['mean_fitted_survival_near_line']:.4f}/{rec['observed_survival_near_line']:.4f}", flush=True)
    r = [out["fits"][str(x)] for x in a.ridges]
    gap = [np.array(f["c1"]) - np.array(f["c0"]) for f in r]
    out["max_abs_diff"] = {"c0": float(np.max(np.abs(np.array(r[0]["c0"]) - np.array(r[1]["c0"])))),
                           "c1": float(np.max(np.abs(np.array(r[0]["c1"]) - np.array(r[1]["c1"])))),
                           "gap": float(np.max(np.abs(gap[0] - gap[1])))}
    print("max |difference| on the window:", out["max_abs_diff"], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"ridge_{a.variant}_seed{a.seed}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
