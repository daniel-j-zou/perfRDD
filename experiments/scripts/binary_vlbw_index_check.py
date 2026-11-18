"""Check the index form of the VLBW logit where each arm is observed (Claude, 2026-11-17).

The outcome model is E{Y(d) | X, eta} = F(c_d(eta) + X'delta_d): no interaction of the residual
and the covariates on the link scale. That form is what carries each arm to covariate values
the arm does not reach at a residual (manuscript/binary_outcome_vlbw.tex, Section 2.1). It
cannot be checked where it is used, but it can be checked within each arm, on the covariates
the arm does reach. This script fits, on the partially linear fold of one rotation (seed 1),
the model of the routes (no penalty) and two larger models that add, within each arm,
  V1: a gestation slope that varies with the residual (B-spline in eta times gestation);
  V2: a linear eta-by-covariate interaction for every covariate;
reports likelihood-ratio statistics, and compares, on the evaluation fold, the fitted effect
F(a1) - F(a0) of the infants a higher or lower line would move (weights 1500-1800 g, untreated,
and 1200-1500 g, treated): that effect is extrapolated, so its sensitivity to the added terms
shows how much the conclusions lean on the index form.

    python -m experiments.scripts.binary_vlbw_index_check --variant pooled
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit
from scipy.stats import chi2

from experiments.scripts import binary_outcome_routes as R

M = R.M


def newton_logit(A, Y, iters=60):
    """Unpenalized logistic regression by Newton steps with step halving."""
    p = A.shape[1]
    theta = np.zeros(p)
    m0 = float(np.clip(Y.mean(), 1e-3, 1 - 1e-3))
    ll_old = -np.inf
    for _ in range(iters):
        u = A @ theta
        ll = float(np.sum(Y * u - np.logaddexp(0, u)))
        grad = A.T @ (Y - expit(u))
        w = expit(u) * (1 - expit(u))
        H = (A * w[:, None]).T @ A
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        while t > 1e-6:
            th = theta + t * step
            u2 = A @ th
            ll2 = float(np.sum(Y * u2 - np.logaddexp(0, u2)))
            if ll2 >= ll - 1e-10:
                break
            t /= 2
        theta = th
        if abs(ll2 - ll_old) < 1e-10 * (1 + abs(ll2)):
            break
        ll_old = ll2
    _ = m0
    return theta, ll2


def design(D, B, Zb, Xe, es, variant):
    """Columns: arm splines, shared baseline slope, effect slope, plus the variant's terms."""
    cols = [(1 - D)[:, None] * B, D[:, None] * B, Zb, D[:, None] * Xe]
    if variant == "V1":           # gestation slope varying with eta, each arm (drop one basis column)
        Bg = B[:, 1:] * Xe[:, :1]
        cols += [(1 - D)[:, None] * Bg, D[:, None] * Bg]
    elif variant == "V2":         # linear eta x every covariate, each arm
        cols += [(1 - D)[:, None] * es[:, None] * Zb, D[:, None] * es[:, None] * Zb]
    return np.column_stack(cols)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pooled")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/binary_vlbw_index_check"))
    a = ap.parse_args(argv)
    data, dsg, meta = R.build_vlbw(a.variant, eps=0.1)
    blocks = np.array_split(np.random.default_rng(a.seed).permutation(len(data.Q)), 4)
    folds = {r: blocks[j] for j, r in enumerate(M.ROLES)}
    g = np.linalg.lstsq(M._xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
    eta = data.Q - M._xt(data.Z) @ g
    T = M._xt(data.Z) @ g
    l_hat, u_hat = R._window(data, dsg, eta, folds, T[folds["index"]])
    keep, D, B, info = R._design_matrix(data, eta, folds["outcome"], dsg.J)
    sd = float(np.std(eta[keep]))
    mu_e = float(np.mean(eta[keep]))
    es = (eta[keep] - mu_e) / sd
    Zb, Xe, Y = data.baseline[keep], data.X[keep], data.Y[keep]

    # evaluation fold, window units
    U = folds["utility"]
    U = U[(eta[U] >= l_hat) & (eta[U] <= u_hat)]
    wU = -data.Q[U]                       # grams
    BU = BSpline.design_matrix(np.clip(eta[U], info["lo"], info["hi"]), info["t"], info["degree"]).toarray()
    esU = (eta[U] - mu_e) / sd
    ZU, XU = data.baseline[U], data.X[U]
    groups = {"untreated 1500-1800 g": (wU >= 1500) & (wU < 1800),
              "treated 1200-1500 g": (wU >= 1200) & (wU < 1500),
              "near the line (1450-1550 g)": (wU >= 1450) & (wU < 1550)}

    out = {"variant": a.variant, "window": [l_hat, u_hat], "n_fit": int(len(Y)),
           "n_treated_fit": int(D.sum()), "models": {}}
    ll0 = None
    for v in ("additive", "V1", "V2"):
        A = design(D, B, Zb, Xe, es, v)
        theta, ll = newton_logit(A, Y)
        rec = {"p": int(A.shape[1]), "loglik": ll}
        if v == "additive":
            ll0, p0 = ll, A.shape[1]
        else:
            lr = 2 * (ll - ll0)
            df = A.shape[1] - p0
            rec.update(LR=lr, df=int(df), p_value=float(chi2.sf(lr, df)),
                       deviance_gain_per_1000=1000 * lr / len(Y))
        a1 = design(np.ones(len(U)), BU, ZU, XU, esU, v) @ theta
        a0 = design(np.zeros(len(U)), BU, ZU, XU, esU, v) @ theta
        eff = expit(a1) - expit(a0)
        rec["mean_effect_points"] = {k: float(100 * eff[m].mean()) for k, m in groups.items()}
        rec["n_group"] = {k: int(m.sum()) for k, m in groups.items()}
        out["models"][v] = rec
        msg = f"{v:9s} p={rec['p']:3d} loglik={ll:.1f}"
        if v != "additive":
            msg += f" LR={rec['LR']:.1f} df={rec['df']} p={rec['p_value']:.2g}"
        print(msg + "  effect (points): " + ", ".join(f"{k} {x:+.2f}" for k, x in rec["mean_effect_points"].items()),
              flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"index_{a.variant}_seed{a.seed}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
