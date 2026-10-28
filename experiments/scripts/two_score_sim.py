"""Simulation check of the two-score pairwise estimator (exploratory; see two_score.py).

Designs: ``symmetric`` and ``complementary`` from two_score_population.py, which have
well-curved interior optima for the both-cutoffs rule. Data:

    X ~ N(0, I_3), eta ~ N(0, Sigma_eta), Q = G X + eta, D = 1{Q_1 > 0, Q_2 > 0},
    Y = b(eta) + X'beta1 + D {alpha(eta) + X'beta2} + N(0, 1),
    b(eta) = 0.5 eta_1 - 0.3 eta_2 + 0.2 eta_1 eta_2,  beta1 = (0.3, -0.2, 0.1).

Rule classes: both cutoffs, either cutoff, a weighted average lam Q1 + (1 - lam) Q2 >= phi
(lam chosen) and the same with lam fixed at 1/2. Targets come from the population code
(quadrature). Estimators: the oracle (true eta, T, alpha, beta2, pi0) and the feasible
rotated estimator with a tensor-product or an additive alpha.

Run from ``code/``:
    python -m experiments.scripts.two_score_sim --n 16000 64000 --reps 200 100 --workers 6
    python -m experiments.scripts.two_score_sim --n 16000 64000 256000 --reps 200 100 50 \
        --estimators tensor tensor_cv --tag _cv --workers 6     # penalty check
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.methods.two_score import (
    Pairs, Rule, TwoScoreProblem, best_halfplane, best_two_cutoffs, estimate, make_folds)
from experiments.scripts.two_score_population import DESIGNS, TwoScorePopulation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "two_score_sim"
BETA1 = np.array([0.3, -0.2, 0.1])
BOX2 = ((-3.0, 3.0), (-3.0, 3.0))
HALF = {"lam_range": (0.05, 0.95), "box": (-3.0, 3.0)}
CLASSES = {
    "both": {"kind": "both", "box": BOX2},
    "either": {"kind": "either", "box": BOX2},
    "half": {"kind": "halfplane", **HALF},
    "half_fixed": {"kind": "halfplane", "lam_fixed": 0.5, **HALF},
}


def b_fun(e):
    return 0.5 * e[:, 0] - 0.3 * e[:, 1] + 0.2 * e[:, 0] * e[:, 1]


# ---------------------------------------------------------------- population targets

def _u_half(pop, lam, phi):
    r = np.hypot(lam, 1 - lam)
    return float(pop.U_comp(np.arctan2(1 - lam, lam), phi / r))


def _u_either(pop, p):
    return (float(pop.U_comp(0.0, p[0])) + float(pop.U_comp(np.pi / 2, p[1]))
            - float(pop.U_conj(np.asarray(p))))


def _hess(f, x, h=1e-3):
    x = np.asarray(x, float)
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def targets(name: str) -> dict:
    d = DESIGNS[name]
    pop = TwoScorePopulation(d)
    out = {}
    specs = {
        "both": (lambda p: float(pop.U_conj(np.asarray(p))), (-0.4, -0.4)),
        "either": (lambda p: _u_either(pop, p), (0.5, 0.5)),
        "half": (lambda p: _u_half(pop, p[0], p[1]), (0.5, 0.1)),
    }
    for k, (f, x0) in specs.items():
        r = minimize(lambda p: -f(p), x0, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000})
        H = _hess(f, r.x)
        out[k] = {"theta": r.x.tolist(), "U": -float(r.fun), "H": H.tolist(),
                  "eig": np.linalg.eigvalsh(H).tolist()}
    g = lambda phi: _u_half(pop, 0.5, phi)
    r = minimize(lambda p: -g(p[0]), [0.1], method="Nelder-Mead", options={"xatol": 1e-8})
    out["half_fixed"] = {"theta": [0.5, float(r.x[0])], "U": -float(r.fun)}
    out["U_deployed"] = float(pop.U_conj(np.asarray(d.phi0)))
    return out


# ---------------------------------------------------------------- data and estimators

def generate(name: str, n: int, seed: int):
    d = DESIGNS[name]
    pop = TwoScorePopulation(d)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, pop.G.shape[1]))
    eta = rng.standard_normal((n, 2)) @ np.linalg.cholesky(pop.Seta).T
    T = X @ pop.G.T
    Q = T + eta
    D = ((Q[:, 0] > d.phi0[0]) & (Q[:, 1] > d.phi0[1])).astype(float)
    W = pop.alpha(eta) + X @ pop.beta2
    Y = b_fun(eta) + X @ BETA1 + D * W + rng.standard_normal(n)
    return pop, X, eta, T, Q, Y


def oracle(pop, X, eta, T, seed):
    d = pop.d
    res = {k: [] for k in CLASSES}
    for folds in make_folds(len(X), seed):
        u_idx, x_idx = folds["utility"], folds["index"]
        e = eta[u_idx]
        p0 = pop.pi0(e)
        I = ((p0 >= d.eps) & (p0 <= 1 - d.eps)).astype(float)
        pr = Pairs(e=e, u=I * (pop.alpha(e) - d.cost), I=I, T=T[x_idx], x=X[x_idx] @ pop.beta2,
                   n_u=len(e), n_x=len(x_idx))
        for k, spec in CLASSES.items():
            if spec["kind"] == "halfplane":
                r = best_halfplane(pr, spec["lam_range"], spec["box"], lam_fixed=spec.get("lam_fixed"))
            else:
                r = best_two_cutoffs(pr, spec["kind"], spec["box"])
            res[k].append(r["theta"])
    return {k: np.mean(v, 0).tolist() for k, v in res.items()}


# feasible estimators: name -> (alpha form, penalty); "cv" = cross-validated penalty
FEASIBLE = {"tensor": ("tensor", 0.03), "additive": ("additive", 0.03),
            "tensor_cv": ("tensor", "cv"), "additive_cv": ("additive", "cv")}


def replicate(task):
    name, n, seed, estimators = task
    pop, X, eta, T, Q, Y = generate(name, n, seed)
    d = pop.d
    out = {"seed": seed, "oracle": oracle(pop, X, eta, T, seed + 1)}
    prob = TwoScoreProblem(Q=Q, X=X, Y=Y, deployed=Rule("both", tuple(d.phi0)),
                           cost=d.cost, eps=d.eps)
    for est in estimators:
        form, pen = FEASIBLE[est]
        r = estimate(prob, CLASSES, seed=seed + 1, form=form, pen=pen)
        out[est] = {k: r[k]["theta"] for k in CLASSES}
        out[est + "_edges"] = {k: r[k]["edge_rotations"] for k in CLASSES}
        out[est + "_m"] = r["m"]
        out[est + "_pen"] = r["pen"]
    return out


def summarize(rows, n, tg, estimators):
    s = {}
    for est in ("oracle", *estimators):
        s[est] = {}
        for k in CLASSES:
            th = np.array([r[est][k] for r in rows])
            truth = np.array(tg[k]["theta"])
            bias = th.mean(0) - truth
            s[est][k] = {"mean": th.mean(0).round(4).tolist(), "truth": truth.round(4).tolist(),
                         "sqrt_n_bias": (np.sqrt(n) * bias).round(3).tolist(),
                         "sd": th.std(0, ddof=1).round(4).tolist(),
                         "n_var": (n * th.var(0, ddof=1)).round(2).tolist(),
                         "mc_se_bias": (th.std(0, ddof=1) / np.sqrt(len(th))).round(4).tolist()}
        if est != "oracle":
            pens = np.array([r[est + "_pen"] for r in rows]).ravel()
            s[est]["pen_chosen"] = {str(v): int(c) for v, c in zip(*np.unique(pens, return_counts=True))}
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["symmetric", "complementary"])
    ap.add_argument("--n", nargs="+", type=int, default=[16000, 64000])
    ap.add_argument("--reps", nargs="+", type=int, default=[200, 100])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--estimators", nargs="+", default=["tensor", "additive"],
                    choices=list(FEASIBLE))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in a.designs:
        tg = targets(name)
        summary[name] = {"targets": tg}
        print(name, json.dumps({k: v.get("theta") for k, v in tg.items() if isinstance(v, dict)}))
        for n, reps in zip(a.n, a.reps):
            t0 = time.time()
            tasks = [(name, n, a.seed + 1000 * r + n, tuple(a.estimators)) for r in range(reps)]
            with ProcessPoolExecutor(a.workers) as ex:
                rows = list(ex.map(replicate, tasks))
            (OUT / f"rows_{name}_n{n}{a.tag}.json").write_text(json.dumps(rows))
            summary[name][str(n)] = summarize(rows, n, tg, a.estimators)
            print(f"  n={n} reps={reps} {time.time() - t0:.0f}s")
            print(json.dumps(summary[name][str(n)], indent=None)[:3000])
    (OUT / f"summary{a.tag}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
