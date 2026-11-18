"""Appendix 6 of ``differing_slopes.tex`` with the estimator of Section 3 and 500 data sets.

Appendix 6 ("Differing-slopes simulation") checks what a common-slope fit gets wrong. Its
numbers came from ``differing_slopes_full_pipeline.py``, an earlier pipeline with outcome
fits linear in eta, 100 (baseline) and 75 (quadratic) replications and n <= 4,000. This
script reruns both designs (author, 2026-11-18, item 5 of the predicted referee concerns)
with the estimator the paper now uses everywhere else:

* designs: those of ``differing_slopes_full_pipeline`` (X ~ N(0, I_2), T = X1,
  eta ~ N(0, 1), Q = T + eta, D = 1{Q > 0}, Y = 0.2 + 0.6 eta + X'(0.3, -0.2)
  + D {0.35 + 0.9 eta + X'(0.8, 0.25) + q2 (X1^2 - 1)} + N(0, 0.5^2), c = 0.25,
  eps = 0.1, Phi = [-3, 3]); ``baseline`` has q2 = 0 and ``quadratic`` q2 = 0.9;
  targets -0.119770 and 0.484416 (``population_truth`` of that script);
* estimator: the four-role split with rotation, OLS first stage, spline g and rho_X on
  T = [-3, 3], outcome splines in eta on J = [-1.75, 1.75], window from the index fold
  (capped at the central 99% of eta-hat, which does not bind here), rotated average and
  pooled maximizer, as in ``ds_review_ablations.estimate``;
* two outcome fits on the same data and split: ``dx`` (arm-wise splines and arm-wise X
  slopes: the D X block of the paper) and ``common`` (arm-wise splines, one X slope for
  both arms: the common-slope model, beta2 = 0 in the utility);
* large-sample limits (``--limits``): the argmax of the pseudo-utility built from the
  population projection of each outcome fit (Monte Carlo normal equations on the exact
  conditional mean of Y).

Point estimates only (no bootstrap): bias, standard deviation and RMSE of each fit, n =
1,000 to 64,000, 500 data sets with common random numbers across fits and designs.

Run (from code/):
    python -m experiments.scripts.ds_appendix6_rerun --workers 8
    python -m experiments.scripts.ds_appendix6_rerun --limits
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.stats import norm

from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_four_role_coverage import SplineTailPieces, _avg_and_pooled
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, Sample, _ls, _n_interior, _xt, ols_first_stage, outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_appendix6_rerun"
SEED = 20260925                     # the seed of differing_slopes_full_pipeline
PHI0, EPS, COST = 0.0, 0.10, 0.25
J, T_SUPPORT, POLICY = (-1.75, 1.75), (-3.0, 3.0), (-3.0, 3.0)
WINDOW_TAIL = 0.005
BETA1, BETA2 = np.array([0.30, -0.20]), np.array([0.80, 0.25])
A0, A1, B0, B1, SIG = 0.35, 0.90, 0.20, 0.60, 0.50
DESIGNS = {"baseline": 0.0, "quadratic": 0.90}
TARGETS = {"baseline": -0.119770, "quadratic": 0.484416}   # population_truth, 6 decimals
FITS = ("dx", "common")
EDGE_TOL = 1e-3


def generate(q2: float, n: int, seed: int) -> Sample:
    """The draws of ``differing_slopes_full_pipeline.generate_sample`` (normal errors)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    eta = rng.standard_normal(n)
    Q = X[:, 0] + eta
    D = (Q > PHI0).astype(float)
    effect = A0 + A1 * eta + X @ BETA2 + q2 * (X[:, 0] ** 2 - 1.0)
    Y = B0 + B1 * eta + X @ BETA1 + D * effect + rng.normal(0.0, SIG, n)
    return Sample(X=X, Q=Q, D=D, Y=Y)


@dataclass(frozen=True)
class CommonFit:
    info: dict
    w0: np.ndarray
    w1: np.ndarray

    def a(self, e):
        e = np.clip(e, self.info["lo"], self.info["hi"])
        return BSpline(self.info["t"], self.w1 - self.w0, self.info["degree"],
                       extrapolate=False)(e)

    @property
    def beta2(self):
        return np.zeros(2)


def common_fit(s: Sample, idx: np.ndarray, eta_hat: np.ndarray) -> CommonFit:
    """Arm-wise splines in eta on J with one X slope for both arms (no D X block); the
    knots are those of ``outcome_fit``."""
    e = eta_hat[idx]
    fit = idx[(e >= J[0]) & (e <= J[1])]
    e, X, D, Y = eta_hat[fit], s.X[fit], s.D[fit], s.Y[fit]
    n1 = int(D.sum())
    info = _basis_params(_n_interior(min(n1, len(fit) - n1)), J)
    B = BSpline.design_matrix(np.clip(e, info["lo"], info["hi"]), info["t"],
                              info["degree"]).toarray()
    nb = B.shape[1]
    coef = _ls(np.column_stack(((1 - D)[:, None] * B, D[:, None] * B, X)), Y)
    return CommonFit(info=info, w0=coef[:nb], w1=coef[nb:2 * nb])


def estimate(s: Sample, kind: str, blocks) -> dict:
    grid = np.linspace(POLICY[0], POLICY[1], 61)
    crit, vals, rots = [], [], []
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        n_u = len(folds["utility"])
        gamma_hat = ols_first_stage(s, folds["gamma"])
        eta_hat = s.Q - _xt(s.X) @ gamma_hat
        T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], T_SUPPORT)
        l_hat = PHI0 - float(np.quantile(T_idx, 1.0 - EPS))
        u_hat = PHI0 - float(np.quantile(T_idx, EPS))
        e = eta_hat[folds["utility"]]
        l_hat = max(l_hat, float(np.quantile(e, WINDOW_TAIL)))
        u_hat = min(u_hat, float(np.quantile(e, 1.0 - WINDOW_TAIL)))
        e = e[(e >= l_hat) & (e <= u_hat)]
        fit = (outcome_fit(s, folds["outcome"], eta_hat, J) if kind == "dx"
               else common_fit(s, folds["outcome"], eta_hat))
        pieces = SplineTailPieces(tails, e, n_u)
        amc = fit.a(e) - COST
        b2 = np.asarray(fit.beta2)
        row = []
        for x in grid:
            (sg,), sh = pieces.sums(x, [amc])
            row.append((sg + float(sh @ b2)) / n_u)
        crit.append(lambda x, p=pieces, a=amc, b=b2: p.value(x, a, b))
        vals.append(row)
    avg, pooled, rot = _avg_and_pooled(crit, grid, np.array(vals))
    return {"avg": avg, "pooled": pooled, "rotations": rot}


def replicate(task) -> dict:
    design, n, rep, seed = task
    t0 = time.perf_counter()
    s = generate(DESIGNS[design], n, seed)
    blocks = np.array_split(np.random.default_rng([seed, 1]).permutation(n), 4)
    out = {"design": design, "n": n, "rep": rep, "seed": seed}
    for kind in FITS:
        out[kind] = estimate(s, kind, blocks)
    out["seconds"] = time.perf_counter() - t0
    return out


def summarize(rows, design) -> dict:
    phi = TARGETS[design]
    out = {}
    for kind in FITS:
        for arm in ("avg", "pooled"):
            est = np.array([r[kind][arm] for r in rows])
            err = est - phi
            n = rows[0]["n"]
            out[f"{kind}_{arm}"] = {
                "mean": float(est.mean()), "bias": float(err.mean()),
                "bias_se": float(err.std(ddof=1) / np.sqrt(len(err))),
                "sd": float(est.std(ddof=1)), "rmse": float(np.sqrt(np.mean(err ** 2))),
                "sqrt_n_bias": float(np.sqrt(n) * err.mean()), "n_var": float(n * err.var(ddof=1)),
                "share_rotation_at_edge": float(np.mean([
                    min(abs(p - POLICY[0]), abs(p - POLICY[1])) < EDGE_TOL
                    for r in rows for p in r[kind]["rotations"]]))}
    return out


# ------------------------------------------------------------ large-sample limits

def utility(phi, alpha, slope1):
    """int_W [(alpha(e) - c) Gbar(phi - e) + slope1 g(phi - e)] dPhi(e); T = X1 ~ N(0, 1),
    E[X 1{T > t}] = (g(t), 0), and slope1 is the X1 slope of the effect."""
    z = norm.ppf(1.0 - EPS)
    f = lambda e: ((alpha(e) - COST) * norm.sf(phi - e) + slope1 * norm.pdf(phi - e)) * norm.pdf(e)
    return quad(f, -z, z, limit=400, epsabs=1e-13, epsrel=1e-12)[0]


def true_utility(phi, q2):
    """The true utility, with E[(X1^2 - 1) 1{X1 > t}] = t g(t) for the quadratic term."""
    z = norm.ppf(1.0 - EPS)
    f = lambda e: ((A0 + A1 * e - COST) * norm.sf(phi - e) + BETA2[0] * norm.pdf(phi - e)
                   + q2 * (phi - e) * norm.pdf(phi - e)) * norm.pdf(e)
    return quad(f, -z, z, limit=400, epsabs=1e-13, epsrel=1e-12)[0]


def argmax(U) -> float:
    grid = np.linspace(POLICY[0], POLICY[1], 241)
    v = np.array([U(x) for x in grid])
    k = int(np.argmax(v))
    res = optimize.minimize_scalar(lambda x: -U(x), bounds=(grid[max(k - 1, 0)],
                                                           grid[min(k + 1, len(grid) - 1)]),
                                   method="bounded", options={"xatol": 1e-10})
    return float(res.x)


def limits(K=20, n_mc=1_000_000, chunks=16, seed=11) -> dict:
    info = _basis_params(K, J)
    out = {}
    for design, q2 in DESIGNS.items():
        acc = {kind: [None, None] for kind in FITS}
        for c in range(chunks):
            rng = np.random.default_rng([seed, c])
            X = rng.standard_normal((n_mc, 2))
            eta = rng.standard_normal(n_mc)
            D = (X[:, 0] + eta > PHI0).astype(float)
            keep = (eta >= J[0]) & (eta <= J[1])
            X, eta, D = X[keep], eta[keep], D[keep]
            ym = (B0 + B1 * eta + X @ BETA1
                  + D * (A0 + A1 * eta + X @ BETA2 + q2 * (X[:, 0] ** 2 - 1.0)))
            B = BSpline.design_matrix(np.clip(eta, info["lo"], info["hi"]), info["t"],
                                      info["degree"]).toarray()
            mats = {"dx": np.column_stack(((1 - D)[:, None] * B, (1 - D)[:, None] * X,
                                           D[:, None] * B, D[:, None] * X)),
                    "common": np.column_stack(((1 - D)[:, None] * B, D[:, None] * B, X))}
            for kind, M in mats.items():
                a, b = acc[kind]
                acc[kind] = [M.T @ M if a is None else a + M.T @ M,
                             M.T @ ym if b is None else b + M.T @ ym]
        nb = B.shape[1]
        res = {"target": TARGETS[design]}
        for kind in FITS:
            coef = np.linalg.solve(*acc[kind])
            if kind == "dx":
                w0, d0, w1, d1 = coef[:nb], coef[nb:nb + 2], coef[nb + 2:2 * nb + 2], coef[2 * nb + 2:]
                slope1 = float(d1[0] - d0[0])
            else:
                w0, w1 = coef[:nb], coef[nb:2 * nb]
                slope1 = 0.0
            spl = BSpline(info["t"], w1 - w0, info["degree"], extrapolate=False)
            alpha = lambda e, s=spl: float(s(np.clip(e, info["lo"], info["hi"])))
            lim = argmax(lambda p: utility(p, alpha, slope1))
            phi = TARGETS[design]
            gain = true_utility(phi, q2) - true_utility(PHI0, q2)
            res[kind] = {"limit": lim, "beta2_x1": slope1,
                         "gain_lost_share": (true_utility(phi, q2) - true_utility(lim, q2)) / gain}
        out[design] = res
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 4000, 16000, 64000])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limits", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.limits:
        res = limits()
        print(json.dumps(res, indent=1))
        (args.out / "limits.json").write_text(json.dumps(res, indent=1) + "\n")
        return
    path = args.out / "rows.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done = {(r["design"], r["n"], r["rep"]) for r in rows}
    tasks = [(d, n, r, args.seed + 104_729 * n + r) for d in DESIGNS for n in args.n
             for r in range(args.reps) if (d, n, r) not in done]
    if tasks:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.workers) as pool, path.open("a") as fh:
            futures = [pool.submit(replicate, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if i % 100 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}, {time.perf_counter() - t0:.0f}s", flush=True)
    summary = {"settings": {"reps": args.reps, "seed": args.seed, "n": args.n},
               "targets": TARGETS, "cells": {}}
    for d in DESIGNS:
        for n in args.n:
            sub = [r for r in rows if r["design"] == d and r["n"] == n and r["rep"] < args.reps]
            summary["cells"][f"{d}_n{n}"] = {"reps": len(sub), **summarize(sub, d)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("| design | n | fit | mean | bias (s.e.) | sd | RMSE | sqrt(n) bias |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for key, cell in summary["cells"].items():
        d, n = key.rsplit("_n", 1)
        for kind in FITS:
            c = cell[f"{kind}_avg"]
            print(f"| {d} | {int(n):,} | {kind} | {c['mean']:.4f} | {c['bias']:+.4f} "
                  f"({c['bias_se']:.4f}) | {c['sd']:.4f} | {c['rmse']:.4f} | {c['sqrt_n_bias']:+.2f} |")


if __name__ == "__main__":
    main()
