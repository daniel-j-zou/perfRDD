"""Where do fitted variance weights lose against the true ones?

Diagnostic companion to ``ds_local_efficiency_weighted_sim.py``. At kappa = 1 the
fully weighted estimator attains the bound with the true variance function but
loses to the paper's estimator with the fitted one (six-function least-squares
spline of squared residuals, refitted on each third of the first-stage fold and
on the outcome fold). On the same data and folds this runs

    paper          unweighted outcome fit, OLS first stage
    w_fit          both weighted, fitted variance (as in the weighted simulation)
    w_oracle       both weighted, true variance
    w_fs_oracle    true variance in the first stage, fitted in the outcome fold
    w_out_oracle   fitted variance in the first stage, true in the outcome fold
    w_glm          both weighted, log-link spline variance (six functions per arm)
                   fitted by Gamma quasi-likelihood; the first stage fits it once
                   on the whole first-stage fold instead of on each third
    w_glm_quad     as w_glm with log sigma_d^2 quadratic in eta (three parameters
                   per arm)
    w_ls_full      the least-squares spline of w_fit, but fitted once on the whole
                   first-stage fold (separates the fitting sample from the log link)

and records the rotated and fixed thresholds and the first-stage slopes of the
first rotation's fold.

Run:
    python -m experiments.scripts.ds_local_efficiency_weight_diag --n 32000 64000 \
        --reps 1000 --workers 6
    python -m experiments.scripts.ds_local_efficiency_weight_diag --kappa 0.5 --n 32000 \
        --reps 1000 --workers 6
    python -m experiments.scripts.ds_local_efficiency_weight_diag --fit-check
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from experiments.methods.perfrdd import _basis_params
from experiments.scripts.ds_local_efficiency import DESIGNS, Population
from experiments.scripts.ds_local_efficiency_hetero import hetero_variances
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES,
    OutcomeFit,
    Sample,
    _design_matrix,
    _xt,
    generate,
    ols_first_stage,
    outcome_fit,
)
from experiments.scripts.ds_local_efficiency_weighted_sim import (
    VAR_BASIS,
    OracleVariance,
    fit_variance,
    threshold,
    weighted_first_stage,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency_weight_diag"
VARIANTS = ("paper", "w_fit", "w_oracle", "w_fs_oracle", "w_out_oracle", "w_glm",
            "w_glm_quad", "w_ls_full")
SLOPE_KEYS = ("ols", "fit", "oracle", "glm", "glm_quad", "ls_full")


def _log_basis(e: np.ndarray, info) -> np.ndarray:
    e = np.clip(e, info["lo"], info["hi"])
    if info.get("kind") == "quad":
        return np.column_stack((np.ones(len(e)), e, e ** 2))
    return BSpline.design_matrix(e, info["t"], info["degree"]).toarray()


@dataclass(frozen=True)
class LogVarianceFit:
    info: dict
    coef: tuple              # per-arm coefficients for log E(R^2 | eta)

    def __call__(self, D: np.ndarray, e: np.ndarray) -> np.ndarray:
        B = _log_basis(e, self.info)
        out = np.empty(len(e))
        for arm in (0, 1):
            on = D == arm
            out[on] = np.exp(B[on] @ self.coef[arm])
        return out


def _gamma_qmle(B: np.ndarray, y: np.ndarray, iters: int = 100) -> np.ndarray:
    """Fisher scoring for E(y | B) = exp(B theta) under the Gamma quasi-likelihood."""
    def objective(th):
        lin = np.clip(B @ th, -30.0, 30.0)
        return float(np.sum(y * np.exp(-lin) + lin))

    theta = np.linalg.lstsq(B, np.full(len(y), np.log(np.mean(y))), rcond=None)[0]
    info = B.T @ B
    obj = objective(theta)
    for _ in range(iters):
        mu = np.exp(np.clip(B @ theta, -30.0, 30.0))
        step = np.linalg.solve(info, B.T @ (y / mu - 1.0))
        t, new = 1.0, np.inf
        while t > 1e-4:
            cand = theta + t * step
            new = objective(cand)
            if new <= obj:
                break
            t *= 0.5
        if new > obj:
            break
        theta, gain, obj = cand, obj - new, new
        if gain < 1e-10 * max(1.0, abs(obj)):
            break
    return theta


def fit_log_variance(s: Sample, idx, eta_hat, J, fit: OutcomeFit,
                     kind: str = "spline") -> LogVarianceFit:
    e = eta_hat[idx]
    keep = (e >= J[0]) & (e <= J[1])
    i = idx[keep]
    e, X, D, Y = eta_hat[i], s.X[i], s.D[i], s.Y[i]
    Z, _ = _design_matrix(e, X, D, fit.info)
    coef = np.concatenate([fit.w0, fit.d0, fit.w1, fit.d1])
    r2 = (Y - Z @ coef) ** 2
    info = (dict(_basis_params(VAR_BASIS - 4, J), kind="spline") if kind == "spline"
            else {"kind": "quad", "lo": J[0], "hi": J[1]})
    B = _log_basis(e, info)
    coefs = tuple(_gamma_qmle(B[D == arm], r2[D == arm]) for arm in (0.0, 1.0))
    return LogVarianceFit(info=info, coef=coefs)


def glm_variance(s: Sample, idx, J, gamma_hat, kind: str = "spline") -> LogVarianceFit:
    """Log-link variance from an unweighted pilot fit on the whole of ``idx``."""
    eta_hat = s.Q - _xt(s.X) @ gamma_hat
    pilot = outcome_fit(s, idx, eta_hat, J)
    return fit_log_variance(s, idx, eta_hat, J, pilot, kind)


def _design(name: str, kappa):
    design = DESIGNS[name]
    return design if kappa is None else replace(design, noise_kappa=float(kappa))


def replicate(task):
    name, n, seed, kappa = task
    design = _design(name, kappa)
    J = design.J
    s = generate(design, n, seed)
    oracle = OracleVariance(design)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    out = {v: [] for v in VARIANTS}
    slopes = {}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        g_ols = ols_first_stage(s, folds["gamma"])
        sub = seed + 11 * k + 3
        g_fit = weighted_first_stage(s, folds["gamma"], J, np.random.default_rng(sub))
        g_orc = weighted_first_stage(s, folds["gamma"], J, np.random.default_rng(sub), oracle)
        eta_g = s.Q - _xt(s.X) @ g_ols
        vf_ls = fit_variance(s, folds["gamma"], eta_g, J, outcome_fit(s, folds["gamma"], eta_g, J))
        g_ls = weighted_first_stage(s, folds["gamma"], J, np.random.default_rng(sub), vf_ls)
        g_glm, vf_o = {}, {}
        for kind in ("spline", "quad"):
            vf_g = glm_variance(s, folds["gamma"], J, g_ols, kind)
            g_glm[kind] = weighted_first_stage(s, folds["gamma"], J,
                                               np.random.default_rng(sub), vf_g)
            eta_o = s.Q - _xt(s.X) @ g_glm[kind]
            vf_o[kind] = fit_log_variance(s, folds["outcome"], eta_o, J,
                                          outcome_fit(s, folds["outcome"], eta_o, J), kind)
        out["paper"].append(threshold(s, design, folds, g_ols, False))
        out["w_fit"].append(threshold(s, design, folds, g_fit, True))
        out["w_oracle"].append(threshold(s, design, folds, g_orc, True, oracle))
        out["w_fs_oracle"].append(threshold(s, design, folds, g_orc, True))
        out["w_out_oracle"].append(threshold(s, design, folds, g_fit, True, oracle))
        out["w_glm"].append(threshold(s, design, folds, g_glm["spline"], True, vf_o["spline"]))
        out["w_glm_quad"].append(threshold(s, design, folds, g_glm["quad"], True, vf_o["quad"]))
        out["w_ls_full"].append(threshold(s, design, folds, g_ls, True))
        if k == 0:
            slopes = {"ols": g_ols[1:].tolist(), "fit": g_fit[1:].tolist(),
                      "oracle": g_orc[1:].tolist(), "glm": g_glm["spline"][1:].tolist(),
                      "glm_quad": g_glm["quad"][1:].tolist(), "ls_full": g_ls[1:].tolist()}
    row = {f"avg_{v}": float(np.mean(out[v])) for v in VARIANTS}
    row.update({f"fixed_{v}": out[v][0] for v in VARIANTS})
    row["slopes"] = slopes
    return row


def fit_check(design_name: str = "paper_hetero", n: int = 32000, samples: int = 40) -> dict:
    """Fitted over true variance on a grid of eta, by arm, fit size and method.

    One OLS first stage on 8000 points gives eta_hat; the variance is then fitted on
    8000 further points (an outcome fold) or on 2667 of them (a third of a fold).
    Returns the 10% and 90% quantiles over ``samples`` data sets.
    """
    design = DESIGNS[design_name]
    J, oracle = design.J, OracleVariance(design)
    grid = np.array([-1.75, -1.5, -1.0, 0.0, 1.0, 1.5, 1.75])
    methods = {"ls": lambda s, f, e, p: fit_variance(s, f, e, J, p),
               "glm": lambda s, f, e, p: fit_log_variance(s, f, e, J, p),
               "glm_quad": lambda s, f, e, p: fit_log_variance(s, f, e, J, p, "quad")}
    out = {"grid": grid.tolist()}
    for size in (8000, 2667):
        ratios = {(m, arm): [] for m in methods for arm in (0, 1)}
        for r in range(samples):
            s = generate(design, n, 1000 + r)
            idx = np.random.default_rng(r).permutation(n)
            g = ols_first_stage(s, idx[:8000])
            eta_hat = s.Q - _xt(s.X) @ g
            fold = idx[8000:8000 + size]
            pilot = outcome_fit(s, fold, eta_hat, J)
            for m, fit in methods.items():
                vf = fit(s, fold, eta_hat, pilot)
                for arm in (0, 1):
                    D = np.full(len(grid), float(arm))
                    ratios[(m, arm)].append(vf(D, grid) / oracle(D, grid))
        out[str(size)] = {f"{m}_arm{arm}": {"q10": np.quantile(v, 0.1, axis=0).tolist(),
                                            "q90": np.quantile(v, 0.9, axis=0).tolist()}
                          for (m, arm), v in ratios.items()}
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="paper_hetero", choices=sorted(DESIGNS))
    parser.add_argument("--kappa", type=float, default=None,
                        help="override the design's noise_kappa")
    parser.add_argument("--n", type=int, nargs="+", default=[32000])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fit-check", action="store_true",
                        help="only compare fitted and true variance functions on a grid")
    args = parser.parse_args(argv)
    if args.fit_check:
        res = fit_check(args.design)
        print("eta grid:", res["grid"])
        for size in ("8000", "2667"):
            for key, q in res[size].items():
                print(f"  fit size {size:>4s} {key:14s} q10 {np.round(q['q10'], 2)}  "
                      f"q90 {np.round(q['q90'], 2)}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "fit_check.json").write_text(json.dumps(res, indent=2) + "\n")
        return
    design = _design(args.design, args.kappa)
    pop = Population(design, allow_hetero=True)
    pred = hetero_variances(pop, design.noise_kappa)
    gamma_x = np.asarray(design.gamma_x)
    result = {"design": args.design, "kappa": design.noise_kappa, "prediction": pred,
              "cells": {}}
    for n in args.n:
        tasks = [(args.design, n, args.seed + 104_729 * n + r, args.kappa)
                 for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(replicate, tasks, chunksize=2))
        cell = {"reps": len(rows)}
        base = np.array([r["avg_paper"] for r in rows])
        for v in VARIANTS:
            a = np.array([r[f"avg_{v}"] for r in rows])
            f = np.array([r[f"fixed_{v}"] for r in rows]) - pop.phi
            cell[v] = {"rot_n_var": float(n * a.var(ddof=1)),
                       "fixed_n_var": float(n * f.var(ddof=1)),
                       "bias": float(np.mean(a) - pop.phi),
                       "paired_vs_paper": float(n * (base.var(ddof=1) - a.var(ddof=1)))}
        m = len(np.array_split(np.arange(n), 4)[0])
        cell["slope_m_var"] = {
            key: float(m * np.mean(np.var([r["slopes"][key] for r in rows], axis=0, ddof=1)))
            for key in SLOPE_KEYS}
        cell["slope_bias"] = {
            key: float(np.mean(np.mean([r["slopes"][key] for r in rows], axis=0) - gamma_x))
            for key in SLOPE_KEYS}
        result["cells"][str(n)] = cell
        print(f"[{args.design}] N={n} reps={len(rows)}; bound {pred['bound']:.2f}, "
              f"paper prediction {pred['paper']:.2f}")
        for v in VARIANTS:
            c = cell[v]
            print(f"  {v:12s}: rotated N Var={c['rot_n_var']:.2f}; fixed={c['fixed_n_var']:.1f}; "
                  f"paired vs paper={c['paired_vs_paper']:+.2f}; bias={c['bias']:+.4f}")
        print(f"  first-stage slopes m Var: {cell['slope_m_var']}")
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.design if args.kappa is None else f"{args.design}_kappa{args.kappa:g}"
    path = args.out / f"{tag}_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {path}")


if __name__ == "__main__":
    main()
