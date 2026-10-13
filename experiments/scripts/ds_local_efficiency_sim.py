"""Simulation check of the efficient first stage for the four-role DS estimator.

Companion to ``ds_local_efficiency.py`` (population side).  The four-role
estimator of ``differing_slopes.tex`` Section 3 is run twice on the same data
and the same folds, differing only in the first stage computed on the
first-stage fold I_gamma:

* ``ols``: least squares of Q on (1, X), as in the paper;
* ``eff``: the stacked first stage of ``this_week.tex`` (efficient first
  stage).  I_gamma is split into three parts.  In each of three internal
  rotations a pilot part gives an OLS fit gamma_p; a second part fits the
  arm-wise spline outcome regression at eta_hat = Q - gamma_p'X~, giving
  xi_hat = c_d'(eta) (c_0 = b, c_1 = a + b) and the arm noise variances; the
  third part solves the weighted least-squares problem

      Q rows:   eta_hat / s_eta = (Delta_0 + X'Delta_X) / s_eta + noise,
      Y rows:   Y / s_D = {N_K(eta_hat)'w_D + X'd_D - xi_hat X'Delta_X} / s_D + noise,
                (only rows with eta_hat in J; separate spline and slope per arm)

  and returns gamma_p + Delta_hat.  The three results are averaged.

The rest of the pipeline is shared: arm-wise stacked spline outcome fit on J,
Lebesgue-Gram spline fits of g and rho_X on the index fold, empirical-quantile
trim endpoints, hard-trimmed evaluation average, argmax over Phi, and the
average of the four rotated thresholds.

Per replication we record the fixed-assignment and rotation-averaged
thresholds for both first stages, and the OLS and efficient first-stage slope
estimates on one fold, so the efficient first-stage variance can be checked
against I_gamma^{-1} directly.

Run:
    python -m experiments.scripts.ds_local_efficiency_sim --design paper_baseline \
        --n 16000 32000 --reps 500 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_local_efficiency import DESIGNS, Design, Population

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_local_efficiency_sim"
KNOT_EXPONENT = 11.0 / 60.0
ROLES = ("gamma", "outcome", "index", "utility")


@dataclass(frozen=True)
class Sample:
    X: np.ndarray
    Q: np.ndarray
    D: np.ndarray
    Y: np.ndarray


def generate(design: Design, n: int, seed: int) -> Sample:
    rng = np.random.default_rng(seed)
    dim = len(design.gamma_x)
    X = rng.standard_normal((n, dim)) * np.sqrt(np.asarray(design.sigma_x))
    eta = design.sigma_eta * rng.standard_normal(n)
    T = design.gamma0 + X @ np.asarray(design.gamma_x)
    Q = T + eta
    D = (Q > design.phi0).astype(float)
    a, b = Polynomial(design.a_coef), Polynomial(design.b_coef)
    sig = np.where(D == 1.0, design.sigma_arm[1], design.sigma_arm[0])
    sig = sig * np.exp(0.5 * design.noise_kappa * eta)
    Y = (b(eta) + X @ np.asarray(design.beta1)
         + D * (a(eta) + X @ np.asarray(design.beta2))
         + sig * rng.standard_normal(n))
    return Sample(X=X, Q=Q, D=D, Y=Y)


def _xt(X: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(X)), X))


def _ls(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares by the normal equations (designs here are well conditioned)."""
    try:
        return np.linalg.solve(A.T @ A, A.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, y, rcond=None)[0]


def ols_first_stage(s: Sample, idx: np.ndarray) -> np.ndarray:
    return _ls(_xt(s.X[idx]), s.Q[idx])


# ------------------------------------------------------------ outcome fit

@dataclass(frozen=True)
class OutcomeFit:
    info: dict
    w0: np.ndarray        # arm-0 spline coefficients (b)
    w1: np.ndarray        # arm-1 spline coefficients (a + b)
    d0: np.ndarray        # arm-0 slope (beta1)
    d1: np.ndarray        # arm-1 slope (beta1 + beta2)
    s2: tuple             # residual variances on arms 0 and 1

    def _spl(self, coef, nu=0):
        spline = BSpline(self.info["t"], coef, self.info["degree"], extrapolate=False)
        return spline.derivative(nu) if nu else spline

    def a(self, e):
        e = np.clip(e, self.info["lo"], self.info["hi"])
        return self._spl(self.w1 - self.w0)(e)

    def c_prime(self, arm, e):
        e = np.clip(e, self.info["lo"], self.info["hi"])
        return self._spl(self.w1 if arm == 1 else self.w0, 1)(e)

    @property
    def beta2(self):
        return self.d1 - self.d0


def _n_interior(n_eff: int) -> int:
    return max(3, int(round(n_eff ** KNOT_EXPONENT)))


def _design_matrix(e, X, D, info):
    B = BSpline.design_matrix(np.clip(e, info["lo"], info["hi"]), info["t"],
                              info["degree"]).toarray()
    D = D[:, None]
    return np.column_stack(((1 - D) * B, (1 - D) * X, D * B, D * X)), B.shape[1]


def outcome_fit(s: Sample, idx: np.ndarray, eta_hat: np.ndarray, J) -> OutcomeFit:
    """Arm-wise stacked spline fit on J (equivalent to the paper's D N, N, X, DX)."""
    e = eta_hat[idx]
    keep = (e >= J[0]) & (e <= J[1])
    fit = idx[keep]
    e, X, D, Y = eta_hat[fit], s.X[fit], s.D[fit], s.Y[fit]
    n1 = int(D.sum())
    info = _basis_params(_n_interior(min(n1, len(fit) - n1)), J)
    Z, nb = _design_matrix(e, X, D, info)
    coef = _ls(Z, Y)
    dim = X.shape[1]
    w0, d0 = coef[:nb], coef[nb:nb + dim]
    w1, d1 = coef[nb + dim:2 * nb + dim], coef[2 * nb + dim:]
    resid = Y - Z @ coef
    p = nb + dim
    s2 = tuple(float(np.sum(resid[D == arm] ** 2) / max(1, (D == arm).sum() - p))
               for arm in (0.0, 1.0))
    return OutcomeFit(info=info, w0=w0, w1=w1, d0=d0, d1=d1, s2=s2)


# ------------------------------------------------------------ efficient first stage

def efficient_first_stage(s: Sample, idx: np.ndarray, J, rng) -> np.ndarray:
    parts = np.array_split(rng.permutation(idx), 3)
    estimates = []
    for r in range(3):
        pilot, deriv, stack = parts[r], parts[(r + 1) % 3], parts[(r + 2) % 3]
        g_p = ols_first_stage(s, pilot)
        res_p = s.Q[pilot] - _xt(s.X[pilot]) @ g_p
        s_eta = float(np.sqrt(np.sum(res_p ** 2) / (len(pilot) - len(g_p))))
        eta_hat = s.Q - _xt(s.X) @ g_p
        fit = outcome_fit(s, deriv, eta_hat, J)
        sd = np.sqrt(np.asarray(fit.s2))

        e, X, D, Y = eta_hat[stack], s.X[stack], s.D[stack], s.Y[stack]
        dim = X.shape[1]
        inJ = (e >= J[0]) & (e <= J[1])
        n_int = _n_interior(int(min(D[inJ].sum(), inJ.sum() - D[inJ].sum())))
        info = _basis_params(n_int, J)
        Zy, nb = _design_matrix(e[inJ], X[inJ], D[inJ], info)
        Dj = D[inJ]
        xi = np.where(Dj == 1.0, fit.c_prime(1, e[inJ]), fit.c_prime(0, e[inJ]))
        wy = 1.0 / np.where(Dj == 1.0, sd[1], sd[0])
        # unknowns: [Delta_0, Delta_X (dim), outcome block (2 nb + 2 dim)]
        n_out = Zy.shape[1]
        rows_q = np.column_stack((np.ones(len(e)), X, np.zeros((len(e), n_out)))) / s_eta
        rows_y = np.column_stack((np.zeros(len(Dj)), -xi[:, None] * X[inJ], Zy)) * wy[:, None]
        lhs = np.vstack((rows_q, rows_y))
        rhs = np.concatenate((e / s_eta, Y[inJ] * wy))
        coef = _ls(lhs, rhs)
        estimates.append(g_p + coef[:1 + dim])
    return np.mean(estimates, axis=0)


# ------------------------------------------------------------ one rotation

def rotation_threshold(s: Sample, design: Design, folds: dict, gamma_hat: np.ndarray) -> float:
    J, eps, phi0 = design.J, design.eps, design.phi0
    eta_hat = s.Q - _xt(s.X) @ gamma_hat
    fit = outcome_fit(s, folds["outcome"], eta_hat, J)
    T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
    tails = fit_weighted_tails(T_idx, s.X[folds["index"]], design.T_support)
    l_hat = phi0 - float(np.quantile(T_idx, 1.0 - eps))
    u_hat = phi0 - float(np.quantile(T_idx, eps))
    e = eta_hat[folds["utility"]]
    keep = (e >= l_hat) & (e <= u_hat)
    e = e[keep]
    amc = fit.a(e) - design.cost
    b2 = fit.beta2

    def value(phi):
        t = phi - e
        return float(np.mean(amc * tails.survival(t) + tails.weighted_tail(t, b2)))

    lo, hi = design.policy
    grid = np.linspace(lo, hi, 61)
    vals = np.array([value(x) for x in grid])
    k = int(np.argmax(vals))
    a_, b_ = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: -value(x), bounds=(a_, b_), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x)


def replicate(task):
    design_name, n, seed = task
    design = DESIGNS[design_name]
    s = generate(design, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    out = {"ols": [], "eff": []}
    gammas = {}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        g_ols = ols_first_stage(s, folds["gamma"])
        g_eff = efficient_first_stage(s, folds["gamma"], design.J,
                                      np.random.default_rng(seed + 11 * k + 3))
        if k == 0:
            gammas = {"ols": g_ols.tolist(), "eff": g_eff.tolist()}
        out["ols"].append(rotation_threshold(s, design, folds, g_ols))
        out["eff"].append(rotation_threshold(s, design, folds, g_eff))
    return {
        "fixed_ols": out["ols"][0], "avg_ols": float(np.mean(out["ols"])),
        "fixed_eff": out["eff"][0], "avg_eff": float(np.mean(out["eff"])),
        "gamma_ols": gammas["ols"], "gamma_eff": gammas["eff"],
    }


def summarize(rows, n, design: Design, pop: dict) -> dict:
    phi = pop["phi_star"]
    get = lambda key: np.array([r[key] for r in rows])
    cell = {}
    for key in ("fixed_ols", "avg_ols", "fixed_eff", "avg_eff"):
        err = get(key) - phi
        cell[key] = {"n_var": float(n * np.var(err, ddof=1)),
                     "bias": float(np.mean(err)),
                     "n_mse": float(n * np.mean(err ** 2))}
    d_avg = get("avg_ols") - get("avg_eff")
    cell["paired"] = {
        "n_var_diff_avg": cell["avg_ols"]["n_var"] - cell["avg_eff"]["n_var"],
        "n_var_of_difference_avg": float(n * np.var(d_avg, ddof=1)),
        "corr_avg": float(np.corrcoef(get("avg_ols"), get("avg_eff"))[0, 1]),
    }
    # first-stage slopes on one fold of size n/4
    gx = np.asarray(design.gamma_x)
    m = n // 4
    for key in ("gamma_ols", "gamma_eff"):
        G = np.array([r[key] for r in rows])[:, 1:] - gx
        cell[key] = {"m_cov_diag": (m * np.var(G, axis=0, ddof=1)).tolist(),
                     "mean_err": G.mean(axis=0).tolist()}
    cell["prediction"] = {
        "V_rotated_ols": pop["V_rotated_ols"], "V_rotated_eff": pop["V_rotated_eff"],
        "V_fixed_ols": pop["V_fixed_ols"], "V_fixed_eff": pop["V_fixed_eff"],
        "gamma_ols_m_var": (design.sigma_eta ** 2 / np.asarray(design.sigma_x)).tolist(),
        "gamma_eff_m_var": np.diag(np.linalg.inv(np.asarray(pop["I_gamma"]))).tolist(),
    }
    cell["reps"] = len(rows)
    cell["relative_mc_se"] = float(np.sqrt(2.0 / (len(rows) - 1)))
    return cell


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="paper_baseline", choices=sorted(DESIGNS))
    parser.add_argument("--n", type=int, nargs="+", default=[16000, 32000])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    design = DESIGNS[args.design]
    if design.noise_kappa:
        from experiments.scripts.ds_local_efficiency_hetero import hetero_variances
        base = Population(design, allow_hetero=True)
        het = hetero_variances(base, design.noise_kappa)
        pop = {"phi_star": base.phi, "V_rotated_ols": het["paper"],
               "V_rotated_eff": het["stacked"], "V_fixed_ols": het["fixed_paper"],
               "V_fixed_eff": het["fixed_stacked"], "hetero": het,
               "I_gamma": np.linalg.inv(np.diag(het["stacked_slope_var"])).tolist()}
    else:
        pop = Population(design).decomposition()
    result = {"design": args.design, "population": pop, "cells": {}}
    for n in args.n:
        tasks = [(args.design, n, args.seed + 104_729 * n + r) for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(replicate, tasks, chunksize=2))
        cell = summarize(rows, n, design, pop)
        result["cells"][str(n)] = cell
        p = cell["prediction"]
        print(f"[{args.design}] N={n} reps={len(rows)} (MC s.e. ~{100 * cell['relative_mc_se']:.0f}%)")
        print(f"  N Var avg: ols={cell['avg_ols']['n_var']:.2f} (pred {p['V_rotated_ols']:.2f}), "
              f"eff={cell['avg_eff']['n_var']:.2f} (pred {p['V_rotated_eff']:.2f}); "
              f"paired diff={cell['paired']['n_var_diff_avg']:.2f} "
              f"(pred {p['V_rotated_ols'] - p['V_rotated_eff']:.2f}), corr={cell['paired']['corr_avg']:.3f}")
        print(f"  N Var fixed: ols={cell['fixed_ols']['n_var']:.2f} (pred {p['V_fixed_ols']:.2f}), "
              f"eff={cell['fixed_eff']['n_var']:.2f} (pred {p['V_fixed_eff']:.2f})")
        print(f"  bias avg: ols={cell['avg_ols']['bias']:+.4f} eff={cell['avg_eff']['bias']:+.4f}")
        print(f"  first stage m Var slopes: ols={np.round(cell['gamma_ols']['m_cov_diag'], 3)} "
              f"(pred {np.round(p['gamma_ols_m_var'], 3)}), eff={np.round(cell['gamma_eff']['m_cov_diag'], 3)} "
              f"(pred {np.round(p['gamma_eff_m_var'], 3)})")
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.design}_summary.json"
    (args.out / tag).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wrote] {args.out / tag}")


if __name__ == "__main__":
    main()
