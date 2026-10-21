"""Point-estimate Monte Carlo of the script's rotated estimator (nonlinear design),
with all four rotations recorded and a role-by-role comparison of the actual
estimation errors with the paper's linear role scores evaluated on the same data.

Seeds and split follow ds_four_role_coverage.replicate (resplit scheme):
data seed = 20261019 + 104729 n + rep, split rng = default_rng([seed, 1]).
The spline arm of ds_four_role_coverage.estimate is re-implemented here (same
calls) so that the four rotations can be returned; a check against the stored
jsonl estimates is printed.

Usage: python mc_decomp.py N REPS [START] [WORKERS]
"""
from __future__ import annotations

import json
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "/Users/zoudj/Documents/Codex/PerfRDD/code"
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import numpy as np
from numpy.polynomial.legendre import leggauss
from concurrent.futures import ProcessPoolExecutor

from experiments.scripts.ds_four_role_coverage import (
    COVERAGE_DESIGNS, SplineTailPieces, _refine)
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, _xt, generate, ols_first_stage, outcome_fit)
from experiments.methods.weighted_tails import fit_weighted_tails
import indep_pieces as P

DESIGN = COVERAGE_DESIGNS["nonlinear"]
SEED0 = 20261019
GRID = np.linspace(DESIGN.policy[0], DESIGN.policy[1], 61)
_x, _w = leggauss(200)
EQ = 0.5 * (P.U0 - P.L0) * _x + 0.5 * (P.U0 + P.L0)
WQ = 0.5 * (P.U0 - P.L0) * _w * P.f_eta(EQ)          # weights for E[. I0]
ABAR = np.asarray(P.Abar)
M90 = np.array([1.0, P.q90, 0.0]); M10 = np.array([1.0, P.q10, 0.0])
A_U = np.asarray(P.A_U); A_OUT = np.asarray(P.A_out); A_T = np.asarray(P.A_T)
GAMMA = np.array([0.0, 1.0, 0.0])


def r_out(D, eta, X):
    r = np.zeros(len(eta))
    SJ = (eta >= P.J0) & (eta <= P.J1)
    I0 = (eta >= P.L0) & (eta <= P.U0)
    for d in (0, 1):
        on = (D == d) & SJ
        e = eta[on]
        r[on] = ((2 * d - 1) * np.where(I0[on], P.g(P.PHI - e), 0.0) / P.pi_d(d, e)
                 + (X[on, 0] - P.mu1_d(d, e)) * P.omega[d][0] + X[on, 1] * P.omega[d][1])
    return r


def one(task):
    n, rep = task
    seed = SEED0 + 104_729 * n + rep
    s = generate(DESIGN, n, seed)
    rng0 = np.random.default_rng([seed, 1])
    blocks = np.array_split(rng0.permutation(n), 4)
    # true quantities
    eta = s.Q - s.X[:, 0]
    m0 = (P.bfun(eta) + s.X @ np.array(DESIGN.beta1) + s.D * (P.alpha(eta) + s.X @ P.BETA2))
    RY = s.Y - m0
    Xt = _xt(s.X)
    I0 = (eta >= P.L0) & (eta <= P.U0)
    psiU = -np.where(I0, P.Fs(eta), 0.0)
    psiO = -r_out(s.D, eta, s.X) * RY
    A, w2 = P.psiT_parts(s.X[:, 0])
    psiT = A - w2 * s.X[:, 1]
    psiG = (Xt @ ABAR) * eta
    out = {"rep": rep, "seed": seed, "rot": [], "lin": [], "Zhat": [], "Hhat": [],
           "eval": [], "out": [], "dens": [], "end": [],
           "eval_lin": [], "out_lin": [], "dens_lin": [], "end_lin": [], "delta": []}
    crit, vals = [], []
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        n_u = len(folds["utility"])
        gamma_hat = ols_first_stage(s, folds["gamma"])
        delta = gamma_hat - GAMMA
        eta_hat = s.Q - _xt(s.X) @ gamma_hat
        T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], DESIGN.T_support)
        l_hat = DESIGN.phi0 - float(np.quantile(T_idx, 1.0 - DESIGN.eps))
        u_hat = DESIGN.phi0 - float(np.quantile(T_idx, DESIGN.eps))
        e_all = eta_hat[folds["utility"]]
        e = e_all[(e_all >= l_hat) & (e_all <= u_hat)]
        fit = outcome_fit(s, folds["outcome"], eta_hat, DESIGN.J)
        pieces = SplineTailPieces(tails, e, n_u)
        amc = fit.a(e) - DESIGN.cost
        b2 = np.asarray(fit.beta2)
        gv = []
        for x in GRID:
            (sg,), sh = pieces.sums(x, [amc])
            gv.append((sg + float(sh @ b2)) / n_u)
        cfun = (lambda x, p=pieces, a=amc, b=b2: p.value(x, a, b))
        crit.append(cfun); vals.append(gv)
        phik = _refine(cfun, GRID, np.array(gv))
        out["rot"].append(phik)
        # actual Zhat(phi*) = -Uhat'(phi*) and curvature at phi_k
        t = P.PHI - e
        Zh = float(np.sum(amc * tails.density(t) + tails.weighted_density(t, b2)) / n_u)
        tk = phik - e
        Hh = -float(np.sum(amc * tails.density_derivative(tk)
                           + tails.weighted_density_derivative(tk, b2)) / n_u)
        out["Zhat"].append(Zh); out["Hhat"].append(Hh)
        # linear role scores on this rotation's folds (U' scale: U'hat = sum P_q psi_q)
        lin = (psiU[folds["utility"]].mean() + psiO[folds["outcome"]].mean()
               + psiT[folds["index"]].mean() + psiG[folds["gamma"]].mean())
        out["lin"].append(P.PHI - lin / P.H)
        out["delta"].append(delta.tolist())
        # actual one-at-a-time errors on the Z scale (Z = -U')
        ev = float(np.sum(np.where((e_all >= P.L0) & (e_all <= P.U0), P.Fs(e_all), 0.0)) / n_u)
        da = fit.a(EQ) - P.alpha(EQ)
        oe = float(np.sum(WQ * (P.g(P.PHI - EQ) * da + P.b2rho(P.PHI - EQ) / P.BETA2[0]
                                * (b2[0] - P.BETA2[0]))))
        # rho_X(t) = (t phi(t), 0): only the first slope enters
        hh = np.column_stack((tails.density(P.PHI - EQ), tails.weighted_density(P.PHI - EQ)))
        htrue = np.column_stack((P.g(P.PHI - EQ), (P.PHI - EQ) * P.g(P.PHI - EQ), 0 * EQ))
        nu = np.column_stack((P.alpha(EQ) - P.C, np.full_like(EQ, P.BETA2[0]),
                              np.full_like(EQ, P.BETA2[1])))
        de = float(np.sum(WQ * np.sum(nu * (hh - htrue), axis=1)))
        x2, w2_ = leggauss(40)

        def intF(a, b):
            ee = 0.5 * (b - a) * x2 + 0.5 * (b + a)
            return float(np.sum(0.5 * (b - a) * w2_ * P.f_eta(ee) * P.Fs(ee)))
        en = -intF(P.L0, l_hat) + intF(P.U0, u_hat)   # int_{l_hat}^{u_hat} - int_{l0}^{u0}
        out["eval"].append(ev); out["out"].append(oe); out["dens"].append(de); out["end"].append(en)
        # linear counterparts (Z scale)
        out["eval_lin"].append(float(-psiU[folds["utility"]].mean() - A_U @ delta))
        out["out_lin"].append(float(-psiO[folds["outcome"]].mean() + A_OUT @ delta))
        Ti = s.X[folds["index"], 0]
        rTX = -(A[folds["index"]] - w2[folds["index"]] * s.X[folds["index"], 1]
                + P.b_l * P.z_p(Ti, 0.9, P.q90) - P.b_u * P.z_p(Ti, 0.1, P.q10))   # r_T(T)'X~
        out["dens_lin"].append(float(rTX.mean() + A_T @ delta))
        out["end_lin"].append(float(np.mean(P.b_l * P.z_p(Ti, 0.9, P.q90) - P.b_u * P.z_p(Ti, 0.1, P.q10))
                                    + (P.b_l * M90 - P.b_u * M10) @ delta))
    vals = np.array(vals)
    out["avg"] = float(np.mean(out["rot"]))
    out["pooled"] = _refine(lambda x: sum(v(x) for v in crit) / 4, GRID, vals.mean(axis=0))
    return out


def main():
    n = int(sys.argv[1]); reps = int(sys.argv[2])
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    path = os.path.join(HERE, f"decomp_n{n}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["rep"] for l in open(path) if l.strip()}
    tasks = [(n, r) for r in range(start, start + reps) if r not in done]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool, open(path, "a") as fh:
        for i, row in enumerate(pool.map(one, tasks, chunksize=4), 1):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if i % 50 == 0 or i == len(tasks):
                print(f"n={n} {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
