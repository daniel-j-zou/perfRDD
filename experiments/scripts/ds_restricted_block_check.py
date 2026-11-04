"""Check of the restricted-effect-slope outcome representer (differing slopes).

The paper's outcome fit uses the columns (D N_K, N_K, X, D X).  Here the effect
slope is restricted to effect covariates X_E = A X (A known, full row rank), so
the fit uses (D N_K, N_K, X, D X_E) and beta2 = A' beta_E.  The claimed
representer of L_out(f) = E[{g(phi* - eta) alpha_f(eta) + rho_X(phi* - eta)' c2_f} I0]
on the restricted space keeps the form of Lemma lem:ds-rout,

    r = S_J sum_d 1(D=d) [(2d-1) I0 g(phi* - eta) / pi_d(eta) + {X - mu_d(eta)}' omega_d],

with coupled arm slopes omega_0 = w, omega_1 = w + A'v, where

    [G0 + G1,  G1 A'  ] [w]   [m_0 - m_1  ]
    [A G1,     A G1 A'] [v] = [A (B - m_1)],   G_d = E[S_J pi_d Sigma_d],
                                                 m_d = E[I0 g(phi* - eta) mu_d].

Design: the displayed design of differing_slopes.tex Section 5 with
beta2 = (0.4, 0, 0) and A selecting X_1.

Parts (``--part``):

* ``population``: target check; the restricted representer; Riesz identities
  by Monte Carlo (``--mc-draws`` draws) for directions in the restricted space,
  and for A = I against the paper's omega_d; a finite-sieve projection check of
  the slopes (MC Gram, discontinuous spline basis with breaks at l0, u0;
  mean and s.e. over SIEVE_SEEDS independent Gram matrices); the
  predicted n Var of the unrestricted and restricted estimators.
* ``mc``: both estimators on the same data and folds (OLS first stage, spline
  outcome fit on J with D X or with D X_E only, spline tails on the index fold,
  hard-trimmed evaluation average; rotated average, pooled maximizer and fixed
  assignment).  Rows are appended to ``<out>/n<n>.jsonl``; reruns resume.
* ``report``: the markdown table of the finished cells.

Run (from code/):
    python -m experiments.scripts.ds_restricted_block_check --part population
    python -m experiments.scripts.ds_restricted_block_check --part mc \
        --cells 16000:1000 64000:500 --workers 3
    python -m experiments.scripts.ds_restricted_block_check --part report
Larger run used in the notebook (extends the 500 data sets above with the same seeds;
copy n64000.jsonl into the new directory first):
    python -m experiments.scripts.ds_restricted_block_check --part mc \
        --cells 64000:1000 --workers 3 \
        --out experiments/runs/ds_restricted_block_check/supp_64k_1000

Notation: the notebook (this_week.tex, section sec:week-taxi-theory, revised
2026-11-04) restricts to a coordinate set E (here E = {1}) instead of a matrix A.
Its unknowns are omega^(0) = w and omega^(1) = w + A'v, and m_d here is its
mu-bar^(d).
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
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_four_role_coverage import (
    SplineTailPieces, _avg_and_pooled, population_summary,
)
from experiments.scripts.ds_local_efficiency import DESIGNS, Design, Population
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, OutcomeFit, Sample, _ls, _n_interior, _xt, generate, ols_first_stage,
    outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_restricted_block_check"
SEL = (0,)                         # X_E = X_1
SIEVE_SEEDS, SIEVE_DRAWS = 8, 4_000_000


def make_design(beta2_1: float = 0.4) -> Design:
    return replace(DESIGNS["paper_baseline"], name=f"restricted_x1_b{beta2_1:g}",
                   beta2=(float(beta2_1), 0.0, 0.0),
                   note="displayed design with beta2 = (b, 0, 0); effect slope on X_1 only")


def selector(dim: int, sel=SEL) -> np.ndarray:
    A = np.zeros((len(sel), dim))
    for i, j in enumerate(sel):
        A[i, j] = 1.0
    return A


# ------------------------------------------------------------ target check

def target_check(design: Design) -> dict:
    pop = Population(design)                     # raises unless one interior maximizer
    summ = population_summary(design)            # adaptive quadrature, must agree to 1e-8
    lo, hi = design.policy
    grid = np.linspace(lo, hi, 3001)
    Z = np.array([pop.Z(x) for x in grid])
    sign_changes = int(np.sum(np.sign(Z[:-1]) != np.sign(Z[1:])))
    return {"phi_star": summ["phi_star"], "phi_star_gl": pop.phi, "H": pop.H,
            "policy": [lo, hi], "Z_sign_changes_on_Phi": sign_changes,
            "Z_at_ends": [float(Z[0]), float(Z[-1])],
            "distance_to_boundary": float(min(pop.phi - lo, hi - pop.phi)),
            "l0": pop.l0, "u0": pop.u0}


# ------------------------------------------------------------ representer

def representer(pop: Population, A: np.ndarray) -> dict:
    """Restricted outcome representer (coupled slopes), its variance and loading."""
    d, dim = pop.d, pop.dim
    phi, l0, u0 = pop.phi, pop.l0, pop.u0
    J0, J1 = d.J
    eJ, wJ = pop._eta_nodes(J0, J1, extra=(l0, u0))
    inI0 = (eJ >= l0) & (eJ <= u0)
    gphi = np.where(inI0, pop.g(phi - eJ), 0.0)
    B = np.sum((wJ * inI0)[:, None] * pop.rho(phi - eJ), axis=0)
    G, N, m, pis, mus = {}, {}, {}, {}, {}
    for arm in (0, 1):
        pis[arm] = pop.pi(arm, eJ)
        mus[arm] = pop.m_arm(arm, eJ)
        V = pop.V_arm(arm, eJ)
        G[arm] = np.einsum("i,ijk->jk", wJ * pis[arm], V)
        N[arm] = np.einsum("i,ijk->jk", wJ * pis[arm] * pop.dc_fun[arm](eJ), V)
        m[arm] = np.sum((wJ * gphi)[:, None] * mus[arm], axis=0)
    p = A.shape[0]
    lhs = np.block([[G[0] + G[1], G[1] @ A.T], [A @ G[1], A @ G[1] @ A.T]])
    rhs = np.concatenate([m[0] - m[1], A @ (B - m[1])])
    sol = np.linalg.solve(lhs, rhs)
    w, v = sol[:dim], sol[dim:]
    omega = {0: w, 1: w + A.T @ v}
    var_o, A_out_X = 0.0, np.zeros(dim)
    A_out_0 = float(np.sum(wJ * gphi * pop.da(eJ)))
    for arm, s in ((0, -1.0), (1, 1.0)):
        sig2 = d.sigma_arm[arm] ** 2
        var_o += sig2 * (float(np.sum(wJ * gphi ** 2 / pis[arm]))
                         + float(omega[arm] @ G[arm] @ omega[arm]))
        A_out_X += (s * np.sum((wJ * pop.dc_fun[arm](eJ) * gphi)[:, None] * mus[arm], axis=0)
                    + N[arm] @ omega[arm])
    return {"A": A, "omega": omega, "w": w, "v": v, "B": B, "m": m, "Gamma": G,
            "var_o": var_o, "A_out": np.concatenate([[A_out_0], A_out_X])}


def predictions(pop: Population, A: np.ndarray) -> dict:
    """Rotated (sigma^2/H^2) and fixed (4 sigma^2/H^2) n Var, unrestricted and restricted."""
    dec = pop.decomposition()
    v = dec["variances"]
    rep = representer(pop, A)
    Abar_res = np.asarray(dec["Abar"]) + np.asarray(dec["A_out"]) - rep["A_out"]
    AX = Abar_res[1:]
    var_g_res = float(pop.var_eta * AX @ np.linalg.solve(pop.Sigma, AX))
    H2 = pop.H ** 2
    s2_full = v["psi_U"] + v["psi_o"] + v["psi_T"] + v["psi_gamma_ols"]
    s2_res = v["psi_U"] + rep["var_o"] + v["psi_T"] + var_g_res
    return {
        "phi_star": pop.phi, "H": pop.H,
        "unrestricted": {"var_U": v["psi_U"], "var_o": v["psi_o"], "var_T": v["psi_T"],
                         "var_gamma": v["psi_gamma_ols"], "sigma2": s2_full,
                         "Abar": dec["Abar"], "A_out": dec["A_out"],
                         "omega": dec["omega"],
                         "V_rotated": s2_full / H2, "V_fixed": 4 * s2_full / H2},
        "restricted": {"var_U": v["psi_U"], "var_o": rep["var_o"], "var_T": v["psi_T"],
                       "var_gamma": var_g_res, "sigma2": s2_res,
                       "Abar": Abar_res.tolist(), "A_out": rep["A_out"].tolist(),
                       "omega": {str(k): x.tolist() for k, x in rep["omega"].items()},
                       "V_rotated": s2_res / H2, "V_fixed": 4 * s2_res / H2},
    }


# ------------------------------------------------------------ Riesz identities by MC

def _draw(design: Design, rng, n: int):
    dim = len(design.gamma_x)
    X = rng.standard_normal((n, dim)) * np.sqrt(np.asarray(design.sigma_x))
    eta = design.sigma_eta * rng.standard_normal(n)
    D = (design.gamma0 + X @ np.asarray(design.gamma_x) + eta > design.phi0).astype(float)
    return X, eta, D


def r_value(pop: Population, omega: dict, X, eta, D) -> np.ndarray:
    J0, J1 = pop.d.J
    inJ = (eta >= J0) & (eta <= J1)
    inI0 = (eta >= pop.l0) & (eta <= pop.u0)
    e = np.where(inJ, eta, 0.0)                  # avoid extreme Mills ratios off J
    g = np.where(inI0, pop.g(pop.phi - e), 0.0)
    r = np.zeros(len(eta))
    for arm, s in ((0, -1.0), (1, 1.0)):
        on = (D == arm)
        pi = pop.pi(arm, e)
        mu = pop.m_arm(arm, e)
        r += on * (s * g / pi + np.sum((X - mu) * omega[arm], axis=1))
    return inJ * r


U_FUNS = {"1": lambda e: np.ones_like(e), "eta": lambda e: e,
          "eta^2-1": lambda e: e ** 2 - 1.0, "sin(2 eta)": lambda e: np.sin(2.0 * e)}


def directions(pop: Population, A: np.ndarray, rng, extra_DX: bool):
    """(label, f(X, eta, D), L-integrand(X, eta, D), L by quadrature) for test directions."""
    phi, l0, u0 = pop.phi, pop.l0, pop.u0
    e_w, w_w = pop._eta_nodes(l0, u0)
    out = []

    def gI0(eta):
        inI0 = (eta >= l0) & (eta <= u0)
        return np.where(inI0, pop.g(phi - eta), 0.0)

    for name, u in U_FUNS.items():
        Lq = float(np.sum(w_w * pop.g(phi - e_w) * u(e_w)))
        out.append((f"D u, u={name}", lambda X, e, D, u=u: D * u(e),
                    lambda X, e, D, u=u: gI0(e) * u(e), Lq))
        out.append((f"u, u={name}", lambda X, e, D, u=u: u(e),
                    lambda X, e, D: np.zeros_like(e), 0.0))
    for i in range(3):
        c = rng.standard_normal(pop.dim)
        out.append((f"X'c, c={np.round(c, 2).tolist()}", lambda X, e, D, c=c: X @ c,
                    lambda X, e, D: np.zeros_like(e), 0.0))
    B = np.sum(w_w[:, None] * pop.rho(phi - e_w), axis=0)
    p = A.shape[0]
    for i in range(2):
        cE = rng.standard_normal(p)
        c2 = A.T @ cE
        out.append((f"D X_E'c_E, c_E={np.round(cE, 2).tolist()}",
                    lambda X, e, D, c2=c2: D * (X @ c2),
                    lambda X, e, D, c2=c2: np.where((e >= l0) & (e <= u0),
                                                     pop.rho(phi - e) @ c2, 0.0),
                    float(B @ c2)))
    if extra_DX:
        for i in range(2):
            c = rng.standard_normal(pop.dim)
            out.append((f"D X'c, c={np.round(c, 2).tolist()}",
                        lambda X, e, D, c=c: D * (X @ c),
                        lambda X, e, D, c=c: np.where((e >= l0) & (e <= u0),
                                                       pop.rho(phi - e) @ c, 0.0),
                        float(B @ c)))
    return out


def riesz_mc(pop: Population, reps: dict, n_draws: int, seed: int, chunk: int = 500_000) -> dict:
    """MC inner products <r, f>_J for each representer in ``reps`` against its directions.

    ``reps`` maps a label to (omega, A, extra_DX).  Also returns MC versions of
    Var(psi_o) and A_out for each representer."""
    d = pop.d
    rng_dirs = np.random.default_rng(seed + 17)
    plan = {k: (om, directions(pop, A, rng_dirs, extra)) for k, (om, A, extra) in reps.items()}
    acc = {k: {"rf": np.zeros(len(dirs)), "diff": np.zeros(len(dirs)),
               "diff2": np.zeros(len(dirs)), "r2s": 0.0, "Aout": np.zeros(pop.dim + 1)}
           for k, (om, dirs) in plan.items()}
    # outside-the-space contrast for the restricted representer: f = D X_2
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        X, eta, D = _draw(d, rng, m)
        sig2 = np.where(D == 1, d.sigma_arm[1], d.sigma_arm[0]) ** 2
        xi = np.where(D == 1, pop.dc_fun[1](eta), pop.dc_fun[0](eta))
        Xt = np.column_stack((np.ones(m), X))
        for k, (om, dirs) in plan.items():
            r = r_value(pop, om, X, eta, D)
            a = acc[k]
            for j, (_, f, lint, _) in enumerate(dirs):
                rf = r * f(X, eta, D)
                df = rf - lint(X, eta, D)
                a["rf"][j] += rf.sum()
                a["diff"][j] += df.sum()
                a["diff2"][j] += (df ** 2).sum()
            a["r2s"] += float(np.sum(sig2 * r ** 2))
            a["Aout"] += (xi * r) @ Xt
        done += m
    out = {}
    for k, (om, dirs) in plan.items():
        a = acc[k]
        mean_d = a["diff"] / n_draws
        se = np.sqrt((a["diff2"] / n_draws - mean_d ** 2) / (n_draws - 1))
        rows = []
        for j, (label, _, _, Lq) in enumerate(dirs):
            rows.append({"direction": label, "inner_mc": float(a["rf"][j] / n_draws),
                         "L_quad": Lq, "diff": float(mean_d[j]), "se": float(se[j]),
                         "z": float(mean_d[j] / se[j]) if se[j] > 0 else 0.0})
        out[k] = {"identities": rows, "var_o_mc": a["r2s"] / n_draws,
                  "A_out_mc": (a["Aout"] / n_draws).tolist()}
    return out


# ------------------------------------------------------------ finite-sieve projection check

def _piece_basis(e: np.ndarray, breaks, n_int: int) -> np.ndarray:
    cols = []
    for i, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:])):
        info = _basis_params(n_int, (lo, hi))
        Bm = BSpline.design_matrix(np.clip(e, lo, hi), info["t"], 3).toarray()
        last = i == len(breaks) - 2
        ind = (e >= lo) & ((e <= hi) if last else (e < hi))
        cols.append(Bm * ind[:, None])
    return np.hstack(cols)


def sieve_check(pop: Population, A: np.ndarray, n_draws: int, seed: int, n_int: int = 6,
                chunk: int = 250_000) -> dict:
    """Slopes of the finite-sieve representer S_J z' Q^{-1} ell, z = (D N, N, X, D X_E).

    N is a cubic spline basis on J that may jump at l0 and u0, so the eta parts
    of the representer are smooth on each piece and the sieve error is
    negligible; Q is a Monte Carlo Gram matrix, ell is computed by quadrature."""
    d, dim, p = pop.d, pop.dim, A.shape[0]
    breaks = [d.J[0], pop.l0, pop.u0, d.J[1]]
    e_w, w_w = pop._eta_nodes(pop.l0, pop.u0)
    NK = _piece_basis(e_w, breaks, n_int)
    K = NK.shape[1]
    ell_a = np.sum((w_w * pop.g(pop.phi - e_w))[:, None] * NK, axis=0)
    B = np.sum(w_w[:, None] * pop.rho(pop.phi - e_w), axis=0)
    ell = np.concatenate([ell_a, np.zeros(K + dim), A @ B])
    P = 2 * K + dim + p
    Q = np.zeros((P, P))
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        X, eta, D = _draw(d, rng, m)
        keep = (eta >= d.J[0]) & (eta <= d.J[1])
        X, eta, D = X[keep], eta[keep], D[keep][:, None]
        Nb = _piece_basis(eta, breaks, n_int)
        Z = np.column_stack((D * Nb, Nb, X, D * (X @ A.T)))
        Q += Z.T @ Z
        done += m
    Q /= n_draws
    vK = np.linalg.solve(Q, ell)
    c1, cE = vK[2 * K:2 * K + dim], vK[2 * K + dim:]
    return {"n_basis_per_block": K, "omega0_sieve": c1.tolist(),
            "omega1_sieve": (c1 + A.T @ cE).tolist()}


# ------------------------------------------------------------ estimators

def outcome_fit_restricted(s: Sample, idx: np.ndarray, eta_hat: np.ndarray, J,
                           sel=SEL) -> OutcomeFit:
    """Stacked spline fit on J with columns spanning (D N_K, N_K, X, D X_E).

    Uses ((1-D) N_K, D N_K, X, D X_E), the same span; the knot rule is that of
    ``ds_local_efficiency_sim.outcome_fit``."""
    e = eta_hat[idx]
    fit = idx[(e >= J[0]) & (e <= J[1])]
    e, X, D, Y = eta_hat[fit], s.X[fit], s.D[fit], s.Y[fit]
    n1 = int(D.sum())
    info = _basis_params(_n_interior(min(n1, len(fit) - n1)), J)
    Bm = BSpline.design_matrix(np.clip(e, info["lo"], info["hi"]), info["t"],
                               info["degree"]).toarray()
    nb, dim = Bm.shape[1], X.shape[1]
    Dc = D[:, None]
    XE = X[:, list(sel)]
    Z = np.column_stack(((1 - Dc) * Bm, Dc * Bm, X, Dc * XE))
    coef = _ls(Z, Y)
    w0, w1 = coef[:nb], coef[nb:2 * nb]
    beta1 = coef[2 * nb:2 * nb + dim]
    beta2 = np.zeros(dim)
    beta2[list(sel)] = coef[2 * nb + dim:]
    return OutcomeFit(info=info, w0=w0, w1=w1, d0=beta1, d1=beta1 + beta2, s2=(np.nan, np.nan))


ESTIMATORS = ("full", "restricted")


def estimate_both(s: Sample, design: Design, rng, sel=SEL) -> dict:
    """Unrestricted and restricted estimators on one data set and one four-block split."""
    n = len(s.Q)
    blocks = np.array_split(rng.permutation(n), 4)
    grid = np.linspace(design.policy[0], design.policy[1], 61)
    crit = {k: [] for k in ESTIMATORS}
    vals = {k: [] for k in ESTIMATORS}
    b2_fold0 = {}
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        n_u = len(folds["utility"])
        gamma_hat = ols_first_stage(s, folds["gamma"])
        eta_hat = s.Q - _xt(s.X) @ gamma_hat
        T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], design.T_support)
        l_hat = design.phi0 - float(np.quantile(T_idx, 1.0 - design.eps))
        u_hat = design.phi0 - float(np.quantile(T_idx, design.eps))
        e = eta_hat[folds["utility"]]
        e = e[(e >= l_hat) & (e <= u_hat)]
        fits = {"full": outcome_fit(s, folds["outcome"], eta_hat, design.J),
                "restricted": outcome_fit_restricted(s, folds["outcome"], eta_hat, design.J, sel)}
        pieces = SplineTailPieces(tails, e, n_u)
        amc = {est: fits[est].a(e) - design.cost for est in ESTIMATORS}
        b2 = {est: np.asarray(fits[est].beta2) for est in ESTIMATORS}
        if k == 0:
            b2_fold0 = {est: b2[est].tolist() for est in ESTIMATORS}
        gv = {est: [] for est in ESTIMATORS}
        for x in grid:
            sg, sh = pieces.sums(x, [amc[est] for est in ESTIMATORS])
            for est, g_sum in zip(ESTIMATORS, sg):
                gv[est].append((g_sum + float(sh @ b2[est])) / n_u)
        for est in ESTIMATORS:
            crit[est].append(lambda x, p=pieces, a=amc[est], b=b2[est]: p.value(x, a, b))
            vals[est].append(gv[est])
    out = {}
    for est in ESTIMATORS:
        avg, pooled, rot = _avg_and_pooled(crit[est], grid, np.array(vals[est]))
        out[f"{est}_avg"], out[f"{est}_pooled"], out[f"{est}_fixed"] = avg, pooled, rot[0]
        out[f"{est}_b2_fold0"] = b2_fold0[est]
    return out


def replicate(task) -> dict:
    beta2_1, n, rep, seed = task
    design = make_design(beta2_1)
    t0 = time.perf_counter()
    s = generate(design, n, seed)
    est = estimate_both(s, design, np.random.default_rng([seed, 1]))
    return {"rep": rep, "seed": seed, "est": est, "seconds": time.perf_counter() - t0}


def _load_rows(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def run_cell(beta2_1: float, n: int, reps: int, seed: int, workers: int, out: Path) -> list:
    path = out / f"n{n}.jsonl"
    rows = _load_rows(path)
    done = {r["rep"] for r in rows}
    tasks = [(beta2_1, n, r, seed + 104_729 * n + r) for r in range(reps) if r not in done]
    if tasks:
        print(f"n={n}: {len(done)} done, running {len(tasks)}", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers) as pool, path.open("a") as fh:
            futures = [pool.submit(replicate, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if i % 50 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}, {time.perf_counter() - t0:.0f}s", flush=True)
    return sorted(rows, key=lambda r: r["rep"])[:reps]


def summarize(rows: list, n: int, pred: dict, n_boot: int = 2000) -> dict:
    phi = pred["phi_star"]
    R = len(rows)
    rel = float(np.sqrt(2.0 / (R - 1)))
    cell = {"n": n, "reps": R, "relative_mc_se": rel, "arms": {}}
    for est, pkey in (("full", "unrestricted"), ("restricted", "restricted")):
        for kind in ("avg", "pooled", "fixed"):
            x = np.array([r["est"][f"{est}_{kind}"] for r in rows])
            err = x - phi
            nv = float(n * err.var(ddof=1))
            V = pred[pkey]["V_fixed" if kind == "fixed" else "V_rotated"]
            cell["arms"][f"{est}_{kind}"] = {
                "sqrt_n_bias": float(np.sqrt(n) * err.mean()),
                "sqrt_n_bias_se": float(np.sqrt(n) * err.std(ddof=1) / np.sqrt(R)),
                "n_var": nv, "n_var_se": nv * rel, "prediction": V,
                "z_var": (nv - V) / (nv * rel)}
    # paired difference of n Var (unrestricted - restricted), rep-bootstrap s.e.
    rng = np.random.default_rng(12345)
    for kind in ("avg", "pooled", "fixed"):
        a = np.array([r["est"][f"full_{kind}"] for r in rows])
        b = np.array([r["est"][f"restricted_{kind}"] for r in rows])
        diff = n * (a.var(ddof=1) - b.var(ddof=1))
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, R, R)
            boots.append(n * (a[i].var(ddof=1) - b[i].var(ddof=1)))
        key = "V_fixed" if kind == "fixed" else "V_rotated"
        cell[f"paired_{kind}"] = {
            "n_var_diff": float(diff), "se": float(np.std(boots, ddof=1)),
            "prediction": pred["unrestricted"][key] - pred["restricted"][key],
            "corr": float(np.corrcoef(a, b)[0, 1])}
    b2r = np.array([r["est"]["restricted_b2_fold0"][0] for r in rows])
    b2f = np.array([r["est"]["full_b2_fold0"] for r in rows])
    cell["beta2_fold0"] = {"restricted_mean": float(b2r.mean()),
                           "full_mean": b2f.mean(axis=0).tolist()}
    cell["mean_seconds_per_rep"] = float(np.mean([r["seconds"] for r in rows]))
    return cell


def report(result: dict) -> str:
    pred = result["population"]["predictions"]
    lines = [f"phi* = {pred['phi_star']:.6f}, H = {pred['H']:.6f}; predicted n Var "
             f"(rotated / fixed): unrestricted {pred['unrestricted']['V_rotated']:.3f} / "
             f"{pred['unrestricted']['V_fixed']:.2f}, restricted "
             f"{pred['restricted']['V_rotated']:.3f} / {pred['restricted']['V_fixed']:.2f}", "",
             "| n | reps | estimator | sqrt(n) bias (s.e.) | n Var (s.e.) | prediction | z |",
             "|---:|---:|---|---:|---:|---:|---:|"]
    for n, cell in sorted(result["cells"].items(), key=lambda kv: int(kv[0])):
        for key, c in cell["arms"].items():
            lines.append(f"| {int(n):,} | {cell['reps']} | {key} | {c['sqrt_n_bias']:+.2f} "
                         f"({c['sqrt_n_bias_se']:.2f}) | {c['n_var']:.2f} ({c['n_var_se']:.2f}) "
                         f"| {c['prediction']:.2f} | {c['z_var']:+.2f} |")
    lines.append("")
    lines.append("| n | paired n Var diff (unres - res) | s.e. | prediction | corr |")
    lines.append("|---:|---|---:|---:|---:|")
    for n, cell in sorted(result["cells"].items(), key=lambda kv: int(kv[0])):
        for kind in ("avg", "pooled", "fixed"):
            c = cell[f"paired_{kind}"]
            lines.append(f"| {int(n):,} | {kind}: {c['n_var_diff']:.2f} | {c['se']:.2f} "
                         f"| {c['prediction']:.2f} | {c['corr']:.3f} |")
    return "\n".join(lines)


# ------------------------------------------------------------ main

def population_part(beta2_1: float, mc_draws: int, seed: int) -> dict:
    design = make_design(beta2_1)
    tc = target_check(design)
    print(f"[target] phi* = {tc['phi_star']:.6f} (GL {tc['phi_star_gl']:.6f}), H = {tc['H']:.6f}, "
          f"Z sign changes on Phi = {tc['Z_sign_changes_on_Phi']}, "
          f"Z(ends) = {np.round(tc['Z_at_ends'], 4).tolist()}", flush=True)
    pop = Population(design)
    dim = pop.dim
    A = selector(dim)
    I = np.eye(dim)
    rep_res = representer(pop, A)
    rep_full = representer(pop, I)
    dec = pop.decomposition()
    om_paper = {int(k): np.asarray(v) for k, v in dec["omega"].items()}
    # paper's closed form, written out independently of ds_local_efficiency
    om_lemma = {d_: (2 * d_ - 1) * np.linalg.solve(rep_full["Gamma"][d_],
                                                   rep_full["B"] - rep_full["m"][d_])
                for d_ in (0, 1)}
    a_eq_I = {"omega_coupled_A=I": {str(k): v.tolist() for k, v in rep_full["omega"].items()},
              "omega_lemma": {str(k): v.tolist() for k, v in om_lemma.items()},
              "omega_ds_local_efficiency": {str(k): v.tolist() for k, v in om_paper.items()},
              "max_abs_diff_vs_lemma": float(max(np.max(np.abs(rep_full["omega"][k] - om_lemma[k]))
                                                 for k in (0, 1))),
              "max_abs_diff_vs_code": float(max(np.max(np.abs(rep_full["omega"][k] - om_paper[k]))
                                                for k in (0, 1))),
              "var_o_coupled": rep_full["var_o"], "var_o_code": dec["variances"]["psi_o"],
              "A_out_coupled": rep_full["A_out"].tolist(), "A_out_code": dec["A_out"]}
    print(f"[A=I] max |omega_coupled - omega_lemma| = {a_eq_I['max_abs_diff_vs_lemma']:.2e}, "
          f"vs code {a_eq_I['max_abs_diff_vs_code']:.2e}", flush=True)
    print(f"[restricted] w = {np.round(rep_res['w'], 6).tolist()}, v = {np.round(rep_res['v'], 6).tolist()}; "
          f"omega0 = {np.round(rep_res['omega'][0], 6).tolist()}, "
          f"omega1 = {np.round(rep_res['omega'][1], 6).tolist()}", flush=True)
    print(f"[full]       omega0 = {np.round(rep_full['omega'][0], 6).tolist()}, "
          f"omega1 = {np.round(rep_full['omega'][1], 6).tolist()}", flush=True)
    t0 = time.perf_counter()
    mc = riesz_mc(pop, {"restricted": (rep_res["omega"], A, False),
                        "full (A=I)": (rep_full["omega"], I, True)}, mc_draws, seed)
    # outside-space contrast: the restricted representer against f = D X_2 (not in its space)
    contrast = riesz_mc(pop, {"restricted vs D X'c": (rep_res["omega"], I, True)},
                        mc_draws // 4, seed + 1)
    print(f"[riesz MC] {mc_draws:,} draws, {time.perf_counter() - t0:.0f}s", flush=True)
    for k, res in mc.items():
        print(f"  representer: {k}")
        for row in res["identities"]:
            print(f"    {row['direction']:44s} <r,f>={row['inner_mc']:+.5f}  L={row['L_quad']:+.5f}  "
                  f"diff={row['diff']:+.5f} (se {row['se']:.5f}, z={row['z']:+.2f})")
    print("  contrast (restricted r, unrestricted DX directions; not expected to match):")
    for row in contrast["restricted vs D X'c"]["identities"]:
        if row["direction"].startswith("D X'c"):
            print(f"    {row['direction']:44s} <r,f>={row['inner_mc']:+.5f}  L={row['L_quad']:+.5f}  "
                  f"diff={row['diff']:+.5f} (se {row['se']:.5f})")
    t0 = time.perf_counter()
    sv = {}
    for k, AA, rep in (("restricted", A, rep_res), ("full (A=I)", I, rep_full)):
        runs = [sieve_check(pop, AA, SIEVE_DRAWS, seed + 100 + j) for j in range(SIEVE_SEEDS)]
        sv[k] = {}
        for d_ in (0, 1):
            o = np.array([r[f"omega{d_}_sieve"] for r in runs])
            mean, se = o.mean(axis=0), o.std(axis=0, ddof=1) / np.sqrt(len(o))
            sv[k][f"omega{d_}"] = {"sieve_mean": mean.tolist(), "sieve_se": se.tolist(),
                                   "closed": rep["omega"][d_].tolist(),
                                   "z": ((mean - rep["omega"][d_]) / se).tolist()}
        sv[k]["n_basis_per_block"] = runs[0]["n_basis_per_block"]
    print(f"[sieve] {SIEVE_SEEDS} x {SIEVE_DRAWS:,} draws, {time.perf_counter() - t0:.0f}s", flush=True)
    for k in sv:
        for d_ in (0, 1):
            c = sv[k][f"omega{d_}"]
            print(f"  {k} omega{d_}: sieve {np.round(c['sieve_mean'], 5).tolist()} "
                  f"(se {np.round(c['sieve_se'], 5).tolist()}) vs closed {np.round(c['closed'], 5).tolist()}, "
                  f"z {np.round(c['z'], 2).tolist()}")
    pred = predictions(pop, A)
    for k in ("unrestricted", "restricted"):
        p = pred[k]
        print(f"[pred {k}] var U={p['var_U']:.5f} o={p['var_o']:.5f} T={p['var_T']:.5f} "
              f"gamma={p['var_gamma']:.5f}; sigma2={p['sigma2']:.5f}; "
              f"n Var rotated={p['V_rotated']:.3f}, fixed={p['V_fixed']:.2f}")
    for k, rep in (("restricted", rep_res), ("full (A=I)", rep_full)):
        print(f"[MC check {k}] var_o formula {rep['var_o']:.5f} MC {mc[k]['var_o_mc']:.5f}; "
              f"A_out formula {np.round(rep['A_out'], 4).tolist()} MC {np.round(mc[k]['A_out_mc'], 4).tolist()}")
    return {"design": repr(design), "target": tc, "A": A.tolist(),
            "restricted": {"w": rep_res["w"].tolist(), "v": rep_res["v"].tolist(),
                           "omega": {str(k): v.tolist() for k, v in rep_res["omega"].items()},
                           "var_o": rep_res["var_o"], "A_out": rep_res["A_out"].tolist()},
            "A_equals_I": a_eq_I, "riesz_mc": mc, "riesz_contrast": contrast,
            "sieve": sv, "mc_draws": mc_draws, "predictions": pred}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("population", "mc", "report"), default="population")
    ap.add_argument("--beta2-1", type=float, default=0.4)
    ap.add_argument("--mc-draws", type=int, default=4_000_000)
    ap.add_argument("--cells", nargs="+", default=["16000:1000", "64000:500"])
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if args.part == "report":
        print(report(result))
        return
    if args.part == "population":
        result["population"] = population_part(args.beta2_1, args.mc_draws, args.seed)
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"[wrote] {summary_path}")
        return
    if "population" not in result:
        result["population"] = {"predictions": predictions(Population(make_design(args.beta2_1)),
                                                           selector(3))}
    pred = result["population"]["predictions"]
    result.setdefault("cells", {})
    result["mc_settings"] = {"seed": args.seed, "beta2_1": args.beta2_1}
    for spec in args.cells:
        n, reps = (int(x) for x in spec.split(":"))
        rows = run_cell(args.beta2_1, n, reps, args.seed, args.workers, args.out)
        result["cells"][str(n)] = summarize(rows, n, pred)
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
    print(report(result))
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
