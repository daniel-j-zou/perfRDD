"""Where does the small positive sqrt(N) bias of the four-role estimator come from?

The four-role differing-slopes estimator (``ds_local_efficiency_sim.py``,
paper design) shows a sqrt(N) bias of about +0.5 to +0.8 at N = 16k to 64k
(``DS_SIMPLIFICATION_CHECKS_20261013.md``).  This script decomposes it.

On each simulated data set and each of the four role rotations, the same
estimator is rerun with each nuisance group either estimated or replaced by
its true value:

    G  first stage (gamma-hat everywhere, including the evaluation residuals)
    O  outcome fit (a-hat, beta2-hat)
    D  density fit (g-hat, rho_X-hat and their tails)
    E  trim endpoints (empirical quantiles of the fitted index)

For all 16 combinations it records, at the true optimum phi*, the score
U-hat'(phi*) and curvature U-hat''(phi*), both normalized by the evaluation
fold size as in the paper.  A score with nonzero mean means a first-order
bias.  A centred score with a biased argmax means second-order (argmax)
nonlinearity.

It also records full argmaxes for four estimators: the fully estimated and
the fully oracle versions; the average of the four rotated argmaxes (the
paper's rotated estimator); and the argmax of the averaged criterion.

Run:
    python -m experiments.scripts.ds_bias_decomposition --n 16000 64000 --reps 1000 --workers 7
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_local_efficiency import DESIGNS
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, _xt, generate, ols_first_stage, outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_bias_decomposition"
DESIGN = DESIGNS["paper_baseline"]
GAMMA = np.concatenate(([DESIGN.gamma0], np.asarray(DESIGN.gamma_x)))
BETA2 = np.asarray(DESIGN.beta2)
S_BETA = float(BETA2 @ np.asarray(DESIGN.gamma_x))  # beta2' E(X|T=t) = S_BETA * t (Var T = 1)
Z_EPS = float(stats.norm.ppf(1 - DESIGN.eps))
L0, U0 = DESIGN.phi0 - Z_EPS, DESIGN.phi0 + Z_EPS
TLO, THI = DESIGN.T_support
PHI_STAR = 0.5451028241935099   # Gauss-Legendre quadrature (ds_identification_check.py)
COMBOS = list(itertools.product((0, 1), repeat=4))  # (G, O, D, E); 1 = estimated
H_STAR = -0.10753350836035151   # U_eps''(phi*)
# Second-order argmax bias constant B = d sigma^2 / (2 H^2) - U3 sigma^2 / (2 H^3), with
# U3 = U_eps'''(phi*), from the analytic four-role variance (ds_local_efficiency.Population)
# evaluated near phi*: sigma^2 = 0.35950, d sigma^2 / d phi = 0.21209, U3 = 0.0857630.
B_STAR = 21.568


class TrueTails:
    """Fixed-support survival and weighted tail for T ~ N(0, 1), E(X|T) = gamma_x T."""

    def survival(self, t):
        t = np.clip(t, TLO, THI)
        return stats.norm.cdf(THI) - stats.norm.cdf(t)

    def weighted_tail(self, t, b2):
        t = np.clip(t, TLO, THI)
        return float(b2 @ np.asarray(DESIGN.gamma_x)) * (stats.norm.pdf(t) - stats.norm.pdf(THI))

    def density(self, t):
        return np.where((t >= TLO) & (t <= THI), stats.norm.pdf(t), 0.0)

    def density_derivative(self, t):
        return np.where((t > TLO) & (t < THI), -t * stats.norm.pdf(t), 0.0)

    def weighted_density(self, t, b2):
        s = float(b2 @ np.asarray(DESIGN.gamma_x))
        return np.where((t >= TLO) & (t <= THI), s * t * stats.norm.pdf(t), 0.0)

    def weighted_density_derivative(self, t, b2):
        s = float(b2 @ np.asarray(DESIGN.gamma_x))
        return np.where((t > TLO) & (t < THI), s * (1 - t ** 2) * stats.norm.pdf(t), 0.0)


TRUE_TAILS = TrueTails()


def a_true(e):
    return np.polynomial.Polynomial(DESIGN.a_coef)(e)


def criterion_pieces(s, folds, gamma_hat):
    """Fitted nuisances for one rotation, with gamma-hat and with the true gamma."""
    out = {}
    for g_flag, gam in ((1, gamma_hat), (0, GAMMA)):
        eta = s.Q - _xt(s.X) @ gam
        fit = outcome_fit(s, folds["outcome"], eta, DESIGN.J)
        T_idx = _xt(s.X[folds["index"]]) @ gam
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], DESIGN.T_support)
        l_hat = DESIGN.phi0 - float(np.quantile(T_idx, 1.0 - DESIGN.eps))
        u_hat = DESIGN.phi0 - float(np.quantile(T_idx, DESIGN.eps))
        out[g_flag] = {"eta_eval": eta[folds["utility"]], "fit": fit, "tails": tails,
                       "ends": (l_hat, u_hat)}
    return out


def make_criterion(pieces, combo, n_u):
    g, o, d, e_ = combo
    p = pieces[g]
    eta = p["eta_eval"]
    lo, hi = p["ends"] if e_ else (L0, U0)
    keep = (eta >= lo) & (eta <= hi)
    e = eta[keep]
    if o:
        amc = p["fit"].a(e) - DESIGN.cost
        b2 = p["fit"].beta2
    else:
        amc = a_true(e) - DESIGN.cost
        b2 = BETA2
    tails = p["tails"] if d else TRUE_TAILS

    def value(phi):
        t = phi - e
        return float(np.sum(amc * tails.survival(t) + tails.weighted_tail(t, b2)) / n_u)

    def score(phi):
        t = phi - e
        return -float(np.sum(amc * tails.density(t) + tails.weighted_density(t, b2)) / n_u)

    def curv(phi):
        t = phi - e
        return -float(np.sum(amc * tails.density_derivative(t)
                             + tails.weighted_density_derivative(t, b2)) / n_u)

    return value, score, curv


def argmax(value):
    lo, hi = DESIGN.policy
    grid = np.linspace(lo, hi, 61)
    vals = np.array([value(x) for x in grid])
    k = int(np.argmax(vals))
    a_, b_ = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: -value(x), bounds=(a_, b_), method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x)


def replicate(task):
    n, seed = task
    s = generate(DESIGN, n, seed)
    rng = np.random.default_rng(seed + 7_000_001)
    blocks = np.array_split(rng.permutation(n), 4)
    scores = np.zeros((4, 16))
    curvs = np.zeros((4, 16))
    phis_full, phis_oracle = [], []
    vals_full, vals_oracle = [], []
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        gamma_hat = ols_first_stage(s, folds["gamma"])
        pieces = criterion_pieces(s, folds, gamma_hat)
        n_u = len(folds["utility"])
        for c, combo in enumerate(COMBOS):
            value, score, curv = make_criterion(pieces, combo, n_u)
            scores[k, c] = score(PHI_STAR)
            curvs[k, c] = curv(PHI_STAR)
            if combo == (1, 1, 1, 1):
                phis_full.append(argmax(value))
                vals_full.append(value)
            if combo == (0, 0, 0, 0):
                phis_oracle.append(argmax(value))
                vals_oracle.append(value)
    pooled_full = argmax(lambda x: sum(v(x) for v in vals_full) / 4.0)
    pooled_oracle = argmax(lambda x: sum(v(x) for v in vals_oracle) / 4.0)
    return {
        "n": n, "scores": scores.tolist(), "curvs": curvs.tolist(),
        "fixed_full": phis_full[0], "avg_full": float(np.mean(phis_full)),
        "pooled_full": pooled_full,
        "fixed_oracle": phis_oracle[0], "avg_oracle": float(np.mean(phis_oracle)),
        "pooled_oracle": pooled_oracle,
    }


def summarize(rows, n):
    out = {"n": n, "reps": len(rows)}
    for key in ("fixed_full", "avg_full", "pooled_full", "fixed_oracle", "avg_oracle",
                "pooled_oracle"):
        err = np.array([r[key] for r in rows]) - PHI_STAR
        out[key] = {"sqrtN_bias": float(np.sqrt(n) * err.mean()),
                    "se": float(np.sqrt(n) * err.std(ddof=1) / np.sqrt(len(err))),
                    "n_var": float(n * err.var(ddof=1))}
    S = np.array([r["scores"] for r in rows])     # reps x 4 x 16
    C = np.array([r["curvs"] for r in rows])
    comb = {}
    for c, combo in enumerate(COMBOS):
        sc = S[:, :, c]
        per_rep = sc.mean(axis=1)                  # average over rotations
        label = "".join(name if f else "-" for name, f in zip("GODE", combo))
        comb[label] = {
            "sqrtN_score_mean": float(np.sqrt(n) * sc.mean()),
            "se": float(np.sqrt(n) * per_rep.std(ddof=1) / np.sqrt(len(per_rep))),
            "curv_mean": float(C[:, :, c].mean()),
        }
    out["score_by_combo"] = comb
    # Paired comparisons: subtract each replication's own score term -Psi/H
    # (averaged over the four rotations). This removes the score's noise and
    # its mean, and leaves the second-order argmax part.
    score_term = -S[:, :, 15].mean(axis=1) / H_STAR
    A = np.array([r["avg_full"] for r in rows])
    P = np.array([r["pooled_full"] for r in rows])
    paired = {}
    for label, d in (("average_minus_score", A - PHI_STAR - score_term),
                     ("pooled_minus_score", P - PHI_STAR - score_term),
                     ("average_minus_pooled", A - P),
                     ("combined_minus_score", (4 * P - A) / 3 - PHI_STAR - score_term)):
        paired[label] = {"sqrtN": float(np.sqrt(n) * d.mean()),
                         "se": float(np.sqrt(n) * d.std(ddof=1) / np.sqrt(len(d)))}
    paired["predicted"] = {"4B": 4 * B_STAR / np.sqrt(n), "B": B_STAR / np.sqrt(n),
                           "3B": 3 * B_STAR / np.sqrt(n), "combined": 0.0}
    out["paired"] = paired
    return out


def density_spline_bias(n: int) -> float:
    """Deterministic score bias of the density fit, in sqrt(N)-bias units.

    With the true index, the density projection is linear in the empirical
    measure, so its mean is the population spline projection. The score bias
    is therefore L_g(g_L - g) + L_rho(rho_L - rho), computed by quadrature
    with the same basis size rule as the estimator (n_T = N/4).
    """
    from experiments.methods.spline_density import (
        _basis_info, evaluate_basis_zero_outside, lebesgue_gram, spline_basis_dimension)
    from scipy.interpolate import BSpline

    x, w = np.polynomial.legendre.leggauss(200)

    def gl(lo, hi):
        return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w

    L = spline_basis_dimension(n // 4)
    info = _basis_info(int(L), DESIGN.T_support)
    G = lebesgue_gram(info)
    knots = np.unique(info["t"])
    pieces = [gl(a, b) for a, b in zip(knots[:-1], knots[1:])]
    tt = np.concatenate([p[0] for p in pieces])
    ww = np.concatenate([p[1] for p in pieces])
    Bm = evaluate_basis_zero_outside(tt, info)
    g = stats.norm.pdf(tt)
    rho = S_BETA * tt * g
    sp_g = BSpline(info["t"], np.linalg.solve(G, Bm.T @ (ww * g)), 3)
    sp_r = BSpline(info["t"], np.linalg.solve(G, Bm.T @ (ww * rho)), 3)
    e, we = gl(L0, U0)
    t = PHI_STAR - e
    dg = sp_g(t) - stats.norm.pdf(t)
    dr = sp_r(t) - S_BETA * t * stats.norm.pdf(t)
    score_bias = -np.sum(we * stats.norm.pdf(e) * ((a_true(e) - DESIGN.cost) * dg + dr))
    return float(np.sqrt(n) * (-score_bias / H_STAR))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, nargs="+", default=[16000, 64000])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, n in enumerate(args.n):
        tasks = [(n, args.seed + 100_000 * i + r) for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(replicate, tasks, chunksize=2))
        (args.out / f"rows_{n}.json").write_text(json.dumps(rows))
        summ = summarize(rows, n)
        results.append(summ)
        print(json.dumps(summ, indent=1), flush=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
