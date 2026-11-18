"""Simulations for the referee concerns predicted on 2026-11-18 (author: "do all five").

Each design changes one feature of a design already in the paper and runs the bootstrap
check of Table 2 (999 full re-estimation draws per data set, percentile and normal
intervals) with the estimator of Section 3 as used in Section 5.5 (four-role split, OLS
first stage, spline outcome fit on J, spline g and rho_X, rotated average; trim window
capped at the central 99% of eta-hat on the evaluation fold).

Section 5 family (the displayed design: X ~ N(0, I_3), T = (1, 1, 1)'X / sqrt3, eta ~ N(0, 1),
alpha = 2 + 0.4 e + 0.8 tanh(1.5 e), b = e^2/2 + 0.3 sin(2 e), c = 2.25, eps = 0.1,
phi0 = 0, J = [-1.75, 1.75], T = [-3, 3], Phi = [-1.5, 1.5]).  Data sets use Table 2's
seeds, so X, eta and the noise are those of Table 2's data sets:

* ``cluster_g31``: the units fall into G = 31 equal groups ("days").  The latent ability
  has a group component, eta = sqrt(1 - rho) eta0 + sqrt(rho) u_g with rho = 0.025 (the
  share of the taxi residual's variance between calendar days), and the treatment
  effect has a group shift w_g ~ N(0, tau^2), with tau set so that at n = 64,000 the
  group part of Var(phi-hat) equals the independent part (resampling rides understates
  the variance by about the taxi factor 1.39^2).  eta is N(0, 1) and E w_g = 0, so the
  target is Table 2's.  Both the unit bootstrap and the group bootstrap (resample the
  31 groups) are run on every data set.
* ``interact_k10``, ``interact_k20``: the effect has an extra term kappa T eta (kappa =
  0.1, 0.2), outside the model (the effect is no longer additive in X and eta).  The
  target maximizes the true utility; ``--limits`` computes the estimator's large-sample
  limit from the population projection of its outcome fit.
* ``grid_h27``: the score is observed on a grid of width 0.27 (the taxi's $0.40 fare step
  is 0.27 standard deviations of its residual near $13), Q = 0.27 round(Q*/0.27), and
  D = 1{Q >= 0}.  Targets: the latent optimum, and the best grid cell.
* ``flat_l50``, ``flat_l25``: alpha - c and beta2 multiplied by lambda = 0.5, 0.25, so
  the utility and its gain over phi0 are lambda times Table 2's (the taxi gain is under a
  cent per ride); the target is unchanged.
* ``skew``: eta has a standardized Johnson SU law with the skewness (2.03) and kurtosis
  (17.5) of the taxi residual; the target is recomputed for that law.

Appendix 7 family (``location_scale/hetero.py``: X1 ~ N(0, 1), X2 = +-1/2, T = X1 + X2/2,
phi0 = 1, eps = 0.05, Phi = [-0.2, 2.2], alpha(e) = 0.4 e - 0.35 + 0.1 tanh(2 e),
b(e) = sin e, beta1 = (0.5, 0.2), beta2 = (0.3, 0), outcome noise s.d. 0.5, c = 0):

* ``hetero_a148``, ``hetero_b148``: the Appendix 7 designs (a) and (b) with spread
  factor 1.48 (k = 0.2791): eta = s(X) zeta, s(X) = exp(k (0.4 X1 + 0.2 X2)); the effect
  and baseline act on zeta (a) or on eta (b).
* ``hetero_ca``, ``hetero_cb``: the dependence of eta on X matched to all three taxi
  diagnostics (R^2 of eta on functions of X 0.093, of eta^2 0.203, spread factor across
  X1 deciles 1.48): eta = m(X1) + s(X) zeta, with m(X1) = A (e^{-B X1} - e^{B^2/2})
  + A B e^{B^2/2} X1 (the nonlinear part, orthogonal to 1, X1 and X2, so T stays the
  linear predictor), A = 1.056159, B = 0.588515, k = 0.335396 (``--calibrate``).
  Effect and baseline act on zeta (ca) or on eta (cb).
  Targets as in Appendix 7: with effects of zeta, the supported optimum with the window
  on zeta (eps quantiles of (phi0 - T - m)/s); with effects of eta, the window on eta of
  Section 2.3, capped at the central 99% of eta as the estimator caps it.

Rows go to ``<out>/<design>_n<n>.jsonl`` as data sets finish; ``--rep-start a --reps b``
shards a cell for Slurm arrays; seeds depend only on (seed, n, rep), so the shards merge
into one run, and designs of one family share data-set draws (common random numbers).

Run (from code/):
    python -m experiments.scripts.ds_review_ablations --population
    python -m experiments.scripts.ds_review_ablations --limits
    python -m experiments.scripts.ds_review_ablations --design skew --n 4000 --reps 4 \
        --boot 19 --workers 4 --out experiments/runs/ds_review_test
    python -m experiments.scripts.ds_review_ablations --check-table2 \
        experiments/runs/ds_four_role_coverage_nl_999
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.stats import johnsonsu, norm

from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_four_role_coverage import (
    SplineTailPieces, _avg_and_pooled, _load_rows, summarize,
)
from experiments.scripts.ds_local_efficiency import DESIGNS, Curve
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, Sample, _design_matrix, _xt, ols_first_stage, outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_review_ablations"
SEED = 20261019                # Table 2's seed
WINDOW_TAIL = 0.005            # trim window capped at the central 99% of eta-hat
J_TAIL = 0.0025                # Appendix 7 family: J = central 99.5% of eta
EDGE_TOL = 1e-3
BASE = DESIGNS["paper_nonlinear"]
GAMMA_X = np.asarray(BASE.gamma_x, dtype=float)
LAM0 = float(np.asarray(BASE.beta2) @ GAMMA_X)     # beta2'gamma: E(beta2'X | T = t) = LAM0 t
A_CURVE, B_CURVE = Curve(BASE.a_coef), Curve(BASE.b_coef)

# Johnson SU law with the taxi residual's skewness 2.03 and kurtosis 17.5 (excess 14.5):
# eta = (sinh((Z - SU_A) / SU_B) - mean) / sd, Z ~ N(0, 1) (``--calibrate``).
SU_A, SU_B = -0.659853, 1.271042
_MU = -np.sinh(SU_A / SU_B) * np.exp(0.5 / SU_B ** 2)
_SD = float(np.sqrt(0.5 * (np.cosh(-2 * SU_A / SU_B) * np.exp(2 / SU_B ** 2) - 1) - _MU ** 2))

# Appendix 7 family constants (location_scale/common.py, hetero.py).
HET_BETA1, HET_BETA2 = np.array([0.5, 0.2]), np.array([0.3, 0.0])
HET_SIG_Y = 0.5
K148 = 0.2791                  # spread factor 1.48 (hetero.py calib; the saved targets use it)
C_A, C_B, C_K = 1.056159, 0.588515, 0.335396   # taxi-matched dependence (``--calibrate``)


@dataclasses.dataclass(frozen=True)
class Cfg:
    key: str
    family: str                # "s5" (Section 5 design) or "het" (Appendix 7 design)
    note: str
    phi0: float
    eps: float
    J: Tuple[float, float]
    T_support: Tuple[float, float]
    policy: Tuple[float, float]
    cost: float
    law: str = "normal"        # s5: "normal" or "su"
    lam: float = 1.0           # s5: alpha - c and beta2 times lam
    kappa: float = 0.0         # s5: effect + kappa T eta
    grid_h: float = 0.0        # s5: score observed on multiples of grid_h, D = 1{Q >= phi0}
    groups: int = 0            # s5: number of groups (0: none)
    rho: float = 0.0           # s5: share of Var(eta) between groups
    tau: float = 0.0           # s5: s.d. of the group shift of the effect
    k: float = 0.0             # het: s(X) = exp(k (0.4 X1 + 0.2 X2))
    A: float = 0.0             # het: nonlinear mean m(X1), amplitude
    B: float = 1.0             # het: nonlinear mean m(X1), rate
    latent: str = "zeta"       # het: effect and baseline act on "zeta" or "eta"


def _s5(key, note, **kw) -> Cfg:
    return Cfg(key=key, family="s5", note=note, phi0=BASE.phi0, eps=BASE.eps, J=BASE.J,
               T_support=BASE.T_support, policy=BASE.policy, cost=BASE.cost, **kw)


# ------------------------------------------------------------ Appendix 7 family pieces

def m_tilde(x1, A, B):
    """Nonlinear part of the score mean, orthogonal to 1, X1 and X2 (X1 ~ N(0, 1))."""
    e = np.exp(0.5 * B * B)
    return A * (np.exp(-B * np.asarray(x1, float)) - e) + A * B * e * np.asarray(x1, float)


def het_scale(x1, x2, k):
    return np.exp(k * (0.4 * np.asarray(x1, float) + 0.2 * np.asarray(x2, float)))


def het_alpha(e):
    return 0.4 * e - 0.35 + 0.1 * np.tanh(2.0 * e)


class XGrid:
    """Quadrature over X = (X1, X2): a fine X1 grid times X2 = -1/2, 1/2."""

    def __init__(self, A, B, k, m=1601, lim=8.0):
        x1 = np.linspace(-lim, lim, m)
        w1 = norm.pdf(x1)
        w1 /= w1.sum()
        self.X1 = np.concatenate([x1, x1])
        self.X2 = np.concatenate([np.full(m, -0.5), np.full(m, 0.5)])
        self.w = 0.5 * np.concatenate([w1, w1])
        self.T = self.X1 + 0.5 * self.X2
        self.m = m_tilde(self.X1, A, B)
        self.s = het_scale(self.X1, self.X2, k)


def _mixture_quantile(cdf, p, lo=-30.0, hi=30.0):
    return float(optimize.brentq(lambda t: cdf(t) - p, lo, hi, xtol=1e-13))


def het_eta_quantile(p, A, B, k):
    xg = XGrid(A, B, k, m=4001, lim=9.0)
    return _mixture_quantile(lambda e: float(np.sum(xg.w * norm.cdf((e - xg.m) / xg.s))), p)


def het_T_quantile(p):
    return _mixture_quantile(lambda t: 0.5 * (norm.cdf(t + 0.25) + norm.cdf(t - 0.25)), p)


def _het(key, note, k, A=0.0, B=1.0, latent="zeta") -> Cfg:
    J = (het_eta_quantile(J_TAIL, A, B, k), het_eta_quantile(1 - J_TAIL, A, B, k))
    return Cfg(key=key, family="het", note=note, phi0=1.0, eps=0.05, J=J,
               T_support=(-3.5, 3.5), policy=(-0.2, 2.2), cost=0.0, k=k, A=A, B=B,
               latent=latent)


# tau for cluster_g31: a common shift w of the effect moves the target by
# d phi* / d w = -1.9047 (``--calibrate``); Table 2's n Var at 64,000 is 28.3, and
# 1.9047^2 tau^2 / 31 = 28.3 / 64000 gives tau = 0.0615.
TAU_G31 = 0.0615

CFGS: Dict[str, Cfg] = {c.key: c for c in (
    _s5("base", "Section 5 design (reproduces Table 2)"),
    _s5("cluster_g31", "31 groups: eta group share 0.025, effect group shift",
        groups=31, rho=0.025, tau=TAU_G31),
    _s5("interact_k10", "effect + 0.1 T eta", kappa=0.1),
    _s5("interact_k20", "effect + 0.2 T eta", kappa=0.2),
    _s5("grid_h27", "score on a grid of width 0.27", grid_h=0.27),
    _s5("flat_l50", "utility scaled by 0.5", lam=0.5),
    _s5("flat_l25", "utility scaled by 0.25", lam=0.25),
    _s5("skew", "eta Johnson SU, skewness 2.03, kurtosis 17.5", law="su"),
    _het("hetero_a148", "Appendix 7 design (a), spread factor 1.48", K148),
    _het("hetero_b148", "Appendix 7 design (b), spread factor 1.48", K148, latent="eta"),
    _het("hetero_ca", "taxi-matched dependence, effect of zeta", C_K, C_A, C_B),
    _het("hetero_cb", "taxi-matched dependence, effect of eta", C_K, C_A, C_B,
         latent="eta"),
)}


# ------------------------------------------------------------ data

def su_eta(z):
    return (np.sinh((z - SU_A) / SU_B) - _MU) / _SD


def su_pdf(e):
    x = _MU + _SD * np.asarray(e, float)
    return _SD * SU_B / np.sqrt(1.0 + x * x) * norm.pdf(SU_A + SU_B * np.arcsinh(x))


def su_ppf(p):
    return su_eta(norm.ppf(p))


def generate(cfg: Cfg, n: int, seed: int):
    """Sample and, for grouped designs, the group of each unit."""
    rng = np.random.default_rng(seed)
    if cfg.family == "s5":
        # Draw order of ds_local_efficiency_sim.generate: X, eta, noise (Table 2's data).
        X = rng.standard_normal((n, 3))
        z = rng.standard_normal(n)
        noise = rng.standard_normal(n)
        group = None
        w = 0.0
        if cfg.law == "su":
            eta = su_eta(z)
        else:
            eta = z
        if cfg.groups:
            grng = np.random.default_rng([seed, 6])
            u = grng.standard_normal(cfg.groups)
            wg = cfg.tau * grng.standard_normal(cfg.groups)
            group = np.arange(n) % cfg.groups
            eta = np.sqrt(1.0 - cfg.rho) * eta + np.sqrt(cfg.rho) * u[group]
            w = wg[group]
        T = X @ GAMMA_X
        Q = T + eta
        if cfg.grid_h:
            Q = cfg.grid_h * np.round(Q / cfg.grid_h)
            D = (Q >= cfg.phi0 - 1e-12).astype(float)
        else:
            D = (Q > cfg.phi0).astype(float)
        if cfg.lam == 1.0:       # bit-identical to Table 2's outcome when nothing else changes
            effect = A_CURVE(eta) + X @ np.asarray(BASE.beta2)
        else:
            effect = (cfg.cost + cfg.lam * (A_CURVE(eta) - cfg.cost)
                      + cfg.lam * (X @ np.asarray(BASE.beta2)))
        if cfg.kappa:
            effect = effect + cfg.kappa * T * eta
        if cfg.groups:
            effect = effect + w
        Y = (B_CURVE(eta) + X @ np.asarray(BASE.beta1) + D * effect
             + BASE.sigma_arm[0] * noise)
        return Sample(X=X, Q=Q, D=D, Y=Y), group
    X1 = rng.standard_normal(n)
    X2 = rng.integers(0, 2, n) - 0.5
    zeta = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    eta = m_tilde(X1, cfg.A, cfg.B) + het_scale(X1, X2, cfg.k) * zeta
    Q = X1 + 0.5 * X2 + eta
    D = (Q > cfg.phi0).astype(float)
    lat = zeta if cfg.latent == "zeta" else eta
    X = np.column_stack([X1, X2])
    Y = (np.sin(lat) + X @ HET_BETA1 + D * (het_alpha(lat) + X @ HET_BETA2)
         + HET_SIG_Y * noise)
    return Sample(X=X, Q=Q, D=D, Y=Y), None


def _resample(s: Sample, idx: np.ndarray) -> Sample:
    return Sample(X=s.X[idx], Q=s.Q[idx], D=s.D[idx], Y=s.Y[idx])


# ------------------------------------------------------------ estimator

def estimate(s: Sample, cfg: Cfg, blocks) -> tuple:
    """Rotated average, pooled maximizer and fixed assignment on one split (the estimator
    of ``ds_r2_balance_coverage.estimate`` without the oracle arm)."""
    grid = np.linspace(cfg.policy[0], cfg.policy[1], 61)
    crit, vals = [], []
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        n_u = len(folds["utility"])
        gamma_hat = ols_first_stage(s, folds["gamma"])
        eta_hat = s.Q - _xt(s.X) @ gamma_hat
        T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], cfg.T_support)
        l_hat = cfg.phi0 - float(np.quantile(T_idx, 1.0 - cfg.eps))
        u_hat = cfg.phi0 - float(np.quantile(T_idx, cfg.eps))
        e = eta_hat[folds["utility"]]
        l_hat = max(l_hat, float(np.quantile(e, WINDOW_TAIL)))
        u_hat = min(u_hat, float(np.quantile(e, 1.0 - WINDOW_TAIL)))
        e = e[(e >= l_hat) & (e <= u_hat)]
        fit = outcome_fit(s, folds["outcome"], eta_hat, cfg.J)
        pieces = SplineTailPieces(tails, e, n_u)
        amc = fit.a(e) - cfg.cost
        b2 = np.asarray(fit.beta2)
        row = []
        for x in grid:
            (sg,), sh = pieces.sums(x, [amc])
            row.append((sg + float(sh @ b2)) / n_u)
        crit.append(lambda x, p=pieces, a=amc, b=b2: p.value(x, a, b))
        vals.append(row)
    out = {}
    out["avg"], out["pooled"], rot = _avg_and_pooled(crit, grid, np.array(vals))
    out["fixed"] = rot[0]
    return out, rot


def at_edge(cfg: Cfg, x: float) -> bool:
    return min(abs(x - cfg.policy[0]), abs(x - cfg.policy[1])) < EDGE_TOL


def _bootstrap(s, cfg, n_boot, idx_draw, split_seed) -> tuple:
    boot = {"avg": [], "pooled": [], "fixed": []}
    failures = edge = 0
    for b in range(n_boot):
        idx = idx_draw()
        m = len(idx)
        try:
            draw, drot = estimate(_resample(s, idx), cfg, np.array_split(
                np.random.default_rng(split_seed(b)).permutation(m), 4))
        except (np.linalg.LinAlgError, ValueError):
            failures += 1
            continue
        edge += int(any(at_edge(cfg, p) for p in drot) or at_edge(cfg, draw["pooled"]))
        for key, val in draw.items():
            boot[key].append(val)
    return boot, failures, edge


def replicate(task) -> dict:
    key, n, rep, seed, n_boot = task
    cfg = CFGS[key]
    t0 = time.perf_counter()
    s, group = generate(cfg, n, seed)
    est, rot = estimate(s, cfg, np.array_split(
        np.random.default_rng([seed, 1]).permutation(n), 4))
    brng = np.random.default_rng([seed, 2])
    boot, failures, edge = _bootstrap(s, cfg, n_boot, lambda: brng.integers(0, n, n),
                                      lambda b: [seed, 3, b])
    row = {"rep": rep, "seed": seed, "design": key, "est": est, "rotations": rot,
           "edge_rotations": int(sum(at_edge(cfg, p) for p in rot)),
           "edge_pooled": bool(at_edge(cfg, est["pooled"])), "boot_draws_at_edge": edge,
           "boot": boot, "failures": failures}
    if group is not None:
        members = [np.flatnonzero(group == g) for g in range(cfg.groups)]
        grng = np.random.default_rng([seed, 4])
        gboot, gfail, gedge = _bootstrap(
            s, cfg, n_boot,
            lambda: np.concatenate([members[g] for g in grng.integers(0, cfg.groups,
                                                                       cfg.groups)]),
            lambda b: [seed, 5, b])
        row.update({"gboot": gboot, "gfailures": gfail, "gboot_draws_at_edge": gedge})
    row["seconds"] = time.perf_counter() - t0
    return row


# ------------------------------------------------------------ population: Section 5 family

def _s5_law(cfg: Cfg):
    if cfg.law == "su":
        return su_pdf, su_ppf
    return norm.pdf, norm.ppf


def s5_window(cfg: Cfg) -> Tuple[float, float]:
    _, ppf = _s5_law(cfg)
    z = float(norm.ppf(1.0 - cfg.eps))               # T ~ N(0, 1)
    return (max(cfg.phi0 - z, float(ppf(WINDOW_TAIL))),
            min(cfg.phi0 + z, float(ppf(1.0 - WINDOW_TAIL))))


def s5_curves(cfg: Cfg, alpha=None, slope=None, cost=None):
    """U(phi) and Z(phi) = -U'(phi) of a Section 5 family design by quadrature over eta.

    U(phi) = int_W [(a(e) - c) Gbar(phi - e) + (slope + kappa e) g(phi - e)] f(e) de, with
    a = c + lam (alpha - c), slope = lam beta2'gamma (E[beta2'X 1{T > t}] = beta2'gamma g(t)
    and E[T 1{T > t}] = g(t) for T ~ N(0, 1)).  ``alpha``/``slope`` replace the effect
    (the pseudo-utility of a limit)."""
    pdf, _ = _s5_law(cfg)
    l, u = s5_window(cfg)
    c = cfg.cost if cost is None else cost
    if alpha is None:
        alpha = lambda e: cfg.cost + cfg.lam * (A_CURVE(e) - cfg.cost)
        slope = cfg.lam * LAM0
        kappa = cfg.kappa
    else:
        kappa = 0.0

    def U(phi):
        f = lambda e: ((float(alpha(np.array([e]))[0]) - c) * norm.sf(phi - e)
                       + (slope + kappa * e) * norm.pdf(phi - e)) * pdf(e)
        return quad(f, l, u, limit=400, epsabs=1e-13, epsrel=1e-12)[0]

    def Z(phi):
        f = lambda e: ((float(alpha(np.array([e]))[0]) - c) + (slope + kappa * e) * (phi - e)) \
            * norm.pdf(phi - e) * pdf(e)
        return quad(f, l, u, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
    return U, Z


def _argmax_by_Z(U, Z, policy, m=241) -> Tuple[float, bool]:
    grid = np.linspace(policy[0], policy[1], m)
    uv = np.array([U(x) for x in grid])
    k = int(np.argmax(uv))
    if k in (0, m - 1):
        return float(grid[k]), False
    zv = [Z(grid[k - 1]), Z(grid[k + 1])]
    if not (zv[0] < 0.0 < zv[1]):
        res = optimize.minimize_scalar(lambda x: -U(x), bounds=(grid[k - 1], grid[k + 1]),
                                       method="bounded", options={"xatol": 1e-12})
        return float(res.x), True
    return float(optimize.brentq(Z, grid[k - 1], grid[k + 1], xtol=1e-14)), True


def s5_population(cfg: Cfg) -> dict:
    U, Z = s5_curves(cfg)
    phi, interior = _argmax_by_Z(U, Z, cfg.policy)
    h = 1e-5
    out = {"key": cfg.key, "family": cfg.family, "note": cfg.note, "phi_star": phi,
           "interior": interior, "H": -(Z(phi + h) - Z(phi - h)) / (2 * h),
           "window": list(s5_window(cfg)), "J": list(cfg.J), "policy": list(cfg.policy),
           "U_gain_over_phi0": U(phi) - U(cfg.phi0), "V_rotated": None}
    if cfg.grid_h:
        hg = cfg.grid_h
        cells = hg * np.arange(np.ceil(cfg.policy[0] / hg), np.floor(cfg.policy[1] / hg) + 1)
        ucell = np.array([U(q - hg / 2) for q in cells])   # rule Q >= q treats Q* >= q - h/2
        best = int(np.argmax(ucell))
        out.update({"grid_cells": cells.tolist(), "U_cells": ucell.tolist(),
                    "best_cell": float(cells[best]),
                    "best_set": [float(cells[best] - hg), float(cells[best])]})
    return out


def s5_limit(cfg: Cfg, n_mc=1_000_000, chunks=16, n_interior=(12, 20), seed=7) -> dict:
    """Large-sample limit of the estimator: argmax of the pseudo-utility built from the
    population projection of its outcome fit (arm-wise splines in eta on J and arm-wise
    X slopes; true gamma), by Monte Carlo normal equations on n_mc * chunks draws of the
    conditional mean of Y."""
    pdf, _ = _s5_law(cfg)
    out = {}
    for K in n_interior:
        info = _basis_params(int(K), cfg.J)
        AtA = Atb = None
        for c in range(chunks):
            rng = np.random.default_rng([seed, c])
            X = rng.standard_normal((n_mc, 3))
            eta = rng.standard_normal(n_mc)
            if cfg.law == "su":
                eta = su_eta(eta)
            T = X @ GAMMA_X
            D = (T + eta > cfg.phi0).astype(float)
            keep = (eta >= cfg.J[0]) & (eta <= cfg.J[1])
            X, eta, T, D = X[keep], eta[keep], T[keep], D[keep]
            ym = (B_CURVE(eta) + X @ np.asarray(BASE.beta1)
                  + D * (cfg.cost + cfg.lam * (A_CURVE(eta) - cfg.cost)
                         + cfg.lam * (X @ np.asarray(BASE.beta2)) + cfg.kappa * T * eta))
            Zm, nb = _design_matrix(eta, X, D, info)
            AtA = Zm.T @ Zm if AtA is None else AtA + Zm.T @ Zm
            Atb = Zm.T @ ym if Atb is None else Atb + Zm.T @ ym
        coef = np.linalg.solve(AtA, Atb)
        w0, d0 = coef[:nb], coef[nb:nb + 3]
        w1, d1 = coef[nb + 3:2 * nb + 3], coef[2 * nb + 3:]
        spl = BSpline(info["t"], w1 - w0, info["degree"], extrapolate=False)
        a_lim = lambda e, s=spl: s(np.clip(e, info["lo"], info["hi"]))
        U, Z = s5_curves(cfg, alpha=a_lim, slope=float((d1 - d0) @ GAMMA_X))
        out[f"K{K}"] = {"limit": _argmax_by_Z(U, Z, cfg.policy)[0],
                        "beta2_lim": (d1 - d0).tolist()}
    Ut, _ = s5_curves(cfg)
    pop = s5_population(cfg)
    lim = out[f"K{max(n_interior)}"]["limit"]
    gain = Ut(pop["phi_star"]) - Ut(cfg.phi0)
    out.update({"target": pop["phi_star"], "limit": lim,
                "gain_lost_share": (Ut(pop["phi_star"]) - Ut(lim)) / gain})
    return out


# ------------------------------------------------------------ population: Appendix 7 family

def het_population(cfg: Cfg) -> dict:
    xg = XGrid(cfg.A, cfg.B, cfg.k, m=40001, lim=9.0)
    xb = HET_BETA2[0] * xg.X1 + HET_BETA2[1] * xg.X2 - cfg.cost
    out = {"key": cfg.key, "family": cfg.family, "note": cfg.note, "J": list(cfg.J),
           "policy": list(cfg.policy), "V_rotated": None}
    if cfg.latent == "zeta":
        # Window on zeta: eps quantiles of R = (phi0 - T - m)/s.
        x1 = np.linspace(-10, 10, 400_001)
        p1 = norm.pdf(x1)
        p1 /= p1.sum()
        R = np.concatenate([(cfg.phi0 - x1 - 0.5 * x2 - m_tilde(x1, cfg.A, cfg.B))
                            / het_scale(x1, x2, cfg.k) for x2 in (-0.5, 0.5)])
        o = np.argsort(R)
        cw = np.cumsum(np.concatenate([p1, p1])[o] * 0.5)
        l0, u0 = (float(np.interp(p, cw, R[o])) for p in (cfg.eps, 1.0 - cfg.eps))
        ez = np.linspace(-12, 12, 480_001)
        af = het_alpha(ez) * norm.pdf(ez)
        Acum = np.concatenate([[0.0], np.cumsum(0.5 * (af[1:] + af[:-1]) * (ez[1] - ez[0]))])

        def U(phi):
            lo = np.clip((phi - xg.T - xg.m) / xg.s, l0, u0)
            val = (np.interp(u0, ez, Acum) - np.interp(lo, ez, Acum)
                   + xb * (norm.cdf(u0) - norm.cdf(lo)))
            return float(np.sum(xg.w * val))
        out["window"] = [l0, u0]
        out["window_on"] = "zeta"
    else:
        z = het_T_quantile(1.0 - cfg.eps), het_T_quantile(cfg.eps)
        overlap = (cfg.phi0 - z[0], cfg.phi0 - z[1])
        cap = (het_eta_quantile(WINDOW_TAIL, cfg.A, cfg.B, cfg.k),
               het_eta_quantile(1.0 - WINDOW_TAIL, cfg.A, cfg.B, cfg.k))
        out["window_uncapped"] = list(overlap)

        def make_U(l, u, me=2001, m1=4001, lim=9.0):
            # U(phi) = int_l^u E_X[(alpha(e) + X'beta2 - c) f_{eta|X}(e) 1{T > phi - e}] de.
            # For each e the X1 integral runs from the threshold phi - e - X2/2 upward; it is
            # read off a cumulative integral along a fine X1 grid (exact limit, linear
            # interpolation), and the e integral is a trapezoid on a fine grid.
            e = np.linspace(l, u, me)
            x1 = np.linspace(-lim, lim, m1)
            dx = x1[1] - x1[0]
            p1 = norm.pdf(x1)
            al = het_alpha(e)
            parts = []
            for x2 in (-0.5, 0.5):
                sx = het_scale(x1, x2, cfg.k)
                f = (norm.pdf((e[:, None] - m_tilde(x1, cfg.A, cfg.B)[None, :]) / sx[None, :])
                     / sx[None, :] * p1[None, :])
                xbx = HET_BETA2[0] * x1 + HET_BETA2[1] * x2 - cfg.cost
                v = al[:, None] * f + xbx[None, :] * f
                seg = 0.5 * (v[:, 1:] + v[:, :-1]) * dx
                cum = np.concatenate([np.cumsum(seg[:, ::-1], axis=1)[:, ::-1],
                                      np.zeros((me, 1))], axis=1)   # int_{x1_j}^{lim}
                parts.append((x2, cum))
            rows = np.arange(me)

            def U(phi):
                tot = np.zeros(me)
                for x2, cum in parts:
                    t = phi - e - 0.5 * x2
                    pos = np.clip((t + lim) / dx, 0.0, m1 - 1.0)
                    i = np.minimum(np.floor(pos).astype(int), m1 - 2)
                    fr = pos - i
                    tot += 0.5 * (cum[rows, i] * (1 - fr) + cum[rows, i + 1] * fr)
                return float(np.sum(0.5 * (tot[1:] + tot[:-1])) * (e[1] - e[0]))
            return U
        Uu = make_U(*overlap)
        phi_u = _argmax_numeric(Uu, cfg.policy)
        out["phi_star_uncapped"] = phi_u
        l0, u0 = max(overlap[0], cap[0]), min(overlap[1], cap[1])
        U = make_U(l0, u0)
        out["window"] = [l0, u0]
        out["window_on"] = "eta"
    phi = _argmax_numeric(U, cfg.policy)
    out.update({"phi_star": phi, "interior": bool(cfg.policy[0] + 1e-6 < phi
                                                  < cfg.policy[1] - 1e-6),
                "U_gain_over_phi0": U(phi) - U(cfg.phi0)})
    return out


def _argmax_numeric(U, policy, m=241) -> float:
    grid = np.linspace(policy[0], policy[1], m)
    v = np.array([U(x) for x in grid])
    k = int(np.argmax(v))
    res = optimize.minimize_scalar(lambda x: -U(x), bounds=(grid[max(k - 1, 0)],
                                                           grid[min(k + 1, m - 1)]),
                                   method="bounded", options={"xatol": 1e-10})
    return float(res.x)


@lru_cache(maxsize=None)
def population(key: str) -> dict:
    cfg = CFGS[key]
    return s5_population(cfg) if cfg.family == "s5" else het_population(cfg)


# ------------------------------------------------------------ summaries

def _summarize(rows, n, pop, boot_key="boot") -> dict:
    rs = [dict(r, boot=r[boot_key], failures=r.get("gfailures" if boot_key == "gboot"
                                                   else "failures", 0)) for r in rows]
    p = dict(pop, V_rotated=1.0, V_fixed=1.0, V_oracle=1.0)
    cell = summarize(rs, n, p)
    for arm in cell["arms"].values():
        arm.pop("coverage_analytic", None)
        arm.pop("analytic_n_var", None)
    return cell


def summarize_cell(key: str, rows, n: int) -> dict:
    pop = population(key)
    cell = _summarize(rows, n, pop)
    cell["edge"] = {
        "rotation_thresholds_at_edge": int(sum(r["edge_rotations"] for r in rows)),
        "pooled_at_edge": int(sum(r["edge_pooled"] for r in rows)),
        "boot_draws_at_edge": int(sum(r["boot_draws_at_edge"] for r in rows)),
        "boot_draws_total": int(sum(len(r["boot"]["avg"]) for r in rows))}
    if rows and "gboot" in rows[0]:
        cell["group_bootstrap"] = _summarize(rows, n, pop, "gboot")
    if "best_set" in pop:
        lo_b, hi_b = pop["best_set"]
        cov = []
        for r in rows:
            b = np.asarray(r["boot"]["avg"])
            q_lo, q_hi = np.quantile(b, [0.025, 0.975])
            cov.append(q_hi > lo_b and q_lo <= hi_b)
        cell["coverage_best_cell_percentile"] = float(np.mean(cov))
        cells = np.asarray(pop["grid_cells"])
        ucell = np.asarray(pop["U_cells"])
        chosen = [ucell[min(int(np.searchsorted(cells, r["est"]["avg"] - 1e-12)),
                            len(cells) - 1)] for r in rows]
        gain = max(ucell) - ucell[int(np.argmin(np.abs(cells - CFGS[key].phi0)))]
        cell["share_best_cell"] = float(np.mean(np.isclose(chosen, max(ucell))))
        cell["mean_regret_share"] = float(np.mean((max(ucell) - np.asarray(chosen)) / gain))
    return cell


# ------------------------------------------------------------ driver

def data_seed(seed: int, n: int, rep: int) -> int:
    return seed + 104_729 * n + rep


def run_cell(key, n, reps, n_boot, seed, workers, out: Path, rep_start=0) -> dict:
    path = out / f"{key}_n{n}.jsonl"
    rows = _load_rows(path)
    done = {r["rep"] for r in rows}
    tasks = [(key, n, r, data_seed(seed, n, r), n_boot)
             for r in range(rep_start, reps) if r not in done]
    if tasks:
        print(f"[{key}] n={n}: {len(done)} done, running {len(tasks)}", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers) as pool, path.open("a") as fh:
            futures = [pool.submit(replicate, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if i % 10 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} reps, {time.perf_counter() - t0:.0f}s",
                          flush=True)
    rows = sorted((r for r in rows if rep_start <= r["rep"] < reps), key=lambda r: r["rep"])
    cell = summarize_cell(key, rows, n)
    cell["rep_range"] = [rep_start, reps]
    return cell


def report(result: dict) -> str:
    pop = result["population"]
    lines = [f"Design `{result['design']}` ({pop['note']}): phi* = {pop['phi_star']:.6f}", "",
             "| n | bootstrap | sqrt(n) bias (s.e.) | n Var | mean n Var* | percentile "
             "| normal | edge (est / boot) |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for n, cell in sorted(result["cells"].items(), key=lambda kv: int(kv[0])):
        for label, c in (("units", cell), ("groups", cell.get("group_bootstrap"))):
            if c is None:
                continue
            a = c["arms"]["avg"]
            e = cell["edge"]
            lines.append(f"| {int(n):,} | {label} | {a['sqrt_n_bias']:+.2f} "
                         f"({a['sqrt_n_bias_se']:.2f}) | {a['n_var']:.2f} "
                         f"| {a['mean_n_boot_var']:.2f} | {a['coverage_percentile']:.3f} "
                         f"| {a['coverage_normal']:.3f} | {e['rotation_thresholds_at_edge']} / "
                         f"{e['boot_draws_at_edge']} |")
        if "coverage_best_cell_percentile" in cell:
            lines.append(f"| {int(n):,} | best cell | covered "
                         f"{cell['coverage_best_cell_percentile']:.3f}; picked "
                         f"{cell['share_best_cell']:.3f}; mean regret share "
                         f"{cell['mean_regret_share']:.4f} | | | | | |")
    return "\n".join(lines)


def check_table2(path: Path, k: int = 3) -> None:
    """The base design reproduces Table 2's point estimates on the same data and split."""
    for n in (4000, 16000):
        rows = _load_rows(path / f"displayed_n{n}.jsonl")[:k]
        for r in rows:
            s, _ = generate(CFGS["base"], n, r["seed"])
            est, _ = estimate(s, CFGS["base"], np.array_split(
                np.random.default_rng([r["seed"], 1]).permutation(n), 4))
            print(f"n={n} seed {r['seed']}: avg {est['avg']:.12f} vs {r['est']['avg']:.12f} "
                  f"(diff {est['avg'] - r['est']['avg']:.1e}); pooled diff "
                  f"{est['pooled'] - r['est']['pooled']:.1e}", flush=True)


def calibrate() -> None:
    """Recompute the constants fixed above (Johnson SU law, design c, tau)."""
    m, v, s, kx = johnsonsu.stats(SU_A, SU_B, moments="mvsk")
    z = np.random.default_rng(1).standard_normal(20_000_000)
    e = su_eta(z)
    print(f"SU: closed-form mean {_MU:.6f} sd {_SD:.6f} (scipy {float(m):.6f}, "
          f"{float(np.sqrt(v)):.6f}); skew {float(s):.3f}, excess kurtosis {float(kx):.2f}; "
          f"MC mean {e.mean():.4f} sd {e.std():.4f} skew {np.mean(e ** 3):.3f} "
          f"kurt {np.mean(e ** 4):.2f}")
    xg = XGrid(C_A, C_B, C_K, m=40001, lim=9.0)
    E2 = np.sum(xg.w * (xg.m ** 2 + xg.s ** 2))
    c2 = xg.m ** 2 + xg.s ** 2
    e4 = xg.m ** 4 + 6 * xg.m ** 2 * xg.s ** 2 + 3 * xg.s ** 4
    print(f"design c: R2(eta) {np.sum(xg.w * xg.m ** 2) / E2:.4f}, R2(eta^2) "
          f"{(np.sum(xg.w * c2 ** 2) - E2 ** 2) / (np.sum(xg.w * e4) - E2 ** 2):.4f}, "
          f"orthogonality E m {np.sum(xg.w * xg.m):.1e}, E m X1 {np.sum(xg.w * xg.m * xg.X1):.1e}")
    for k_, A_, B_ in ((K148, 0.0, 1.0), (C_K, C_A, C_B)):
        rng = np.random.default_rng(3)
        x1 = rng.standard_normal(4_000_000)
        x2 = rng.integers(0, 2, len(x1)) - 0.5
        eta = m_tilde(x1, A_, B_) + het_scale(x1, x2, k_) * rng.standard_normal(len(x1))
        dec = np.searchsorted(np.quantile(x1, np.linspace(0.1, 0.9, 9)), x1)
        sd = np.array([eta[dec == d].std() for d in range(10)])
        print(f"k={k_}, A={A_}: spread factor across X1 deciles {sd.max() / sd.min():.3f}; "
              f"first-stage R^2 {1 - eta.var() / (x1 + 0.5 * x2 + eta).var():.3f}")
    base = s5_population(CFGS["base"])
    dphi = []
    for d in (-0.01, 0.01):
        U, Z = s5_curves(CFGS["base"], cost=CFGS["base"].cost - d)
        dphi.append(_argmax_by_Z(U, Z, CFGS["base"].policy)[0])
    slope = (dphi[1] - dphi[0]) / 0.02
    print(f"base phi* {base['phi_star']:.6f}; d phi*/d w = {slope:.4f}; tau for equal "
          f"variance parts at 64,000 with n Var 28.3: "
          f"{np.sqrt(28.3 / 64000 * 31) / abs(slope):.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", choices=sorted(CFGS))
    ap.add_argument("--n", type=int, nargs="+", default=[16000, 64000, 256000])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--population", action="store_true")
    ap.add_argument("--limits", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--check-table2", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.calibrate:
        calibrate()
        return
    if args.check_table2 is not None:
        check_table2(args.check_table2)
        return
    if args.population:
        out = {key: population(key) for key in CFGS}
        for key, p in out.items():
            print(key, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in p.items()
                                   if k not in ("grid_cells", "U_cells")}), flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "population.json").write_text(json.dumps(out, indent=1) + "\n")
        return
    if args.limits:
        res = {key: s5_limit(CFGS[key]) for key in ("base", "interact_k10", "interact_k20")}
        print(json.dumps(res, indent=1))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "limits.json").write_text(json.dumps(res, indent=1) + "\n")
        return
    if args.design is None:
        raise SystemExit("--design is required")
    summary_path = args.out / f"{args.design}_summary.json"
    if args.report:
        print(report(json.loads(summary_path.read_text())))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    pop = population(args.design)
    print(f"[{args.design}] phi* = {pop['phi_star']:.6f}", flush=True)
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result.update({"design": args.design, "config": dataclasses.asdict(CFGS[args.design]),
                   "population": pop,
                   "settings": {"boot": args.boot, "seed": args.seed, "scheme": "resplit"}})
    result.setdefault("cells", {})
    for n in args.n:
        cell = run_cell(args.design, n, args.reps, args.boot, args.seed, args.workers,
                        args.out, args.rep_start)
        result["cells"][str(n)] = cell
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
    print(report(result), flush=True)
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
