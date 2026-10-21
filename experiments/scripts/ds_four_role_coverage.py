"""Bootstrap coverage of the rotated four-role estimator, with an oracle-nuisance arm.

Closes the simulation gaps in the todo box of ``differing_slopes.tex`` Section 5.

Designs (``--design``):

* ``displayed``: the Section 5.1 design (``ds_local_efficiency.DESIGNS["paper_baseline"]``),
  X ~ N(0, I_3), gamma = (0, 1, 1, 1)/sqrt3, alpha = 2 + eta, b = eta^2/2,
  sigma = 0.5, c = 2.25, eps = 0.1; target 0.545103.
* ``nonlinear``: the Section 5.3 design, T = X_1, X ~ N(0, I_2),
  alpha = 0.35 + 0.9 eta + 0.35 (eta^2 - 1), b = 0.2 + 0.6 eta + 0.5 (eta^2 - 1),
  beta1 = (0.3, -0.2), beta2 = (0.8, 0.25), sigma = 0.5, c = 0.25, eps = 0.1;
  target 0.152143.

Both use the paper's windows J = [-1.75, 1.75], T = [-3, 3] and Phi = [-1.5, 1.5],
for which (DS1)(i) holds.

Estimator (``ds_local_efficiency_sim.py``): the sample is split at random into four
blocks; rotation k gives the roles (first stage, outcome, index, evaluation) to
blocks k, k+1, k+2, k+3 (mod 4).  OLS first stage; arm-wise stacked cubic-spline
outcome fit on J; spline g and rho_X on the index fold; empirical-quantile trim
endpoints; the hard-trimmed evaluation average, normalized by the evaluation fold
size; argmax over Phi on a 61-point grid refined by a bounded scalar search.
Recorded per data set:

* ``avg``: the average of the four rotated thresholds (the paper's estimator);
* ``pooled``: the maximizer of the average of the four rotated criteria;
* ``oracle_avg``, ``oracle_pooled``: the same with every nuisance at its true
  value (gamma, alpha, beta2, g, rho_X and [l0, u0]), on the same folds, so only
  the evaluation averages are estimated;
* ``linear_avg``, ``linear_pooled`` (``nonlinear`` only, supplementary): the
  spline outcome fit replaced by a fit linear in eta on each arm.

Bootstrap: full re-estimation.  Each draw resamples n observations with
replacement, splits the resample into four new random blocks and re-runs every
fit.  For each arm we form 95% percentile and normal (estimate +- 1.96 bootstrap
s.d.) intervals.  As a cross-check only, we also report the coverage of
estimate +- 1.96 sqrt(V / n) with the analytic first-order variance V
(sigma^2/H^2 = 31.09 in the displayed design; Var psi_U / H^2 for the oracle).

Diagnostic (``--scheme within``): keep the data set's own split and resample
within each of the four blocks, so no observation is copied into two roles.

Rows are appended to ``<out>/<design>_n<n>.jsonl`` as replications finish, and a
rerun resumes from them.  ``--report`` prints the markdown tables of a finished
run.

Run:
    python -m experiments.scripts.ds_four_role_coverage --design displayed \
        --n 4000 16000 --reps 500 --boot 199 --workers 7
    python -m experiments.scripts.ds_four_role_coverage --design displayed --report
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
from typing import Callable, Dict, List

import numpy as np
from numpy.polynomial import Polynomial
from scipy import optimize, stats
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from experiments.methods.trimmed_quadrature import trimmed_normal_mean
from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_local_efficiency import DESIGNS, Design, Population
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, Sample, _ls, _xt, generate, ols_first_stage, outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_four_role_coverage"
Z95 = float(stats.norm.ppf(0.975))

COVERAGE_DESIGNS: Dict[str, Design] = {
    "displayed": DESIGNS["paper_baseline"],
    "nonlinear": Design(
        name="nonlinear_ab",
        sigma_x=(1.0, 1.0), gamma0=0.0, gamma_x=(1.0, 0.0), sigma_eta=1.0,
        # alpha = 0.35 + 0.9 e + 0.35 (e^2 - 1), b = 0.2 + 0.6 e + 0.5 (e^2 - 1)
        a_coef=(0.0, 0.9, 0.35), b_coef=(-0.3, 0.6, 0.5),
        beta1=(0.3, -0.2), beta2=(0.8, 0.25),
        sigma_arm=(0.5, 0.5), cost=0.25,
        note="differing_slopes.tex Section 5.3 nonlinear alpha, b design",
    ),
}
LINEAR_ARM = {"displayed": False, "nonlinear": True}


# ------------------------------------------------------------ truth

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class TrueTails:
    """Fixed-support survival and weighted tail of a Gaussian index T = gamma0 + gamma_x'X."""

    def __init__(self, design: Design):
        Sigma = np.diag(np.asarray(design.sigma_x, dtype=float))
        gx = np.asarray(design.gamma_x, dtype=float)
        self.kappa = Sigma @ gx                       # Cov(X, T)
        self.mu = float(design.gamma0)
        self.sT = float(np.sqrt(gx @ Sigma @ gx))
        self.lo, self.hi = design.T_support
        self.z_hi = (self.hi - self.mu) / self.sT

    def _z(self, t):
        return (np.clip(t, self.lo, self.hi) - self.mu) / self.sT

    def survival(self, t):
        return ndtr(self.z_hi) - ndtr(self._z(t))

    def weighted_tail(self, t, b2):
        """b2' E[X 1{t < T <= hi}] = (b2'kappa / sT) {phi(z_t) - phi(z_hi)}."""
        z = self._z(t)
        s = float(np.asarray(b2) @ self.kappa) / self.sT
        return s * _INV_SQRT_2PI * (np.exp(-0.5 * z ** 2) - np.exp(-0.5 * self.z_hi ** 2))


def target_by_quadrature(design: Design, pop: Population) -> float:
    """phi*_eps as the zero of Z(phi) = -U_eps'(phi), by adaptive quadrature on [l0, u0]."""
    a = Polynomial(design.a_coef)
    beta2 = np.asarray(design.beta2, dtype=float)

    def Z(phi):
        return trimmed_normal_mean(
            lambda e: (a(e) - design.cost) * pop.g(phi - e) + pop.rho(phi - e) @ beta2,
            pop.l0, pop.u0, sd=design.sigma_eta)

    return float(optimize.brentq(Z, pop.phi - 0.05, pop.phi + 0.05, xtol=1e-13))


def population_summary(design: Design) -> dict:
    pop = Population(design)
    dec = pop.decomposition()
    phi_quad = target_by_quadrature(design, pop)
    if abs(phi_quad - dec["phi_star"]) > 1e-8:
        raise RuntimeError(f"targets disagree: {phi_quad} vs {dec['phi_star']}")
    H2 = dec["H"] ** 2
    return {
        "phi_star": phi_quad, "phi_star_gl": dec["phi_star"], "H": dec["H"],
        "l0": dec["l0"], "u0": dec["u0"], "variances": dec["variances"],
        "V_rotated": dec["V_rotated_ols"], "V_fixed": dec["V_fixed_ols"],
        "V_oracle": dec["variances"]["psi_U"] / H2,
    }


# ------------------------------------------------------------ linear outcome fit

@dataclass(frozen=True)
class LinearOutcomeFit:
    """Arm-wise fit linear in eta: c_d(e) = w_d0 + w_d1 e, slopes d_d."""
    w0: np.ndarray
    w1: np.ndarray
    d0: np.ndarray
    d1: np.ndarray

    def a(self, e):
        w = self.w1 - self.w0
        return w[0] + w[1] * np.asarray(e)

    @property
    def beta2(self):
        return self.d1 - self.d0


def outcome_fit_linear(s: Sample, idx: np.ndarray, eta_hat: np.ndarray, J) -> LinearOutcomeFit:
    e = eta_hat[idx]
    fit = idx[(e >= J[0]) & (e <= J[1])]
    e, X, D, Y = eta_hat[fit], s.X[fit], s.D[fit][:, None], s.Y[fit]
    B = np.column_stack((np.ones(len(e)), e))
    Z = np.column_stack(((1 - D) * B, (1 - D) * X, D * B, D * X))
    coef = _ls(Z, Y)
    dim = X.shape[1]
    return LinearOutcomeFit(w0=coef[:2], d0=coef[2:2 + dim],
                            w1=coef[2 + dim:4 + dim], d1=coef[4 + dim:])


# ------------------------------------------------------------ criteria and argmax

def _criterion(e: np.ndarray, amc: np.ndarray, b2: np.ndarray, tails, n_u: int) -> Callable:
    """U-hat(phi) = sum over kept evaluation units of (a - c) Gbar + b2' H_X, over n_u."""
    def value(phi):
        t = phi - e
        return float(np.sum(amc * tails.survival(t) + tails.weighted_tail(t, b2)) / n_u)
    return value


class SplineTailPieces:
    """Gbar-hat and H_X-hat of one ``WeightedTails`` fit at phi - e, in one spline call.

    The survival and the weighted tails are integrals of splines on one knot
    vector, so a single antiderivative spline with columns (omega_g, omega_x)
    gives both.  ``value(phi, amc, b2)`` equals ``_criterion(e, amc, b2,
    tails, n_u)(phi)``; several outcome fits can share the pieces.
    """

    def __init__(self, tails, e: np.ndarray, n_u: int):
        coef = np.column_stack((tails.omega_g, tails.omega_x))
        self.A = BSpline(np.asarray(tails.info["t"], dtype=float), coef,
                         int(tails.info["degree"]), extrapolate=False).antiderivative()
        self.lo, self.hi = tails.support
        self.top = self.A(self.hi)
        self.e, self.n_u = e, n_u

    def sums(self, phi: float, amc_list) -> tuple:
        """(sum_i amc_i Gbar(phi - e_i) for each amc, sum_i H_X(phi - e_i))."""
        tail = self.top - self.A(np.clip(phi - self.e, self.lo, self.hi))
        return [float(amc @ tail[:, 0]) for amc in amc_list], tail[:, 1:].sum(axis=0)

    def value(self, phi: float, amc: np.ndarray, b2: np.ndarray) -> float:
        (sg,), sh = self.sums(phi, [amc])
        return (sg + float(sh @ b2)) / self.n_u


def _refine(value: Callable, grid: np.ndarray, vals: np.ndarray) -> float:
    k = int(np.argmax(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: -value(x), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x)


def _avg_and_pooled(criteria: List[Callable], grid: np.ndarray, vals=None) -> tuple:
    """Average of the rotated argmaxes, and the argmax of the averaged criterion."""
    if vals is None:
        vals = np.array([[v(x) for x in grid] for v in criteria])
    phis = [_refine(v, grid, row) for v, row in zip(criteria, vals)]
    pooled = _refine(lambda x: sum(v(x) for v in criteria) / len(criteria), grid,
                     vals.mean(axis=0))
    return float(np.mean(phis)), pooled, phis


@dataclass(frozen=True)
class Truth:
    gamma: np.ndarray
    a: Polynomial
    beta2: np.ndarray
    tails: TrueTails
    window: tuple


def make_truth(design: Design, pop: dict) -> Truth:
    return Truth(gamma=np.concatenate(([design.gamma0], np.asarray(design.gamma_x))),
                 a=Polynomial(design.a_coef), beta2=np.asarray(design.beta2, dtype=float),
                 tails=TrueTails(design), window=(pop["l0"], pop["u0"]))


def estimate(s: Sample, design: Design, truth: Truth, rng, linear_arm: bool,
             blocks=None) -> dict:
    """All arms on one data set and one four-block split (random unless ``blocks`` is given)."""
    n = len(s.Q)
    if blocks is None:
        blocks = np.array_split(rng.permutation(n), 4)
    grid = np.linspace(design.policy[0], design.policy[1], 61)
    arms = ("spline", "linear") if linear_arm else ("spline",)
    crit = {arm: [] for arm in arms + ("oracle",)}
    vals = {arm: [] for arm in arms}
    eta_true = s.Q - _xt(s.X) @ truth.gamma
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
        fits = {"spline": outcome_fit(s, folds["outcome"], eta_hat, design.J)}
        if linear_arm:
            fits["linear"] = outcome_fit_linear(s, folds["outcome"], eta_hat, design.J)
        pieces = SplineTailPieces(tails, e, n_u)
        amc = {arm: fits[arm].a(e) - design.cost for arm in arms}
        b2 = {arm: np.asarray(fits[arm].beta2) for arm in arms}
        grid_vals = {arm: [] for arm in arms}
        for x in grid:
            sg, sh = pieces.sums(x, [amc[arm] for arm in arms])
            for arm, g_sum in zip(arms, sg):
                grid_vals[arm].append((g_sum + float(sh @ b2[arm])) / n_u)
        for arm in arms:
            crit[arm].append(lambda x, p=pieces, a=amc[arm], b=b2[arm]: p.value(x, a, b))
            vals[arm].append(grid_vals[arm])
        eo = eta_true[folds["utility"]]
        eo = eo[(eo >= truth.window[0]) & (eo <= truth.window[1])]
        crit["oracle"].append(_criterion(eo, truth.a(eo) - design.cost, truth.beta2,
                                         truth.tails, n_u))
    out = {}
    out["avg"], out["pooled"], rot = _avg_and_pooled(crit["spline"], grid,
                                                     np.array(vals["spline"]))
    out["fixed"] = rot[0]
    out["oracle_avg"], out["oracle_pooled"], _ = _avg_and_pooled(crit["oracle"], grid)
    if linear_arm:
        out["linear_avg"], out["linear_pooled"], _ = _avg_and_pooled(
            crit["linear"], grid, np.array(vals["linear"]))
    return out


def _resample(s: Sample, idx: np.ndarray) -> Sample:
    return Sample(X=s.X[idx], Q=s.Q[idx], D=s.D[idx], Y=s.Y[idx])


def replicate(task) -> dict:
    design_key, n, rep, seed, n_boot, pop = task[:6]
    scheme = task[6] if len(task) > 6 else "resplit"
    linear_arm = task[7] if len(task) > 7 and task[7] is not None else LINEAR_ARM[design_key]
    old = task[8] if len(task) > 8 else None
    design = COVERAGE_DESIGNS[design_key]
    truth = make_truth(design, pop)
    t0 = time.perf_counter()
    s = generate(design, n, seed)
    rng0 = np.random.default_rng([seed, 1])
    brng = np.random.default_rng([seed, 2])
    if old is not None:
        # Extend a finished run: reuse its estimate and its first draws, which used
        # the same seeds, and continue the bootstrap stream where it stopped.
        if scheme != "resplit" or old["seed"] != seed:
            raise ValueError("extension needs the resplit scheme and the same seed")
        est = {k: v for k, v in old["est"].items() if linear_arm or not k.startswith("linear")}
        boot = {k: list(old["boot"][k]) for k in est}
        failures = int(old["failures"])
        start = len(old["boot"]["avg"]) + failures
        for _ in range(start):
            brng.integers(0, n, n)
    else:
        if scheme == "within":
            blocks = np.array_split(rng0.permutation(n), 4)
            sizes = np.cumsum([len(blk) for blk in blocks])[:-1]
            est = estimate(s, design, truth, None, linear_arm, blocks=blocks)
        else:
            est = estimate(s, design, truth, rng0, linear_arm)
        boot = {key: [] for key in est}
        failures = 0
        start = 0
    for b in range(start, n_boot):
        try:
            if scheme == "within":
                idx = np.concatenate([blk[brng.integers(0, len(blk), len(blk))] for blk in blocks])
                draw = estimate(_resample(s, idx), design, truth, None, linear_arm,
                                blocks=np.split(np.arange(n), sizes))
            else:
                idx = brng.integers(0, n, n)
                draw = estimate(_resample(s, idx), design, truth,
                                np.random.default_rng([seed, 3, b]), linear_arm)
        except (np.linalg.LinAlgError, ValueError):
            failures += 1
            continue
        for key, val in draw.items():
            boot[key].append(val)
    seconds = time.perf_counter() - t0 + (float(old["seconds"]) if old is not None else 0.0)
    return {"rep": rep, "seed": seed, "est": est, "boot": boot, "failures": failures,
            "seconds": seconds}


# ------------------------------------------------------------ summaries

ARMS_FOR_ANALYTIC = {"avg": "V_rotated", "pooled": "V_rotated", "fixed": "V_fixed",
                     "oracle_avg": "V_oracle", "oracle_pooled": "V_oracle"}


def summarize(rows: List[dict], n: int, pop: dict) -> dict:
    phi = pop["phi_star"]
    out = {"n": n, "reps": len(rows), "boot_draws_per_rep": int(np.median(
        [len(r["boot"]["avg"]) for r in rows])), "boot_failures": int(sum(r["failures"] for r in rows)),
        "arms": {}}
    for key in rows[0]["est"]:
        est = np.array([r["est"][key] for r in rows])
        err = est - phi
        cell = {"bias": float(err.mean()), "sqrt_n_bias": float(np.sqrt(n) * err.mean()),
                "sqrt_n_bias_se": float(np.sqrt(n) * err.std(ddof=1) / np.sqrt(len(err))),
                "n_var": float(n * err.var(ddof=1)), "n_mse": float(n * np.mean(err ** 2))}
        if key == "fixed":
            out["arms"][key] = cell
            continue
        B = [np.asarray(r["boot"][key]) for r in rows]
        sd = np.array([b.std(ddof=1) for b in B])
        q_lo = np.array([np.quantile(b, 0.025) for b in B])
        q_hi = np.array([np.quantile(b, 0.975) for b in B])
        pct = (q_lo <= phi) & (phi <= q_hi)
        nrm = np.abs(est - phi) <= Z95 * sd
        cell.update({
            "mean_n_boot_var": float(np.mean(n * sd ** 2)),
            "median_n_boot_var": float(np.median(n * sd ** 2)),
            "boot_var_ratio": float(np.mean(sd ** 2) / err.var(ddof=1)),
            "sqrt_n_boot_bias": float(np.sqrt(n) * np.mean([b.mean() for b in B] - est)),
            "coverage_percentile": float(pct.mean()),
            "coverage_normal": float(nrm.mean()),
            "miss_low_percentile": float(np.mean(q_hi < phi)),
            "miss_high_percentile": float(np.mean(q_lo > phi)),
            "width_percentile": float(np.mean(q_hi - q_lo)),
            "width_normal": float(np.mean(2 * Z95 * sd)),
        })
        if key in ARMS_FOR_ANALYTIC:
            V = pop[ARMS_FOR_ANALYTIC[key]]
            cell["coverage_analytic"] = float(np.mean(np.abs(err) <= Z95 * np.sqrt(V / n)))
            cell["analytic_n_var"] = V
        out["arms"][key] = cell
    out["coverage_mc_se"] = float(np.sqrt(0.95 * 0.05 / len(rows)))
    if "avg" in out["arms"] and "pooled" in out["arms"]:
        A = np.array([r["est"]["avg"] for r in rows])
        P = np.array([r["est"]["pooled"] for r in rows])
        C = (4 * P - A) / 3
        out["arms"]["combined_point"] = {
            "sqrt_n_bias": float(np.sqrt(n) * (C - phi).mean()),
            "sqrt_n_bias_se": float(np.sqrt(n) * C.std(ddof=1) / np.sqrt(len(C))),
            "n_var": float(n * C.var(ddof=1))}
    out["mean_seconds_per_rep"] = float(np.mean([r["seconds"] for r in rows]))
    return out


def _load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _suffix(scheme: str) -> str:
    return "" if scheme == "resplit" else f"_{scheme}"


def run_cell(design_key: str, n: int, reps: int, n_boot: int, seed: int, workers: int,
             out: Path, pop: dict, scheme: str = "resplit", linear_arm=None,
             extend_from: Path = None) -> dict:
    path = out / f"{design_key}_n{n}{_suffix(scheme)}.jsonl"
    rows = _load_rows(path)
    done = {r["rep"] for r in rows}
    old = {}
    if extend_from is not None:
        old = {r["rep"]: r for r in _load_rows(extend_from / path.name)}
    tasks = [(design_key, n, r, seed + 104_729 * n + r, n_boot, pop, scheme, linear_arm,
              old.get(r)) for r in range(reps) if r not in done]
    if tasks:
        print(f"[{design_key}] n={n}: {len(done)} done, running {len(tasks)}", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(workers) as pool, path.open("a") as fh:
            futures = [pool.submit(replicate, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if i % 25 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} reps, {time.perf_counter() - t0:.0f}s", flush=True)
    rows = sorted(rows, key=lambda r: r["rep"])[:reps]
    return summarize(rows, n, pop)


LABELS = {"avg": "rotated average", "pooled": "pooled maximizer",
          "oracle_avg": "oracle, average", "oracle_pooled": "oracle, pooled",
          "linear_avg": "linear fit, average", "linear_pooled": "linear fit, pooled"}


def report(result: dict, arms=("avg", "pooled", "oracle_avg", "linear_avg")) -> str:
    """Markdown table of a summary: bias, variance, interval coverage and width."""
    pop = result["population"]
    lines = [f"Design `{result['design']}`: phi* = {pop['phi_star']:.6f}, "
             f"H = {pop['H']:.6f}; first-order n Var {pop['V_rotated']:.2f} (rotated), "
             f"{pop['V_fixed']:.1f} (fixed), {pop['V_oracle']:.4f} (oracle).", "",
             "| n | estimator | sqrt(n) bias (s.e.) | n Var | mean n Var* | percentile "
             "| normal | analytic (check) | percentile width |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, cell in sorted(result["cells"].items(), key=lambda kv: int(kv[0])):
        for arm in arms:
            if arm not in cell["arms"]:
                continue
            c = cell["arms"][arm]
            ana = f"{c['coverage_analytic']:.3f}" if "coverage_analytic" in c else "-"
            lines.append(
                f"| {int(n):,} | {LABELS[arm]} | {c['sqrt_n_bias']:+.2f} ({c['sqrt_n_bias_se']:.2f}) "
                f"| {c['n_var']:.2f} | {c['mean_n_boot_var']:.2f} | {c['coverage_percentile']:.3f} "
                f"| {c['coverage_normal']:.3f} | {ana} | {c['width_percentile']:.4f} |")
    lines.append("")
    lines.append("; ".join(f"n = {int(n):,}: {c['reps']} data sets, {c['boot_draws_per_rep']} draws"
                           for n, c in sorted(result["cells"].items(), key=lambda kv: int(kv[0]))))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", choices=sorted(COVERAGE_DESIGNS), default="displayed")
    ap.add_argument("--n", type=int, nargs="+", default=[4000, 16000])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--boot", type=int, default=199)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scheme", choices=("resplit", "within"), default="resplit")
    ap.add_argument("--report", action="store_true", help="print the tables of a finished run")
    ap.add_argument("--no-linear", action="store_true",
                    help="drop the supplementary linear-in-eta arm (nonlinear design)")
    ap.add_argument("--extend-from", type=Path, default=None,
                    help="directory of a finished run with the same seeds; reuse its estimates "
                         "and draws and add draws up to --boot")
    args = ap.parse_args(argv)
    summary_path = args.out / f"{args.design}_summary{_suffix(args.scheme)}.json"
    if args.report:
        print(report(json.loads(summary_path.read_text())))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    design = COVERAGE_DESIGNS[args.design]
    pop = population_summary(design)
    print(f"[{args.design}] phi* = {pop['phi_star']:.6f}, H = {pop['H']:.6f}, "
          f"V_rotated = {pop['V_rotated']:.3f}, V_oracle = {pop['V_oracle']:.4f}", flush=True)
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result.update({"design": args.design, "design_params": repr(design), "population": pop,
                   "settings": {"boot": args.boot, "seed": args.seed, "scheme": args.scheme,
                                "linear_arm": not args.no_linear,
                                "extended_from": str(args.extend_from) if args.extend_from else None}})
    result.setdefault("cells", {})
    for n in args.n:
        cell = run_cell(args.design, n, args.reps, args.boot, args.seed, args.workers,
                        args.out, pop, args.scheme, False if args.no_linear else None,
                        args.extend_from)
        result["cells"][str(n)] = cell
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(cell, indent=1), flush=True)
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
