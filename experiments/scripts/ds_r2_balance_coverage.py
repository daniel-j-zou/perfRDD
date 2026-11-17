"""Bootstrap coverage across the first-stage R^2 and the treated share (author, 2026-11-17).

The author asked for simulations that match two features of the taxi application
(Section 6): an unbalanced treatment (11% of rides treated) and a first stage that
explains most of the score (R^2 = 0.925).  This script runs the bootstrap check of
Table 2 (``ds_four_role_coverage.py``, 999 full re-estimation draws per data set) on a
grid of designs that vary those two features and keep everything else of the
Section 5 design.

Designs (key ``r2<100 R^2>_<bal|unbal>``, e.g. ``r2_90_unbal``):

* X ~ N(0, I_3), T = s (1, 1, 1)'X / sqrt3 ~ N(0, s^2), eta ~ N(0, 1), Q = T + eta,
  so the first-stage R^2 = Var T / Var Q = s^2 / (1 + s^2); R^2 in
  {0.1, 0.3, 0.5, 0.7, 0.9}, s = sqrt(R^2 / (1 - R^2)).
* alpha, b, beta1, beta2, the noise and eps = 0.1 are those of the Section 5 design.
* ``bal``: deployed threshold phi0 = 0 (half the units treated) and c = 2.25, as in
  Section 5.  ``unbal``: phi0 at the 89th percentile of Q (11% treated, as in the taxi
  data), and the cost c set so that the target treats 16% of units (it lies below
  phi0, as in the taxi application, where $12.87 < $15).
* Windows scale with s, so that at R^2 = 0.5 the ``bal`` design is exactly the
  Section 5 design (J = [-1.75, 1.75], T = [-3, 3], Phi = [-1.5, 1.5]):
  - overlap window [l0, u0] = phi0 -+ s z_{0.9} (Section 2.3);
  - the trim window is capped at the central 99% of the first-stage residuals,
    re-estimated from the evaluation fold in every rotation and bootstrap draw, as in
    the taxi application (``taxi_section6.py``); the population cap is +-z_{0.995}.
    The cap binds in the population for r2_90_bal and the unbal designs with
    R^2 >= 0.5; there the analytic variance is not computed (the capped-window
    variance of Appendix 5 is not implemented here), and the tables compare the
    bootstrap with the Monte Carlo variance only;
  - J = [phi0 - 1.75 s, phi0 + 1.75 s], where both arms have probability at least
    Phi(-1.75) = 0.04, intersected with the central 99.5% of eta (as the taxi J);
  - T = [-3 s, 3 s] (three standard deviations of T);
  - Phi = [-3 s + u + b s, 3 s + l - b s] with b = 3 - z_{0.9} - 1.5 = 0.218, so that the
    argument set {phi - eta} stays inside T with the Section 5 buffer.
* Every design uses the same seeds (seed + 104729 n + rep, seed 20261019 as in
  Table 2), so the data sets are common random numbers across designs, and the
  r2_50_bal cell reproduces Table 2 data set by data set (checked by ``--check``).

Estimator, bootstrap and summaries are those of ``ds_four_role_coverage.py``: four-role
split, OLS first stage, spline outcome fit on J, spline g and rho_X on T, rotated
average and pooled maximizer, and an oracle arm with every nuisance at its true value
on the same folds.  Rows go to ``<out>/<design>_n<n>.jsonl`` as replications finish;
``--rep-start a --reps b`` shards a cell for Slurm arrays.

Run:
    python -m experiments.scripts.ds_r2_balance_coverage --population
    python -m experiments.scripts.ds_r2_balance_coverage --design r2_90_unbal --n 4000 \
        --reps 4 --boot 19 --workers 4 --out experiments/runs/ds_r2bal_test
    python -m experiments.scripts.ds_r2_balance_coverage --design r2_50_bal --check \
        experiments/runs/ds_four_role_coverage_nl_999
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import optimize
from scipy.integrate import quad
from scipy.stats import norm

from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_four_role_coverage import (
    SplineTailPieces, TrueTails, _avg_and_pooled, _criterion, _load_rows, _resample,
    summarize,
)
from experiments.scripts.ds_local_efficiency import DESIGNS, Curve, Design, Population
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, _xt, generate, ols_first_stage, outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_r2_balance_coverage"
BASE = DESIGNS["paper_nonlinear"]
R2_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
SHARE = {"bal": 0.5, "unbal": 0.11}
TARGET_SHARE_UNBAL = 0.16
WINDOW_TAIL = 0.005            # window cap: central 99% of eta-hat on the evaluation fold
J_TAIL = 0.0025                # J inside the central 99.5% of eta
J_HALF = 1.75                  # J = phi0 -+ 1.75 s (Section 5: [-1.75, 1.75])
EDGE_TOL = 1e-3
SEED = 20261019                # Table 2's seed


@dataclasses.dataclass(frozen=True)
class Spec:
    key: str
    r2: float
    balance: str
    s: float
    design: Design
    window: Tuple[float, float]          # population trim window (capped)
    overlap: Tuple[float, float]         # population overlap window (uncapped)
    capped: bool


def design_key(r2: float, balance: str) -> str:
    return f"r2_{int(round(100 * r2)):02d}_{balance}"


def _target_and_H(design: Design, s: float, window, policy) -> Tuple[float, float]:
    """Unique maximizer of U_eps over Phi and H = U_eps'' there, by adaptive quadrature."""
    a = Curve(design.a_coef)
    lam = float(np.asarray(design.beta2) @ np.asarray(design.gamma_x)) / s ** 2  # lambda(t) = lam t
    l, u = window

    def Z(phi):           # -U_eps'(phi)
        f = lambda e: ((float(a(np.array([e]))[0]) - design.cost + lam * (phi - e))
                       * norm.pdf((phi - e) / s) / s * norm.pdf(e))
        return quad(f, l, u, limit=400, epsabs=1e-13, epsrel=1e-12)[0]

    grid = np.linspace(policy[0], policy[1], 241)
    vals = np.array([Z(x) for x in grid])
    ups = [i for i in range(len(grid) - 1) if vals[i] < 0.0 <= vals[i + 1]]
    downs = [i for i in range(len(grid) - 1) if vals[i] >= 0.0 > vals[i + 1]]
    if len(ups) != 1 or downs:
        raise ValueError(f"{design.name}: expected one interior maximizer on Phi "
                         f"(up-crossings {len(ups)}, down-crossings {len(downs)})")
    phi = optimize.brentq(Z, grid[ups[0]], grid[ups[0] + 1], xtol=1e-14)
    h = 1e-5
    H = -(Z(phi + h) - Z(phi - h)) / (2 * h)
    return float(phi), float(H)


def _unbal_cost(base: Design, s: float, window, phi_t: float) -> float:
    """Cost c at which the target is phi_t: E_w[(alpha + lambda - c) g] = 0 at phi_t."""
    a = Curve(base.a_coef)
    gx = s * np.ones(3) / np.sqrt(3.0)
    lam = float(np.asarray(base.beta2) @ gx) / s ** 2
    l, u = window
    w = lambda e: norm.pdf((phi_t - e) / s) / s * norm.pdf(e)
    num = quad(lambda e: (float(a(np.array([e]))[0]) + lam * (phi_t - e)) * w(e), l, u,
               limit=400, epsabs=1e-14, epsrel=1e-13)[0]
    den = quad(w, l, u, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
    return num / den


def make_spec(r2: float, balance: str) -> Spec:
    key = design_key(r2, balance)
    if key == "r2_50_bal":
        d = BASE
        z = float(norm.ppf(1.0 - d.eps))
        return Spec(key, r2, balance, 1.0, d, (-z, z), (-z, z), False)
    s = float(np.sqrt(r2 / (1.0 - r2)))
    sq = float(np.sqrt(1.0 + s * s))
    z = float(norm.ppf(1.0 - BASE.eps))
    phi0 = 0.0 if balance == "bal" else float(norm.ppf(1.0 - SHARE[balance])) * sq
    overlap = (phi0 - s * z, phi0 + s * z)
    qc = float(norm.ppf(1.0 - WINDOW_TAIL))
    qj = float(norm.ppf(1.0 - J_TAIL))
    window = (max(overlap[0], -qc), min(overlap[1], qc))
    capped = window != overlap
    J = (max(phi0 - J_HALF * s, -qj), min(phi0 + J_HALF * s, qj))
    T_support = (-3.0 * s, 3.0 * s)
    b = 3.0 - z - 1.5            # Section 5 buffer: -3 + z + b = -1.5
    policy = (T_support[0] + window[1] + b * s, T_support[1] + window[0] - b * s)
    cost = BASE.cost
    if balance == "unbal":
        phi_t = float(norm.ppf(1.0 - TARGET_SHARE_UNBAL)) * sq
        cost = _unbal_cost(BASE, s, window, phi_t)
    d = dataclasses.replace(
        BASE, name=key, gamma_x=tuple(s * np.ones(3) / np.sqrt(3.0)), cost=float(cost),
        phi0=phi0, J=J, T_support=T_support, policy=policy,
        note=f"R^2 = {r2}, treated share {SHARE[balance]} (ds_r2_balance_coverage)")
    if not (J[0] < window[0] < window[1] < J[1]):
        raise ValueError(f"{key}: window {window} not inside J {J}")
    return Spec(key, r2, balance, s, d, window, overlap, capped)


SPECS: Dict[str, Spec] = {design_key(r2, b): make_spec(r2, b)
                          for b in ("bal", "unbal") for r2 in R2_GRID}


def population(spec: Spec) -> dict:
    d = spec.design
    phi, H = _target_and_H(d, spec.s, spec.window, d.policy)
    sq = float(np.sqrt(1.0 + spec.s ** 2))
    out = {"key": spec.key, "r2": spec.r2, "balance": spec.balance, "s": spec.s,
           "phi0": d.phi0, "cost": d.cost, "phi_star": phi, "H": H,
           "window": list(spec.window), "overlap_window": list(spec.overlap),
           "capped": spec.capped, "J": list(d.J), "T_support": list(d.T_support),
           "policy": list(d.policy),
           "treated_share_phi0": float(norm.sf(d.phi0 / sq)),
           "treated_share_target": float(norm.sf(phi / sq)),
           "window_share": float(norm.cdf(spec.window[1]) - norm.cdf(spec.window[0])),
           "V_rotated": None, "V_fixed": None, "V_oracle": None}
    if not spec.capped:
        dec = Population(d).decomposition()
        if abs(dec["phi_star"] - phi) > 1e-7:
            raise RuntimeError(f"{spec.key}: targets disagree, {dec['phi_star']} vs {phi}")
        H2 = dec["H"] ** 2
        out.update({"H_gl": dec["H"], "variances": dec["variances"],
                    "V_rotated": dec["V_rotated_ols"], "V_fixed": dec["V_fixed_ols"],
                    "V_oracle": dec["variances"]["psi_U"] / H2})
    return out


# ------------------------------------------------------------ estimator

def estimate(s, spec: Spec, rng=None, blocks=None) -> tuple:
    """Rotated average, pooled maximizer, fixed assignment and oracle on one split.

    As ``ds_four_role_coverage.estimate`` (without the linear arm), with the trim window
    capped at the central 99% of eta-hat on the evaluation fold.
    """
    d = spec.design
    n = len(s.Q)
    if blocks is None:
        blocks = np.array_split(rng.permutation(n), 4)
    grid = np.linspace(d.policy[0], d.policy[1], 61)
    gamma = np.concatenate(([d.gamma0], np.asarray(d.gamma_x)))
    beta2 = np.asarray(d.beta2, dtype=float)
    a_true = Curve(d.a_coef)
    true_tails = TrueTails(d)
    eta_true = s.Q - _xt(s.X) @ gamma
    crit, vals, ocrit = [], [], []
    for k in range(4):
        folds = {role: blocks[(j + k) % 4] for j, role in enumerate(ROLES)}
        n_u = len(folds["utility"])
        gamma_hat = ols_first_stage(s, folds["gamma"])
        eta_hat = s.Q - _xt(s.X) @ gamma_hat
        T_idx = _xt(s.X[folds["index"]]) @ gamma_hat
        tails = fit_weighted_tails(T_idx, s.X[folds["index"]], d.T_support)
        l_hat = d.phi0 - float(np.quantile(T_idx, 1.0 - d.eps))
        u_hat = d.phi0 - float(np.quantile(T_idx, d.eps))
        e = eta_hat[folds["utility"]]
        l_hat = max(l_hat, float(np.quantile(e, WINDOW_TAIL)))
        u_hat = min(u_hat, float(np.quantile(e, 1.0 - WINDOW_TAIL)))
        e = e[(e >= l_hat) & (e <= u_hat)]
        fit = outcome_fit(s, folds["outcome"], eta_hat, d.J)
        pieces = SplineTailPieces(tails, e, n_u)
        amc = fit.a(e) - d.cost
        b2 = np.asarray(fit.beta2)
        row = []
        for x in grid:
            (sg,), sh = pieces.sums(x, [amc])
            row.append((sg + float(sh @ b2)) / n_u)
        crit.append(lambda x, p=pieces, a=amc, b=b2: p.value(x, a, b))
        vals.append(row)
        eo = eta_true[folds["utility"]]
        eo = eo[(eo >= spec.window[0]) & (eo <= spec.window[1])]
        ocrit.append(_criterion(eo, a_true(eo) - d.cost, beta2, true_tails, n_u))
    out = {}
    out["avg"], out["pooled"], rot = _avg_and_pooled(crit, grid, np.array(vals))
    out["fixed"] = rot[0]
    out["oracle_avg"], out["oracle_pooled"], _ = _avg_and_pooled(ocrit, grid)
    return out, rot


def at_edge(spec: Spec, x: float) -> bool:
    lo, hi = spec.design.policy
    return min(abs(x - lo), abs(x - hi)) < EDGE_TOL


def replicate(task) -> dict:
    key, n, rep, seed, n_boot = task
    spec = SPECS[key]
    t0 = time.perf_counter()
    s = generate(spec.design, n, seed)
    est, rot = estimate(s, spec, np.random.default_rng([seed, 1]))
    brng = np.random.default_rng([seed, 2])
    boot = {k: [] for k in est}
    failures = edge_draws = 0
    for b in range(n_boot):
        idx = brng.integers(0, n, n)
        try:
            draw, drot = estimate(_resample(s, idx), spec, np.random.default_rng([seed, 3, b]))
        except (np.linalg.LinAlgError, ValueError):
            failures += 1
            continue
        edge_draws += int(any(at_edge(spec, p) for p in drot) or at_edge(spec, draw["pooled"]))
        for k, v in draw.items():
            boot[k].append(v)
    return {"rep": rep, "seed": seed, "design": key, "est": est, "rotations": rot,
            "edge_rotations": int(sum(at_edge(spec, p) for p in rot)),
            "edge_pooled": bool(at_edge(spec, est["pooled"])), "boot_draws_at_edge": edge_draws,
            "boot": boot, "failures": failures, "seconds": time.perf_counter() - t0}


def _summarize(rows, n, pop) -> dict:
    analytic = pop["V_rotated"] is not None
    p = dict(pop)
    if not analytic:
        p.update({"V_rotated": 1.0, "V_fixed": 1.0, "V_oracle": 1.0})
    cell = summarize(rows, n, p)
    if not analytic:
        for arm in cell["arms"].values():
            arm.pop("coverage_analytic", None)
            arm.pop("analytic_n_var", None)
    return cell


def run_cell(key, n, reps, n_boot, seed, workers, out: Path, pop: dict, rep_start=0) -> dict:
    path = out / f"{key}_n{n}.jsonl"
    rows = _load_rows(path)
    done = {r["rep"] for r in rows}
    tasks = [(key, n, r, seed + 104_729 * n + r, n_boot)
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
                    print(f"  {i}/{len(tasks)} reps, {time.perf_counter() - t0:.0f}s", flush=True)
    rows = sorted((r for r in rows if rep_start <= r["rep"] < reps), key=lambda r: r["rep"])
    cell = _summarize(rows, n, pop)
    cell["rep_range"] = [rep_start, reps]
    cell["edge"] = {
        "rotation_thresholds_at_edge": int(sum(r["edge_rotations"] for r in rows)),
        "pooled_at_edge": int(sum(r["edge_pooled"] for r in rows)),
        "boot_draws_at_edge": int(sum(r["boot_draws_at_edge"] for r in rows)),
        "boot_draws_total": int(sum(len(r["boot"]["avg"]) for r in rows)),
    }
    return cell


def report(result: dict) -> str:
    pop = result["population"]
    V = pop["V_rotated"]
    lines = [f"Design `{result['design']}`: R^2 {pop['r2']}, treated {pop['treated_share_phi0']:.2f} "
             f"at phi0 = {pop['phi0']:.4f}, {pop['treated_share_target']:.2f} at the target "
             f"phi* = {pop['phi_star']:.6f}; c = {pop['cost']:.4f}, H = {pop['H']:.6f}; window "
             f"[{pop['window'][0]:.3f}, {pop['window'][1]:.3f}] ({'capped' if pop['capped'] else 'overlap'}, "
             f"{pop['window_share']:.2f} of units); first-order n Var "
             + (f"{V:.2f} (rotated), {pop['V_oracle']:.3f} (oracle)" if V is not None else "not computed"),
             "", "| n | estimator | sqrt(n) bias (s.e.) | n Var | mean n Var* | percentile "
             "| normal | edge (est / boot draws) |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for n, cell in sorted(result["cells"].items(), key=lambda kv: int(kv[0])):
        e = cell["edge"]
        for arm, label in (("avg", "rotated average"), ("pooled", "pooled maximizer"),
                           ("oracle_avg", "oracle")):
            c = cell["arms"][arm]
            lines.append(
                f"| {int(n):,} | {label} | {c['sqrt_n_bias']:+.2f} ({c['sqrt_n_bias_se']:.2f}) "
                f"| {c['n_var']:.2f} | {c['mean_n_boot_var']:.2f} | {c['coverage_percentile']:.3f} "
                f"| {c['coverage_normal']:.3f} | {e['rotation_thresholds_at_edge']} / "
                f"{e['boot_draws_at_edge']} |")
    lines.append("")
    lines.append("; ".join(f"n = {int(n):,}: {c['reps']} data sets, {c['boot_draws_per_rep']} draws, "
                           f"{c['boot_failures']} failed" for n, c in
                           sorted(result["cells"].items(), key=lambda kv: int(kv[0]))))
    return "\n".join(lines)


def check_table2(key: str, ref_dir: Path, n: int, k: int, workers: int) -> None:
    """The r2_50_bal point estimates must equal Table 2's rows (same seeds and data)."""
    ref = {r["rep"]: r for r in _load_rows(ref_dir / f"displayed_n{n}.jsonl")}
    spec = SPECS[key]
    worst = 0.0
    for rep in sorted(ref)[:k]:
        seed = SEED + 104_729 * n + rep
        if ref[rep]["seed"] != seed:
            raise SystemExit(f"seed mismatch at rep {rep}: {ref[rep]['seed']} vs {seed}")
        s = generate(spec.design, n, seed)
        est, _ = estimate(s, spec, np.random.default_rng([seed, 1]))
        for arm in ("avg", "pooled", "fixed", "oracle_avg", "oracle_pooled"):
            worst = max(worst, abs(est[arm] - ref[rep]["est"][arm]))
        print(f"rep {rep}: avg {est['avg']:.12f} vs {ref[rep]['est']['avg']:.12f}")
    print(f"largest difference over {k} data sets and five arms: {worst:.3e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=sorted(SPECS))
    ap.add_argument("--n", type=int, nargs="+", default=[4000, 16000, 64000])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--population", action="store_true", help="print every design's population")
    ap.add_argument("--check", type=Path, default=None,
                    help="Table 2 run directory: compare r2_50_bal point estimates with its rows")
    args = ap.parse_args(argv)
    if args.population:
        for key in SPECS:
            p = population(SPECS[key])
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in p.items() if k != "variances"}))
        return
    if args.design is None:
        raise SystemExit("--design is required")
    if args.check is not None:
        check_table2(args.design, args.check, args.n[0], 3, args.workers)
        return
    summary_path = args.out / f"{args.design}_summary.json"
    if args.report:
        print(report(json.loads(summary_path.read_text())))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    pop = population(SPECS[args.design])
    print(json.dumps({k: v for k, v in pop.items() if k != "variances"}), flush=True)
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result.update({"design": args.design, "population": pop,
                   "design_params": repr(SPECS[args.design].design),
                   "settings": {"boot": args.boot, "seed": args.seed, "scheme": "resplit",
                                "window_tail": WINDOW_TAIL}})
    result.setdefault("cells", {})
    for n in args.n:
        cell = run_cell(args.design, n, args.reps, args.boot, args.seed, args.workers,
                        args.out, pop, args.rep_start)
        result["cells"][str(n)] = cell
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: cell[k] for k in ("n", "reps", "boot_draws_per_rep", "edge")}),
              flush=True)
    print(report(result), flush=True)
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
