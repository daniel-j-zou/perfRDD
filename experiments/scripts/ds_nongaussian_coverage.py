"""Bootstrap coverage of the rotated four-role estimator in the t5 and mixture designs.

Appendix 7 of ``differing_slopes.tex`` ("Non-Gaussian linear predictor") reports
point estimates only for the two non-Gaussian designs of
``ds_four_role_nongaussian.py``.  This script adds the bootstrap check that
Table 2 (``tab:ds-coverage-displayed``) gives the displayed design: for each
data set, 95% percentile and normal intervals from the full re-estimation
bootstrap of Section 3, with ``--boot`` draws (999 in the paper).

Pieces reused unchanged:

* data: ``ds_four_role_nongaussian.generate`` (T = X1 with a t5 or mixture law,
  X2, X3, eta ~ N(0, 1); effect alpha = 2 + 0.4 eta + 0.8 tanh(1.5 eta),
  b = eta^2/2 + 0.3 sin(2 eta) with DS_NG_EFFECT=nonlinear, the default);
* estimator settings: those of ``ds_four_role_nongaussian`` (OLS first stage,
  J = [-1.75, 1.75], T = [-3, 3], Phi = [-1.5, 1.5], c = 2.25, eps = 0.1,
  phi0 = 0), i.e. the estimator of Section 5.1;
* target and first-order variances: ``NonGaussianPopulation``;
* bootstrap scheme, seeds and summaries: ``ds_four_role_coverage`` (resample n
  with replacement, four new random blocks, every fit redone; percentile and
  normal intervals; the analytic interval as a check).  There is no oracle arm.

The point estimate on a data set equals that of ``ds_four_role_nongaussian.replicate``
for the same data and split (``--check-against`` compares them on saved rows).

Rows go to ``<out>/<law>_n<n>.jsonl`` as replications finish; a rerun resumes.
Seeds depend only on (seed, law, n, rep), so ``--rep-start a --reps b`` shards
a cell across Slurm array tasks and the merged shards equal one run.

Run:
    python -m experiments.scripts.ds_nongaussian_coverage --law t5 --n 4000 \
        --reps 4 --boot 19 --workers 4 --out experiments/runs/ds_ng_cov_test
    python -m experiments.scripts.ds_nongaussian_coverage --law t5 --report --out <dir>
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
from pathlib import Path

import numpy as np

from experiments.methods.weighted_tails import fit_weighted_tails
from experiments.scripts.ds_four_role_coverage import (
    SplineTailPieces, _avg_and_pooled, _load_rows, _resample, summarize,
)
from experiments.scripts.ds_four_role_nongaussian import (
    COST, EFFECT, EPS, J, LAWS, PHI0, POLICY, T_SUPPORT, NonGaussianPopulation, at_edge,
    generate,
)
from experiments.scripts.ds_local_efficiency_sim import (
    ROLES, _xt, ols_first_stage, outcome_fit,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_nongaussian_coverage"
GRID = np.linspace(POLICY[0], POLICY[1], 61)


def estimate(s, blocks) -> tuple:
    """(avg, pooled, fixed) and the four rotated thresholds on one split."""
    crit, vals = [], []
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
        e = e[(e >= l_hat) & (e <= u_hat)]
        fit = outcome_fit(s, folds["outcome"], eta_hat, J)
        pieces = SplineTailPieces(tails, e, n_u)
        amc = fit.a(e) - COST
        b2 = np.asarray(fit.beta2)
        row = []
        for x in GRID:
            (sg,), sh = pieces.sums(x, [amc])
            row.append((sg + float(sh @ b2)) / n_u)
        crit.append(lambda x, p=pieces, a=amc, b=b2: p.value(x, a, b))
        vals.append(row)
    avg, pooled, rot = _avg_and_pooled(crit, GRID, np.array(vals))
    return {"avg": avg, "pooled": pooled, "fixed": rot[0]}, rot


def data_seed(seed: int, law: str, n: int, rep: int) -> int:
    return seed + 10_000_000 * LAWS.index(law) + 104_729 * n + rep


def replicate(task) -> dict:
    law, n, rep, seed, n_boot = task
    t0 = time.perf_counter()
    s = generate(law, n, seed)
    blocks = np.array_split(np.random.default_rng([seed, 1]).permutation(n), 4)
    est, rot = estimate(s, blocks)
    brng = np.random.default_rng([seed, 2])
    boot = {key: [] for key in est}
    failures = edge_draws = 0
    for b in range(n_boot):
        idx = brng.integers(0, n, n)
        split = np.array_split(np.random.default_rng([seed, 3, b]).permutation(n), 4)
        try:
            draw, drot = estimate(_resample(s, idx), split)
        except (np.linalg.LinAlgError, ValueError):
            failures += 1
            continue
        edge_draws += int(any(at_edge(p) for p in drot) or at_edge(draw["pooled"]))
        for key, val in draw.items():
            boot[key].append(val)
    return {"rep": rep, "seed": seed, "law": law, "est": est, "rotations": rot,
            "edge_rotations": int(sum(at_edge(p) for p in rot)),
            "edge_pooled": bool(at_edge(est["pooled"])), "boot_draws_at_edge": edge_draws,
            "boot": boot, "failures": failures, "seconds": time.perf_counter() - t0}


def population(law: str) -> dict:
    d = NonGaussianPopulation(law).decomposition()
    return {k: d[k] for k in ("law", "phi_star", "H", "l0", "u0", "variances",
                              "V_rotated", "V_fixed", "V_oracle")}


def run_cell(law, n, reps, n_boot, seed, workers, out: Path, pop: dict, rep_start=0) -> dict:
    path = out / f"{law}_n{n}.jsonl"
    rows = _load_rows(path)
    done = {r["rep"] for r in rows}
    tasks = [(law, n, r, data_seed(seed, law, n, r), n_boot)
             for r in range(rep_start, reps) if r not in done]
    if tasks:
        print(f"[{law}] n={n}: {len(done)} done, running {len(tasks)}", flush=True)
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
    cell = summarize(rows, n, pop)
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
    lines = [f"Law `{result['law']}` (effect {result['effect']}): phi* = {pop['phi_star']:.6f}, "
             f"H = {pop['H']:.6f}; first-order n Var {pop['V_rotated']:.2f} (rotated), "
             f"{pop['V_fixed']:.1f} (fixed).", "",
             "| n | estimator | sqrt(n) bias (s.e.) | n Var | mean n Var* | percentile "
             "| normal | analytic (check) | edge (est / boot draws) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, cell in sorted(result["cells"].items(), key=lambda kv: int(kv[0])):
        e = cell["edge"]
        for arm, label in (("avg", "rotated average"), ("pooled", "pooled maximizer")):
            c = cell["arms"][arm]
            lines.append(
                f"| {int(n):,} | {label} | {c['sqrt_n_bias']:+.2f} ({c['sqrt_n_bias_se']:.2f}) "
                f"| {c['n_var']:.2f} | {c['mean_n_boot_var']:.2f} | {c['coverage_percentile']:.3f} "
                f"| {c['coverage_normal']:.3f} | {c['coverage_analytic']:.3f} "
                f"| {e['rotation_thresholds_at_edge']} / {e['boot_draws_at_edge']} |")
    lines.append("")
    lines.append("; ".join(f"n = {int(n):,}: {c['reps']} data sets, {c['boot_draws_per_rep']} "
                           f"draws, {c['boot_failures']} failed draws"
                           for n, c in sorted(result["cells"].items(), key=lambda kv: int(kv[0]))))
    return "\n".join(lines)


def check_against(law: str, n: int, rows_path: Path, k: int) -> None:
    """Point estimates on the data and split of saved ds_four_role_nongaussian rows."""
    for r in json.loads(rows_path.read_text())[:k]:
        s = generate(law, n, r["seed"])
        rng = np.random.default_rng(r["seed"] + 7_000_001)
        est, rot = estimate(s, np.array_split(rng.permutation(n), 4))
        print(f"seed {r['seed']}: average {est['avg']:.10f} vs {r['average']:.10f}, "
              f"pooled {est['pooled']:.10f} vs {r['pooled']:.10f}, "
              f"max rotation diff {max(abs(a - b) for a, b in zip(rot, r['rotations'])):.2e}",
              flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--law", choices=LAWS, required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[4000, 16000, 64000])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--check-against", type=Path, default=None,
                    help="rows_<law>_<n>.json of ds_four_role_nongaussian; compare point estimates")
    args = ap.parse_args(argv)
    if EFFECT != "nonlinear":
        raise SystemExit("this check is for the nonlinear effect (DS_NG_EFFECT=nonlinear)")
    if args.check_against is not None:
        check_against(args.law, args.n[0], args.check_against, 3)
        return
    summary_path = args.out / f"{args.law}_summary.json"
    if args.report:
        print(report(json.loads(summary_path.read_text())))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    pop = population(args.law)
    print(f"[{args.law}] phi* = {pop['phi_star']:.6f}, H = {pop['H']:.6f}, "
          f"V_rotated = {pop['V_rotated']:.3f}", flush=True)
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result.update({"law": args.law, "effect": EFFECT, "population": pop,
                   "settings": {"boot": args.boot, "seed": args.seed, "scheme": "resplit"}})
    result.setdefault("cells", {})
    for n in args.n:
        cell = run_cell(args.law, n, args.reps, args.boot, args.seed, args.workers, args.out,
                        pop, args.rep_start)
        result["cells"][str(n)] = cell
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: cell[k] for k in ("n", "reps", "boot_draws_per_rep", "edge")}),
              flush=True)
    print(report(result), flush=True)
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
