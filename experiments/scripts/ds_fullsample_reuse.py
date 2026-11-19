"""Full-sample reuse against the four-role estimator in the Section 5 design (review item m19).

Section 3.4 of ``differing_slopes.tex`` says that full-sample reuse, which fits every
nuisance on all observations, "is not covered by this theory and is reported only in
simulations", but no simulation reported it (author, 2026-11-19: "Do both, with as many
draws as you think would be good for the paper").

The full-sample estimator is the estimator of Section 3 with one fold in every role: the
score regression, the spline estimates of g and rho_X, the trim endpoints (capped at the
central 99% of eta-hat, as in ``ds_review_ablations.estimate``), the outcome fit on J (knots
by the n^{11/60} rule applied to the whole sample) and the evaluation average all use the
same n observations. Its threshold maximizes that single criterion (61-point grid, then a
bounded refinement, as for each rotation of the four-role estimator).

Data sets are those of Table 2 (``ds_review_ablations.generate`` for the base design, seed
20261019 + 104729 n + rep), so the four-role results of Table 2 (and of r2_50_bal in
``ds_r2_balance_coverage``, which reproduces them) apply to the same data. Bootstrap draw b
resamples the indices of Table 2's draw b (stream [seed, 2]); there is no split. Rows go to
``<out>/full_n<n>.jsonl``; ``--rep-start a --reps b`` shards a cell.

Run (from code/):
    python -m experiments.scripts.ds_fullsample_reuse --n 4000 --reps 4 --boot 19 \
        --workers 4 --out experiments/runs/ds_fullsample_test
    python -m experiments.scripts.ds_fullsample_reuse --report --out <dir> \
        --four-role experiments/runs/ds_r2_balance_coverage_999
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
from experiments.scripts.ds_four_role_coverage import SplineTailPieces, _load_rows, _refine
from experiments.scripts.ds_local_efficiency_sim import _xt, ols_first_stage, outcome_fit
from experiments.scripts.ds_review_ablations import (
    CFGS, SEED, WINDOW_TAIL, _resample, data_seed, generate, population,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "ds_fullsample_reuse"
CFG = CFGS["base"]
Z95 = 1.959963984540054


def estimate_full(s) -> float:
    """Every nuisance and the evaluation average on all observations."""
    n = len(s.Q)
    idx = np.arange(n)
    gamma_hat = ols_first_stage(s, idx)
    eta_hat = s.Q - _xt(s.X) @ gamma_hat
    T = _xt(s.X) @ gamma_hat
    tails = fit_weighted_tails(T, s.X, CFG.T_support)
    l_hat = CFG.phi0 - float(np.quantile(T, 1.0 - CFG.eps))
    u_hat = CFG.phi0 - float(np.quantile(T, CFG.eps))
    l_hat = max(l_hat, float(np.quantile(eta_hat, WINDOW_TAIL)))
    u_hat = min(u_hat, float(np.quantile(eta_hat, 1.0 - WINDOW_TAIL)))
    e = eta_hat[(eta_hat >= l_hat) & (eta_hat <= u_hat)]
    fit = outcome_fit(s, idx, eta_hat, CFG.J)
    pieces = SplineTailPieces(tails, e, n)
    amc = fit.a(e) - CFG.cost
    b2 = np.asarray(fit.beta2)
    grid = np.linspace(CFG.policy[0], CFG.policy[1], 61)
    vals = np.array([pieces.value(x, amc, b2) for x in grid])
    return _refine(lambda x: pieces.value(x, amc, b2), grid, vals)


def replicate(task) -> dict:
    n, rep, seed, n_boot = task
    t0 = time.perf_counter()
    s, _ = generate(CFG, n, seed)
    est = estimate_full(s)
    brng = np.random.default_rng([seed, 2])     # Table 2's resampling stream
    boot, failures = [], 0
    for _ in range(n_boot):
        idx = brng.integers(0, n, n)
        try:
            boot.append(estimate_full(_resample(s, idx)))
        except (np.linalg.LinAlgError, ValueError):
            failures += 1
    return {"rep": rep, "seed": seed, "est": est, "boot": boot, "failures": failures,
            "seconds": time.perf_counter() - t0}


def summarize(rows, n, phi) -> dict:
    est = np.array([r["est"] for r in rows])
    err = est - phi
    B = [np.asarray(r["boot"]) for r in rows]
    sd = np.array([b.std(ddof=1) for b in B])
    lo = np.array([np.quantile(b, 0.025) for b in B])
    hi = np.array([np.quantile(b, 0.975) for b in B])
    return {"n": n, "reps": len(rows), "boot_draws_per_rep": int(np.median([len(b) for b in B])),
            "boot_failures": int(sum(r["failures"] for r in rows)),
            "bias": float(err.mean()), "sqrt_n_bias": float(np.sqrt(n) * err.mean()),
            "sqrt_n_bias_se": float(np.sqrt(n) * err.std(ddof=1) / np.sqrt(len(err))),
            "n_var": float(n * err.var(ddof=1)), "n_mse": float(n * np.mean(err ** 2)),
            "mean_n_boot_var": float(np.mean(n * sd ** 2)),
            "boot_var_ratio": float(np.mean(sd ** 2) / err.var(ddof=1)),
            "coverage_percentile": float(np.mean((lo <= phi) & (phi <= hi))),
            "coverage_normal": float(np.mean(np.abs(err) <= Z95 * sd)),
            "width_percentile": float(np.mean(hi - lo)),
            "mean_seconds_per_rep": float(np.mean([r["seconds"] for r in rows]))}


def run_cell(n, reps, n_boot, seed, workers, out: Path, rep_start=0):
    path = out / f"full_n{n}.jsonl"
    rows = _load_rows(path)
    done = {r["rep"] for r in rows}
    tasks = [(n, r, data_seed(seed, n, r), n_boot) for r in range(rep_start, reps)
             if r not in done]
    if tasks:
        print(f"[full] n={n}: {len(done)} done, running {len(tasks)}", flush=True)
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
    return sorted((r for r in rows if rep_start <= r["rep"] < reps), key=lambda r: r["rep"])


def report(out: Path, four_role: Path | None) -> str:
    res = json.loads((out / "full_summary.json").read_text())
    lines = ["| n | estimator | sqrt(n) bias (s.e.) | n Var | boot/MC var | percentile | normal |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    fr = None
    if four_role is not None:
        fr = json.loads((four_role / "r2_50_bal_summary.json").read_text())["cells"]
    for n, c in sorted(res["cells"].items(), key=lambda kv: int(kv[0])):
        if fr is not None and n in fr:
            a = fr[n]["arms"]["avg"]
            lines.append(f"| {int(n):,} | four-role | {a['sqrt_n_bias']:+.2f} ({a['sqrt_n_bias_se']:.2f}) "
                         f"| {a['n_var']:.1f} | {a['boot_var_ratio']:.2f} "
                         f"| {a['coverage_percentile']:.3f} | {a['coverage_normal']:.3f} |")
        lines.append(f"| {int(n):,} | full sample | {c['sqrt_n_bias']:+.2f} ({c['sqrt_n_bias_se']:.2f}) "
                     f"| {c['n_var']:.1f} | {c['boot_var_ratio']:.2f} "
                     f"| {c['coverage_percentile']:.3f} | {c['coverage_normal']:.3f} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, nargs="+", default=[4000, 16000, 64000])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--four-role", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.report:
        print(report(args.out, args.four_role))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    phi = population("base")["phi_star"]
    summary_path = args.out / "full_summary.json"
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result.update({"phi_star": phi, "settings": {"boot": args.boot, "seed": args.seed}})
    result.setdefault("cells", {})
    for n in args.n:
        rows = run_cell(n, args.reps, args.boot, args.seed, args.workers, args.out, args.rep_start)
        cell = summarize(rows, n, phi)
        cell["rep_range"] = [args.rep_start, args.reps]
        result["cells"][str(n)] = cell
        summary_path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: cell[k] for k in ("n", "reps", "sqrt_n_bias", "n_var",
                                                 "boot_var_ratio", "coverage_percentile",
                                                 "coverage_normal")}), flush=True)
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
