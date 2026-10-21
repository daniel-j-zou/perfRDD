"""Taxi application for ``differing_slopes.tex`` Section 6: the two chosen specifications.

At the author's request (weekly scope review, 2026-10-20), Section 6 reports two
specifications of ``taxi_four_role_grid.py`` side by side, with no single headline:

  S1-E1  score linear in distance;                effect linear in distance
  S2-E1  score in distance and log distance;      effect linear in distance

Both scores also contain the hour, day-of-week, weekday-hour x weekend and group-ride
indicators, and the untreated baseline uses the score covariates. Fare in dollars,
eps = 0.05, outcome fit on the central 99.5% of residuals, policy set [$6, $24], as in
the nine-way grid. The trim window is capped at the central 99% of the first-stage
residuals, re-estimated in every rotation and bootstrap draw from the index fold (author
decision, 2026-10-20): at eps = 0.05 the overlap endpoints l0, u0 lie beyond the
residuals, so the supported population is the central 99% of latent types.

For each specification the script records:

* the four-role estimate (average of the four rotated thresholds), the pooled
  maximizer, the rotations, the own-score check, the gain over the deployed $15
  cutoff, beta2-hat, the window share, and the curves tau-hat, U-hat and alpha-hat;
* a full re-estimation bootstrap (resample rides, re-split, re-fit every nuisance)
  with ``--boot`` draws: 95% and 90% percentile intervals for the cutoff (the paper's
  interval, author decision 2026-10-20), the normal interval as a check, and
  pointwise percentile bands for U-hat and alpha-hat. Draw b uses the seeds of the
  grid script, so its first 20 draws repeat the grid's;
* the independence diagnostics of the score (R^2 of eta-hat and eta-hat^2 on rich
  covariate functions, the spread ratio of eta-hat across distance deciles) and the
  identification and held-out fit of the effect block;
* robustness without the bootstrap: eps in {0.075, 0.10}; the tail excluded from J
  in {0.0005, 0.001, 0.004} with the window cap fixed at the central 99% (main
  0.0025), so the window stays inside J; J and the window cap scaled together,
  (tail, window tail) in {(0.001, 0.002), (0.005, 0.01)} (main (0.0025, 0.005)),
  which also changes the trimmed population; and four further fold seeds.

Draws are appended to ``<out>/<spec>_draws.jsonl`` as they finish, and a rerun
resumes from them. ``--figures`` redraws the figures from the saved results.

Run (from code/):
    python -m experiments.scripts.taxi_section6 --boot 999 --workers 3
    python -m experiments.scripts.taxi_section6 --figures
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_four_role_grid as G

OUT = M.ROOT / "runs" / "taxi_section6"
SPECS = {"S1-E1": "linear", "S2-E1": "distlog"}
LABEL = {"S1-E1": "score: distance", "S2-E1": "score: distance + log distance"}
EFFECT = "distance"
EPS = G.EPS
LAM = G.LAM
TAIL, WINDOW_TAIL = 0.0025, 0.005
GRID_POINTS = G.GRID_POINTS
ROBUST_EPS = (0.075, 0.10)
ROBUST_TAIL = (0.0005, 0.001, 0.004)
ROBUST_CAPS = ((0.001, 0.002), (0.005, 0.01))
ROBUST_SEEDS = (2, 3, 4, 5)


def build(df, spec, eps=EPS, tail=TAIL, window_tail=WINDOW_TAIL):
    data = M.make_data(df, LAM, EFFECT, score=SPECS[spec])
    design = M.make_design(data, eps, effect=True, tail=tail, window_tail=window_tail,
                           window_estimated=True)
    return data, design


def estimate(data, design, seed, curve_grid=None, alpha_grid=None):
    r = M.four_role(data, design, seed, curves=None if curve_grid is None else M.boxcox(curve_grid, LAM),
                    grid_points=GRID_POINTS, alpha_grid=alpha_grid)
    d = lambda p: M.dollars(p, LAM)
    out = {"phi": d(r["phi_avg"]), "cell": M.fare_cell(d(r["phi_avg"]), data.fare),
           "phi_pooled": d(r["phi_pooled"]), "rotations": [d(p) for p in r["phi_rotations"]],
           "at_boundary": r["at_boundary"], "local_maxima": r["local_maxima"],
           "gain_cents": 100 * r["gain_vs_deployed"], "beta2": r["beta2"],
           "window": [float(v) for v in r["window"]], "window_share": r["window_share"]}
    if curve_grid is not None:
        out.update(U=r["U"].tolist(), tau=r["tau"].tolist(),
                   own_cell=M.fare_cell(d(r["own_cutoff"]), data.fare))
    if alpha_grid is not None:
        out["alpha"] = np.asarray(r["alpha"]).tolist()
    return out


_B: dict = {}


def _init(data, design, seed, curve_grid, alpha_grid):
    _B.update(data=data, design=design, seed=seed, curve_grid=curve_grid, alpha_grid=alpha_grid)


def _draw(b):
    data, design, seed = _B["data"], _B["design"], _B["seed"]
    idx = np.random.default_rng(seed + 1_000 + b).integers(0, len(data.Q), len(data.Q))
    r = M.four_role(data.subset(idx), design, seed=seed + 50_000 + b, grid_points=GRID_POINTS,
                    curves=M.boxcox(_B["curve_grid"], LAM), alpha_grid=_B["alpha_grid"])
    d = lambda p: M.dollars(p, LAM)
    return {"b": b, "phi": d(r["phi_avg"]), "phi_pooled": d(r["phi_pooled"]),
            "at_boundary": bool(r["at_boundary"]), "U": np.round(r["U"], 7).tolist(),
            "alpha": np.round(np.asarray(r["alpha"]), 6).tolist()}


def _load(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap(spec, data, design, seed, curve_grid, alpha_grid, boot, workers, out):
    path = out / f"{spec}_draws.jsonl"
    rows = _load(path)
    done = {r["b"] for r in rows}
    todo = [b for b in range(boot) if b not in done]
    if todo:
        t0 = time.time()
        print(f"[{spec}] bootstrap: {len(done)} done, running {len(todo)}", flush=True)
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(data, design, seed, curve_grid, alpha_grid)) as ex, \
                path.open("a") as fh:
            futures = [ex.submit(_draw, b) for b in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} draws, {time.time() - t0:.0f}s", flush=True)
    rows = sorted(rows, key=lambda r: r["b"])[:boot]
    return rows


def summarize_boot(est, rows, fares):
    phis = np.array([r["phi"] for r in rows])
    U = np.array([r["U"] for r in rows])
    A = np.array([r["alpha"] for r in rows])
    sd = float(phis.std(ddof=1))
    q = lambda p: float(np.quantile(phis, p))
    return {"B": len(rows), "sd": sd,
            "pct95": [q(0.025), q(0.975)], "pct90": [q(0.05), q(0.95)],
            "normal95": [est["phi"] - 1.96 * sd, est["phi"] + 1.96 * sd],
            "share_boundary": float(np.mean([r["at_boundary"] for r in rows])),
            "share_below_15": float(np.mean(phis < M.FARE0)),
            "cells": {str(k): int(v) for k, v in zip(*np.unique(
                [M.fare_cell(p, fares) for p in phis], return_counts=True))},
            "U_band95": [np.quantile(U, 0.025, axis=0).tolist(), np.quantile(U, 0.975, axis=0).tolist()],
            "alpha_band95": [np.quantile(A, 0.025, axis=0).tolist(),
                             np.quantile(A, 0.975, axis=0).tolist()]}


def run(boot, workers, seed, out):
    out.mkdir(parents=True, exist_ok=True)
    df = M.load_frame("full")
    curve_grid = np.linspace(*M.POLICY_DOLLARS, 73)
    result = {"eps": EPS, "lambda": LAM, "fare_grid": curve_grid.tolist(), "specs": {}}
    t0 = time.time()
    for spec in SPECS:
        data, design = build(df, spec)
        alpha_grid = np.linspace(design.window[0], design.window[1], 61)
        est = estimate(data, design, seed, curve_grid, alpha_grid)
        est["eta_grid"] = alpha_grid.tolist()
        est["independence"] = G.independence(df, data)
        est["effect_block"] = M.effect_diagnostics(data, design, seed)
        est["design"] = {"J": list(design.J), "window_caps": list(design.window),
                         "T_support": list(design.T_support), "phi0": design.phi0}
        print(f"[{spec}] phi {est['phi']:.2f} cell {est['cell']:.2f} pooled {est['phi_pooled']:.2f} "
              f"own {est['own_cell']:.2f} R2eta {est['independence']['R2_eta']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        robust = []
        for eps in ROBUST_EPS:
            d2, g2 = build(df, spec, eps=eps)
            robust.append({"kind": "eps", "value": eps, **estimate(d2, g2, seed)})
        for tail in ROBUST_TAIL:
            d2, g2 = build(df, spec, tail=tail)
            robust.append({"kind": "J_tail", "value": tail, **estimate(d2, g2, seed)})
        for tail, wt in ROBUST_CAPS:
            d2, g2 = build(df, spec, tail=tail, window_tail=wt)
            robust.append({"kind": "J_and_window", "value": [tail, wt], **estimate(d2, g2, seed)})
        for s in ROBUST_SEEDS:
            robust.append({"kind": "fold_seed", "value": s, **estimate(data, design, s)})
        est["robustness"] = robust
        print(f"[{spec}] robustness: " + ", ".join(f"{r['kind']}={r['value']}: {r['phi']:.2f}"
                                                   for r in robust), flush=True)
        if boot:
            rows = bootstrap(spec, data, design, seed, curve_grid, alpha_grid, boot, workers, out)
            est["boot"] = summarize_boot(est, rows, data.fare)
            b = est["boot"]
            print(f"[{spec}] B={b['B']} sd {b['sd']:.3f} 95% pct [{b['pct95'][0]:.2f}, "
                  f"{b['pct95'][1]:.2f}] boundary {b['share_boundary']:.3f}", flush=True)
        result["specs"][spec] = est
        (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    figures(result, out)
    print(f"wrote {out} [{time.time() - t0:.0f}s]")


def figures(result, out):
    """Section 6 figures: U-hat and alpha-hat with 95% bootstrap bands, and the tau-hat curves."""
    grid = np.array(result["fare_grid"])
    specs = list(result["specs"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, spec in zip(axes, specs):
        r = result["specs"][spec]
        U = 100 * np.array(r["U"])
        if "boot" in r:
            lo, hi = (100 * np.array(v) for v in r["boot"]["U_band95"])
            ax.fill_between(grid, lo, hi, color="C0", alpha=0.25, lw=0)
        ax.plot(grid, U, color="C0", lw=1.6)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(r["phi"], color="C3", ls="--", lw=1)
        ax.axvline(M.FARE0, color="0.5", ls=":", lw=1)
        ax.set_title(f"{LABEL[spec]} (\\${r['phi']:.2f})", fontsize=10)
        ax.set_xlabel("fare cutoff (\\$)")
    axes[0].set_ylabel("tips relative to the \\$15 cutoff (cents per ride)")
    fig.tight_layout()
    fig.savefig(out / "taxi_U.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, spec in zip(axes, specs):
        r = result["specs"][spec]
        x, a = np.array(r["eta_grid"]), np.array(r["alpha"])
        if "boot" in r:
            lo, hi = (np.array(v) for v in r["boot"]["alpha_band95"])
            ax.fill_between(x, lo, hi, color="C2", alpha=0.25, lw=0)
        ax.plot(x, a, color="C2", lw=1.6)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(LABEL[spec], fontsize=10)
        ax.set_xlabel(r"latent type $\eta$ (\$)")
    axes[0].set_ylabel(r"$\hat\alpha(\eta)$ (\$ per ride)")
    fig.tight_layout()
    fig.savefig(out / "taxi_alpha.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, spec in enumerate(specs):
        r = result["specs"][spec]
        ax.plot(grid, r["tau"], color=f"C{k}", lw=1.6, label=f"{LABEL[spec]} (\\${r['phi']:.2f})")
        ax.axvline(r["phi"], color=f"C{k}", ls="--", lw=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(M.FARE0, color="0.5", ls=":", lw=1.1)
    ax.set_xlabel("fare cutoff (\\$)")
    ax.set_ylabel("marginal net effect (\\$ per ride)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "taxi_tau.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--figures", action="store_true", help="redraw figures from summary.json")
    a = ap.parse_args(argv)
    if a.figures:
        figures(json.loads((a.out / "summary.json").read_text()), a.out)
        return
    run(a.boot, a.workers, a.seed, a.out)


if __name__ == "__main__":
    main()
