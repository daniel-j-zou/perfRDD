"""Taxi application for ``differing_slopes.tex`` Section 6.

Samples (``--sample``):

  daytime   the main sample: daytime rides without tolls or surcharges, every fare,
            effect linear in distance. Output in ``runs/taxi_section6/``.
  allhours  robustness row (author, 2026-11-12: tried as the main sample, then "Back to
            daytime" after the residual diagnostics): VTS credit-card rides at every
            hour, the score on the fare plus the time-of-day surcharge (the base amount
            the terminal switches on), the effect linear in distance and a daytime
            indicator, hour indicators for every hour. Output in
            ``runs/taxi_section6_allhours/``.

The paper writes the square-root score as Q = sqrt(fare) (author, 2026-11-12); the code
uses the Box-Cox form 2(sqrt(fare) - 1). The two give the same thresholds, U-hat and
alpha-hat values (checked to 1e-12); only eta's unit differs, and eta in the paper is
``lambda`` times eta here (``eta_scale`` in summary.json). The paper figure converts.

Specifications:

  R2-E1  square root of the fare; score in distance and log distance   (main, author
         decision 2026-10-27: "for now" the paper's use case)
  S1-E1  fare in dollars; score linear in distance                     (weekly review
  S2-E1  fare in dollars; score in distance and log distance            2026-10-20)
  R3-E1  square root of the fare; score a cubic B-spline in log distance (robustness
         row, author decision 2026-11-04; its D x distance column is weakly identified
         and its fitted score is not monotone in distance, see the research log)

Every score also contains the hour, day-of-week, weekday-hour x weekend and group-ride
indicators, and the untreated baseline uses the score covariates. The fare scale is a
Box-Cox power (1/2: square root; 1: dollars); a cutoff on the transformed fare is a
cutoff on the fare. eps = 0.05, outcome fit on the central 99.5% of residuals, policy set [$6, $24], as in
the nine-way grid. The trim window is capped at the central 99% of the first-stage
residuals, re-estimated in every rotation and bootstrap draw from the evaluation fold (author
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

Each draw also stores beta2-hat and the gain over $15, so the bootstrap gives their
intervals directly (``taxi_section6_beta2.py`` did this separately for the daytime
sample). Draws are appended to ``<out>/<spec>_draws.jsonl`` as they finish, and a rerun
resumes from them. ``--figures`` redraws the figures from the saved results.

Results for each specification are merged into ``<out>/summary.json``.

Run (from code/; ``--sample daytime`` is the default):
    python -m experiments.scripts.taxi_section6 --specs R2-E1 --boot 999 --workers 6
    python -m experiments.scripts.taxi_section6 --sample allhours --specs R2-E1 --boot 0
    python -m experiments.scripts.taxi_section6 --sample allhours --comparisons --specs R2-E1
    python -m experiments.scripts.taxi_section6 --specs S1-E1 S2-E1 --boot 999 --workers 3
    python -m experiments.scripts.taxi_section6 --comparisons --specs R2-E1
    python -m experiments.scripts.taxi_section6 --figures --specs R2-E1
    python -m experiments.scripts.taxi_section6 --specs R3-E1 --boot 0 \
        --out experiments/runs/taxi_section6_R3-E1
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

OUT = M.ROOT / "runs" / "taxi_section6"          # daytime sample
SAMPLES = {
    "allhours": {"frame": "allhours", "hours": "all", "effect": "distance+daytime",
                 "effect_cols": ["Trip_Distance", "daytime"],
                 "out": M.ROOT / "runs" / "taxi_section6_allhours"},
    "daytime": {"frame": "full", "hours": "day", "effect": "distance",
                "effect_cols": ["Trip_Distance"], "out": OUT},
}
SAMPLE = "daytime"
SPECS = {"R2-E1": (0.5, "distlog"), "S1-E1": (1.0, "linear"), "S2-E1": (1.0, "distlog"),
         "R3-E1": (0.5, "spline")}
MAIN = "R2-E1"
LABEL = {"R2-E1": "square root of fare; distance + log distance",
         "S1-E1": "dollars; distance", "S2-E1": "dollars; distance + log distance",
         "R3-E1": "square root of fare; spline in log distance"}
EFFECT = "distance"          # the daytime sample's effect (taxi_locscale_row.py uses it)
EPS = G.EPS
TAIL, WINDOW_TAIL = 0.0025, 0.005
GRID_POINTS = G.GRID_POINTS
ROBUST_EPS = (0.075, 0.10)
ROBUST_TAIL = (0.0005, 0.001, 0.004)
ROBUST_CAPS = ((0.001, 0.002), (0.005, 0.01))
ROBUST_SEEDS = (2, 3, 4, 5)


def build(df, spec, eps=EPS, tail=TAIL, window_tail=WINDOW_TAIL, sample=SAMPLE):
    lam, score = SPECS[spec]
    cfg = SAMPLES[sample]
    data = M.make_data(df, lam, cfg["effect"], score=score, hour_set=cfg["hours"])
    design = M.make_design(data, eps, effect=True, tail=tail, window_tail=window_tail,
                           window_estimated=True)
    return data, design


def estimate(data, design, seed, curve_grid=None, alpha_grid=None):
    r = M.four_role(data, design, seed, curves=None if curve_grid is None else M.boxcox(curve_grid, data.lam),
                    grid_points=GRID_POINTS, alpha_grid=alpha_grid)
    d = lambda p: M.dollars(p, data.lam)
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
                    curves=M.boxcox(_B["curve_grid"], data.lam), alpha_grid=_B["alpha_grid"],
                    tau_and_own=False)
    d = lambda p: M.dollars(p, data.lam)
    return {"b": b, "phi": d(r["phi_avg"]), "phi_pooled": d(r["phi_pooled"]),
            "at_boundary": bool(r["at_boundary"]), "U": np.round(r["U"], 7).tolist(),
            "beta2": np.round(np.asarray(r["beta2"], float), 7).tolist(),
            "gain_cents": float(100 * r["gain_vs_deployed"]),
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


def summarize_boot(est, rows, fares, effect_sd=None):
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
                             np.quantile(A, 0.975, axis=0).tolist()],
            **_boot_extras(rows, effect_sd)}


def _boot_extras(rows, effect_sd):
    """Percentile intervals for beta2-hat (per unit of each effect column) and for the
    gain over $15, when the draws carry them."""
    out = {}
    if rows and "beta2" in rows[0] and effect_sd is not None:
        B2 = np.array([r["beta2"] for r in rows]) / np.asarray(effect_sd)
        out["beta2_raw_pct95"] = np.quantile(B2, [0.025, 0.975], axis=0).T.tolist()
        out["beta2_raw_sd"] = B2.std(0, ddof=1).tolist()
    if rows and "gain_cents" in rows[0]:
        g = np.array([r["gain_cents"] for r in rows])
        out["gain_cents_pct95"] = np.quantile(g, [0.025, 0.975]).tolist()
    return out


def _save(path, mutate):
    """Read summary.json, apply ``mutate``, write it back: concurrent runs (the main
    specification, its comparisons) each change only their own entries."""
    res = json.loads(path.read_text()) if path.exists() else {}
    res.setdefault("specs", {})
    mutate(res)
    path.write_text(json.dumps(res, indent=1) + "\n")
    return res


_R: dict = {}


def _rinit(frame, spec, sample, seed):
    _R.update(df=M.load_frame(frame), spec=spec, sample=sample, seed=seed)


def _robust_one(task):
    kind, value, kw, seed = task
    d2, g2 = build(_R["df"], _R["spec"], sample=_R["sample"], **kw)
    return {"kind": kind, "value": value, **estimate(d2, g2, seed if seed is not None else _R["seed"])}


def robustness_tasks():
    tasks = [("eps", eps, {"eps": eps}, None) for eps in ROBUST_EPS]
    tasks += [("J_tail", tail, {"tail": tail}, None) for tail in ROBUST_TAIL]
    tasks += [("J_and_window", [tail, wt], {"tail": tail, "window_tail": wt}, None)
              for tail, wt in ROBUST_CAPS]
    tasks += [("fold_seed", s, {}, s) for s in ROBUST_SEEDS]
    return tasks


def run(boot, workers, seed, out, specs, sample=SAMPLE, robust=True):
    out.mkdir(parents=True, exist_ok=True)
    cfg = SAMPLES[sample]
    df = M.load_frame(cfg["frame"])
    effect_sd = df[cfg["effect_cols"]].to_numpy(float).std(0).tolist()
    curve_grid = np.linspace(*M.POLICY_DOLLARS, 73)
    path = out / "summary.json"

    def head(res):
        res.update(eps=EPS, fare_grid=curve_grid.tolist(), sample=sample, n=int(len(df)))
        res.pop("lambda", None)                 # now recorded per specification
    _save(path, head)
    t0 = time.time()
    for spec in specs:
        data, design = build(df, spec, sample=sample)
        alpha_grid = np.linspace(design.window[0], design.window[1], 61)
        est = estimate(data, design, seed, curve_grid, alpha_grid)
        est["eta_grid"] = alpha_grid.tolist()
        est["lambda"], est["score"] = SPECS[spec]
        est["eta_scale"] = est["lambda"] or 1.0      # paper's eta = eta_scale x eta here
        est["effect_cols"], est["effect_sd"] = cfg["effect_cols"], effect_sd
        est["beta2_raw"] = (np.asarray(est["beta2"]) / np.asarray(effect_sd)).tolist()
        est["independence"] = G.independence(df, data)
        A = M._xt(data.Z)
        resid = data.Q - A @ np.linalg.lstsq(A, data.Q, rcond=None)[0]
        est["score_R2"] = float(1.0 - resid.var() / data.Q.var())
        est["share_at_or_above_15"] = float(np.mean(data.fare >= M.FARE0))
        if "daytime" in df:
            est["daytime_share"] = float(df["daytime"].mean())
        est["effect_block"] = M.effect_diagnostics(data, design, seed)
        est["design"] = {"J": list(design.J), "window_caps": list(design.window),
                         "T_support": list(design.T_support), "phi0": design.phi0}
        print(f"[{spec}] phi {est['phi']:.2f} cell {est['cell']:.2f} pooled {est['phi_pooled']:.2f} "
              f"own {est['own_cell']:.2f} R2eta {est['independence']['R2_eta']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

        def put(res, est=est, spec=spec):        # keep entries written by other runs
            res["specs"][spec] = {**res["specs"].get(spec, {}), **est}
        _save(path, put)                         # before the refits and the bootstrap
        if robust:
            tasks = robustness_tasks()
            with ProcessPoolExecutor(min(workers, len(tasks)), initializer=_rinit,
                                     initargs=(cfg["frame"], spec, sample, seed)) as ex:
                rob = list(ex.map(_robust_one, tasks))
            est["robustness"] = rob
            print(f"[{spec}] robustness: " + ", ".join(f"{r['kind']}={r['value']}: {r['phi']:.2f}"
                                                       for r in rob), flush=True)
            _save(path, put)
        if boot:
            rows = bootstrap(spec, data, design, seed, curve_grid, alpha_grid, boot, workers, out)
            est["boot"] = summarize_boot(est, rows, data.fare, effect_sd)
            b = est["boot"]
            print(f"[{spec}] B={b['B']} sd {b['sd']:.3f} 95% pct [{b['pct95'][0]:.2f}, "
                  f"{b['pct95'][1]:.2f}] boundary {b['share_boundary']:.3f}", flush=True)
            _save(path, put)
    figures(json.loads(path.read_text()), out, specs)
    print(f"wrote {out} [{time.time() - t0:.0f}s]")


def comparisons(seed, out, spec=MAIN, sample=SAMPLE):
    """Point estimates for the Section 6 comparison rows (author choice, 2026-10-27):
    the common slope (W = alpha(eta), no D*X block) and the restricted $5-$25 sample
    of Haggag and Paci, both with the main specification's scale and score; for the
    all-hours sample also the daytime sample (the main sample until 2026-11-12)."""
    lam, score = SPECS[spec]
    cfg = SAMPLES[sample]
    path = out / "summary.json"
    rows = {}
    df = M.load_frame(cfg["frame"])
    data = M.make_data(df, lam, "none", score=score, hour_set=cfg["hours"])
    design = M.make_design(data, EPS, effect=False, tail=TAIL, window_tail=WINDOW_TAIL,
                           window_estimated=True)
    rows["common_slope"] = estimate(data, design, seed)
    dfr = M.load_frame("restricted")
    data = M.make_data(dfr, lam, "distance", score=score)
    design = M.make_design(data, EPS, effect=True, tail=TAIL, window_tail=WINDOW_TAIL,
                           window_estimated=True)
    rows["restricted"] = {**estimate(data, design, seed), "n": int(len(data.Q)),
                          "independence": G.independence(dfr, data)}
    if sample == "allhours":
        dfd = M.load_frame("full")
        data, design = build(dfd, spec, sample="daytime")
        rows["daytime"] = {**estimate(data, design, seed), "n": int(len(data.Q))}
    for k, r in rows.items():
        print(f"[{spec} {k}] phi {r['phi']:.2f} cell {r['cell']:.2f} boundary {r['at_boundary']} "
              f"beta2 {np.round(r['beta2'], 3).tolist()}", flush=True)
    def put(res):
        res["specs"].setdefault(spec, {})["comparisons"] = rows
    _save(path, put)


def paper_figure(result, out, spec):
    """The Section 6 figure for one specification: tips relative to the $15 threshold
    against the threshold (left) and alpha-hat against the latent ability (right), with
    95% pointwise percentile bands from the bootstrap."""
    r = result["specs"][spec]
    grid = np.array(result["fare_grid"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    U = 100 * np.array(r["U"])
    if "boot" in r:
        lo, hi = (100 * np.array(v) for v in r["boot"]["U_band95"])
        a1.fill_between(grid, lo, hi, color="C0", alpha=0.25, lw=0)
    a1.plot(grid, U, color="C0", lw=1.6)
    a1.axhline(0, color="k", lw=0.7)
    a1.axvline(r["phi"], color="C3", ls="--", lw=1)
    a1.axvline(M.FARE0, color="0.5", ls=":", lw=1)
    allhours = result.get("sample") == "allhours"
    a1.set_xlabel("threshold on fare + surcharge (\\$)" if allhours else "fare threshold (\\$)")
    a1.set_ylabel("tips vs the \\$15 threshold (cents per ride)")
    scale = r.get("lambda") or 1.0      # paper writes Q = sqrt(fare): eta is lambda x code's eta
    x, al = scale * np.array(r["eta_grid"]), np.array(r["alpha"])
    if "boot" in r:
        lo, hi = (np.array(v) for v in r["boot"]["alpha_band95"])
        a2.fill_between(x, lo, hi, color="C2", alpha=0.25, lw=0)
    a2.plot(x, al, color="C2", lw=1.6)
    a2.axhline(0, color="k", lw=0.7)
    unit = r"($\sqrt{\$}$)" if r.get("lambda", 1.0) == 0.5 else r"(\$)"
    a2.set_xlabel(r"latent ability $\eta$ " + unit)
    a2.set_ylabel(r"$\hat\alpha(\eta)$ (\$ per ride)")
    fig.tight_layout()
    fig.savefig(out / f"taxi_section6_{spec}.png", dpi=200, bbox_inches="tight")
    if spec == MAIN:                     # the file the paper includes
        fig.savefig(out / "taxi_section6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figures(result, out, specs):
    """Section 6 figures for ``specs``: U-hat and alpha-hat with 95% bootstrap bands, and
    the tau-hat curves. Files are named after the specifications."""
    grid = np.array(result["fare_grid"])
    tag = "_".join(specs)
    k = len(specs)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 3.6), sharey=True, squeeze=False)
    axes = axes[0]
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
        ax.set_xlabel("fare threshold (\\$)")
    axes[0].set_ylabel("tips relative to the \\$15 threshold (cents per ride)")
    fig.tight_layout()
    fig.savefig(out / f"taxi_U_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, k, figsize=(5 * k, 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, spec in zip(axes, specs):
        r = result["specs"][spec]
        x, a = np.array(r["eta_grid"]), np.array(r["alpha"])
        if "boot" in r:
            lo, hi = (np.array(v) for v in r["boot"]["alpha_band95"])
            ax.fill_between(x, lo, hi, color="C2", alpha=0.25, lw=0)
        ax.plot(x, a, color="C2", lw=1.6)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(LABEL[spec], fontsize=10)
        unit = r"(2$\sqrt{\$}$)" if r.get("lambda", 1.0) == 0.5 else r"(\$)"
        ax.set_xlabel(r"latent ability $\eta$ " + unit)
    axes[0].set_ylabel(r"$\hat\alpha(\eta)$ (\$ per ride)")
    fig.tight_layout()
    fig.savefig(out / f"taxi_alpha_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, spec in enumerate(specs):
        r = result["specs"][spec]
        ax.plot(grid, r["tau"], color=f"C{k}", lw=1.6, label=f"{LABEL[spec]} (\\${r['phi']:.2f})")
        ax.axvline(r["phi"], color=f"C{k}", ls="--", lw=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(M.FARE0, color="0.5", ls=":", lw=1.1)
    ax.set_xlabel("fare threshold (\\$)")
    ax.set_ylabel("marginal net effect (\\$ per ride)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"taxi_tau_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--specs", nargs="+", default=[MAIN], choices=list(SPECS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample", default=SAMPLE, choices=list(SAMPLES))
    ap.add_argument("--no-robust", action="store_true",
                    help="skip the design-variation refits (comparison rows)")
    ap.add_argument("--out", type=Path, default=None, help="default: the sample's run folder")
    ap.add_argument("--figures", action="store_true", help="redraw figures from summary.json")
    ap.add_argument("--comparisons", action="store_true",
                    help="common-slope and restricted-sample rows for the first --specs entry")
    a = ap.parse_args(argv)
    a.out = a.out or SAMPLES[a.sample]["out"]
    if a.figures:
        result = json.loads((a.out / "summary.json").read_text())
        figures(result, a.out, a.specs)
        for spec in a.specs:
            paper_figure(result, a.out, spec)
        return
    if a.comparisons:
        comparisons(a.seed, a.out, a.specs[0], a.sample)
        return
    run(a.boot, a.workers, a.seed, a.out, a.specs, a.sample, robust=not a.no_robust)


if __name__ == "__main__":
    main()
