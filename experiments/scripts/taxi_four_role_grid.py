"""Nine taxi specifications: score covariates x linear part of the effect.

Score (fare in dollars), each with hour, day-of-week, weekday-hour x weekend and
group-ride indicators; the untreated baseline uses the same covariates:
  linear   distance
  distlog  distance and log distance (the meter's $2/mile plus a time charge
           growing with log distance)
  spline   cubic B-spline in log distance

Effect W = alpha(eta) + X'beta2 with X:
  distance  trip distance
  controls  distance, passengers, hour of day, day of week (the paper's taxi
            controls; tolls and surcharges are zero in this sample), all linear
  none      common slope, W = alpha(eta)

Each runs the four-role estimator (eps = 0.05; outcome fit on the central 99.5%
of residuals, window capped at the central 99%), reports the independence
diagnostics of the score, the identification and held-out fit of the effect
block, the own-score check, and a 20-draw full re-estimation bootstrap with a
90% interval estimate +- 1.645 sd.

Run (from code/):
    python -m experiments.scripts.taxi_four_role_grid --boot 20 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.scripts import taxi_four_role as M

OUT = M.ROOT / "runs" / "taxi_four_role_grid"
SCORES = ("linear", "distlog", "spline")
EFFECTS = ("distance", "controls", "none")
LAM = 1.0            # fare in dollars
EPS = 0.05
GRID_POINTS = 61


def rich_features(df: pd.DataFrame) -> np.ndarray:
    """Test functions of the covariates, richer than any score specification."""
    d = df["Trip_Distance"].to_numpy(float)
    S = M._bspline_cols(np.log(d), np.linspace(0.05, 0.95, 15))
    hr = np.floor(df["hour_of_day"].to_numpy(float)).astype(int)
    dow = df["day_of_week"].to_numpy(int)
    pc = np.clip(df["Passenger_Count"].to_numpy(int), 1, 6)
    H = np.eye(24)[hr][:, 7:20]
    W = np.eye(7)[dow][:, 1:]
    P = np.eye(7)[pc][:, 2:]
    return np.column_stack([np.ones(len(d)), S, H, W, P, H * (dow >= 5)[:, None]])


def _r2(y, F):
    c = np.linalg.lstsq(F, y, rcond=None)[0]
    return float(1.0 - np.var(y - F @ c) / np.var(y))


def independence(df, data) -> dict:
    A = M._xt(data.Z)
    eta = data.Q - A @ np.linalg.lstsq(A, data.Q, rcond=None)[0]
    F = rich_features(df)
    dec = pd.qcut(df["Trip_Distance"].to_numpy(float), 10, labels=False)
    sd = pd.Series(eta).groupby(dec).std()
    mean = pd.Series(eta).groupby(dec).mean()
    return {"R2_eta": _r2(eta, F), "R2_eta2": _r2(eta ** 2, F),
            "spread_ratio": float(sd.max() / sd.min()),
            "decile_mean_range": [float(mean.min()), float(mean.max())],
            "sd_eta": float(eta.std())}


_B: dict = {}


def _init(data, design, seed, curve_grid, alpha_grid):
    _B.update(data=data, design=design, seed=seed, curve_grid=curve_grid, alpha_grid=alpha_grid)


def _draw(b):
    data, design = _B["data"], _B["design"]
    idx = np.random.default_rng(_B["seed"] + 1_000 + b).integers(0, len(data.Q), len(data.Q))
    r = M.four_role(data.subset(idx), design, seed=_B["seed"] + 50_000 + b,
                    grid_points=GRID_POINTS, curves=_B["curve_grid"], alpha_grid=_B["alpha_grid"])
    return r["phi_avg"], r["at_boundary"], r["U"], r["alpha"]


def run_one(df, score, effect, boot, workers, seed, curve_grid):
    t0 = time.time()
    data = M.make_data(df, LAM, effect if effect != "none" else "none", score=score)
    design = M.make_design(data, EPS, effect=(effect != "none"), tail=0.0025, window_tail=0.005)
    alpha_grid = np.linspace(design.window[0], design.window[1], 61)
    main = M.four_role(data, design, seed, curves=M.boxcox(curve_grid, LAM),
                       grid_points=GRID_POINTS, alpha_grid=alpha_grid)
    d = lambda p: M.dollars(p, LAM)
    fares = data.fare
    row = {"score": score, "effect": effect,
           "phi": d(main["phi_avg"]), "cell": M.fare_cell(d(main["phi_avg"]), fares),
           "phi_pooled": d(main["phi_pooled"]),
           "rotations": [d(p) for p in main["phi_rotations"]],
           "own_cell": M.fare_cell(d(main["own_cutoff"]), fares),
           "at_boundary": main["at_boundary"], "local_maxima": main["local_maxima"],
           "gain_cents": 100 * main["gain_vs_deployed"], "beta2": main["beta2"],
           "window_share": main["window_share"],
           "effect_block": M.effect_diagnostics(data, design, seed),
           "tau": main["tau"].tolist(), "U": main["U"].tolist(),
           "eta_grid": alpha_grid.tolist(), "alpha": main["alpha"].tolist()}
    if boot:
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(data, design, seed, M.boxcox(curve_grid, LAM),
                                           alpha_grid)) as ex:
            draws = list(ex.map(_draw, range(boot)))
        phis = np.array([d(x[0]) for x in draws])
        sd = float(phis.std(ddof=1))
        U = np.array([x[2] for x in draws])
        A = np.array([x[3] for x in draws])
        # 90% pointwise bands: estimate +- 1.645 bootstrap sd, as for phi
        row["boot"] = {"B": boot, "sd": sd, "ci90": [row["phi"] - 1.645 * sd, row["phi"] + 1.645 * sd],
                       "draws": phis.round(3).tolist(),
                       "share_boundary": float(np.mean([x[1] for x in draws])),
                       "U_sd": U.std(0, ddof=1).tolist(), "alpha_sd": A.std(0, ddof=1).tolist()}
    row["seconds"] = time.time() - t0
    return row, data


LABEL_SCORE = {"linear": "distance", "distlog": "distance + log distance",
               "spline": "spline in log distance"}
LABEL_EFFECT = {"distance": "effect: distance", "controls": "effect: all controls",
                "none": "common slope"}


def _panel_grid(rows, path, xkey, ykey, sdkey, xlabel, ylabel, scale=1.0, mark_phi=False):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex=True, sharey=True)
    for i, score in enumerate(SCORES):
        for j, eff in enumerate(EFFECTS):
            ax = axes[i, j]
            r = next(r for r in rows if r["score"] == score and r["effect"] == eff)
            x = np.array(r[xkey])
            y = scale * np.array(r[ykey])
            if "boot" in r:
                sd = scale * np.array(r["boot"][sdkey])
                ax.fill_between(x, y - 1.645 * sd, y + 1.645 * sd, color="C0", alpha=0.25, lw=0)
            ax.plot(x, y, color="C0", lw=1.6)
            ax.axhline(0, color="k", lw=0.7)
            if mark_phi:
                ax.axvline(r["phi"], color="C3", ls="--", lw=1)
                ax.axvline(15, color="0.5", ls=":", lw=1)
                ax.text(0.97, 0.05, f"\${r['phi']:.2f}", transform=ax.transAxes,
                        ha="right", fontsize=8, color="C3")
            if i == 0:
                ax.set_title(LABEL_EFFECT[eff], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"score: {LABEL_SCORE[score]}\n{ylabel}", fontsize=8)
            if i == 2:
                ax.set_xlabel(xlabel, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def phi_distribution(rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = []
    for k, r in enumerate(rows):
        y = len(rows) - 1 - k
        labels.append(f"{LABEL_SCORE[r['score']]} | {LABEL_EFFECT[r['effect']]}")
        if "boot" in r:
            draws = np.array(r["boot"]["draws"])
            jitter = np.random.default_rng(k).uniform(-0.18, 0.18, len(draws))
            ax.scatter(draws, y + jitter, s=9, color="C0", alpha=0.6)
            lo, hi = r["boot"]["ci90"]
            ax.plot([lo, hi], [y, y], color="k", lw=1.2)
        ax.plot(r["phi"], y, "D", color="C3", ms=5)
    ax.axvline(15, color="0.5", ls=":", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=7.5)
    ax.set_xlabel("estimated cutoff \$ (red: estimate; black: 90% interval; blue: bootstrap draws)",
                  fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure(rows, grid, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    colors = {"linear": "C0", "distlog": "C2", "spline": "C1"}
    labels = {"linear": "distance", "distlog": "distance + log distance",
              "spline": "spline in log distance"}
    titles = {"distance": "effect linear in distance", "controls": "effect linear in all controls",
              "none": "common slope"}
    for ax, eff in zip(axes, EFFECTS):
        for r in rows:
            if r["effect"] == eff:
                ax.plot(grid, r["tau"], color=colors[r["score"]], lw=1.8,
                        label=f"score: {labels[r['score']]} (\\${r['phi']:.2f})")
                ax.axvline(r["phi"], color=colors[r["score"]], ls="--", lw=0.9)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(15, color="0.5", ls=":", lw=1.1)
        ax.set_title(titles[eff], fontsize=10)
        ax.set_xlabel("fare cutoff (\\$)")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    axes[0].set_ylabel("marginal effect (\\$ per ride)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    df = M.load_frame("full")
    grid = np.linspace(*M.POLICY_DOLLARS, 73)
    rows, indep = [], {}
    for score in SCORES:
        for effect in EFFECTS:
            row, data = run_one(df, score, effect, a.boot, a.workers, a.seed, grid)
            if score not in indep:
                indep[score] = independence(df, data)
            row["independence"] = indep[score]
            rows.append(row)
            b = row.get("boot", {})
            print(f"{score:8s} {effect:9s} phi {row['phi']:6.2f} cell {row['cell']:5.2f} "
                  f"pooled {row['phi_pooled']:6.2f} rot {min(row['rotations']):5.2f}-{max(row['rotations']):5.2f} "
                  f"own {row['own_cell']:5.2f} sd {b.get('sd', float('nan')):.2f} "
                  f"R2eta {indep[score]['R2_eta']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "summary.json", "w") as f:
        json.dump({"lambda": LAM, "eps": EPS, "fare_grid": grid.tolist(), "rows": rows}, f, indent=1)
    figure(rows, grid, a.out / "tau_by_spec.png")
    for r in rows:
        r["fare_grid"] = grid.tolist()
    _panel_grid(rows, a.out / "alpha_by_spec.png", "eta_grid", "alpha", "alpha_sd",
                r"latent type $\eta$ (\$)", r"$\hat\alpha(\eta)$ (\$)")
    _panel_grid(rows, a.out / "U_by_spec.png", "fare_grid", "U", "U_sd",
                "fare cutoff (\$)", "tips vs \$15 (cents/ride)", scale=100.0, mark_phi=True)
    phi_distribution(rows, a.out / "phi_draws.png")
    print(f"wrote {a.out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
