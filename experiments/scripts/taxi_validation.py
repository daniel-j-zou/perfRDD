"""Taxi Section 6: validation of the main specification's outcome model (R2-E1, daytime
sample) against what the data identify without the model.

Referee-style checks (author, 2026-11-19: "Validation analyses"):

1. Out-of-sample fit by fare cell. The paper's four-role folds are reused: in rotation k
   the score is fitted on the gamma fold, the outcome model on the outcome fold (rides
   with eta-hat in J), exactly as in ``taxi_four_role.outcome_fit`` but keeping every
   coefficient, and tips are predicted on the utility fold. Every ride is in one utility
   fold, so every ride gets one out-of-sample prediction of its tip, of both arm means
   and of its effect alpha-hat(eta-hat) + beta2-hat d.
2. The jump at $15, the one quantity identified without the model: local linear
   regression on each side (uniform kernel) at several bandwidths, with
   heteroskedasticity-robust standard errors and standard errors clustered by fare cell
   (Haggag and Paci's choice), against the model's mean effect by fare cell.
3. Hold-outs. The rides of a band of fares are removed from the outcome fit and their
   tips predicted: treated $15.30-$16.90 and $15.30-$19.70 (the treated arm extrapolated
   down by about $2 and $4.5, the direction the estimate needs), untreated
   $12.90-$14.90 and $10.50-$14.90 (the untreated arm extrapolated up).
4. A simple local comparator in the spirit of Marinescu et al. (2022): local linear fits
   on each side, extrapolated, and the fare where the fitted effect crosses zero, at
   several bandwidths; outcome in dollars (theirs is tip as a percentage of the fare).

Run (from code/):
    python -m experiments.scripts.taxi_validation                      # main specification
    python -m experiments.scripts.taxi_validation --sample daytime_logdist
Output (git-ignored): experiments/runs/taxi_validation[_<sample>]/summary.json and figures.
"""
from __future__ import annotations

import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from experiments.methods.perfrdd import _basis_params
from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_section6 as S

OUT = M.ROOT / "runs" / "taxi_validation"
SPEC, SEED = "R2-E1", 1
BANDWIDTHS = (2.0, 3.0, 4.5, 6.0, 10.0)
HOLDOUTS = {
    "treated_15.30-16.90": (15.29, 16.91),
    "treated_15.30-19.70": (15.29, 19.71),
    "untreated_12.90-14.90": (12.89, 14.91),
    "untreated_10.50-14.90": (10.49, 14.91),
}


def fit_full(data, eta, idx, design, drop=None):
    """``taxi_four_role.outcome_fit`` with every coefficient kept. ``drop`` is a boolean
    mask over all rides; those rides are left out of the fit (hold-outs)."""
    e = eta[idx]
    keep = idx[(e >= design.J[0]) & (e <= design.J[1])]
    if drop is not None:
        keep = keep[~drop[keep]]
    D = data.D[keep]
    n1 = int(D.sum())
    info = _basis_params(max(3, int(round(min(n1, len(keep) - n1) ** (11 / 60)))), design.J)
    B = BSpline.design_matrix(np.clip(eta[keep], info["lo"], info["hi"]), info["t"],
                              info["degree"]).toarray()
    A = np.column_stack(((1 - D)[:, None] * B, D[:, None] * B, data.baseline[keep],
                         D[:, None] * data.X[keep]))
    coef = np.linalg.lstsq(A, data.Y[keep], rcond=None)[0]
    nb, nz = B.shape[1], data.baseline.shape[1]
    return {"info": info, "w0": coef[:nb], "w1": coef[nb:2 * nb],
            "b1": coef[2 * nb:2 * nb + nz], "b2": coef[2 * nb + nz:]}


def predict(fit, data, eta, idx):
    info = fit["info"]
    B = BSpline.design_matrix(np.clip(eta[idx], info["lo"], info["hi"]), info["t"],
                              info["degree"]).toarray()
    base = data.baseline[idx] @ fit["b1"]
    mu0 = B @ fit["w0"] + base
    mu1 = B @ fit["w1"] + base + data.X[idx] @ fit["b2"]
    return mu0, mu1


def out_of_sample(data, design, drop=None):
    """One out-of-sample prediction per ride from the four rotations."""
    n = len(data.Q)
    blocks = np.array_split(np.random.default_rng(SEED).permutation(n), 4)
    mu0, mu1, eta_all, inJ = (np.full(n, np.nan) for _ in range(4))
    for k in range(4):
        folds = {r: blocks[(j + k) % 4] for j, r in enumerate(M.ROLES)}
        g = np.linalg.lstsq(M._xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
        eta = data.Q - M._xt(data.Z) @ g
        fit = fit_full(data, eta, folds["outcome"], design, drop)
        U = folds["utility"]
        mu0[U], mu1[U] = predict(fit, data, eta, U)
        eta_all[U] = eta[U]
        inJ[U] = (eta[U] >= design.J[0]) & (eta[U] <= design.J[1])
    return mu0, mu1, eta_all, inJ.astype(bool)


def local_linear(fare, y, h, cell_se=True):
    k = np.abs(fare - M.FARE0) <= h
    x, D = fare[k] - M.FARE0, (fare[k] >= M.FARE0).astype(float)
    X = np.column_stack((np.ones(k.sum()), D, x, D * x))
    b = np.linalg.lstsq(X, y[k], rcond=None)[0]
    r = y[k] - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * r ** 2) @ X @ XtXi
    out = {"h": h, "n": int(k.sum()), "jump": float(b[1]), "slope_diff": float(b[3]),
           "se_jump_hc": float(np.sqrt(V[1, 1])), "se_slope_hc": float(np.sqrt(V[3, 3]))}
    if cell_se:                                   # clustered by fare cell
        cells = np.round(fare[k], 2)
        meat = np.zeros((4, 4))
        for c in np.unique(cells):
            s = (X[cells == c] * r[cells == c][:, None]).sum(0)
            meat += np.outer(s, s)
        Vc = XtXi @ meat @ XtXi
        out.update(se_jump_cell=float(np.sqrt(Vc[0 + 1, 1])), se_slope_cell=float(np.sqrt(Vc[3, 3])),
                   n_cells=int(len(np.unique(cells))))
    # Marinescu-style extrapolated crossing: effect(x) = jump + slope_diff (x - 15) = 0
    out["zero_crossing"] = (float(M.FARE0 - b[1] / b[3]) if b[3] > 0 else None)
    return out


def by_cell(frame, cols):
    g = frame.groupby("cell")
    res = g[cols].mean()
    res["n"] = g.size()
    return res


def redraw(sample="daytime"):
    """Redraw the figure from a saved summary.json."""
    global OUT
    OUT = M.ROOT / "runs" / ("taxi_validation" if sample == "daytime" else f"taxi_validation_{sample}")
    figures(json.loads((OUT / "summary.json").read_text()))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", default="daytime", choices=["daytime", "daytime_logdist"])
    a = ap.parse_args(argv)
    global OUT
    OUT = M.ROOT / "runs" / ("taxi_validation" if a.sample == "daytime" else f"taxi_validation_{a.sample}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = M.load_frame(S.SAMPLES[a.sample]["frame"])
    data, design = S.build(df, SPEC, sample=a.sample)
    fare, y, D = data.fare, data.Y, data.D
    sd_d = float(df["Trip_Distance"].std(ddof=0))

    # 0. the copied outcome fit reproduces the estimator's own (rotation 0)
    blocks = np.array_split(np.random.default_rng(SEED).permutation(len(data.Q)), 4)
    folds = {r: blocks[j] for j, r in enumerate(M.ROLES)}
    g = np.linalg.lstsq(M._xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
    e0 = data.Q - M._xt(data.Z) @ g
    ref, mine = M.outcome_fit(data, e0, folds["outcome"], design), fit_full(data, e0, folds["outcome"], design)
    gap = max(np.abs(ref.w0 - mine["w0"]).max(), np.abs(ref.w1 - mine["w1"]).max(),
              np.abs(np.asarray(ref.beta2) - mine["b2"]).max())
    print(f"[check] copied outcome fit vs estimator's: max coefficient gap {gap:.2e}", flush=True)
    assert gap < 1e-6

    # 1. out-of-sample fit by fare cell
    mu0, mu1, eta, inJ = out_of_sample(data, design)
    fr = pd.DataFrame({"cell": np.round(fare, 2), "y": y, "D": D, "mu0": mu0, "mu1": mu1,
                       "inJ": inJ})
    fr["pred"] = np.where(fr.D == 1, fr.mu1, fr.mu0)
    fr["effect"] = fr.mu1 - fr.mu0
    fr["resid"] = fr.y - fr.pred
    cells = by_cell(fr[fr.inJ], ["y", "pred", "resid", "effect", "mu0", "mu1"])
    near = cells.loc[(cells.index >= 6.0) & (cells.index <= 30.0)]
    rmse_cell = float(np.sqrt(np.average(near.resid ** 2, weights=near.n)))
    print(f"[fit] rides in J {int(fr.inJ.sum())} of {len(fr)}; weighted RMS of cell residuals "
          f"$6-$30: {rmse_cell:.4f}; max |cell residual| {near.resid.abs().max():.3f}", flush=True)

    # 2. the jump at $15
    rd = [local_linear(fare, y, h) for h in BANDWIDTHS]
    for r in rd:
        print(f"[RD] h={r['h']}: jump {r['jump']:.3f} (se hc {r['se_jump_hc']:.3f}, cell "
              f"{r['se_jump_cell']:.3f}), slope diff {r['slope_diff']:.4f} (cell se "
              f"{r['se_slope_cell']:.4f}), crossing {r['zero_crossing']}", flush=True)
    eff = cells["effect"]
    model_at_15 = float(np.interp(M.FARE0, eff.index, eff.values))
    model_slope = {}
    for h in (2.0, 4.5):
        w = (eff.index >= M.FARE0 - h) & (eff.index <= M.FARE0 + h)
        model_slope[str(h)] = float(np.polyfit(eff.index[w], eff.values[w], 1,
                                               w=np.sqrt(cells.n[w]))[0])
    top = json.loads((S.SAMPLES[a.sample]["out"] / "summary.json").read_text())
    tau15 = float(np.interp(M.FARE0, top["fare_grid"], top["specs"][SPEC]["tau"]))
    # where the model's mean effect by cell crosses zero
    sgn = np.sign(eff.values)
    cross = [float(eff.index[i + 1]) for i in range(len(sgn) - 1)
             if sgn[i] < 0 <= sgn[i + 1] and 6 <= eff.index[i] <= 24]
    print(f"[model] mean effect at $15 (cells interpolated) {model_at_15:.3f}; paper tau(15) "
          f"{tau15:.3f}; effect slope per $ within 2 / 4.5: {model_slope}; effect turns "
          f"positive at cells {cross}", flush=True)

    # 3. hold-outs
    hold = {}
    for name, (lo, hi) in HOLDOUTS.items():
        band = (fare >= lo) & (fare <= hi)
        h0, h1, _, hJ = out_of_sample(data, design, drop=band)
        pred = np.where(D == 1, h1, h0)
        m = band & hJ
        err = pd.DataFrame({"cell": np.round(fare[m], 2), "e": y[m] - pred[m]})
        per = err.groupby("cell").e.agg(["mean", "size"])
        hold[name] = {"mean_error": float(err.e.mean()), "n": int(m.sum()),
                      "se": float(err.e.std(ddof=1) / np.sqrt(m.sum())),
                      "by_cell": {f"{c:.2f}": [float(v), int(n)] for c, (v, n) in per.iterrows()}}
        print(f"[hold-out {name}] mean error (observed - predicted) {hold[name]['mean_error']:+.3f} "
              f"(se {hold[name]['se']:.3f}); by cell " +
              ", ".join(f"{c}: {v[0]:+.3f}" for c, v in hold[name]["by_cell"].items()), flush=True)

    # 4. outputs
    res = {"spec": SPEC, "sample": a.sample, "seed": SEED, "n": int(len(fr)), "n_inJ": int(fr.inJ.sum()),
           "rms_cell_resid_6_30": rmse_cell,
           "cells": {f"{c:.2f}": {k: float(v) for k, v in row.items()} for c, row in cells.iterrows()},
           "rd": rd, "model_effect_at_15": model_at_15, "paper_tau_15": tau15,
           "model_effect_slope_per_dollar": model_slope, "model_effect_turns_positive": cross,
           "holdouts": hold, "sd_distance": sd_d}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    figures(res)
    print(f"wrote {OUT}", flush=True)


def figures(res):
    c = pd.DataFrame(res["cells"]).T
    c.index = c.index.astype(float)
    c = c.loc[(c.index >= 5.0) & (c.index <= 30.0)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    t, u = c.index >= M.FARE0, c.index < M.FARE0
    ax[0].plot(c.index, c.y, "o", ms=3, color="k", label="observed mean tip")
    ax[0].plot(c.index[u], c.pred[u], color="C0", label="fitted, untreated arm")
    ax[0].plot(c.index[t], c.pred[t], color="C1", label="fitted, treated arm")
    ax[0].plot(c.index[u], c.mu1[u], color="C1", ls="--", label="treated arm, extrapolated")
    ax[0].plot(c.index[t], c.mu0[t], color="C0", ls="--", label="untreated arm, extrapolated")
    ax[0].axvline(M.FARE0, color="0.5", ls=":")
    ax[0].set_xlabel("fare cell ($)"); ax[0].set_ylabel("tip ($)")
    ax[0].set_title("Out-of-sample fit by fare cell", fontsize=10); ax[0].legend(fontsize=7)
    ax[1].bar(c.index, c.resid, width=0.3, color=np.where(t, "C1", "C0"))
    for name, col in (("treated_15.30-16.90", "C3"), ("untreated_12.90-14.90", "C2")):
        h = res["holdouts"][name]["by_cell"]
        xs = np.array([float(k) for k in h]); ys = np.array([v[0] for v in h.values()])
        side, band = name.split("_")
        lo, hi = band.split("-")
        ax[1].plot(xs, ys, "o-", color=col, ms=3, label=f"{side} \\${lo}-\\${hi} held out")
    ax[1].axhline(0, color="k", lw=0.7); ax[1].axvline(M.FARE0, color="0.5", ls=":")
    ax[1].set_xlabel("fare cell ($)"); ax[1].set_ylabel("observed - predicted ($)")
    ax[1].set_title("Residuals by cell; hold-out errors", fontsize=10); ax[1].legend(fontsize=7)
    ax[2].plot(c.index, c.effect, "o-", ms=3, color="C2", label="model: mean effect of the rides in the cell")
    x = np.linspace(9, 21, 50)
    for r, ls in zip([r for r in res["rd"] if r["h"] in (2.0, 4.5, 10.0)], ("-", "--", ":")):
        ax[2].plot(x, r["jump"] + r["slope_diff"] * (x - M.FARE0), color="k", ls=ls, lw=1,
                   label=f"local linear, h=\\${r['h']:g}: jump {r['jump']:.2f}")
    ax[2].plot([M.FARE0], [res["paper_tau_15"]], "s", color="C3",
               label=f"model, $\\hat\\tau_\\epsilon(15)$ = {res['paper_tau_15']:.2f}")
    ax[2].axhline(0, color="k", lw=0.7); ax[2].axvline(M.FARE0, color="0.5", ls=":")
    ax[2].set_xlim(9, 21); ax[2].set_ylim(-0.6, 0.9)
    ax[2].set_xlabel("fare ($)"); ax[2].set_ylabel("effect of the percentage menu ($)")
    ax[2].set_title("Effect near $15: model against local linear", fontsize=10); ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "taxi_validation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
