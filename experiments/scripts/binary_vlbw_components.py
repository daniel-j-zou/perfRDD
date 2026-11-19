"""VLBW: the fitted nonparametric components b(eta) and alpha(eta) of the logistic outcome model.

For manuscript/binary_outcome_vlbw.tex (the GLM note), proof-of-concept section (Claude,
2026-11-19, at the author's request). Pooled 1995-2002 births of 500-2500 g, the fit of
binary_outcome_routes.fit_logit without a penalty, seed 1, the four rotations of the four-role
split. For each rotation the score regression is refitted on its own fold, so the residual scale
moves slightly between rotations; the curves are drawn on each rotation's residual.

The fit uses the mirrored score Q = -weight (the rule treats below the line). The figure is drawn on
the weight scale, against the weight residual r = -eta (weight minus its prediction from X), so
b(r) and alpha(r) here are the fitted b(-r) and alpha(-r) (2026-11-19, at the author's request).

Writes components.json and binary_vlbw_components.png to the folder given; with a third argument
``plot`` it redraws the figure from an existing components.json without refitting:

    python -m experiments.scripts.binary_vlbw_components experiments/runs/binary_vlbw_components pooled
    python -m experiments.scripts.binary_vlbw_components experiments/runs/binary_vlbw_components pooled plot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.scripts import binary_outcome_routes as R  # noqa: E402
from experiments.scripts import taxi_four_role as M  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "pooled"
SEED = 1
PLOT_ONLY = len(sys.argv) > 3 and sys.argv[3] == "plot"


def fit_components():
    data, design, meta = R.build_vlbw(VARIANT, eps=0.1)
    blocks = np.array_split(np.random.default_rng(SEED).permutation(len(data.Q)), 4)
    grid = np.linspace(design.J[0], design.J[1], 201)
    rows = []
    for k in range(4):
        folds = {r: blocks[(j + k) % 4] for j, r in enumerate(M.ROLES)}
        g = np.linalg.lstsq(M._xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
        eta = data.Q - M._xt(data.Z) @ g
        T_idx = M._xt(data.Z[folds["index"]]) @ g
        l_hat, u_hat = R._window(data, design, eta, folds, T_idx)
        fit = R.fit_logit(data, eta, folds["outcome"], design)
        b = fit.c(grid, 0)
        a = fit.c(grid, 1) - b
        rows.append({"rotation": k, "window": [l_hat, u_hat], "b": b.tolist(), "alpha": a.tolist(),
                     "beta2": np.asarray(fit.beta2).tolist(), "n_spline": int(len(fit.w0))})
        print(f"rotation {k}: window [{l_hat:.1f}, {u_hat:.1f}] g, beta2 = {np.round(fit.beta2, 3).tolist()}, "
              f"{len(fit.w0)} spline functions per arm", flush=True)

    summary = {"variant": VARIANT, "seed": SEED, "n": int(len(data.Q)), "J": list(design.J),
               "eta_sd": float(np.std(data.Q - M._xt(data.Z) @ np.linalg.lstsq(M._xt(data.Z), data.Q, rcond=None)[0])),
               "grid": grid.tolist(), "rotations": rows}
    (OUT / "components.json").write_text(json.dumps(summary))
    return summary


summary = (json.loads((OUT / "components.json").read_text()) if PLOT_ONLY else fit_components())
grid = -np.array(summary["grid"])[::-1]                    # weight residual r = -eta, increasing
rows = summary["rotations"]

B = np.array([r["b"][::-1] for r in rows]); A = np.array([r["alpha"][::-1] for r in rows])
win = -np.mean([r["window"] for r in rows], axis=0)[::-1]       # window on the weight residual
inside = (grid >= win[0]) & (grid <= win[1])
for name, M_ in (("b", B), ("alpha", A)):
    m = M_.mean(0)
    print(f"{name}: over the window {m[inside][0]:.2f} at {grid[inside][0]:.0f} g to {m[inside][-1]:.2f} at "
          f"{grid[inside][-1]:.0f} g; range [{m[inside].min():.2f}, {m[inside].max():.2f}]; largest spread across "
          f"rotations {np.max(M_.max(0)[inside] - M_.min(0)[inside]):.2f}")
expit = R.expit
s0, s1 = expit(B.mean(0)), expit(B.mean(0) + A.mean(0))
print(f"survival at average covariates over the window: without care {s0[inside].min():.3f}-{s0[inside].max():.3f}, "
      f"with care {s1[inside].min():.3f}-{s1[inside].max():.3f}; gain {100 * (s1 - s0)[inside].min():.1f}-"
      f"{100 * (s1 - s0)[inside].max():.1f} points")

BG = "#fcfcfb"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#888",
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "savefig.facecolor": BG})
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
for ax, M_, title in ((axs[0], B, r"$\hat b$: log-odds of survival without the care"),
                      (axs[1], A, r"$\hat\alpha$: log-odds ratio of the care")):
    for r in M_:
        ax.plot(grid, r, color="#9bb8e0", lw=0.8)
    ax.plot(grid, M_.mean(0), color="#2a78d6", lw=1.6)
    ax.set_title(title, fontsize=9.5)
ax = axs[2]
ax.plot(grid, s0, color="#eb6834", lw=1.6, label="without the care")
ax.plot(grid, s1, color="#2c9a5f", lw=1.6, label="with the care")
ax.set_title("survival at average covariates", fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
for ax in axs:
    ax.axvspan(win[0], win[1], color="#f1efe8", zorder=0)
    ax.set_xlabel("weight residual: weight minus its prediction (g)")
fig.tight_layout()
fig.savefig(OUT / "binary_vlbw_components.png", dpi=150)
print("wrote", OUT / "binary_vlbw_components.png")
