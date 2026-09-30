"""Knot-count check for the differing-slopes outcome splines (alpha and b).

For one dataset, refit the stacked outcome regression with several interior-knot
counts (ridge chosen by the same 80/20 holdout as the screen) and report, per count:
holdout MSE at the chosen ridge, alpha_hat(eta) on a grid, U_J(phi), the maximizer and
its treated share of the trim window. The trim window, index and densities do not
depend on the knot count.

    PYTHONPATH=. python experiments/scripts/ds_knots_check.py OUT_DIR NAME K1 [K2 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.methods.weighted_tails import uj_utility
from experiments.scripts.ds_curve_check import _setup
from experiments.scripts.taxi_differing_slopes import LAMBDA_GRID, _design, _solve


def main(out_dir: Path, name: str, knots):
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _setup(name)
    eta_grid = np.linspace(s["lo"], s["hi"], 300)
    pw = s["win"].mean()
    in_win = s["win"] > 0
    res, arrays = {"dataset": name, "window_eta_orig": sorted([s["sign"] * s["l0"], s["sign"] * s["u0"]]),
                   "rule_knots": s["kn"], "fits": {}}, {"eta_grid_orig": s["sign"] * eta_grid,
                                                        "phi": s["sign"] * s["grid"]}
    for kn in knots:
        info = _basis_params(kn, (s["lo"], s["hi"]))
        H, pen, nb, nx = _design(s["Qs"], s["X"], s["eta"], s["D"], info, True)
        tr = np.random.default_rng(0).random(len(s["Y"])) < 0.8
        best = (np.inf, None)
        for lam in LAMBDA_GRID:
            c = _solve(H[tr], s["Y"][tr], pen, lam)
            mse = float(np.mean((s["Y"][~tr] - H[~tr] @ c) ** 2))
            if mse < best[0]:
                best = (mse, lam)
        mse, lam = best
        c = _solve(H, s["Y"], pen, lam)
        ca, b2 = c[-nb:], c[1 + nx:1 + 2 * nx]
        a = _eval_basis(s["eta"], info) @ ca
        U = np.array([uj_utility(p, s["eta"], s["win"], a, b2, s["tails"]) for p in s["grid"]]) / pw
        j = int(np.argmax(U))
        share = float(np.mean(s["tails"].survival(s["grid"][j] - s["eta"][in_win]))
                      / max(s["tails"].survival(-np.inf), 1e-12))
        dep = float(uj_utility(s["thr_s"], s["eta"], s["win"], a, b2, s["tails"]) / pw)
        res["fits"][str(kn)] = {"holdout_mse": mse, "lambda": lam, "phi_opt": float(s["sign"] * s["grid"][j]),
                                "treated_share": share, "U_opt": float(U[j]), "U_deployed": dep,
                                "U_ends": [float(U[0]), float(U[-1])],
                                "beta2_first2": [float(b2[0]), float(b2[5]) if len(b2) > 5 else float("nan")]}
        arrays[f"alpha_{kn}"] = _eval_basis(eta_grid, info) @ ca
        arrays[f"U_{kn}"] = U
        f = res["fits"][str(kn)]
        print(f"knots {kn:3d}: holdout MSE {mse:.6f} (lambda {lam}) phi {f['phi_opt']:.3f} "
              f"share {share:.2f} U_opt-U_dep {f['U_opt'] - dep:.4f}", flush=True)
    (out_dir / f"{name}_knots.json").write_text(json.dumps(res, indent=2) + "\n")
    np.savez(out_dir / f"{name}_knots.npz", **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], [int(k) for k in sys.argv[3:]])
