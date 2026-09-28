"""Flatness diagnostics for datasets in the differing-slopes screen.

Repeats the steps of ``differing_slopes_screen.screen_dataset`` and reports, for the
alpha-only and differing-slopes fits, the fitted effect in the trim window (mean, share
negative, 10th/90th percentiles) and U_J per window student at the grid optimum, at
both ends of the candidate range (treat all / treat none of the window), and at the
deployed cutoff. The gap between the optimum and the better end shows how flat the
utility is.

    PYTHONPATH=. python experiments/scripts/screen_flatness.py OUT.json NAME [NAME ...]
"""
import json, sys
import numpy as np
from experiments.scripts import differing_slopes_screen as S
from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails, uj_utility
from experiments.scripts.taxi_differing_slopes import fit_effect

out = {}
for name in sys.argv[2:]:
    loader, direction = S.DATASETS[name]
    Q, X, Y, thr = loader()
    ok = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y = Q[ok], S._standardize(X[ok]), Y[ok]
    sign = 1.0 if direction == "above" else -1.0
    Qs, thr_s = sign * Q, sign * thr
    Xd = np.column_stack((np.ones(len(Qs)), X))
    gamma, *_ = np.linalg.lstsq(Xd, Qs, rcond=None)
    eta = Qs - Xd @ gamma; T = Qs - eta
    l0 = thr_s - np.quantile(T, 1 - S.EPS); u0 = thr_s - np.quantile(T, S.EPS)
    lo, hi = np.percentile(eta, 0.5), np.percentile(eta, 99.5)
    l0, u0 = max(min(l0, u0), lo), min(max(l0, u0), hi)
    D = (Qs >= thr_s).astype(float)
    kn = max(4, int(round(int(D.sum()) ** (1 / 5)))) + 1
    info = _basis_params(kn, (lo, hi))
    grid = np.linspace(*np.quantile(Qs, [0.005, 0.995]), S.N_GRID)
    tails = fit_weighted_tails(T, X, tuple(np.quantile(T, S.DENSITY_QUANTILES)))
    win = ((eta >= l0) & (eta <= u0)).astype(float); iw = win > 0
    rng = np.random.default_rng(0)
    res = {"y_mean_window": float(Y[iw].mean()), "y_sd_window": float(Y[iw].std()),
           "Q_window_quantiles": [float(sign * v) for v in np.quantile(Qs[iw], [0.1, 0.5, 0.9])]}
    for label, inter in (("alpha_only", False), ("differing_slopes", True)):
        a, b, lam = fit_effect(Qs, X, Y, eta, D, info, inter, rng)
        eff = a + X @ b
        U = np.array([uj_utility(p, eta, win, a, b, tails) for p in grid])
        j = int(np.argmax(U)); pw = win.mean()
        res[label] = {
            "mean_effect_window": float(eff[iw].mean()),
            "share_effect_negative_window": float((eff[iw] < 0).mean()),
            "effect_q10_q90_window": [float(v) for v in np.quantile(eff[iw], [0.1, 0.9])],
            # utilities per window student, relative to treating no one in the window
            "U_opt": float(U[j] / pw),
            "U_grid_start": float(U[0] / pw), "U_grid_end": float(U[-1] / pw),
            "U_deployed": float(uj_utility(thr_s, eta, win, a, b, tails) / pw),
            "phi_opt": float(sign * grid[j]),
        }
    out[name] = res
    print(name, json.dumps(res, indent=1), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)
