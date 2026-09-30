"""Sanity check of the differing-slopes curves for candidate datasets.

For each dataset (names from ``differing_slopes_screen.DATASETS``), using the screen's
pipeline:
  * alpha_hat(eta) on a grid over the outer region, for the alpha-only and DS fits (the
    DS a(eta) is the effect at the covariate means, since X is standardized);
  * U_J(phi) per window student for both fits, with the maximizer and the deployed cutoff;
  * DS maximizer under alternative tuning: trim eps 0.05 / 0.20, knots -2 / +2, and fixed
    ridge penalties 0.1 / 1 / 10 / 30 instead of the cross-validated one.
Below-cutoff designs are mirrored for fitting; eta is reported in the original direction
(Q minus its prediction).

    PYTHONPATH=. python experiments/scripts/ds_curve_check.py compute OUT_DIR NAME [NAME ...]
    PYTHONPATH=. python experiments/scripts/ds_curve_check.py plot OUT_DIR NAME [NAME ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.scripts import differing_slopes_screen as S
from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.methods.weighted_tails import fit_weighted_tails, uj_utility
from experiments.scripts.taxi_differing_slopes import LAMBDA_GRID, _design, _solve


def _fit(Qs, X, Y, eta, D, info, interact, rng, lam=None):
    """fit_effect, also returning the coefficients (same CV draw when lam is None)."""
    H, pen, nb, nx = _design(Qs, X, eta, D, info, interact)
    if lam is None:
        tr = rng.random(len(Y)) < 0.8
        best = (np.inf, None)
        for cand in LAMBDA_GRID:
            c = _solve(H[tr], Y[tr], pen, cand)
            mse = np.mean((Y[~tr] - H[~tr] @ c) ** 2)
            if mse < best[0]:
                best = (mse, cand)
        lam = best[1]
    c = _solve(H, Y, pen, lam)
    beta2 = c[1 + nx:1 + 2 * nx] if interact else np.zeros(nx)
    return c[-nb:], beta2, lam


def _setup(name, eps=S.EPS, dknots=0):
    loader, direction = S.DATASETS[name]
    Q, X, Y, thr = loader()
    ok = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y = Q[ok], S._standardize(X[ok]), Y[ok]
    sign = 1.0 if direction == "above" else -1.0
    Qs, thr_s = sign * Q, sign * thr
    Xd = np.column_stack((np.ones(len(Qs)), X))
    gamma, *_ = np.linalg.lstsq(Xd, Qs, rcond=None)
    eta = Qs - Xd @ gamma
    T = Qs - eta
    l0, u0 = thr_s - np.quantile(T, 1 - eps), thr_s - np.quantile(T, eps)
    lo, hi = np.percentile(eta, 0.5), np.percentile(eta, 99.5)
    l0, u0 = max(min(l0, u0), lo), min(max(l0, u0), hi)
    D = (Qs >= thr_s).astype(float)
    kn = max(4, int(round(int(D.sum()) ** (1 / 5)))) + 1 + dknots
    info = _basis_params(kn, (lo, hi))
    grid = np.linspace(*np.quantile(Qs, [0.005, 0.995]), S.N_GRID)
    tails = fit_weighted_tails(T, X, tuple(np.quantile(T, S.DENSITY_QUANTILES)))
    win = ((eta >= l0) & (eta <= u0)).astype(float)
    return dict(Q=Q, X=X, Y=Y, thr=thr, sign=sign, Qs=Qs, thr_s=thr_s, eta=eta, T=T,
                l0=l0, u0=u0, lo=lo, hi=hi, D=D, kn=kn, info=info, grid=grid,
                tails=tails, win=win)


def _curve(s, coef_a, beta2):
    a = _eval_basis(s["eta"], s["info"]) @ coef_a
    pw = s["win"].mean()
    U = np.array([uj_utility(p, s["eta"], s["win"], a, beta2, s["tails"]) for p in s["grid"]]) / pw
    j = int(np.argmax(U))
    in_win = s["win"] > 0
    share = float(np.mean(s["tails"].survival(s["grid"][j] - s["eta"][in_win]))
                  / max(s["tails"].survival(-np.inf), 1e-12))
    dep = float(uj_utility(s["thr_s"], s["eta"], s["win"], a, beta2, s["tails"]) / pw)
    return U, j, share, dep


def compute(out_dir: Path, names):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        s = _setup(name)
        rng = np.random.default_rng(0)
        res, arrays = {"dataset": name, "cutoff": s["thr"], "knots": s["kn"],
                       "window_eta_orig": sorted([s["sign"] * s["l0"], s["sign"] * s["u0"]]),
                       "window_share": float(s["win"].mean())}, {}
        eta_grid = np.linspace(s["lo"], s["hi"], 200)
        arrays["eta_grid_orig"] = s["sign"] * eta_grid
        arrays["phi"] = s["sign"] * s["grid"]
        for model, interact in (("alpha_only", False), ("differing_slopes", True)):
            ca, b2, lam = _fit(s["Qs"], s["X"], s["Y"], s["eta"], s["D"], s["info"], interact, rng)
            U, j, share, dep = _curve(s, ca, b2)
            arrays[f"{model}_alpha"] = _eval_basis(eta_grid, s["info"]) @ ca
            arrays[f"{model}_U"] = U
            xb = s["X"] @ b2
            res[model] = {"phi_opt": float(s["sign"] * s["grid"][j]), "treated_share": share,
                          "boundary": bool(j in (0, len(U) - 1) or share > 0.99 or share < 0.01),
                          "U_opt": float(U[j]), "U_deployed": dep,
                          "U_ends": [float(U[0]), float(U[-1])], "lambda": lam,
                          "beta2": [float(v) for v in b2],
                          "sd_xbeta2_window": float(np.std(xb[s["win"] > 0]))}
            if interact:
                base = dict(ca=ca, b2=b2, lam=lam)
        sens = {}
        for label, kw, lam in (("eps 0.05", {"eps": 0.05}, None), ("eps 0.20", {"eps": 0.20}, None),
                               ("knots -2", {"dknots": -2}, None), ("knots +2", {"dknots": 2}, None),
                               ("ridge 0.1", {}, 0.1), ("ridge 1", {}, 1.0),
                               ("ridge 10", {}, 10.0), ("ridge 30", {}, 30.0)):
            s2 = _setup(name, **kw) if kw else s
            ca, b2, used = _fit(s2["Qs"], s2["X"], s2["Y"], s2["eta"], s2["D"], s2["info"], True,
                                np.random.default_rng(0), lam)
            U, j, share, dep = _curve(s2, ca, b2)
            sens[label] = {"phi_opt": float(s2["sign"] * s2["grid"][j]), "treated_share": share,
                           "gain_over_best_end": float(U[j] - max(U[0], U[-1])),
                           "gain_over_deployed": float(U[j] - dep), "lambda": used}
            print(f"  {name} {label}: phi {sens[label]['phi_opt']:.3f} share {share:.2f}", flush=True)
        res["sensitivity"] = sens
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        np.savez(out_dir / f"{name}.npz", **arrays)
        print(f"[done] {name}: DS phi {res['differing_slopes']['phi_opt']:.3f}", flush=True)


def plot(out_dir: Path, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid_c, base = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    ds, ao, band = "#2a78d6", "#eb6834", "#cde2fb"
    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9.5,
                         "axes.edgecolor": base, "axes.labelcolor": ink2, "xtick.color": muted,
                         "ytick.color": muted, "axes.titlecolor": ink, "axes.titlesize": 10.5,
                         "axes.titlelocation": "left"})
    labels = {"chile_retention_gpa_next_level": ("Retention: average in next grade level", "grade points"),
              "chile_admission_sel_2024": ("Admission: selectivity of 2024 program", "PAES points"),
              "chile_admission_sel_2025": ("Admission: selectivity of 2025 program", "PAES points"),
              "chile_admission_acred_2024": ("Admission: accreditation years, 2024", "years")}
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 3.6 * len(names)), facecolor="#fcfcfb",
                             gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        r = json.loads((out_dir / f"{name}.json").read_text())
        a = np.load(out_dir / f"{name}.npz")
        title, unit = labels.get(name, (name, "units of Y"))
        for ax in axes[row]:
            ax.set_facecolor("#fcfcfb")
            ax.grid(True, color=grid_c, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        # alpha(eta)
        ax = axes[row, 0]
        lo_w, hi_w = r["window_eta_orig"]
        ax.axvspan(lo_w, hi_w, color=band, alpha=0.6, lw=0, label="trim window")
        order = np.argsort(a["eta_grid_orig"])
        for model, color, lab in (("alpha_only", ao, "original: alpha(eta)"),
                                  ("differing_slopes", ds, "DS: a(eta) at mean X")):
            ax.plot(a["eta_grid_orig"][order], a[f"{model}_alpha"][order], color=color, lw=2, label=lab)
        ax.axhline(0, color=base, lw=1)
        ax.set_title(f"{title}\nfitted effect by latent type")
        ax.set_xlabel("eta = Q minus predicted Q")
        ax.set_ylabel(f"effect ({unit})")
        if row == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="best")
        # U(phi)
        ax = axes[row, 1]
        phi = a["phi"]
        for model, color, lab in (("alpha_only", ao, "original"), ("differing_slopes", ds, "DS")):
            U = a[f"{model}_U"]
            ref = U[np.argmin(phi)] if r["cutoff"] < np.median(phi) or True else 0.0
            ax.plot(phi, U - ref, color=color, lw=2, label=lab)
            j = int(np.argmax(U))
            ax.plot(phi[j], U[j] - ref, "o", color=color, ms=8, mec="#fcfcfb", mew=1.5, zorder=5)
            dep = r[model]["U_deployed"] - ref
            ax.plot(r["cutoff"], dep, "s", color=color, ms=6, mec="#fcfcfb", mew=1.2, zorder=5)
        ax.axvline(r["cutoff"], color=base, lw=1, ls=(0, (3, 3)))
        m = r["differing_slopes"]
        ax.set_title(f"utility per window student (c = 0)\nDS max {m['phi_opt']:.2f}"
                     f" [{100 * m['treated_share']:.0f}% of window]; deployed {r['cutoff']:g}")
        ax.set_xlabel("candidate cutoff (Q units)")
        ax.set_ylabel(f"U relative to lowest cutoff ({unit})")
        if row == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="best")
        # sensitivity
        ax = axes[row, 2]
        sens = r["sensitivity"]
        keys = ["baseline"] + list(sens)
        vals = [m["phi_opt"]] + [sens[k]["phi_opt"] for k in sens]
        ax.scatter(vals, range(len(keys)), color=[ink] + [ds] * len(sens), s=30, zorder=5)
        ax.axvline(r["cutoff"], color=base, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=8.5)
        ax.invert_yaxis()
        lo_p, hi_p = float(np.min(phi)), float(np.max(phi))
        ax.set_xlim(lo_p, hi_p)
        ax.set_title("DS maximizer under alternative tuning")
        ax.set_xlabel("maximizer (Q units)")
    fig.tight_layout()
    path = out_dir / "ds_curve_check.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"[wrote] {path}")


if __name__ == "__main__":
    mode, out, names = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    (compute if mode == "compute" else plot)(out, names)
