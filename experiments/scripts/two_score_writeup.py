"""Figures and numbers for the two-score settings write-up (continuous outcomes only).

For each setting the estimator is run as in ``two_score_apply.py`` (c = 0, eps = 0.05,
cross-validated penalty, the same seed and four rotations). Per rotation the script keeps
alpha_hat on a grid of eta, pi0 on that grid, and U_hat as a function of the rule
parameters; the figures show the averages over the four rotations:

  (a) alpha_hat(eta1, eta2), the effect at the covariate mean, shown only where U counts
      (eps <= pi0 <= 1 - eps) and where there are data, with the density of eta_hat;
  (b) U_hat(phi) with the deployed weights;
  (c) U_hat(lam, phi) with chosen weights;
  (d) U_hat(phi1, phi2) for the both-cutoffs rule.

Axes are in the original score units (for NC low-performing, where designation is the
low side, the cutoffs are shown as "designated if Q1 < phi1 and Q2 < phi2").

Run from ``code/``:
    python -m experiments.scripts.two_score_writeup
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from experiments.methods.two_score import (
    _value_from_grid, best_halfplane, best_two_cutoffs, make_folds, prob_in_rule,
    rotation_pairs)
from experiments.scripts.two_score_apply import problem

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "writeups" / "two_score_settings"
SEED = 20261028                          # two_score_apply's default seed

SETTINGS = {
    "tx_school_ratings": {"labels": ("M = max(achievement, progress)", "closing the gaps"),
                          "short": ("M", "CtG")},
    "fl_school_grades": {"labels": ("achievement mean", "learning-gains mean"),
                         "short": ("Ach", "Gains")},
    "nc_school_grades": {"labels": ("achievement", "growth"), "short": ("A", "G")},
    "nc_low_performing": {"labels": ("score 0.8A + 0.2G", "growth"), "short": ("S", "G")},
    "nyc_progress_reports_ab": {"labels": ("environment + performance (%)", "progress (%)"),
                                "short": ("EP", "P")},
}


def _r2(Z, X):
    A = np.column_stack([np.ones(len(X)), X - X.mean(0)])
    T = A @ np.linalg.lstsq(A, Z, rcond=None)[0]
    E = Z - T
    return 1 - E.var(0) / Z.var(0), T, E


def run(name: str) -> dict:
    s, prob, classes = problem(name)
    sg = np.asarray(prob.signs, float)
    dep = prob.deployed
    r2, T_all, E_all = _r2(prob.Z, prob.X)
    eta = E_all * sg                                 # original units
    lo, hi = np.quantile(eta, 0.01, axis=0), np.quantile(eta, 0.99, axis=0)
    g = [np.linspace(lo[k], hi[k], 121) for k in (0, 1)]
    G0, G1 = np.meshgrid(*g, indexing="ij")
    grid = np.column_stack([G0.ravel(), G1.ravel()])
    halfplane = dep.kind == "halfplane"
    if halfplane:
        lam0 = dep.theta[0]
        phi_box = classes["fixed_weights"]["box"]
        phis = np.linspace(*phi_box, 201)
        lam_range = classes["weights"]["lam_range"]
        lams = np.linspace(*lam_range, 37)
    both_box = classes["both"]["box"]
    b0 = np.linspace(*both_box[0], 121)
    b1 = np.linspace(*both_box[1], 121)

    A, P, Uf, Ul, Ub, Udep, est, infos, xb = [], [], [], [], [], [], {}, [], []
    for folds in make_folds(len(prob.Y), SEED):
        pairs, fit, info = rotation_pairs(prob, folds)
        infos.append(info)
        A.append(fit.alpha(grid * sg).reshape(G0.shape))
        xb.append(float(np.std(prob.Xc @ fit.beta2)))
        P.append(prob_in_rule(info["deployed"], grid * sg, pairs.T).reshape(G0.shape))
        Udep.append(pairs.value(info["deployed"]) if halfplane else _value_from_grid(pairs, dep))
        if halfplane:
            x, S = pairs.curve1(lam0)
            Uf.append(np.interp(phis, x, S))
            Ul.append(np.array([np.interp(phis, *pairs.curve1(l)) for l in lams]))
        (n0, n1), S2 = pairs.surface2()
        f = RegularGridInterpolator((n0, n1), S2, bounds_error=False, fill_value=None)
        B0, B1 = np.meshgrid(b0, b1, indexing="ij")
        Ub.append(f(np.column_stack([B0.ravel(), B1.ravel()])).reshape(B0.shape))
        for label, spec in classes.items():
            if spec["kind"] == "halfplane":
                r = best_halfplane(pairs, spec.get("lam_range", (0.0, 1.0)), spec["box"],
                                   lam_fixed=spec.get("lam_fixed"))
            else:
                r = best_two_cutoffs(pairs, spec["kind"], spec["box"])
            est.setdefault(label, []).append(list(map(float, r["theta"])) + [float(r["U"]), bool(r["edge"])])

    fit_alpha = np.mean(A, 0)
    pi0 = np.mean(P, 0)
    res = {
        "name": name, "n": int(len(prob.Y)), "p": int(prob.X.shape[1]),
        "treated_share": float(prob.D.mean()), "signs": list(map(int, sg)),
        "deployed": {"kind": dep.kind, "theta": list(map(float, dep.theta))},
        "R2": r2.round(3).tolist(), "sd_T": T_all.std(0).round(3).tolist(),
        "sd_eta": E_all.std(0).round(3).tolist(),
        "corr_T": float(np.corrcoef(T_all.T)[0, 1]), "corr_eta": float(np.corrcoef(E_all.T)[0, 1]),
        "trim_share": float(np.mean([i["trim_share"] for i in infos])),
        "pen": [float(i["pen"]) for i in infos], "m": int(infos[0]["m"]),
        "sd_xbeta2": float(np.mean(xb)),
        "U_deployed": float(np.mean(Udep)), "Y_mean": float(prob.Y.mean()), "Y_sd": float(prob.Y.std()),
        "estimates": {k: {"theta": np.mean([v[:-2] for v in vs], 0).tolist(),
                          "theta_rot": [v[:-2] for v in vs],
                          "U": float(np.mean([v[-2] for v in vs])),
                          "edge_rotations": int(sum(v[-1] for v in vs))} for k, vs in est.items()},
        "alpha_trimmed": {},
    }
    # where U counts and there are data: the trim, and a non-empty 40 x 40 histogram cell
    H, e0, e1 = np.histogram2d(eta[:, 0], eta[:, 1], bins=40, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    k0 = np.clip(np.searchsorted(e0, G0, side="right") - 1, 0, 39)
    k1 = np.clip(np.searchsorted(e1, G1, side="right") - 1, 0, 39)
    inside = (pi0 >= 0.05) & (pi0 <= 0.95) & (H[k0, k1] > 0)
    if inside.any():
        a = fit_alpha[inside]
        res["alpha_trimmed"] = {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
                                "share_positive": float((a > 0).mean())}
    arrays = {"g0": g[0], "g1": g[1], "alpha": fit_alpha, "pi0": pi0, "eta": eta, "inside": inside,
              "b0": b0, "b1": b1, "Ub": np.mean(Ub, 0)}
    if halfplane:
        arrays.update({"phis": phis, "Uf": np.mean(Uf, 0), "lams": lams, "Ul": np.mean(Ul, 0)})
    return res, arrays


def figure(res: dict, arr: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8})
    name = res["name"]
    lab, short = SETTINGS[name]["labels"], SETTINGS[name]["short"]
    sg = np.array(res["signs"])
    halfplane = res["deployed"]["kind"] == "halfplane"
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6)) if halfplane else plt.subplots(1, 2, figsize=(7.0, 2.9))
    axs = np.atleast_1d(axs).ravel()

    # (a) alpha_hat
    ax = axs[0]
    a = np.where(arr["inside"], arr["alpha"], np.nan)
    vmax = np.nanquantile(np.abs(a), 0.98)       # a few sparse edge cells would set the scale
    ax.set_facecolor("0.9")
    im = ax.pcolormesh(arr["g0"], arr["g1"], a.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    if np.nanmin(a) < 0 < np.nanmax(a):
        ax.contour(arr["g0"], arr["g1"], a.T, levels=[0], colors="k", linewidths=0.8)
    ax.contour(arr["g0"], arr["g1"], arr["pi0"].T, levels=[0.05, 0.95], colors="0.2",
               linewidths=0.8, linestyles="--")
    H, e0, e1 = np.histogram2d(arr["eta"][:, 0], arr["eta"][:, 1], bins=40,
                               range=[[arr["g0"][0], arr["g0"][-1]], [arr["g1"][0], arr["g1"][-1]]])
    c0, c1 = 0.5 * (e0[1:] + e0[:-1]), 0.5 * (e1[1:] + e1[:-1])
    lv = np.quantile(H[H > 0], [0.5, 0.8, 0.95])
    ax.contour(c0, c1, H.T, levels=np.unique(lv), colors="0.55", linewidths=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_xlabel(f"$\\eta_1$ ({short[0]})")
    ax.set_ylabel(f"$\\eta_2$ ({short[1]})")
    ax.set_title("(a) $\\hat\\alpha(\\eta)$ where $\\hat U$ counts; solid $\\hat\\alpha=0$, dashed trim")

    if halfplane:
        lam0, phi0 = res["deployed"]["theta"]
        ef = res["estimates"]["fixed_weights"]["theta"]
        ew = res["estimates"]["weights"]["theta"]
        # (b) U(phi), deployed weights
        ax = axs[1]
        ax.plot(arr["phis"], arr["Uf"], "k-", lw=1)
        ax.axvline(phi0, color="C3", ls="--", lw=0.9, label=f"deployed {phi0:g}")
        ax.axvline(ef[1], color="C0", lw=0.9, label=f"estimate {ef[1]:.1f}")
        ax.set_xlabel(f"cutoff $\\phi$ on {lam0:.2f} {short[0]} + {1 - lam0:.2f} {short[1]}")
        ax.set_ylabel("$\\hat U$ (outcome units)")
        ax.set_title("(b) $\\hat U(\\phi)$, deployed weights")
        ax.legend(frameon=False)
        # (c) U(lam, phi)
        ax = axs[2]
        im = ax.pcolormesh(arr["phis"], arr["lams"], arr["Ul"], cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.plot(phi0, lam0, "r^", ms=5, label="deployed")
        ax.plot(ew[1], ew[0], "wo", ms=5, mec="k", label="estimate")
        ax.set_xlabel("cutoff $\\phi$")
        ax.set_ylabel(f"weight $\\lambda$ on {short[0]}")
        ax.set_title(f"(c) $\\hat U(\\lambda,\\phi)$: $\\lambda$ {short[0]} + $(1-\\lambda)$ {short[1]} $\\geq\\phi$")
        ax.legend(frameon=False, loc="lower left")
        axb = axs[3]
    else:
        axb = axs[1]
    # (d) U(phi1, phi2), both cutoffs
    b0, b1 = arr["b0"] * sg[0], arr["b1"] * sg[1]
    im = axb.pcolormesh(b0, b1, arr["Ub"].T, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=axb, fraction=0.046, pad=0.03)
    eb = np.array(res["estimates"]["both"]["theta"]) * sg
    if res["deployed"]["kind"] == "both":
        db = np.array(res["deployed"]["theta"]) * sg
        axb.plot(*db, "r^", ms=5, label="deployed")
    axb.plot(*eb, "wo", ms=5, mec="k", label="estimate")
    if sg[0] < 0:
        axb.set_title("(b) $\\hat U$, designated if $Q_1<\\phi_1$ and $Q_2<\\phi_2$")
    else:
        axb.set_title("(d) $\\hat U$, treated if $Q_1\\geq\\phi_1$ and $Q_2\\geq\\phi_2$")
    axb.set_xlabel(f"$\\phi_1$ ({lab[0]})")
    axb.set_ylabel(f"$\\phi_2$ ({lab[1]})")
    axb.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    out = {}
    for name in SETTINGS:
        res, arr = run(name)
        figure(res, arr, OUT / "figures" / f"{name}.pdf")
        out[name] = res
        print(name, json.dumps({k: v["theta"] for k, v in res["estimates"].items()}))
    (OUT / "numbers.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
