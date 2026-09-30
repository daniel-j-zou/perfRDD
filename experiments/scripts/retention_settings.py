"""Chile retention (next-level average): differing slopes in three samples with fixed
hyperparameters (eps 0.10, 8 interior knots, ridge 0.1, eta support 0.5-99.5%, c = 0).

Samples: all grades (grades 2-8 and secondary 1-3), primary grades 2-6, grades 2-4.
For each: maximizer, treated share of the trim window, utility gains, beta2 for prior
average and grade level, window share, first stage and local ITT at 4.5 (bandwidth 0.5)
against the model's mean fitted effect at Q = 4.4/4.5, and the window decomposition of the
fitted effect into a(eta) and X'beta2 by Q. Also draws a 3x2 figure of a(eta) and U_J.

    PYTHONPATH=. python experiments/scripts/retention_settings.py OUT.json FIG.png
    PYTHONPATH=. python experiments/scripts/retention_settings.py --plot OUT.json FIG.png
"""
from __future__ import annotations

import json
import sys

import numpy as np

from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.scripts.ds_curve_check import _curve, _fit, _setup
from experiments.scripts.local_rd_checks import rd

SETTINGS = [("All grades", "chile_retention_gpa_next_level"),
            ("Grades 2-6", "chile_retention_gpa_next_level_grades2to6"),
            ("Grades 2-4", "chile_retention_gpa_next_level_grades2to4")]
KNOTS, RIDGE = 8, 0.1


def fit_setting(name: str) -> dict:
    s = _setup(name)
    s["info"] = _basis_params(KNOTS, (s["lo"], s["hi"]))
    ca, b2, _ = _fit(s["Qs"], s["X"], s["Y"], s["eta"], s["D"], s["info"], True, None, lam=RIDGE)
    U, j, share, dep = _curve(s, ca, b2)
    sg = s["sign"]
    a = _eval_basis(s["eta"], s["info"]) @ ca
    xb = s["X"] @ b2
    q = np.round(s["Q"], 1)
    w = s["win"] > 0
    near = np.isin(q, (4.4, 4.5))
    # Actual retention (for the first stage) from the same sample; _setup drops no rows
    # for these outcomes, which the length check confirms.
    from experiments.datasets.chile_retention import adapter
    sample = getattr(adapter, "load_" + name.replace("chile_retention_", ""))()
    retained = np.asarray(sample.extras["retained"], float)
    fs = rd(s["Q"], retained, 4.45, 0.5, True) if len(retained) == len(s["Q"]) else None
    itt = rd(s["Q"], s["Y"], 4.45, 0.5, True)
    decomp = []
    for v in np.round(np.arange(4.0, 6.01, 0.2), 1):
        m = w & (q == v)
        if m.sum() > 50:
            decomp.append({"Q": float(v), "n": int(m.sum()), "a": float(a[m].mean()),
                           "xb2": float(xb[m].mean()), "effect": float((a[m] + xb[m]).mean())})
    eg = np.linspace(s["lo"], s["hi"], 300)
    phi = sg * s["grid"]
    return {
        "n": int(len(s["Q"])), "eligible": int((q <= 4.4).sum()), "window_share": float(s["win"].mean()),
        "window": sorted([sg * s["l0"], sg * s["u0"]]),
        "phi_opt": float(phi[j]), "treated_share": share,
        "gain_vs_deployed": float(U[j] - dep), "gain_vs_treat_all": float(U[j] - U[int(np.argmax(phi))]),
        "beta2_prior": float(b2[0]), "beta2_grade": float(b2[5]),
        "first_stage_45": fs["est"] if fs else None, "local_itt_45": itt["est"], "local_itt_45_se": itt["se"],
        "model_itt_45": float((a + xb)[near].mean()),
        "decomposition": decomp,
        "curves": {"eta": (sg * eg).tolist(), "alpha": (_eval_basis(eg, s["info"]) @ ca).tolist(),
                   "phi": phi.tolist(), "U": U.tolist(), "U_deployed": dep},
    }


def plot(res: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, muted, grid_c, base, ds, band = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6", "#cde2fb"
    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10.5,
                         "axes.edgecolor": base, "axes.labelcolor": ink2, "xtick.color": muted,
                         "ytick.color": muted, "axes.titlecolor": ink, "axes.titlesize": 11,
                         "axes.titlelocation": "left"})
    fig, ax = plt.subplots(3, 2, figsize=(12, 11), facecolor="#fcfcfb")
    for i, (lab, _) in enumerate(SETTINGS):
        r = res[lab]
        c = r["curves"]
        for a in ax[i]:
            a.set_facecolor("#fcfcfb")
            a.grid(True, color=grid_c, lw=0.6)
            a.set_axisbelow(True)
            for sd in ("top", "right"):
                a.spines[sd].set_visible(False)
        a = ax[i, 0]
        a.axvspan(*r["window"], color=band, alpha=0.7, lw=0, label="trim window")
        eta, alpha = np.asarray(c["eta"]), np.asarray(c["alpha"])
        o = np.argsort(eta)
        a.plot(eta[o], alpha[o], color=ds, lw=2.2, label="a(eta) at mean X")
        a.axhline(0, color=base, lw=1)
        a.set_title(f"{lab}: fitted effect by latent type")
        a.set_xlabel("eta = average minus its prediction")
        a.set_ylabel("effect (grade points)")
        if i == 0:
            a.legend(frameon=False, fontsize=9)
        a = ax[i, 1]
        phi, U = np.asarray(c["phi"]), np.asarray(c["U"])
        ref = U[np.argmin(phi)]
        j = int(np.argmax(U))
        a.plot(phi, U - ref, color=ds, lw=2.2)
        a.plot(phi[j], U[j] - ref, "o", color=ds, ms=9, mec="#fcfcfb", mew=2, zorder=5)
        a.annotate(f"maximizer {phi[j]:.2f}", (phi[j], U[j] - ref), textcoords="offset points",
                   xytext=(0, 10), ha="center", fontsize=9.5, color=ink)
        a.set_ylim(top=1.15 * (U[j] - ref))
        a.plot(4.45, c["U_deployed"] - ref, "s", color=ink2, ms=7, mec="#fcfcfb", mew=1.5, zorder=5)
        a.annotate("deployed rule", (4.45, c["U_deployed"] - ref), textcoords="offset points",
                   xytext=(8, -14), fontsize=9, color=ink2)
        a.set_title(f"{lab}: utility (c = 0; window = {100 * r['window_share']:.1f}% of students)")
        a.set_xlabel("candidate cutoff: eligible if average <= cutoff")
        a.set_ylabel("U per window student,\nrelative to lowest cutoff")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    if sys.argv[1] == "--plot":        # redraw from saved results
        with open(sys.argv[2]) as f:
            plot(json.load(f), sys.argv[3])
        sys.exit()
    res = {}
    for lab, name in SETTINGS:
        res[lab] = fit_setting(name)
        r = res[lab]
        print(f"{lab}: n {r['n']:,} eligible {r['eligible']:,} window {r['window_share']:.4f} phi {r['phi_opt']:.3f} "
              f"share {r['treated_share']:.2f} gain dep {r['gain_vs_deployed']:.4f} all {r['gain_vs_treat_all']:.4f} "
              f"b2 {r['beta2_prior']:+.3f}/{r['beta2_grade']:+.3f} FS {r['first_stage_45']} "
              f"ITT local {r['local_itt_45']:+.3f} model {r['model_itt_45']:+.3f}", flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump(res, f)
    plot(res, sys.argv[2])
    print(f"[wrote] {sys.argv[1]} {sys.argv[2]}")
