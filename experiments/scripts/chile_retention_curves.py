"""Estimated curves for the Chile retention next-grade-level outcome, and their figure.

Compute (same steps as ``screen_flatness.py``):
  * U_J(phi) per trim-window student for the alpha-only and differing-slopes fits, 2017
    cohort, plus the DS curve for the 2016 and 2015 cohorts and the 2017 donut;
  * mean fitted effect by Q value (0.1 grid) for both 2017 fits;
  * binned first stage and outcome means by Q, and local Wald estimates at the 4.5 and
    5.0 cutoffs (per retained student; delta-method SE ignoring the first-stage error).

    PYTHONPATH=. python experiments/scripts/chile_retention_curves.py compute OUT.npz
    PYTHONPATH=. python experiments/scripts/chile_retention_curves.py plot OUT.npz FIG.png
"""
from __future__ import annotations

import sys

import numpy as np

SPECS = [  # (label, dataset, models)
    ("2017", "chile_retention_gpa_next_level", ("alpha_only", "differing_slopes")),
    ("2017 donut", "chile_retention_gpa_next_level_donut", ("differing_slopes",)),
    ("2016", "chile_retention_gpa_next_level_2016", ("differing_slopes",)),
    ("2015", "chile_retention_gpa_next_level_2015", ("differing_slopes",)),
]
CUTS = (4.45, 4.95)
BANDS = (0.5, 0.3)


def compute(out: str) -> None:
    from experiments.scripts import differing_slopes_screen as S
    from experiments.methods.perfrdd import _basis_params
    from experiments.methods.weighted_tails import fit_weighted_tails, uj_utility
    from experiments.scripts.taxi_differing_slopes import fit_effect
    from experiments.scripts.local_rd_checks import rd
    from experiments.datasets.chile_retention.adapter import build_continuous

    res = {}
    for label, name, models in SPECS:
        loader, _ = S.DATASETS[name]
        Q, X, Y, thr = loader()
        ok = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
        Q, X, Y = Q[ok], S._standardize(X[ok]), Y[ok]
        Qs, thr_s = -Q, -thr                                  # below-cutoff design, mirrored
        Xd = np.column_stack((np.ones(len(Qs)), X))
        gamma, *_ = np.linalg.lstsq(Xd, Qs, rcond=None)
        eta = Qs - Xd @ gamma
        T = Qs - eta
        l0, u0 = thr_s - np.quantile(T, 1 - S.EPS), thr_s - np.quantile(T, S.EPS)
        lo, hi = np.percentile(eta, 0.5), np.percentile(eta, 99.5)
        l0, u0 = max(min(l0, u0), lo), min(max(l0, u0), hi)
        D = (Qs >= thr_s).astype(float)
        kn = max(4, int(round(int(D.sum()) ** (1 / 5)))) + 1
        info = _basis_params(kn, (lo, hi))
        grid = np.linspace(*np.quantile(Qs, [0.005, 0.995]), S.N_GRID)
        tails = fit_weighted_tails(T, X, tuple(np.quantile(T, S.DENSITY_QUANTILES)))
        win = ((eta >= l0) & (eta <= u0)).astype(float)
        pw = win.mean()
        rng = np.random.default_rng(0)
        for model in ("alpha_only", "differing_slopes"):
            a, b, _ = fit_effect(Qs, X, Y, eta, D, info, model == "differing_slopes", rng)
            if model not in models:
                continue
            U = np.array([uj_utility(p, eta, win, a, b, tails) for p in grid]) / pw
            key = f"{label}|{model}"
            res[f"{key}|phi"] = -grid                             # back to Q units
            res[f"{key}|U"] = U
            res[f"{key}|U_deployed"] = np.array(uj_utility(thr_s, eta, win, a, b, tails) / pw)
            if label == "2017":
                eff = a + X @ b
                qv = np.round(Q, 1)
                values = np.round(np.arange(3.0, 7.01, 0.1), 1)
                res[f"{key}|effect_q"] = values
                res[f"{key}|effect_mean"] = np.array([eff[qv == v].mean() if (qv == v).any() else np.nan
                                                      for v in values])
        print(f"[done] {label}", flush=True)

    df = build_continuous(2017)
    df = df[df.gpa_next_level.notna()]
    values = np.round(np.arange(3.0, 7.01, 0.1), 1)
    qv = df.Q.round(1).to_numpy()
    res["bins|q"] = values
    res["bins|n"] = np.array([(qv == v).sum() for v in values])
    res["bins|retained"] = np.array([df.retained[qv == v].mean() if (qv == v).any() else np.nan for v in values])
    res["bins|y"] = np.array([df.gpa_next_level[qv == v].mean() if (qv == v).any() else np.nan for v in values])
    q = df.Q.to_numpy()
    for cut, h in zip(CUTS, BANDS):
        fs = rd(q, df.retained.to_numpy(), cut, h, True)
        itt = rd(q, df.gpa_next_level.to_numpy(), cut, h, True)
        res[f"wald|{cut}"] = np.array([itt["est"] / fs["est"], itt["se"] / fs["est"],
                                       fs["est"], itt["est"]])
    np.savez(out, **res)
    print(f"[wrote] {out}")


def plot(npz: str, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.load(npz)
    ink, ink2, muted, grid_c, base = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    ds, ao = "#2a78d6", "#eb6834"                                # categorical slots 1, 2
    cohort = {"2015": "#86b6ef", "2016": "#3987e5", "2017": "#1c5cab", "2017 donut": "#1c5cab"}
    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.edgecolor": base, "axes.labelcolor": ink2, "xtick.color": muted,
                         "ytick.color": muted, "axes.titlecolor": ink, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "axes.titlelocation": "left"})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.6), facecolor="#fcfcfb")
    for a in ax.flat:
        a.set_facecolor("#fcfcfb")
        a.grid(True, color=grid_c, linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for c in (4.45, 4.95):
            a.axvline(c, color=base, linewidth=1, linestyle=(0, (3, 3)), zorder=0)

    # A. first stage
    a = ax[0, 0]
    keep = r["bins|n"] > 500
    a.plot(r["bins|q"][keep], r["bins|retained"][keep], color=ink2, linewidth=2, marker="o", markersize=4)
    a.set_title("A. Share held back, by this year's average")
    a.set_ylabel("share retained")
    a.set_xlim(3.0, 6.5)
    a.text(4.40, 0.35, "cutoff 4.5\n(1 failed subject)", color=muted, fontsize=8, va="top", ha="right")
    a.text(5.02, 0.35, "cutoff 5.0\n(2 failed subjects)", color=muted, fontsize=8, va="top")

    # B. outcome
    a = ax[0, 1]
    a.plot(r["bins|q"][keep], r["bins|y"][keep], color=ink2, linewidth=2, marker="o", markersize=4)
    a.set_title("B. Average in the next grade level (first time reached)")
    a.set_ylabel("grade average (1-7)")
    a.set_xlim(3.0, 6.5)

    # C. utility curves, 2017
    a = ax[1, 0]
    for model, color, name in (("alpha_only", ao, "original (alpha only)"),
                               ("differing_slopes", ds, "differing slopes")):
        phi, U = r[f"2017|{model}|phi"], r[f"2017|{model}|U"]
        base_u = U[np.argmin(phi)]                        # treat no one (lowest cutoff)
        a.plot(phi, U - base_u, color=color, linewidth=2, label=name)
        j = int(np.argmax(U))
        a.plot(phi[j], U[j] - base_u, "o", color=color, markersize=9,
               markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=5)
        a.annotate(f"max at {phi[j]:.2f}", (phi[j], U[j] - base_u), textcoords="offset points",
                   xytext=(8, 10) if model == "differing_slopes" else (-10, 10),
                   ha="left" if model == "differing_slopes" else "right", fontsize=9, color=ink)
        a.plot(4.45, float(r[f'2017|{model}|U_deployed']) - base_u, "s", color=color, markersize=7,
               markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=5)
    a.set_title("C. Estimated utility of each retention cutoff (2017, c = 0)")
    a.set_xlabel("candidate cutoff: hold back if average <= cutoff")
    a.set_ylabel("gain vs. lowest candidate cutoff (4.3)\n(grade points per window student)")
    a.set_xlim(4.2, 7.0)
    a.legend(frameon=False, loc="upper left")
    a.text(4.5, a.get_ylim()[0] + 0.02 * np.ptp(a.get_ylim()), "squares = deployed cutoff",
           color=muted, fontsize=8)

    # D. DS utility by cohort
    a = ax[1, 1]
    for label, ls in (("2015", "-"), ("2016", "-"), ("2017", "-"), ("2017 donut", (0, (4, 2)))):
        phi, U = r[f"{label}|differing_slopes|phi"], r[f"{label}|differing_slopes|U"]
        base_u = U[np.argmin(phi)]
        a.plot(phi, U - base_u, color=cohort[label], linewidth=2, linestyle=ls,
               label=f"{label} ({phi[int(np.argmax(U))]:.2f})")
        j = int(np.argmax(U))
        a.plot(phi[j], U[j] - base_u, "o", color=cohort[label], markersize=7,
               markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=5)
    a.set_title("D. Differing slopes: same curve across cohorts")
    a.set_xlabel("candidate cutoff: hold back if average <= cutoff")
    a.set_ylabel("gain vs. lowest candidate cutoff")
    a.set_xlim(4.2, 7.0)
    a.set_ylim(bottom=-0.01)
    a.legend(frameon=False, loc="lower right", title="cohort (optimum)", title_fontsize=9)

    fig.suptitle("Chile grade retention: outcome = average in the next grade level",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=ink)
    fig.text(0.01, 0.005, "Source: MINEDUC open student-performance files, 2014-2019, linked by MRUN. "
             "Utility is the trim-window U_J per window student; window = students whose "
             "covariates put them near the cutoff (2-4% of all).", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(fig_path, dpi=160, facecolor=fig.get_facecolor())

    # Effect-by-Q figure (validation)
    fig2, b = plt.subplots(figsize=(8.5, 5), facecolor="#fcfcfb")
    b.set_facecolor("#fcfcfb")
    b.grid(True, color=grid_c, linewidth=0.6)
    b.set_axisbelow(True)
    for side in ("top", "right"):
        b.spines[side].set_visible(False)
    fs45 = float(r["wald|4.45"][2])
    for model, color, name in (("alpha_only", ao, "original (alpha only)"),
                               ("differing_slopes", ds, "differing slopes")):
        qv, m = r[f"2017|{model}|effect_q"], r[f"2017|{model}|effect_mean"]
        keep2 = (qv >= 3.6) & (qv <= 6.2) & np.isfinite(m)
        b.plot(qv[keep2], m[keep2] / fs45, color=color, linewidth=2, label=name)
    for cut in (4.45, 4.95):
        w, se, *_ = r[f"wald|{cut}"]
        b.errorbar(cut, w, yerr=1.96 * se, fmt="o", color=ink, markersize=8, capsize=4,
                   linewidth=1.5, zorder=5)
        b.annotate(f"local RD at {cut + 0.05:.1f}\n{w:+.2f} [{w - 1.96 * se:.2f}, {w + 1.96 * se:.2f}]",
                   (cut, w), textcoords="offset points", xytext=(-12, -28), ha="right",
                   fontsize=9, color=ink)
    b.axhline(0, color=base, linewidth=1)
    b.set_title("Fitted effect of being held back, by this year's average (2017)", loc="left")
    b.set_xlabel("this year's average (Q)")
    b.set_ylabel("effect on next-grade average\nper held-back student (grade points)")
    b.legend(frameon=False, loc="upper right")
    b.text(3.62, b.get_ylim()[0] + 0.03 * np.ptp(b.get_ylim()),
           f"Model lines: mean fitted eligibility effect at each Q, divided by the 4.5 first stage ({fs45:.2f}).\n"
           "Black points: local Wald estimates with 95% intervals (first-stage error ignored).",
           fontsize=8, color=muted)
    fig2.tight_layout()
    fig2.savefig(fig_path.replace(".png", "_effects.png"), dpi=160, facecolor=fig2.get_facecolor())
    print(f"[wrote] {fig_path} and {fig_path.replace('.png', '_effects.png')}")


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute(sys.argv[2])
    else:
        plot(sys.argv[2], sys.argv[3])
