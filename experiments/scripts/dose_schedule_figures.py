"""Figures for the dose-schedule applications: alpha (dose effect by latent eta) and U (welfare of a
one-parameter schedule family), plus the transported effect by score bin.

    python -m experiments.scripts.dose_schedule_figures ui taxi2 lending

alpha_a(e) = f_hat(e, a) + (a - a_ref) mean(tau_hat'Z): the effect of dose a against the reference
dose for units with latent part e. Solid where e lies inside the 1-99% eta range of the units given
a dose near a, dashed outside (extrapolation). Thin lines are the three role rotations.

U(theta) - U(theta_0): the change in the mean outcome when every unit gets the dose of schedule
d_theta instead of the status quo, averaged over the evaluation fold of each rotation; the lower
panel gives the share of units whose new dose is supported for their eta.

Figures go to experiments/datasets/simulations/dose_schedule/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.methods import dose_schedule as ds  # noqa: E402
from experiments.scripts import dose_schedule_apply as app  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "datasets" / "simulations" / "dose_schedule"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # reference palette, categorical slots 1-3
GRID = "#d9d8d2"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2})


def rotation_fits(prob, seed=0, cluster=None):
    n = len(prob.Y)
    fold = ds.make_folds(n, seed, cluster)
    n_fit = (prob.in_range & (fold == 1)).sum()
    K, k = max(3, int(round(n_fit ** 0.2))), max(3, int(round(n_fit ** (1 / 7))))
    out = []
    for r0, r1, r2 in ds.ROTATIONS:
        _, eta, _ = ds.first_stage(prob.Q, prob.X, fold == r0)
        out.append((ds.outcome_fit(prob, eta, fold == r1, K, k), eta, fold == r2))
    return out


def alpha_panel(ax, prob, fits, doses, unit, label_fmt):
    a_ref = prob.basis.a_ref
    eta_all = fits[0][1]
    e_grid = np.linspace(*np.quantile(eta_all[prob.in_range], [0.025, 0.975]), 200)
    width = 0.05 * (prob.A.max() - prob.A.min())
    ylo, yhi = 0.0, 0.0
    for c, a in zip(COLORS, doses):
        near = np.abs(prob.A - a) <= (width if prob.basis.levels is None else 1e-9)
        lo, hi = np.quantile(eta_all[near], [0.025, 0.975]) if near.sum() > 30 else (np.nan, np.nan)
        inside = (e_grid >= lo) & (e_grid <= hi)
        curves = []
        for fit, eta, ev in fits:
            zbar = float(np.mean(prob.Z[ev] @ fit.tau)) if prob.Z.shape[1] else 0.0
            curves.append(fit.f(e_grid, np.full(len(e_grid), a)) + (a - a_ref) * zbar)
            ax.plot(e_grid, curves[-1], color=c, lw=0.6, alpha=0.5)
        m = np.mean(curves, axis=0)
        if inside.any():
            sup_vals = np.array(curves)[:, inside]
            ylo, yhi = min(ylo, sup_vals.min()), max(yhi, sup_vals.max())
        ax.plot(e_grid[inside], m[inside], color=c, label=label_fmt(a))
        ax.plot(e_grid, np.where(inside, np.nan, m), color=c, ls="--", lw=1.2)
    ax.axhline(0, color="#8a8984", lw=0.8)
    pad = 0.15 * (yhi - ylo + 1e-12)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xlabel("latent part eta (solid: supported, dashed: extrapolated)")
    ax.set_ylabel(f"effect vs reference dose ({unit})")
    ax.legend(frameon=False, fontsize=8)


def psi_panel(ax, prob, cfg, bins, unit, label_fmt):
    est = ds.estimate(prob, seed=0, a_grid=cfg["a_grid"], score_edges=cfg["edges"], n_bins=cfg["n_bins"] or 10,
                      cluster=cfg["cluster"])
    _, eta, _ = ds.first_stage(prob.Q, prob.X, np.ones(len(prob.Q), bool))
    cov = ds.coverage(eta, prob.A, prob.Q, est["edges"], est["a_grid"], prob.basis.a_ref,
                      levels=cfg["levels"], band=0.04)
    j0 = int(np.argmin(np.abs(est["a_grid"] - prob.basis.a_ref)))
    sup = []
    for c, (i, name) in zip(COLORS, bins):
        eff = est["psi"][i] - est["psi"][i, j0]
        ok = cov[i] >= 0.9
        sup.extend(eff[ok])
        ax.plot(est["a_grid"][ok], eff[ok], "o-", color=c, ms=5, label=name)
        ax.plot(est["a_grid"][~ok], eff[~ok], "o", mfc="white", color=c, ms=5)
    lo, hi = min(sup + [0.0]), max(sup + [0.0])
    ax.set_ylim(lo - 0.25 * (hi - lo + 1e-12), hi + 0.25 * (hi - lo + 1e-12))
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_xlabel(label_fmt)
    ax.set_ylabel(f"psi(q, a) - psi(q, a_ref) ({unit})")
    ax.legend(frameon=False, fontsize=8, title="score bin (hollow: <90% supported)", title_fontsize=7)
    return est


def u_panels(ax, ax2, prob, cfg, family, thetas, theta0, unit, xlabel, weight=None):
    vals, sup = ds.family_value(prob, family, thetas, seed=0, cluster=cfg["cluster"], weight=weight)
    j0 = int(np.argmin(np.abs(thetas - theta0)))
    vals = vals - vals[:, [j0]]
    for v in vals:
        ax.plot(thetas, v, color=COLORS[0], lw=0.6, alpha=0.5)
    ax.plot(thetas, vals.mean(0), color=COLORS[0])
    ax.axvline(theta0, color="#8a8984", lw=0.8, ls=":")
    ax.set_ylabel(f"U(theta) - U(status quo) ({unit})")
    ax2.plot(thetas, sup.mean(0), color="#5f5e5a", lw=1.5)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("supported share")
    ax2.set_xlabel(xlabel)
    print(f"  U [{xlabel}]: theta, mean, rotation min, max, supported share")
    for t, m, lo, hi, sh in zip(thetas, vals.mean(0), vals.min(0), vals.max(0), sup.mean(0)):
        print(f"    {t:8.4g} {m:10.4g} {lo:10.4g} {hi:10.4g} {sh:6.3f}", flush=True)
    return vals, sup


def ui(outcome):
    args = argparse.Namespace(x="linear", outcome=outcome, seed=0)
    cfg = app.ui(args)
    prob = cfg["prob"]
    unit = "weeks" if outcome == "weeks" else "$"
    fits = rotation_fits(prob)
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.35)
    alpha_panel(fig.add_subplot(gs[:, 0]), prob, fits, [0.45, 0.6, 0.75], unit, lambda a: f"benefit = {a:g} x cap")
    edges = cfg["edges"]
    bins = [(1, "s/kink %.2f-%.2f" % tuple(np.exp(edges[1:3]))), (4, "s/kink %.2f-%.2f" % tuple(np.exp(edges[4:6]))),
            (6, "s/kink %.2f-%.2f" % tuple(np.exp(edges[6:8])))]
    psi_panel(fig.add_subplot(gs[:, 1]), prob, cfg, bins, unit, "weekly benefit as a share of the cap")
    floor = 0.24

    def family(th, Q, idx):
        return np.clip(th * np.exp(Q), floor, 1.0)
    thetas = np.linspace(0.6, 1.0, 9)
    u_panels(fig.add_subplot(gs[:3, 2]), fig.add_subplot(gs[3, 2]), prob, cfg, family, thetas, 1.0, unit,
             "replacement-rate multiplier theta")
    fig.suptitle(f"UI (WA 1988 + IL 1984): Y = {'weeks of UI paid' if outcome == 'weeks' else 'UI dollars paid'}",
                 fontsize=10, x=0.01, ha="left")
    return fig


BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # reference ordinal ramp, steps 250-650
MUTED = "#5f5e5a"


def _panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold")


def cwbh(outcome):
    with plt.rc_context({"font.size": 10}):
        return _cwbh(outcome)


def _cwbh(outcome):
    """Paper figure (Treatment Intensities, Fig. 1): the fitted nonparametric parts.

    A: baseline b_hat(e) + mean(X)'beta1_hat, mean weeks paid at the cap by latent ability.
    B: effect surface alpha_hat(e, a) + (a - 1) mean(Z'tau_hat) against e, at 60, 75, 90% of the cap.
    C: the same surface against the benefit, at three latent abilities.
    Lines average the three rotations, bands span them over the supported part; dashed where the pair
    (e, a) lies outside the 2.5-97.5% eta range of claimants given a benefit near a.
    """
    args = argparse.Namespace(x="linear", outcome=outcome, seed=0)
    cfg = app.cwbh(args)
    prob = cfg["prob"]
    a_ref = prob.basis.a_ref
    fits = rotation_fits(prob, cluster=cfg["cluster"])
    _, eta_all, _ = ds.first_stage(prob.Q, prob.X, np.ones(len(prob.Q), bool))
    e_grid = np.linspace(*np.quantile(eta_all[prob.in_range], [0.025, 0.975]), 200)
    width = 0.05 * (prob.A.max() - prob.A.min())

    def eta_range(a):
        near = np.abs(prob.A - a) <= width
        return np.quantile(eta_all[near], [0.025, 0.975]) if near.sum() > 30 else (np.nan, np.nan)

    def surface(e, a):
        out = []
        for fit, eta, ev in fits:
            zbar = float(np.mean(prob.Z[ev] @ fit.tau)) if prob.Z.shape[1] else 0.0
            out.append(fit.f(e, a) + (a - a_ref) * zbar)
        return np.array(out)

    fig = plt.figure(figsize=(9.2, 3.4))
    gs = fig.add_gridspec(1, 3, wspace=0.42)

    # A. baseline
    ax = fig.add_subplot(gs[0, 0])
    base = np.array([fit.baseline(e_grid, prob.X[ev].mean(0)) for fit, eta, ev in fits])
    ax.fill_between(e_grid, base.min(0), base.max(0), color=COLORS[0], alpha=0.18, lw=0)
    ax.plot(e_grid, base.mean(0), color=COLORS[0], lw=1.8)
    ax.set_xlabel("latent ability $\\eta$")
    ax.set_ylabel("weeks paid at the cap")
    ax.set_ylim(12.5, 21)
    _panel_title(ax, "A. Baseline $\\hat b$")
    print("  A: baseline weeks at the cap: " + " ".join(
        f"eta={e:+.1f}:{np.interp(e, e_grid, base.mean(0)):.1f}" for e in (-0.7, -0.4, 0.0, 0.3, 0.6)))

    # B. effect against latent ability
    ax = fig.add_subplot(gs[0, 1])
    lo_y, hi_y = 0.0, 0.0
    print("  B: effect by latent ability")
    for c, a in zip(COLORS, (0.6, 0.75, 0.9)):
        e_lo, e_hi = eta_range(a)
        inside = (e_grid >= e_lo) & (e_grid <= e_hi)
        cur = surface(e_grid, np.full(len(e_grid), a))
        m = cur.mean(0)
        ax.fill_between(e_grid[inside], cur.min(0)[inside], cur.max(0)[inside], color=c, alpha=0.18, lw=0)
        ax.plot(e_grid[inside], m[inside], color=c, lw=1.8, label=f"{a:.0%} of the cap")
        ax.plot(e_grid, np.where(inside, np.nan, m), color=c, ls="--", lw=1.1)
        lo_y, hi_y = min(lo_y, cur.min(0)[inside].min()), max(hi_y, cur.max(0)[inside].max())
        print(f"    {a:.2f}: supported eta {e_lo:.2f} to {e_hi:.2f}; " +
              " ".join(f"eta={e:+.1f}:{np.interp(e, e_grid, m):+.1f}" for e in (-0.7, -0.4, 0.0, 0.2, 0.4)))
    ax.axhline(0, color="#8a8984", lw=0.8)
    pad = 0.08 * (hi_y - lo_y)
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.set_xlabel("latent ability $\\eta$")
    ax.set_ylabel("weeks paid vs. the full cap")
    ax.legend(frameon=False, fontsize=8, loc="lower left", title="benefit", title_fontsize=8)
    _panel_title(ax, "B. Effect $\\hat\\alpha$ by $\\eta$")
    ylim_b = ax.get_ylim()

    # C. effect against the benefit
    ax = fig.add_subplot(gs[0, 2])
    a_grid = np.round(np.arange(0.3, 1.0001, 0.025), 3)
    rng = np.array([eta_range(a) for a in a_grid])
    print("  C: effect by benefit")
    for c, e in zip(["#3d3c38", "#7a7873", "#b4b2ab"], (-0.4, 0.0, 0.3)):
        cur = surface(np.full(len(a_grid), e), a_grid)
        m = cur.mean(0)
        inside = (rng[:, 0] <= e) & (e <= rng[:, 1])
        inside[a_grid >= 1.0] = True
        ax.fill_between(a_grid[inside], cur.min(0)[inside], cur.max(0)[inside], color=c, alpha=0.18, lw=0)
        ax.plot(a_grid[inside], m[inside], color=c, lw=1.8, label=f"$\\eta$ = {e:+.1f}")
        ax.plot(a_grid, np.where(inside, np.nan, m), color=c, ls="--", lw=1.1)
        print(f"    eta={e:+.1f}: supported benefits {a_grid[inside].min():.3f}-{a_grid[inside].max():.3f}; " +
              " ".join(f"{a:g}:{v:+.1f}" for a, v in zip(a_grid, m) if np.isclose(a % 0.15, 0) or a in (0.45, 0.6, 0.75, 0.9)))
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_ylim(*ylim_b)
    ax.set_xlim(0.38, 1.02)
    ax.set_xlabel("benefit as a share of the cap")
    ax.set_ylabel("weeks paid vs. the full cap")
    ax.legend(frameon=False, fontsize=8, loc="lower right", title="latent ability", title_fontsize=8)
    _panel_title(ax, "C. Effect $\\hat\\alpha$ by benefit")
    return fig


def cwbh_schedules():
    """Paper figure (Treatment Intensities, Fig. 2): utility curves and schedules, weeks paid as the goal.

    A: no budget: every rate scaled by theta, cap fixed (change in mean weeks paid, band over rotations).
    B: budget-neutral family, cap kappa with the rate set by the budget (redesign_cwbh_weeks.json).
    C: schedules against the earnings ratio: today's, the best family member, and the budget-neutral
       schedules chosen bin by bin, free and monotone (redesign_cwbh_free.json); grey = supported benefits.
    """
    with plt.rc_context({"font.size": 10}):
        cfg = app.cwbh(argparse.Namespace(x="linear", outcome="weeks_paid", seed=0))
        prob = cfg["prob"]
        floor = prob.a_lo
        fig = plt.figure(figsize=(9.2, 3.4))
        gs = fig.add_gridspec(1, 3, wspace=0.45, width_ratios=[1, 1, 1.3])

        ax = fig.add_subplot(gs[0, 0])
        thetas = np.linspace(0.6, 1.0, 9)
        vals, sup = ds.family_value(prob, lambda th, Q, idx: np.clip(th * np.exp(Q), floor, 1.0), thetas, seed=0,
                                    cluster=cfg["cluster"])
        vals = vals - vals[:, [len(thetas) - 1]]
        ax.fill_between(thetas, vals.min(0), vals.max(0), color=COLORS[0], alpha=0.18, lw=0)
        ax.plot(thetas, vals.mean(0), color=COLORS[0], lw=1.8)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_xlabel("rate as a share of today's")
        ax.set_ylabel("weeks paid vs. today")
        _panel_title(ax, "A. No budget")
        print("  A: theta, mean, rotation min, max, supported share")
        for t, m_, lo_, hi_, sh in zip(thetas, vals.mean(0), vals.min(0), vals.max(0), sup.mean(0)):
            print(f"    {t:5.2f} {m_:7.2f} {lo_:7.2f} {hi_:7.2f} {sh:6.3f}", flush=True)

        ax = fig.add_subplot(gs[0, 1])
        fam = json.load(open(FIG / "redesign_cwbh_weeks.json"))
        rows = [r for r in fam["rows"] if r["s"] == r["s"]]
        k = np.array([r["x"] for r in rows])
        ax.fill_between(k, [r["lo"] for r in rows], [r["hi"] for r in rows], color=COLORS[1], alpha=0.18, lw=0)
        ax.plot(k, [r["mean"] for r in rows], "o-", color=COLORS[1], lw=1.8, ms=4.5)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_xlim(0.83, 1.02)
        ax.set_xticks(k)
        ax.set_xticklabels([f"{r['x']:.2f}\n\u00d7{r['s']:.2f}" for r in rows], fontsize=8.5)
        ax.set_xlabel("cap / today's, with rate multiplier")
        ax.set_ylabel("weeks paid vs. today")
        _panel_title(ax, "B. Budget-neutral family")

        ax = fig.add_subplot(gs[0, 2])
        free = json.load(open(FIG / "redesign_cwbh_free.json"))
        e = np.array(free["edges"])
        xs = np.exp(0.5 * (e[:-1] + e[1:]))
        lo = np.array([np.nan if v is None else v for v in free["allowed_lo"]])
        hi = np.array([np.nan if v is None else v for v in free["allowed_hi"]])
        ax.fill_between(xs, lo, hi, step="mid", color=GRID, alpha=0.8, lw=0, label="supported")
        q = np.exp(np.linspace(e[0], e[-1], 300))
        ax.plot(q, np.clip(q, floor, 1.0), color=MUTED, lw=1.5, label="today")
        best = fam["best"]
        ax.plot(q, np.clip(best["s"] * q, floor, best["x"]), color=COLORS[1], lw=1.5, ls="--",
                label=f"family (cap {best['x']:.2f}, \u00d7{best['s']:.2f})")
        ax.step(xs, np.mean([r["dose"] for r in free["free"]], 0), where="mid", color=COLORS[0], lw=1.8,
                label="free")
        ax.step(xs, np.mean([r["dose"] for r in free["monotone"]], 0), where="mid", color=COLORS[2], lw=1.5,
                label="free, monotone")
        ax.set_xscale("log")
        ticks = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("earnings ratio (log scale)")
        ax.set_ylabel("benefit as a share of the cap")
        ax.set_ylim(0, 1.04)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right", handlelength=1.6)
        _panel_title(ax, "C. Budget-neutral schedules")
    return fig


def taxi2():
    args = argparse.Namespace(x="linear", outcome=None, seed=0)
    cfg = app.taxi2(args)
    prob = cfg["prob"]
    fits = rotation_fits(prob)
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.35)
    alpha_panel(fig.add_subplot(gs[:, 0]), prob, fits, [2.0, 4.5, 6.25], "$", lambda a: f"middle suggestion ${a:g}")
    psi_panel(fig.add_subplot(gs[:, 1]), prob, cfg, [(0, "fare $5-8"), (3, "fare $14-17"), (5, "fare $20-25")],
              "$", "middle tip suggestion ($)")

    def family(th, Q, idx):
        return np.where(Q >= 15, th * Q, 3.0)
    thetas = np.linspace(0.15, 0.25, 11)
    u_panels(fig.add_subplot(gs[:3, 2]), fig.add_subplot(gs[3, 2]), prob, cfg, family, thetas, 0.25, "$ per ride",
             "percentage default theta above \\$15 (\\$3 below)")
    fig.suptitle("NYC taxi 2009 (VTS + CMT): Y = tip", fontsize=10, x=0.01, ha="left")
    return fig


def lending():
    args = argparse.Namespace(x="linear", outcome="return", seed=0)
    cfg = app.lending(args)
    prob = cfg["prob"]
    fits = rotation_fits(prob)
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.35)
    alpha_panel(fig.add_subplot(gs[:, 0]), prob, fits, [9.0, 15.0, 18.0], "return per $", lambda a: f"rate {a:g}%")
    psi_panel(fig.add_subplot(gs[:, 1]), prob, cfg, [(1, "grade B"), (2, "grade C"), (3, "grade D")],
              "return per $", "interest rate (%)")

    def family(th, Q, idx):
        return prob.A[idx] + th
    w = ((prob.A >= 9) & (prob.A <= 19)).astype(float)
    thetas = np.linspace(-3, 3, 13)
    u_panels(fig.add_subplot(gs[:3, 2]), fig.add_subplot(gs[3, 2]), prob, cfg, family, thetas, 0.0, "return per $",
             "shift of every rate (pp), loans at 9-19%", weight=w)
    fig.suptitle("Lending Club 2009-13, 36-month loans: Y = realized return per dollar", fontsize=10, x=0.01, ha="left")
    return fig


def dsh(outcome):
    args = argparse.Namespace(x="linear", outcome=outcome, seed=0)
    cfg = app.dsh(args)
    prob = cfg["prob"]
    unit = "log cases" if outcome == "log_cases" else "probability"
    fits = rotation_fits(prob, cluster=cfg["cluster"])
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.35)
    alpha_panel(fig.add_subplot(gs[:, 0]), prob, fits, [2.5, 5.0, 8.0], unit, lambda a: f"DSH add-on {a:g}%")
    psi_panel(fig.add_subplot(gs[:, 1]), prob, cfg, [(2, "DPP 15-20%"), (4, "DPP 25-30%"), (5, "DPP 30-40%")],
              unit, "DSH adjustment (%)")
    a_bar = prob.basis.a_bar

    def family(th, Q, idx):
        return np.minimum(th * prob.A[idx], a_bar)
    thetas = np.linspace(0.25, 1.0, 13)
    u_panels(fig.add_subplot(gs[:3, 2]), fig.add_subplot(gs[3, 2]), prob, cfg, family, thetas, 1.0, unit,
             "DSH scaled by theta, hospitals at <= 16%", weight=(prob.A <= 16).astype(float))
    fig.suptitle("Medicare DSH, FY 2008-13: Y = " + ("log Medicare cases in t+1" if outcome == "log_cases"
                                                      else "exit by t+2"), fontsize=10, x=0.01, ha="left")
    return fig


def lcff(subject):
    args = argparse.Namespace(x="linear", outcome=subject, seed=0)
    cfg = app.lcff(args)
    prob = cfg["prob"]
    fits = rotation_fits(prob, cluster=cfg["cluster"])
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.35)
    alpha_panel(fig.add_subplot(gs[:, 0]), prob, fits, [500.0, 1000.0, 1500.0], "pp met",
                lambda a: f"\\${a:,.0f} per ADA")
    psi_panel(fig.add_subplot(gs[:, 1]), prob, cfg, [(1, "UPP 0.30-0.45"), (3, "UPP 0.55-0.65"), (5, "UPP 0.75-0.85")],
              "pp met", "concentration grant (\\$ per ADA)")
    rate = 0.65

    def family(th, Q, idx):                      # concentration rate th instead of 0.65
        return np.minimum(prob.A[idx] * th / rate, prob.basis.a_bar)
    thetas = np.linspace(0.0, 0.8, 9)
    u_panels(fig.add_subplot(gs[:3, 2]), fig.add_subplot(gs[3, 2]), prob, cfg, family, thetas, rate, "pp met",
             "concentration rate (law: 0.50, then 0.65)")
    fig.suptitle(f"California LCFF 2020-21 to 2024-25: Y = CAASPP {subject.upper()} percent meeting the standard, next spring",
                 fontsize=10, x=0.01, ha="left")
    return fig


MAKERS = {"lcff": lambda: lcff("ela"), "lcff_math": lambda: lcff("math"), "ui": lambda: ui("weeks"), "ui_benefits": lambda: ui("benefits"),
          "cwbh": lambda: cwbh("weeks_paid"), "cwbh_sched": cwbh_schedules, "cwbh_duration": lambda: cwbh("duration"), "taxi2": taxi2, "lending": lending,
          "dsh": lambda: dsh("log_cases"), "dsh_exit": lambda: dsh("exit")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=sorted(MAKERS))
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for name in a.names:
        print("==", name, flush=True)
        fig = MAKERS[name]()
        fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        print("wrote", FIG / f"{name}.png", flush=True)


if __name__ == "__main__":
    main()
