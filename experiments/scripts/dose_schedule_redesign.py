"""Budget-neutral redesigns of the dose schedule: UI, Medicare DSH, California LCFF.

Each redesign is a two-parameter family with one parameter fixed by the budget, so the utility is
a curve in the other parameter:

- LCFF: grant = k x m_year x max(min(UPP, cap) - t, 0) $/ADA (m_year = the year's $ slope, so
  t = 0.55, k = 1 is the law). For each start t, k keeps total grant dollars (sum of ADA x grant)
  at the status quo. U(t) = change in the share of tested students meeting the standard.
- DSH: add-on = min(s x max(DPP - t, 0), 12 for the capped class), t in [0, 20]. For each t, s keeps total
  add-on dollars (proxy weight: lagged Medicare cases x case-mix index) at the status quo, among
  hospitals with DPP <= 45. U(t) = change in the share of hospitals exiting (lower is better).
- UI: benefit / cap = clip(theta x score / cap kink, 0.24, kappa); the status quo is theta = kappa = 1
  (the formula without rounding, floors and dependents' allowances). For each cap kappa <= 1,
  theta keeps predicted UI dollars paid (including the behavioural response, from the
  dollars-paid fit) at the status quo. U(kappa) = change in weeks paid.

All changes are against the status quo, per role rotation (evaluation fold), with the share of
units whose new dose is supported for their eta. Figures and numbers go to
experiments/datasets/simulations/dose_schedule/.

    python -m experiments.scripts.dose_schedule_redesign lcff dsh ui
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.methods import dose_schedule as ds  # noqa: E402
from experiments.scripts import dose_schedule_apply as app  # noqa: E402
from experiments.scripts.dose_schedule_figures import COLORS, FIG, rotation_fits  # noqa: E402


def solve_budget(make, budget, lo, hi, iters=60):
    """Smallest-error s in [lo, hi] with make(s) = budget, make increasing in s (bisection)."""
    if make(hi) < budget:
        return np.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if make(mid) < budget else (lo, mid)
    return 0.5 * (lo + hi)


def mechanical_redesign(prob, fits, family, cost_w, pop, ts, s_hi, u_w, scale=1.0):
    """U(t) for budget-neutral members of family(t, s, idx); budget = sum over pop of cost_w x dose."""
    a_ref = prob.basis.a_ref
    idx_pop = np.where(pop)[0]
    budget = np.sum(cost_w[idx_pop] * prob.A[idx_pop])
    s_of_t = np.array([solve_budget(lambda s: np.sum(cost_w[idx_pop] * family(t, s, idx_pop)), budget, 0.0, s_hi)
                       for t in ts])
    U = np.full((3, len(ts)), np.nan)
    sup = np.full((3, len(ts)), np.nan)
    over = np.full(len(ts), np.nan)
    for r, (fit, eta, ev) in enumerate(fits):
        idx = np.where(ev & pop & prob.in_range)[0]
        fi = np.where(~ev & prob.in_range)[0]
        g0 = ds.gain(fit, eta[idx], prob.Z[idx], prob.A[idx], a_ref)
        for j, (t, s) in enumerate(zip(ts, s_of_t)):
            if np.isnan(s):
                continue
            a1 = family(t, s, idx)
            g1 = ds.gain(fit, eta[idx], prob.Z[idx], np.minimum(a1, prob.basis.a_bar), a_ref)
            U[r, j] = scale * np.sum(u_w[idx] * (g1 - g0)) / np.sum(u_w[idx])
            sup[r, j] = ds.support_share(eta[fi], prob.A[fi], eta[idx], a1)
            over[j] = np.mean(family(t, s, idx_pop) > prob.basis.a_bar)
    return s_of_t, U, sup, over


def plot(name, xs, U, sup, x0, xlab, ylab, schedules, sched_xlab, sched_ylab, title, better="higher"):
    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.28)
    ax, ax2, axs = fig.add_subplot(gs[:3, 0]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[:, 1])
    for u in U:
        ax.plot(xs, u, color=COLORS[0], lw=0.6, alpha=0.5)
    m = np.nanmean(U, 0)
    ax.plot(xs, m, color=COLORS[0])
    if x0 is not None:
        ax.axvline(x0, color="#8a8984", lw=0.8, ls=":")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_ylabel(ylab)
    ax2.plot(xs, np.nanmean(sup, 0), color="#5f5e5a", lw=1.5)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("supported")
    ax2.set_xlabel(xlab)
    for c, (lab, q, a) in zip(["#8a8984", COLORS[1], COLORS[2]], schedules):
        axs.plot(q, a, color=c, label=lab, ls="-" if lab.startswith("status") else "-")
    axs.set_xlabel(sched_xlab)
    axs.set_ylabel(sched_ylab)
    axs.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title} ({better} U is better)", fontsize=10, x=0.01, ha="left", y=1.02)
    fig.savefig(FIG / f"redesign_{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def report(name, xs, s_of_t, U, sup, over, x0, better="higher"):
    m = np.nanmean(U, 0)
    j = int(np.nanargmax(m) if better == "higher" else np.nanargmin(m))
    rows = [dict(x=float(x), s=float(s), mean=float(mm), lo=float(np.nanmin(u)), hi=float(np.nanmax(u)),
                 supported=float(np.nanmean(sp)), above_max=float(ov))
            for x, s, mm, u, sp, ov in zip(xs, s_of_t, m, U.T, sup.T, over)]
    print(f"== {name}: x, budget parameter, mean change, rotation min, max, supported, share above max dose")
    for r in rows:
        print("  %8.4g %10.4g %10.4g %10.4g %10.4g %6.3f %6.3f" % tuple(r.values()), flush=True)
    print(f"  best x = {xs[j]:.4g} (status quo {x0:.4g}): change {m[j]:.4g} "
          f"(rotations {np.nanmin(U[:, j]):.4g} to {np.nanmax(U[:, j]):.4g}), supported {np.nanmean(sup[:, j]):.3f}")
    json.dump({"rows": rows, "best": rows[j]}, open(FIG / f"redesign_{name}.json", "w"), indent=1)
    return j


def lcff(subject="ela"):
    args = argparse.Namespace(x="linear", outcome=subject, seed=0)
    cfg = app.lcff(args)
    prob = cfg["prob"]
    from experiments.datasets.ca_lcff import adapter as ad
    e = ad.load_dose(subject=subject).extras
    cap, fy, ada, tested = e["cap"], e["fy"], e["ada"], np.nan_to_num(e["tested"], nan=0.0)
    excess = np.maximum(np.minimum(prob.Q, cap) - 0.55, 0)
    m_year = {y: np.median(prob.A[(fy == y) & (excess > 1e-6)] / excess[(fy == y) & (excess > 1e-6)]) for y in np.unique(fy)}
    m = np.array([m_year[y] for y in fy])

    def family(t, k, idx):
        return k * m[idx] * np.maximum(np.minimum(prob.Q[idx], cap[idx]) - t, 0)
    fits = rotation_fits(prob, cluster=cfg["cluster"])
    ts = np.round(np.arange(0.30, 0.751, 0.025), 3)
    pop = np.ones(len(prob.Q), bool)
    s_of_t, U, sup, over = mechanical_redesign(prob, fits, family, ada, pop, ts, 20.0, tested)
    j = report(f"lcff_{subject}", ts, s_of_t, U, sup, over, 0.55)
    q = np.linspace(0, 1, 201)
    m25 = m_year[max(m_year)]
    sched = [("status quo (start 0.55, rate 0.65)", q, m25 * np.maximum(q - 0.55, 0)),
             (f"proposed: start {ts[j]:.3f}", q, s_of_t[j] * m25 * np.maximum(q - ts[j], 0))]
    plot(f"lcff_{subject}", ts, U, sup, 0.55, "UPP at which the grant starts (slope set by the budget)",
         f"change in % meeting ({subject.upper()})", sched, "UPP (high-need share)", "grant, $ per ADA (2024-25 rates)",
         "LCFF, budget-neutral: same total concentration-grant dollars, grant starts at t")


def dsh(outcome="exit"):
    args = argparse.Namespace(x="linear", outcome=outcome, seed=0)
    cfg = app.dsh(args)
    prob = cfg["prob"]
    from experiments.datasets.medicare_dsh import adapter as ad
    s = ad.load_dose(outcome=outcome)
    fn = s.feature_names
    w = np.exp(s.X[:, fn.index("log_cases_lag")]) * s.X[:, fn.index("cmi")]
    capped = s.extras["capped_class"].astype(bool)
    capv = np.where(capped, 12.0, np.inf)

    def family(t, k, idx):
        return np.minimum(k * np.maximum(prob.Q[idx] - t, 0), capv[idx])
    fits = rotation_fits(prob, cluster=cfg["cluster"])
    ts = np.arange(0.0, 20.01, 1.25)       # later starts push doses past the data (estimates explode)
    pop = np.isfinite(w) & (prob.Q <= 45)
    scale = 100.0 if outcome == "exit" else 100.0
    s_of_t, U, sup, over = mechanical_redesign(prob, fits, family, np.nan_to_num(w), pop, ts, 10.0,
                                               np.ones(len(prob.Q)), scale=scale)
    better = "lower" if outcome == "exit" else "higher"
    j = report(f"dsh_{outcome}", ts, s_of_t, U, sup, over, 15.0, better=better)
    q = np.linspace(0, 45, 301)
    sq = np.where(q <= 15, 0, np.where(q <= 20.2, 2.5 + 0.65 * (q - 15), 5.88 + 0.825 * (q - 20.2)))
    sched = [("status quo (large urban)", q, sq), (f"proposed: start {ts[j]:.2f}%", q, s_of_t[j] * np.maximum(q - ts[j], 0))]
    ylab = "change in exit, pp" if outcome == "exit" else "change in Medicare cases, %"
    plot(f"dsh_{outcome}", ts, U, sup, None, "DPP at which the add-on starts (slope set by the budget)", ylab, sched,
         "DPP (%)", "DSH add-on (%)", "Medicare DSH, budget-neutral: same total add-on dollars, linear add-on from t",
         better=better)


def ui():
    base = argparse.Namespace(x="linear", outcome="weeks", seed=0)
    pw = app.ui(base)["prob"]
    pb = app.ui(argparse.Namespace(x="linear", outcome="benefits", seed=0))["prob"]
    assert np.allclose(pw.Q, pb.Q)
    fw, fb = rotation_fits(pw), rotation_fits(pb)
    kappas = np.round(np.arange(0.6, 1.001, 0.05), 3)
    U = np.full((3, len(kappas)), np.nan)
    sup = np.full((3, len(kappas)), np.nan)
    th = np.full((3, len(kappas)), np.nan)
    for r in range(3):
        (fit_w, eta, ev), (fit_b, _, _) = fw[r], fb[r]
        idx = np.where(ev & pw.in_range)[0]
        fi = np.where(~ev & pw.in_range)[0]
        x = np.exp(pw.Q[idx])
        A0 = np.clip(x, 0.24, 1.0)          # the family member theta = 1, kappa = 1 (the formula without rounding)
        gw0 = ds.gain(fit_w, eta[idx], pw.Z[idx], A0, 1.0)
        gb0 = ds.gain(fit_b, eta[idx], pb.Z[idx], A0, 1.0)
        for j, kap in enumerate(kappas):
            def cost(t):
                a1 = np.clip(t * x, 0.24, kap)
                return np.mean(ds.gain(fit_b, eta[idx], pb.Z[idx], a1, 1.0) - gb0)
            t = solve_budget(cost, 0.0, 0.5, 3.0)
            if np.isnan(t):
                continue
            a1 = np.clip(t * x, 0.24, kap)
            th[r, j] = t
            U[r, j] = np.mean(ds.gain(fit_w, eta[idx], pw.Z[idx], a1, 1.0) - gw0)
            sup[r, j] = ds.support_share(eta[fi], pw.A[fi], eta[idx], a1)
    j = report("ui_weeks", kappas, np.nanmean(th, 0), U, sup, np.zeros(len(kappas)), 1.0, better="lower")
    q = np.linspace(0.2, 1.6, 200)
    sched = [("status quo", q, np.clip(q, 0.24, 1.0)),
             (f"proposed: cap {kappas[j]:.2f}, rate x{np.nanmean(th[:, j]):.2f}", q, np.clip(np.nanmean(th[:, j]) * q, 0.24, kappas[j]))]
    plot("ui_weeks", kappas, U, sup, 1.0, "cap as a share of today's cap (replacement rate set by the budget)",
         "change in weeks paid", sched, "earnings / cap kink", "weekly benefit / today's cap",
         "UI, budget-neutral: same predicted UI dollars paid (with behaviour), lower cap, higher rate", better="lower")


MAKERS = {"lcff": lambda: lcff("ela"), "lcff_math": lambda: lcff("math"), "dsh": lambda: dsh("exit"),
          "dsh_cases": lambda: dsh("log_cases"), "ui": ui}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=sorted(MAKERS))
    for name in p.parse_args().names:
        MAKERS[name]()


if __name__ == "__main__":
    main()
