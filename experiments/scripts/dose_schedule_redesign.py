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

Free versions (cwbh_free, dsh_free): no family. The score range is cut into bins and each bin
gets its own dose from a grid, chosen to optimise the summed utility subject to the same budget,
by a price lam on spending (Lagrangian; bisection on lam). Only doses supported in the bin
(coverage >= 90%) and the bin's current dose are allowed. The monotone version also requires the
dose not to fall as the score rises (dynamic programming over bins).

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


def cwbh():
    """Same family as ``ui`` on the CWBH claims. Benefits = weeks paid x weekly benefit (in units of
    the claimant's cap), with predicted weeks under the new schedule (behaviour included)."""
    cfg = app.cwbh(argparse.Namespace(x="linear", outcome="weeks_paid", seed=0))
    pw = cfg["prob"]
    floor = pw.a_lo
    fw = rotation_fits(pw, cluster=cfg["cluster"])
    kappas = np.round(np.arange(0.6, 1.001, 0.05), 3)
    U = np.full((3, len(kappas)), np.nan)
    sup = np.full((3, len(kappas)), np.nan)
    th = np.full((3, len(kappas)), np.nan)
    for r in range(3):
        fit_w, eta, ev = fw[r]
        idx = np.where(ev & pw.in_range)[0]
        fi = np.where(~ev & pw.in_range)[0]
        x = np.exp(pw.Q[idx])
        A0 = np.clip(x, floor, 1.0)
        gw0 = ds.gain(fit_w, eta[idx], pw.Z[idx], A0, 1.0)
        w0 = pw.Y[idx]
        cost0 = np.mean(w0 * A0)
        for j, kap in enumerate(kappas):
            def cost(t):
                a1 = np.clip(t * x, floor, kap)
                return np.mean((w0 + ds.gain(fit_w, eta[idx], pw.Z[idx], a1, 1.0) - gw0) * a1)
            t = solve_budget(cost, cost0, 0.5, 3.0)
            if np.isnan(t):
                continue
            a1 = np.clip(t * x, floor, kap)
            th[r, j] = t
            U[r, j] = np.mean(ds.gain(fit_w, eta[idx], pw.Z[idx], a1, 1.0) - gw0)
            sup[r, j] = ds.support_share(eta[fi], pw.A[fi], eta[idx], a1)
    j = report("cwbh_weeks", kappas, np.nanmean(th, 0), U, sup, np.zeros(len(kappas)), 1.0, better="lower")
    q = np.linspace(0.2, 1.6, 200)
    sched = [("status quo", q, np.clip(q, floor, 1.0)),
             (f"proposed: cap {kappas[j]:.2f}, rate x{np.nanmean(th[:, j]):.2f}", q,
              np.clip(np.nanmean(th[:, j]) * q, floor, kappas[j]))]
    plot("cwbh_weeks", kappas, U, sup, 1.0, "cap as a share of today's cap (replacement rate set by the budget)",
         "change in weeks paid", sched, "earnings / cap kink", "weekly benefit / today's cap",
         "UI (CWBH), budget-neutral: same predicted UI dollars, lower cap, higher rate", better="lower")


def _pick(L, C, lam, allowed, monotone):
    """Dose index per bin minimising L - lam C over allowed doses (non-decreasing in the bin if monotone)."""
    obj = np.where(allowed, L - lam * C, np.inf)
    if not monotone:
        return obj.argmin(1)
    B, G = obj.shape
    best, back = obj[0].copy(), []
    for b in range(1, B):
        idx = np.zeros(G, int)
        cur = 0
        for g in range(G):
            if best[g] < best[cur]:
                cur = g
            idx[g] = cur
        back.append(idx)
        best = obj[b] + best[idx]
    j = np.zeros(B, int)
    j[-1] = int(np.argmin(best))
    for b in range(B - 1, 0, -1):
        j[b - 1] = back[b - 1][j[b]]
    return j


def free_redesign(L, C, allowed, monotone, bound=1e4, iters=100):
    """Minimise sum_b L[b, j_b] subject to sum_b C[b, j_b] = 1 (C is spending over today's spending).

    Spending rises with the price lam, so lam is found by bisection; the returned design spends at
    least today's budget (the dose grid is discrete).
    """
    lo, hi = -bound, bound
    rows = np.arange(L.shape[0])
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        j = _pick(L, C, lam, allowed, monotone)
        if C[rows, j].sum() < 1.0:
            lo = lam
        else:
            hi = lam
    # Many bins can switch at the same price, so spending jumps past the budget. Blend the two
    # solutions on either side of the budget: bins below m keep the cheaper solution, bins from m up
    # take the dearer one, with the largest m that still spends the budget.
    j_lo, j_hi = _pick(L, C, lo, allowed, monotone), _pick(L, C, hi, allowed, monotone)
    best = j_hi
    for m in range(len(rows), -1, -1):
        j = np.concatenate([j_lo[:m], j_hi[m:]])
        if monotone:
            j = np.maximum.accumulate(j)
        if C[rows, j].sum() >= 1.0:
            best = j
            break
    return best, float(C[rows, best].sum()), hi


def _free_run(prob, fits, edges, grid, unit_dose, today, loss, cost, pop, min_cov=0.90):
    """Run the free redesign in each rotation. unit_dose(a, idx) maps a bin dose to each unit's dose;
    loss(fit, eta, idx, a_units) and cost(idx, a_units, loss_units) return unit-level arrays."""
    _, eta_all, _ = ds.first_stage(prob.Q, prob.X, np.ones(len(prob.Q), bool))
    cov = ds.coverage(eta_all, prob.A, prob.Q, edges, grid, prob.basis.a_ref, band=0.04)
    out = {"free": [], "monotone": []}
    B = len(edges) - 1
    for fit, eta, ev in fits:
        idx = np.where(ev & prob.in_range & pop)[0]
        b = np.clip(np.searchsorted(edges, prob.Q[idx], side="right") - 1, 0, B - 1)
        a0 = today(idx)
        l0 = loss(fit, eta, idx, a0)
        c0 = cost(idx, a0, l0).sum()
        L = np.zeros((B, len(grid)))
        C = np.zeros((B, len(grid)))
        for k, a in enumerate(grid):
            au = unit_dose(a, idx)
            lu = loss(fit, eta, idx, au)
            L[:, k] = np.bincount(b, lu, B) / len(idx)
            C[:, k] = np.bincount(b, cost(idx, au, lu), B) / c0
        allowed = cov >= min_cov
        for bb in range(B):
            m = b == bb
            if m.any():
                allowed[bb, np.argmin(np.abs(grid - np.median(a0[m])))] = True
        for name, mono in (("free", False), ("monotone", True)):
            j, spend, lam = free_redesign(L, C, allowed, mono)
            change = L[np.arange(B), j].sum() - l0.sum() / len(idx)
            out[name].append(dict(change=float(change), spend=spend, price=float(lam),
                                  dose=[float(grid[k]) for k in j]))
    out["allowed_lo"] = [float(grid[allowed_row(cov, r, min_cov)][0]) if allowed_row(cov, r, min_cov).any() else None
                         for r in range(B)]
    out["allowed_hi"] = [float(grid[allowed_row(cov, r, min_cov)][-1]) if allowed_row(cov, r, min_cov).any() else None
                         for r in range(B)]
    out["edges"] = [float(e) for e in edges]
    out["grid"] = [float(g) for g in grid]
    for name in ("free", "monotone"):
        ch = [r["change"] for r in out[name]]
        print(f"  {name}: change {np.mean(ch):.4g} (rotations {min(ch):.4g} to {max(ch):.4g}); spending "
              f"{[round(r['spend'], 3) for r in out[name]]}", flush=True)
        for r in out[name]:
            print("    doses by bin:", r["dose"], flush=True)
    return out


def allowed_row(cov, r, min_cov):
    return cov[r] >= min_cov


def _free_plot(name, out, xs, today_x, today_a, family, xlab, ylab, title, xlog=False):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    lo = np.array([np.nan if v is None else v for v in out["allowed_lo"]])
    hi = np.array([np.nan if v is None else v for v in out["allowed_hi"]])
    ax.fill_between(xs, lo, hi, step="mid", color="#d9d8d2", alpha=0.7, label="supported doses (coverage >= 90%)")
    ax.plot(today_x, today_a, color="#5f5e5a", lw=1.5, label="status quo")
    if family is not None:
        ax.plot(family[0], family[1], color=COLORS[1], lw=1.5, ls="--", label=family[2])
    for r in out["free"]:
        ax.step(xs, r["dose"], where="mid", color=COLORS[0], lw=0.7, alpha=0.5)
    ax.step(xs, np.mean([r["dose"] for r in out["free"]], 0), where="mid", color=COLORS[0], lw=2,
            label="free, by score bin (mean of rotations)")
    ax.step(xs, np.mean([r["dose"] for r in out["monotone"]], 0), where="mid", color=COLORS[2], lw=1.5,
            label="free and monotone (mean of rotations)")
    if xlog:
        ax.set_xscale("log")
        ticks = [t for t in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0) if xs.min() * 0.8 <= t <= xs.max() * 1.25]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    fig.savefig(FIG / f"redesign_{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def cwbh_free():
    """UI without a family: a benefit per score bin (20 bins, grid 0.25-1.00 by 0.025), same predicted
    UI dollars (weeks paid x benefit, with predicted weeks); utility = weeks paid (lower is better)."""
    cfg = app.cwbh(argparse.Namespace(x="linear", outcome="weeks_paid", seed=0))
    p = cfg["prob"]
    floor = p.a_lo
    fits = rotation_fits(p, cluster=cfg["cluster"])
    edges = np.quantile(p.Q, np.linspace(0, 1, 21))
    edges[-1] += 1e-9
    grid = np.round(np.arange(0.25, 1.0001, 0.025), 3)
    x = np.exp(p.Q)
    a_today = np.clip(x, floor, 1.0)
    g0 = {}

    def loss(fit, eta, idx, a):
        key = id(fit)
        if key not in g0:
            g0[key] = ds.gain(fit, eta[idx], p.Z[idx], a_today[idx], 1.0)
        return p.Y[idx] + ds.gain(fit, eta[idx], p.Z[idx], a, 1.0) - g0[key]

    out = _free_run(p, fits, edges, grid, unit_dose=lambda a, idx: np.full(len(idx), a),
                    today=lambda idx: a_today[idx], loss=loss, cost=lambda idx, a, weeks: weeks * a,
                    pop=np.ones(len(p.Q), bool))
    json.dump(out, open(FIG / "redesign_cwbh_free.json", "w"), indent=1)
    xs = np.exp(0.5 * (edges[:-1] + edges[1:]))
    q = np.exp(np.linspace(p.Q.min(), p.Q.max(), 300))
    fam = json.load(open(FIG / "redesign_cwbh_weeks.json"))["best"]
    _free_plot("cwbh_free", out, xs, q, np.clip(q, floor, 1.0),
               (q, np.clip(fam["s"] * q, floor, fam["x"]), f"family: cap {fam['x']:.2f}, rate x{fam['s']:.2f}"),
               "earnings / cap kink (log scale)", "weekly benefit / today's cap",
               "UI (CWBH), budget-neutral without a family: benefit by score bin", xlog=True)


def dsh_free(outcome="log_cases"):
    """DSH without a family: an add-on per DPP bin (20 bins among DPP <= 45, grid 0-20 by 0.5; the 12%
    cap still applies to the capped class), same add-on dollars (lagged cases x case-mix index);
    utility = 100 x change in log Medicare cases (higher is better)."""
    cfg = app.dsh(argparse.Namespace(x="linear", outcome=outcome, seed=0))
    p = cfg["prob"]
    from experiments.datasets.medicare_dsh import adapter as ad
    s = ad.load_dose(outcome=outcome)
    fn = s.feature_names
    w = np.nan_to_num(np.exp(s.X[:, fn.index("log_cases_lag")]) * s.X[:, fn.index("cmi")])
    capped = s.extras["capped_class"].astype(bool)
    pop = (p.Q <= 45) & (w > 0)
    fits = rotation_fits(p, cluster=cfg["cluster"])
    edges = np.quantile(p.Q[pop], np.linspace(0, 1, 21))
    edges[-1] += 1e-9
    grid = np.round(np.arange(0.0, min(20.0, p.basis.a_bar) + 1e-9, 0.5), 2)
    g0 = {}

    def unit_dose(a, idx):
        return np.where(capped[idx], np.minimum(a, 12.0), a)

    def loss(fit, eta, idx, a):
        key = id(fit)
        if key not in g0:
            g0[key] = ds.gain(fit, eta[idx], p.Z[idx], np.minimum(p.A[idx], p.basis.a_bar), 0.0)
        return -100.0 * (ds.gain(fit, eta[idx], p.Z[idx], np.minimum(a, p.basis.a_bar), 0.0) - g0[key])

    out = _free_run(p, fits, edges, grid, unit_dose=unit_dose, today=lambda idx: p.A[idx], loss=loss,
                    cost=lambda idx, a, l: w[idx] * a, pop=pop)
    json.dump(out, open(FIG / f"redesign_dsh_free_{outcome}.json", "w"), indent=1)
    xs = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
    q = np.linspace(0, 45, 301)
    sq = np.where(q < 15, 0, np.where(q < 20.2, 2.5 + 0.65 * (q - 15), 5.88 + 0.825 * (q - 20.2)))
    fam = json.load(open(FIG / f"redesign_dsh_{outcome}.json"))["best"]
    _free_plot(f"dsh_free_{outcome}", out, xs, q, sq,
               (q, fam["s"] * np.maximum(q - fam["x"], 0), f"family: start {fam['x']:.1f}%, slope {fam['s']:.2f}"),
               "DPP (%)", "DSH add-on (%), large urban hospitals",
               "Medicare DSH, budget-neutral without a family: add-on by DPP bin (utility: next year's Medicare cases)")


def _cross(fits, evaluate, schedules):
    """M[r, c]: change in utility of the schedule chosen with rotation r's fit, evaluated with rotation c's
    fit on c's evaluation fold; S[r, c]: its spending over today's under that fit."""
    M = np.zeros((3, 3))
    S = np.zeros((3, 3))
    for r in range(3):
        for c, fc in enumerate(fits):
            M[r, c], S[r, c] = evaluate(fc, np.array(schedules[r]))
    return M, S


def free_cross():
    """Cross-evaluation of the free schedules: does a schedule chosen with one rotation's fit keep its gain
    under the other rotations' fits (whose outcome fits use other folds)? Needs cwbh_free and dsh_free."""
    out = {}
    # UI: change in weeks paid
    cfg = app.cwbh(argparse.Namespace(x="linear", outcome="weeks_paid", seed=0))
    p = cfg["prob"]
    fits = rotation_fits(p, cluster=cfg["cluster"])
    o = json.load(open(FIG / "redesign_cwbh_free.json"))
    edges = np.array(o["edges"])
    a_today = np.clip(np.exp(p.Q), p.a_lo, 1.0)

    def ev_ui(fc, sched):
        fit, eta, ev = fc
        idx = np.where(ev & p.in_range)[0]
        b = np.clip(np.searchsorted(edges, p.Q[idx], side="right") - 1, 0, len(edges) - 2)
        a1 = sched[b]
        w1 = p.Y[idx] + ds.gain(fit, eta[idx], p.Z[idx], a1, 1.0) - ds.gain(fit, eta[idx], p.Z[idx], a_today[idx], 1.0)
        return (w1 - p.Y[idx]).mean(), (w1 * a1).sum() / (p.Y[idx] * a_today[idx]).sum()
    for kind in ("free", "monotone"):
        M, S = _cross(fits, ev_ui, [r["dose"] for r in o[kind]])
        out[f"ui_{kind}"] = dict(change=M.tolist(), spend=S.tolist())
    # DSH: 100 x change in log Medicare cases
    cfg = app.dsh(argparse.Namespace(x="linear", outcome="log_cases", seed=0))
    p = cfg["prob"]
    from experiments.datasets.medicare_dsh import adapter as ad
    s = ad.load_dose(outcome="log_cases")
    fn = s.feature_names
    w = np.nan_to_num(np.exp(s.X[:, fn.index("log_cases_lag")]) * s.X[:, fn.index("cmi")])
    capped = s.extras["capped_class"].astype(bool)
    pop = (p.Q <= 45) & (w > 0)
    fits = rotation_fits(p, cluster=cfg["cluster"])
    o = json.load(open(FIG / "redesign_dsh_free_log_cases.json"))
    edges = np.array(o["edges"])
    abar = p.basis.a_bar

    def ev_dsh(fc, sched):
        fit, eta, ev = fc
        idx = np.where(ev & p.in_range & pop)[0]
        b = np.clip(np.searchsorted(edges, p.Q[idx], side="right") - 1, 0, len(edges) - 2)
        a1 = np.where(capped[idx], np.minimum(sched[b], 12.0), sched[b])
        d = (ds.gain(fit, eta[idx], p.Z[idx], np.minimum(a1, abar), 0.0)
             - ds.gain(fit, eta[idx], p.Z[idx], np.minimum(p.A[idx], abar), 0.0))
        return 100 * d.mean(), (w[idx] * a1).sum() / (w[idx] * p.A[idx]).sum()
    for kind in ("free", "monotone"):
        M, S = _cross(fits, ev_dsh, [r["dose"] for r in o[kind]])
        out[f"dsh_{kind}"] = dict(change=M.tolist(), spend=S.tolist())
    json.dump(out, open(FIG / "redesign_free_cross.json", "w"), indent=1)
    off = ~np.eye(3, dtype=bool)
    print("own-fit vs cross-evaluated change (mean, range of the six cross values), spending range under other fits")
    for k, v in out.items():
        M, S = np.array(v["change"]), np.array(v["spend"])
        print(f"  {k:14s} own {np.diag(M).mean():+.3f}  cross {M[off].mean():+.3f} ({M[off].min():+.3f} to "
              f"{M[off].max():+.3f})  spending {S[off].min():.3f}-{S[off].max():.3f}", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (setting, unit) in zip(axes, (("ui", "change in weeks paid (lower is better)"),
                                          ("dsh", "change in Medicare cases, % (higher is better)"))):
        for i, kind in enumerate(("free", "monotone")):
            M = np.array(out[f"{setting}_{kind}"]["change"])
            ax.scatter(np.full(3, i - 0.08), np.diag(M), color=COLORS[0], s=28,
                       label="fit that chose it" if i == 0 else None)
            ax.scatter(np.full(6, i + 0.08), M[off], color=COLORS[1], marker="x", s=30,
                       label="other rotations' fits" if i == 0 else None)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["free", "free, monotone"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title("UI (CWBH)" if setting == "ui" else "Medicare DSH", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "redesign_free_cross.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


MAKERS = {"lcff": lambda: lcff("ela"), "lcff_math": lambda: lcff("math"), "dsh": lambda: dsh("exit"),
          "dsh_cases": lambda: dsh("log_cases"), "ui": ui, "cwbh": cwbh,
          "cwbh_free": cwbh_free, "dsh_free": dsh_free,
          "free_cross": free_cross}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=sorted(MAKERS))
    for name in p.parse_args().names:
        MAKERS[name]()


if __name__ == "__main__":
    main()
