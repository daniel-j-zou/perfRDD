"""UI (CWBH): does the dose-schedule estimator work on one state, or one schedule, alone?

On the share-of-cap scale every state-period schedule is the same, A = min(exp(Q), 1) above a
floor, so the pooled sample is already a single-schedule design. This script reruns the
estimator on the pooled sample, on each state, and on the two largest single state-period
schedules, with the default spline sizes and with small ones (K = 4, k = 3, dose basis l = 2),
and reports how often claimants reach the cap (the reference region) by quintile of the
latent part eta, and how many fitting-fold claims support the thinnest column of the tensor spline
(a cell of eta x dose). Then, on the pooled sample: (i) a test of a common dose response across
states, adding (A - 1) x state columns (Idaho is the reference) to the outcome fit, three seeds x
three score folds; (ii) the estimator with state indicators added to the modifiers Z, by score bin
and for each state's own claimants at 0.59-0.91 of the kink.

    python -m experiments.scripts.dose_schedule_by_state

Writes simulations/dose_schedule/ui_by_state.json.
"""
from __future__ import annotations

import json

import numpy as np

from experiments.datasets.ui_cwbh import adapter as ad
from experiments.methods import dose_schedule as ds
from experiments.scripts.dose_schedule_figures import FIG, rotation_fits

A_GRID = np.array([0.45, 0.6, 0.75, 0.9, 1.0])
# the pooled run's bins from 0.44 to 1.40 of the cap kink, plus two outer bins so that no claim is
# lumped into an end bin (the trimmed sample spans about 0.23 to 3.7)
EDGES = np.log(np.array([0.2, 0.44, 0.59, 0.74, 0.91, 1.12, 1.40, 4.0]))


def main():
    s = ad.load_dose()
    fn = s.feature_names
    lq = np.log(s.Q)
    keep = (lq > np.quantile(lq, 0.01)) & (lq < np.quantile(lq, 0.99))
    state, sched = s.extras["state"], s.extras["schedule"]
    zc = [fn.index(c) for c in ("age", "male", "educ")]
    _, eta_pooled, _ = ds.first_stage(lq[keep], s.X[keep], np.ones(keep.sum(), bool))
    q_edges = np.quantile(eta_pooled, np.linspace(0, 1, 6))
    samples = [("pooled", np.ones(len(lq), bool))] + [(st, state == st) for st in ("LA", "ID", "MO", "NM")] + [
        ("MO_1979_83", sched == "MO_105_15_22.22"), ("LA_1982_83", sched == "LA_205_10_25.0")]
    out = {"edges": EDGES.tolist(), "a_grid": A_GRID.tolist(), "runs": []}
    for label, m0 in samples:
        m = m0 & keep
        X = s.X[m][:, s.X[m].std(0) > 0]
        A = s.extras["A"][m]
        a_lo = float(np.floor(100 * A.min()) / 100)
        _, eta, _ = ds.first_stage(lq[m], X, np.ones(m.sum(), bool))
        quint = np.clip(np.searchsorted(q_edges, eta_pooled[m[keep]], side="right") - 1, 0, 4)
        at_cap = [float(np.mean(A[quint == k] == 1)) for k in range(5)]
        for size, (K, k, l) in (("default", (None, None, 3)), ("small", (4, 3, 2))):
            prob = ds.DoseProblem(lq[m], X, A, s.Y[m], s.X[m][:, zc], ds.DoseBasis(1.0, 1.0, l=l, a_lo=a_lo), a_lo=a_lo)
            est = ds.estimate(prob, n_bins=10, seed=0, a_grid=A_GRID, score_edges=EDGES,
                              cluster=s.extras["claim_id"][m], K=K, k=k)
            cov = ds.coverage(eta, prob.A, prob.Q, est["edges"], est["a_grid"], 1.0, band=0.04)
            cells = min_cell_claims(prob, s.extras["claim_id"][m], est["K"], est["k"])
            eff = est["psi"] - est["psi"][:, [-1]]
            sd = (est["psi_rot"] - est["psi_rot"][:, :, [-1]]).std(0)
            run = dict(sample=label, splines=size, K=est["K"], k=est["k"], l=l, n=int(m.sum()),
                       r2=float(1 - eta.var() / lq[m].var()), share_at_cap=float(np.mean(A == 1)),
                       at_cap_by_eta_quintile=at_cap, effect=eff.tolist(), rotation_sd=sd.tolist(),
                       coverage=cov.tolist(), min_cell_claims=cells)
            out["runs"].append(run)
            print(f"\n== {label} [{size}: K={est['K']} k={est['k']} l={l}] n {m.sum():,} R2 {run['r2']:.2f} "
                  f"at cap {run['share_at_cap']:.2f}; at cap by pooled-eta quintile {np.round(at_cap, 2).tolist()}; "
                  f"fewest claims in a tensor cell {cells}")
            for i in range(len(EDGES) - 1):
                print(f"   {np.exp(EDGES[i]):.2f}-{np.exp(EDGES[i + 1]):.2f}  " + "  ".join(
                    f"{eff[i, j]:9.2f} ({sd[i, j]:7.2f},{100 * cov[i, j]:3.0f}%)" for j in range(4)), flush=True)
    out["pooling_test"] = pooling_test(s, lq, keep, zc)
    out["state_modifier"] = state_modifier(s, lq, keep, zc)
    out["share_at_floor"] = float(np.mean(s.extras["A"][keep] <= (s.extras["min_b"] / s.extras["max_b"])[keep] + 1e-9))
    print("\nshare of the trimmed sample at its floor: %.4f" % out["share_at_floor"])
    json.dump(out, open(FIG / "ui_by_state.json", "w"), indent=1)


def min_cell_claims(prob, cluster, K, k):
    """Fewest claims with a nonzero value in a tensor-spline column, in the first fitting fold."""
    fold = ds.make_folds(len(prob.Q), 0, cluster)
    _, eta, _ = ds.first_stage(prob.Q, prob.X, fold == 0)
    use = (fold == 1) & prob.in_range
    ebox = tuple(np.quantile(eta[prob.in_range], [0.005, 0.995]))
    Pk = ds.bspline(eta[use], *ebox, k)
    B = prob.basis(prob.A[use])
    T = (Pk[:, :, None] * B[:, None, :]).reshape(use.sum(), -1)
    return int((np.abs(T) > 1e-12).sum(0).min())


def pooling_test(s, lq, keep, zc):
    st, A, cl = s.extras["state"][keep], s.extras["A"][keep], s.extras["claim_id"][keep]
    a_lo = float(np.floor(100 * A.min()) / 100)
    prob = ds.DoseProblem(lq[keep], s.X[keep], A, s.Y[keep], s.X[keep][:, zc],
                          ds.DoseBasis(1.0, 1.0, l=3, a_lo=a_lo), a_lo=a_lo)
    others = ["LA", "MO", "NM"]
    cols = np.column_stack([(A - 1.0) * (st == g) for g in others])
    res = []
    print("\ncommon dose response across states? extra slope on (A - 1) relative to Idaho")
    for seed in (0, 1, 2):
        fold = ds.make_folds(len(A), seed, cl)
        for r0 in range(3):
            _, eta, _ = ds.first_stage(prob.Q, prob.X, fold == r0)
            use = fold != r0
            n_fit = (prob.in_range & use).sum()
            K, k = max(3, int(round(n_fit ** 0.2))), max(3, int(round(n_fit ** (1 / 7))))
            fit = ds.outcome_fit(prob, eta, use, K, k, extra_cols=cols, cluster=cl)
            res.append(dict(seed=seed, score_fold=r0, coef=fit.extra["coef"].tolist(), se=fit.extra["se"].tolist()))
            print(f"  seed {seed} fold {r0}: " + "  ".join(
                f"{g} {fit.extra['coef'][j]:+.2f} (se {fit.extra['se'][j]:.2f})" for j, g in enumerate(others)), flush=True)
    return dict(states=others, reference="ID", runs=res)


def state_modifier(s, lq, keep, zc):
    st, A, cl = s.extras["state"][keep], s.extras["A"][keep], s.extras["claim_id"][keep]
    a_lo = float(np.floor(100 * A.min()) / 100)
    out = {}
    for label, Z in (("base", s.X[keep][:, zc]),
                     ("state", np.column_stack([s.X[keep][:, zc]] + [(st == g).astype(float) for g in ("LA", "MO", "NM")]))):
        prob = ds.DoseProblem(lq[keep], s.X[keep], A, s.Y[keep], Z, ds.DoseBasis(1.0, 1.0, l=3, a_lo=a_lo), a_lo=a_lo)
        est = ds.estimate(prob, n_bins=10, seed=0, a_grid=A_GRID, score_edges=EDGES, cluster=cl)
        eff = est["psi"] - est["psi"][:, [-1]]
        sd = (est["psi_rot"] - est["psi_rot"][:, :, [-1]]).std(0)
        fits = rotation_fits(prob, cluster=cl)
        by_state = {}
        for g in ("ID", "LA", "MO", "NM"):
            v = []
            for fit, eta, ev in fits:
                mm = ev & prob.in_range & (st == g) & (prob.Q > EDGES[2]) & (prob.Q < EDGES[4])
                v.append([float(np.mean(ds.gain(fit, eta[mm], prob.Z[mm], np.full(mm.sum(), a), 1.0))) for a in (0.6, 0.75)])
            by_state[g] = dict(rotations=v, mean_weeks=float(prob.Y[st == g].mean()))
        out[label] = dict(effect=eff.tolist(), rotation_sd=sd.tolist(), by_state=by_state)
        print(f"\nmodifiers Z = {label}: effect vs the cap by bin; then each state's own claimants at 0.59-0.91")
        for i in range(len(EDGES) - 1):
            print(f"   {np.exp(EDGES[i]):.2f}-{np.exp(EDGES[i + 1]):.2f}  " + "  ".join(
                f"{eff[i, j]:6.2f} ({sd[i, j]:4.2f})" for j in range(4)))
        for g, d in by_state.items():
            r = np.array(d["rotations"])
            print(f"   {g}: 60% {r[:, 0].mean():.2f} ({r[:, 0].min():.2f} to {r[:, 0].max():.2f}); "
                  f"75% {r[:, 1].mean():.2f} ({r[:, 1].min():.2f} to {r[:, 1].max():.2f}); mean weeks {d['mean_weeks']:.1f}", flush=True)
    return out


if __name__ == "__main__":
    main()
