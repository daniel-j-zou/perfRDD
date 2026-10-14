"""Exploratory: does redesigning the score beat moving the cutoff? (taxi, GPA)

A possible follow-up paper chooses a whole linear score S = kappa*eta + lambda'X
and a cutoff, not just a cutoff on the observed score Q = gamma'X~ + eta.  This
script asks how much that could matter in the two applications already in the
repository, under the paper's differing-slopes effect model

    effect(x, e) = alpha(e) + x'beta2,

fitted exactly as in ``taxi_differing_slopes.py`` (spline in eta with
n_treated^{1/5} knots, ridge on the beta2 and spline blocks chosen by
cross-validation; ``--ridge none`` refits without the penalty).

Population and rules.  Everything is evaluated on the hard-trimmed window
{l0 <= eta_hat <= u0} (eps = 0.1), with the historical rule kept outside it,
so values are per unit in the window and relative to treating no one there.
Compared rules:

* deployed:      the historical cutoff on Q;
* best cutoff:   the best cutoff on Q;
* best score:    the best rule 1{w'z >= phi}, z = (eta/sd(eta), X), ||w|| = 1;
* model optimum: treat iff the predicted net effect is positive, the best
                 rule of any shape under the fitted model (an upper bound).

Honest split.  The sample is split in two halves; the outcome model is fitted
on each half.  Rules are selected with one half's model on the other half's
units and evaluated with the other half's model on fresh units, then the
roles are swapped and the two evaluations averaged.  The first stage gamma
is fitted once on the full sample.

Extrapolation.  At fixed eta, a unit's historical arm is decided by Q alone,
so a unit switched to the other arm lies at standardized-X distance
|Q - cutoff| / ||gamma_X|| from the nearest covariate value at which that arm
was observed for its eta.  The script reports that distance for switched units.

The values are model predictions: every number relies on the linear-in-X
effect holding across the window, which is exactly what score redesign leans
on.  Point estimates only.

Run:
    python -m experiments.scripts.score_redesign_explore --out experiments/runs/score_redesign_explore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from experiments._core.registry import load
from experiments.datasets.taxi.adapter import load_haggag_paci_vendor
from experiments.methods.perfrdd import _basis_params, _eval_basis
from experiments.scripts.taxi_differing_slopes import LAMBDA_GRID, _clean, _design, _solve

EPS = 0.10
N_RANDOM_STARTS = 6
BANDWIDTHS = (0.3, 0.1, 0.03)   # surrogate smoothing, in units of the unit-norm score


def load_taxi() -> dict:
    sample = load_haggag_paci_vendor("VTS")
    Q, X, Y = _clean(sample)          # X already standardized by the adapter
    return dict(name="taxi", Q=Q, X=X, Y=Y, thr=15.0, direction="above",
                xnames=["distance (mi)", "passengers", "hour of day", "day of week"],
                xscale=np.asarray(sample.extras["control_standardization_scales"], float),
                costs=[0.0, 0.20], unit="$ tip per ride")


def load_gpa() -> dict:
    sample = load("gpa")
    X = np.asarray(sample.X, float)
    scale = X.std(axis=0)
    return dict(name="gpa", Q=np.asarray(sample.Q, float), X=(X - X.mean(axis=0)) / scale,
                Y=np.asarray(sample.Y, float), thr=0.0, direction="below",
                xnames=list(sample.feature_names), xscale=scale,
                costs=[0.0, "p50", "p75"], unit="next-term GPA points (selected sample)")


LOADERS = {"taxi": load_taxi, "gpa": load_gpa}


def fit_model(Qs, X, Y, eta, D, info, ridge, rng) -> dict:
    """Differing-slopes outcome fit; returns the alpha spline coefficients and beta2."""
    H, pen, nb, nx = _design(Qs, X, eta, D, info, True)
    if ridge == "cv":
        tr = rng.random(len(Y)) < 0.8
        mses = [np.mean((Y[~tr] - H[~tr] @ _solve(H[tr], Y[tr], pen, lam)) ** 2)
                for lam in LAMBDA_GRID]
        lam = LAMBDA_GRID[int(np.argmin(mses))]
        coef = _solve(H, Y, pen, lam)
    else:   # the intercept and baseline spline are collinear; lstsq returns the min-norm fit
        lam = 0.0
        coef = np.linalg.lstsq(H, Y, rcond=None)[0]
    return dict(alpha=coef[-nb:], beta2=coef[1 + nx:1 + 2 * nx], lam=lam)


def predict(model, eta, X, info) -> np.ndarray:
    return _eval_basis(eta, info) @ model["alpha"] + X @ model["beta2"]


def best_threshold(score: np.ndarray, gain: np.ndarray) -> tuple[float, float]:
    """Exact best rule 1{score >= t}; ties are moved together.  Returns (t, mean value)."""
    order = np.argsort(-score, kind="stable")
    s, g = score[order], gain[order]
    cut = np.r_[s[:-1] > s[1:], True]                 # a cut after position k is admissible
    values = np.r_[0.0, np.cumsum(g)[cut]]
    ends = np.r_[-1, np.nonzero(cut)[0]]
    j = int(np.argmax(values))
    t = np.inf if ends[j] < 0 else float(s[ends[j]])
    return t, float(values[j] / len(s))


def _surrogate(params, Z, gain, h):
    v, phi = params[:-1], params[-1]
    nv = np.linalg.norm(v)
    w = v / nv
    u = (Z @ w - phi) / h
    val = np.mean(gain * norm.cdf(u))
    dens = gain * norm.pdf(u) / h
    gw = Z.T @ dens / len(gain)
    gv = (gw - w * (w @ gw)) / nv
    return -val, -np.r_[gv, -np.mean(dens)]


def best_score(Z: np.ndarray, gain: np.ndarray, starts: list[np.ndarray], rng) -> dict:
    """Best rule 1{w'z >= phi}: smoothed search from several starts, exact final threshold."""
    starts = list(starts) + [rng.standard_normal(Z.shape[1]) for _ in range(N_RANDOM_STARTS)]
    best = dict(value=-np.inf)
    for w0 in starts:
        w = w0 / np.linalg.norm(w0)
        phi, _ = best_threshold(Z @ w, gain)
        if not np.isfinite(phi):
            phi = float(np.max(Z @ w)) + 1.0
        x = np.r_[w, phi]
        for h in BANDWIDTHS:
            x = minimize(_surrogate, x, args=(Z, gain, h), jac=True, method="L-BFGS-B").x
        w = x[:-1] / np.linalg.norm(x[:-1])
        phi, value = best_threshold(Z @ w, gain)
        if value > best["value"]:
            best = dict(w=w, phi=phi, value=value)
    return best


def as_cutoff_rule(w, phi, sd_eta, gamma, sign, xscale, xnames) -> dict | None:
    """Write 1{w'z >= phi} as a covariate-dependent cutoff on the original score Q."""
    w0, wx = w[0], w[1:]
    if w0 <= 1e-8:
        return None                                    # not monotone in Q
    k = w0 / sd_eta                                    # coefficient on eta = Qs - gamma0 - gamma_X'X
    q0 = phi / k + gamma[0]
    delta = wx / k - gamma[1:]                         # Qs >= q0 - delta'X
    per_sd = -sign * delta                             # change of the Q cutoff per +1 sd of X_j
    return dict(cutoff_at_mean_covariates=float(sign * q0),
                cutoff_shift_per_sd={n: float(v) for n, v in zip(xnames, per_sd)},
                cutoff_shift_per_unit={n: float(v / s) for n, v, s in zip(xnames, per_sd, xscale)})


def run_dataset(name: str, ridge: str, seed: int = 0) -> dict:
    data = LOADERS[name]()
    Q, X, Y, thr = data["Q"], data["X"], data["Y"], data["thr"]
    ok = np.isfinite(Q) & np.isfinite(Y) & np.isfinite(X).all(axis=1)
    Q, X, Y = Q[ok], X[ok], Y[ok]
    sign = 1.0 if data["direction"] == "above" else -1.0
    Qs, thr_s = sign * Q, sign * thr                    # treated iff Qs >= thr_s

    Xd = np.column_stack((np.ones(len(Qs)), X))
    gamma = np.linalg.lstsq(Xd, Qs, rcond=None)[0]
    eta = Qs - Xd @ gamma
    T = Qs - eta
    lo, hi = np.percentile(eta, 0.5), np.percentile(eta, 99.5)
    l0 = max(min(thr_s - np.quantile(T, 1 - EPS), thr_s - np.quantile(T, EPS)), lo)
    u0 = min(max(thr_s - np.quantile(T, 1 - EPS), thr_s - np.quantile(T, EPS)), hi)
    D = (Qs >= thr_s).astype(float)
    kn = max(4, int(round(int(D.sum()) ** (1 / 5)))) + 1
    info = _basis_params(kn, (lo, hi))
    win = (eta >= l0) & (eta <= u0)
    sd_eta = float(eta.std())
    gnorm = float(np.linalg.norm(gamma[1:]))
    Z = np.column_stack((eta / sd_eta, X))

    rng = np.random.default_rng(seed)
    half = rng.random(len(Y)) < 0.5
    halves = (half, ~half)
    models = [fit_model(Qs[h], X[h], Y[h], eta[h], D[h], info, ridge, rng) for h in halves]
    # out-of-half predicted effect for every unit (used to set data-driven costs only)
    pooled = np.empty(len(Y))
    for k, h in enumerate(halves):
        pooled[h] = predict(models[1 - k], eta[h], X[h], info)

    costs = []
    for c in data["costs"]:
        costs.append(float(np.quantile(pooled[win], int(c[1:]) / 100)) if isinstance(c, str) else float(c))

    out = dict(dataset=name, ridge=ridge, n=int(len(Y)), n_window=int(win.sum()),
               window_share=float(win.mean()), window_eta=[float(l0), float(u0)],
               treated_share_window=float(D[win].mean()), first_stage_r2=float(1 - eta.var() / Qs.var()),
               sd_eta=sd_eta, gamma_X_norm=gnorm, knots=kn, unit=data["unit"],
               ridge_lambda=[m["lam"] for m in models],
               beta2_per_sd=[{n: float(v) for n, v in zip(data["xnames"], m["beta2"])} for m in models],
               effect_in_window=dict(mean=float(pooled[win].mean()), sd=float(pooled[win].std()),
                                     share_positive=float(np.mean(pooled[win] > 0))),
               results=[])

    for c in costs:
        rounds = []
        for sel in (0, 1):                       # select on half `sel`, evaluate on the other half
            ev = 1 - sel
            hs, he = halves[sel] & win, halves[ev] & win
            g_sel = predict(models[ev], eta[hs], X[hs], info) - c      # selection model: fitted on half ev
            g_eval = predict(models[sel], eta[he], X[he], info) - c    # evaluation model: fitted on half sel
            # the unrestricted rule uses the selection model at the evaluation units
            g_rule = predict(models[ev], eta[he], X[he], info) - c

            a_hat = _eval_basis(eta[hs], info) @ models[ev]["alpha"]
            slope = np.polyfit(eta[hs], a_hat, 1)[0]
            starts = [np.r_[sd_eta, gamma[1:]], np.r_[slope * sd_eta, models[ev]["beta2"]]]
            t_q, _ = best_threshold(Qs[hs], g_sel)
            score = best_score(Z[hs], g_sel, starts, rng)

            deployed = Qs[he] >= thr_s
            rules = {
                "deployed": deployed,
                "best_cutoff": Qs[he] >= t_q,
                "best_score": Z[he] @ score["w"] >= score["phi"],
                "model_optimum": g_rule > 0,
                "treat_all_window": np.ones(he.sum(), bool),
            }
            r = {}
            for key, treat in rules.items():
                switched = treat != deployed
                gap = np.abs(Qs[he][switched] - thr_s)
                r[key] = dict(value=float(np.mean(g_eval * treat)), treated_share=float(treat.mean()),
                              switched_share=float(switched.mean()),
                              extrap_sd_median=float(np.median(gap) / gnorm) if gap.size else 0.0,
                              extrap_sd_p90=float(np.quantile(gap, 0.9) / gnorm) if gap.size else 0.0)
            r["best_cutoff"]["cutoff"] = float(sign * t_q)
            r["best_score"]["w"] = score["w"].tolist()
            r["best_score"]["as_cutoff_rule"] = as_cutoff_rule(
                score["w"], score["phi"], sd_eta, gamma, sign, data["xscale"], data["xnames"])
            rounds.append(r)
        avg = {k: {m: float(np.mean([rd[k][m] for rd in rounds]))
                   for m in ("value", "treated_share", "switched_share", "extrap_sd_median", "extrap_sd_p90")}
               for k in rounds[0]}
        out["results"].append(dict(cost=c, average=avg, rounds=rounds))
    return out


def summarize(res: dict) -> str:
    lines = [f"## {res['dataset']} (ridge={res['ridge']}): n={res['n']:,}, window {res['n_window']:,} "
             f"({100 * res['window_share']:.0f}%), first-stage R2={res['first_stage_r2']:.2f}, "
             f"unit = {res['unit']}",
             f"predicted effect in window: mean {res['effect_in_window']['mean']:.3f}, "
             f"sd {res['effect_in_window']['sd']:.3f}, positive for "
             f"{100 * res['effect_in_window']['share_positive']:.0f}%"]
    for block in res["results"]:
        a = block["average"]
        lines.append(f"\ncost c = {block['cost']:.3f}")
        lines.append("| rule | value | gain over deployed | treated | switched | extrap. sd (median / p90) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key in ("deployed", "best_cutoff", "best_score", "model_optimum", "treat_all_window"):
            v = a[key]
            lines.append(f"| {key} | {v['value']:.4f} | {v['value'] - a['deployed']['value']:+.4f} | "
                         f"{100 * v['treated_share']:.0f}% | {100 * v['switched_share']:.0f}% | "
                         f"{v['extrap_sd_median']:.2f} / {v['extrap_sd_p90']:.2f} |")
        for k, rd in enumerate(block["rounds"]):
            rule = rd["best_score"]["as_cutoff_rule"]
            desc = ("not monotone in Q" if rule is None else
                    f"cutoff {rule['cutoff_at_mean_covariates']:.2f} at mean X; shift per unit: " +
                    ", ".join(f"{n} {v:+.3f}" for n, v in rule["cutoff_shift_per_unit"].items()))
            lines.append(f"- round {k + 1}: best cutoff {rd['best_cutoff']['cutoff']:.2f}; best score: {desc}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(LOADERS), default=sorted(LOADERS, reverse=True))
    parser.add_argument("--ridge", nargs="+", choices=("cv", "none"), default=["cv", "none"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    text = []
    for name in args.datasets:
        for ridge in args.ridge:
            res = run_dataset(name, ridge)
            (args.out / f"{name}_{ridge}.json").write_text(json.dumps(res, indent=2) + "\n")
            block = summarize(res)
            print(block + "\n", flush=True)
            text.append(block)
    (args.out / "summary.md").write_text("\n\n".join(text) + "\n")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
