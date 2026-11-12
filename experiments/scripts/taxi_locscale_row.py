"""Taxi Section 6: a location-scale first stage for the main specification R2-E1.

Robustness row only (scope, 2026-10-22: "location-scale first stage kept as a taxi
robustness check only"); the paper's theory does not cover it.

``--sample`` (default ``daytime``, the paper's row) picks the sample, score hours and
effect columns of ``taxi_section6.SAMPLES`` (output in runs/taxi_locscale_row/;
``--sample allhours`` writes to runs/taxi_locscale_row_allhours/).

Everything is R2-E1 (``taxi_section6.py``) except the first stage:

* Q = 2(sqrt(fare) - 1); score covariates distance, log distance, hour, day-of-week,
  weekday-hour x weekend and group indicators (``score_covariates(df, "distlog")``);
  baseline = score covariates; effect linear in distance; eps = 0.05;
* first stage Q = T + s(X) zeta, T = gamma'(1, X) by OLS, zeta independent of X,
  log s^2 linear in the scale covariates (default: the cubic B-spline in log distance
  plus the indicators, as in the 2026-10-21 runs), s normalized to median one.  The
  scale is fitted by Poisson pseudo-likelihood of eta-hat^2 (default; its score is
  orthogonal to gamma, the change the 2026-10-21 theory check named) or by Harvey's
  least squares of log eta-hat^2.  Both come from ``taxi_score_scales.py``.  Location
  and scale are fitted on the gamma fold;
* J = [max(l0 - 0.7 sd(zeta), q_0.0025(zeta)), q_0.9975(zeta)] from the full-sample
  pilot, held fixed, where l0 = q_eps((phi0 - T)/s) (the R2-E1 rule applied to zeta);
* trim window: endpoints q_eps and q_{1-eps} of t_phi0 = (phi0 - T)/s on the index
  fold, capped at the 0.5% and 99.5% quantiles of zeta-hat on the evaluation fold,
  re-estimated in every rotation and draw (the R2-E1 rule);
* utility U(phi) = mean_j sum_{i in window} [alpha(zeta_i) + d_j beta2] 1{zeta_i >=
  (phi - T_j)/s_j} / n_u over index-fold rides j and evaluation-fold rides i (the
  empirical index law of ``taxi_score_scales.rotation_ls``; R2-E1 instead uses a spline
  density of T);
* policy set [$6, $24], 61-point grid plus refinement; average of the four rotated
  thresholds; fold seed 1; bootstrap draw b uses the seeds of R2-E1's draw b (same
  resample, same split), so draws can be paired with ``taxi_section6/R2-E1_draws.jsonl``.

Variants in ``--point``: Poisson scale (the row), Harvey scale, Poisson scale with the
score covariates as scale covariates, Poisson scale with J and the window caps taken
within each arm (the 2026-10-21 rule: under the location-scale fit almost no treated
ride lies in the left tail of zeta-hat, so the pooled rule leaves alpha-hat at the
left end of the window to the treated spline's extrapolation), and an additive bridge
(s = 1, same empirical index law and window rules), which separates the scale from the
index-law estimator.

Run (from code/):
    python -m experiments.scripts.taxi_locscale_row --point
    python -m experiments.scripts.taxi_locscale_row --boot 999 --workers 2
    python -m experiments.scripts.taxi_locscale_row --summarize
Outputs in experiments/runs/taxi_locscale_row/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_score_scales as S
from experiments.scripts import taxi_section6 as R6

OUT = M.ROOT / "runs" / "taxi_locscale_row"            # daytime sample
OUTS = {"daytime": OUT, "allhours": M.ROOT / "runs" / "taxi_locscale_row_allhours"}
CFG = R6.SAMPLES["daytime"]                          # set from --sample in main()
SPEC = "R2-E1"
LAM, SCORE = R6.SPECS[SPEC]
STAGE = {"poisson": "locscale_pois", "harvey": "locscale", "additive": "additive"}
VARIANTS = {                     # key: (scale fit, scale covariates, both-arms J and caps)
    "ls_poisson": ("poisson", "spline", False),
    "ls_harvey": ("harvey", "spline", False),
    "ls_poisson_scorecov": ("poisson", "score", False),
    "ls_poisson_botharms": ("poisson", "spline", True),
    "additive_empirical": ("additive", "spline", False),
}
MAIN = "ls_poisson"


# ----------------------------------------------------------------------------- setup

def scale_covariates(df, data, kind):
    return M.score_covariates(df, "spline", CFG["hours"]) if kind == "spline" else data.Z


def central(z, D, q, both_arms=False):
    """Central range of z excluding q per tail: pooled (R2-E1), or within each arm and
    intersected (the 2026-10-21 rule, so that both arms are observed throughout)."""
    if not both_arms:
        return float(np.quantile(z, q)), float(np.quantile(z, 1 - q))
    return (max(float(np.quantile(z[D], q)), float(np.quantile(z[~D], q))),
            min(float(np.quantile(z[D], 1 - q)), float(np.quantile(z[~D], 1 - q))))


def make_design(data, Zs, stage, eps=R6.EPS, tail=R6.TAIL, window_tail=R6.WINDOW_TAIL,
                margin_sd=0.7, both_arms=False):
    """R2-E1's design rules (``M.make_design``) applied to zeta-hat from the full sample."""
    T, s, z = S.latent(data, Zs, np.arange(len(data.Q)), stage)
    D = data.D.astype(bool)
    phi0 = float(M.boxcox(M.FARE0, data.lam))
    r = (phi0 - T) / s                                   # treated iff zeta >= r
    l0, u0 = float(np.quantile(r, eps)), float(np.quantile(r, 1 - eps))
    Jc = central(z, D, tail, both_arms)
    J = (max(l0 - margin_sd * float(z.std()), Jc[0]), Jc[1])
    policy = tuple(float(v) for v in M.boxcox(np.array(M.POLICY_DOLLARS), data.lam))
    window = central(z, D, window_tail, both_arms)
    design = M.Design(J=J, eps=eps, phi0=phi0, T_support=(np.nan, np.nan), policy=policy,
                      effect=True, window=window, window_tail=window_tail,
                      window_estimated=True)
    return design, {"l0": l0, "u0": u0, "J": list(J), "window_caps_full": list(window),
                    "sd_zeta": float(z.std())}


# ----------------------------------------------------------------------------- estimator

def rotation(data, design, folds, Zs, stage, both_arms=False):
    """``S.rotation_ls`` with the window caps re-estimated on the evaluation fold."""
    T, s, e = S.latent(data, Zs, folds["gamma"], stage)
    fit = M.outcome_fit(data, e, folds["outcome"], design)
    I, U = folds["index"], folds["utility"]
    r = (design.phi0 - T[I]) / s[I]
    e_u = e[U]
    if design.window_estimated:
        caps = central(e_u, data.D[U].astype(bool), design.window_tail, both_arms)
    else:
        caps = design.window
    l_hat = max(float(np.quantile(r, design.eps)), caps[0])
    u_hat = min(float(np.quantile(r, 1 - design.eps)), caps[1])
    n_u = len(U)
    inside = (e_u >= l_hat) & (e_u <= u_hat)
    es = np.sort(e_u[inside])
    a = fit.alpha(es) - design.cost
    sufA = np.concatenate((np.cumsum(a[::-1])[::-1], [0.0])) / n_u
    sufF = (len(es) - np.arange(len(es) + 1)) / n_u
    T_I, s_I, xb_I = T[I], s[I], data.X[I] @ fit.beta2

    def value(phi):
        k = np.searchsorted(es, (phi - T_I) / s_I, side="left")
        return float(np.mean(sufA[k] + xb_I * sufF[k]))

    effect_all = fit.alpha(e_u) + data.X[U] @ fit.beta2 - design.cost

    def own(phi):
        return float(np.mean(inside * effect_all * (data.Q[U] >= phi)))

    binds = (float(np.quantile(r, design.eps)) > caps[0],
             float(np.quantile(r, 1 - design.eps)) < caps[1])
    rot = M.Rotation(value, None, own, fit.alpha, fit.beta2, (l_hat, u_hat), float(inside.mean()))
    return rot, binds


def four_role(data, design, seed, Zs, stage, curves=None, alpha_grid=None, own=True,
              grid_points=R6.GRID_POINTS, both_arms=False):
    blocks = np.array_split(np.random.default_rng(seed).permutation(len(data.Q)), 4)
    rots, phis, binds = [], [], []
    for k in range(4):
        folds = {r: blocks[(j + k) % 4] for j, r in enumerate(M.ROLES)}
        rot, b = rotation(data, design, folds, Zs, stage, both_arms)
        rots.append(rot)
        binds.append(b)
        phis.append(M._argmax(rot.value, *design.policy, m=grid_points)[0])
    pooled = lambda x: float(np.mean([r.value(x) for r in rots]))
    phi_pool, at_bound, n_local = M._argmax(pooled, *design.policy, m=grid_points)
    out = {"phi_avg": float(np.mean(phis)), "phi_pooled": phi_pool, "phi_rotations": phis,
           "at_boundary": at_bound, "local_maxima": n_local,
           "gain_vs_deployed": pooled(phi_pool) - pooled(design.phi0),
           "beta2": np.mean([r.beta2 for r in rots], axis=0).tolist(),
           "window": np.mean([r.window for r in rots], axis=0).tolist(),
           "window_share": float(np.mean([r.window_share for r in rots])),
           "trim_binds": [[bool(x) for x in b] for b in binds]}
    if curves is not None:
        out["U"] = np.array([pooled(x) for x in curves]) - pooled(design.phi0)
    if alpha_grid is not None:
        out["alpha"] = np.mean([r.alpha(alpha_grid) for r in rots], axis=0)
    if own:
        cuts = np.unique(data.Q)
        cuts = cuts[(cuts >= design.policy[0]) & (cuts <= design.policy[1])]
        vals = np.array([np.mean([r.own(c) for r in rots]) for c in cuts])
        out["own_cutoff"] = float(cuts[int(np.argmax(vals))])
    return out


# ----------------------------------------------------------------------------- diagnostics

def diagnostics(df, data, Zs, stage, design):
    """``S.diagnostics`` (R^2 of zeta-hat and zeta-hat^2 on the rich covariate functions
    of the paper's diagnostics, spread ratio across distance deciles, assignment gap)
    plus the spread ratio without the shortest decile, arm coverage at the ends of J,
    and folds of x -> T(x) + e s(x) in distance at the window caps."""
    g = S.diagnostics(df, data, Zs, stage, design)
    idx = np.arange(len(data.Q))
    T, s, z = S.latent(data, Zs, idx, stage)
    d = df["Trip_Distance"].to_numpy(float)
    dec = pd.qcut(d, 10, labels=False)
    sd = pd.Series(z).groupby(dec).std()
    out = {k: g[k] for k in ("R2_latent", "R2_latent2", "spread_ratio", "sd_by_decile",
                             "decile_mean_range", "assignment_gap", "scale_range")}
    out["spread_ratio_without_shortest_decile"] = float(sd.iloc[1:].max() / sd.iloc[1:].min())
    D = data.D.astype(bool)
    inJ = (z >= design.J[0]) & (z <= design.J[1])
    zq = np.quantile(z[inJ], [0.005, 0.995])
    out["share_treated_outer_half_percent_of_J"] = [float(D[inJ & (z <= zq[0])].mean()),
                                                   float(D[inJ & (z >= zq[1])].mean())]
    w = design.window
    below = inJ & (z < w[0])
    near = (z >= w[0]) & (z < np.quantile(z, 0.01))
    out["treated_left_end"] = {"J_to_cap": [int(D[below].sum()), int(below.sum())],
                               "cap_to_q01": [int(D[near].sum()), int(near.sum())]}
    if stage != "additive":
        out["folds"] = folds(df, data, Zs, s, design.window)
    return out


def folds(df, data, Zs, s, window):
    """Rides where distance -> T + e s decreases, for e at the window caps (as in
    ``taxi_ls_checks.folds``; log s^2 is recovered exactly by regressing 2 log s on
    (1, Zs), since the fit is linear in them)."""
    d = df["Trip_Distance"].to_numpy(float)
    A = M._xt(data.Z)
    g = np.linalg.lstsq(A, data.Q, rcond=None)[0]
    dT = g[1] / d.std() + g[2] / (np.log(d).std() * d)        # columns: std. d, std. log d
    delta = np.linalg.lstsq(M._xt(Zs), 2 * np.log(s), rcond=None)[0]
    x = np.log(d)
    lo, hi = x.min(), x.max()
    t = np.concatenate([np.repeat(lo, 4), np.quantile(x, [0.2, 0.4, 0.6, 0.8]), np.repeat(hi, 4)])
    B = BSpline.design_matrix(np.clip(x, lo, hi), t, 3).toarray()[:, 1:]
    nb = B.shape[1]
    if Zs.shape[1] == data.Z.shape[1] and np.allclose(Zs, data.Z):
        dlogs2 = delta[1] / d.std() + delta[2] / (np.log(d).std() * d)
    else:
        dB = np.column_stack([BSpline(t, np.eye(nb + 1)[k + 1], 3).derivative()(x) for k in range(nb)])
        dlogs2 = ((dB / B.std(0)) @ delta[1:1 + nb]) / d
    ds = s * 0.5 * dlogs2
    out = {}
    for name, e in (("lower", window[0]), ("upper", window[1])):
        bad = dT + e * ds <= 0
        out[name] = {"e": float(e), "share_folding": float(bad.mean())}
        if bad.any():
            out[name].update(distance_range=[float(d[bad].min()), float(d[bad].max())],
                             median_fare=float(np.median(data.fare[bad])))
    # Fold points e* = -T'/s' inside the window: the density of t_phi(X) = (phi - T)/s
    # has a spike at e* when phi = T + e* s; report those fares.
    with np.errstate(divide="ignore", invalid="ignore"):
        estar = np.where(ds != 0, -dT / ds, np.nan)
    crit = np.isfinite(estar) & (estar >= window[0]) & (estar <= window[1])
    if crit.any():
        fare = M.inv_boxcox(T_of(data, g)[crit] + estar[crit] * s[crit], data.lam)
        out["spikes"] = {"share_rides": float(crit.mean()),
                         "fare_q05_q50_q95": np.quantile(fare, [0.05, 0.5, 0.95]).tolist(),
                         "share_in_policy_set": float(np.mean((fare >= 6) & (fare <= 24))),
                         "share_within_1_dollar_of_12.5": float(np.mean(np.abs(fare - 12.5) <= 1))}
    return out


def T_of(data, g):
    return M._xt(data.Z) @ g


# ----------------------------------------------------------------------------- point estimates

def point(df, key, seed=1):
    t0 = time.time()
    fit, zs, both = VARIANTS[key]
    stage = STAGE[fit]
    data = M.make_data(df, LAM, CFG["effect"], score=SCORE, hour_set=CFG["hours"])
    Zs = scale_covariates(df, data, zs)
    design, dinfo = make_design(data, Zs, stage, both_arms=both)
    fare_grid = np.linspace(*M.POLICY_DOLLARS, 73)
    alpha_grid = np.linspace(design.window[0], design.window[1], 61)
    r = four_role(data, design, seed, Zs, stage, curves=M.boxcox(fare_grid, LAM),
                  alpha_grid=alpha_grid, both_arms=both)
    dol = lambda p: M.dollars(p, LAM)
    sd_eff = df[CFG["effect_cols"]].to_numpy(float).std(0)
    row = {"key": key, "scale_fit": fit, "scale_covariates": zs, "both_arms": both, "stage": stage,
           "phi": dol(r["phi_avg"]), "cell": M.fare_cell(dol(r["phi_avg"]), data.fare),
           "phi_pooled": dol(r["phi_pooled"]), "rotations": [dol(p) for p in r["phi_rotations"]],
           "own_cell": M.fare_cell(dol(r["own_cutoff"]), data.fare),
           "at_boundary": r["at_boundary"], "local_maxima": r["local_maxima"],
           "gain_cents": 100 * r["gain_vs_deployed"], "beta2_per_sd": r["beta2"],
           "beta2_per_mile": [b / sd for b, sd in zip(r["beta2"], sd_eff)],   # [0]: per mile
           "window": r["window"], "window_share": r["window_share"],
           "trim_binds": r["trim_binds"], "design": dinfo,
           "fare_grid": fare_grid.tolist(), "U": r["U"].tolist(),
           "latent_grid": alpha_grid.tolist(), "alpha": r["alpha"].tolist(),
           "diagnostics": diagnostics(df, data, Zs, stage, design)}
    row["seconds"] = time.time() - t0
    return row


# ----------------------------------------------------------------------------- bootstrap

_B: dict = {}


def _init(data, design, Zs, stage, seed, curves, alpha_grid, both):
    _B.update(data=data, design=design, Zs=Zs, stage=stage, seed=seed, curves=curves,
              alpha_grid=alpha_grid, both=both)


def _draw(b):
    t0 = time.time()
    data, seed = _B["data"], _B["seed"]
    idx = np.random.default_rng(seed + 1_000 + b).integers(0, len(data.Q), len(data.Q))
    r = four_role(data.subset(idx), _B["design"], seed + 50_000 + b, _B["Zs"][idx], _B["stage"],
                  curves=_B["curves"], alpha_grid=_B["alpha_grid"], own=False,
                  both_arms=_B["both"])
    dol = lambda p: M.dollars(p, data.lam)
    return {"b": b, "phi": dol(r["phi_avg"]), "phi_pooled": dol(r["phi_pooled"]),
            "rotations": [round(dol(p), 4) for p in r["phi_rotations"]],
            "at_boundary": bool(r["at_boundary"]), "beta2": r["beta2"],
            "U": np.round(r["U"], 7).tolist(), "alpha": np.round(r["alpha"], 6).tolist(),
            "seconds": round(time.time() - t0, 1)}


def bootstrap(df, key, boot, workers, seed, out):
    fit, zs, both = VARIANTS[key]
    stage = STAGE[fit]
    data = M.make_data(df, LAM, CFG["effect"], score=SCORE, hour_set=CFG["hours"])
    Zs = scale_covariates(df, data, zs)
    design, _ = make_design(data, Zs, stage, both_arms=both)
    curves = M.boxcox(np.linspace(*M.POLICY_DOLLARS, 73), LAM)
    alpha_grid = np.linspace(design.window[0], design.window[1], 61)
    path = out / f"{key}_draws.jsonl"
    done = {r["b"] for r in R6._load(path)}
    todo = [b for b in range(boot) if b not in done]
    t0 = time.time()
    print(f"[{key}] bootstrap: {len(done)} done, running {len(todo)} with {workers} workers",
          flush=True)
    if todo:
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(data, design, Zs, stage, seed, curves, alpha_grid, both)) as ex, \
                path.open("a") as fh:
            futures = [ex.submit(_draw, b) for b in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                fh.write(json.dumps(fut.result()) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)} draws, {el:.0f}s, eta {el / i * (len(todo) - i) / 60:.0f} min",
                          flush=True)


def summarize(key, out, boot):
    rows = sorted(R6._load(out / f"{key}_draws.jsonl"), key=lambda r: r["b"])
    rows = [r for r in rows if r["b"] < boot]
    pts = json.loads((out / "point.json").read_text())["rows"]
    est = next(r for r in pts if r["key"] == key)
    fares = np.unique(M.load_frame(CFG["frame"])["Fare_Amt"].to_numpy(float))
    s = R6.summarize_boot(est, rows, fares)
    s["seconds_per_draw_median"] = float(np.median([r["seconds"] for r in rows]))
    # pair with the main specification's draws (same resample and split for each b)
    main_path = CFG["out"] / f"{SPEC}_draws.jsonl"
    main = {r["b"]: r["phi"] for r in R6._load(main_path)}
    common = [r for r in rows if r["b"] in main]
    if common:
        diff = np.array([r["phi"] - main[r["b"]] for r in common])
        s["paired_with_R2E1"] = {"n": len(common), "mean_diff": float(diff.mean()),
                                 "sd_diff": float(diff.std(ddof=1)),
                                 "pct95_diff": np.quantile(diff, [0.025, 0.975]).tolist(),
                                 "corr": float(np.corrcoef([r["phi"] for r in common],
                                                           [main[r["b"]] for r in common])[0, 1])}
    path = out / "summary.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res[key] = {"estimate": {k: est[k] for k in ("phi", "cell", "phi_pooled", "rotations",
                                                   "own_cell", "beta2_per_mile", "window_share")},
                "boot": s}
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[{key}] B={s['B']} sd {s['sd']:.3f} 95% pct [{s['pct95'][0]:.2f}, {s['pct95'][1]:.2f}] "
          f"90% [{s['pct90'][0]:.2f}, {s['pct90'][1]:.2f}] boundary {s['share_boundary']:.3f} "
          f"cells {s['cells']}", flush=True)
    if "paired_with_R2E1" in s:
        p = s["paired_with_R2E1"]
        print(f"  paired with R2-E1 (n={p['n']}): mean diff {p['mean_diff']:+.3f}, sd {p['sd_diff']:.3f}, "
              f"95% [{p['pct95_diff'][0]:+.2f}, {p['pct95_diff'][1]:+.2f}], corr {p['corr']:.2f}")
    return s


def fmt(r):
    g = r["diagnostics"]
    return (f"{r['key']:20s} phi {r['phi']:6.2f} cell {r['cell']:5.2f} pooled {r['phi_pooled']:6.2f} "
            f"rot {min(r['rotations']):5.2f}-{max(r['rotations']):5.2f} own {r['own_cell']:5.2f} "
            f"gain {r['gain_cents']:5.2f}c b2/mi {r['beta2_per_mile'][0]:.3f} win {r['window_share']:.3f} "
            f"| R2 {g['R2_latent']:.3f} R2sq {g['R2_latent2']:.3f} spread {g['spread_ratio']:.2f} "
            f"(w/o shortest {g['spread_ratio_without_shortest_decile']:.2f}) gap {g['assignment_gap']:.3f} "
            f"[{r['seconds']:.0f}s]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--point", action="store_true", help="point estimates for all variants")
    ap.add_argument("--keys", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--boot-key", default=MAIN, choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--sample", default="daytime", choices=list(R6.SAMPLES))
    ap.add_argument("--out", type=Path, default=None, help="default: the sample's run folder")
    a = ap.parse_args(argv)
    global CFG
    CFG = R6.SAMPLES[a.sample]
    a.out = a.out or OUTS[a.sample]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.point:
        df = M.load_frame(CFG["frame"])
        path = a.out / "point.json"
        res = json.loads(path.read_text()) if path.exists() else {"rows": []}
        res.update(spec=SPEC, lam=LAM, score=SCORE, eps=R6.EPS, tail=R6.TAIL,
                   window_tail=R6.WINDOW_TAIL, grid_points=R6.GRID_POINTS, seed=a.seed)
        for key in a.keys:
            row = point(df, key, a.seed)
            print(fmt(row), flush=True)
            res["rows"] = [r for r in res["rows"] if r["key"] != key] + [row]
            path.write_text(json.dumps(res, indent=1) + "\n")
    if a.boot:
        df = M.load_frame(CFG["frame"])
        bootstrap(df, a.boot_key, a.boot, a.workers, a.seed, a.out)
        summarize(a.boot_key, a.out, a.boot)
    elif a.summarize:
        summarize(a.boot_key, a.out, 999)


if __name__ == "__main__":
    main()
