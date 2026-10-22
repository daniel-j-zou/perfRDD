"""Taxi: score transforms and a location-scale first stage (exploratory).

The nine-way comparison (``taxi_four_role_grid.py``) keeps the fare in dollars, where
the spread of eta-hat doubles across trip length and the share treated is U-shaped in
eta-hat, which X independent of eta rules out.  This script tries two remedies:

* Box-Cox scales for Q (square root, log) with distance terms suited to each scale;
* a location-scale first stage in dollars, Q = gamma'(1, Z) + s(Z) zeta with zeta
  independent of Z, where log s(Z)^2 = gamma_s'(1, Zs) (Zs: spline in log distance and
  the indicators), normalized to median one.  gamma_s is fitted either by least squares
  of log eta-hat^2 (Harvey) or by Poisson pseudo-likelihood of eta-hat^2 on
  exp(gamma_s'(1, Zs)), whose score is orthogonal to gamma (this_week.tex, "A
  location-scale first stage").  The standardized residual zeta is the time charge
  beyond typical in dollars for a ride of median variability.  The outcome model, the
  trim and the utility use zeta; the utility integrates over the joint distribution of
  (T, s, X) on the index fold:

      U(phi) = mean_j  sum_{i in window} [alpha(zeta_i) + X_j'beta2] 1{zeta_i >= (phi - T_j)/s_j} / n_u,

  which is the paper's convolution with the density of T replaced by that of (T, s).
  With s = 1 it is the additive estimator with an empirical index distribution.

Every setting: score covariates = distance terms + hour, day-of-week, weekday-hour x
weekend and group indicators (also the baseline); effect linear in distance; eps = 0.05;
J = the range holding all but the 0.25% tails of the latent type in each arm, and the
trim window capped at the 0.5% tails in each arm, so that both arms are observed
throughout (the nine-way grid used the central 99.5% and 99% of all rides).

Outputs, per setting: alpha on the window, the share treated against the latent type
(observed, and implied by independence), and U(phi) - U(15) against the fare cutoff.

Run (from code/):
    python -m experiments.scripts.taxi_score_scales
"""
from __future__ import annotations

import argparse
import json
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from experiments.scripts import taxi_four_role as M
from experiments.scripts import taxi_four_role_grid as G

OUT = M.ROOT / "runs" / "taxi_score_scales"
EPS = 0.05
GRID_POINTS = 121

# (key, Box-Cox lambda, distance terms, first stage)
SETTINGS = [
    ("sqrt_d", 0.5, "d", "additive"),
    ("sqrt_dlogd", 0.5, "d+logd", "additive"),
    ("sqrt_meter", 0.5, "sqrtmeter", "additive"),
    ("sqrt_spline", 0.5, "spline", "additive"),
    ("log_logd", 0.0, "logd", "additive"),
    ("log_meter", 0.0, "logmeter", "additive"),
    ("log_meter_logd", 0.0, "logmeter+logd", "additive"),
    ("log_spline", 0.0, "spline", "additive"),
    ("usd_dlogd", 1.0, "d+logd", "additive"),
    ("usd_dlogd_ls", 1.0, "d+logd", "locscale"),
    ("usd_dlogd_ls_pois", 1.0, "d+logd", "locscale_pois"),
    ("usd_spline_ls", 1.0, "spline", "locscale"),
]
LABEL_TERMS = {"d": "distance", "d+logd": "distance + log distance",
               "sqrtmeter": r"$\sqrt{2.5+2\,\mathrm{distance}}$", "spline": "spline in log distance",
               "logd": "log distance", "logmeter": r"$\log(2.5+2\,\mathrm{distance})$",
               "logmeter+logd": r"$\log(2.5+2\,\mathrm{distance})$ + log distance"}
LABEL_SCALE = {0.5: "square root", 0.0: "log", 1.0: "dollars"}


def label(key):
    _, lam, terms, stage = next(s for s in SETTINGS if s[0] == key)
    tail = {"locscale": ", location-scale", "locscale_pois": ", location-scale (Poisson scale)"}.get(stage, "")
    return f"{LABEL_SCALE[lam]}: {LABEL_TERMS[terms]}{tail}"


def dist_terms(d, kind):
    if kind == "d":
        return d[:, None]
    if kind == "d+logd":
        return np.column_stack((d, np.log(d)))
    if kind == "sqrtmeter":
        return np.sqrt(2.5 + 2 * d)[:, None]
    if kind == "logd":
        return np.log(d)[:, None]
    if kind == "logmeter":
        return np.log(2.5 + 2 * d)[:, None]
    if kind == "logmeter+logd":
        return np.column_stack((np.log(2.5 + 2 * d), np.log(d)))
    if kind == "spline":
        return M._bspline_cols(np.log(d), [0.2, 0.4, 0.6, 0.8])
    raise ValueError(kind)


def score_Z(df, kind):
    indicators = M.score_covariates(df, "linear")[:, 1:]          # already standardized
    return np.column_stack((M._standardize(dist_terms(df["Trip_Distance"].to_numpy(float), kind)),
                            indicators))


def make_data(df, lam, terms):
    fare = df["Fare_Amt"].to_numpy(float)
    return M.Data(M.boxcox(fare, lam), score_Z(df, terms), M.effect_covariates(df, "distance"),
                  df["Tip_Amt"].to_numpy(float), fare, lam)


# ----------------------------------------------------------------------------- location-scale

def fit_scale(Zs, eta, idx):
    """s(Z) with log s^2 = delta'(1, Zs), fitted by least squares of log eta^2 on
    ``idx`` (Harvey's multiplicative heteroskedasticity; under eps independent of Z
    the intercept absorbs E log eps^2), normalized to median one."""
    A = M._xt(Zs)
    y = np.log(eta[idx] ** 2 + 1e-6 * np.var(eta[idx]))
    delta = np.linalg.lstsq(A[idx], y, rcond=None)[0]
    s = np.exp(0.5 * (A @ delta))
    return s / np.median(s[idx])


def fit_scale_poisson(Zs, eta, idx, iters=200):
    """s(Z) with s^2 = exp(gamma_s'(1, Zs)) fitted to eta^2 on ``idx`` by Poisson
    pseudo-likelihood, damped Newton from the Harvey fit; median one.  Its score
    E[(1, Zs)(eta^2 - s^2)] has zero derivative in gamma because E(eta | Z) = 0, so the
    generated residuals do not enter its first-order expansion."""
    A = M._xt(Zs)
    Ai, y = A[idx], eta[idx] ** 2
    d = np.linalg.lstsq(Ai, np.log(y + 1e-6 * y.mean()), rcond=None)[0]
    d[0] = np.log(y.mean() / np.mean(np.exp(Ai[:, 1:] @ d[1:])))
    obj = lambda v: np.mean(np.exp(Ai @ v) - y * (Ai @ v))
    for _ in range(iters):
        mu = np.exp(Ai @ d)
        step = np.linalg.solve((Ai * mu[:, None]).T @ Ai, Ai.T @ (y - mu))
        t = 1.0
        while obj(d + t * step) > obj(d) and t > 1e-8:
            t /= 2
        d = d + t * step
        if np.max(np.abs(t * step)) < 1e-9:
            break
    s = np.exp(0.5 * (A @ d))
    return s / np.median(s[idx])


def latent(data, Zs, idx, stage):
    """Index T, scale s and standardized residual from the first stage fitted on ``idx``."""
    A = M._xt(data.Z)
    g = np.linalg.lstsq(A[idx], data.Q[idx], rcond=None)[0]
    T = A @ g
    eta = data.Q - T
    if stage == "locscale":
        s = fit_scale(Zs, eta, idx)
    elif stage == "locscale_pois":
        s = fit_scale_poisson(Zs, eta, idx)
    else:
        s = np.ones_like(eta)
    return T, s, eta / s


def make_design_ls(data, Zs, eps, stage, tail=0.0025, window_tail=0.005):
    idx = np.arange(len(data.Q))
    T, s, e = latent(data, Zs, idx, stage)
    D = data.D.astype(bool)

    def both_arms(q):
        """Central range observed in both arms: in the tails of a heteroskedastic
        residual one arm can be absent (e.g. long fast rides, all treated)."""
        return (max(float(np.quantile(e[D], q)), float(np.quantile(e[~D], q))),
                min(float(np.quantile(e[D], 1 - q)), float(np.quantile(e[~D], 1 - q))))

    J, window = both_arms(tail), both_arms(window_tail)
    policy = tuple(float(v) for v in M.boxcox(np.array(M.POLICY_DOLLARS), data.lam))
    phi0 = float(M.boxcox(M.FARE0, data.lam))
    return M.Design(J=J, eps=eps, phi0=phi0, T_support=(np.nan, np.nan), policy=policy,
                    effect=True, window=window)


def rotation_ls(data, design, folds, Zs, stage):
    T, s, e = latent(data, Zs, folds["gamma"], stage)
    fit = M.outcome_fit(data, e, folds["outcome"], design)
    I = folds["index"]
    r = (design.phi0 - T[I]) / s[I]                 # treated iff latent >= r
    l_hat = max(float(np.quantile(r, design.eps)), design.window[0])
    u_hat = min(float(np.quantile(r, 1 - design.eps)), design.window[1])
    U = folds["utility"]
    n_u = len(U)
    e_u = e[U]
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

    return M.Rotation(value, None, own, fit.alpha, fit.beta2, (l_hat, u_hat), float(inside.mean()))


def four_role_ls(data, design, seed, Zs, stage, curves, alpha_grid, grid_points=GRID_POINTS):
    blocks = np.array_split(np.random.default_rng(seed).permutation(len(data.Q)), 4)
    rots, phis = [], []
    for k in range(4):
        folds = {r: blocks[(j + k) % 4] for j, r in enumerate(M.ROLES)}
        rot = rotation_ls(data, design, folds, Zs, stage)
        rots.append(rot)
        phis.append(M._argmax(rot.value, *design.policy, m=grid_points)[0])
    pooled = lambda x: float(np.mean([r.value(x) for r in rots]))
    phi_pool, at_bound, n_local = M._argmax(pooled, *design.policy, m=grid_points)
    cuts = np.unique(data.Q)
    cuts = cuts[(cuts >= design.policy[0]) & (cuts <= design.policy[1])]
    own = np.array([np.mean([r.own(c) for r in rots]) for c in cuts])
    return {"phi_avg": float(np.mean(phis)), "phi_pooled": phi_pool, "phi_rotations": phis,
            "at_boundary": at_bound, "local_maxima": n_local,
            "gain_vs_deployed": pooled(phi_pool) - pooled(design.phi0),
            "beta2": np.mean([r.beta2 for r in rots], axis=0).tolist(),
            "window": np.mean([r.window for r in rots], axis=0).tolist(),
            "window_share": float(np.mean([r.window_share for r in rots])),
            "alpha": np.mean([r.alpha(alpha_grid) for r in rots], axis=0),
            "U": np.array([pooled(x) for x in curves]) - pooled(design.phi0),
            "own_cutoff": float(cuts[int(np.argmax(own))])}


# ----------------------------------------------------------------------------- diagnostics

def diagnostics(df, data, Zs, stage, design):
    """Independence diagnostics of the latent type and the share treated against it."""
    idx = np.arange(len(data.Q))
    T, s, e = latent(data, Zs, idx, stage)
    F = G.rich_features(df)

    def r2(v):
        c = np.linalg.lstsq(F, v, rcond=None)[0]
        return float(1 - np.var(v - F @ c) / np.var(v))

    dec = pd.qcut(df["Trip_Distance"].to_numpy(float), 10, labels=False)
    sd = pd.Series(e).groupby(dec).std()
    mean = pd.Series(e).groupby(dec).mean()
    D = data.D.astype(bool)
    edges = np.linspace(design.J[0], design.J[1], 51)
    mid = 0.5 * (edges[1:] + edges[:-1])
    b = np.digitize(e, edges) - 1
    counts = np.array([(b == k).sum() for k in range(50)])
    share = np.array([D[b == k].mean() if counts[k] >= 100 else np.nan for k in range(50)])
    r = np.sort((design.phi0 - T) / s)                   # treated iff latent >= r
    grid = np.linspace(design.J[0], design.J[1], 200)
    implied = np.searchsorted(r, grid, side="right") / len(r)
    imp_mid = np.searchsorted(r, mid, side="right") / len(r)
    inw = (mid >= design.window[0]) & (mid <= design.window[1]) & np.isfinite(share)
    gap = float(np.sum(counts[inw] * np.abs(share[inw] - imp_mid[inw])) / counts[inw].sum())
    return {"R2_latent": r2(e), "R2_latent2": r2(e ** 2), "spread_ratio": float(sd.max() / sd.min()),
            "sd_by_decile": (sd / sd.median()).round(3).tolist(),
            "decile_mean_range": [float(mean.min()), float(mean.max())],
            "assignment_gap": gap, "share_mid": mid.tolist(), "share": share.tolist(),
            "implied_grid": grid.tolist(), "implied": implied.tolist(),
            "scale_range": [float(np.quantile(s, 0.01)), float(np.quantile(s, 0.99))]}


# ----------------------------------------------------------------------------- one setting

_B: dict = {}


def _init(data, design, seed, Zs, stage, curves, alpha_grid):
    _B.update(data=data, design=design, seed=seed, Zs=Zs, stage=stage, curves=curves,
              alpha_grid=alpha_grid)


def _draw(b):
    d = _B
    idx = np.random.default_rng(d["seed"] + 1_000 + b).integers(0, len(d["data"].Q), len(d["data"].Q))
    r = four_role_ls(d["data"].subset(idx), d["design"], d["seed"] + 50_000 + b, d["Zs"][idx],
                     d["stage"], d["curves"], d["alpha_grid"])
    return r["phi_avg"], r["U"], r["alpha"]


def run_setting(df, key, seed=1, boot=0, workers=2):
    _, lam, terms, stage = next(s for s in SETTINGS if s[0] == key)
    t0 = time.time()
    data = make_data(df, lam, terms)
    Zs = M.score_covariates(df, "spline")
    design = make_design_ls(data, Zs, EPS, stage)
    fare_grid = np.linspace(*M.POLICY_DOLLARS, 73)
    curves = M.boxcox(fare_grid, lam)
    alpha_grid = np.linspace(design.window[0], design.window[1], 61)
    main = four_role_ls(data, design, seed, Zs, stage, curves, alpha_grid)
    d = lambda p: M.dollars(p, lam)
    row = {"key": key, "label": label(key), "lam": lam, "terms": terms, "stage": stage,
           "phi": d(main["phi_avg"]), "cell": M.fare_cell(d(main["phi_avg"]), data.fare),
           "phi_pooled": d(main["phi_pooled"]), "rotations": [d(p) for p in main["phi_rotations"]],
           "own_cell": M.fare_cell(d(main["own_cutoff"]), data.fare),
           "at_boundary": main["at_boundary"], "local_maxima": main["local_maxima"],
           "gain_cents": 100 * main["gain_vs_deployed"], "beta2": main["beta2"],
           "window": main["window"], "window_share": main["window_share"],
           "fare_grid": fare_grid.tolist(), "U": main["U"].tolist(),
           "latent_grid": alpha_grid.tolist(), "alpha": main["alpha"].tolist(),
           "diagnostics": diagnostics(df, data, Zs, stage, design)}
    if boot:
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(data, design, seed, Zs, stage, curves, alpha_grid)) as ex:
            draws = list(ex.map(_draw, range(boot)))
        phis = np.array([d(x[0]) for x in draws])
        sd = float(phis.std(ddof=1))
        row["boot"] = {"B": boot, "sd": sd, "ci90": [row["phi"] - 1.645 * sd, row["phi"] + 1.645 * sd],
                       "draws": phis.round(3).tolist(),
                       "U_sd": np.std([x[1] for x in draws], 0, ddof=1).tolist(),
                       "alpha_sd": np.std([x[2] for x in draws], 0, ddof=1).tolist()}
    row["seconds"] = time.time() - t0
    return row


def fmt(r):
    g = r["diagnostics"]
    b = r.get("boot")
    ci = f" 90% [{b['ci90'][0]:.2f}, {b['ci90'][1]:.2f}]" if b else ""
    return (f"{r['key']:15s} phi {r['phi']:6.2f} cell {r['cell']:5.2f} pooled {r['phi_pooled']:6.2f} "
            f"rot {min(r['rotations']):5.2f}-{max(r['rotations']):5.2f} own {r['own_cell']:5.2f} "
            f"gain {r['gain_cents']:5.2f}c b2 {r['beta2'][0]:.3f} | R2 {g['R2_latent']:.3f} "
            f"R2sq {g['R2_latent2']:.3f} spread {g['spread_ratio']:.2f} gap {g['assignment_gap']:.3f}{ci}")


# ----------------------------------------------------------------------------- figures

def figure(rows, path, reference=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.55 * n), squeeze=False)
    for i, r in enumerate(rows):
        g = r["diagnostics"]
        unit = {0.5: r"(2$\sqrt{\$}$)", 0.0: "(log points)", 1.0: "(\\$)"}[r["lam"]]
        name = r"$\zeta$" if r["stage"].startswith("locscale") else r"$\eta$"
        x = np.array(r["latent_grid"]); a = np.array(r["alpha"])
        ax = axes[i, 0]
        if "boot" in r:
            sd = np.array(r["boot"]["alpha_sd"])
            ax.fill_between(x, a - 1.645 * sd, a + 1.645 * sd, color="C0", alpha=0.25, lw=0)
        ax.plot(x, a, color="C0", lw=1.6)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel(r["label"].replace(": ", ":\n"), fontsize=7.5)
        ax.set_xlabel(f"{name} {unit}", fontsize=8)
        ax.set_title(r"$\hat\alpha$ (\$ per ride) on the trim window" if i == 0 else "", fontsize=9)
        ax = axes[i, 1]
        ax.axvspan(*r["window"], color="0.92", lw=0, zorder=0)
        ax.plot(g["implied_grid"], g["implied"], color="C1", lw=1.5, label="implied by independence")
        ax.plot(g["share_mid"], g["share"], "o", color="C0", ms=2.5, label="observed")
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{name} {unit}", fontsize=8)
        ax.text(0.03, 0.93, f"gap {g['assignment_gap']:.3f}, spread ratio {g['spread_ratio']:.2f}",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.set_title("share treated (grey: trim window)" if i == 0 else "", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="center right")
        ax = axes[i, 2]
        fg = np.array(r["fare_grid"]); U = 100 * np.array(r["U"])
        if reference is not None:
            ax.plot(fg, 100 * np.array(reference["U"]), color="0.6", ls="--", lw=1)
        if "boot" in r:
            sd = 100 * np.array(r["boot"]["U_sd"])
            ax.fill_between(fg, U - 1.645 * sd, U + 1.645 * sd, color="C0", alpha=0.25, lw=0)
        ax.plot(fg, U, color="C0", lw=1.6)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(r["phi"], color="C3", ls="--", lw=1)
        ax.axvline(15, color="0.5", ls=":", lw=1)
        ax.text(0.97, 0.06, f"\\${r['phi']:.2f} (cell \\${r['cell']:.2f}; own \\${r['own_cell']:.2f})",
                transform=ax.transAxes, ha="right", fontsize=7.5, color="C3")
        ax.set_xlabel("fare cutoff (\\$)", fontsize=8)
        ax.set_title("tips vs \\$15 (cents per ride); grey dashed: dollars, distance + log distance"
                     if i == 0 else "", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=[s[0] for s in SETTINGS])
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="point")
    a = ap.parse_args(argv)
    df = M.load_frame("full")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for key in a.keys:
        row = run_setting(df, key, a.seed, a.boot, a.workers)
        rows.append(row)
        print(fmt(row), f"[{time.time() - t0:.0f}s]", flush=True)
        with open(OUT / f"summary_{a.tag}.json", "w") as f:
            json.dump({"eps": EPS, "rows": rows}, f, indent=1)
    ref = next((r for r in rows if r["key"] == "usd_dlogd"), None)
    groups = {"sqrt": [r for r in rows if r["lam"] == 0.5], "log": [r for r in rows if r["lam"] == 0.0],
              "dollars": [r for r in rows if r["lam"] == 1.0]}
    for name, rs in groups.items():
        if rs:
            figure(rs, OUT / f"curves_{name}_{a.tag}.png", reference=ref)
    print("wrote", OUT, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
