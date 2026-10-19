"""GPA (academic probation; Lindo, Sanders and Oreopoulos 2010) with the four-role
differing-slopes estimator of differing_slopes.tex, Section 3.

Students below the probation cutoff (first-year GPA minus the campus cutoff,
``dist_from_cut`` < 0) are placed on probation.  The estimator treats above the
cutoff, so the score is the mirrored distance, S = -tr(dist), and probation is
S >= s0 with s0 = -tr(-0.005): GPAs lie on a 0.01 grid, so this is dist < 0.  A
policy s is reported as the GPA-distance cutoff c = tr^{-1}(-s) (probation if
dist <= c).  ``tr`` is the identity or, for the bounded-score check, a logit of the
distance between the floor and the ceiling of the GPA scale; a cutoff on tr(dist)
is a cutoff on dist, so the policy class is unchanged.

Model, as in the taxi application (score and outcome may use different covariates):

    score    S = gamma'(1, Z) + eta,             Z independent of eta,
    outcome  Y = b(eta) + D alpha(eta) + Z'beta1 + D X'beta2 + R,
    effect   W = alpha(eta) + X'beta2.

The estimator itself (fixed J, arm-wise spline outcome fit, spline g and rho_X on
the index fold, quantile trim endpoints, average of four rotated thresholds) is the
one in ``taxi_four_role.py``; this file supplies the GPA data, design and options.

Run (from code/):
    python -m experiments.scripts.gpa_four_role --screen
    python -m experiments.scripts.gpa_four_role --boot 20 --workers 6
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.scripts import taxi_four_role as M

DATA_PATH = M.ROOT / "datasets" / "gpa" / "Dep_Data" / "final_processed_data.csv"
OUT = M.ROOT / "runs" / "gpa_four_role"
POLICY_DIST = (-0.6, 0.6)      # GPA-distance cutoffs considered (the earlier GPA work's grid)
DEPLOYED = -0.005              # between the grid points -0.01 and 0: probation iff dist < 0
LSO = ["hsgrade_pct", "totcredits_year1", "loc_campus1", "loc_campus2", "male",
       "bpl_north_america", "age_at_entry", "english"]

# Outcomes, all with c = 0.  Graduation is missing only for late cohorts (not by
# treatment), so it is a full-population outcome on the cohorts observed long enough.
# nextGPA exists only for students with a later evaluation: a selected diagnostic.
OUTCOMES = {
    "grad6": "graduated within 6 years (cohorts 1996-99)",
    "grad5": "graduated within 5 years (cohorts 1996-2000)",
    "grad4": "graduated within 4 years (cohorts 1996-2001)",
    "return2": "registered in fall of year 2",
    "stay": "did not leave voluntarily",
    "credits2": "credits earned in year 2",
    "nextGPA": "next GPA minus cutoff (selected: continuing students only)",
}


def load_frame() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def outcome(df: pd.DataFrame, name: str) -> np.ndarray:
    col = {"grad6": "gradin6", "grad5": "gradin5", "grad4": "gradin4",
           "return2": "fallreg_year2", "credits2": "credits_earned2",
           "nextGPA": "nextGPA"}.get(name)
    if name == "stay":
        return 1.0 - df["left_school"].to_numpy(float)
    return df[col].to_numpy(float)


# ----------------------------------------------------------------------------- score scale

LO, HI = -1.65, 2.85           # just outside the GPA floor (0 - 1.6) and ceiling (4.3 - 1.5)


def tr(dist, scale: str):
    dist = np.asarray(dist, float)
    if scale == "raw":
        return dist
    if scale == "logit":
        return np.log((dist - LO) / (HI - dist))
    raise ValueError(scale)


def inv_tr(y, scale: str):
    y = np.asarray(y, float)
    if scale == "raw":
        return y
    return (LO + HI * np.exp(y)) / (1.0 + np.exp(y))


# ----------------------------------------------------------------------------- covariates

def score_covariates(df: pd.DataFrame, kind: str) -> np.ndarray:
    """``lso``: the eight covariates of the earlier GPA work, linear.  ``flex``: a
    cubic B-spline in the high-school grade percentile, indicators for year-1
    credits, age at entry and entry cohort, and the binary covariates."""
    if kind == "lso":
        return M._standardize(df[LSO].to_numpy(float))
    S = M._bspline_cols(df["hsgrade_pct"].to_numpy(float), np.linspace(0.1, 0.9, 9))
    return _standardize(np.column_stack([S, _indicators(df)]))


def _standardize(A):
    """Standardize, dropping columns that are constant in this sample (e.g. the cohort
    indicators of cohorts without graduation records)."""
    A = np.asarray(A, float)
    return M._standardize(A[:, A.std(0) > 0])


def _indicators(df: pd.DataFrame) -> np.ndarray:
    cr = df["totcredits_year1"].to_numpy(float)
    credits = np.column_stack([cr == v for v in (3.0, 3.5, 4.5, 5.0)] + [cr >= 5.5]).astype(float)
    age = df["age_at_entry"].to_numpy(int)
    ages = np.column_stack([age == a for a in (17, 18, 20, 21)]).astype(float)
    year = df["firstyear"].to_numpy(int)
    years = np.column_stack([year == y for y in range(1997, 2004)]).astype(float)
    binary = df[["loc_campus1", "loc_campus2", "male", "bpl_north_america", "english"]].to_numpy(float)
    return np.column_stack([credits, ages, years, binary])


def effect_covariates(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind in ("none", "hsgrade"):        # "none": placeholder, not used
        return M._standardize(df["hsgrade_pct"].to_numpy(float))
    if kind == "lso":
        return M._standardize(df[LSO].to_numpy(float))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- data and design

@dataclass
class GPAData:
    Q: np.ndarray          # S = -tr(dist)
    Z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    s0: float
    Zb: np.ndarray | None = None

    @property
    def D(self):
        return (self.Q >= self.s0).astype(float)

    @property
    def baseline(self):
        return self.Z if self.Zb is None else self.Zb

    def subset(self, idx):
        return GPAData(self.Q[idx], _standardize(self.Z[idx]), _standardize(self.X[idx]),
                       self.Y[idx], self.s0, None if self.Zb is None else _standardize(self.Zb[idx]))


def make_data(df, y: str, score: str = "flex", effect: str = "hsgrade", scale: str = "raw"):
    Y = outcome(df, y)
    keep = np.isfinite(Y)
    df = df.loc[keep]
    s0 = float(-tr(DEPLOYED, scale))
    data = GPAData(-tr(df["dist_from_cut"].to_numpy(float), scale), score_covariates(df, score),
                   effect_covariates(df, effect), Y[keep], s0)
    assert np.array_equal(data.D, (df["dist_from_cut"].to_numpy(float) < 0).astype(float))
    return data, df


def make_design(data: GPAData, eps: float, scale: str, effect: bool = True, tail: float = 0.0025,
                window_tail: float = 0.005, margin_sd: float = 0.7) -> M.Design:
    """J, density interval and policy set from (S, Z) only (D is a function of S).

    The trim endpoints are interior here, so J covers the population trim endpoints
    with a margin of ``margin_sd`` sd(eta).  Both arms must be observed throughout J:
    the score index has a short range, so no student far above the GPA predicted by
    the covariates is on probation, and few far below it are not.  J therefore stops
    at the ``tail`` quantiles of eta among the treated (below) and the untreated
    (above), and the trim window is capped at the ``window_tail`` quantiles, inside J."""
    A = M._xt(data.Z)
    T = A @ np.linalg.lstsq(A, data.Q, rcond=None)[0]
    eta = data.Q - T
    D = data.D.astype(bool)
    l0 = data.s0 - float(np.quantile(T, 1 - eps))
    u0 = data.s0 - float(np.quantile(T, eps))
    sd = float(eta.std())
    J = (max(l0 - margin_sd * sd, float(np.quantile(eta[D], tail))),
         min(u0 + margin_sd * sd, float(np.quantile(eta[~D], 1 - tail))))
    policy = tuple(sorted(float(v) for v in -tr(np.array(POLICY_DIST), scale)))
    pad = 0.5 * sd
    tlo, thi = np.quantile(T, [0.0005, 0.9995])
    T_support = (float(min(tlo, policy[0] - J[1]) - pad), float(max(thi, policy[1] - J[0]) + pad))
    window = (float(np.quantile(eta[D], window_tail)), float(np.quantile(eta[~D], 1 - window_tail)))
    return M.Design(J=J, eps=eps, phi0=data.s0, T_support=T_support, policy=policy,
                    effect=effect, window=window)


def _eps_endpoints(data, eps):
    A = M._xt(data.Z)
    T = A @ np.linalg.lstsq(A, data.Q, rcond=None)[0]
    return [data.s0 - float(np.quantile(T, 1 - eps)), data.s0 - float(np.quantile(T, eps))]


def to_dist(s, scale):
    return float(inv_tr(-np.asarray(s, float), scale))


def dist_cell(c, dists):
    """Largest observed GPA distance placed on probation by the rule dist <= c."""
    below = np.unique(dists)[np.unique(dists) <= c + 1e-9]
    return float(below.max()) if below.size else float("-inf")


# ----------------------------------------------------------------------------- diagnostics

def independence(df, data) -> dict:
    A = M._xt(data.Z)
    eta = data.Q - A @ np.linalg.lstsq(A, data.Q, rcond=None)[0]
    hs = df["hsgrade_pct"].to_numpy(float)
    rich = np.column_stack([np.ones(len(hs)), M._bspline_cols(hs, np.linspace(0.05, 0.95, 15)),
                            _indicators(df),
                            M._bspline_cols(hs, np.linspace(0.1, 0.9, 5)) * df["male"].to_numpy(float)[:, None]])

    def r2(v):
        c = np.linalg.lstsq(rich, v, rcond=None)[0]
        return float(1 - np.var(v - rich @ c) / np.var(v))

    dec = pd.qcut(hs, 10, labels=False, duplicates="drop")
    sd = pd.Series(eta).groupby(dec).std()
    mean = pd.Series(eta).groupby(dec).mean()
    return {"R2_score": float(1 - eta.var() / data.Q.var()), "R2_eta": r2(eta), "R2_eta2": r2(eta ** 2),
            "spread_ratio": float(sd.max() / sd.min()),
            "decile_mean_range": [float(mean.min()), float(mean.max())]}


# ----------------------------------------------------------------------------- one specification

_B: dict = {}


def _init(data, design, seed, curves, alpha_grid):
    _B.update(data=data, design=design, seed=seed, curves=curves, alpha_grid=alpha_grid)


def _draw(b):
    data, design = _B["data"], _B["design"]
    idx = np.random.default_rng(_B["seed"] + 1_000 + b).integers(0, len(data.Q), len(data.Q))
    r = M.four_role(data.subset(idx), design, seed=_B["seed"] + 50_000 + b, grid_points=121,
                    curves=_B["curves"], alpha_grid=_B["alpha_grid"])
    return r["phi_avg"], r["at_boundary"], r["U"], r["alpha"], r["tau"]


def run_one(df, y, score, effect, scale="raw", eps=0.1, seed=1, boot=0, workers=6):
    data, sub = make_data(df, y, score, effect, scale)
    design = make_design(data, eps, scale, effect=(effect != "none"))
    dist_grid = np.linspace(*POLICY_DIST, 49)
    curves = -tr(dist_grid, scale)
    alpha_grid = np.linspace(design.J[0], design.J[1], 49)
    main = M.four_role(data, design, seed, curves=curves, grid_points=121, alpha_grid=alpha_grid)
    dists = sub["dist_from_cut"].to_numpy(float)
    c = to_dist(main["phi_avg"], scale)
    row = {"outcome": y, "score": score, "effect": effect, "scale": scale, "eps": eps,
           "n": int(len(data.Q)), "treated_share": float(data.D.mean()),
           "c": c, "cell": dist_cell(c, dists), "c_pooled": to_dist(main["phi_pooled"], scale),
           "c_rotations": [to_dist(p, scale) for p in main["phi_rotations"]],
           "own_cell": dist_cell(to_dist(main["own_cutoff"], scale), dists),
           "at_boundary": main["at_boundary"], "local_maxima": main["local_maxima"],
           "gain": main["gain_vs_deployed"], "beta2": main["beta2"],
           "window": main["window"], "window_share": main["window_share"],
           "J": list(design.J), "caps": list(design.window),
           "eps_endpoints": _eps_endpoints(data, eps),
           "dist_grid": dist_grid.tolist(), "U": main["U"].tolist(), "tau": main["tau"].tolist(),
           "eta_grid": alpha_grid.tolist(), "alpha": main["alpha"].tolist(),
           "effect_block": M.effect_diagnostics(data, design, seed)}
    if boot:
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(data, design, seed, curves, alpha_grid)) as ex:
            draws = list(ex.map(_draw, range(boot)))
        cs = np.array([to_dist(x[0], scale) for x in draws])
        sd = float(cs.std(ddof=1))
        row["boot"] = {"B": boot, "sd": sd, "ci90": [c - 1.645 * sd, c + 1.645 * sd],
                       "draws": cs.round(4).tolist(),
                       "share_boundary": float(np.mean([x[1] for x in draws])),
                       "U_sd": np.std([x[2] for x in draws], 0, ddof=1).tolist(),
                       "alpha_sd": np.std([x[3] for x in draws], 0, ddof=1).tolist(),
                       "tau_sd": np.std([x[4] for x in draws], 0, ddof=1).tolist()}
    return row, data, sub


def fmt(r):
    b = r.get("boot")
    ci = f" 90% [{b['ci90'][0]:+.2f}, {b['ci90'][1]:+.2f}] bnd {b['share_boundary']:.2f}" if b else ""
    eb = r["effect_block"]
    left = eb.get("min_share_left_after_partialling", float("nan"))
    return (f"{r['outcome']:8s} {r['score']:4s} {r['effect']:7s} {r['scale']:5s} eps {r['eps']:.2f} "
            f"c {r['c']:+.3f} cell {r['cell']:+.2f} pooled {r['c_pooled']:+.3f} "
            f"rot {min(r['c_rotations']):+.2f}..{max(r['c_rotations']):+.2f} own {r['own_cell']:+.2f} "
            f"bound {int(r['at_boundary'])} lm {r['local_maxima']} gain {r['gain']:+.4f} "
            f"win {r['window_share']:.2f} [{r['window'][0]:+.2f},{r['window'][1]:+.2f}] eps-end "
            f"[{r['eps_endpoints'][0]:+.2f},{r['eps_endpoints'][1]:+.2f}] left {left:.3f} "
            f"b2 {np.round(r['beta2'][:2], 3).tolist()}{ci}")


def figure(rows, path):
    """Marginal effect tau and value U against the GPA-distance cutoff, common slope
    against the effect linear in the high-school grade, with 90% pointwise bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outs = list(dict.fromkeys(r["outcome"] for r in rows))
    fig, axes = plt.subplots(2, len(outs), figsize=(3.3 * len(outs), 5.6), sharex=True)
    style = {"none": ("C0", "common slope"), "hsgrade": ("C3", "effect linear in HS grade")}
    for j, y in enumerate(outs):
        for r in (r for r in rows if r["outcome"] == y):
            col, lab = style[r["effect"]]
            g = np.array(r["dist_grid"])
            for i, (key, sdkey) in enumerate((("tau", "tau_sd"), ("U", "U_sd"))):
                v = np.array(r[key])
                ax = axes[i, j]
                if "boot" in r:
                    sd = np.array(r["boot"][sdkey])
                    ax.fill_between(g, v - 1.645 * sd, v + 1.645 * sd, color=col, alpha=0.2, lw=0)
                ax.plot(g, v, color=col, lw=1.6, label=lab)
                ax.axhline(0, color="k", lw=0.7)
                ax.axvline(0, color="0.5", ls=":", lw=1)
        axes[0, j].set_title(OUTCOMES[y].split(" (")[0], fontsize=9)
        axes[1, j].set_xlabel("probation if GPA - cutoff <= c", fontsize=8)
    axes[0, 0].set_ylabel(r"marginal effect $\hat\tau_\epsilon(c)$", fontsize=9)
    axes[1, 0].set_ylabel(r"$\hat U_\epsilon(c)-\hat U_\epsilon(0)$", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true", help="all outcomes x scores x effects, no bootstrap")
    ap.add_argument("--outcomes", nargs="*", default=list(OUTCOMES))
    ap.add_argument("--scores", nargs="*", default=["lso", "flex"])
    ap.add_argument("--effects", nargs="*", default=["none", "hsgrade", "lso"])
    ap.add_argument("--scale", default="raw", choices=["raw", "logit"])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="screen")
    a = ap.parse_args(argv)
    df = load_frame()
    t0 = time.time()
    rows, indep = [], {}
    for y in a.outcomes:
        for score in a.scores:
            for effect in a.effects:
                row, data, sub = run_one(df, y, score, effect, a.scale, a.eps, a.seed, a.boot, a.workers)
                key = f"{y}/{score}"
                if key not in indep:
                    indep[key] = independence(sub, data)
                row["independence"] = indep[key]
                rows.append(row)
                print(fmt(row), f"[{time.time() - t0:.0f}s]", flush=True)
    for k, v in indep.items():
        print(k, {kk: (np.round(vv, 3).tolist() if not isinstance(vv, float) else round(vv, 3))
                  for kk, vv in v.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"summary_{a.tag}.json", "w") as f:
        json.dump({"rows": rows, "policy_dist": POLICY_DIST}, f, indent=1)
    print("wrote", OUT / f"summary_{a.tag}.json")
    if a.boot:
        figure(rows, OUT / f"tau_U_{a.tag}.png")


if __name__ == "__main__":
    main()
