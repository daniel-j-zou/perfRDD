"""Taxi application of the four-role differing-slopes estimator (differing_slopes.tex, Section 3).

Data: NYC TLC January 2009 yellow-taxi trips, VTS credit-card rides, with the
Haggag--Paci restrictions (daytime, no tolls/taxes/surcharges, standard meter
grid).  The main sample keeps every fare; ``--sample restricted`` adds their
$5--$25 fare range, which selects on the running variable and is kept only as a
comparison.  Rides with zero recorded distance and tip entry errors (tip > fare
and tip >= $10) are dropped.

Model (score and outcome use different covariate vectors, as in Mukherjee et al.):

    score    Q = tr(fare) = gamma'(1, Z) + eta,           Z independent of eta,
    outcome  Y = b(eta) + D alpha(eta) + Z'beta1 + D X'beta2 + R,
    effect   W = alpha(eta) + X'beta2,  X = trip distance (standardized),

with D = 1{fare >= $15} and tr a Box-Cox transform of the fare (default
lambda = 1/2, the square root).  A cutoff on tr(fare) is a cutoff on the fare, so
the policy class is unchanged; the transform only chooses the scale on which Z
independent of eta is maintained.  The power is chosen from (fare, Z) alone: the
metered time charge grows with distance, so in dollars the spread of eta rises
with distance and in logs it falls; lambda in [0.4, 0.5] makes it flat.  Z is a cubic B-spline in log distance plus hour, day-of-week,
weekday-hour x weekend and group-ride indicators.  The utility is the
hard-trimmed differing-slopes criterion with spline g and rho_X on the index
fold, quantile trim endpoints, and the average of the four rotated thresholds.

Design rules fixed before any outcome is used (they depend on (Q, Z) only):
* J = [max(l0 - 0.7 sd(eta), q_0.005(eta)), q_0.995(eta)] from a full-sample OLS pilot
  (robustness: the excluded tail 0.25%-2% and the margin 0.35-1.4 sd);
  when the upper trim endpoint lies beyond J (it does here: few rides are treated),
  the upper trim is inactive and the window is [l, J_hi];
* the density interval covers the pilot index range and the argument set;
* Phi = [$6, $24]; outcome spline knots n_eff^(11/60) as in the simulations.

Inference: full re-estimation bootstrap over trips (iid; the public file has no
driver identifiers), percentile intervals.

Run (from code/):
    python -m experiments.scripts.taxi_four_role --boot 199 --workers 5
    python -m experiments.scripts.taxi_four_role --sample restricted --boot 199
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
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from experiments.datasets.taxi.adapter import DATA_PATH, prepare_haggag_paci_frame
from experiments.methods.perfrdd import _basis_params
from experiments.methods.weighted_tails import fit_weighted_tails

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "runs" / "taxi_four_role"
FARE0 = 15.0
POLICY_DOLLARS = (6.0, 24.0)
ROLES = ("gamma", "outcome", "index", "utility")


# ----------------------------------------------------------------------------- data

def load_frame(sample: str) -> pd.DataFrame:
    """Haggag--Paci restrictions; ``full`` drops only their $5--$25 fare range."""
    raw = pd.read_parquet(DATA_PATH)
    if sample == "restricted":
        df = prepare_haggag_paci_frame(raw, vendor="VTS")
    elif sample == "full":
        df = raw[raw["Payment_Type"].astype(str).str.upper() == "CREDIT"].copy()
        pickup = pd.to_datetime(df["Trip_Pickup_DateTime"])
        hour = pickup.dt.hour + pickup.dt.minute / 60.0 + pickup.dt.second / 3600.0
        day = pickup.dt.dayofweek.astype(float)
        daytime = (((day < 5) & (hour >= 6) & (hour < 16))
                   | ((day >= 5) & (hour >= 6) & (hour < 20)))
        units = np.rint((df["Fare_Amt"] - 2.5) / 0.4)
        keep = ((pickup < pd.Timestamp("2009-11-01")) & daytime
                & df["Tolls_Amt"].fillna(0.0).eq(0.0) & df["surcharge"].fillna(0.0).eq(0.0)
                & df["mta_tax"].fillna(0.0).eq(0.0) & (df["Fare_Amt"] >= 2.5)
                & np.isclose(df["Fare_Amt"], 2.5 + 0.4 * units, atol=1e-6))
        df = df.loc[keep].copy()
        df["hour_of_day"] = hour.loc[keep]
        df["day_of_week"] = day.loc[keep]
    else:
        raise ValueError(sample)
    cols = ["Trip_Distance", "Passenger_Count", "hour_of_day", "day_of_week",
            "Fare_Amt", "Tip_Amt"]
    df = df[df[cols].notna().all(axis=1)]
    fare, tip = df["Fare_Amt"].to_numpy(float), df["Tip_Amt"].to_numpy(float)
    entry_error = (tip / fare > 1.0) & (tip >= 10.0)
    df = df.loc[~entry_error & (df["Trip_Distance"].to_numpy(float) > 0)]
    return df.reset_index(drop=True)


def boxcox(x, lam: float):
    x = np.asarray(x, float)
    return np.log(x) if lam == 0 else (x ** lam - 1.0) / lam


def inv_boxcox(y, lam: float):
    y = np.asarray(y, float)
    return np.exp(y) if lam == 0 else (lam * y + 1.0) ** (1.0 / lam)


def _bspline_cols(x, knot_quantiles, degree=3):
    lo, hi = float(x.min()), float(x.max())
    t = np.concatenate([np.repeat(lo, degree + 1), np.quantile(x, knot_quantiles),
                        np.repeat(hi, degree + 1)])
    B = BSpline.design_matrix(np.clip(x, lo, hi), t, degree).toarray()
    return B[:, 1:]                    # the basis sums to one; Z carries no constant


def _standardize(M):
    M = np.asarray(M, float)
    M = M[:, None] if M.ndim == 1 else M
    return (M - M.mean(0)) / M.std(0)


def score_covariates(df: pd.DataFrame, kind: str = "spline") -> np.ndarray:
    """Distance terms (``kind``) plus hour, day-of-week, weekday-hour x weekend and
    group-ride indicators.  ``linear``: distance; ``distlog``: distance and log
    distance (the meter's $2/mile plus a time charge growing with log distance);
    ``spline``: cubic B-spline in log distance with knots at the quintiles."""
    d = df["Trip_Distance"].to_numpy(float)
    hr = np.floor(df["hour_of_day"].to_numpy(float)).astype(int)
    dow = df["day_of_week"].to_numpy(float)
    if kind == "linear":
        dist = d[:, None]
    elif kind == "distlog":
        dist = np.column_stack((d, np.log(d)))
    elif kind == "spline":
        dist = _bspline_cols(np.log(d), [0.2, 0.4, 0.6, 0.8])
    else:
        raise ValueError(kind)
    hours = np.column_stack([hr == h for h in range(7, 20)]).astype(float)   # base 6am
    days = np.column_stack([dow == k for k in range(1, 7)]).astype(float)    # base Monday
    weekend_hours = hours[:, :9] * (dow >= 5)[:, None]   # 16-19h occur only on weekends
    group = (df["Passenger_Count"].to_numpy(float) >= 2).astype(float)
    return _standardize(np.column_stack([dist, hours, days, weekend_hours, group]))


def effect_covariates(df: pd.DataFrame, kind: str = "distance") -> np.ndarray:
    d = df["Trip_Distance"].to_numpy(float)
    if kind == "distance":
        return _standardize(d)
    if kind == "none":                      # common slope: placeholder, not used
        return _standardize(d)
    if kind == "distance2":                 # robustness: curvature in distance
        return _standardize(np.column_stack((d, d ** 2)))
    if kind == "controls":                  # comparison: all four controls, entering linearly
        return _standardize(np.column_stack((
            d, df["Passenger_Count"].to_numpy(float), df["hour_of_day"].to_numpy(float),
            df["day_of_week"].to_numpy(float))))
    if kind == "distance+Z":                # outside the model: effect nonlinear in distance
        return _standardize(np.column_stack((d, score_covariates(df))))
    raise ValueError(kind)


def linear_baseline_covariates(df: pd.DataFrame) -> np.ndarray:
    """Robustness: an untreated-tip baseline linear in distance, hour, weekend, group."""
    return _standardize(np.column_stack([
        df["Trip_Distance"].to_numpy(float), df["hour_of_day"].to_numpy(float),
        df["day_of_week"].to_numpy(float) >= 5,
        df["Passenger_Count"].to_numpy(float) >= 2]))


@dataclass
class Data:
    Q: np.ndarray        # transformed fare
    Z: np.ndarray        # score covariates (centered); also the baseline unless Zb is set
    X: np.ndarray        # effect covariates (centered)
    Y: np.ndarray        # tip, dollars
    fare: np.ndarray
    lam: float
    Zb: np.ndarray | None = None   # baseline covariates, if different from Z

    @property
    def D(self):
        return (self.fare >= FARE0).astype(float)

    @property
    def baseline(self):
        return self.Z if self.Zb is None else self.Zb

    def subset(self, idx):
        Zb = None if self.Zb is None else _standardize(self.Zb[idx])
        return Data(self.Q[idx], _standardize(self.Z[idx]), _standardize(self.X[idx]),
                    self.Y[idx], self.fare[idx], self.lam, Zb)


def make_data(df: pd.DataFrame, lam: float, effect: str = "distance",
              baseline: str = "flexible", score: str = "spline") -> Data:
    fare = df["Fare_Amt"].to_numpy(float)
    Zb = None if baseline == "flexible" else linear_baseline_covariates(df)
    return Data(boxcox(fare, lam), score_covariates(df, score), effect_covariates(df, effect),
                df["Tip_Amt"].to_numpy(float), fare, lam, Zb)


# ----------------------------------------------------------------------------- design

@dataclass
class Design:
    J: tuple
    eps: float
    phi0: float
    T_support: tuple
    policy: tuple
    cost: float = 0.0
    effect: bool = True
    window: tuple | None = None    # fixed caps on the trim window, inside J
    # With ``window_tail`` set and ``window_estimated`` true, the caps are instead
    # re-estimated in every rotation as the window_tail and 1 - window_tail quantiles
    # of the first-stage residuals on the index fold (author decision, 2026-10-20).
    window_tail: float | None = None
    window_estimated: bool = False


def _xt(Z):
    return np.column_stack((np.ones(len(Z)), Z))


def make_design(data: Data, eps: float, margin_sd: float = 0.7, effect: bool = True,
                tail: float = 0.005, window_tail: float | None = None,
                window_estimated: bool = False) -> Design:
    """Fixed nuisance region, density interval and policy set, from (Q, Z) only.

    With ``window_tail`` the trim window is also capped at the central
    1 - 2*window_tail of the residuals, strictly inside J (which excludes only
    ``tail`` in each direction), as the theorem's margin condition requires."""
    g = np.linalg.lstsq(_xt(data.Z), data.Q, rcond=None)[0]
    T = _xt(data.Z) @ g
    eta = data.Q - T
    phi0 = float(boxcox(FARE0, data.lam))
    l0 = phi0 - float(np.quantile(T, 1 - eps))
    J = (max(l0 - margin_sd * float(eta.std()), float(np.quantile(eta, tail))),
         float(np.quantile(eta, 1 - tail)))
    policy = tuple(float(v) for v in boxcox(np.array(POLICY_DOLLARS), data.lam))
    pad = 0.5 * float(eta.std())
    tlo, thi = np.quantile(T, [0.0005, 0.9995])
    T_support = (float(min(tlo, policy[0] - J[1]) - pad),
                 float(max(thi, policy[1] - max(l0, J[0])) + pad))
    window = None
    if window_tail is not None:
        window = (float(np.quantile(eta, window_tail)), float(np.quantile(eta, 1 - window_tail)))
    return Design(J=J, eps=eps, phi0=phi0, T_support=T_support, policy=policy,
                  effect=effect, window=window, window_tail=window_tail,
                  window_estimated=bool(window_estimated and window_tail is not None))


# ----------------------------------------------------------------------------- estimator

@dataclass
class OutcomeFit:
    info: dict
    w0: np.ndarray
    w1: np.ndarray
    beta2: np.ndarray

    def alpha(self, e):
        e = np.clip(e, self.info["lo"], self.info["hi"])
        return BSpline(self.info["t"], self.w1 - self.w0, self.info["degree"],
                       extrapolate=False)(e)


def outcome_fit(data: Data, eta, idx, design: Design) -> OutcomeFit:
    """Stacked spline regression on J: arm-wise splines in eta, Z, and D*X."""
    e = eta[idx]
    keep = idx[(e >= design.J[0]) & (e <= design.J[1])]
    e, D = eta[keep], data.D[keep]
    n1 = int(D.sum())
    info = _basis_params(max(3, int(round(min(n1, len(keep) - n1) ** (11 / 60)))), design.J)
    B = BSpline.design_matrix(np.clip(e, info["lo"], info["hi"]), info["t"],
                              info["degree"]).toarray()
    cols = [(1 - D)[:, None] * B, D[:, None] * B, data.baseline[keep]]
    if design.effect:
        cols.append(D[:, None] * data.X[keep])
    A = np.column_stack(cols)
    coef = np.linalg.solve(A.T @ A, A.T @ data.Y[keep])
    nb = B.shape[1]
    beta2 = (coef[2 * nb + data.baseline.shape[1]:] if design.effect
             else np.zeros(data.X.shape[1]))
    return OutcomeFit(info, coef[:nb], coef[nb:2 * nb], beta2)


@dataclass
class Rotation:
    value: Callable          # U-hat(phi), per ride in the evaluation fold
    marginal: Callable       # tau-hat(phi) = -U-hat'(phi) / p-hat(phi)
    own: Callable            # own-score objective (does not use Z independent of eta)
    alpha: Callable          # fitted alpha(eta), the effect at average covariates
    beta2: np.ndarray
    window: tuple
    window_share: float


def rotation(data: Data, design: Design, folds: dict) -> Rotation:
    g = np.linalg.lstsq(_xt(data.Z[folds["gamma"]]), data.Q[folds["gamma"]], rcond=None)[0]
    eta = data.Q - _xt(data.Z) @ g
    fit = outcome_fit(data, eta, folds["outcome"], design)
    T_idx = _xt(data.Z[folds["index"]]) @ g
    tails = fit_weighted_tails(T_idx, data.X[folds["index"]], design.T_support)
    # Trim endpoints from the index fold; an endpoint beyond the caps (no residuals
    # there to fit) is inactive, and the window stops at the cap.
    if design.window_estimated:
        e_idx = eta[folds["index"]]
        caps = (float(np.quantile(e_idx, design.window_tail)),
                float(np.quantile(e_idx, 1 - design.window_tail)))
    else:
        caps = design.window if design.window is not None else design.J
    l_hat = max(design.phi0 - float(np.quantile(T_idx, 1 - design.eps)), caps[0])
    u_hat = min(design.phi0 - float(np.quantile(T_idx, design.eps)), caps[1])
    U = folds["utility"]
    e_all = eta[U]
    inside = (e_all >= l_hat) & (e_all <= u_hat)
    e = e_all[inside]
    amc = fit.alpha(e) - design.cost
    b2 = fit.beta2
    n_u = len(U)

    def value(phi):
        t = phi - e
        return float(np.sum(amc * tails.survival(t) + tails.weighted_tail(t, b2)) / n_u)

    def marginal(phi):
        t = phi - e
        num = np.sum(amc * tails.density(t) + tails.weighted_density(t, b2))
        den = np.sum(tails.density(t))
        return float(num / den)

    effect_all = fit.alpha(e_all) + data.X[U] @ b2 - design.cost

    def own(phi):
        return float(np.mean(inside * effect_all * (data.Q[U] >= phi)))

    return Rotation(value, marginal, own, fit.alpha, b2, (l_hat, u_hat), float(inside.mean()))


def _argmax(f, lo, hi, m=181):
    grid = np.linspace(lo, hi, m)
    vals = np.array([f(x) for x in grid])
    k = int(np.argmax(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, m - 1)]
    res = minimize_scalar(lambda x: -f(x), bounds=(a, b), method="bounded",
                          options={"xatol": 1e-8})
    n_local = int(np.sum((vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:])))
    return float(res.x), bool(k in (0, m - 1)), n_local


def four_role(data: Data, design: Design, seed: int, curves: np.ndarray | None = None,
              grid_points: int = 181, alpha_grid: np.ndarray | None = None) -> dict:
    """Fixed-fold four-role estimator, rotated over the four roles."""
    blocks = np.array_split(np.random.default_rng(seed).permutation(len(data.Q)), 4)
    rots, phis = [], []
    for k in range(4):
        folds = {r: blocks[(j + k) % 4] for j, r in enumerate(ROLES)}
        rot = rotation(data, design, folds)
        rots.append(rot)
        phis.append(_argmax(rot.value, *design.policy, m=grid_points)[0])
    pooled = lambda x: float(np.mean([r.value(x) for r in rots]))
    phi_pool, at_bound, n_local = _argmax(pooled, *design.policy, m=grid_points)
    phi_avg = float(np.mean(phis))
    out = {
        "phi_avg": phi_avg, "phi_pooled": phi_pool, "phi_rotations": phis,
        "fixed_assignment": phis[0], "at_boundary": at_bound, "local_maxima": n_local,
        "gain_vs_deployed": pooled(phi_pool) - pooled(design.phi0),
        "beta2": np.mean([r.beta2 for r in rots], axis=0).tolist(),
        "window": np.mean([r.window for r in rots], axis=0).tolist(),
        "window_share": float(np.mean([r.window_share for r in rots])),
    }
    if alpha_grid is not None:
        out["alpha"] = np.mean([r.alpha(alpha_grid) for r in rots], axis=0)
    if curves is not None:
        out["U"] = np.array([pooled(x) for x in curves]) - pooled(design.phi0)
        out["tau"] = np.array([np.mean([r.marginal(x) for r in rots]) for x in curves])
        cuts = np.unique(data.Q)
        cuts = cuts[(cuts >= design.policy[0]) & (cuts <= design.policy[1])]
        own = np.array([np.mean([r.own(c) for r in rots]) for c in cuts])
        out["own_cutoff"] = float(cuts[int(np.argmax(own))])
    return out


def dollars(phi, lam):
    return float(inv_boxcox(phi, lam))


def fare_cell(phi_dollars, fares):
    """Smallest observed fare treated by the rule fare >= phi."""
    above = np.unique(fares)[np.unique(fares) >= phi_dollars - 1e-9]
    return float(above.min()) if above.size else float("inf")


# ----------------------------------------------------------------------------- effect block

def effect_diagnostics(data: Data, design: Design, seed: int) -> dict:
    """How well the D*X block is identified, and whether it predicts held-out tips.

    Identification: each column of D*X is regressed on the other outcome regressors
    (arm-wise splines in eta and the baseline covariates) within J; the share of its
    variance left over and the smallest eigenvalue of the correlation matrix of the
    residuals measure how much separate variation the effect block has.
    Prediction: the outcome model is fitted on a random half and evaluated on the
    other half (rides in J), overall and among treated rides.
    """
    g = np.linalg.lstsq(_xt(data.Z), data.Q, rcond=None)[0]
    eta = data.Q - _xt(data.Z) @ g
    inJ = np.flatnonzero((eta >= design.J[0]) & (eta <= design.J[1]))
    D = data.D[inJ]
    info = _basis_params(max(3, int(round(min(D.sum(), len(D) - D.sum()) ** (11 / 60)))),
                         design.J)
    B = BSpline.design_matrix(np.clip(eta[inJ], info["lo"], info["hi"]), info["t"],
                              info["degree"]).toarray()
    nuis = np.column_stack(((1 - D)[:, None] * B, D[:, None] * B, data.baseline[inJ]))
    if not design.effect:
        rng = np.random.default_rng(seed + 99)
        half = rng.random(len(inJ)) < 0.5
        fit = np.linalg.lstsq(nuis[half], data.Y[inJ][half], rcond=None)[0]
        err = data.Y[inJ][~half] - nuis[~half] @ fit
        return {"n_effect_columns": 0, "holdout_mse": float(np.mean(err ** 2)),
                "holdout_mse_treated": float(np.mean(err[D[~half] == 1] ** 2)),
                "holdout_n": int((~half).sum())}
    eff = D[:, None] * data.X[inJ]
    coef = np.linalg.lstsq(nuis, eff, rcond=None)[0]
    resid = eff - nuis @ coef
    left = resid.var(0) / eff.var(0)
    corr = np.corrcoef(resid, rowvar=False) if resid.shape[1] > 1 else np.ones((1, 1))
    eig = np.linalg.eigvalsh(np.atleast_2d(corr))
    # held-out prediction
    rng = np.random.default_rng(seed + 99)
    half = rng.random(len(inJ)) < 0.5
    A = np.column_stack((nuis, eff))
    fit = np.linalg.lstsq(A[half], data.Y[inJ][half], rcond=None)[0]
    err = data.Y[inJ][~half] - A[~half] @ fit
    treated = D[~half] == 1
    return {"n_effect_columns": int(data.X.shape[1]),
            "min_share_left_after_partialling": float(left.min()),
            "median_share_left_after_partialling": float(np.median(left)),
            "min_eigenvalue_residual_correlation": float(eig.min()),
            "condition_number_residual_correlation": float(eig.max() / max(eig.min(), 1e-300)),
            "holdout_mse": float(np.mean(err ** 2)),
            "holdout_mse_treated": float(np.mean(err[treated] ** 2)),
            "holdout_n": int((~half).sum())}


# ----------------------------------------------------------------------------- bootstrap

_BOOT: dict = {}


def _boot_init(data, design, seed, grid):
    _BOOT.update(data=data, design=design, seed=seed, grid=grid)


def _boot_draw(b):
    data, design = _BOOT["data"], _BOOT["design"]
    idx = np.random.default_rng(_BOOT["seed"] + 1_000 + b).integers(0, len(data.Q), len(data.Q))
    r = four_role(data.subset(idx), design, seed=_BOOT["seed"] + 50_000 + b,
                  curves=_BOOT["grid"])
    return r["phi_avg"], r["phi_pooled"], r["at_boundary"], r["U"], r["tau"]


# ----------------------------------------------------------------------------- main

def run(sample, lam, eps, boot, workers, seed, out_dir, extras=True, effect="distance",
        baseline="flexible"):
    t0 = time.time()
    df = load_frame(sample)
    data = make_data(df, lam, effect, baseline)
    design = make_design(data, eps)
    grid_d = np.linspace(*POLICY_DOLLARS, 73)
    grid = boxcox(grid_d, lam)
    main = four_role(data, design, seed, curves=grid)
    fares = data.fare
    res = {
        "sample": sample, "n": int(len(data.Q)), "treated_share": float(data.D.mean()),
        "lambda": lam, "eps": eps, "effect": effect, "baseline": baseline, "design": {
            "J": design.J, "T_support": design.T_support, "policy_dollars": POLICY_DOLLARS},
        "estimate": {
            "phi_avg_dollars": dollars(main["phi_avg"], lam),
            "cell_avg": fare_cell(dollars(main["phi_avg"], lam), fares),
            "phi_pooled_dollars": dollars(main["phi_pooled"], lam),
            "rotations_dollars": [dollars(p, lam) for p in main["phi_rotations"]],
            "own_score_cell": fare_cell(dollars(main["own_cutoff"], lam), fares),
            "gain_vs_deployed_cents_per_ride": 100 * main["gain_vs_deployed"],
            "beta2_per_sd": main["beta2"], "window": main["window"],
            "window_share": main["window_share"], "local_maxima": main["local_maxima"],
            "at_boundary": main["at_boundary"]},
        "curves": {"fare": grid_d.tolist(), "U_minus_U15": main["U"].tolist(),
                   "tau": main["tau"].tolist()},
    }
    res["effect_diagnostics"] = effect_diagnostics(data, design, seed)
    print(json.dumps({**res["estimate"], **res["effect_diagnostics"]}, indent=1),
          f"[{time.time() - t0:.0f}s]", flush=True)
    if extras:
        alpha_only = four_role(data, make_design(data, eps, effect=False), seed)
        res["alpha_only"] = {"phi_dollars": dollars(alpha_only["phi_avg"], lam),
                             "at_boundary": alpha_only["at_boundary"]}
        res["eps_curve"] = {}
        for e in (0.03, 0.05, 0.075, 0.10, 0.125, 0.15):
            r = four_role(data, make_design(data, e), seed)
            res["eps_curve"][str(e)] = {"phi_dollars": dollars(r["phi_avg"], lam),
                                        "cell": fare_cell(dollars(r["phi_avg"], lam), fares),
                                        "window_share": r["window_share"]}
        res["J_margin"] = {}
        for m in (0.35, 0.7, 1.4):
            r = four_role(data, make_design(data, eps, margin_sd=m), seed)
            res["J_margin"][str(m)] = {"phi_dollars": dollars(r["phi_avg"], lam),
                                       "cell": fare_cell(dollars(r["phi_avg"], lam), fares),
                                       "beta2": r["beta2"]}
        res["J_tail"] = {}
        for q in (0.0025, 0.01, 0.02):
            r = four_role(data, make_design(data, eps, tail=q), seed)
            res["J_tail"][str(q)] = {"phi_dollars": dollars(r["phi_avg"], lam),
                                     "cell": fare_cell(dollars(r["phi_avg"], lam), fares),
                                     "beta2": r["beta2"], "window_share": r["window_share"]}
        res["fold_seeds"] = [dollars(four_role(data, design, s)["phi_avg"], lam)
                             for s in (seed + 1, seed + 2, seed + 3)]
        print(json.dumps({k: res[k] for k in ("alpha_only", "eps_curve", "J_margin",
                                               "J_tail", "fold_seeds")}, indent=1),
              f"[{time.time() - t0:.0f}s]", flush=True)
    if boot:
        with ProcessPoolExecutor(workers, initializer=_boot_init,
                                 initargs=(data, design, seed, grid)) as ex:
            draws = list(ex.map(_boot_draw, range(boot)))
        avg = np.array([dollars(d[0], lam) for d in draws])
        pool = np.array([dollars(d[1], lam) for d in draws])
        U = np.array([d[3] for d in draws])
        tau = np.array([d[4] for d in draws])
        res["bootstrap"] = {
            "B": boot, "sd_avg": float(avg.std(ddof=1)), "sd_pooled": float(pool.std(ddof=1)),
            "ci95_avg": np.percentile(avg, [2.5, 97.5]).tolist(),
            "ci95_pooled": np.percentile(pool, [2.5, 97.5]).tolist(),
            "share_boundary": float(np.mean([d[2] for d in draws])),
            "share_below_15": float(np.mean(avg < FARE0)),
            "cells_avg": pd.Series([fare_cell(a, fares) for a in avg]).value_counts().sort_index().to_dict(),
            "U_band": np.percentile(U, [2.5, 97.5], axis=0).tolist(),
            "tau_band": np.percentile(tau, [2.5, 97.5], axis=0).tolist(),
            "draws_avg": avg.round(4).tolist(),
        }
        print(json.dumps({k: v for k, v in res["bootstrap"].items()
                          if k not in ("U_band", "tau_band", "draws_avg")}, indent=1),
              f"[{time.time() - t0:.0f}s]", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{sample}_lam{lam:g}_eps{eps:g}" + ("" if effect == "distance" else f"_{effect}")
           + ("" if baseline == "flexible" else f"_{baseline}-baseline"))
    path = out_dir / f"summary_{tag}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote {path} [{time.time() - t0:.0f}s]")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", choices=("full", "restricted"), default="full")
    ap.add_argument("--lam", type=float, default=0.5, help="Box-Cox power for the fare")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--effect", choices=("distance", "distance2", "controls", "distance+Z"),
                    default="distance")
    ap.add_argument("--baseline", choices=("flexible", "linear"), default="flexible")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    run(a.sample, a.lam, a.eps, a.boot, a.workers, a.seed, a.out, extras=not a.no_extras,
        effect=a.effect, baseline=a.baseline)


if __name__ == "__main__":
    main()
