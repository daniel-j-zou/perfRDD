"""Cross-fitted nonparametric residualizers for first-stage Q ~ X (and Y ~ X).

Models considered, picked per outer fold by held-out R^2 on a 20% inner split:
  - GAM   : additive cubic B-splines (kn=8 per feature, ridge lam=1.0)
  - MLP   : 2-layer ReLU (128, 64), early stopping

Empirical reasoning (see /tmp/bench_first_stage.py):
  * MLP wins decisively on HMDA where strong X interactions exist (R^2 0.90 vs 0.79
    for the GAM at kn=8); it ties or marginally beats GAM on lending_club; it loses
    to GAM on small datasets (NHANES n~5k) where it overfits, and on noisy ones
    (OULAD) where there is little structure to learn.
  * GAM is fast (<1s) and consistent; MLP needs more compute but captures
    interactions GAMs cannot.
  * We auto-select per fold to get the better of both worlds. For n < 5000 we
    skip MLP entirely (small-sample overfit risk).

Cross-fitting: 2-fold (KFold, shuffled). Within each training fold we hold out
20% as the candidate-selection validation set, refit the winner on the full
training fold, then predict on the held-out outer fold. The combined out-of-fold
predictions are subtracted from `target` to yield the residualization.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


MAX_FIT_N = 10_000
MLP_MIN_N = 5_000
GAM_KN_DEFAULT = 8
GAM_LAM_DEFAULT = 1.0
MLP_HIDDEN_DEFAULT = (128, 64)
MLP_MAX_ITER = 120


# --------------------------------------------------------------- GAM helpers

def _spline_basis_1d(x: np.ndarray, kn: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    lo = float(np.percentile(x, 0.5))
    hi = float(np.percentile(x, 99.5))
    if hi - lo < 1e-12:
        # Constant feature — fall back to a trivial single-column basis (the value 1).
        return np.ones((len(x), 1)), {"trivial": True, "lo": lo, "hi": hi, "t": None}
    interior = np.linspace(lo, hi, kn + 2)[1:-1]
    t = np.concatenate([np.repeat(lo, 4), interior, np.repeat(hi, 4)])
    pts = np.clip(x, lo, hi)
    B = BSpline.design_matrix(pts, t, 3).toarray()
    return B[:, 1:], {"trivial": False, "lo": lo, "hi": hi, "t": t}


def _gam_features(X: np.ndarray, kn: int, infos: List[Dict[str, Any]] | None = None):
    cols = []
    new_infos = []
    if infos is None:
        for j in range(X.shape[1]):
            B, info = _spline_basis_1d(X[:, j], kn)
            cols.append(B)
            new_infos.append(info)
        return np.column_stack(cols), new_infos
    for j in range(X.shape[1]):
        info = infos[j]
        if info["trivial"]:
            cols.append(np.ones((X.shape[0], 1)))
        else:
            pts = np.clip(X[:, j], info["lo"], info["hi"])
            B = BSpline.design_matrix(pts, info["t"], 3).toarray()
            cols.append(B[:, 1:])
    return np.column_stack(cols), infos


@dataclass
class GAMModel:
    ridge: Ridge
    infos: List[Dict[str, Any]]
    kn: int
    lam: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        F, _ = _gam_features(X, self.kn, infos=self.infos)
        return self.ridge.predict(F)


def _fit_gam(X: np.ndarray, y: np.ndarray, kn: int, lam: float) -> GAMModel:
    F, infos = _gam_features(X, kn=kn)
    ridge = Ridge(alpha=lam, fit_intercept=True).fit(F, y)
    return GAMModel(ridge=ridge, infos=infos, kn=kn, lam=lam)


# --------------------------------------------------------------- MLP helpers

@dataclass
class MLPModel:
    scaler: StandardScaler
    mlp: MLPRegressor
    hidden: Tuple[int, ...]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mlp.predict(self.scaler.transform(X))


def _fit_mlp(X: np.ndarray, y: np.ndarray, hidden, seed: int) -> MLPModel:
    sc = StandardScaler().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=seed,
        ).fit(sc.transform(X), y)
    return MLPModel(scaler=sc, mlp=mlp, hidden=hidden)


# --------------------------------------------------------------- residualizer

def _r2(y, yhat) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return 0.0
    ssr = float(((y - yhat) ** 2).sum())
    return float(1.0 - ssr / sst)


def residualize(
    target: np.ndarray, X: np.ndarray, *,
    n_folds: int = 2, seed: int = 0,
    max_fit_n: int = MAX_FIT_N,
    label: str = "Q",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (residuals, predictions, info)."""
    n = len(target)
    if n < 2 * n_folds:
        # Degenerate, fall back to mean.
        m = float(target.mean())
        pred = np.full(n, m)
        return target - pred, pred, {"method": "mean", "oof_r2": 0.0, "label": label}

    rng = np.random.default_rng(seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yhat = np.empty(n)
    fold_records: List[Dict[str, Any]] = []

    for fid, (tr_all, te) in enumerate(kf.split(X)):
        # Inner held-out 20% for candidate selection.
        perm = rng.permutation(len(tr_all))
        n_val = max(50, len(tr_all) // 5)
        v_idx_local = perm[:n_val]
        f_idx_local = perm[n_val:]
        v = tr_all[v_idx_local]
        f = tr_all[f_idx_local]

        # Cap training subsample for speed.
        if len(f) > max_fit_n:
            sub = rng.choice(len(f), size=max_fit_n, replace=False)
            f = f[sub]

        candidates = []
        # GAM
        try:
            gam = _fit_gam(X[f], target[f], kn=GAM_KN_DEFAULT, lam=GAM_LAM_DEFAULT)
            r2 = _r2(target[v], gam.predict(X[v]))
            candidates.append(("gam", {"kn": GAM_KN_DEFAULT, "lam": GAM_LAM_DEFAULT}, r2))
        except Exception as e:
            candidates.append(("gam", {"error": repr(e)}, -np.inf))

        # MLP (skip for very small training sets)
        if len(f) >= MLP_MIN_N:
            try:
                mlp = _fit_mlp(X[f], target[f], MLP_HIDDEN_DEFAULT, seed + fid)
                r2 = _r2(target[v], mlp.predict(X[v]))
                candidates.append(("mlp", {"hidden": MLP_HIDDEN_DEFAULT}, r2))
            except Exception as e:
                candidates.append(("mlp", {"error": repr(e)}, -np.inf))

        winner = max(candidates, key=lambda c: c[2])
        # Refit on all training data (subsampled if needed) using the winner
        tr_use = tr_all
        if len(tr_use) > max_fit_n:
            tr_use = rng.choice(tr_use, size=max_fit_n, replace=False)
        if winner[0] == "gam":
            best = _fit_gam(X[tr_use], target[tr_use],
                            kn=winner[1]["kn"], lam=winner[1]["lam"])
        else:
            best = _fit_mlp(X[tr_use], target[tr_use],
                            winner[1]["hidden"], seed + fid + 100)

        yhat[te] = best.predict(X[te])
        fold_records.append({
            "fold": fid, "winner": winner[0], "winner_params": winner[1],
            "val_r2": float(winner[2]),
            "candidates": [(c[0], c[1], float(c[2])) for c in candidates],
            "n_train_used": int(len(tr_use)), "n_val": int(len(v)), "n_test": int(len(te)),
        })

    info = {
        "method": "auto",
        "label": label,
        "n_folds": n_folds,
        "fold_records": fold_records,
        "oof_r2": _r2(target, yhat),
    }
    return target - yhat, yhat, info
