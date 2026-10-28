"""Readiness check for datasets meant for the threshold-rule extensions.

For a registered sample with one or more scores, this reports the facts the
rule-class analysis needs before any estimator is run:

* the deployed rule: treated share, and agreement with a published flag when the
  adapter supplies one in ``extras["eligible_published"]``;
* the score model Q_j = gamma_j' X~ + eta_j, fitted by OLS on [1, X]: R^2, sd(T_j)
  and sd(eta_j), and how much sd(eta_j) changes across deciles of T_j (X independent
  of eta implies no change);
* with two or more scores: corr(T) and corr(eta), and the spread of T along its
  weakest direction. A root-n limit for a rule in score space needs T to have a
  full-dimensional density; a thin weakest direction means the trade-off between the
  scores is weakly identified;
* the mass near the deployed boundary (|index - cutoff| < 0.25 sd(eta_index));
* with ``extras["group"]``: per-group n, treated share and R^2.

Run from the repo root::

    python -m experiments.scripts.multi_rule_readiness chile_two_score gpa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments._core.registry import load

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "multi_rule_readiness"


def _ols(X: np.ndarray, Q: np.ndarray):
    Z = np.column_stack([np.ones(len(X)), X])
    B = np.linalg.lstsq(Z, Q, rcond=None)[0]
    T = Z @ B
    return T, Q - T


def _sd_ratio(t: np.ndarray, e: np.ndarray) -> float:
    edges = np.quantile(t, np.linspace(0, 1, 11)[1:-1])
    dec = np.searchsorted(edges, t)
    sds = np.array([e[dec == d].std() for d in range(10) if (dec == d).sum() > 50])
    return float(sds.max() / sds.min())


def readiness(name: str) -> dict:
    s = load(name)
    Q = s.Q[:, None] if s.Q.ndim == 1 else s.Q
    D = s.D
    T, E = _ols(s.X, Q)
    r = {
        "name": s.name, "n": int(s.n), "k": int(Q.shape[1]), "p": int(s.p),
        "continuous_X": [f for f, c in zip(s.feature_names, s.X.T) if np.unique(c).size > 20],
        "treated_share": float(D.mean()),
        "R2": (1 - E.var(0) / Q.var(0)).round(3).tolist(),
        "sd_T": T.std(0).round(3).tolist(), "sd_eta": E.std(0).round(3).tolist(),
        "sd_eta_ratio_T_deciles": [round(_sd_ratio(T[:, j], E[:, j]), 2) for j in range(Q.shape[1])],
    }
    if "eligible_published" in s.extras:
        r["agreement_with_published_flag"] = float(np.mean(D == s.extras["eligible_published"]))
    w = np.asarray(s.extras.get("weights", np.ones(Q.shape[1]) / Q.shape[1]), float)
    cut = float(s.extras.get("cutoff", np.atleast_1d(s.threshold)[0]))
    if Q.shape[1] > 1:
        r["corr_T"] = np.corrcoef(T.T).round(3).tolist()
        r["corr_eta"] = np.corrcoef(E.T).round(3).tolist()
        evals, evecs = np.linalg.eigh(np.cov(T.T))
        r["sd_T_weakest_direction"] = float(np.sqrt(evals[0]))
        r["weakest_direction"] = evecs[:, 0].round(3).tolist()
        r["sd_T_weakest_over_sd_eta"] = float(np.sqrt(evals[0]) / np.sqrt(np.mean(E.var(0))))
    index = Q @ w
    h = 0.25 * (E @ w).std()
    r["near_boundary_share"] = float(np.mean(np.abs(index - cut) < h))
    if "group" in s.extras:
        g = np.asarray(s.extras["group"])
        r["groups"] = {}
        for v in np.unique(g):
            m = g == v
            Tg, Eg = _ols(s.X[m], Q[m])
            r["groups"][str(v)] = {"n": int(m.sum()), "treated_share": float(D[m].mean()),
                                   "R2": (1 - Eg.var(0) / Q[m].var(0)).round(3).tolist()}
        if "group_cutoffs" in s.extras:
            r["group_cutoffs"] = {str(k): v for k, v in s.extras["group_cutoffs"].items()}
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in args.datasets:
        out[name] = readiness(name)
        print(json.dumps(out[name], indent=1))
    (RUNS / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
