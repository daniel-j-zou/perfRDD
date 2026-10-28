"""Apply the two-score pairwise estimator to the real two-score datasets (exploratory).

Datasets (``experiments/datasets/<name>/``) and their deployed rules, on sign-adjusted
scores (every rule treats high values):
  * chile_two_score:  (reading + math-1)/2 >= 458          -> halfplane (lam 0.5, phi 458);
  * us_tracts_lic:    poverty >= 20 or income ratio <= 80  -> either (20, -80) on (pov, -mfi);
  * fl_school_grades: (3 ach + 4 gains)/7 >= B/C cutoff     -> halfplane (lam 3/7, phi 53.5).
``chile_two_score`` uses any undergraduate enrollment in 2025, ``chile_two_score_univ``
university enrollment in 2025 (the year after admission). Replications: the 2023 cohort
with university enrollment the next year, and the 2024 and 2025 cohorts with university
enrollment in the admission year.

For each dataset the rotated estimator is computed for four rule classes: the weighted
average with the deployed weights (only the cutoff moves), with chosen weights, both
cutoffs, and either cutoff. Reported: the estimated rule, U_hat at it and at the deployed
rule (per unit of the evaluation fold, in outcome units), the rotation spread, and
whether a rotation hit the edge of the policy box. Optional bootstrap with full
re-estimation (``--boot``).

Run from ``code/``:
    python -m experiments.scripts.two_score_apply                 # point estimates
    python -m experiments.scripts.two_score_apply --boot 40 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.methods.two_score import Rule, TwoScoreProblem, estimate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "two_score_apply"


def _chile(y="any_2025"):
    from experiments.datasets.chile_two_score import adapter as a
    s = {"any_2025": a.load, "univ_2025": a.load_univ_2025, "univ_2024": a.load_univ_2024,
         "2023_univ_2024": a.load_2023_univ_2024, "2025_univ_2025": a.load_2025_univ_2025}[y]()
    dep = Rule("halfplane", (0.5, 458.0))
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": 0.5, "box": (300.0, 700.0)},
        "weights": {"kind": "halfplane", "lam_range": (0.2, 0.8), "box": (300.0, 700.0)},
        "both": {"kind": "both", "box": ((250.0, 750.0), (250.0, 750.0))},
        "either": {"kind": "either", "box": ((300.0, 900.0), (300.0, 900.0))},
    }
    return s, dep, (1, 1), classes


def _tracts():
    from experiments.datasets.us_tracts_lic import adapter as a
    s = a.load()
    dep = Rule("either", (20.0, -80.0), signs=(1, -1))
    classes = {
        "either": {"kind": "either", "box": ((5.0, 45.0), (-130.0, -40.0))},
        "both": {"kind": "both", "box": ((0.0, 45.0), (-150.0, -40.0))},
        "weights": {"kind": "halfplane", "lam_range": (0.1, 0.9), "box": (-120.0, 20.0)},
    }
    return s, dep, (1, -1), classes


def _florida():
    from experiments.datasets.fl_school_grades import adapter as a
    s = a.load_two_score()
    lam0 = float(s.extras["weights"][0])
    cut = float(s.extras["cutoff"])
    dep = Rule("halfplane", (lam0, cut))
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": lam0, "box": (30.0, 80.0)},
        "weights": {"kind": "halfplane", "lam_range": (0.1, 0.9), "box": (30.0, 80.0)},
        "both": {"kind": "both", "box": ((20.0, 90.0), (20.0, 90.0))},
        "either": {"kind": "either", "box": ((30.0, 100.0), (30.0, 100.0))},
    }
    return s, dep, (1, 1), classes


def _nc():
    from experiments.datasets.nc_school_grades import adapter as a
    s = a.load()
    dep = Rule("halfplane", (0.8, 69.5))
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": 0.8, "box": (40.0, 95.0)},
        "weights": {"kind": "halfplane", "lam_range": (0.5, 0.95), "box": (40.0, 95.0)},
        "both": {"kind": "both", "box": ((30.0, 100.0), (40.0, 100.0))},
        "either": {"kind": "either", "box": ((40.0, 100.0), (50.0, 100.0))},
    }
    return s, dep, (1, 1), classes


def _nc_low_performing():
    """Designated if 0.8 A + 0.2 G < 54.5 and G < 85. On the scores (0.8 A + 0.2 G, G),
    sign-reversed, this is a both-cutoffs rule; a linear map of the scores keeps the
    score model."""
    from experiments.datasets.nc_school_grades import adapter as a
    s = a.load_low_performing()
    A, G = s.Q[:, 0], s.Q[:, 1]
    s.Q = np.column_stack([0.8 * A + 0.2 * G, G])
    s.treatment_rule = lambda Q: ((Q[:, 0] < 54.5) & (Q[:, 1] < 85.0)).astype(int)
    # strict inequalities: -Q >= -cutoff + tiny
    dep = Rule("both", (-54.5 + 1e-9, -85.0 + 1e-9), signs=(-1, -1))
    classes = {
        "both": {"kind": "both", "box": ((-80.0, -30.0), (-100.0, -55.0))},
        "either": {"kind": "either", "box": ((-80.0, -20.0), (-100.0, -40.0))},
    }
    return s, dep, (-1, -1), classes


def _la():
    from experiments.datasets.la_school_grades import adapter as a
    s = a.load_two_score()
    lam0, cut = float(s.extras["weights"][0]), float(s.extras["cutoff"])
    dep = Rule("halfplane", (lam0, cut))
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": lam0, "box": (50.0, 100.0)},
        "weights": {"kind": "halfplane", "lam_range": (0.4, 0.95), "box": (50.0, 100.0)},
        "both": {"kind": "both", "box": ((40.0, 130.0), (20.0, 150.0))},
        "either": {"kind": "either", "box": ((50.0, 150.0), (40.0, 150.0))},
    }
    return s, dep, (1, 1), classes


def _ciia():
    """Tracts failing the income test: eligible if poverty >= 20 or unemployment >= 1.5
    times the national rate (an either rule on two scores)."""
    from experiments.datasets.us_tracts_ciia import adapter as a
    s = a.load_two_score()
    dep = Rule("either", (20.0, 1.5))
    classes = {
        "either": {"kind": "either", "box": ((5.0, 45.0), (0.5, 3.0))},
        "both": {"kind": "both", "box": ((0.0, 45.0), (0.0, 3.0))},
    }
    return s, dep, (1, 1), classes


def _tx():
    from experiments.datasets.tx_school_ratings import adapter as a
    s = a.load_two_score()
    lam0, cut = float(s.extras["weights"][0]), float(s.extras["cutoff"])
    dep = Rule("halfplane", (lam0, cut))
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": lam0, "box": (70.0, 100.0)},
        "weights": {"kind": "halfplane", "lam_range": (0.4, 0.95), "box": (70.0, 100.0)},
        "both": {"kind": "both", "box": ((60.0, 105.0), (40.0, 105.0))},
        "either": {"kind": "either", "box": ((70.0, 110.0), (60.0, 110.0))},
    }
    return s, dep, (1, 1), classes


DATASETS = {"chile_two_score": _chile, "chile_two_score_univ": lambda: _chile("univ_2025"),
            "chile_2024_univ_same_year": lambda: _chile("univ_2024"),
            "chile_2023_univ_next_year": lambda: _chile("2023_univ_2024"),
            "chile_2025_univ_same_year": lambda: _chile("2025_univ_2025"),
            "us_tracts_lic": _tracts, "fl_school_grades": _florida,
            "nc_school_grades": _nc, "nc_low_performing": _nc_low_performing,
            "la_school_grades": _la, "us_tracts_ciia": _ciia, "tx_school_ratings": _tx}


def problem(name, cost=0.0, eps=0.05, idx=None):
    s, dep, signs, classes = DATASETS[name]()
    Q, X, Y = s.Q, s.X, s.Y
    if idx is not None:
        Q, X, Y = Q[idx], X[idx], Y[idx]
    # drop constant columns (state dummies absent in a resample, etc.)
    keep = X.std(0) > 0
    prob = TwoScoreProblem(Q=Q, X=X[:, keep], Y=Y, deployed=dep, cost=cost, eps=eps, signs=signs)
    return s, prob, classes


def _one(task):
    name, cost, eps, seed, boot = task
    s, _, _ = problem(name, cost, eps)
    idx = None
    if boot:
        idx = np.random.default_rng(seed).integers(0, s.n, s.n)
    _, prob, classes = problem(name, cost, eps, idx)
    r = estimate(prob, classes, seed=seed)
    r["D_agreement"] = float(np.mean(prob.D == s.D[idx] if idx is not None else prob.D == s.D))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["chile_two_score", "us_tracts_lic", "fl_school_grades"])
    ap.add_argument("--cost", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    allres = {}
    for name in a.datasets:
        t0 = time.time()
        point = _one((name, a.cost, a.eps, a.seed, False))
        res = {"point": point, "cost": a.cost, "eps": a.eps}
        if a.boot:
            tasks = [(name, a.cost, a.eps, a.seed + 101 * b + 1, True) for b in range(a.boot)]
            with ProcessPoolExecutor(a.workers) as ex:
                boots = list(ex.map(_one, tasks))
            res["boot"] = {k: {"theta_sd": np.std([b[k]["theta"] for b in boots], 0, ddof=1).tolist(),
                               "U_sd": float(np.std([b[k]["U"] for b in boots], ddof=1)),
                               "edge_share": float(np.mean([b[k]["edge_rotations"] > 0 for b in boots]))}
                           for k in point if isinstance(point[k], dict) and "theta" in point[k]}
            res["boot"]["U_deployed_sd"] = float(np.std([b["U_deployed"] for b in boots], ddof=1))
        allres[name] = res
        print(name, f"{time.time() - t0:.0f}s")
        print(json.dumps(res, indent=1)[:4000])
        tag = f"_c{a.cost:g}_eps{a.eps:g}" + (f"_boot{a.boot}" if a.boot else "")
        (OUT / f"{name}{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
