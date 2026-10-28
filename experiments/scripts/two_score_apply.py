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
re-estimation (``--boot``); for datasets in CLUSTER the bootstrap resamples whole
schools (or campuses), since they repeat across years.

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


def _chile_top10():
    """All elective takers: eligible if the PAES average >= 458 or the grade average is in
    the top 10% of the class (grade minus the class cutoff >= 0; cutoff at the bracket
    midpoint). Scores are on different scales, so no weighted-sum class."""
    from experiments.datasets.chile_top10 import adapter as a
    s = a.load_two_score_mid()
    dep = Rule("either", (458.0, 0.0))
    classes = {
        "either": {"kind": "either", "box": ((380.0, 560.0), (-0.6, 0.4))},
        "both": {"kind": "both", "box": ((250.0, 560.0), (-2.5, 0.4))},
    }
    return s, dep, (1, 1), classes


def _wi():
    """Wisconsin 4-star boundary on (Achievement, Growth). The deployed weight and cutoff
    differ by school: the achievement weight falls with the share of economically
    disadvantaged students, and the other priority areas move the cutoff
    (wi_report_cards/README.md). The counterfactual rules use one weight and cutoff."""
    from experiments.datasets.wi_report_cards import adapter as a
    s = a.load_two_score()
    w, c = s.extras["unit_weights"][:, 0], s.extras["unit_cutoff"]
    dep = Rule("unit_halfplane", (float(np.median(w)), float(np.median(c))), unit=(w, c))
    classes = {
        "weights": {"kind": "halfplane", "lam_range": (0.1, 0.9), "box": (45.0, 90.0)},
        "fixed_weights": {"kind": "halfplane", "lam_fixed": float(np.median(w)), "box": (45.0, 90.0)},
        "both": {"kind": "both", "box": ((30.0, 100.0), (20.0, 100.0))},
        "either": {"kind": "either", "box": ((50.0, 100.0), (50.0, 100.0))},
    }
    return s, dep, (1, 1), classes


def _indiana(boundary="DF"):
    """Grades 3-8 schools, 2016-17 and 2017-18 grades pooled: 0.5 Performance + 0.5 Growth
    (Growth runs 0-150). Small: 2,590 school-years."""
    from experiments.datasets.in_school_grades import adapter as a
    s = a.load_two_score_pooled(boundary)
    cut = float(s.extras["cutoff"])
    dep = Rule("halfplane", (0.5, cut))
    box = (cut - 15.0, cut + 15.0)
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": 0.5, "box": box},
        "weights": {"kind": "halfplane", "lam_range": (0.3, 0.7), "box": box},
        "both": {"kind": "both", "box": ((20.0, 100.0), (40.0, 150.0))},
        "either": {"kind": "either", "box": ((40.0, 110.0), (60.0, 150.0))},
    }
    return s, dep, (1, 1), classes


def _ca(subject="ela"):
    """California Dashboard, all students: Red = status Very Low (centred status < 0) and
    change below +3.0, a both-cutoffs rule on sign-reversed (status, change)."""
    from experiments.datasets.ca_dashboard import adapter as a
    s = a.load_two_score() if subject == "ela" else a.load_math_two_score()
    # scores on a 0.1 grid: -status >= 0.05 is status < -0.05; -change >= -2.95 is change < 2.95
    dep = Rule("both", (0.05, -2.95), signs=(-1, -1))
    classes = {
        "both": {"kind": "both", "box": ((-30.0, 20.0), (-15.0, 15.0))},
        "either": {"kind": "either", "box": ((-20.0, 60.0), (0.0, 40.0))},
        "weights": {"kind": "halfplane", "lam_range": (0.1, 0.9), "box": (-25.0, 25.0)},
    }
    return s, dep, (-1, -1), classes


def _ge(fail=False):
    """Gainful-employment programs (2015 D/E rates): not passing = annual rate > 8 and
    discretionary rate > 20 (fail: > 12 and > 30), a both-cutoffs rule; Y = no completions
    in 2019-20. Rates are truncated to 0.01, so "> 8" is ">= 8.005"."""
    from experiments.datasets.gainful_employment import adapter as a
    s = a.load_two_score_fail() if fail else a.load_two_score()
    dep = Rule("both", (12.005, 30.005) if fail else (8.005, 20.005))
    classes = {
        "both": {"kind": "both", "box": ((4.0, 16.0), (10.0, 45.0))},
        "either": {"kind": "either", "box": ((6.0, 25.0), (15.0, 99.0))},
    }
    return s, dep, (1, 1), classes


def _nyc(boundary="CD"):
    """NYC Progress Reports (elementary, K-8, middle; 2008-2012): overall = Environment +
    Performance + Progress + credit with maxima 40 (the first two) and 60, so in percent
    of each maximum, net of its share of the cut, the grade rule is 0.4 Q1 + 0.6 Q2 >= 0."""
    from experiments.datasets.nyc_progress_reports import adapter as a
    s = a.load_two_score(boundary)
    dep = Rule("halfplane", (0.4, -1e-6))     # units at the cut have Q = 0 up to rounding
    if boundary == "CD":
        classes = {
            "fixed_weights": {"kind": "halfplane", "lam_fixed": 0.4, "box": (-10.0, 15.0)},
            "weights": {"kind": "halfplane", "lam_range": (0.1, 0.9), "box": (-15.0, 20.0)},
            "both": {"kind": "both", "box": ((-30.0, 30.0), (-40.0, 20.0))},
            "either": {"kind": "either", "box": ((-10.0, 70.0), (-20.0, 60.0))},
        }
    else:
        classes = {
            "fixed_weights": {"kind": "halfplane", "lam_fixed": 0.4, "box": (-20.0, 20.0)},
            "weights": {"kind": "halfplane", "lam_range": (0.1, 0.9), "box": (-20.0, 20.0)},
            "both": {"kind": "both", "box": ((-40.0, 40.0), (-50.0, 30.0))},
            "either": {"kind": "either", "box": ((-20.0, 60.0), (-30.0, 50.0))},
        }
    return s, dep, (1, 1), classes


def _hacrp():
    """CMS HAC Reduction Program, FY2018-2019: penalized if 0.15 PSI-90 domain + 0.85
    infection domain (each standardized, net of the cutoff) >= 0. Y is the Total HAC Score
    three years later, which is worse when higher, so it is negated here: the estimator
    maximizes the outcome."""
    from experiments.datasets.cms_hacrp import adapter as a
    s = a.load_two_score()
    s.Y = -s.Y
    dep = Rule("halfplane", (0.15, 0.0))
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": 0.15, "box": (-0.5, 0.5)},
        "weights": {"kind": "halfplane", "lam_range": (0.05, 0.6), "box": (-0.5, 0.5)},
        "both": {"kind": "both", "box": ((-1.5, 2.0), (-1.5, 2.0))},
        "either": {"kind": "either", "box": ((-0.5, 3.0), (-0.5, 3.0))},
    }
    return s, dep, (1, 1), classes


def _esrd():
    """ESRD QIP, payment years 2015 and 2017: no payment reduction if 0.75 clinical +
    0.25 reporting >= 59.5 (the TPS is an integer, minimum 60); Y = the TPS two payment
    years later. D = 1 is the side without a reduction."""
    from experiments.datasets.esrd_qip import adapter as a
    s = a.load_two_score()
    dep = Rule("halfplane", (0.75, 59.5))
    classes = {
        "fixed_weights": {"kind": "halfplane", "lam_fixed": 0.75, "box": (45.0, 75.0)},
        "weights": {"kind": "halfplane", "lam_range": (0.5, 0.95), "box": (45.0, 75.0)},
        "both": {"kind": "both", "box": ((20.0, 90.0), (30.0, 100.0))},
        "either": {"kind": "either", "box": ((40.0, 100.0), (60.0, 100.0))},
    }
    return s, dep, (1, 1), classes


def _nursing():
    """Nursing homes, 5-star staffing: RN hours >= c_RN and total hours >= c_total (both,
    per resident day, net of the period's cut points). Y = log(1 + deficiency score of the
    next standard survey), worse when higher, so it is negated here."""
    from experiments.datasets.nursing_homes import adapter as a
    s = a.load_two_score()
    s.Y = -s.Y
    dep = Rule("both", (0.0, 0.0))
    classes = {
        "both": {"kind": "both", "box": ((-0.3, 0.3), (-1.0, 1.0))},
        "either": {"kind": "either", "box": ((-0.1, 0.6), (-0.3, 1.5))},
    }
    return s, dep, (1, 1), classes


def _fdic():
    """Banks, Q4 1992-2024: well capitalized if total risk-based capital >= 10% and
    leverage >= 5% (the binding pair; tier 1 and CET1 almost never bind alone), a
    both-cutoffs rule; Y = log loan growth over the next four quarters. Banks bunch just
    above both cutoffs, so the score density is not continuous there."""
    from experiments.datasets.fdic_capital import adapter as a
    s = a.load_two_score()
    dep = Rule("both", (10.0, 5.0))
    classes = {
        "both": {"kind": "both", "box": ((8.0, 13.0), (3.5, 7.0))},
        "either": {"kind": "either", "box": ((9.0, 16.0), (4.5, 10.0))},
    }
    return s, dep, (1, 1), classes


DATASETS = {"chile_two_score": _chile, "chile_two_score_univ": lambda: _chile("univ_2025"),
            "chile_top10": _chile_top10, "wi_report_cards": _wi,
            "ca_dashboard_ela": _ca, "ca_dashboard_math": lambda: _ca("math"),
            "nyc_progress_reports": _nyc, "nyc_progress_reports_ab": lambda: _nyc("AB"),
            "fdic_capital": _fdic, "nursing_homes": _nursing, "cms_hacrp": _hacrp, "esrd_qip": _esrd,
            "gainful_employment": _ge, "gainful_employment_fail": lambda: _ge(True),
            "in_school_grades_df": _indiana, "in_school_grades_bc": lambda: _indiana("BC"),
            "chile_2024_univ_same_year": lambda: _chile("univ_2024"),
            "chile_2023_univ_next_year": lambda: _chile("2023_univ_2024"),
            "chile_2025_univ_same_year": lambda: _chile("2025_univ_2025"),
            "us_tracts_lic": _tracts, "fl_school_grades": _florida,
            "nc_school_grades": _nc, "nc_low_performing": _nc_low_performing,
            "la_school_grades": _la, "us_tracts_ciia": _ciia, "tx_school_ratings": _tx}


# unit id for the cluster bootstrap (schools repeat across years)
CLUSTER = {"tx_school_ratings": "campus", "fl_school_grades": "school",
           "nc_school_grades": "agency_code", "nc_low_performing": "agency_code",
           "nyc_progress_reports": "dbn", "nyc_progress_reports_ab": "dbn"}


def _draw(s, name, seed):
    """Bootstrap row indices: whole clusters when the dataset has a unit id."""
    rng = np.random.default_rng(seed)
    if name not in CLUSTER:
        return rng.integers(0, s.n, s.n)
    _, g = np.unique(np.asarray(s.extras[CLUSTER[name]]), return_inverse=True)
    rows = np.split(np.argsort(g, kind="stable"), np.cumsum(np.bincount(g))[:-1])
    pick = rng.integers(0, len(rows), len(rows))
    return np.concatenate([rows[k] for k in pick])


def problem(name, cost=0.0, eps=0.05, idx=None):
    s, dep, signs, classes = DATASETS[name]()
    Q, X, Y = s.Q, s.X, s.Y
    if idx is not None:
        Q, X, Y = Q[idx], X[idx], Y[idx]
        dep = dep.subset(idx)
    # drop constant columns (state dummies absent in a resample, etc.)
    keep = X.std(0) > 0
    prob = TwoScoreProblem(Q=Q, X=X[:, keep], Y=Y, deployed=dep, cost=cost, eps=eps, signs=signs)
    return s, prob, classes


def _one(task):
    name, cost, eps, seed, boot = task
    s, _, _ = problem(name, cost, eps)
    idx = None
    if boot:
        idx = _draw(s, name, seed)
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
        boot = ("_cboot" if name in CLUSTER else "_boot") + str(a.boot)
        tag = f"_c{a.cost:g}_eps{a.eps:g}" + (boot if a.boot else "")
        (OUT / f"{name}{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
