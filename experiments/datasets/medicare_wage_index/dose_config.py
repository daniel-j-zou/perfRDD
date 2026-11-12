"""Estimator config for the low wage index policy (same dict as the loaders in dose_schedule_apply.py).

    import argparse
    from experiments.scripts import dose_schedule_apply as app
    from experiments.datasets.medicare_wage_index import dose_config as c
    app.DATASETS["medicare_wage_index"] = c.config
    app.run("medicare_wage_index", argparse.Namespace(seed=0, x="linear", outcome=None))

``args.outcome``: log_ahw (default), dlog_ahw or log_cases. Optional attributes on ``args``:
``wi_x`` ("full", "state", "nowage"; default "full"), ``wi_years`` (default (2016, 2023); log_cases
uses (2016, 2024)), ``wi_placebo`` (e.g. (2018, 2019) with wi_years=(2016, 2019)).

Sample: Q < 100 and eta inside the dosed units' 0.5-99.5% range (adapter defaults). The dose
basis has one interior knot (l=1): with l=3 the fit is unstable at doses above 4.5 points,
where there are few hospital-years.
"""
from __future__ import annotations

import numpy as np

from experiments.methods import dose_schedule as ds

EDGES = np.array([60, 75, 80, 84.7, 90, 100.0])        # wage index points (x100)
A_GRID = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 6.0])


def config(args):
    from experiments.datasets.medicare_wage_index import adapter as ad
    outcome = getattr(args, "outcome", None) or "log_ahw"
    x = getattr(args, "wi_x", "full")
    years = getattr(args, "wi_years", None) or ((2016, 2024) if outcome == "log_cases" else (2016, 2023))
    placebo = getattr(args, "wi_placebo", None)
    s = ad.load_dose(years=years, outcome=outcome, x=x, placebo=placebo)
    e, fn = s.extras, s.feature_names
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    A = e["A"]
    a_bar = float(np.quantile(A[A > 0], 0.98))
    prob = ds.DoseProblem(s.Q, s.X, A, s.Y, Z, ds.DoseBasis(0.0, a_bar, l=1), a_lo=0.0)
    return dict(prob=prob, cluster=e["unit_id"], levels=None, edges=EDGES, n_bins=None, a_grid=A_GRID,
                note=(f"low wage index increase on the wage index before the policy (both in points, x100), "
                      f"FY {years[0]}-{years[1]}{' placebo ' + str(placebo) if placebo else ''}; Y = {outcome}; "
                      f"X = {x}; Q < 100, eta trimmed to the dosed range; reference 0"))
