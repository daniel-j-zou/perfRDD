"""Estimator config for experiments/scripts/dose_schedule_apply.run (registered at run time, the
shared script is not edited):

    from experiments.scripts import dose_schedule_apply as app
    from experiments.datasets.mi_proposal_a import dose_config as c
    app.DATASETS["mi_proposal_a"] = c.config
    app.run("mi_proposal_a", argparse.Namespace(seed=0, x="linear", outcome=None))

Michigan foundation allowance, local districts x FY 2001-2024 (hold-harmless districts and the
district-years the formula does not reproduce are left out). Q = prior-year allowance above the
prior-year minimum ($/pupil); A = the year's increase beyond the basic increase ($/pupil),
reference 0; Y = change in F-33 current spending per pupil (default) or ``--outcome`` one of
adapter.OUTCOMES. ``args.x == "county"`` adds county indicators to X; ``args.years`` (optional)
restricts the fiscal years (e.g. range(2001, 2021) leaves out the COVID / ESSER years).
"""
from __future__ import annotations

import numpy as np

from experiments.methods import dose_schedule as ds

EDGES = np.array([-0.5, 0.5, 100.0, 300.0, 600.0, 1000.0, 1500.0])   # first bin: districts at the minimum
A_GRID = np.array([0.0, 25.0, 60.0, 120.0, 200.0, 418.0])


def config(args):
    from experiments.datasets.mi_proposal_a import adapter as ad
    outcome = getattr(args, "outcome", None) or "dcur"
    years = getattr(args, "years", None) or range(2001, 2025)
    s = ad.load_dose(outcome=outcome, years=years, county=(getattr(args, "x", "linear") == "county"))
    e, fn = s.extras, s.feature_names
    Z = s.X[:, [fn.index(c) for c in e["Z_names"]]]
    a_bar = float(np.max(e["A"]))
    prob = ds.DoseProblem(s.Q, s.X, e["A"], s.Y, Z, ds.DoseBasis(0.0, a_bar, l=3), a_lo=0.0)
    a_grid = A_GRID[A_GRID <= a_bar + 1e-9]
    return dict(prob=prob, cluster=e["district_id"], levels=None, edges=EDGES, n_bins=None, a_grid=a_grid,
                note=(f"MI foundation allowance: increase beyond the basic increase ($/pupil) on the prior-year "
                      f"allowance above the prior-year minimum; Y = {ad.OUTCOMES[outcome]}; FY {min(years)}-{max(years)}, "
                      f"{len(np.unique(e['group']))} yearly schedules; ref 0"))
