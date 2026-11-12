"""Estimator config for the IME adjustment (same dict as the loaders in dose_schedule_apply.py).

    import argparse
    from experiments.scripts import dose_schedule_apply as app
    from experiments.datasets.medicare_ime import dose_config as c
    app.DATASETS["medicare_ime"] = c.config
    app.run("medicare_ime", argparse.Namespace(seed=0, x="linear", outcome=None))

``args.outcome``: log_cases (default) or cmi_lead. Optional attributes on ``args``:
``ime_years`` (default (2005, 2024)), ``ime_teaching_only`` (default False: then the reference
is the median teaching dose, 5.5%, and doses below it are fitted, a_lo = 0).
"""
from __future__ import annotations

import numpy as np

from experiments.methods import dose_schedule as ds

EDGES = np.array([-0.5, 0.5, 5, 10, 25, 50, 200.0])    # residents per 100 beds; first bin = non-teaching
A_GRID = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 25.0, 35.0])


def config(args):
    from experiments.datasets.medicare_ime import adapter as ad
    outcome = getattr(args, "outcome", None) or "log_cases"
    years = getattr(args, "ime_years", None) or (2005, 2024)
    teach = getattr(args, "ime_teaching_only", False)
    s = ad.load_dose(years=years, outcome=outcome, teaching_only=teach)
    e, fn = s.extras, s.feature_names
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    A = e["A"]
    a_bar = float(np.quantile(A[A > 0], 0.98))
    if teach:
        a_ref = 5.5
        basis = ds.DoseBasis(a_ref, a_bar, l=3, a_lo=0.0)
        edges = EDGES[1:]
        grid = np.array([0.5, 2.0, 5.5, 10.0, 15.0, 25.0, 35.0])
    else:
        basis = ds.DoseBasis(0.0, a_bar, l=3)
        edges, grid = EDGES, A_GRID
    prob = ds.DoseProblem(s.Q, s.X, A, s.Y, Z, basis, a_lo=0.0)
    return dict(prob=prob, cluster=e["unit_id"], levels=None, edges=edges, n_bins=None, a_grid=grid,
                note=(f"IME adjustment (%) on residents per 100 beds, FY {years[0]}-{years[1]}"
                      f"{', teaching hospitals only, reference 5.5%' if teach else ', reference 0 (non-teaching)'}; "
                      f"Y = {outcome}"))
