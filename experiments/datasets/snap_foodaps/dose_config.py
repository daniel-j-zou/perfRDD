"""Estimator config for the FoodAPS SNAP dose design (same dict as the loaders in
experiments/scripts/dose_schedule_apply.py). Run from code/:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -c "import argparse; \
        from experiments.scripts import dose_schedule_apply as app; \
        from experiments.datasets.snap_foodaps import dose_config as c; \
        app.DATASETS['snap_foodaps']=c.config; \
        app.run('snap_foodaps', argparse.Namespace(seed=0, x='linear', outcome=None))"

Q = 0.3 x reconstructed net income / maximum allotment (the score; <= 0 means the maximum
allotment), A = formula benefit / maximum allotment, a_ref = 1. Q trimmed to its 1-99% range.
--outcome: fah_food (default), fah_paid, total_food, fafh_paid.
"""
from __future__ import annotations

import numpy as np

from experiments.methods import dose_schedule as ds


def config(args):
    from experiments.datasets.snap_foodaps import adapter as ad
    outcome = getattr(args, "outcome", None) or "fah_food"
    s = ad.load_dose(outcome=outcome, dose="relative")
    fn, e = s.feature_names, s.extras
    Q = s.Q
    keep = (Q > np.quantile(Q, 0.01)) & (Q < np.quantile(Q, 0.99))
    Z = s.X[:, [fn.index(c) for c in e["Z_names"]]]
    prob = ds.DoseProblem(Q[keep], s.X[keep], e["A"][keep], s.Y[keep], Z[keep],
                          ds.DoseBasis(1.0, 1.0, l=3, a_lo=0.0), a_lo=0.0)
    return dict(prob=prob, cluster=e["psu"][keep], levels=None,
                edges=np.array([-0.85, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 4.0]), n_bins=None,
                a_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                note=f"FoodAPS SNAP: formula benefit / max allotment on 0.3 x net income / max allotment; "
                     f"reference = max allotment; Y = {outcome}; clusters = PSU")


CORE = ["size_2", "size_3", "size_4", "size_5", "size_6", "size_7p", "n_children", "n_elderly", "n_disabled",
        "n_employed", "n_earners", "n_retdis", "n_transfer", "pr_age", "pr_female", "educ_somecoll", "educ_ba",
        "own_home", "public_housing", "subsidized", "rural", "region_2", "region_3", "region_4"]


def config_core(args):
    """Same as ``config`` with 24 core covariates instead of 54 (stability check)."""
    from experiments.datasets.snap_foodaps import adapter as ad
    outcome = getattr(args, "outcome", None) or "fah_food"
    s = ad.load_dose(outcome=outcome, dose="relative")
    fn, e = s.feature_names, s.extras
    X = s.X[:, [fn.index(c) for c in CORE]]
    Q = s.Q
    keep = (Q > np.quantile(Q, 0.01)) & (Q < np.quantile(Q, 0.99))
    Z = s.X[:, [fn.index(c) for c in e["Z_names"]]]
    prob = ds.DoseProblem(Q[keep], X[keep], e["A"][keep], s.Y[keep], Z[keep],
                          ds.DoseBasis(1.0, 1.0, l=3, a_lo=0.0), a_lo=0.0)
    return dict(prob=prob, cluster=e["psu"][keep], levels=None,
                edges=np.array([-0.85, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 4.0]), n_bins=None,
                a_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                note=f"FoodAPS SNAP (24 core covariates): formula benefit / max allotment on 0.3 x net income / "
                     f"max allotment; reference = max allotment; Y = {outcome}; clusters = PSU")
