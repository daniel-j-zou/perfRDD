"""Estimator configuration for the Texas CEP dataset (same dict as the loaders in
experiments/scripts/dose_schedule_apply.py), without editing the shared script:

    from experiments.scripts import dose_schedule_apply as app
    from experiments.datasets.cep_meals import dose_config as c
    app.DATASETS['cep_meals'] = c.config
    app.run('cep_meals', argparse.Namespace(seed=0, x='linear', outcome=None))

Dose: the free claiming share OFFERED on the site's own ISP, 1{Q >= c_t} min(1, 1.6 Q), with
c_t = 0.40 (program years to 2023-24) and 0.25 (from 2024-25); reference 0 (not offered).
There are no doses in (0, 0.40): the spline in the dose (l = 2 interior knots; with l = 3 a basis
function lives mostly in the gap and a = 0.40-0.52 explodes) only joins 0 to [0.40, 1].
``args.outcome``: "lunch" (default), "breakfast", "meals"; or "received:<outcome>" to use the
received dose (CEP x implied claiming share), which is chosen, not set by the schedule.
"""
from __future__ import annotations

import numpy as np

from experiments.methods import dose_schedule as ds

EDGES = np.array([0, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.625, 0.70, 0.80, 1.0])
A_GRID = np.array([0, 0.45, 0.55, 0.64, 0.75, 0.85, 1.0])


def variant(outcome: str = "lunch", dose: str = "offered", years=None, l: int = 2, a_grid=None):
    """A config function for a variant (outcome, offered / received dose, program years, knots)."""
    def cfg(args):
        from experiments.datasets.cep_meals import adapter as ad
        kw = {} if years is None else {"years": tuple(years)}
        s = ad.load_dose(outcome=outcome, dose=dose, **kw)
        e, fn = s.extras, s.feature_names
        Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
        prob = ds.DoseProblem(s.Q, s.X, e["A"], s.Y, Z, ds.DoseBasis(0.0, 1.0, l=l), a_lo=0.0)
        return dict(prob=prob, cluster=e["ce_id"], levels=None, edges=EDGES, n_bins=None,
                    a_grid=A_GRID if a_grid is None else np.asarray(a_grid, float),
                    note=f"CEP free claiming share ({dose}) on the site ISP, thresholds 0.40 / 0.25; "
                         f"Y = {outcome} participation (meals per enrolled student-day); reference 0; "
                         f"years {years or 'default'}; l = {l}")
    return cfg


def config(args):
    spec = args.outcome or "lunch"
    dose, outcome = ("received", spec.split(":", 1)[1]) if spec.startswith("received:") else ("offered", spec)
    return variant(outcome=outcome, dose=dose)(args)
