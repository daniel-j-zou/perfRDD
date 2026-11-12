"""Loader for experiments/scripts/dose_schedule_apply.py (registered at run time, not in DATASETS).

    from experiments.scripts import dose_schedule_apply as app
    from experiments.datasets.mimic_titration import dose_config as c
    app.DATASETS["mimic_titration"] = c.config
    app.run("mimic_titration", argparse.Namespace(seed=0, x="linear", outcome=None))

``args.protocol`` (default "potassium") picks the protocol; ``args.source`` (default "demo")
the data. Potassium: the KCl dose (mEq) is collapsed to the step levels 0, 20 (10-30),
40 (40-50) and 60 (>= 60); reference 0; Y = the next serum potassium within 24 h.
"""
from __future__ import annotations

import numpy as np

from experiments.methods import dose_schedule as ds
from experiments.datasets.mimic_titration import adapter as ad

COLLAPSE = {
    "potassium": (np.array([0.0, 20.0, 40.0, 60.0]), np.array([5.0, 35.0, 55.0])),
    "insulin": (np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]), np.array([1.0, 3.0, 5.0, 7.0, 9.0])),
    "insulin_emar": (np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]), np.array([1.0, 3.0, 5.0, 7.0, 9.0])),
    "heparin": (np.array([-4.0, -2.0, 0.0, 2.0, 4.0]), np.array([-3.0, -0.5, 0.5, 3.0])),
}
EDGES = {
    "potassium": np.array([1.5, 3.4, 3.7, 4.0, 4.3, 4.7, 9.0]),
    "insulin": np.array([20, 120, 150, 200, 250, 300, 1500.0]),
    "insulin_emar": np.array([20, 120, 150, 200, 250, 300, 1500.0]),
    "heparin": np.array([10, 40, 55, 70, 90, 120, 250.0]),
}


def collapse(protocol: str, A) -> np.ndarray:
    levels, cuts = COLLAPSE[protocol]
    return levels[np.searchsorted(cuts, np.asarray(A, float), side="right")]


def config(args):
    protocol = getattr(args, "protocol", "potassium")
    source = getattr(args, "source", "demo")
    s = ad.load_dose(protocol, source=source)
    e, fn = s.extras, s.feature_names
    A = collapse(protocol, e["A"])
    levels = np.array(sorted(set(A) - {e["a_ref"]}), float)
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    prob = ds.DoseProblem(s.Q, s.X, A, s.Y, Z, ds.DoseBasis(e["a_ref"], float(levels.max()), levels=levels),
                          a_lo=float(min(A.min(), e["a_ref"])))
    return dict(prob=prob, cluster=e["patient"], levels=levels, edges=EDGES[protocol], n_bins=None, a_grid=None,
                note=(f"MIMIC-IV {source} {protocol}: dose levels {levels.tolist()} vs {e['a_ref']:g} on the lab value; "
                      f"Y = the next value; n = {s.n} events, {len(set(e['patient']))} patients"))
