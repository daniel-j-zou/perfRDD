"""Estimator configuration for the Chapter 70 dose, in the form of the loaders in
experiments/scripts/dose_schedule_apply.py (run without editing that file; see README).

    from experiments.scripts import dose_schedule_apply as app
    from experiments.datasets.ma_chapter70 import dose_config as c
    app.DATASETS["ma_chapter70"] = c.config
    app.run("ma_chapter70", argparse.Namespace(seed=0, x="linear", outcome=None))

args.outcome: None / "ela" (default), "math", "ela_ss", "math_ss", "ela_t0", "math_t0", "ppx";
append "_nolag" (e.g. "ela_nolag") to leave last year's score out of X, "_notrim" to keep
the first-stage outliers.

Sample: operating districts, FY2016-18 and FY2020-25, published aid = formula (1); the 1% tails
of the score are dropped, and the 2% tails of the full-sample first-stage residual (score jumps
of tiny districts and a few urban districts in the SOA years). Without that trim, the outcome
fit has empty tensor cells (eta tails x doses above $250) and the transported values explode.
The dose basis is a cubic in A on [0, 95% quantile] (l=0): with l >= 1 the fit is unstable.
"""
from __future__ import annotations

import numpy as np

from experiments.methods import dose_schedule as ds


def config(args):
    """Aid increase per foundation pupil ($) on the foundation gap per pupil, FY2016-25 (FY2019 has no
    t+1 test); nine schedules max(Q, m_t) with floors m_t = $0-104; reference $30."""
    from experiments.datasets.ma_chapter70 import adapter as ad
    out = args.outcome or "ela"
    score_lag, trim = not out.endswith("_nolag"), not out.endswith("_notrim")
    out = out.replace("_nolag", "").replace("_notrim", "")
    s = ad.load_dose(outcome=out, score_lag=score_lag)
    e, fn = s.extras, s.feature_names
    Q, A = s.Q, e["A"]
    keep = (Q > np.quantile(Q, 0.01)) & (Q < np.quantile(Q, 0.99))
    if trim:
        _, eta, _ = ds.first_stage(Q, s.X, keep)
        lo, hi = np.quantile(eta[keep], [0.02, 0.98])
        keep &= (eta > lo) & (eta < hi)
    Z = s.X[:, [fn.index(c) for c in e["Z_names"] if c in fn]]
    a_bar = float(np.quantile(A[keep], 0.95))
    prob = ds.DoseProblem(Q[keep], s.X[keep], A[keep], s.Y[keep], Z[keep],
                          ds.DoseBasis(30.0, a_bar, l=0, a_lo=0.0), a_lo=0.0)
    return dict(prob=prob, cluster=e["lea_id"][keep], levels=None,
                edges=np.array([-4000, -1000, -500, -250, 0, 250, 500, 1000, 2500.0]), n_bins=None,
                a_grid=np.array([0, 30, 60, 104, 250, 500, 750.0]),
                note=f"Chapter 70 aid increase ($/pupil) = max(Q, m_t) on the foundation gap per pupil Q; "
                     f"Y = {out}; Q_lag in X: {score_lag}; residual trim: {trim}; ref $30; a_bar = {a_bar:.0f}")
