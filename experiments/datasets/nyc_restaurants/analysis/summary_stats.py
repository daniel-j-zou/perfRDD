"""The numbers quoted in README.md: n, treated share, base rate of Y, R-squared of Q on X,
the bunching counts at 12-15 points, grade compliance and follow-up timing.

    python -m experiments.datasets.nyc_restaurants.analysis.summary_stats
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.datasets.nyc_restaurants.adapter import (
    CYCLE_INITIAL, CYCLE_RE, build, load, load_exit, sample_frame)


def r2(Q, X):
    A = np.column_stack([np.ones(len(Q)), X])
    beta, *_ = np.linalg.lstsq(A, Q, rcond=None)
    res = Q - A @ beta
    return 1 - res.var() / Q.var()


def describe(s, d, label):
    D = s.D
    print(f"\n== {label}: n = {s.n}, p = {s.p}")
    print(f"  treated share (Q <= 13): {D.mean():.4f}   base rate of Y: {s.Y.mean():.4f}")
    print(f"  Y | treated: {s.Y[D == 1].mean():.4f}   Y | control: {s.Y[D == 0].mean():.4f}")
    print(f"  R2 of Q on X: {r2(s.Q, s.X):.4f}")
    print(f"  years: {d.year.value_counts().sort_index().astype(int).to_dict()}")
    vc = d.score.value_counts()
    print(f"  bunching in sample 12/13/14/15: {[int(vc.get(k, 0)) for k in (12, 13, 14, 15)]}")
    print(f"  grade at the initial inspection, Q <= 13: {d.grade[d.score <= 13].value_counts().to_dict()}")
    print(f"  grade at the initial inspection, Q >= 14: {d.grade[d.score >= 14].value_counts().to_dict()}")
    if "next_inspection_type" in d:
        print("  next inspection type by treatment:")
        print(pd.crosstab(d.score <= 13, d.next_inspection_type))
    print(f"  days to next inspection, treated: {d.next_days[d.score <= 13].describe()[['25%', '50%', '75%']].to_dict()}")
    print(f"  days to next inspection, control: {d.next_days[d.score >= 14].describe()[['25%', '50%', '75%']].to_dict()}")


if __name__ == "__main__":
    insp = build()
    ci = insp[(insp.inspection_type == CYCLE_INITIAL) & insp.score.notna()]
    vc = ci.score.value_counts()
    print(f"all cycle initial inspections with a score: {len(ci)}, restaurants {ci.camis.nunique()}")
    print(f"  bunching 12/13/14/15: {[int(vc.get(k, 0)) for k in (12, 13, 14, 15)]}")
    print(f"  score 10..20: {[int(vc.get(k, 0)) for k in range(10, 21)]}")
    re = insp[(insp.inspection_type == CYCLE_RE) & insp.score.notna()]
    vc = re.score.value_counts()
    print(f"all cycle re-inspections with a score: {len(re)}; bunching 12/13/14/15: {[int(vc.get(k, 0)) for k in (12, 13, 14, 15)]}")
    print(f"record date: {insp.record_date.max().date()}; inspection dates {insp.inspection_date.min().date()} .. {insp.inspection_date.max().date()}")

    s = load(); d = sample_frame("next_score_14"); describe(s, d, "load() next_kind=any")
    s = load(next_kind="initial"); d = sample_frame("next_score_14", next_kind="initial"); describe(s, d, "load(next_kind='initial')")
    s = load_exit(); d = sample_frame("no_inspection_18m"); describe(s, d, "load_exit()")
    s = load(with_critical=True); print(f"\nR2 of Q on X with n_critical: {r2(s.Q, s.X):.4f}")
