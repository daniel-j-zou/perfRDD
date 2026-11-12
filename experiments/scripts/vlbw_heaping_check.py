"""Heaping at 1500 g and the local jump in one-year death, with and without a donut.

Barreca, Guldi, Lindo and Waddell (2011, QJE) show that births recorded at exactly 1500 g
(a round-number heap) have higher mortality, which inflates the very-low-birth-weight
discontinuity; a donut that drops the heap shrinks it. This script reports, for the 1983-1991
and 1995-2002 NCHS cohorts: the heap ratio at 1500 g and at other round weights, the death
rate at the heap against its neighbours, and local linear jumps (below minus above 1500 g,
triangular kernel, HC1 standard errors) at 50 g and 100 g bandwidths, with no donut, a donut
that drops 1500 g, and one that drops 1495-1505 g.

    python -m experiments.scripts.vlbw_heaping_check
"""
from __future__ import annotations

import numpy as np

from experiments.datasets.vlbw import adapter as V


def jump(w, y, h, drop=None):
    m = np.abs(w - 1499.5) <= h
    if drop is not None:
        m &= ~((w >= drop[0]) & (w <= drop[1]))
    q = w[m] - 1499.5
    d = (q < 0).astype(float)
    k = 1 - np.abs(q) / h                       # triangular kernel
    X = np.column_stack([np.ones(m.sum()), d, q, d * q])
    sw = np.sqrt(k)
    b, *_ = np.linalg.lstsq(X * sw[:, None], y[m] * sw, rcond=None)
    r = y[m] - X @ b
    A = np.linalg.inv((X * k[:, None]).T @ X)
    V_ = A @ ((X * (k * r)[:, None]).T @ (X * (k * r)[:, None])) @ A
    return 100 * b[1], 100 * np.sqrt(V_[1, 1]), int(m.sum())


def report(label, years):
    d = V.sample_frame((1000.0, 2000.0), years)
    w, y = d.weight.to_numpy(float), d.death.to_numpy(float)
    print(f"\n== {label}: n = {len(w):,} births 1000-2000 g ==")
    for c in (1400, 1450, 1500, 1550, 1600):
        at = np.sum(w == c)
        nb = np.mean([np.sum(w == c + k) for k in (-3, -2, -1, 1, 2, 3)])
        print(f"  heap at {c} g: {at:6d} births vs {nb:7.1f} on average at +-1..3 g (ratio {at / max(nb, 1):.1f})")
    at = w == 1500
    below = (w >= 1490) & (w <= 1499)
    above = (w >= 1501) & (w <= 1510)
    print(f"  death rate: 1490-1499 g {100 * y[below].mean():.2f}%, exactly 1500 g {100 * y[at].mean():.2f}%, "
          f"1501-1510 g {100 * y[above].mean():.2f}%")
    for h in (50, 100):
        for name, drop in (("no donut", None), ("drop 1500", (1500, 1500)), ("drop 1495-1505", (1495, 1505))):
            j, se, n = jump(w, y, h, drop)
            print(f"  h={h:3d} {name:15s}: jump {j:+.2f} pp (se {se:.2f}), n={n:,}")


if __name__ == "__main__":
    # 1983-1984 are left out: they keep a 50% sample of survivors (record weight 2) in some
    # states, so unweighted death rates are too high.
    early = tuple(y for y in V.EARLY_YEARS if y >= 1985)
    report("1985-1991", early)
    report("1995-2002", V.YEARS)
    report("1985-2002", early + tuple(V.YEARS))
