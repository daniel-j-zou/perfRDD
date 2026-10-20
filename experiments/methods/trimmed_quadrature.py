"""Hard-trimmed population expectations by adaptive quadrature on the trim window.

The hard-trimmed target integrates over eta only on the trim window
[l0, u0].  Multiplying a Gauss-Hermite rule by the trim indicator leaves only
a handful of nodes inside the window and moves the optimum (for the four-role
check design, from -0.1197699 to -0.1200638; ``ds_identification_check.py``,
check 10).  Integrating on the window itself, where the integrand is smooth,
removes that error.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
from scipy.stats import norm


def trimmed_normal_mean(
    fun: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    *,
    sd: float = 1.0,
    epsabs: float = 1e-13,
    epsrel: float = 1e-12,
) -> float:
    """E[fun(eta) 1{lo <= eta <= hi}] for eta ~ N(0, sd^2).

    ``fun`` takes and returns arrays; it is called on one point at a time.
    Adaptive Gauss-Kronrod quadrature (``scipy.integrate.quad``) on [lo, hi].
    """
    def integrand(e: float) -> float:
        return float(np.asarray(fun(np.array([e])), dtype=float).ravel()[0]) * float(
            norm.pdf(e, scale=sd))

    value, _ = integrate.quad(integrand, float(lo), float(hi),
                              epsabs=epsabs, epsrel=epsrel, limit=200)
    return float(value)
