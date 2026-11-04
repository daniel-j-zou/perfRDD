"""Separable sieve for a link function in its intercept.

For a generalized linear outcome model on both arms,
m_d(X, eta) = F(c_d(eta) + X'delta_d), the paper's one-projection density block
(multiplier X~) does not apply, because the effect is not affine in X. The note
manuscript/binary_outcome_sketch.tex proposes expanding the link in its intercept,

    F(c + v) = sum_{k<=M} phi_k(c) psi_k(v) + e_M(c, v),

so that E_X[F(c_d(eta) + X'delta_d) 1{T > s}] is a sum of M one-dimensional tails
of projections with multipliers psi_k(X'delta_d). This script reports the uniform
error sup |e_M| of a Chebyshev interpolant of degree M-1 in c over c in [-3, 3] and
v in [-6, 6], for the logit and probit links. The bias must be o(n^{-1/2}).

Run:  python -I experiments/scripts/binary_link_sieve_check.py
"""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import chebyshev as C

C_HALF = 3.0          # c_d(eta) ranges over [-C_HALF, C_HALF]
V_HALF = 6.0          # X'delta_d ranges over [-V_HALF, V_HALF]
DEGREES = [4, 6, 8, 10, 12, 16]


def logit(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def probit(u: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(u / math.sqrt(2.0)))


def uniform_error(link, M: int, n_c: int = 241, n_v: int = 481) -> float:
    cs = np.linspace(-C_HALF, C_HALF, n_c)
    vs = np.linspace(-V_HALF, V_HALF, n_v)
    nodes = C_HALF * np.cos(np.pi * (np.arange(M) + 0.5) / M)
    err = 0.0
    for v in vs:
        coef = C.chebfit(nodes / C_HALF, link(nodes + v), M - 1)
        err = max(err, float(np.max(np.abs(C.chebval(cs / C_HALF, coef) - link(cs + v)))))
    return err


def main() -> None:
    print(f"uniform error of the separable expansion, c in [-{C_HALF},{C_HALF}], "
          f"v in [-{V_HALF},{V_HALF}]")
    print("link   " + "  ".join(f"M={M:>2}" for M in DEGREES))
    for name, link in [("logit ", logit), ("probit", probit)]:
        errs = [uniform_error(link, M) for M in DEGREES]
        print(name + " " + "  ".join(f"{e:7.1e}" for e in errs))


if __name__ == "__main__":
    main()
