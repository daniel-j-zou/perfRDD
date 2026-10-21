"""Monte Carlo of the four role scores (direct evaluation of the paper's formulas),
and a formula-free sieve check of the outcome representer.

1. Draw N observations, evaluate psi_U, psi_o, psi_T, psi_gamma, and estimate
   their means and 4x4 covariance matrix (role variances and cross-covariances).
2. Outcome representer without the closed form: on a large simulated sample,
   the finite-sieve representer r_K = z_K' Q^{-1} l_K (paper eq. prelim-alpha-riesz)
   with Q the empirical arm-wise Gram matrix of (N_K(eta), X) on J and l_K the
   functional L_out applied to the basis.  The basis has knots of multiplicity 4
   at l0 and u0 so the sieve can follow the jumps of r_out.  ||r_K||^2 and
   A_out,K = E[xi X~ r_K] are compared with the closed-form values.
"""
from __future__ import annotations

import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_pieces as P  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def scores(rng, n):
    X = rng.standard_normal((n, 2))
    eta = rng.standard_normal(n)
    eps = P.SIG * rng.standard_normal(n)
    T = X[:, 0]
    D = (T + eta > 0).astype(float)
    I0 = (eta >= P.L0) & (eta <= P.U0)
    SJ = (eta >= P.J0) & (eta <= P.J1)
    psiU = -np.where(I0, P.Fs(eta), 0.0)
    # outcome representer, closed form (Lemma ds-rout)
    r = np.zeros(n)
    for d in (0, 1):
        on = (D == d) & SJ
        e = eta[on]
        part = (2 * d - 1) * np.where(I0[on], P.g(P.PHI - e), 0.0) / P.pi_d(d, e)
        part += (X[on, 0] - P.mu1_d(d, e)) * P.omega[d][0] + X[on, 1] * P.omega[d][1]
        r[on] = part
    psiO = -r * eps
    A, w2 = P.psiT_parts(T)
    psiT = A - w2 * X[:, 1]
    Ab = np.asarray(P.Abar)
    psiG = (Ab[0] + X @ Ab[1:]) * eta
    xi = np.where(D == 1, P.dalpha(eta), 0.0) + P.db(eta)
    Aout_mc = np.column_stack((np.ones(n), X)) * (xi * r)[:, None]
    return np.column_stack((psiU, psiO, psiT, psiG)), Aout_mc


def part1(N=20_000_000, chunk=2_000_000, seed=7):
    rng = np.random.default_rng(seed)
    S1 = np.zeros(4); S2 = np.zeros((4, 4)); SA = np.zeros(3); SA2 = np.zeros(3)
    sums_sq = []
    m = 0
    for _ in range(N // chunk):
        s, a = scores(rng, chunk)
        S1 += s.sum(0); S2 += s.T @ s; SA += a.sum(0); SA2 += (a ** 2).sum(0)
        m += chunk
        tot = s.sum(1)
        sums_sq.append(((tot - 0) ** 2).mean())
    mean = S1 / m
    cov = S2 / m - np.outer(mean, mean)
    # s.e. of each variance from chunk-to-chunk variation is crude; use the 4th moment instead
    Aout = SA / m
    Aout_se = np.sqrt((SA2 / m - Aout ** 2) / m)
    return {"N": m, "mean": mean.tolist(), "cov": cov.tolist(),
            "var_sum": float(cov.sum()), "A_out_mc": Aout.tolist(), "A_out_mc_se": Aout_se.tolist(),
            "var_sum_chunks_sd_over_sqrtk": float(np.std(sums_sq, ddof=1) / np.sqrt(len(sums_sq)))}


def part1_se(N=4_000_000, seed=8):
    """Delta-free s.e. for each variance: Var(psi^2)/N."""
    rng = np.random.default_rng(seed)
    s, _ = scores(rng, N)
    v = s.var(0)
    se = np.sqrt(((s - s.mean(0)) ** 2).var(0) / N)
    tot = s.sum(1)
    return {"var": v.tolist(), "se_at_N=4e6": se.tolist(),
            "var_sum": float(tot.var()), "var_sum_se": float(np.sqrt(((tot - tot.mean()) ** 2).var() / N))}


def rich_knots(n_uniform=40):
    lo, hi = P.J0, P.J1
    inner = list(np.linspace(lo, hi, n_uniform + 2)[1:-1])
    inner = [x for x in inner if abs(x - P.L0) > 1e-3 and abs(x - P.U0) > 1e-3]
    inner += [P.L0] * 4 + [P.U0] * 4
    inner = sorted(inner)
    return np.concatenate([[lo] * 4, inner, [hi] * 4])


def basis(e, t, sparse=False):
    M = BSpline.design_matrix(np.clip(e, t[0], t[-1]), t, 3)
    return M if sparse else M.toarray()


def gl_nodes(breaks, order=20):
    x, w = leggauss(order)
    nodes, weights = [], []
    br = np.unique(np.asarray(breaks))
    for a, b in zip(br[:-1], br[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (b + a)); weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def ell_vector(t):
    """l_K for one arm (sign (2d-1) applied later): (E[I0 g(phi*-eta) N(eta)], B)."""
    e, w = gl_nodes(np.unique(np.concatenate([t, [P.L0, P.U0]])), 30)
    inI = (e >= P.L0) & (e <= P.U0)
    N = basis(e, t)
    lN = (w * P.f_eta(e) * inI * P.g(P.PHI - e)) @ N
    return np.concatenate([lN, P.Bvec])


def part2(t, N=4_000_000, seed=11):
    """Formula-free: empirical Gram on a simulated sample."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 2)); eta = rng.standard_normal(N)
    D = (X[:, 0] + eta > 0)
    SJ = (eta >= P.J0) & (eta <= P.J1)
    ell = ell_vector(t)
    norm2 = 0.0
    Aout = np.zeros(3)
    for d in (0, 1):
        on = SJ & (D == d)
        Bs = basis(eta[on], t, sparse=True)
        Xo = X[on]
        k = Bs.shape[1]
        Q = np.zeros((k + 2, k + 2))
        Q[:k, :k] = (Bs.T @ Bs).toarray()
        Q[:k, k:] = Bs.T @ Xo
        Q[k:, :k] = Q[:k, k:].T
        Q[k:, k:] = Xo.T @ Xo
        Q /= N
        v = np.linalg.solve(Q, (2 * d - 1) * ell)
        norm2 += float(((2 * d - 1) * ell) @ v)
        rK = Bs @ v[:k] + Xo @ v[k:]
        xi = (P.dalpha(eta[on]) if d == 1 else 0.0) + P.db(eta[on])
        Aout += (np.column_stack((np.ones(on.sum()), X[on])) * (xi * rK)[:, None]).sum(0) / N
    return {"n_basis": len(t) - 4, "Er2_K": norm2, "varO_K": P.SIG ** 2 * norm2, "A_out_K": Aout.tolist()}


def part2_quad(t):
    """Same with the Gram by quadrature over eta using the arm moments."""
    e, w = gl_nodes(np.unique(np.concatenate([t, [P.J0, P.J1, P.L0, P.U0]])), 30)
    w = w * P.f_eta(e)
    N = basis(e, t)
    ell = ell_vector(t)
    norm2 = 0.0; Aout = np.zeros(3)
    for d in (0, 1):
        pi = P.pi_d(d, e); m1 = P.mu1_d(d, e); v1 = P.v1_d(d, e)
        k = N.shape[1]
        Q = np.zeros((k + 2, k + 2))
        Q[:k, :k] = N.T @ ((w * pi)[:, None] * N)
        Q[:k, k] = N.T @ (w * pi * m1)
        Q[k, :k] = Q[:k, k]
        Q[k, k] = np.sum(w * pi * (v1 + m1 ** 2))
        Q[k + 1, k + 1] = np.sum(w * pi)
        v = np.linalg.solve(Q, (2 * d - 1) * ell)
        norm2 += float(((2 * d - 1) * ell) @ v)
        xi = (P.dalpha(e) if d == 1 else 0.0) + P.db(e)
        # E[S_J 1_d xi X~ z'] v with X~ = (1, X1, X2), z = (N, X1, X2)
        M = np.zeros((3, k + 2))
        M[0, :k] = (w * pi * xi) @ N; M[0, k] = np.sum(w * pi * xi * m1)
        M[1, :k] = (w * pi * xi * m1) @ N; M[1, k] = np.sum(w * pi * xi * (v1 + m1 ** 2))
        M[2, k + 1] = np.sum(w * pi * xi)
        Aout += M @ v
    return {"n_basis": len(t) - 4, "Er2_K": norm2, "varO_K": P.SIG ** 2 * norm2, "A_out_K": Aout.tolist()}


if __name__ == "__main__":
    res = {"closed_form": {"varU": P.varU, "varO": P.varO, "varT": P.varT, "varG": P.varG,
                           "Er2": P.Er2, "A_out": list(P.A_out)}}
    res["part1"] = part1()
    print("part1", json.dumps(res["part1"], indent=1), flush=True)
    res["part1_se"] = part1_se()
    print("part1_se", res["part1_se"], flush=True)
    res["sieve_quad"] = {str(nu): part2_quad(rich_knots(nu)) for nu in (10, 20, 40, 80)}
    print("sieve_quad", json.dumps(res["sieve_quad"], indent=1), flush=True)
    res["sieve_mc"] = part2(rich_knots(20))
    print("sieve_mc", res["sieve_mc"], flush=True)
    with open(os.path.join(OUT, "mc_scores.json"), "w") as fh:
        json.dump(res, fh, indent=1)
