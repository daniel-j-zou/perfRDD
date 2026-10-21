"""First-order variance at the FINITE sieve sizes the simulation actually uses.

For fixed (K, L) the estimator is a parametric M-estimator of the pseudo-true
threshold phi_L (alpha, b are quadratics, so the outcome sieve is exact; the density
sieve gives P_L h).  Its role scores are those of the paper with the representers
replaced by their sieve versions:
  psi_U^L = -F_L(eta) I0,                 F_L = nu' (P_L h)(phi_L - .)
  psi_o^K = -r_{out,K} R_Y,                r_{out,K} = z_K' Q^{-1} l_K   (outcome knots of the script)
  psi_T^L = -(P_L r_T)(T)' X~ - b_l z_{.9} + b_u z_{.1}
  psi_g   = Abar_{KL}' X~ eta,  Abar_{KL} = A_U^L - A_out,K - A_T,L - b_l m_.9 + b_u m_.1
and n Var(rotated) = Var(sum of scores) / H_L^2 (cross-role covariances included).
"""
from __future__ import annotations

import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import optimize
from scipy.interpolate import BSpline
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_pieces as P  # noqa: E402

TS = (-3.0, 3.0)


def knots(n_interior, lo, hi, extra=()):
    inner = list(np.linspace(lo, hi, n_interior + 2)[1:-1]) + list(extra)
    return np.concatenate([[lo] * 4, sorted(inner), [hi] * 4])


def gl(breaks, order=24):
    x, w = leggauss(order)
    br = np.unique(np.asarray(breaks, dtype=float))
    nodes, weights = [], []
    for a, b in zip(br[:-1], br[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (b + a)); weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def bas(x, t, nu=0):
    x = np.asarray(x, dtype=float)
    nb = len(t) - 4
    inside = (x >= t[0]) & (x <= t[-1])
    out = np.zeros((len(x), nb))
    if nu == 0:
        out[inside] = BSpline.design_matrix(x[inside], t, 3).toarray()
    else:
        for j in range(nb):
            c = np.zeros(nb); c[j] = 1
            out[inside, j] = BSpline(t, c, 3).derivative(nu)(x[inside])
    return out


class DensitySieve:
    """P_L h for h = (g, rho_X1, rho_X2) = (1, t, 0) phi(t) on [-3, 3]."""

    def __init__(self, L, tvec=None):
        self.t = knots(L - 4, *TS) if tvec is None else tvec
        x, w = gl(self.t, 24)
        Nx = bas(x, self.t)
        self.G = Nx.T @ (w[:, None] * Nx)
        hx = np.column_stack((P.phi(x), x * P.phi(x), 0 * x))
        self.coef = np.linalg.solve(self.G, Nx.T @ (w[:, None] * hx))   # nb x 3

    def h(self, s, nu=0):
        return bas(s, self.t, nu) @ self.coef

    def project(self, fun, breaks):
        """Coefficients of P_L applied to a vector function fun(t) -> (n, 3)."""
        x, w = gl(np.concatenate([self.t, [b for b in breaks if TS[0] < b < TS[1]]]), 24)
        return np.linalg.solve(self.G, bas(x, self.t).T @ (w[:, None] * fun(x)))


def nu_vec(e):
    e = np.asarray(e, dtype=float)
    return np.column_stack((P.alpha(e) - P.C, np.full_like(e, P.BETA2[0]), np.full_like(e, P.BETA2[1])))


def finite_sieve(K_int, L, exact_density=False, outcome_extra=()):
    ds = None if exact_density else DensitySieve(L)
    hfun = (lambda s, nu=0: np.column_stack(
        (P.phi(s), s * P.phi(s), 0 * s)) if nu == 0 else np.column_stack(
        (P.dphi(s), (1 - s ** 2) * P.phi(s), 0 * s))) if exact_density else ds.h
    eW, wW = gl([P.L0, P.U0], 60)
    fW = P.f_eta(eW)

    def Z(p, nu=0):
        return float(np.sum(wW * fW * np.sum(nu_vec(eW) * hfun(p - eW, nu), axis=1)))

    phiL = optimize.brentq(lambda p: Z(p), P.PHI - 0.2, P.PHI + 0.2, xtol=1e-14)
    HL = -Z(phiL, 1)
    FL = lambda e: np.sum(nu_vec(e) * hfun(phiL - e), axis=1)
    sL = lambda e: (P.dalpha(e) * hfun(phiL - e)[:, 0] - np.sum(nu_vec(e) * hfun(phiL - e, 1), axis=1))
    varU = float(np.sum(wW * fW * FL(eW) ** 2))
    meanU = float(np.sum(wW * fW * FL(eW)))
    E_eta_psiU = -float(np.sum(wW * fW * eW * FL(eW)))
    b_l = float(FL(np.array([P.L0]))[0] * P.f_eta(P.L0))
    b_u = float(FL(np.array([P.U0]))[0] * P.f_eta(P.U0))
    A_U = np.array([float(np.sum(wW * fW * sL(eW))) + b_l - b_u, 0.0, 0.0])

    # ---- outcome sieve (script knots: uniform interior knots on J, clamped cubic)
    tK = knots(K_int, P.J0, P.J1, outcome_extra)
    e, w = gl(np.concatenate([tK, [P.L0, P.U0]]), 30)
    w = w * P.f_eta(e)
    N = bas(e, tK)
    inI = (e >= P.L0) & (e <= P.U0)
    hh = hfun(phiL - e)
    lN = (w * inI * hh[:, 0]) @ N
    BL = np.array([np.sum(w * inI * hh[:, 1]), np.sum(w * inI * hh[:, 2])])
    ell = np.concatenate([lN, BL])
    norm2 = 0.0; A_out = np.zeros(3)
    k = N.shape[1]
    for d in (0, 1):
        pi = P.pi_d(d, e); m1 = P.mu1_d(d, e); v1 = P.v1_d(d, e)
        Q = np.zeros((k + 2, k + 2))
        Q[:k, :k] = N.T @ ((w * pi)[:, None] * N)
        Q[:k, k] = N.T @ (w * pi * m1); Q[k, :k] = Q[:k, k]
        Q[k, k] = np.sum(w * pi * (v1 + m1 ** 2)); Q[k + 1, k + 1] = np.sum(w * pi)
        v = np.linalg.solve(Q, (2 * d - 1) * ell)
        norm2 += float(((2 * d - 1) * ell) @ v)
        xi = (P.dalpha(e) if d == 1 else 0.0) + P.db(e)
        M = np.zeros((3, k + 2))
        M[0, :k] = (w * pi * xi) @ N; M[0, k] = np.sum(w * pi * xi * m1)
        M[1, :k] = (w * pi * xi * m1) @ N; M[1, k] = np.sum(w * pi * xi * (v1 + m1 ** 2))
        M[2, k + 1] = np.sum(w * pi * xi)
        A_out += M @ v
    varO = P.SIG ** 2 * norm2

    # ---- density representer and its projection
    TL, TU = phiL - P.U0, phiL - P.L0

    def rT(t):
        t = np.asarray(t, dtype=float)
        ind = ((t >= TL) & (t <= TU)).astype(float)
        return (P.f_eta(phiL - t) * ind)[:, None] * nu_vec(phiL - t)

    if exact_density:
        rTL = rT
    else:
        cT = ds.project(rT, [TL, TU])
        rTL = lambda t: bas(t, ds.t) @ cT
    # Var psi_T^L over T = X1 ~ N(0,1), X2 ~ N(0,1) independent
    br = [-12, TS[0], TS[1], 12, TL, TU, P.q10, P.q90] + (list(ds.t) if ds else [])
    x, wx = gl(br, 30)
    wx = wx * P.phi(x)
    R = rTL(x)
    A = -(R[:, 0] + R[:, 1] * x) - b_l * P.z_p(x, 0.9, P.q90) + b_u * P.z_p(x, 0.1, P.q10)
    meanT = float(np.sum(wx * A))
    varT = float(np.sum(wx * (A ** 2 + R[:, 2] ** 2))) - meanT ** 2
    # A_T,L = -int h_X~'(t) r_TL(t) dt   (h_X~ exact; r_TL vanishes off [-3,3])
    xt, wt = gl([TS[0], TS[1], TL, TU] + (list(ds.t) if ds else []), 30)
    Rt = rTL(xt)
    A_T = -np.array([np.sum(wt * np.einsum("ij,ij->i", np.stack([P.hXt_prime(s)[j] for s in xt]), Rt))
                     for j in range(3)])
    m90 = np.array([1.0, P.q90, 0.0]); m10 = np.array([1.0, P.q10, 0.0])
    Abar = A_U - A_out - A_T - b_l * m90 + b_u * m10
    varG = float(Abar @ Abar)
    covUG = Abar[0] * E_eta_psiU
    sigma2 = varU + varO + varT + varG + 2 * covUG
    return {"K_interior": K_int, "L": L, "phi_L": phiL, "H_L": HL, "varU": varU, "meanU": meanU,
            "varO": varO, "varT": varT, "meanT": meanT, "varG": varG, "covUG": covUG,
            "Abar": Abar.tolist(), "A_out": A_out.tolist(), "A_T": A_T.tolist(),
            "sigma2": sigma2, "V_rot": sigma2 / HL ** 2, "V_rot_over_limit": sigma2 / HL ** 2 / P.V}


def sizes(n):
    m = 0.25 * n * 0.45994
    K = max(3, int(round(m ** (11 / 60))))
    L = max(8, int(round(2 * (n / 4) ** (11 / 60))))
    return K, L


if __name__ == "__main__":
    rows = []
    print(f"limit V = {P.V:.4f}  (varU {P.varU:.5f} varO {P.varO:.5f} varT {P.varT:.5f} varG {P.varG:.6f})")
    exact = finite_sieve(3, 8, exact_density=True)
    print("exact density, K=3:", {k: exact[k] for k in ("varO", "varT", "varG", "V_rot")})
    for n in (3200, 6400, 12800, 25600, 51200, 102400):
        K, L = sizes(n)
        r = finite_sieve(K, L)
        r["n"] = n
        rows.append(r)
        print(f"n={n:6d} K_int={K} L={L}: phi_L={r['phi_L']:.5f} H_L={r['H_L']:.5f} varU={r['varU']:.5f} "
              f"varO={r['varO']:.5f} varT={r['varT']:.5f} varG={r['varG']:.6f} covUG={r['covUG']:.2e} "
              f"V={r['V_rot']:.3f} ratio={r['V_rot_over_limit']:.3f}", flush=True)
    grid = []
    for L in (8, 9, 10, 12, 16, 24, 40, 80):
        for K in (3, 4, 10):
            r = finite_sieve(K, L)
            grid.append(r)
            print(f"  K_int={K:2d} L={L:3d}: varO={r['varO']:.5f} varT={r['varT']:.5f} varG={r['varG']:.6f} "
                  f"H_L={r['H_L']:.5f} V={r['V_rot']:.3f}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sieve_variance.json"), "w") as fh:
        json.dump({"by_n": rows, "grid": grid, "exact_density_K3": exact}, fh, indent=1)
