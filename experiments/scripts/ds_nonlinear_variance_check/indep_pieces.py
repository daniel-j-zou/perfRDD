"""Independent recomputation of phi*, H and the four role-score variances for the
"nonlinear" design of ds_four_role_coverage.py, from the formulas of
differing_slopes.tex (Section 4.4, Appendix 2).  Does not import the project code.

Design: X ~ N(0, I_2), T = X1, eta ~ N(0,1), Q = T + eta, D = 1{Q > 0},
alpha(e) = 0.9 e + 0.35 e^2, b(e) = -0.3 + 0.6 e + 0.5 e^2, beta2 = (0.8, 0.25),
sigma_d = 0.5, c = 0.25, eps = 0.1, J = [-1.75, 1.75].
"""
from __future__ import annotations

import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate, optimize
from scipy.stats import norm

OUT = os.path.dirname(os.path.abspath(__file__))
QK = dict(epsabs=1e-14, epsrel=1e-13, limit=400)

C = 0.25
EPS = 0.10
BETA2 = np.array([0.8, 0.25])
SIG = 0.5
J0, J1 = -1.75, 1.75
q90 = norm.ppf(0.9)
q10 = norm.ppf(0.1)
L0, U0 = -q90, -q10          # phi0 - Q_{1-eps}, phi0 - Q_eps


def alpha(e): return 0.9 * e + 0.35 * e ** 2
def dalpha(e): return 0.9 + 0.7 * e
def bfun(e): return -0.3 + 0.6 * e + 0.5 * e ** 2
def db(e): return 0.6 + 1.0 * e
def phi(t): return np.exp(-0.5 * np.asarray(t) ** 2) / np.sqrt(2 * np.pi)
def dphi(t): return -np.asarray(t) * phi(t)
f_eta = phi
g = phi
# rho_X(t) = E[X | T=t] g(t) = (t, 0) phi(t); beta2' rho_X = 0.8 t phi(t)
def b2rho(t): return BETA2[0] * t * phi(t)
def db2rho(t): return BETA2[0] * (1 - t ** 2) * phi(t)


def quad(fun, a, b, points=None):
    pts = None if points is None else [p for p in points if a < p < b]
    val, err = integrate.quad(fun, a, b, points=pts or None, **QK)
    return val, err


# ---------------------------------------------------------------- target, curvature
def F(e, p):
    return (alpha(e) - C) * g(p - e) + b2rho(p - e)


def dF_dphi(e, p):
    return (alpha(e) - C) * dphi(p - e) + db2rho(p - e)


def Zfun(p):
    return quad(lambda e: F(e, p) * f_eta(e), L0, U0)[0]


def dZfun(p):
    return quad(lambda e: dF_dphi(e, p) * f_eta(e), L0, U0)[0]


grid = np.linspace(-1.5, 1.5, 301)
zv = np.array([Zfun(x) for x in grid])
roots = [optimize.brentq(Zfun, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15)
         for i in range(len(grid) - 1) if zv[i] * zv[i + 1] < 0]
assert len(roots) == 1, roots
PHI = roots[0]
H = -dZfun(PHI)                     # U'' = -Z'
h_fd = 1e-4
H_fd = -(Zfun(PHI + h_fd) - Zfun(PHI - h_fd)) / (2 * h_fd)


# ---------------------------------------------------------------- psi_U
def Fs(e):
    return F(e, PHI)


def s_star(e):  # dF*/de
    return dalpha(e) * g(PHI - e) - (alpha(e) - C) * dphi(PHI - e) - db2rho(PHI - e)


meanU, _ = quad(lambda e: Fs(e) * f_eta(e), L0, U0)
EU2, eU2 = quad(lambda e: Fs(e) ** 2 * f_eta(e), L0, U0)
varU = EU2 - meanU ** 2
b_l = Fs(L0) * f_eta(L0)
b_u = Fs(U0) * f_eta(U0)


# ---------------------------------------------------------------- psi_T
TL, TU = PHI - U0, PHI - L0      # support of r_T in t


def w_T(t):
    t = np.asarray(t, dtype=float)
    return np.where((t >= TL) & (t <= TU), f_eta(PHI - t), 0.0)


def z_p(t, p, qp):
    return (p - (np.asarray(t) <= qp)) / g(qp)


def psiT_parts(t):
    """psi_T = A(t) - 0.25 w(t) X2 with X1 = t."""
    w = w_T(t)
    A = (-w * ((alpha(PHI - t) - C) + BETA2[0] * t)
         - b_l * z_p(t, 0.9, q90) + b_u * z_p(t, 0.1, q10))
    return A, BETA2[1] * w


bps = sorted([TL, TU, q10, q90])
meanT = quad(lambda t: psiT_parts(t)[0] * phi(t), -12, 12, bps)[0]
ET2 = quad(lambda t: (psiT_parts(t)[0] ** 2 + psiT_parts(t)[1] ** 2) * phi(t), -12, 12, bps)
varT = ET2[0] - meanT ** 2


# ---------------------------------------------------------------- arm moments (own derivation)
def pi_d(d, e):
    return norm.cdf(e) if d == 1 else norm.cdf(-e)


def mu1_d(d, e):   # E[X1 | eta=e, D=d]; E[X2|...] = 0
    return phi(e) / norm.cdf(e) if d == 1 else -phi(e) / norm.cdf(-e)


def v1_d(d, e):    # Var(X1 | eta=e, D=d); Var(X2|...) = 1
    if d == 1:
        lam = phi(e) / norm.cdf(e)
        return 1 - e * lam - lam ** 2
    lam = phi(e) / norm.cdf(-e)
    return 1 + e * lam - lam ** 2


# numeric check of the arm moments at a few points
chk = []
for e in (-1.6, -0.3, 0.0, 0.9, 1.7):
    for d in (0, 1):
        lo, hi = ((-e, 40) if d == 1 else (-40, -e))
        p = quad(phi, lo, hi)[0]
        m = quad(lambda x: x * phi(x), lo, hi)[0] / p
        v = quad(lambda x: x * x * phi(x), lo, hi)[0] / p - m ** 2
        chk.append(max(abs(p - pi_d(d, e)), abs(m - mu1_d(d, e)), abs(v - v1_d(d, e))))
arm_moment_maxerr = max(chk)

# Gamma_d = E[S_J pi_d Sigma_d]  (diag);  B = E[I0 rho_X(phi*-eta)];  Gm_d = E[I0 g mu_d]
Gam = {}
for d in (0, 1):
    G11 = quad(lambda e: f_eta(e) * pi_d(d, e) * v1_d(d, e), J0, J1)[0]
    G22 = quad(lambda e: f_eta(e) * pi_d(d, e), J0, J1)[0]
    Gam[d] = np.diag([G11, G22])
B1 = quad(lambda e: f_eta(e) * (PHI - e) * g(PHI - e), L0, U0)[0]
Bvec = np.array([B1, 0.0])
omega = {}
for d in (0, 1):
    Gm1 = quad(lambda e: f_eta(e) * g(PHI - e) * mu1_d(d, e), L0, U0)[0]
    omega[d] = (2 * d - 1) * np.linalg.solve(Gam[d], Bvec - np.array([Gm1, 0.0]))

Er2 = 0.0
Er2_parts = {}
for d in (0, 1):
    a = quad(lambda e: f_eta(e) * g(PHI - e) ** 2 / pi_d(d, e), L0, U0)[0]
    bq = float(omega[d] @ Gam[d] @ omega[d])
    Er2_parts[d] = (a, bq)
    Er2 += a + bq
varO = SIG ** 2 * Er2


# ---------------------------------------------------------------- loadings
def xi_d(d, e):
    return db(e) + (dalpha(e) if d == 1 else 0.0)


A_U = np.array([quad(lambda e: s_star(e) * f_eta(e), L0, U0)[0] + b_l - b_u, 0.0, 0.0])

# A_out = E[xi X~ r_out]; intercept = sum_d (2d-1) int_J f xi_d I0 g; X block below
A_out = np.zeros(3)
A_out[0] = sum((2 * d - 1) * quad(lambda e: f_eta(e) * xi_d(d, e) * g(PHI - e), L0, U0)[0]
               for d in (0, 1))
for d in (0, 1):
    # X1: int_J f xi_d [(2d-1) I0 g mu1_d + pi_d v1_d omega_d1];  X2: int_J f xi_d pi_d * 1 * omega_d2
    A_out[1] += (2 * d - 1) * quad(lambda e: f_eta(e) * xi_d(d, e) * g(PHI - e) * mu1_d(d, e), L0, U0)[0]
    A_out[1] += omega[d][0] * quad(lambda e: f_eta(e) * xi_d(d, e) * pi_d(d, e) * v1_d(d, e), J0, J1)[0]
    A_out[2] += omega[d][1] * quad(lambda e: f_eta(e) * xi_d(d, e) * pi_d(d, e), J0, J1)[0]
A_out_intercept_check = quad(lambda e: f_eta(e) * dalpha(e) * g(PHI - e), L0, U0)[0]


# A_T = -int h_X~'(t) r_T(t) dt with h_X~ = E[X~X~'|T=t] g(t), X~ = (1, t, X2)
def r_T(t):
    w = w_T(t)
    return np.array([w * (alpha(PHI - t) - C), w * BETA2[0], w * BETA2[1]])


def hXt_prime(t):
    p = phi(t)
    return np.array([[-t * p, (1 - t ** 2) * p, 0.0],
                     [(1 - t ** 2) * p, (2 * t - t ** 3) * p, 0.0],
                     [0.0, 0.0, -t * p]])


A_T = np.array([-quad(lambda t: float(hXt_prime(t)[j] @ r_T(t)), TL, TU)[0] for j in range(3)])
m90 = np.array([1.0, q90, 0.0])
m10 = np.array([1.0, q10, 0.0])
Abar = A_U - A_out - A_T - b_l * m90 + b_u * m10
varG = float(Abar @ Abar)      # Sigma_X~ = I_3, E eta^2 = 1

sigma2 = varU + varO + varT + varG
V = sigma2 / H ** 2


# ---------------------------------------------------------------- FD check of the non-outcome loadings
def Psi_partial(delta):
    """Z(phi*) with h, the quantiles and the law of eta_hat at gamma + delta; nu at truth."""
    d0, d1, d2 = delta
    s2 = (1 + d1) ** 2 + d2 ** 2
    s = np.sqrt(s2)
    se = np.sqrt(1 + d1 ** 2 + d2 ** 2)         # sd of eta_delta = eta - d0 - d1 X1 - d2 X2
    l = -(d0 + s * q90)
    u = -(d0 + s * q10)

    def h_delta(t):
        dens = phi((t - d0) / s) / s
        return dens, (1 + d1) * (t - d0) / s2 * dens, d2 * (t - d0) / s2 * dens

    def integrand(v):
        hg, h1, h2 = h_delta(PHI - v)
        Fv = (alpha(v) - C) * hg + BETA2[0] * h1 + BETA2[1] * h2
        return Fv * phi((v + d0) / se) / se

    return quad(integrand, l, u)[0]


hh = 1e-4
grad_partial = np.zeros(3)
for j in range(3):
    e = np.zeros(3); e[j] = hh
    grad_partial[j] = (Psi_partial(e) - Psi_partial(-e)) / (2 * hh)
# -grad = A_U - A_T - b_l m90 + b_u m10
nonout_formula = A_U - A_T - b_l * m90 + b_u * m10
nonout_fd = -grad_partial

res = {
    "phi_star": PHI, "H": H, "H_fd": H_fd, "l0": L0, "u0": U0,
    "meanU": meanU, "varU": varU, "varU_quad_err": eU2,
    "b_l": b_l, "b_u": b_u,
    "meanT": meanT, "varT": varT, "varT_quad_err": ET2[1],
    "arm_moment_maxerr": arm_moment_maxerr,
    "Gamma": {d: Gam[d].tolist() for d in Gam}, "B": Bvec.tolist(),
    "omega": {d: omega[d].tolist() for d in omega}, "Er2_parts": Er2_parts,
    "Er2": Er2, "varO": varO,
    "A_U": A_U.tolist(), "A_out": A_out.tolist(), "A_out_intercept_check": A_out_intercept_check,
    "A_T": A_T.tolist(), "Abar": Abar.tolist(), "varG": varG,
    "nonout_formula": nonout_formula.tolist(), "nonout_fd": nonout_fd.tolist(),
    "sigma2": sigma2, "V": V,
}

if __name__ == "__main__":
    for k, v in res.items():
        print(f"{k:24s} {v}")
    with open(os.path.join(OUT, "indep_pieces.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
