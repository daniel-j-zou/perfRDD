"""Location-scale threshold model: DGP, population target and four-role estimators.

Independent numerical review script (scratch only).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats, optimize
from scipy.interpolate import BSpline

# ---------------------------------------------------------------- configuration
GAMMA = np.array([0.0, 1.0, 0.5])          # T = g0 + g1 X1 + g2 X2
DELTA = np.array([0.8, 0.4])               # s = exp(0.5 (d1 X1 + d2 X2))
PHI0 = 1.0
BETA1 = np.array([0.5, 0.2])
BETA2 = np.array([0.3, 0.0])
SIG_Y = 0.5
EPS = 0.05
PHI_LO, PHI_HI = PHI0 - 1.2, PHI0 + 1.2
TRIM_J = 0.0025
A_INT = -0.35                               # alpha intercept tuned from -0.15 (see explore_alpha.py)


def alpha_fn(e, a_int=None):
    a_int = A_INT if a_int is None else a_int
    return 0.4 * e + a_int + 0.1 * np.tanh(2.0 * e)


def b_fn(e):
    return np.sin(e)


# ---------------------------------------------------------------- xi distributions
T5_SCALE = np.sqrt(3.0 / 5.0)              # unit variance t(5)
LOGIS_SCALE = np.sqrt(3.0) / np.pi          # unit variance logistic


def draw_xi(dist, n, rng):
    if dist == "normal":
        return rng.standard_normal(n)
    if dist == "t5":
        return rng.standard_t(5, n) * T5_SCALE
    if dist == "logistic":
        return rng.logistic(0.0, 1.0, n) * LOGIS_SCALE
    raise ValueError(dist)


def xi_frozen(dist):
    if dist == "normal":
        return stats.norm()
    if dist == "t5":
        return stats.t(5, scale=T5_SCALE)
    if dist == "logistic":
        return stats.logistic(scale=LOGIS_SCALE)
    raise ValueError(dist)


def T_true(X1, X2):
    return GAMMA[0] + GAMMA[1] * X1 + GAMMA[2] * X2


def s_true(X1, X2):
    return np.exp(0.5 * (DELTA[0] * X1 + DELTA[1] * X2))


def simulate(n, rng, dist="normal", a_int=None):
    X1 = rng.standard_normal(n)
    X2 = rng.integers(0, 2, n) - 0.5
    T = T_true(X1, X2)
    s = s_true(X1, X2)
    xi = draw_xi(dist, n, rng)
    Q = T + s * xi
    D = (Q > PHI0).astype(float)
    X = np.column_stack([X1, X2])
    Y = (b_fn(xi) + D * alpha_fn(xi, a_int) + X @ BETA1 + D * (X @ BETA2)
         + SIG_Y * rng.standard_normal(n))
    return dict(X=X, Q=Q, D=D, Y=Y, xi=xi)


# ---------------------------------------------------------------- population quantities
def R_cdf(r):
    """P(R <= r), R = (phi0 - T)/s, exact via root finding in X1 per X2 value."""
    tot = 0.0
    for x2 in (-0.5, 0.5):
        g = lambda x1: (PHI0 - T_true(x1, x2)) / s_true(x1, x2) - r
        # g(x1)+r is decreasing then increasing (minimum at x1m), -> 0^- as x1 -> inf
        # d/dx1 [(c - x1) e^{-a x1}] = 0  ->  x1m = c + 1/a, with c, a below
        a = 0.5 * DELTA[0]
        c = (PHI0 - GAMMA[0] - GAMMA[2] * x2) / GAMMA[1]
        x1m = c + 1.0 / a
        gmin = g(x1m)
        if gmin >= 0:  # r below minimum of R
            continue
        lo = optimize.brentq(g, -60.0, x1m)
        if r < 0:
            hi = optimize.brentq(g, x1m, 200.0)
            tot += 0.5 * (stats.norm.cdf(hi) - stats.norm.cdf(lo))
        else:
            tot += 0.5 * stats.norm.sf(lo)
    return tot


def window_pop(eps=EPS):
    l0 = optimize.brentq(lambda r: R_cdf(r) - eps, -0.9, 20.0, xtol=1e-12)
    u0 = optimize.brentq(lambda r: R_cdf(r) - (1 - eps), -0.9, 50.0, xtol=1e-12)
    return l0, u0


class XQuad:
    """Deterministic quadrature over X: fine X1 grid times the two X2 values."""

    def __init__(self, m=40001, lim=9.0):
        x1 = np.linspace(-lim, lim, m)
        w1 = stats.norm.pdf(x1)
        w1 /= w1.sum()
        self.X1 = np.concatenate([x1, x1])
        self.X2 = np.concatenate([np.full(m, -0.5), np.full(m, 0.5)])
        self.w = np.concatenate([w1, w1]) * 0.5
        self.T = T_true(self.X1, self.X2)
        self.s = s_true(self.X1, self.X2)


class XiCum:
    """Cumulative integrals F(e) and A(e) = int_{-inf}^e afun(u) f(u) du on a fine grid."""

    def __init__(self, dist, afun, lim=60.0, m=1_200_001):
        fz = xi_frozen(dist)
        e = np.linspace(-lim, lim, m)
        f = fz.pdf(e)
        af = afun(e) * f
        h = e[1] - e[0]
        self.e = e
        self.F = fz.cdf(e)
        self.A = np.concatenate([[0.0], np.cumsum(0.5 * (af[1:] + af[:-1]) * h)])
        self.fz = fz

    def Fi(self, x):
        return np.interp(x, self.e, self.F)

    def Ai(self, x):
        return np.interp(x, self.e, self.A)


def U_pop(phi, xq, xc, l0, u0, beta2=BETA2):
    """U(phi) = E[(alpha(xi) + X'beta2) 1(xi > (phi-T)/s) 1(l0 <= xi <= u0)]."""
    c = (phi - xq.T) / xq.s
    lo = np.minimum(np.maximum(c, l0), u0)
    xb = beta2[0] * xq.X1 + beta2[1] * xq.X2
    val = (xc.Ai(u0) - xc.Ai(lo)) + xb * (xc.Fi(u0) - xc.Fi(lo))
    return float(np.sum(xq.w * val))


def argmax_pop(xq, xc, l0, u0, beta2=BETA2, grid=241):
    ph = np.linspace(PHI_LO, PHI_HI, grid)
    U = np.array([U_pop(p, xq, xc, l0, u0, beta2) for p in ph])
    k = int(np.argmax(U))
    a, b = ph[max(k - 1, 0)], ph[min(k + 1, grid - 1)]
    res = optimize.minimize_scalar(lambda p: -U_pop(p, xq, xc, l0, u0, beta2),
                                   bounds=(a, b), method="bounded",
                                   options=dict(xatol=1e-9))
    return res.x, -res.fun, ph, U


# ---------------------------------------------------------------- estimator pieces
def _Z(X):
    return np.column_stack([np.ones(len(X)), X])


def fit_index(Xg, Qg, scale=True):
    Z = _Z(Xg)
    gam = np.linalg.lstsq(Z, Qg, rcond=None)[0]
    eta = Qg - Z @ gam
    if not scale:
        return gam, None, 1.0
    dlt = np.linalg.lstsq(Z, np.log(eta ** 2 + 1e-8), rcond=None)[0]
    med = np.median(np.exp(0.5 * Z @ dlt))
    return gam, dlt, med


def eval_index(X, idx):
    gam, dlt, med = idx
    Z = _Z(X)
    T = Z @ gam
    s = np.ones(len(X)) if dlt is None else np.exp(0.5 * Z @ dlt) / med
    return T, s


def _arm_basis(e):
    lo, hi = np.quantile(e, [TRIM_J, 1 - TRIM_J])
    keep = (e >= lo) & (e <= hi)
    neff = int(keep.sum())
    K = max(1, int(round(neff ** 0.2)))
    inner = np.quantile(e[keep], np.linspace(0, 1, K + 2)[1:-1])
    t = np.r_[[lo] * 4, inner, [hi] * 4]
    return keep, t, lo, hi


def _dm(x, t, lo, hi):
    return BSpline.design_matrix(np.clip(x, lo, hi), t, 3).toarray()


def fit_outcome(xi, X, D, Y):
    """Arm-wise cubic B-splines in xi-hat + X + D X on J (0.25% arm tails trimmed)."""
    arms = {}
    rows, blocks = [], []
    for d in (0, 1):
        m = D == d
        keep, t, lo, hi = _arm_basis(xi[m])
        idx = np.flatnonzero(m)[keep]
        arms[d] = (t, lo, hi)
        rows.append(idx)
        blocks.append(_dm(xi[idx], t, lo, hi))
    p0, p1 = blocks[0].shape[1], blocks[1].shape[1]
    n0, n1 = len(rows[0]), len(rows[1])
    idx = np.r_[rows[0], rows[1]]
    Bfull = np.zeros((n0 + n1, p0 + p1))
    Bfull[:n0, :p0] = blocks[0]
    Bfull[n0:, p0:] = blocks[1]
    Xs, Ds = X[idx], D[idx]
    W = np.column_stack([Bfull, Xs, Ds[:, None] * Xs])
    coef = np.linalg.lstsq(W, Y[idx], rcond=None)[0]
    c0, c1 = coef[:p0], coef[p0:p0 + p1]
    beta2 = coef[p0 + p1 + 2:]

    def alpha_hat(e):
        t1, lo1, hi1 = arms[1]
        t0, lo0, hi0 = arms[0]
        return _dm(e, t1, lo1, hi1) @ c1 - _dm(e, t0, lo0, hi0) @ c0

    return alpha_hat, beta2, arms


def make_Uhat(xi_u, a_u, T_j, s_j, Xb_j):
    """Return a vectorised U-hat(phi) using sorting + suffix sums."""
    o = np.argsort(xi_u)
    e = xi_u[o]
    a = a_u[o]
    m = len(e)
    SA = np.r_[np.cumsum(a[::-1])[::-1], 0.0]       # SA[k] = sum_{i>=k} a_i
    nI = len(T_j)

    def U(phis):
        phis = np.atleast_1d(phis)
        out = np.empty(len(phis))
        for r, p in enumerate(phis):
            c = (p - T_j) / s_j
            k = np.searchsorted(e, c, side="right")   # count e <= c
            out[r] = (SA[k].sum() + np.dot(Xb_j, m - k)) / nI
        return out
    return U


def argmax_step(U, grid=121, refine=((21, 1.0), (21, 1.0))):
    ph = np.linspace(PHI_LO, PHI_HI, grid)
    h = ph[1] - ph[0]
    v = U(ph)
    best = ph[int(np.argmax(v))]
    for (m, frac) in refine:
        half = h * frac
        a, b = max(PHI_LO, best - half), min(PHI_HI, best + half)
        g = np.linspace(a, b, m)
        vv = U(g)
        best = g[int(np.argmax(vv))]
        h = g[1] - g[0]
    return best


def four_role(data, scale=True, eps=EPS, return_diag=False):
    """Average of the four cyclic role rotations (gamma, outcome, index, utility)."""
    X, Q, D, Y = data["X"], data["Q"], data["D"], data["Y"]
    n = len(Q)
    folds = data.get("folds")
    if folds is None:
        folds = np.arange(n) % 4
    ests, diags = [], []
    for r in range(4):
        fg, fo, fi, fu = [(folds == ((r + k) % 4)) for k in range(4)]
        idx = fit_index(X[fg], Q[fg], scale=scale)
        # outcome fold
        To, so = eval_index(X[fo], idx)
        xio = (Q[fo] - To) / so
        alpha_hat, b2, arms = fit_outcome(xio, X[fo], D[fo], Y[fo])
        # index fold
        Tj, sj = eval_index(X[fi], idx)
        Rj = (PHI0 - Tj) / sj
        lh, uh = np.quantile(Rj, [eps, 1 - eps])
        # utility fold
        Tu, su = eval_index(X[fu], idx)
        xiu = (Q[fu] - Tu) / su
        w = (xiu >= lh) & (xiu <= uh)
        e = xiu[w]
        a = alpha_hat(e)
        Uh = make_Uhat(e, a, Tj, sj, X[fi] @ b2)
        ph = argmax_step(Uh)
        ests.append(ph)
        if return_diag:
            diags.append(dict(l=lh, u=uh, beta2=b2,
                              hi0=arms[0][2], lo1=arms[1][1],
                              frac_above_J0=float(np.mean(e > arms[0][2])),
                              frac_below_J1=float(np.mean(e < arms[1][1]))))
    ests = np.array(ests)
    if return_diag:
        return ests.mean(), ests, diags
    return ests.mean(), ests
