"""Check the root-n limit of the GLM threshold estimator by simulation (Claude, 2026-11-18).

manuscript/binary_outcome_vlbw.tex ("GLM Setting"), Section 3: for one fixed assignment of the
four roles, sqrt(n)(phi_hat - phi*) -> N(0, 4 sigma^2/H^2), and the average of the four
rotations and the maximizer of the pooled criterion -> N(0, sigma^2/H^2), where sigma^2 is the
sum of the variances of the four role scores psi_U, psi_beta, psi_T, psi_gamma and H = U''(phi*).

The design is a logit model with two Gaussian covariates. The script
  1. computes phi*, H and the four score variances from the formulas of the note, by quadrature
     (the representer r_out on a rich piecewise-cubic basis with breaks at the window ends);
  2. checks that the intercept coordinate of the gradient A-bar is zero;
  3. checks the generated-regressor gradient A_out, the new term of the GLM fit, against a
     finite difference of the population fit when the residual is formed with gamma + delta;
  4. runs the estimator of the note (empirical tails, Chebyshev split with M = 12, logistic
     spline fit with no penalty, window from the density fold, no caps) on R simulated samples
     and compares the Monte Carlo spread with the formula.

    python -m experiments.scripts.binary_glm_clt_check --n 40000 --reps 1000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from numpy.polynomial import chebyshev as Cheb
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "binary_glm_clt_check"

# ----------------------------------------------------------------------------- design
GX = np.array([0.6, 0.4])          # T = gamma' X~, intercept 0
SD_T = float(np.linalg.norm(GX))
UPERP = np.array([-GX[1], GX[0]]) / SD_T
S_ETA = 0.7
PHI0 = 0.0
EPS = 0.1
B1 = np.array([0.5, -0.4])         # baseline slope
B2 = 0.6                           # effect slope on X_e = X1
PHI_TARGET = 0.3                   # the cost is set so that phi* is here
M_SPLIT = 12


def b(e):
    return 0.8 + 0.6 * np.tanh(e)


def db(e):
    return 0.6 / np.cosh(e) ** 2


def alpha(e):
    return -0.2 + 2.5 * e + 0.2 * np.sin(2 * e)


def dalpha(e):
    return 2.5 + 0.4 * np.cos(2 * e)


def expit(u):
    return 1.0 / (1.0 + np.exp(-u))


def dexpit(u):
    m = expit(u)
    return m * (1 - m)


def f_eta(e):
    return norm.pdf(e, scale=S_ETA)


def g(t):
    return norm.pdf(t, scale=SD_T)


Q_EPS, Q_1MEPS = norm.ppf(EPS, scale=SD_T), norm.ppf(1 - EPS, scale=SD_T)
L0, U0 = PHI0 - Q_1MEPS, PHI0 - Q_EPS
J = (L0 - 0.7 * S_ETA, U0 + 0.7 * S_ETA)

ZN, ZW = hermegauss(40)
ZW = ZW / np.sqrt(2 * np.pi)


def x_given_t(t, z):
    """X given T = t is GX t / |GX|^2 + z UPERP, z ~ N(0, 1). Shapes broadcast; last axis 2."""
    t = np.asarray(t, float)[..., None]
    z = np.asarray(z, float)[..., None]
    return GX * t / SD_T ** 2 + z * UPERP


def lin_pred(X, e):
    a0 = b(e) + X @ B1
    a1 = a0 + alpha(e) + B2 * X[..., 0]
    return a0, a1


def k_cond(e, t, c, weight="W", out="1"):
    """E[ w(X, e) * m(X) | T = t] g(t), w = W-bar - c (or mu'(a_d)), m = 1 or X~."""
    e = np.asarray(e, float)
    t = np.asarray(t, float)
    X = x_given_t(t[..., None], ZN)                 # (..., nz, 2)
    a0, a1 = lin_pred(X, e[..., None])
    if weight == "W":
        w = expit(a1) - expit(a0) - c
    elif weight == "mu1":
        w = dexpit(a1)
    elif weight == "mu0":
        w = dexpit(a0)
    else:
        raise ValueError(weight)
    if out == "1":
        val = np.sum(ZW * w, axis=-1)
    else:                                            # X~ = (1, X1, X2)
        Xt = np.concatenate([np.ones(X.shape[:-1] + (1,)), X], axis=-1)
        val = np.sum((ZW * w)[..., None] * Xt, axis=-2)
        return val * g(t)[..., None]
    return val * g(t)


EN, EW = leggauss(200)


def e_nodes(lo, hi, n=200):
    x, w = (EN, EW) if n == 200 else leggauss(n)
    return lo + (x + 1) * (hi - lo) / 2, w * (hi - lo) / 2


def upsilon(phi, c):
    """Upsilon(phi) = -U'(phi) = int_{l0}^{u0} f_eta(e) k1(e, phi - e) de."""
    e, w = e_nodes(L0, U0)
    return float(np.sum(w * f_eta(e) * k_cond(e, phi - e, c)))


def p_eps(phi):
    e, w = e_nodes(L0, U0)
    return float(np.sum(w * f_eta(e) * g(phi - e)))


# ----------------------------------------------------------------------------- population

def piece_basis(e, kn=6):
    """Cubic B-splines on [J0, L0], [L0, U0], [U0, J1] separately (breaks at the window ends)."""
    e = np.asarray(e, float)
    cols = []
    for lo, hi in ((J[0], L0), (L0, U0), (U0, J[1])):
        interior = np.linspace(lo, hi, kn + 2)[1:-1]
        t = np.concatenate([np.repeat(lo, 4), interior, np.repeat(hi, 4)])
        inside = (e >= lo) & (e < hi) if hi < J[1] else (e >= lo) & (e <= hi)
        B = BSpline.design_matrix(np.clip(e, lo, hi), t, 3).toarray()
        cols.append(B * inside[:, None])
    return np.column_stack(cols)


def population(verbose=True):
    # cost putting phi* at PHI_TARGET
    c = upsilon(PHI_TARGET, 0.0) / p_eps(PHI_TARGET)
    phis = np.linspace(-0.6, 1.2, 37)
    tau = np.array([upsilon(p, c) / p_eps(p) for p in phis])
    phi_star = brentq(lambda p: upsilon(p, c), PHI_TARGET - 0.3, PHI_TARGET + 0.3, xtol=1e-12)
    h = 1e-4
    H = -(upsilon(phi_star + h, c) - upsilon(phi_star - h, c)) / (2 * h)   # U'' = -Upsilon'

    def F(e):
        return k_cond(e, phi_star - e, c)

    b_l = float(F(np.array(L0)) * f_eta(L0))
    b_u = float(F(np.array(U0)) * f_eta(U0))
    e, w = e_nodes(L0, U0)
    var_U = float(np.sum(w * f_eta(e) * F(e) ** 2))

    # psi_T by quadrature over (t, z), t split at the window ends and the quantiles
    brk = sorted([-7 * SD_T, phi_star - U0, phi_star - L0, Q_EPS, Q_1MEPS, 7 * SD_T])
    m1, m2, mm = 0.0, 0.0, 0.0
    for lo, hi in zip(brk[:-1], brk[1:]):
        t, wt = e_nodes(lo, hi, 80)
        X = x_given_t(t[:, None], ZN)
        ee = phi_star - t
        a0, a1 = lin_pred(X, ee[:, None])
        inwin = ((ee >= L0) & (ee <= U0)).astype(float)
        main = -(expit(a1) - expit(a0) - c) * (f_eta(ee) * inwin)[:, None]
        zl = ((1 - EPS) - (t <= Q_1MEPS)) / g(Q_1MEPS)
        zu = (EPS - (t <= Q_EPS)) / g(Q_EPS)
        psi = main - b_l * zl[:, None] + b_u * zu[:, None]
        m1 += float(np.sum(wt * g(t) * np.sum(ZW * psi, axis=1)))
        m2 += float(np.sum(wt * g(t) * np.sum(ZW * psi ** 2, axis=1)))
        mm += float(np.sum(wt * g(t) * np.sum(ZW * main, axis=1)))
    var_T = m2 - m1 ** 2

    # representer r_out on the piecewise basis; Gram matrix by quadrature over (e, t, z)
    pts_e, pts_w = [], []
    for lo, hi in ((J[0], L0), (L0, U0), (U0, J[1])):
        ee, ww = e_nodes(lo, hi, 48)
        pts_e.append(ee); pts_w.append(ww)
    ee = np.concatenate(pts_e); we = np.concatenate(pts_w) * f_eta(ee)
    Be = piece_basis(ee)
    nb = Be.shape[1]
    dim = 2 * nb + 3
    Jm = np.zeros((dim, dim))
    cvec = np.zeros((dim, 3))          # E[S mu' z z'_D X~]
    for i in range(len(ee)):
        e1 = ee[i]
        for lo, hi in ((-7 * SD_T, PHI0 - e1), (PHI0 - e1, 7 * SD_T)):
            t, wt = e_nodes(lo, hi, 48)
            X = x_given_t(t[:, None], ZN).reshape(-1, 2)
            wts = (wt * g(t))[:, None] * ZW[None, :]
            wts = wts.reshape(-1) * we[i]
            d = 1.0 if lo >= PHI0 - e1 - 1e-12 else 0.0
            a0, a1 = lin_pred(X, e1)
            a = a1 if d else a0
            mp = dexpit(a)
            Z = np.zeros((len(X), dim))
            if d:
                Z[:, nb:2 * nb] = Be[i]
            else:
                Z[:, :nb] = Be[i]
            Z[:, 2 * nb:2 * nb + 2] = X
            Z[:, 2 * nb + 2] = d * X[:, 0]
            ww = wts * mp
            Jm += Z.T @ (Z * ww[:, None])
            zprime = db(e1) + d * dalpha(e1)
            Xt = np.column_stack([np.ones(len(X)), X])
            cvec += Z.T @ (Xt * (ww * zprime)[:, None])
    # ell_out on the basis: integrate over the window, X given T = phi* - e
    e, w = e_nodes(L0, U0)
    t = phi_star - e
    wg = w * f_eta(e) * g(t)
    X = x_given_t(t[:, None], ZN)
    a0, a1 = lin_pred(X, e[:, None])
    mu1, mu0 = dexpit(a1), dexpit(a0)
    Bw = piece_basis(e)
    ell = np.zeros(dim)
    ell[:nb] = -Bw.T @ (wg * np.sum(ZW * mu0, axis=1))
    ell[nb:2 * nb] = Bw.T @ (wg * np.sum(ZW * mu1, axis=1))
    for m in range(2):
        ell[2 * nb + m] = np.sum(wg * np.sum(ZW * (mu1 - mu0) * X[..., m], axis=1))
    ell[2 * nb + 2] = np.sum(wg * np.sum(ZW * mu1 * X[..., 0], axis=1))
    coef = np.linalg.solve(Jm, ell)
    var_beta = float(ell @ coef)
    A_out = cvec.T @ coef                                       # (3,)
    ell_bprime = float(np.sum(wg * np.sum(ZW * (mu1 * (db(e) + dalpha(e))[:, None]
                                                - mu0 * db(e)[:, None]), axis=1)))

    # tails gradient G_T = E[I0 d/dt k_X~(eta, t)] at t = phi* - eta
    hh = 1e-5
    dk = (k_cond(e, t + hh, c, out="X") - k_cond(e, t - hh, c, out="X")) / (2 * hh)
    G_T = (w * f_eta(e)) @ dk                                    # (3,)
    # evaluation gradient (intercept only) and the window
    dF = (F(e + hh) - F(e - hh)) / (2 * hh)
    EFp = float(np.sum(w * f_eta(e) * dF))
    A_U = np.array([EFp + b_l - b_u, 0.0, 0.0])
    m_lo = np.concatenate([[1.0], GX * Q_1MEPS / SD_T ** 2])     # m_{1-eps} = E[X~ | T = q]
    m_hi = np.concatenate([[1.0], GX * Q_EPS / SD_T ** 2])       # m_eps
    Abar = A_U - A_out + G_T - b_l * m_lo + b_u * m_hi
    var_gamma = float(S_ETA ** 2 * Abar @ Abar)                  # Sigma_X~ = I, X indep eta
    sigma2 = var_U + var_beta + var_T + var_gamma
    pop = dict(c=c, phi_star=phi_star, H=H, p_eps=p_eps(phi_star),
               tau_grid=dict(phi=phis.tolist(), tau=tau.tolist()),
               window=[L0, U0], J=list(J), b_l=b_l, b_u=b_u,
               var_U=var_U, var_T=var_T, mean_psi_T=m1, mean_main_T=mm,
               var_beta=var_beta, var_gamma=var_gamma, sigma2=sigma2,
               A_U=A_U.tolist(), A_out=A_out.tolist(), G_T=G_T.tolist(),
               m_lo=m_lo.tolist(), m_hi=m_hi.tolist(), Abar=Abar.tolist(),
               intercept_check=dict(A_out_0=float(A_out[0]), ell_out_bprime=ell_bprime,
                                    Abar_0=float(Abar[0]),
                                    terms=[float(A_U[0]), float(-A_out[0]), float(G_T[0]),
                                           float(-b_l), float(b_u)]),
               sd_rootn_fixed=2 * np.sqrt(sigma2) / abs(H),
               sd_rootn_rotated=np.sqrt(sigma2) / abs(H),
               _basis=dict(nb=nb, coef=coef.tolist()))
    if verbose:
        print(f"c = {c:.5f}, phi* = {phi_star:.6f}, H = {H:.5f}, window = [{L0:.4f}, {U0:.4f}], "
              f"J = [{J[0]:.4f}, {J[1]:.4f}]")
        print(f"tau monotone on grid: {bool(np.all(np.diff(tau) > 0))}")
        print(f"Var psi_U = {var_U:.6f}, psi_beta = {var_beta:.6f}, psi_T = {var_T:.6f} "
              f"(mean {m1:.2e}, main part {mm:.2e}), psi_gamma = {var_gamma:.6f}")
        print(f"A_out = {A_out}, G_T = {G_T}, A_U = {A_U}")
        print(f"intercept: e1'A_out = {A_out[0]:.6f} vs ell_out(b', alpha', 0) = {ell_bprime:.6f}; "
              f"e1'Abar = {Abar[0]:.2e}; Abar = {Abar}")
        print(f"sigma^2 = {sigma2:.6f}; sd of sqrt(n)(phi_hat - phi*): fixed "
              f"{pop['sd_rootn_fixed']:.4f}, rotated {pop['sd_rootn_rotated']:.4f}")
    return pop


# ----------------------------------------------------------------------------- A_out by finite difference

def population_fit(eta_s, D, X, mu_star, w_obs, kn=6, iters=30):
    """Maximize sum w_obs {mu* a - log(1 + e^a)} over the piecewise basis in the fitted residual
    (units with eta_s in J only): the population logistic fit when Y is replaced by its mean."""
    keep = (eta_s >= J[0]) & (eta_s <= J[1])
    e, d, x, m, ww = eta_s[keep], D[keep], X[keep], mu_star[keep], w_obs[keep]
    B = piece_basis(e, kn)
    A = np.column_stack([(1 - d)[:, None] * B, d[:, None] * B, x, d * x[:, 0]])
    theta = np.zeros(A.shape[1])
    for _ in range(iters):
        u = A @ theta
        grad = A.T @ (ww * (m - expit(u)))
        Hs = (A * (ww * dexpit(u))[:, None]).T @ A
        step = np.linalg.solve(Hs, grad)
        theta += step
        if np.max(np.abs(step)) < 1e-11:
            break
    nb = B.shape[1]
    return theta[:nb], theta[nb:2 * nb], theta[2 * nb:2 * nb + 2], theta[2 * nb + 2]


def upsilon_fit(fit, phi_star, c, kn=6):
    """Upsilon(phi*) with the linear predictors of a fit in place of the truth."""
    w0, w1, beta1, beta2 = fit
    e, w = e_nodes(L0, U0)
    t = phi_star - e
    X = x_given_t(t[:, None], ZN)
    B = piece_basis(e, kn)
    z0, z1 = B @ w0, B @ w1
    a0 = z0[:, None] + X @ beta1
    a1 = z1[:, None] + X @ beta1 + beta2 * X[..., 0]
    k1 = np.sum(ZW * (expit(a1) - expit(a0) - c), axis=1) * g(t)
    return float(np.sum(w * f_eta(e) * k1))


def check_A_out(pop, N=1_000_000, h=0.02, seed=7):
    """d/d delta of Upsilon(phi*) through the outcome fit alone, delta in the slope coordinates."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 2))
    eta = S_ETA * rng.standard_normal(N)
    T = X @ GX
    D = (T + eta > PHI0).astype(float)
    a0, a1 = lin_pred(X, eta)
    mu_star = expit(np.where(D > 0, a1, a0))
    w_obs = np.ones(N) / N
    Xt = np.column_stack([np.ones(N), X])
    out = {}
    base = population_fit(eta, D, X, mu_star, w_obs)
    out["upsilon_fit_at_truth"] = upsilon_fit(base, pop["phi_star"], pop["c"])
    for j in range(3):
        vals = []
        for s in (+1, -1):
            delta = np.zeros(3); delta[j] = s * h
            eta_s = eta - Xt @ delta
            fit = population_fit(eta_s, D, X, mu_star, w_obs)
            vals.append(upsilon_fit(fit, pop["phi_star"], pop["c"]))
        out[f"dUpsilon_d{j}"] = (vals[0] - vals[1]) / (2 * h)
    out["A_out_formula"] = pop["A_out"]
    print("A_out finite difference:", [round(out[f'dUpsilon_d{j}'], 5) for j in range(3)],
          "formula:", np.round(pop["A_out"], 5), "Upsilon(fit at truth) =",
          f"{out['upsilon_fit_at_truth']:.2e}")
    return out


# ----------------------------------------------------------------------------- the estimator

ROLES = ("gamma", "beta", "T", "U")


class Split:
    """Chebyshev interpolant of z -> F(z + v) on [lo, hi] with M terms."""

    def __init__(self, lo, hi, M=M_SPLIT):
        self.lo, self.hi, self.M = float(lo), float(hi), int(M)
        x = np.cos(np.pi * (np.arange(M) + 0.5) / M)
        self.nodes = self.lo + (x + 1) * (self.hi - self.lo) / 2
        Tk = np.array([Cheb.chebval(x, np.eye(M)[k]) for k in range(M)])
        self.proj = (2.0 / M) * Tk
        self.proj[0] /= 2.0

    def chi(self, z):
        x = np.clip(2 * (np.asarray(z, float) - self.lo) / (self.hi - self.lo) - 1, -1, 1)
        return Cheb.chebvander(x, self.M - 1)

    def xi(self, v):
        return expit(self.nodes[None, :] + np.asarray(v, float)[:, None]) @ self.proj.T


def fit_logit(e, D, X, Y, iters=50):
    n1 = int(D.sum()); n0 = len(D) - n1
    kn = max(3, int(round(min(n1, n0) ** (11 / 60))))
    interior = np.linspace(J[0], J[1], kn + 2)[1:-1]
    knots = np.concatenate([np.repeat(J[0], 4), interior, np.repeat(J[1], 4)])
    B = BSpline.design_matrix(np.clip(e, J[0], J[1]), knots, 3).toarray()
    A = np.column_stack([(1 - D)[:, None] * B, D[:, None] * B, X, D * X[:, 0]])
    nb = B.shape[1]
    theta = np.zeros(A.shape[1])
    for d, sl in ((0, slice(0, nb)), (1, slice(nb, 2 * nb))):
        m = float(np.clip(Y[D == d].mean(), 1e-3, 1 - 1e-3))
        theta[sl] = np.log(m / (1 - m))
    ll_old = -np.inf
    for _ in range(iters):
        u = A @ theta
        ll = float(np.sum(Y * u - np.logaddexp(0, u)))
        grad = A.T @ (Y - expit(u))
        Hs = (A * dexpit(u)[:, None]).T @ A
        step = np.linalg.solve(Hs, grad)
        tt = 1.0
        while tt > 1e-4:
            th = theta + tt * step
            u2 = A @ th
            ll2 = float(np.sum(Y * u2 - np.logaddexp(0, u2)))
            if ll2 >= ll - 1e-10:
                break
            tt /= 2
        theta = th
        if abs(ll2 - ll_old) < 1e-10 * (1 + abs(ll2)):
            break
        ll_old = ll2

    def z(ee, d):
        Be = BSpline.design_matrix(np.clip(ee, J[0], J[1]), knots, 3).toarray()
        return Be @ (theta[nb:2 * nb] if d else theta[:nb])

    return z, theta[2 * nb:2 * nb + 2], float(theta[2 * nb + 2]), kn


def rotation_criterion(Q, X, Y, folds, c):
    Xt = np.column_stack([np.ones(len(Q)), X])
    ig, ib, iT, iU = (folds[r] for r in ROLES)
    gam = np.linalg.lstsq(Xt[ig], Q[ig], rcond=None)[0]
    That = Xt @ gam
    ehat = Q - That
    D = (Q > PHI0).astype(float)
    l_hat = PHI0 - float(np.quantile(That[iT], 1 - EPS))
    u_hat = PHI0 - float(np.quantile(That[iT], EPS))
    keep = ib[(ehat[ib] >= J[0]) & (ehat[ib] <= J[1])]
    z, beta1, beta2, kn = fit_logit(ehat[keep], D[keep], X[keep], Y[keep])
    eU = ehat[iU]
    e = eU[(eU >= l_hat) & (eU <= u_hat)]
    z0, z1 = z(e, 0), z(e, 1)
    allz = np.concatenate([z0, z1])
    sp = Split(allz.min() - 0.25, allz.max() + 0.25)
    XT = X[iT]
    v0 = XT @ beta1
    v1 = v0 + beta2 * XT[:, 0]
    Psi = np.column_stack([sp.xi(v1), sp.xi(v0), np.ones(len(iT))])
    o = np.argsort(That[iT])
    Ts = That[iT][o]
    cum = np.cumsum(Psi[o][::-1], axis=0)[::-1] / len(iT)
    cum = np.vstack([cum, np.zeros((1, Psi.shape[1]))])
    C1, C0 = sp.chi(z1), sp.chi(z0)
    nU = len(iU)
    M = sp.M

    def crit(phi):
        pos = np.searchsorted(Ts, phi - e, side="right")
        Hs = cum[pos]
        val = np.sum(C1 * Hs[:, :M], axis=1) - np.sum(C0 * Hs[:, M:2 * M], axis=1) - c * Hs[:, -1]
        return float(np.sum(val) / nU)

    return crit, dict(gamma=gam.tolist(), window=[l_hat, u_hat], beta1=beta1.tolist(),
                      beta2=beta2, kn=kn)


def argmax(f, center, half, m=241, fine=81):
    grid = np.linspace(center - half, center + half, m)
    vals = np.array([f(x) for x in grid])
    k = int(np.argmax(vals))
    a, b_ = grid[max(k - 1, 0)], grid[min(k + 1, m - 1)]
    fg = np.linspace(a, b_, fine)
    fv = np.array([f(x) for x in fg])
    j = int(np.argmax(fv))
    return float(fg[j]), bool(k in (0, m - 1))


def role_scores(pop, Q, X, Y, eta):
    """The four role scores of the theorem at the true parameters, for every unit."""
    c, ps = pop["c"], pop["phi_star"]
    D = (Q > PHI0).astype(float)
    T = X @ GX
    a0, a1 = lin_pred(X, eta)
    inwin = (eta >= L0) & (eta <= U0)
    psi_U = -k_cond(eta, ps - eta, c) * inwin
    S = (eta >= J[0]) & (eta <= J[1])
    Bp = piece_basis(eta)
    nb = pop["_basis"]["nb"]
    Z = np.column_stack([(1 - D)[:, None] * Bp, D[:, None] * Bp, X, D * X[:, 0]])
    r = (Z @ np.asarray(pop["_basis"]["coef"])) * S
    psi_beta = -r * (Y - expit(np.where(D > 0, a1, a0)))
    ee = ps - T
    Xa0, Xa1 = lin_pred(X, ee)
    inw = (ee >= L0) & (ee <= U0)
    psi_T = (-(expit(Xa1) - expit(Xa0) - c) * f_eta(ee) * inw
             - pop["b_l"] * ((1 - EPS) - (T <= Q_1MEPS)) / g(Q_1MEPS)
             + pop["b_u"] * (EPS - (T <= Q_EPS)) / g(Q_EPS))
    Xt = np.column_stack([np.ones(len(Q)), X])
    psi_gamma = (Xt @ np.asarray(pop["Abar"])) * eta          # Sigma_X~ = I in this design
    assert nb == Bp.shape[1]
    return dict(U=psi_U, beta=psi_beta, T=psi_T, gamma=psi_gamma)


def one_rep(args):
    seed, n, pop, half = args
    c, phi_star = pop["c"], pop["phi_star"]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    eta = S_ETA * rng.standard_normal(n)
    Q = X @ GX + eta
    D = Q > PHI0
    a0, a1 = lin_pred(X, eta)
    Y = (rng.random(n) < expit(np.where(D, a1, a0))).astype(float)
    sc = role_scores(pop, Q, X, Y, eta)
    blocks = np.array_split(np.arange(n), 4)          # X, eta iid, so fixed blocks are random
    crits, ests, edge, comps = [], [], [], []
    for k in range(4):
        folds = {r: blocks[(j + k) % 4] for j, r in enumerate(ROLES)}
        f, _ = rotation_criterion(Q, X, Y, folds, c)
        crits.append(f)
        x, at_edge = argmax(f, phi_star, half)
        ests.append(x); edge.append(at_edge)
        # the theorem's linear term for this assignment: -(1/H) sum_q P_q psi_q, by fold
        comps.append([float(-np.mean(sc[q][folds[r]]) / pop["H"])
                      for q, r in (("U", "U"), ("beta", "beta"), ("T", "T"), ("gamma", "gamma"))])
    pooled, pe = argmax(lambda p: sum(cr(p) for cr in crits) / 4, phi_star, half)
    return dict(seed=seed, rot=ests, avg=float(np.mean(ests)), pooled=pooled,
                edge=bool(any(edge) or pe), comps=comps)


def monte_carlo(pop, n, reps, workers, seed0=20261118):
    from multiprocessing import Pool
    half = 6 * pop["sd_rootn_fixed"] / np.sqrt(n)
    args = [(seed0 + r, n, pop, half) for r in range(reps)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(one_rep, args, chunksize=4)
    el = time.time() - t0
    rot = np.array([r["rot"] for r in res])
    avg = np.array([r["avg"] for r in res])
    pooled = np.array([r["pooled"] for r in res])
    ps = pop["phi_star"]
    rn = np.sqrt(n)

    def summ(x, sd_th):
        d = rn * (x - ps)
        se_sd = d.std(ddof=1) / np.sqrt(2 * (len(d) - 1))
        cover = float(np.mean(np.abs(d) <= 1.959964 * sd_th))
        return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), sd_se=float(se_sd),
                    sd_formula=float(sd_th), ratio=float(d.std(ddof=1) / sd_th),
                    cover95_formula_sd=cover,
                    skew=float(np.mean(((d - d.mean()) / d.std()) ** 3)),
                    exkurt=float(np.mean(((d - d.mean()) / d.std()) ** 4) - 3))

    comps = rn * np.array([r["comps"] for r in res])          # reps x 4 rotations x 4 scores
    err = rn * (rot - ps)                                       # reps x 4
    lin = comps.sum(axis=2)
    E, Lm, C = err.reshape(-1), lin.reshape(-1), comps.reshape(-1, 4)
    coefs, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(E)), C]), E, rcond=None)
    resid = E - np.column_stack([np.ones(len(E)), C]) @ coefs
    cov_c = np.cov(C.T)
    coef_se = np.sqrt(np.diag(np.linalg.inv(cov_c)) * resid.var() / len(E))
    err_avg, lin_avg = err.mean(axis=1), lin.mean(axis=1)
    linear = dict(corr=float(np.corrcoef(E, Lm)[0, 1]), sd_err=float(E.std()),
                  sd_lin=float(Lm.std()), sd_err_minus_lin=float((E - Lm).std()),
                  mean_err_minus_lin=float((E - Lm).mean()),
                  coef_on_scores=dict(zip(("U", "beta", "T", "gamma"), coefs[1:].tolist())),
                  coef_se=dict(zip(("U", "beta", "T", "gamma"), coef_se.tolist())),
                  sd_scores=dict(zip(("U", "beta", "T", "gamma"), C.std(axis=0).tolist())),
                  sd_scores_formula=dict(
                      U=2 * np.sqrt(pop["var_U"]) / abs(pop["H"]),
                      beta=2 * np.sqrt(pop["var_beta"]) / abs(pop["H"]),
                      T=2 * np.sqrt(pop["var_T"]) / abs(pop["H"]),
                      gamma=2 * np.sqrt(pop["var_gamma"]) / abs(pop["H"])),
                  rotated=dict(corr=float(np.corrcoef(err_avg, lin_avg)[0, 1]),
                               sd_err_minus_lin=float((err_avg - lin_avg).std()),
                               sd_err=float(err_avg.std())))
    print(f"n={n} linear term: corr {linear['corr']:.4f}, sd(err) {linear['sd_err']:.3f}, "
          f"sd(lin) {linear['sd_lin']:.3f}, sd(err - lin) {linear['sd_err_minus_lin']:.3f}, "
          f"mean(err - lin) {linear['mean_err_minus_lin']:+.3f}")
    print("   coefficients on the four scores (1 expected):",
          {k: f"{v:.3f}+-{linear['coef_se'][k]:.3f}" for k, v in linear["coef_on_scores"].items()})
    print("   sd of each score term, simulated vs formula:",
          {k: f"{linear['sd_scores'][k]:.3f}/{linear['sd_scores_formula'][k]:.3f}" for k in linear["sd_scores"]})
    out = dict(n=n, reps=reps, seconds=el, edge=int(sum(r["edge"] for r in res)), linear=linear,
               fixed=summ(rot[:, 0], pop["sd_rootn_fixed"]),
               fixed_all4=summ(rot.reshape(-1), pop["sd_rootn_fixed"]),
               rotated_avg=summ(avg, pop["sd_rootn_rotated"]),
               pooled=summ(pooled, pop["sd_rootn_rotated"]),
               corr_rotations=np.corrcoef(rot.T).tolist(),
               raw=dict(rot=rot.tolist(), pooled=pooled.tolist()))
    for k in ("fixed", "fixed_all4", "rotated_avg", "pooled"):
        s = out[k]
        print(f"n={n} {k:11s}: mean {s['mean']:+.3f}, sd {s['sd']:.4f} (+-{s['sd_se']:.4f}) vs "
              f"formula {s['sd_formula']:.4f}, ratio {s['ratio']:.3f}, cover {s['cover95_formula_sd']:.3f}, "
              f"skew {s['skew']:+.2f}, exkurt {s['exkurt']:+.2f}")
    print(f"edge hits {out['edge']}, {el:.0f} s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[40000])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-aout", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    pop = population()
    res = dict(population={k: v for k, v in pop.items() if not k.startswith("_")})
    if not a.skip_aout:
        res["A_out_check"] = check_A_out(pop)
    res["monte_carlo"] = []
    for n in a.n:
        res["monte_carlo"].append(monte_carlo(pop, n, a.reps, a.workers))
        (a.out / "summary.json").write_text(json.dumps(res, indent=1))
    (a.out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
