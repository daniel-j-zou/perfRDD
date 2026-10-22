"""Quick checks of two proposed fixes for the location-scale theory.

(1) Smoothed criterion: replace the empirical law of xi-hat in the utility fold by an
    undersmoothed kernel density (h = sd * n_u^{-1/3}).  U-hat becomes smooth in phi with
    an analytic derivative, so the paper's derivative-based argmax argument applies.
    Compare with the step-function estimator on the same data.
(2) Scale first stage: Harvey (OLS of log eta-hat^2) against Poisson pseudo-likelihood
    (eta-hat^2 on exp(delta'X), damped Newton from Harvey).  Measure how much using
    eta-hat instead of the true eta moves delta-hat: sqrt(n) * sd(delta(eta-hat) - delta(eta)).
    A first-order term J (gamma-hat - gamma) keeps this O(1); without one it shrinks.
"""
import json
import sys
import time

import numpy as np
from scipy import optimize

import common as C

rng0 = np.random.default_rng(20261021)


# ----------------------------------------------------------------------------- (1)
def smooth_parts(xiu, alpha_hat, lh, uh, m=2001):
    n_u = len(xiu)
    h = xiu.std() * n_u ** (-1 / 3)
    g = np.linspace(lh, uh, m)
    f = np.zeros(m)
    for k in range(0, n_u, 4000):
        z = (g[:, None] - xiu[None, k:k + 4000]) / h
        f += np.exp(-0.5 * z * z).sum(1)
    f /= n_u * h * np.sqrt(2 * np.pi)
    a = alpha_hat(g)
    dg = g[1] - g[0]

    def tail(v):                                 # int_{g_k}^{u} v
        c = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * dg)]
        return c[-1] - c

    return g, f, a, tail(a * f), tail(f)


def make_smooth_U(parts, T_j, s_j, Xb_j):
    g, f, a, A, F = parts
    lo, hi = g[0], g[-1]

    def U(phi):
        c = np.clip((phi - T_j) / s_j, lo, hi)
        return float(np.mean(np.interp(c, g, A) + Xb_j * np.interp(c, g, F)))

    def dU(phi):
        c = (phi - T_j) / s_j
        inside = (c > lo) & (c < hi)
        cc = np.clip(c, lo, hi)
        return float(-np.mean(inside * (np.interp(cc, g, a) + Xb_j) * np.interp(cc, g, f) / s_j))

    return U, dU


def smooth_argmax(U, dU, grid=121):
    ph = np.linspace(C.PHI_LO, C.PHI_HI, grid)
    v = np.array([U(p) for p in ph])
    k = int(np.argmax(v))
    a, b = ph[max(k - 1, 0)], ph[min(k + 1, grid - 1)]
    if dU(a) > 0 > dU(b):
        return optimize.brentq(dU, a, b, xtol=1e-10), False
    return ph[k], True


def four_role_both(data, eps=C.EPS):
    X, Q, D, Y = data["X"], data["Q"], data["D"], data["Y"]
    folds = np.arange(len(Q)) % 4
    step, smooth, dcheck = [], [], []
    for r in range(4):
        fg, fo, fi, fu = [(folds == ((r + k) % 4)) for k in range(4)]
        idx = C.fit_index(X[fg], Q[fg], scale=True)
        To, so = C.eval_index(X[fo], idx)
        alpha_hat, b2, _ = C.fit_outcome((Q[fo] - To) / so, X[fo], D[fo], Y[fo])
        Tj, sj = C.eval_index(X[fi], idx)
        lh, uh = np.quantile((C.PHI0 - Tj) / sj, [eps, 1 - eps])
        Tu, su = C.eval_index(X[fu], idx)
        xiu = (Q[fu] - Tu) / su
        w = (xiu >= lh) & (xiu <= uh)
        Uh = C.make_Uhat(xiu[w], alpha_hat(xiu[w]), Tj, sj, X[fi] @ b2)
        step.append(C.argmax_step(Uh))
        U, dU = make_smooth_U(smooth_parts(xiu, alpha_hat, lh, uh), Tj, sj, X[fi] @ b2)
        p, at_edge = smooth_argmax(U, dU)
        smooth.append(p)
        if r == 0:                               # analytic derivative against a difference
            for q in (0.5, 0.8, 1.1):
                fd = (U(q + 1e-4) - U(q - 1e-4)) / 2e-4
                dcheck.append(abs(dU(q) - fd) / max(abs(fd), 1e-12))
    return float(np.mean(step)), float(np.mean(smooth)), max(dcheck)


def check_smoothing(reps):
    phi_star = json.load(open("target_results.json"))
    phi_star = phi_star.get("phi_star", phi_star.get("normal", {}).get("phi_star")) if isinstance(phi_star, dict) else None
    out = {}
    for n, R in reps:
        rows = []
        t0 = time.time()
        for rep in range(R):
            data = C.simulate(n, np.random.default_rng(7_000 + 100 * n + rep))
            rows.append(four_role_both(data))
        a = np.array(rows)
        out[n] = a
        st, sm, dc = a[:, 0], a[:, 1], a[:, 2]
        print(f"n={n:6d} reps={R}  step mean {st.mean():.4f} sd {st.std(ddof=1):.4f} (sqrt(n) sd {np.sqrt(n)*st.std(ddof=1):.2f})  "
              f"smooth mean {sm.mean():.4f} sd {sm.std(ddof=1):.4f} (sqrt(n) sd {np.sqrt(n)*sm.std(ddof=1):.2f})  "
              f"sqrt(n) sd(smooth-step) {np.sqrt(n)*np.std(sm-st, ddof=1):.2f}  max rel dU error {dc.max():.1e}  [{time.time()-t0:.0f}s]",
              flush=True)
    return out


# ----------------------------------------------------------------------------- (2)
def harvey(Z, eta):
    return np.linalg.lstsq(Z, np.log(eta ** 2), rcond=None)[0]


def poisson(Z, eta, start):
    y = eta ** 2
    d = start.copy()
    d[0] = np.log(np.mean(y) / np.mean(np.exp(Z[:, 1:] @ d[1:])))
    obj = lambda dd: np.mean(np.exp(Z @ dd) - y * (Z @ dd))
    for _ in range(100):
        mu = np.exp(Z @ d)
        step = np.linalg.solve((Z * mu[:, None]).T @ Z, Z.T @ (y - mu))
        t = 1.0
        while obj(d + t * step) > obj(d) and t > 1e-6:
            t /= 2
        d = d + t * step
        if np.max(np.abs(t * step)) < 1e-10:
            break
    return d


def draw(dist, n, rng):
    if dist == "normal":
        return rng.standard_normal(n)
    if dist == "gamma2":                         # skewness sqrt(2), mean 0, var 1
        return (rng.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0)
    raise ValueError(dist)


def check_scale(ns=(4000, 16000, 64000), R=150):
    for dist in ("normal", "gamma2"):
        for n in ns:
            dh, dp = [], []
            for rep in range(R):
                rng = np.random.default_rng(90_000 + n + rep)
                X1 = rng.standard_normal(n); X2 = rng.integers(0, 2, n) - 0.5
                Z = np.column_stack([np.ones(n), X1, X2])
                s = np.exp(0.5 * (C.DELTA[0] * X1 + C.DELTA[1] * X2))
                eta = s * draw(dist, n, rng)
                Q = Z @ C.GAMMA + eta
                g = np.linalg.lstsq(Z, Q, rcond=None)[0]
                eh = Q - Z @ g
                h_true, h_hat = harvey(Z, eta), harvey(Z, eh)
                p_true, p_hat = poisson(Z, eta, h_true), poisson(Z, eh, h_hat)
                dh.append(h_hat[1:] - h_true[1:]); dp.append(p_hat[1:] - p_true[1:])
            dh, dp = np.array(dh), np.array(dp)
            print(f"{dist:6s} n={n:6d}  sqrt(n) sd[delta(eta-hat) - delta(eta)] slopes: "
                  f"Harvey {np.round(np.sqrt(n) * dh.std(0, ddof=1), 3).tolist()}  "
                  f"Poisson {np.round(np.sqrt(n) * dp.std(0, ddof=1), 3).tolist()}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("scale", "all"):
        check_scale()
    if which in ("smooth", "all"):
        check_smoothing([(16000, 30), (64000, 25)])
